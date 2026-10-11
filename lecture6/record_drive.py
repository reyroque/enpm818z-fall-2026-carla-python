"""Record the L6 drive from CARLA: a mapping pass and a localization pass.

GNSS is simulated from the true pose in local meters (noise, bias, an
outage): CARLA's GNSS sensor models no multipath or blocked sky, so the
outage is ours, and it is labeled as chosen in l6_drive_meta.json.
Wheel speed is the true speed with a chosen scale error and noise: CARLA has
no wheel encoder.

Never run this against a CARLA you are driving. Start a private server:
  docker run -d --name carla-test --runtime=nvidia --gpus all --net=host \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 /bin/bash -c "./CarlaUE4.sh -RenderOffScreen \
    -vulkan -nosound -quality-level=Low -carla-rpc-port=2200"
"""
import argparse
import csv
import json
import math
import queue
import subprocess
from datetime import date

import numpy as np

from l6common import DRIVE_COLUMNS, save_scans

DT = 0.05
CHOSEN = {"gnss_rate_hz": 1.0, "gnss_sigma_m": 1.5, "gnss_bias_m": [0.5, -0.3],
          "outage_length_m": 400.0, "wheel_scale": 0.005, "wheel_sigma_mps": 0.05,
          "gyro_bias_rps": 0.001, "gyro_sigma_rps": 0.01, "lidar_hz": 10.0,
          "lidar_channels": 32, "lidar_range_m": 50.0, "slice_half_height_m": 0.3, "voxel_m": 0.2}


def preflight(docker_ps_names, our_name="carla-test"):
    others = [n for n in docker_ps_names if "carla" in n.lower() and n != our_name]
    if others:
        raise RuntimeError(f"a CARLA container is running ({', '.join(others)}): "
                           "it may be the user's session; stop and ask before recording")


def running_containers():
    out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    return [n for n in out.stdout.split() if n]


def to_local(loc):
    """CARLA is left-handed (y to the right); flip y so x east, y north."""
    return loc.x, -loc.y


def heading(tf):
    return -math.radians(tf.rotation.yaw)


def slice_2d(points_xyz, half_height, voxel):
    pts = points_xyz[np.abs(points_xyz[:, 2]) < half_height][:, :2]
    pts[:, 1] *= -1                                   # vehicle frame: y left
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--town", required=False)
    ap.add_argument("--port", type=int, default=2200)
    ap.add_argument("--tm-port", type=int, default=8200)
    ap.add_argument("--seed", type=int, default=818)
    ap.add_argument("--list-towns", action="store_true")
    ap.add_argument("--route", help="JSON file: list of [x, y] CARLA locations to drive through, in order")
    ap.add_argument("--outage-start-m", type=float, help="distance along the route where the GNSS outage starts")
    ap.add_argument("--laps", type=int, default=1, help="how many times the route returns to its start")
    args = ap.parse_args()
    preflight(running_containers())

    import carla
    client = carla.Client("localhost", args.port)
    client.set_timeout(30.0)
    if args.list_towns:
        print("\n".join(sorted(client.get_available_maps())))
        return
    world = client.load_world(args.town)
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, DT
    world.apply_settings(settings)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    route = json.load(open(args.route))
    passes = {}
    try:
        for name in ("mapping", "localization"):
            passes[name] = drive_once(world, tm, route, rng, carla, args.laps)
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
    write_outputs(passes, route, args, rng)


def wait_for(q, frame, timeout=5.0):
    """The sensor message for this tick (older ones are dropped)."""
    while True:
        m = q.get(timeout=timeout)
        if m.frame >= frame:
            return m


def drive_once(world, tm, route, rng, carla, laps=1):
    bp = world.get_blueprint_library()
    start = carla.Transform(carla.Location(x=route[0][0], y=route[0][1], z=0.5))
    pose = world.get_map().get_waypoint(start.location).transform
    pose.location.z += 0.5                           # a waypoint is on the road surface
    vehicle = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), pose)
    imu = world.spawn_actor(bp.find("sensor.other.imu"), carla.Transform(), attach_to=vehicle)
    lbp = bp.find("sensor.lidar.ray_cast")
    # In synchronous mode each tick returns the points swept during that tick,
    # so the sensor turns once per tick (1 / DT) to give a full circle per scan.
    for k, v in (("channels", CHOSEN["lidar_channels"]), ("range", CHOSEN["lidar_range_m"]),
                 ("rotation_frequency", 1.0 / DT), ("points_per_second", 1200000)):
        lbp.set_attribute(k, str(v))
    lidar = world.spawn_actor(lbp, carla.Transform(carla.Location(z=1.9)), attach_to=vehicle)
    queues = {"imu": queue.Queue(), "lidar": queue.Queue()}
    imu.listen(queues["imu"].put)
    lidar.listen(queues["lidar"].put)
    vehicle.set_autopilot(True, tm.get_port())
    tm.set_path(vehicle, [carla.Location(x=x, y=y, z=0.5) for x, y in route[1:]])
    tm.ignore_lights_percentage(vehicle, 100.0)
    rows, scans, scan_t = [], [], []
    goal = carla.Location(x=route[-1][0], y=route[-1][1])
    away, done = False, 0                              # laps: count returns to the goal
    scan_every = int(round(1.0 / (CHOSEN["lidar_hz"] * DT)))
    try:
        for tick in range(1, int(600 / DT) + 1):             # at most 10 minutes
            frame = world.tick()
            t = tick * DT
            latest = {name: wait_for(q, frame) for name, q in queues.items()}
            tf, vel = vehicle.get_transform(), vehicle.get_velocity()
            x, y = to_local(tf.location)
            v = math.hypot(vel.x, vel.y)
            gz = -latest["imu"].gyroscope.z
            rows.append([t, x, y, heading(tf), v, gz])
            if tick % scan_every == 0:
                raw = np.frombuffer(latest["lidar"].raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
                scans.append(slice_2d(raw.copy(), CHOSEN["slice_half_height_m"], CHOSEN["voxel_m"]))
                scan_t.append(t)
            gap = tf.location.distance(goal)
            if gap > 30.0:
                away = True
            elif away and gap < 5.0:
                away, done = False, done + 1
                if done == laps:
                    break
    finally:
        # Hand the car back from Traffic Manager and stop the sensors before
        # destroying anything: TM touching a destroyed actor aborts the client.
        vehicle.set_autopilot(False, tm.get_port())
        for s in (lidar, imu):
            s.stop()
        world.tick()
        for a in (lidar, imu, vehicle):
            a.destroy()
        world.tick()
    return {"rows": np.array(rows), "scans": scans, "scan_t": np.array(scan_t)}


def write_outputs(passes, route, args, rng):
    m = passes["mapping"]
    world_pts = []
    for st, scan in zip(m["scan_t"], m["scans"]):
        k = int(np.argmin(np.abs(m["rows"][:, 0] - st)))
        _, x, y, th, _, _ = m["rows"][k]
        c, s = np.cos(th), np.sin(th)
        world_pts.append(scan @ np.array([[c, s], [-s, c]]) + [x, y])
    pts = np.vstack(world_pts)
    keys = np.floor(pts / CHOSEN["voxel_m"]).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    np.savez_compressed("l6_map.npz", points=pts[np.sort(first)])

    L = passes["localization"]["rows"]
    t, x, y, th, v, gz = L.T
    along = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    o0 = args.outage_start_m
    o1 = o0 + CHOSEN["outage_length_m"]
    wheel = v * (1 + CHOSEN["wheel_scale"]) + rng.normal(0, CHOSEN["wheel_sigma_mps"], len(v))
    gyro = gz + CHOSEN["gyro_bias_rps"] + rng.normal(0, CHOSEN["gyro_sigma_rps"], len(v))
    every = int(round(1 / (CHOSEN["gnss_rate_hz"] * DT)))
    with open("l6_drive.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DRIVE_COLUMNS)
        for k in range(len(t)):
            fix = (k % every == 0) and not (o0 <= along[k] <= o1)
            gx = x[k] + CHOSEN["gnss_bias_m"][0] + rng.normal(0, CHOSEN["gnss_sigma_m"]) if fix else ""
            gy = y[k] + CHOSEN["gnss_bias_m"][1] + rng.normal(0, CHOSEN["gnss_sigma_m"]) if fix else ""
            w.writerow([f"{t[k]:.2f}", f"{x[k]:.4f}", f"{y[k]:.4f}", f"{th[k]:.6f}", f"{v[k]:.4f}",
                        f"{gyro[k]:.6f}", f"{wheel[k]:.4f}",
                        f"{gx:.4f}" if gx != "" else "", f"{gy:.4f}" if gy != "" else ""])
    save_scans("l6_scans.npz", passes["localization"]["scan_t"], passes["localization"]["scans"])
    json.dump({"town": args.town, "route": route, "laps": args.laps, "fixed_step_s": DT, "chosen": CHOSEN,
               "outage_m": [o0, o1], "seed": args.seed, "carla": "0.9.16", "date": str(date.today())},
              open("l6_drive_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
