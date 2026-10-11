#!/usr/bin/env python3
"""
record_stereo.py: the L6 route again, with a stereo camera.

Same town (Town10HD_Opt), same route and laps, same car and autopilot as
../record_drive.py, no other traffic. Two cameras look
forward from the roof, BASELINE_M apart, and save a grayscale image each
every 0.1 s (10 Hz), with the left camera's true pose.

    stereo/left/00000.png, stereo/right/00000.png
    stereo/poses.csv   t, then the left camera's 4 x 4 pose in CARLA axes
                       (camera -> world, row by row), then x, y, yaw in the
                       L6 map frame (x east, y north, yaw counterclockwise)
    stereo/meta.json   intrinsics, baseline, the chosen values

Needs a running CARLA 0.9.16 server. Pass --port and --tm-port to match
yours (CARLA's own defaults are 2000 and 8000):

    python3 record_stereo.py --route ../route.json --out stereo --port 2000 --tm-port 8000

About 900 MB of images.
"""

import argparse
import json
import math
import queue
from pathlib import Path

import cv2
import numpy as np

DT = 0.05
W, H, FOV = 1024, 512, 90.0
BASELINE_M = 0.54            # chosen: KITTI's stereo baseline
MOUNT = dict(x=1.2, z=1.7)   # chosen: on the roof, ahead of the LiDAR
EVERY = 2                    # save every 2nd 0.05 s tick: 10 Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', required=True)
    ap.add_argument('--laps', type=int, default=2)
    ap.add_argument('--port', type=int, default=2200)
    ap.add_argument('--tm-port', type=int, default=8200)
    ap.add_argument('--seed', type=int, default=818)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import carla
    out = Path(a.out)
    for s in ('left', 'right'):
        (out / s).mkdir(parents=True, exist_ok=True)
    client = carla.Client('localhost', a.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != 'Town10HD_Opt':
        world = client.load_world('Town10HD_Opt')
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    world.apply_settings(s)
    tm = client.get_trafficmanager(a.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(a.seed)
    route = json.load(open(a.route))
    bp = world.get_blueprint_library()
    start = carla.Transform(carla.Location(x=route[0][0], y=route[0][1], z=0.5))
    pose = world.get_map().get_waypoint(start.location).transform
    pose.location.z += 0.5
    actors = []
    try:
        car = world.spawn_actor(bp.find('vehicle.lincoln.mkz_2020'), pose)
        actors.append(car)
        cams, qs = [], []
        cbp = bp.find('sensor.camera.rgb')
        for k, v in (('image_size_x', W), ('image_size_y', H), ('fov', FOV)):
            cbp.set_attribute(k, str(v))
        for y in (-BASELINE_M / 2, BASELINE_M / 2):         # CARLA y right: left camera first
            c = world.spawn_actor(cbp, carla.Transform(carla.Location(x=MOUNT['x'], y=y, z=MOUNT['z'])), attach_to=car)
            q = queue.Queue(); c.listen(q.put)
            cams.append(c); qs.append(q); actors.append(c)
        car.set_autopilot(True, tm.get_port())
        tm.set_path(car, [carla.Location(x=x, y=y, z=0.5) for x, y in route[1:]])
        tm.ignore_lights_percentage(car, 100.0)
        goal = carla.Location(x=route[-1][0], y=route[-1][1])
        away, done, saved, rows = False, 0, 0, []
        for tick in range(1, int(600 / DT) + 1):
            frame = world.tick()
            imgs = []
            for q in qs:
                m = q.get(timeout=10.0)
                while m.frame < frame:
                    m = q.get(timeout=10.0)
                imgs.append(m)
            if tick % EVERY == 0:
                for side, m in zip(('left', 'right'), imgs):
                    bgra = np.frombuffer(m.raw_data, dtype=np.uint8).reshape(H, W, 4)
                    cv2.imwrite(str(out / side / f'{saved:05d}.png'), cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY))
                T = np.array(imgs[0].transform.get_matrix())
                loc = imgs[0].transform.location
                yaw = -math.radians(imgs[0].transform.rotation.yaw)
                rows.append([tick * DT, *T.ravel(), loc.x, -loc.y, yaw])
                saved += 1
            gap = car.get_location().distance(goal)
            if gap > 30.0:
                away = True
            elif away and gap < 5.0:
                away, done = False, done + 1
                if done == a.laps:
                    break
    finally:
        car.set_autopilot(False, tm.get_port())
        for c in cams:
            c.stop()
        for x in actors[::-1]:
            x.destroy()
        world.tick()
        tm.set_synchronous_mode(False)
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
    np.savetxt(out / 'poses.csv', np.array(rows), delimiter=',',
               header='t,' + ','.join(f'T{i}{j}' for i in range(4) for j in range(4)) + ',x,y,yaw')
    f = W / (2 * math.tan(math.radians(FOV) / 2))
    (out / 'meta.json').write_text(json.dumps({'town': 'Town10HD_Opt', 'frames': saved, 'rate_hz': 1 / (DT * EVERY),
                                               'K': [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], 'width': W, 'height': H,
                                               'baseline_m': BASELINE_M, 'mount': MOUNT, 'laps': a.laps,
                                               'traffic': 'none', 'seed': a.seed}, indent=1))
    print(f'saved {saved} stereo pairs to {out}')


if __name__ == '__main__':
    main()
