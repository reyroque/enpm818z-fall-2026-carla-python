"""Export the L9 road stretch from CARLA's Town03 map, with no CARLA server.

    python3 export_scene.py /path/to/Town03.xodr          # writes l9_scene.npz here

The stretch is road 67 of Town03, driven east and then north: the AV's lane is
lane 3 (the right lane), the lane beside it is lane 2 (the left lane, same
direction), then a bidirectional center lane (lane 1), then the two lanes the
other way. The script follows each lane's waypoints with get_next() every 0.5 m
and saves them in the map frame used by every L9 script:

    x  = CARLA's x,   y = minus CARLA's y   (so the frame is right-handed),
    heading = minus CARLA's yaw, in radians, counterclockwise from +x.

It reads only the map: carla.Map(name, xodr_text) works offline (CARLA 0.9.16
client). The .xodr file is the map's OpenDRIVE description; a running CARLA server also
returns it with world.get_map().to_opendrive().
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROAD = 67            # Town03 road with two lanes each way and two left curves
START_S = 290.0      # road s where the stretch starts (lane 3 drives toward s = 0)
STEP = 0.5           # meters between exported points
LENGTH = 230.0       # meters of the right lane exported


def follow(m, carla, lane_id, start_s, length, step):
    """Points along one lane, in its driving direction, every `step` meters."""
    wp = m.get_waypoint_xodr(ROAD, lane_id, start_s)
    if wp is None:
        raise RuntimeError(f"no waypoint on road {ROAD}, lane {lane_id}, s = {start_s}")
    pts = []
    for _ in range(int(round(length / step)) + 1):
        t = wp.transform
        pts.append((t.location.x, -t.location.y, -np.radians(t.rotation.yaw), wp.lane_width,
                    wp.road_id, wp.lane_id, float(wp.is_junction)))
        nxt = wp.next(step)
        if not nxt:
            break
        # stay on the same lane: at a fork, take the successor closest in heading
        wp = min(nxt, key=lambda w: abs(((w.transform.rotation.yaw - t.rotation.yaw) + 180) % 360 - 180))
    return np.array(pts)


def overview(m, carla, step=2.0):
    """Every lane segment of the town's topology as a polyline, for the overview figure.
    Returns rows (x, y, segment index)."""
    lines = []
    for a, b in m.get_topology():
        pts, w = [], a
        for _ in range(400):
            pts.append((w.transform.location.x, -w.transform.location.y, -np.radians(w.transform.rotation.yaw)))
            nxt = w.next(step)
            if not nxt or (nxt[0].road_id != a.road_id and w.road_id != a.road_id):
                break
            w = nxt[0]
            if w.transform.location.distance(b.transform.location) < step:
                pts.append((b.transform.location.x, -b.transform.location.y, -np.radians(b.transform.rotation.yaw)))
                break
        pts = np.array(pts)
        lines.append(np.column_stack([pts[:, :2], np.full(len(pts), len(lines))]))
    return np.vstack(lines)


def tight_curve(m, road=2, lane_id=-1, step=1.0):
    """Radius of the tight left curve on Town03 road 2 (lane -1), for the slide on
    where the Frenet frame breaks: the heading change over the arc divided by its
    length, on the stretch where the curvature is constant."""
    wps = sorted([w for w in m.generate_waypoints(step) if w.road_id == road and w.lane_id == lane_id],
                 key=lambda w: w.s)
    hd = np.unwrap(np.radians([-w.transform.rotation.yaw for w in wps]))
    s = np.array([w.s for w in wps])
    xy = np.array([[w.transform.location.x, -w.transform.location.y] for w in wps])
    k = np.gradient(hd, s)
    on = np.abs(k - np.median(k[np.abs(k) > 1e-3])) < 1e-4
    arc = np.sum(np.hypot(*np.diff(xy[on], axis=0).T))
    return float(arc / abs(hd[on][-1] - hd[on][0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xodr", help="path to Town03.xodr")
    ap.add_argument("--out", default=str(HERE / "l9_scene.npz"))
    a = ap.parse_args(argv)
    import carla
    m = carla.Map("Town03", Path(a.xodr).read_text())
    right = follow(m, carla, 3, START_S, LENGTH, STEP)
    left = follow(m, carla, 2, START_S, LENGTH, STEP)
    town = overview(m, carla)
    np.savez(a.out, right=right, left=left, town=town, road2_radius=tight_curve(m),
             columns=np.array(["x", "y", "heading", "lane_width", "road_id", "lane_id", "is_junction"]),
             source=np.array(f"Town03.xodr road {ROAD}, lanes 3 and 2 from road s = {START_S} m, "
                             f"every {STEP} m along get_next(); x = CARLA x, y = -CARLA y"))
    print(f"wrote {a.out}: right lane {len(right)} points, left lane {len(left)} points")


if __name__ == "__main__":
    main()
