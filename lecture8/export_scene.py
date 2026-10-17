"""Export the lanes of one Town01 junction for the L8 scene, with no CARLA server.

    python3 export_scene.py PATH/TO/Town01.xodr            # writes l8_scene.npz here

Reads the OpenDRIVE file through carla.Map (the CARLA 0.9.16 Python package, no
server, no GPU) and writes, in the map frame (x east, y north, meters; CARLA's y is
flipped, as in lecture6):

  * path_<name>: the six ways through junction 54, each one lane centerline from
    120 m before the junction to 60 m after it, one point every 0.5 m. The junction
    itself lists these six (junction.get_waypoints); before and after it, each lane is
    followed straight on. Nothing here searches the road network.
  * lane_<k>: the edges of every driving lane in the window (left edge x, y, right
    edge x, y, seen in the direction of travel), for drawing; lane_in_junction says
    which of them lie inside a junction.

The hands-on (l8_handson.py) and l8sim.py read only l8_scene.npz, so they run with
numpy and matplotlib alone.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JUNCTION = 54                  # the T-junction at x = 156 m, y = -2 m (map frame)
WINDOW = (95.0, 235.0, -60.0, 25.0)   # x min, x max, y min, y max (map frame, m)
STEP = 0.5                     # m between centerline points
BEFORE, AFTER = 120.0, 60.0    # m of lane kept before and after the junction


def xy(wp):
    loc = wp.transform.location
    return (loc.x, -loc.y)     # CARLA's y points south; the map frame's y points north


def walk(wp, dist, back):
    """Follow one lane for dist meters, forward or backward. Where the lane splits (at
    another junction), keep the branch whose point 10 m on lies most nearly straight
    ahead, so the road continues straight. Returns the points, nearest first."""
    pts, d = [], 0.0
    while d < dist:
        nxt = wp.previous(STEP) if back else wp.next(STEP)
        if not nxt:
            break
        if len(nxt) > 1:
            here = np.array(xy(wp))
            yaw = -np.radians(wp.transform.rotation.yaw) + (np.pi if back else 0.0)
            ahead = np.array([np.cos(yaw), np.sin(yaw)])

            def straightness(w):
                far = w.previous(10.0) if back else w.next(10.0)
                v = np.array(xy(far[0] if far else w)) - here
                return float(v @ ahead) / (np.linalg.norm(v) + 1e-9)
            wp = max(nxt, key=straightness)
        else:
            wp = nxt[0]
        pts.append(xy(wp))
        d += STEP
    return pts


def heading_name(p0, p1):
    a = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    return {0: "east", 90: "north", 180: "west", -180: "west", -90: "south"}[int(np.round(a / 90.0) * 90)]


def main(argv):
    import carla
    if len(argv) != 1:
        sys.exit(__doc__)
    xodr = Path(argv[0])
    m = carla.Map("Town01", xodr.read_text())
    wps = m.generate_waypoints(2.0)
    junction = next(w.get_junction() for w in wps if w.is_junction and w.get_junction().id == JUNCTION)
    out = {}
    for start, end in junction.get_waypoints(carla.LaneType.Driving):
        inside = [xy(start)] + [xy(w) for w in start.next_until_lane_end(STEP)]
        before = walk(start, BEFORE, back=True)[::-1]
        after = walk(end, AFTER, back=False)
        pts = np.array(before + inside + after)
        # name: where it comes from and where it goes, by heading at both ends
        name = f"{heading_name(inside[0], inside[2])}_to_{heading_name(inside[-3], inside[-1])}"
        out[f"path_{name}"] = pts
        out[f"entry_{name}"] = np.array(xy(start))
        out[f"exit_{name}"] = np.array(xy(end))
    # lane edges for drawing: every driving lane in the window, from 0.5 m waypoints
    lanes = {}
    for w in m.generate_waypoints(0.5):
        x, y = xy(w)
        if not (WINDOW[0] - 5 <= x <= WINDOW[1] + 5 and WINDOW[2] - 5 <= y <= WINDOW[3] + 5):
            continue
        key = (w.is_junction, w.road_id, w.section_id, w.lane_id)
        yaw = -np.radians(w.transform.rotation.yaw)
        lanes.setdefault(key, []).append((w.s, x, y, yaw, w.lane_width))
    k, junc = 0, []
    for key, rows in sorted(lanes.items()):
        rows = np.array(sorted(rows))
        if len(rows) < 2:
            continue
        nx, ny = -np.sin(rows[:, 3]), np.cos(rows[:, 3])
        half = rows[:, 4] / 2
        left = np.c_[rows[:, 1] + nx * half, rows[:, 2] + ny * half]
        right = np.c_[rows[:, 1] - nx * half, rows[:, 2] - ny * half]
        out[f"lane_{k:03d}"] = np.c_[left, right]
        junc.append(key[0])
        k += 1
    out["lane_in_junction"] = np.array(junc, bool)
    out["window"] = np.array(WINDOW)
    out["junction_center"] = np.array([junction.bounding_box.location.x, -junction.bounding_box.location.y])
    out["source"] = np.array(f"{xodr.name}, junction {JUNCTION}, carla.Map offline (CARLA 0.9.16)")
    np.savez_compressed(HERE / "l8_scene.npz", **out)
    print("wrote", HERE / "l8_scene.npz", "with", sum(1 for n in out if n.startswith("path_")), "paths and", k, "lanes")
    for n in sorted(out):
        if n.startswith("path_"):
            p = out[n]
            print(f"  {n}: {len(p)} points, {np.sum(np.hypot(*np.diff(p, axis=0).T)):.1f} m")


if __name__ == "__main__":
    main(sys.argv[1:])
