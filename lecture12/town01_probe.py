"""Read the street values L12 uses from CARLA's Town01 map, with no CARLA server.

    /usr/bin/python3 town01_probe.py PATH/TO/Town01.xodr     # writes town01.json

Needs the carla Python client (pip install carla), which can open an OpenDRIVE
file offline: carla.Map("Town01", xodr_text). The values it writes are the ones
l12_sim.py uses: lane width, shoulder and sidewalk width, the speed limit, and a
long straight road for the crossing.
"""
import collections
import json
import re
import sys
from pathlib import Path

import carla
import numpy as np


def main(xodr_path):
    xodr = Path(xodr_path).read_text()
    m = carla.Map("Town01", xodr)
    wps = m.generate_waypoints(2.0)
    widths = collections.Counter(round(wp.lane_width, 2) for wp in wps)
    roads = collections.defaultdict(list)
    for wp in wps:
        if not wp.is_junction:
            roads[(wp.road_id, wp.lane_id)].append(wp)
    straight = []
    for (road, lane), v in roads.items():
        yaw = np.unwrap(np.radians([wp.transform.rotation.yaw for wp in v]))
        if np.ptp(yaw) < 0.01:
            straight.append((len(v) * 2.0, road))
    length, road = max(straight)
    wp = roads[(road, -1)][len(roads[(road, -1)]) // 2]
    side = []
    lane = wp
    while True:
        lane = lane.get_right_lane()
        if lane is None or lane.lane_type == carla.LaneType.Driving:
            break
        side.append((str(lane.lane_type), round(lane.lane_width, 2)))
        if lane.lane_type == carla.LaneType.Sidewalk:
            break
    speeds = collections.Counter(re.findall(r'<speed max="([0-9.]+)" unit="([a-z/]+)"', xodr))
    out = {"source": Path(xodr_path).name, "waypoints_at_2m": len(wps),
           "driving_lane_widths_m": {str(k): n for k, n in widths.items()},
           "straight_road": {"road_id": road, "length_m": length},
           "beside_the_right_lane": side,
           "speed_limits": {f"{v} {u}": n for (v, u), n in speeds.items()}}
    Path(__file__).with_name("town01.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
