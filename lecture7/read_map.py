"""Read CARLA's Town01 map file with no CARLA server: the code on L7's slide
"Read the map file yourself, with no CARLA server".

  cd lecture7
  python3 read_map.py

Needs CARLA's Python package (pip install carla==0.9.16). No server, no GPU.
It takes no options and opens no window. The clickable map is a different script:
python3 l7_handson.py --part map.
"""
import sys

if len(sys.argv) > 1:
    sys.exit("read_map.py takes no options; it prints three counts. "
             "For the clickable map, run: python3 l7_handson.py --part map")

import carla

text = open("maps/Town01.xodr").read()   # OpenDRIVE: plain XML text, one <road> per road
m = carla.Map("Town01", text)            # CARLA builds roads, lanes and waypoints from the text;
                                         # a running simulator is not needed for this
wps = m.generate_waypoints(2.0)          # a point every 2 m along the middle of every lane
segs = m.get_topology()                  # (start, end) waypoint pairs: one lane of one road
                                         # each, listed once per lane it can lead into

lights = text.count('type="1000001"')    # OpenDRIVE's signal type code for a traffic light
print(f"waypoints at 2 m, len(wps): {len(wps)}")
print(f"(start, end) waypoint pairs, len(segs): {len(segs)}")
print(f"traffic lights in the file: {lights}")
