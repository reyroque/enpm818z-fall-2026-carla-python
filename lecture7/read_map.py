"""Read CARLA's Town01 map file with no CARLA server: the code on L7's slide
"Read the map file yourself, with no CARLA server".

  cd lecture7
  python3 read_map.py

Needs CARLA's Python package (pip install carla==0.9.16). No server, no GPU.
"""
import carla

text = open("maps/Town01.xodr").read()
m = carla.Map("Town01", text)      # no server needed
wps = m.generate_waypoints(2.0)    # every 2 m, every lane
segs = m.get_topology()            # (start, end) of each piece

lights = text.count('type="1000001"')  # OpenDRIVE's signal type for a traffic light
print(f"waypoints at 2 m, len(wps): {len(wps)}")
print(f"topology segments, len(segs): {len(segs)}")
print(f"traffic lights in the file: {lights}")
