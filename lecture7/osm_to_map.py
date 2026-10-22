"""Turn an OpenStreetMap export into a CARLA map, with no CARLA server: the code on L7's
slide "From your .osm file to a map, in Python".

  cd lecture7
  python3 osm_to_map.py                      # the campus around the Kim building
  python3 osm_to_map.py my_area.osm          # your own export from openstreetmap.org

Needs CARLA's Python package (pip install carla==0.9.16). No server, no GPU.
maps/umd_engineering.osm: map data (c) OpenStreetMap contributors, ODbL.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import carla

HERE = Path(__file__).resolve().parent


def find_osm(name):
    """The file as given, else the same name in lecture7/maps/ or next to this script."""
    for p in (Path(name), HERE / "maps" / Path(name).name, HERE / Path(name).name):
        if p.is_file():
            return p
    sys.exit(f"osm_to_map: no file {name!r} here, in {HERE / 'maps'}, or in {HERE}")


path = find_osm(sys.argv[1] if len(sys.argv) > 1 else "umd_engineering.osm")
print(f"reading {path}")
osm = path.read_text()

# Center the projection on the file's own area (its <bounds>), not on longitude 0.
b = ET.fromstring(osm).find("bounds")
lat0 = (float(b.get("minlat")) + float(b.get("maxlat"))) / 2
lon0 = (float(b.get("minlon")) + float(b.get("maxlon"))) / 2

settings = carla.Osm2OdrSettings()
settings.proj_string = (f"+proj=tmerc +lat_0={lat0} +lon_0={lon0}"
                        " +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m")
settings.center_map = False                  # keep the projection's own origin
settings.use_offsets = False
xodr = carla.Osm2Odr.convert(osm, settings)  # OpenDRIVE text, as in Town01.xodr

m = carla.Map("my_area", xodr)               # then read it as Town01
root = ET.fromstring(xodr)
print(f"projection centered on lat {lat0:.5f}, lon {lon0:.5f}")
print(f"roads: {len(root.findall('road'))}, intersections: {len(root.findall('junction'))}, "
      f"waypoints at 2 m: {len(m.generate_waypoints(2.0))}")
open("my_area.xodr", "w").write(xodr)
print("wrote my_area.xodr")
