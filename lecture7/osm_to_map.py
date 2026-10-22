"""Turn an OpenStreetMap export into a CARLA map, with no CARLA server: the code on L7's
slide "From your .osm file to a map, in Python".

  cd lecture7
  python3 osm_to_map.py                         # the campus around the Kim building
  python3 osm_to_map.py ~/Downloads/map.osm     # your own export from openstreetmap.org

A bare file name (umd_engineering.osm) is also looked for in lecture7/maps/.
Needs CARLA's Python package (pip install carla==0.9.16). No server, no GPU.
Writes my_area.xodr into the folder you run it from (git ignores it).
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
osm = path.read_text()      # an .osm file is XML text: nodes (points), ways (streets), tags

# An OpenStreetMap export stores positions as latitude and longitude, in degrees. A map in
# meters needs a projection: a rule that flattens the Earth's curved surface onto a plane.
# Transverse Mercator ("tmerc") is exact along one north-south line, the central meridian
# lon_0, and stretches the map more the farther you go from it. CARLA's default lon_0 is 0
# (Greenwich, England), 77 degrees from College Park, which made our map 1.94 times too tall.
# So we put lat_0 and lon_0 at the middle of the exported box: <bounds> holds its edges.
b = ET.fromstring(osm).find("bounds")
lat0 = (float(b.get("minlat")) + float(b.get("maxlat"))) / 2
lon0 = (float(b.get("minlon")) + float(b.get("maxlon"))) / 2

settings = carla.Osm2OdrSettings()
# proj4 string: Transverse Mercator centered on our area; k=1, no scale factor; x_0 and y_0
# are 0, so (0, 0) m is the point (lat0, lon0); WGS84 is the Earth model GPS uses; units: m.
settings.proj_string = (f"+proj=tmerc +lat_0={lat0} +lon_0={lon0}"
                        " +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m")
# Both options below would move the map's origin and override parts of proj_string
# (CARLA's docs). Off: (0, 0) stays at the middle of the box, as the projection put it.
settings.center_map = False
settings.use_offsets = False
xodr = carla.Osm2Odr.convert(osm, settings)  # the OpenDRIVE text, the same format as Town01.xodr

# carla.Map reads the OpenDRIVE text into roads, lanes and waypoints, with no server.
m = carla.Map("my_area", xodr)
root = ET.fromstring(xodr)
print(f"projection centered on lat {lat0:.5f}, lon {lon0:.5f}")
# <road> elements include the short roads inside intersections; <junction> elements are
# the intersections; generate_waypoints(2.0) puts a point every 2 m in every lane.
print(f"roads: {len(root.findall('road'))}, intersections: {len(root.findall('junction'))}, "
      f"waypoints at 2 m: {len(m.generate_waypoints(2.0))}")
open("my_area.xodr", "w").write(xodr)        # save it to open in a text editor or in CARLA
print("wrote my_area.xodr")
