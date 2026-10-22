import matplotlib
matplotlib.use("Agg")
import pytest
import l7_handson as h


@pytest.mark.parametrize("part", ["graph", "costs", "search", "replan"])
def test_each_part_writes_a_snapshot(part, tmp_path):
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out)])
    assert out.exists() and out.stat().st_size > 10000


def test_map_part_reads_the_file(tmp_path):
    pytest.importorskip("carla")
    out = tmp_path / "map.png"
    h.main(["--part", "map", "--snapshot", str(out)])
    assert out.exists() and out.stat().st_size > 10000


def test_read_map_prints_the_slide_numbers():
    pytest.importorskip("carla")
    import subprocess
    import sys
    from pathlib import Path
    out = subprocess.run([sys.executable, "read_map.py"], cwd=Path(h.__file__).parent,
                         capture_output=True, text=True, check=True).stdout
    assert "len(wps): 3266" in out and "len(segs): 160" in out and "file: 36" in out


def test_map_search_finds_roads_lanes_junctions_segments():
    pytest.importorskip("carla")
    M = h.load_map()
    pts, msg = h.map_search("road 25", M)
    assert pts is not None and "35.18 m" in msg and "junction 54" in msg and "junction 332" in msg
    pts, msg = h.map_search("lane 25 -1", M)
    assert pts is not None and "(south)" in msg
    assert h.map_search("junction 332", M)[0] is not None
    assert h.map_search("segment 12", M)[0] is not None
    assert h.map_search("road 999", M)[0] is None
    assert h.map_search("segment 999", M)[0] is None


def test_map_point_finds_our_start_on_road_4_lane_1():
    pytest.importorskip("carla")
    M = h.load_map()
    pts, msg = h.map_point(133.6, -129.4, M)
    assert "road 4, lane 1, s = 32.2 m" in msg and "line 407" in msg
    assert h.map_search("133.6 -129.4", M)[1] == msg


def test_osm_to_map_converts_the_campus_as_the_slide_says(tmp_path):
    pytest.importorskip("carla")
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    here = Path(h.__file__).parent
    shutil.copytree(here / "maps", tmp_path / "maps")
    out = subprocess.run([sys.executable, str(here / "osm_to_map.py")], cwd=tmp_path,
                         capture_output=True, text=True, check=True).stdout
    assert "roads: 314, intersections: 46, waypoints at 2 m: 9144" in out
    assert (tmp_path / "my_area.xodr").exists()
