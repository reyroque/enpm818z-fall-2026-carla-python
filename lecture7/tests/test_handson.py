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
