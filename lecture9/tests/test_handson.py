import matplotlib
matplotlib.use("Agg")
import json
from pathlib import Path

import numpy as np
import pytest

import l9_handson as h
import l9common as c

HERE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("part", ["scene", "sample", "drive", "rrt", "search"])
def test_each_part_writes_a_snapshot(part, tmp_path):
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out)])
    assert out.exists() and out.stat().st_size > 10000


def test_load_search_names_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing.json"):
        h.load_search(tmp_path / "nothing.json")


def test_load_search_names_missing_key(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"toy": {}, "grid": {}}))
    with pytest.raises(KeyError, match="hybrid"):
        h.load_search(p)


def test_search_results_are_collision_free():
    d = h.load_search()
    rects = c.garage_obstacles()
    Q = np.array(d["hybrid"]["poses"])
    assert all(c.pose_free(x, y, t, rects) for x, y, t in Q)
    gx, gy, gt = c.garage_goal()
    assert np.hypot(Q[-1, 0] - gx, Q[-1, 1] - gy) < 0.2
    steps = np.array(d["toy"]["steps"])
    path = d["toy"]["path"]
    assert [steps[i][j] for i, j in path] == list(range(len(path)))
