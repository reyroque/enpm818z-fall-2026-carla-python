import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import numpy as np
import pytest

import l10_handson as h
import l10common as c

HERE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("part", ["bicycle", "profile", "pid", "track"])
def test_each_part_writes_a_snapshot(part, tmp_path):
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out)])
    assert out.exists() and out.stat().st_size > 10000


def test_every_track_run_opens():
    res = c.load_results()
    for lab in res["track_grid"]["labels"]:
        h._label_text(str(lab))


def test_load_results_names_a_missing_file(tmp_path):
    (tmp_path / "worked.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="pid_grid.npz"):
        c.load_results(tmp_path)
