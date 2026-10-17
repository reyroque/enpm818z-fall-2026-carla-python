import matplotlib
matplotlib.use("Agg")
import pytest

import l11_handson as h
import l11common as c


@pytest.fixture(autouse=True)
def chosen_values():
    keep = dict(c.CHOSEN)
    yield
    c.CHOSEN.clear()
    c.CHOSEN.update(keep)


@pytest.mark.parametrize("part", ["road", "bc", "shift", "dagger", "eval"])
def test_each_part_writes_a_snapshot(part, tmp_path):
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out), "--iterations", "3"])
    assert out.exists() and out.stat().st_size > 10000
