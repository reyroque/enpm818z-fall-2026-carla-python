import matplotlib
matplotlib.use("Agg")
import pytest

import l13_handson as h


@pytest.mark.parametrize("part", list(h.PARTS))
def test_each_part_writes_a_snapshot(part, tmp_path):
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out)])
    assert out.exists() and out.stat().st_size > 10000
