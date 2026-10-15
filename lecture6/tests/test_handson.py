import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pytest
import l6_handson as h

HERE = Path(__file__).resolve().parents[1]


def test_load_results_names_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_gps0_wheels1_map0.npz"):
        h.load_results(tmp_path, outage=True, wheels=True, map_=False)


def test_load_results_names_missing_key(tmp_path):
    np.savez(tmp_path / "run_gps0_wheels1_map0.npz", t=np.zeros(2))
    with pytest.raises(KeyError, match="xy"):
        h.load_results(tmp_path, outage=True, wheels=True, map_=False)


@pytest.mark.parametrize("part", ["gnss", "dr", "ekf", "mcl", "icp"])
def test_each_part_writes_a_snapshot(part, tmp_path):
    if not (HERE / "l6_drive.csv").exists():
        pytest.skip("data not recorded yet")
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out), "--particles", "300"])
    assert out.exists() and out.stat().st_size > 10000


def test_gnss_stats_match_the_errors_we_chose():
    # The recorder adds a bias of 0.5 m east and 0.3 m south and noise sigma 1.5 m.
    # A mean of n fixes is off by about 1.5 / sqrt(n); allow three of those.
    if not (HERE / "l6_drive.csv").exists():
        pytest.skip("data not recorded yet")
    import l6common as c
    s = h.gnss_stats(c.load_drive(HERE / "l6_drive.csv"))
    tol = 3 * 1.5 / np.sqrt(s["n"])
    assert abs(s["mean"][0] - 0.5) < tol and abs(s["mean"][1] + 0.3) < tol
    assert 1.0 < s["std"][0] < 2.0 and 1.0 < s["std"][1] < 2.0
    assert 50 < s["outage_s"] < 60 and 380 < s["outage_m"] < 420


def test_mcl_kidnap_and_random_write_a_snapshot(tmp_path):
    if not (HERE / "l6_drive.csv").exists():
        pytest.skip("data not recorded yet")
    out = tmp_path / "kidnap.png"
    h.main(["--part", "mcl", "--snapshot", str(out), "--particles", "300",
            "--kidnap", "40", "0", "10", "--random", "20"])
    assert out.exists() and out.stat().st_size > 10000
