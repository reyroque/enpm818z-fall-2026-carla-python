import numpy as np
import pytest
import l6common as c


def test_wrap_range():
    a = np.array([np.pi, -np.pi, 3 * np.pi, 0.1])
    w = c.wrap(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.isclose(w[3], 0.1)


def test_load_drive_names_missing_column(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("t,x,y\n0,0,0\n")
    with pytest.raises(ValueError, match="theta"):
        c.load_drive(p)


def test_load_drive_empty_cells_are_nan(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(",".join(c.DRIVE_COLUMNS) + "\n0,0,0,0,1,0,1,,\n")
    d = c.load_drive(p)
    assert np.isnan(d["gnss_x"][0]) and d["v"][0] == 1.0


def test_dead_reckon_straight_line():
    t = np.arange(0, 10.01, 0.1)
    x, y, th = c.dead_reckon(t, np.full_like(t, 2.0), np.zeros_like(t), 0.0, 0.0, 0.0)
    assert np.isclose(x[-1], 20.0) and np.isclose(y[-1], 0.0)


def test_dead_reckon_quarter_circle():
    t = np.arange(0, 10.0001, 0.001)
    w = np.pi / 2 / 10.0
    x, y, th = c.dead_reckon(t, np.full_like(t, 1.0), np.full_like(t, w), 0.0, 0.0, 0.0)
    r = 1.0 / w
    assert np.isclose(x[-1], r, atol=0.02) and np.isclose(y[-1], r, atol=0.02)


def test_dead_reckon_holds_last_valid():
    t = np.arange(0, 1.01, 0.1)
    v = np.full_like(t, 1.0)
    v[5] = np.nan
    x, _, _ = c.dead_reckon(t, v, np.zeros_like(t), 0.0, 0.0, 0.0)
    assert np.isclose(x[-1], 1.0)


def test_ape_ignores_nan_rows():
    est = np.array([[0.0, 0.0], [np.nan, np.nan], [3.0, 4.0]])
    tru = np.zeros((3, 2))
    r = c.ape(est, tru)
    assert np.isclose(r["max"], 5.0) and np.isnan(r["errors"][1])
    assert np.isclose(r["rmse"], np.sqrt((0 + 25) / 2))


def test_ape_all_nan_raises():
    with pytest.raises(ValueError, match="no valid rows"):
        c.ape(np.full((2, 2), np.nan), np.zeros((2, 2)))


def test_scans_round_trip(tmp_path):
    scans = [np.array([[1.0, 0.0], [2.0, 1.0]]), np.zeros((0, 2)), np.array([[5.0, -1.0]])]
    c.save_scans(tmp_path / "s.npz", np.array([0.0, 0.1, 0.2]), scans)
    t, back = c.load_scans(tmp_path / "s.npz")
    assert len(back) == 3 and back[1].shape == (0, 2) and np.allclose(back[2], [[5.0, -1.0]])
