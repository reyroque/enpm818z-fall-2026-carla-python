"""The precomputed runs: present, complete, and saying what the slides say."""
import numpy as np

import l10common as c

R = c.load_results()


def pid_run(kp, ki, kd, clamp):
    g = R["pid_grid"]["gains"]
    m = (g[:, 0] == kp) & (g[:, 1] == ki) & (g[:, 2] == kd)
    m &= np.isnan(g[:, 3]) if clamp is None else (g[:, 3] == clamp)
    return R["pid_grid"]["v"][int(np.flatnonzero(m)[0])]


def test_p_alone_leaves_a_steady_error():
    v = pid_run(0.5, 0.0, 0.0, None)
    assert 0.3 < 8.0 - v[np.isfinite(v)][-1] < 0.6


def test_the_integral_removes_it_and_winds_up_without_the_clamp():
    free = pid_run(0.5, 0.3, 0.0, None)
    clamped = pid_run(0.5, 0.3, 0.0, 1.0)
    assert abs(free[np.isfinite(free)][-1] - 8.0) < 0.01
    assert np.nanmax(free) > 11.0 and np.nanmax(clamped) < 9.0


def test_every_track_run_reaches_the_end_of_the_turn():
    g = R["track_grid"]
    for i in range(len(g["labels"])):
        s = g["s"][i][np.isfinite(g["s"][i])]
        assert s.max() > 120.0


def test_summary_matches_the_runs():
    w = R["worked"]
    assert len(w["track_runs"]) == len(R["track_grid"]["labels"])
    assert len(w["pid_runs"]) == len(R["pid_grid"]["gains"])
    assert R["mpc_candidates"]["x"].shape[0] == w["mpc"]["n_candidates"]
