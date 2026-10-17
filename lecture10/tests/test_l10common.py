import numpy as np

import l10common as c


def test_curve_is_the_town01_turn():
    d = c.load_curve()
    assert abs(d["s"][-1] - 135.1) < 0.5
    assert tuple(d["roads"][0]) == (12, 1) and tuple(d["roads"][-1]) == (23, 1)
    # a right turn of 90 degrees: the heading ends at -pi/2
    assert abs(d["heading"][-1] + np.pi / 2) < 0.01
    # two arcs: radius 7.64 m and 4.97 m
    k = np.abs(d["kappa"])
    assert abs(1 / np.median(k[(d["s"] > 83) & (d["s"] < 87)]) - 7.64) < 0.05
    assert abs(1 / np.median(k[(d["s"] > 88) & (d["s"] < 91.5)]) - 4.97) < 0.05


def test_bicycle_step_turns_on_the_circle():
    delta, v = 0.3, 5.0
    R = c.turning_radius(delta)
    xs, ys, _ = c.drive_constant(0.0, 0.0, 0.0, v, delta, 2.0, dt=0.001)
    # center of the circle is R to the left of the start
    assert np.allclose(np.hypot(xs, ys - R), R, atol=0.02)


def test_speed_profile_respects_its_limits():
    d = c.load_curve()
    v = c.speed_profile(d["s"], d["kappa"])
    assert v.max() <= 8.0 + 1e-9
    assert np.all(v ** 2 * np.abs(d["kappa"]) <= 2.0 + 1e-6)
    dv2 = np.diff(v ** 2) / (2 * np.diff(d["s"]))
    assert dv2.max() <= 1.5 + 1e-6 and dv2.min() >= -2.0 - 1e-6


def test_quintic_meets_its_six_conditions():
    q = c.quintic(0.0, 8.0, 0.0, 20.0, 3.0, 0.0, 3.5)
    T = 3.5
    got = [c.poly_eval(q, 0, 0), c.poly_eval(q, 0, 1), c.poly_eval(q, 0, 2),
           c.poly_eval(q, T, 0), c.poly_eval(q, T, 1), c.poly_eval(q, T, 2)]
    assert np.allclose(got, [0, 8, 0, 20, 3, 0], atol=1e-9)


def test_slow_down_quintic_stays_within_the_brake_limit():
    sd = c.slow_down(8.0, 3.153, 2.0)
    assert abs(sd["quintic"]["peak_a"] - 2.0) < 1e-3
    assert sd["quintic"]["D"] > sd["constant"]["D"]
