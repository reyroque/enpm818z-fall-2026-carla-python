import numpy as np
import pytest

import pipeline_sim as ps


def const(ms):
    return {"yolov8s_gpu": np.array([ms]), "slow": np.array([ms])}


def test_keep_last_drops_the_oldest():
    # frames every 10 ms, 25 ms each, depth 2: the worker can never keep up
    arrive = np.arange(10) * 10.0
    start, done = ps.detector_queue(arrive, np.full(10, 25.0), depth=2)
    # frame 0 is taken at once; 1 and 2 wait; 3 arrives, then 4 pushes 2 out of the
    # full queue (keep last drops the oldest)
    assert np.isfinite(done[0]) and np.isfinite(done[1])
    assert np.isnan(done[2])
    assert np.all(np.diff(start[np.isfinite(start)]) >= 25.0)


def test_depth_one_never_waits_more_than_one_service():
    arrive = np.arange(200) * 50.0
    start, done = ps.detector_queue(arrive, np.full(200, 75.0), depth=1)
    ok = np.isfinite(start)
    assert np.all(start[ok] - arrive[ok] <= 75.0 + 1e-9)


def test_hand_chain_is_100_ms():
    # the slides' worked chain: phases 30 and 15 ms, detector 5.0 ms
    cfg = ps.Config(behavior_phase_ms=30.0, control_phase_ms=15.0, duration_s=3.0)
    r = ps.run(cfg, const(5.0), seed=0)
    assert ps.reaction_ms(r)[20] == 100.0


def test_reaction_is_on_the_tick_grid_and_bounded_by_hand():
    rng = np.random.default_rng(1)
    for s in range(10):
        cfg = ps.random_phases(ps.Config(duration_s=10.0), rng)
        lat = ps.reaction_ms(ps.run(cfg, const(5.0), seed=s))[10:-10]
        assert np.all(lat % 50.0 == 0)
        # worst case by hand: 3 + 5 + 4 + 100 + 1 + 50 + 1 + 1 = 165 ms, next tick 200
        assert 50.0 <= lat.min() and lat.max() <= 200.0


def test_freeze_makes_the_age_grow_and_the_timeout_notices():
    cfg = ps.Config(duration_s=30.0, freeze_at_ms=20000.0, behavior_phase_ms=37.0)
    r = ps.run(cfg, const(5.0), seed=0)
    alarm = ps.first_alarm_after(r, 70.0, 20000.0)
    assert 20000.0 < alarm <= 20000.0 + 70.0 + 100.0
    assert ps.ages_at_behavior(r)[-1] > 9000


def test_no_alarm_in_normal_driving_with_a_safe_timeout():
    cfg = ps.Config(duration_s=30.0, behavior_phase_ms=37.0)
    r = ps.run(cfg, const(5.0), seed=0)
    assert np.isnan(ps.first_alarm_after(r, 70.0, 1000.0))


def test_stopping_distance():
    assert ps.stopping_distance_m(8.0, 0.2, 6.0) == pytest.approx(1.6 + 64 / 12)


def test_measured_times_file_loads():
    times, meta = ps.load_times()
    assert set(times) >= {"yolov8s_gpu", "rtdetr_l_gpu", "yolov8s_cpu"}
    assert all(len(v) >= 100 and np.all(v > 0) for v in times.values())
