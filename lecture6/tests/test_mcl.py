import numpy as np
import mcl


def box_map():
    """A 20 m by 10 m box with a 1 m pillar off center. Without the pillar the
    box looks the same turned half a turn, and MCL would keep two clusters."""
    s = np.linspace(0, 20, 201)
    e = np.linspace(0, 1, 11)
    return np.vstack([np.column_stack([s, np.zeros_like(s)]), np.column_stack([s, np.full_like(s, 10)]),
                      np.column_stack([np.zeros(101), np.linspace(0, 10, 101)]),
                      np.column_stack([np.full(101, 20.0), np.linspace(0, 10, 101)]),
                      np.column_stack([14 + e, np.full(11, 7.0)]), np.column_stack([14 + e, np.full(11, 8.0)]),
                      np.column_stack([np.full(11, 14.0), 7 + e]), np.column_stack([np.full(11, 15.0), 7 + e])])


def scan_from(pose, mp, max_range=15.0):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    d = mp - [x, y]
    vehicle = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])
    return vehicle[np.linalg.norm(vehicle, axis=1) < max_range][::3]


def test_systematic_resample_keeps_count_and_favors_heavy():
    rng = np.random.default_rng(1)
    w = np.array([0.7, 0.1, 0.1, 0.1])
    idx = mcl.systematic_resample(w, rng)
    assert len(idx) == 4 and np.sum(idx == 0) >= 2


def test_neff():
    assert np.isclose(mcl.neff(np.full(10, 0.1)), 10.0)
    assert np.isclose(mcl.neff(np.array([1.0, 0, 0])), 1.0)


def test_weigh_prefers_true_pose():
    mp = box_map()
    field = mcl.LikelihoodField(mp)
    true = np.array([8.0, 4.0, 0.3])
    parts = np.array([true, [12.0, 6.0, 0.3], [8.0, 4.0, 1.3]])
    w, lost = mcl.weigh(parts, scan_from(true, mp), field)
    assert np.argmax(w) == 0 and not lost


def test_weigh_all_zero_likelihood_gives_uniform():
    field = mcl.LikelihoodField(box_map(), z_rand=0.0, max_dist=2.0)
    parts = np.array([[500.0, 500.0, 0.0], [600.0, 600.0, 0.0]])
    w, lost = mcl.weigh(parts, np.array([[1.0, 0.0], [2.0, 0.0]]), field)
    assert lost and np.allclose(w, [0.5, 0.5]) and not np.any(np.isnan(w))


def test_mcl_finds_pose_from_lost_start():
    rng = np.random.default_rng(818)
    mp = box_map()
    field = mcl.LikelihoodField(mp)
    true = np.array([4.0, 2.0, 0.0])
    t = np.arange(0, 5.01, 0.1)
    scan_t = t[::5]
    poses = [true + [k * 0.1 * 1.0, 0, 0] for k in range(len(t))]
    scans = [scan_from(poses[int(round(s / 0.1))], mp) for s in scan_t]
    out = mcl.run_mcl(t, np.full_like(t, 1.0), np.zeros_like(t), scan_t, scans, field,
                      n=2000, rng=rng, init="lost", bounds=(0, 20, 0, 10))
    final_true = poses[int(round(scan_t[-1] / 0.1))]
    assert np.linalg.norm(out["mean"][-1, :2] - final_true[:2]) < 1.0


def test_mcl_coarse_start_converges():
    rng = np.random.default_rng(818)
    mp = box_map()
    field = mcl.LikelihoodField(mp)
    true = np.array([4.0, 2.0, 0.0])
    t = np.arange(0, 5.01, 0.1)
    scan_t = t[::5]
    poses = [true + [k * 0.1, 0, 0] for k in range(len(t))]
    scans = [scan_from(poses[int(round(s / 0.1))], mp) for s in scan_t]
    out = mcl.run_mcl(t, np.full_like(t, 1.0), np.zeros_like(t), scan_t, scans, field,
                      n=1000, rng=rng, init="coarse", init_pose=(6.0, 3.0), radius=4.0)
    assert np.linalg.norm(out["mean"][-1, :2] - poses[-1][:2]) < 1.0
