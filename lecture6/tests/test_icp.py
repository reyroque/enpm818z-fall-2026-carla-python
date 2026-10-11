import numpy as np
import pytest
import icp


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_fit_recovers_known_transform():
    rng = np.random.default_rng(6)
    P = rng.uniform(-10, 10, (50, 2))
    R_true, t_true = rot(np.radians(12.0)), np.array([1.5, -0.7])
    Q = P @ R_true.T + t_true
    R, t, info = icp.best_rigid_transform(P, Q)
    assert np.allclose(R, R_true, atol=1e-9) and np.allclose(t, t_true, atol=1e-9)


def test_fit_worked_example_three_pairs_unique():
    # The slide's worked example: three non-collinear pairs, rotation 90 deg, shift (2, 1).
    P = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -2.0]])
    Q = P @ rot(np.pi / 2).T + np.array([2.0, 1.0])
    R, t, info = icp.best_rigid_transform(P, Q)
    assert np.allclose(R, rot(np.pi / 2), atol=1e-12) and np.allclose(t, [2.0, 1.0], atol=1e-12)
    assert info["singular_values"][1] > 1.0


def test_fit_never_returns_reflection():
    P = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    Q = P * np.array([1.0, -1.0])   # a mirror image: the best rotation, not the mirror
    R, _, info = icp.best_rigid_transform(P, Q)
    assert np.isclose(np.linalg.det(R), 1.0) and info["reflection_fixed"]


def test_icp_degenerate_collinear():
    P = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])   # one straight wall
    with pytest.raises(icp.DegenerateFit):
        icp.best_rigid_transform(P, P + [0.5, 0.0])


def room(n):
    """A U-shaped room: one 20 m wall and two 10 m walls, n points per 5 m."""
    return np.vstack([np.column_stack([np.linspace(0, 20, 4 * n), np.zeros(4 * n)]),
                      np.column_stack([np.zeros(2 * n), np.linspace(0, 10, 2 * n)]),
                      np.column_stack([np.full(2 * n, 20.0), np.linspace(0, 10, 2 * n)])])


def test_icp_converges_from_offset():
    # A sparse scan against a dense map, as on the car: the scan's points are
    # not the map's points, so nearest neighbours slide along the walls.
    rng = np.random.default_rng(7)
    target = room(200)
    R_true, t_true = rot(np.radians(4.0)), np.array([0.6, -0.4])
    scan = room(20)[1:-1]
    source = (scan - t_true) @ R_true + rng.normal(0, 0.01, scan.shape)   # target = R_true * source + t_true
    R, t, hist = icp.icp(source, target, max_dist=2.0)
    assert np.allclose(R, R_true, atol=2e-3) and np.allclose(t, t_true, atol=2e-2)
    assert hist[-1]["rmse"] < hist[0]["rmse"]
