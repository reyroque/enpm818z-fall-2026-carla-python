"""2-D ICP, point to point (Besl and McKay, 1992), with the SVD fit (Kabsch).

Five steps per fit, as on the slides: (1) weighted centroids, (2) center the
points, (3) cross-covariance H, (4) SVD and the rotation, with the
reflection check, (5) the translation.
"""
import numpy as np
from scipy.spatial import cKDTree


class DegenerateFit(ValueError):
    """The pairs do not fix a rotation (too few, or all on one line)."""


def apply(R, t, pts):
    return np.asarray(pts, float) @ np.asarray(R).T + np.asarray(t)


def best_rigid_transform(P, Q, w=None):
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if len(P) < 2:
        raise DegenerateFit(f"need at least 2 pairs, got {len(P)}")
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    if w.sum() <= 0:
        raise DegenerateFit("total weight is zero")
    p_bar = (w[:, None] * P).sum(0) / w.sum()                 # step 1
    q_bar = (w[:, None] * Q).sum(0) / w.sum()
    Pc, Qc = P - p_bar, Q - q_bar                              # step 2
    H = (w[:, None] * Pc).T @ Qc                               # step 3
    U, S, Vt = np.linalg.svd(H)                                # step 4
    if S[0] == 0 or S[1] < 1e-9 * S[0]:
        raise DegenerateFit("points are collinear: the rotation is not unique")
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    t = q_bar - R @ p_bar                                      # step 5
    return R, t, {"singular_values": S, "reflection_fixed": bool(d < 0),
                  "p_bar": p_bar, "q_bar": q_bar, "H": H}


def icp(source, target, R0=None, t0=None, max_iter=30, tol=1e-6, max_dist=2.0):
    source, target = np.asarray(source, float), np.asarray(target, float)
    R = np.eye(2) if R0 is None else np.asarray(R0, float)
    t = np.zeros(2) if t0 is None else np.asarray(t0, float)
    tree = cKDTree(target)
    history, prev_rmse = [], None
    for _ in range(max_iter):
        moved = apply(R, t, source)
        dist, idx = tree.query(moved)
        keep = dist < max_dist
        if keep.sum() < 2:
            raise DegenerateFit(f"only {keep.sum()} pairs within {max_dist} m")
        dR, dt, _ = best_rigid_transform(moved[keep], target[idx[keep]])
        R, t = dR @ R, dR @ t + dt
        rmse = float(np.sqrt(np.mean(dist[keep] ** 2)))
        history.append({"R": R.copy(), "t": t.copy(), "rmse": rmse, "pairs": int(keep.sum())})
        step = np.linalg.norm(dt) + abs(np.arctan2(dR[1, 0], dR[0, 0]))
        if step < tol or (prev_rmse is not None and abs(prev_rmse - rmse) < tol):
            break
        prev_rmse = rmse
    return R, t, history
