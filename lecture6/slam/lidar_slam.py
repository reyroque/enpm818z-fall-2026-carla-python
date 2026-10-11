#!/usr/bin/env python3
"""
lidar_slam.py: 2-D LiDAR graph SLAM on the L6 drive.

Measures what the reading page "SLAM with a LiDAR" quotes, on the L6 recording
(../l6_drive.csv and ../l6_scans.npz: 1256 2-D scans at 10 Hz, two laps of one block,
916 m, ground truth for every scan):

  front end   scan-to-scan ICP odometry, and scan-to-submap ICP odometry
  keyframes   a new node every KEY_EVERY_M of travel
  loops       Scan Context (Kim and Kim, 2018) in 2-D: a polar grid of
              point counts, ring keys to propose, column shift for the yaw,
              ICP to verify
  back end    SE(2) pose graph, Gauss-Newton on the sparse normal equations
  maps        occupancy grids from the odometry poses and the optimized ones

Numbers go to measured_lidar.json; lidar_figures.py draws the figures.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

L6 = Path(__file__).resolve().parents[1]      # lecture6/: the drive, icp.py, l6common.py
sys.path.insert(0, str(L6))
import icp as icp_mod          # noqa: E402
import l6common                # noqa: E402

HERE = Path(__file__).resolve().parent
KEY_EVERY_M = 2.0              # chosen: a keyframe every 2 m of odometry travel
SUBMAP_KEYS = 10               # chosen: the submap is the last 10 keyframes
SC_RINGS, SC_SECTORS, SC_MAX_R = 20, 60, 50.0     # Scan Context grid (the paper's 20 x 60)
LOOP_MIN_GAP_M = 100.0         # chosen: a loop must close over 100 m of path or more
LOOP_ICP_RMSE = 0.25           # chosen: accept a loop if ICP's final RMSE is under 0.25 m


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def ang(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def compose(a, b):
    """Pose a then relative pose b (x, y, theta)."""
    x, y, t = a
    c, s = np.cos(t), np.sin(t)
    return np.array([x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], l6common.wrap(t + b[2])])


def between(a, b):
    """Relative pose of b seen from a."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, l6common.wrap(b[2] - a[2])])


# --------------------------------------------------------------- front end
def odometry(scans, mode):
    """ICP odometry. mode 'scan': each scan against the previous one.
    mode 'submap': each scan against the last SUBMAP_KEYS keyframes' points."""
    poses = [np.zeros(3)]
    keys = [0]
    dR, dt = np.eye(2), np.zeros(2)
    last_key_pose = np.zeros(3)
    for j in range(1, len(scans)):
        if mode == 'scan':
            Rr, tr, _ = icp_mod.icp(scans[j], scans[j - 1], R0=dR, t0=dt, max_dist=1.0)
            rel = np.array([tr[0], tr[1], ang(Rr)])
            poses.append(compose(poses[-1], rel))
            dR, dt = Rr, tr
        else:
            sub = []
            for k in keys[-SUBMAP_KEYS:]:
                p = poses[k]
                sub.append(icp_mod.apply(rot(p[2]), p[:2], scans[k]))
            sub = np.vstack(sub)
            guess = compose(poses[-1], np.array([dt[0], dt[1], ang(dR)]))
            Rw, tw, _ = icp_mod.icp(scans[j], sub, R0=rot(guess[2]), t0=guess[:2], max_dist=1.0)
            pose = np.array([tw[0], tw[1], ang(Rw)])
            rel = between(poses[-1], pose)
            dR, dt = rot(rel[2]), rel[:2]
            poses.append(pose)
        if np.hypot(*(poses[-1][:2] - last_key_pose[:2])) >= KEY_EVERY_M:
            keys.append(j)
            last_key_pose = poses[-1]
    return np.array(poses), keys


# ------------------------------------------------------------ scan context
def scan_context(pts):
    r = np.hypot(pts[:, 0], pts[:, 1])
    a = np.arctan2(pts[:, 1], pts[:, 0]) % (2 * np.pi)
    ok = r < SC_MAX_R
    ri = np.minimum((r[ok] / SC_MAX_R * SC_RINGS).astype(int), SC_RINGS - 1)
    si = np.minimum((a[ok] / (2 * np.pi) * SC_SECTORS).astype(int), SC_SECTORS - 1)
    sc = np.zeros((SC_RINGS, SC_SECTORS))
    np.add.at(sc, (ri, si), 1.0)
    return sc


def sc_distance(a, b):
    """Smallest mean column cosine distance over all column shifts, and the shift."""
    best, best_s = 2.0, 0
    na = np.linalg.norm(a, axis=0)
    for s in range(SC_SECTORS):
        bs = np.roll(b, s, axis=1)
        nb = np.linalg.norm(bs, axis=0)
        ok = (na > 0) & (nb > 0)
        if ok.sum() == 0:
            continue
        cos = (a[:, ok] * bs[:, ok]).sum(0) / (na[ok] * nb[ok])
        d = 1.0 - cos.mean()
        if d < best:
            best, best_s = d, s
    return best, best_s


# ------------------------------------------------------------- pose graph
def optimize(nodes, edges, iters=20, robust=None, robust_c=1.0, robust_from=None):
    """Gauss-Newton on an SE(2) pose graph. nodes: (n, 3). edges: list of
    (i, j, z (3,), Omega (3x3)). Node 0 is held fixed. Returns the nodes,
    the cost after each iteration and the sparsity of H.

    robust='cauchy' reweights every edge from index robust_from on (the
    loop edges) by w = 1 / (1 + chi2 / c^2), chi2 = e^T Omega e, each
    iteration (iteratively reweighted least squares): an edge that disagrees
    with the rest of the graph counts less and less."""
    x = nodes.copy()
    n = len(x)
    costs = []
    weights = np.ones(len(edges))
    for it in range(iters):
        rows, cols, vals = [], [], []
        b = np.zeros(3 * n)
        cost = 0.0
        for ei, (i, j, z, Om) in enumerate(edges):
            xi, xj = x[i], x[j]
            e = between(xi, xj) - z
            e[2] = l6common.wrap(e[2])
            if robust == 'cauchy' and ei >= robust_from:
                chi2 = float(e @ Om @ e)
                weights[ei] = 1.0 / (1.0 + chi2 / robust_c ** 2)
                Om = weights[ei] * Om
            c, s = np.cos(xi[2]), np.sin(xi[2])
            dx, dy = xj[0] - xi[0], xj[1] - xi[1]
            A = np.array([[-c, -s, -s * dx + c * dy],
                          [s, -c, -c * dx - s * dy],
                          [0, 0, -1.0]])
            B = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
            cost += float(e @ Om @ e)
            for (p, Jp) in ((i, A), (j, B)):
                b[3 * p:3 * p + 3] += Jp.T @ Om @ e
                for (q, Jq) in ((i, A), (j, B)):
                    blk = Jp.T @ Om @ Jq
                    for r in range(3):
                        for cc in range(3):
                            rows.append(3 * p + r); cols.append(3 * q + cc); vals.append(blk[r, cc])
        costs.append(cost)
        H = sp.coo_matrix((vals, (rows, cols)), shape=(3 * n, 3 * n)).tocsr()
        H = H + sp.diags(np.r_[np.full(3, 1e9), np.zeros(3 * n - 3)])   # hold node 0
        dxs = spla.spsolve(H.tocsc(), -b)
        x = x + dxs.reshape(n, 3)
        x[:, 2] = l6common.wrap(x[:, 2])
        if np.abs(dxs).max() < 1e-6:
            break
    # final cost
    cost = 0.0
    for i, j, z, Om in edges:
        e = between(x[i], x[j]) - z; e[2] = l6common.wrap(e[2])
        cost += float(e @ Om @ e)
    costs.append(cost)
    nnz = H.nnz
    return x, costs, {'size': 3 * n, 'nonzeros': int(nnz), 'share_nonzero': nnz / float((3 * n) ** 2),
                      'final_weights': weights}


# ----------------------------------------------------------- occupancy grid
def occupancy(poses, scans, cell=0.2, lo_hit=0.85, lo_miss=-0.4):
    pts_w = np.vstack([icp_mod.apply(rot(p[2]), p[:2], s) for p, s in zip(poses, scans)])
    lo, hi = pts_w.min(0) - 5, pts_w.max(0) + 5
    shape = tuple(np.ceil((hi - lo) / cell).astype(int))
    L = np.zeros(shape)
    for p, s in zip(poses, scans):
        w = icp_mod.apply(rot(p[2]), p[:2], s)
        o = ((p[:2] - lo) / cell).astype(int)
        e = ((w - lo) / cell).astype(int)
        for ex, ey in e[::2]:
            n = int(max(abs(ex - o[0]), abs(ey - o[1])))
            if n == 0:
                continue
            xs = np.linspace(o[0], ex, n, endpoint=False).astype(int)
            ys = np.linspace(o[1], ey, n, endpoint=False).astype(int)
            L[xs, ys] += lo_miss
        np.add.at(L, (e[:, 0], e[:, 1]), lo_hit)
    return L, lo, cell


def main():
    t0 = time.time()
    d = l6common.load_drive(L6 / 'l6_drive.csv')
    st, scans = l6common.load_scans(L6 / 'l6_scans.npz')
    ks = np.array([int(np.argmin(np.abs(d['t'] - s))) for s in st])
    truth = np.c_[d['x'][ks], d['y'][ks], d['theta'][ks]]
    path = np.r_[0, np.cumsum(np.hypot(*np.diff(truth[:, :2], axis=0).T))]
    m = {'scans': len(scans), 'drive_length_m': round(float(path[-1]), 1)}

    def to_world(rel_poses):
        """Odometry starts at (0, 0, 0); put it at the first true pose."""
        return np.array([compose(truth[0], p) for p in rel_poses])

    res = {}
    for mode in ('scan', 'submap'):
        tic = time.time()
        rel, keys = odometry(scans, mode)
        w = to_world(rel)
        err = np.hypot(*(w[:, :2] - truth[:, :2]).T)
        res[mode] = (w, keys)
        m[f'odometry_{mode}'] = {'final_error_m': round(float(err[-1]), 2),
                                 'rmse_m': round(float(np.sqrt(np.mean(err ** 2))), 2),
                                 'max_error_m': round(float(err.max()), 2),
                                 'heading_error_end_deg': round(float(np.degrees(l6common.wrap(w[-1, 2] - truth[-1, 2]))), 2),
                                 'seconds': round(time.time() - tic, 1)}
        print(mode, m[f'odometry_{mode}'])
    w, keys = res['submap']
    keys = np.array(keys)
    nodes = w[keys].copy()
    m['keyframes'] = {'every_m': KEY_EVERY_M, 'count': int(len(keys))}

    # odometry edges between consecutive keyframes
    Om_odo = np.diag([1 / 0.05 ** 2, 1 / 0.05 ** 2, 1 / np.radians(0.5) ** 2])  # chosen: 5 cm, 0.5 deg per 2 m step
    edges = [(i, i + 1, between(nodes[i], nodes[i + 1]), Om_odo) for i in range(len(nodes) - 1)]

    # loops: scan context on keyframes
    scs = [scan_context(scans[k]) for k in keys]
    rk = np.array([s.sum(1) / max(s.sum(), 1) for s in scs])
    kpath = path[keys]
    tree = cKDTree(rk)
    cand, loops, rejected = 0, [], 0
    true_pairs = 0
    for i in range(len(keys)):
        older = [j for j in tree.query(rk[i], k=10)[1] if kpath[i] - kpath[j] > LOOP_MIN_GAP_M]
        for j in older[:3]:
            dist, shift = sc_distance(scs[i], scs[j])
            if dist > 0.35:          # chosen: Scan Context distance gate
                continue
            cand += 1
            yaw = -shift * 2 * np.pi / SC_SECTORS
            try:
                R, t, h = icp_mod.icp(scans[keys[i]], scans[keys[j]], R0=rot(yaw), t0=np.zeros(2), max_dist=2.0)
            except icp_mod.DegenerateFit:
                rejected += 1
                continue
            if h[-1]['rmse'] > LOOP_ICP_RMSE:
                rejected += 1
                continue
            z = np.array([t[0], t[1], ang(R)])           # pose of i seen from j
            # A loop is right when its measured relative pose agrees with the true one.
            zt = between(truth[keys[j]], truth[keys[i]])
            near_truth = bool(np.hypot(*(z[:2] - zt[:2])) < 1.0 and abs(l6common.wrap(z[2] - zt[2])) < np.radians(5))
            true_pairs += int(near_truth)
            loops.append((j, i, z, np.diag([1 / 0.1 ** 2, 1 / 0.1 ** 2, 1 / np.radians(1.0) ** 2]), bool(near_truth)))
            break
    m['loops'] = {'candidates': cand, 'accepted': len(loops), 'rejected_by_icp': rejected,
                  'accepted_right_within_1m_5deg': true_pairs,
                  'scan_context': {'rings': SC_RINGS, 'sectors': SC_SECTORS, 'max_range_m': SC_MAX_R,
                                   'distance_gate': 0.35}, 'min_gap_m': LOOP_MIN_GAP_M,
                  'icp_rmse_gate_m': LOOP_ICP_RMSE}
    print('loops', m['loops'])
    all_edges = edges + [(j, i, z, Om) for j, i, z, Om, _ in loops]

    before = nodes.copy()
    tic = time.time()
    after, costs, sparsity = optimize(nodes, all_edges)
    sparsity.pop('final_weights')
    tk = truth[keys]
    eb = np.hypot(*(before[:, :2] - tk[:, :2]).T)
    ea = np.hypot(*(after[:, :2] - tk[:, :2]).T)
    m['pose_graph'] = {'nodes': int(len(nodes)), 'odometry_edges': len(edges), 'loop_edges': len(loops),
                       'iterations': len(costs) - 1, 'cost_per_iteration': [round(c, 1) for c in costs],
                       'seconds': round(time.time() - tic, 2), **sparsity,
                       'ape_rmse_before_m': round(float(np.sqrt(np.mean(eb ** 2))), 2),
                       'ape_rmse_after_m': round(float(np.sqrt(np.mean(ea ** 2))), 2),
                       'max_error_before_m': round(float(eb.max()), 2),
                       'max_error_after_m': round(float(ea.max()), 2)}
    print('graph', {k: v for k, v in m['pose_graph'].items() if k != 'cost_per_iteration'})
    np.savez(HERE / 'lidar_slam_results.npz', truth=truth, keys=keys, before=before, after=after,
             scan_odo=res['scan'][0], submap_odo=w, path=path,
             loops=np.array([(j, i, int(tp)) for j, i, z, Om, tp in loops]))
    m['seconds_total'] = round(time.time() - t0, 1)
    (HERE / 'measured_lidar.json').write_text(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
