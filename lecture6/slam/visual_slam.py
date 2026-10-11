#!/usr/bin/env python3
"""
visual_slam.py: stereo visual SLAM on the L6 route.

Measures what the reading page "Visual SLAM" quotes, on stereo/ (record_stereo.py:
1256 grayscale stereo pairs at 10 Hz, two laps, 914 m, true poses):

  features    ORB (Rublee et al., 2011), 4000 per image
  stereo      the disparity d at each left feature, from a dense
              semi-global matching map (Hirschmuller, 2008); depth Z = f b / d
  tracking    frame k+1 against frame k: 3-D points of k, 2-D features of
              k+1, PnP with RANSAC
  mono        the essential matrix between two frames: the direction of
              travel, and why one camera cannot give its length
  loops       a bag of visual words (k-means on ORB bits, tf-idf, cosine),
              then PnP to verify
  back end    the SE(2) pose graph of lidar_slam.py, on keyframes every 2 m

Numbers to measured_visual.json, results to visual_slam_results.npz.
"""

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import lidar_slam as ls

HERE = Path(__file__).resolve().parent
ST = HERE / 'stereo'
N_FEATURES = 4000          # chosen
MAX_DEPTH_M = 40.0         # chosen: farther points have too little disparity to trust
RATIO = 0.8                # chosen: Lowe's ratio test for temporal matches
KEY_EVERY_M = 2.0          # chosen: as for the LiDAR
VOCAB = 500                # chosen: words in the vocabulary
LOOP_MIN_GAP_M = 100.0     # chosen: as for the LiDAR
LOOP_MIN_INLIERS = 60      # chosen: PnP inliers to accept a loop
C2O = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], float)   # CARLA camera axes -> optical


def load():
    if not (ST / 'meta.json').exists():
        raise SystemExit(f'No recording in {ST}. Record it first:\n'
                         '  python3 record_stereo.py --route ../route.json --out stereo --port 2000 --tm-port 8000')
    meta = json.loads((ST / 'meta.json').read_text())
    P = np.loadtxt(ST / 'poses.csv', delimiter=',')
    T_carla = P[:, 1:17].reshape(-1, 4, 4)
    # Optical-frame pose of the left camera: world <- optical
    To = np.zeros_like(T_carla)
    To[:, :3, :3] = T_carla[:, :3, :3] @ C2O.T
    To[:, :3, 3] = T_carla[:, :3, 3]
    To[:, 3, 3] = 1
    return meta, P, To


def planar(T):
    """(x east, y north, yaw ccw) of an optical-frame camera pose in CARLA world axes."""
    fwd = T[:3, :3] @ np.array([0, 0, 1.0])            # optical z = forward
    return np.array([T[0, 3], -T[1, 3], math.atan2(-fwd[1], fwd[0])])


def features(orb, img):
    kp, des = orb.detectAndCompute(img, None)
    return np.array([k.pt for k in kp], float), des


SGBM = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                             P1=8 * 5 * 5, P2=32 * 5 * 5, uniquenessRatio=10,
                             speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def stereo_points(K, b, kl, left, right):
    """3-D points (optical frame of the left camera) for left features, with the
    depth read from a dense disparity map (semi-global block matching,
    Hirschmuller, 2008), to a sixteenth of a pixel. Z = f b / d."""
    disp = SGBM.compute(left, right).astype(np.float32) / 16.0
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    u = np.clip(np.round(kl[:, 0]).astype(int), 0, left.shape[1] - 1)
    v = np.clip(np.round(kl[:, 1]).astype(int), 0, left.shape[0] - 1)
    d = disp[v, u]
    Z = np.where(d > 0.5, f * b / np.maximum(d, 1e-6), np.inf)
    keep = np.where((d > 0.5) & (Z < MAX_DEPTH_M))[0]
    Zk = Z[keep]
    pts = np.c_[(kl[keep, 0] - cx) * Zk / f, (kl[keep, 1] - cy) * Zk / f, Zk]
    return keep, pts


def pnp(K, pts3, des3, kp2, des2):
    """Pose of the second camera from the first's 3-D points: x2 = R x1 + t."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    m = bf.knnMatch(des3, des2, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < RATIO * b.distance]
    if len(good) < 8:
        return None
    obj = np.float32([pts3[g.queryIdx] for g in good])
    img = np.float32([kp2[g.trainIdx] for g in good])
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=2.0,
                                             iterationsCount=200, confidence=0.999)
    if not ok or inl is None:
        return None
    # Refine on the inliers only (Levenberg-Marquardt on the reprojection error).
    ii = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[ii], img[ii], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel(), len(good), len(inl)


def main():
    t0 = time.time()
    meta, P, To = load()
    K = np.array(meta['K'], float); b = meta['baseline_m']
    n = meta['frames']
    orb = cv2.ORB_create(N_FEATURES)
    m = {'frames': n, 'baseline_m': b, 'image': [meta['width'], meta['height']],
         'focal_px': K[0, 0], 'orb_features': N_FEATURES}

    # 1. features, stereo depth, tracking
    feats, est = [], [np.eye(4)]
    stats = {'stereo_points': [], 'matches': [], 'inliers': []}
    prev = None
    last_rel = np.eye(4)
    for k in range(n):
        L = cv2.imread(str(ST / 'left' / f'{k:05d}.png'), cv2.IMREAD_GRAYSCALE)
        R_ = cv2.imread(str(ST / 'right' / f'{k:05d}.png'), cv2.IMREAD_GRAYSCALE)
        kl, dl = features(orb, L)
        idx, p3 = stereo_points(K, b, kl, L, R_)
        feats.append((kl, dl, idx, p3))
        stats['stereo_points'].append(len(idx))
        if prev is not None:
            kl0, dl0, idx0, p30 = prev
            r = pnp(K, p30, dl0[idx0], kl, dl)
            if r is None or r[3] < 20:          # too few inliers: keep the last motion
                T_rel = last_rel
                stats.setdefault('failed', []).append(k)
            else:
                Rm, tm, ng, ni = r
                stats['matches'].append(ng); stats['inliers'].append(ni)
                T_rel = np.eye(4); T_rel[:3, :3] = Rm; T_rel[:3, 3] = tm      # x_k = T_rel x_{k-1}
            last_rel = T_rel
            est.append(est[-1] @ np.linalg.inv(T_rel))                      # world0 <- cam_k
        prev = (kl, dl, idx, p3)
        if k % 200 == 0:
            print(k, len(idx), time.time() - t0)
    est = np.array(est)
    world = np.array([To[0] @ T for T in est])
    vo = np.array([planar(T) for T in world])
    truth = np.array([planar(T) for T in To])
    path = np.r_[0, np.cumsum(np.hypot(*np.diff(truth[:, :2], axis=0).T))]
    err = np.hypot(*(vo[:, :2] - truth[:, :2]).T)
    m['stereo_points_median'] = int(np.median(stats['stereo_points']))
    m['tracking'] = {'frames_kept_last_motion': len(stats.get('failed', [])),
                     'matches_median': int(np.median(stats['matches'])),
                     'inliers_median': int(np.median(stats['inliers'])),
                     'final_error_m': round(float(err[-1]), 2), 'rmse_m': round(float(np.sqrt(np.mean(err ** 2))), 2),
                     'max_error_m': round(float(err.max()), 2),
                     'error_per_100m_end': round(float(err[-1] / path[-1] * 100), 2),
                     'heading_error_end_deg': round(float(np.degrees(ls.l6common.wrap(vo[-1, 2] - truth[-1, 2]))), 2),
                     'height_drift_end_m': round(float(world[-1][2, 3] - To[-1][2, 3]), 2)}
    print('VO', m['tracking'])

    # 2. one camera: the essential matrix between frames 300 and 302
    kA, dA, _, _ = feats[300]; kB, dB, _, _ = feats[302]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [a for a, c in (p for p in bf.knnMatch(dA, dB, k=2) if len(p) == 2) if a.distance < RATIO * c.distance]
    pA = np.float32([kA[g.queryIdx] for g in good]); pB = np.float32([kB[g.trainIdx] for g in good])
    E, inl = cv2.findEssentialMat(pA, pB, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, Rm, tm, _ = cv2.recoverPose(E, pA, pB, K, mask=inl)
    T_true = np.linalg.inv(To[302]) @ To[300]                      # x_302 = T x_300
    t_true = T_true[:3, 3]
    cosang = float(np.dot(tm.ravel(), t_true) / np.linalg.norm(t_true))
    m['mono_essential'] = {'frames': [300, 302], 'matches': len(good), 'inliers': int(inl.sum()),
                           't_length_from_E': round(float(np.linalg.norm(tm)), 3),
                           'true_motion_m': round(float(np.linalg.norm(t_true)), 3),
                           'direction_error_deg': round(float(np.degrees(np.arccos(np.clip(abs(cosang), -1, 1)))), 2)}
    print('mono', m['mono_essential'])

    # 3. keyframes
    keys, last = [0], vo[0]
    for k in range(1, n):
        if np.hypot(*(vo[k, :2] - last[:2])) >= KEY_EVERY_M:
            keys.append(k); last = vo[k]
    keys = np.array(keys)
    m['keyframes'] = {'every_m': KEY_EVERY_M, 'count': int(len(keys))}

    # 4. bag of words
    rng = np.random.default_rng(0)
    sample = np.vstack([feats[k][1][rng.choice(len(feats[k][1]), 200, replace=False)] for k in keys[::3]])
    bits = np.unpackbits(sample, axis=1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, _, centers = cv2.kmeans(bits, VOCAB, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    ctree = cKDTree(centers)
    hist = np.zeros((len(keys), VOCAB))
    for r_, k in enumerate(keys):
        _, w = ctree.query(np.unpackbits(feats[k][1], axis=1).astype(np.float32))
        np.add.at(hist[r_], w, 1)
    idf = np.log(len(keys) / np.maximum((hist > 0).sum(0), 1))
    vec = hist * idf
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    np.save(HERE / 'bow_similarity.npy', vec @ vec.T)
    kpath = np.r_[0, np.cumsum(np.hypot(*np.diff(vo[keys, :2], axis=0).T))]   # the AV's own odometry
    loops, cand, rejected, true_ok, loop_err = [], 0, 0, 0, []
    sims_true, sims_false = [], []
    for i in range(len(keys)):
        older = np.where(kpath[i] - kpath > LOOP_MIN_GAP_M)[0]
        if len(older) == 0:
            continue
        s = vec[older] @ vec[i]
        j = older[int(np.argmax(s))]
        same_place = np.hypot(*(truth[keys[i], :2] - truth[keys[j], :2])) < 5.0
        (sims_true if same_place else sims_false).append(float(s.max()))
        if s.max() < 0.30:                # chosen: similarity gate
            continue
        cand += 1
        kl0, dl0, idx0, p30 = feats[keys[j]]
        kl1, dl1, _, _ = feats[keys[i]]
        r = pnp(K, p30, dl0[idx0], kl1, dl1)
        if r is None or r[3] < LOOP_MIN_INLIERS:
            rejected += 1
            continue
        Rm, tm, ng, ni = r
        Trel = np.eye(4); Trel[:3, :3] = Rm; Trel[:3, 3] = tm       # x_i = Trel x_j
        Twj = To[0] @ est[keys[j]]
        Twi_meas = Twj @ np.linalg.inv(Trel)
        z = ls.between(planar(Twj), planar(Twi_meas))
        # Right when the measured relative pose agrees with the true one.
        zt = ls.between(truth[keys[j]], truth[keys[i]])
        same = bool(np.hypot(*(z[:2] - zt[:2])) < 1.0 and abs(ls.l6common.wrap(z[2] - zt[2])) < np.radians(5))
        loops.append((int(j), int(i), z, same, ni))
        true_ok += int(same)
        loop_err.append(float(np.hypot(*(z[:2] - zt[:2]))))
    m['loops'] = {'vocabulary_words': VOCAB, 'similarity_gate': 0.30, 'candidates': cand, 'accepted': len(loops),
                  'rejected_by_pnp': rejected, 'accepted_right_within_1m_5deg': true_ok,
                  'measured_vs_true_offset_m_sorted_top5': sorted([round(e_, 2) for e_ in loop_err])[-5:],
                  'min_inliers': LOOP_MIN_INLIERS,
                  'best_similarity_median_same_place': round(float(np.median(sims_true)), 3) if sims_true else None,
                  'best_similarity_median_elsewhere': round(float(np.median(sims_false)), 3) if sims_false else None}
    print('loops', m['loops'])

    # 5. pose graph
    nodes = vo[keys].copy()
    Om_odo = np.diag([1 / 0.05 ** 2, 1 / 0.05 ** 2, 1 / np.radians(0.5) ** 2])
    edges = [(i, i + 1, ls.between(nodes[i], nodes[i + 1]), Om_odo) for i in range(len(nodes) - 1)]
    Om_loop = np.diag([1 / 0.1 ** 2, 1 / 0.1 ** 2, 1 / np.radians(1.0) ** 2])
    edges += [(j, i, z, Om_loop) for j, i, z, ok, ni in loops]
    tk = truth[keys]
    eb = np.hypot(*(nodes[:, :2] - tk[:, :2]).T)
    n_odo = len(nodes) - 1
    after, costs, info = ls.optimize(nodes, edges)
    ea = np.hypot(*(after[:, :2] - tk[:, :2]).T)
    after_r, costs_r, info_r = ls.optimize(nodes, edges, iters=30, robust='cauchy', robust_c=3.0, robust_from=n_odo)
    er = np.hypot(*(after_r[:, :2] - tk[:, :2]).T)
    wl = info_r['final_weights'][n_odo:]
    okl = np.array([ok for j, i, z, ok, ni in loops], bool)
    m['pose_graph'] = {'nodes': int(len(nodes)), 'loop_edges': len(loops), 'wrong_loop_edges': int((~okl).sum()),
                       'ape_rmse_before_m': round(float(np.sqrt(np.mean(eb ** 2))), 2),
                       'max_error_before_m': round(float(eb.max()), 2),
                       'plain': {'iterations': len(costs) - 1,
                                 'ape_rmse_after_m': round(float(np.sqrt(np.mean(ea ** 2))), 2),
                                 'max_error_after_m': round(float(ea.max()), 2)},
                       'cauchy': {'c': 3.0, 'iterations': len(costs_r) - 1,
                                  'ape_rmse_after_m': round(float(np.sqrt(np.mean(er ** 2))), 2),
                                  'max_error_after_m': round(float(er.max()), 2),
                                  'weight_median_right_loops': round(float(np.median(wl[okl])), 3),
                                  'weight_max_wrong_loops': round(float(wl[~okl].max()), 4) if (~okl).any() else None}}
    print('graph', m['pose_graph'])
    np.savez(HERE / 'visual_slam_results.npz', truth=truth, vo=vo, keys=keys, before=nodes, after=after, after_robust=after_r,
             loops=np.array([(j, i, int(ok), ni) for j, i, z, ok, ni in loops]).reshape(-1, 4),
             stereo_points=np.array(stats['stereo_points']), path=path)
    m['seconds_total'] = round(time.time() - t0, 1)
    (HERE / 'measured_visual.json').write_text(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
