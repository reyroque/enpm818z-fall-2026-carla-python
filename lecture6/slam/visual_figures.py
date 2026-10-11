#!/usr/bin/env python3
"""
visual_figures.py: the figures of the reading page "Visual SLAM".

Runs after visual_slam.py. Draws, on stereo/ (the L6 route, stereo camera):

  visual_stereo_pair     one left and right image, with a row line
  visual_orb             ORB features on one left image
  visual_disparity       the dense disparity map of that pair, and depth
  visual_matches         frame k against frame k+1: matches, RANSAC inliers
  visual_trajectory      stereo visual odometry against the truth
  visual_bow_similarity  keyframe-against-keyframe word similarity
  visual_graph           the pose graph: before, plain, robust (Cauchy)

and adds the numbers these figures quote to measured_visual.json.
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import visual_slam as vs

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figs'
DPI = 200
plt.rcParams.update({'font.size': 12, 'figure.facecolor': 'white'})
FRAME = 300


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    m = json.loads((HERE / 'measured_visual.json').read_text())
    meta, P, To = vs.load()
    K = np.array(meta['K']); b = meta['baseline_m']
    L = cv2.imread(str(vs.ST / 'left' / f'{FRAME:05d}.png'), cv2.IMREAD_GRAYSCALE)
    R_ = cv2.imread(str(vs.ST / 'right' / f'{FRAME:05d}.png'), cv2.IMREAD_GRAYSCALE)
    L2 = cv2.imread(str(vs.ST / 'left' / f'{FRAME + 1:05d}.png'), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(vs.N_FEATURES)

    # 1. stereo pair with one row
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.9))
    for ax, im, t in ((axes[0], L, 'left camera'), (axes[1], R_, 'right camera, 0.54 m to the right')):
        ax.imshow(im, cmap='gray'); ax.axhline(300, color='#c62828', lw=1.2); ax.set_title(t); ax.axis('off')
    save(fig, 'visual_stereo_pair')

    # 2. ORB features
    kp, des = orb.detectAndCompute(L, None)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.imshow(L, cmap='gray')
    pts = np.array([k.pt for k in kp]); octv = np.array([k.octave for k in kp])
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=6, c=octv, cmap='plasma', vmin=0, vmax=7)
    ax.set_title(f'{len(kp)} ORB features, colored by pyramid level (0 = full size)')
    ax.axis('off'); fig.colorbar(sc, ax=ax, fraction=0.025, label='pyramid level')
    save(fig, 'visual_orb')
    m['figure_frame'] = {'frame': FRAME, 'orb_features': len(kp),
                         'share_at_full_size': round(float(np.mean(octv == 0)), 3)}

    # 3. disparity and depth
    disp = vs.SGBM.compute(L, R_).astype(np.float32) / 16.0
    valid = disp > 0.5
    depth = np.where(valid, K[0, 0] * b / np.maximum(disp, 1e-6), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.9))
    im0 = axes[0].imshow(np.where(valid, disp, np.nan), cmap='magma'); axes[0].set_title('disparity d (pixels)')
    fig.colorbar(im0, ax=axes[0], fraction=0.025)
    im1 = axes[1].imshow(np.clip(depth, 0, 60), cmap='viridis'); axes[1].set_title('depth Z = f b / d (m), cut at 60 m')
    fig.colorbar(im1, ax=axes[1], fraction=0.025)
    for ax in axes:
        ax.axis('off')
    save(fig, 'visual_disparity')
    m['figure_frame']['disparity_valid_share'] = round(float(valid.mean()), 3)

    # 4. matches frame k -> k+1 with RANSAC inliers
    idx, p3 = vs.stereo_points(K, b, pts, L, R_)
    kp2, des2 = orb.detectAndCompute(L2, None)
    pts2 = np.array([k.pt for k in kp2])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    mm = bf.knnMatch(des[idx], des2, k=2)
    good = [a for a, c in (p for p in mm if len(p) == 2) if a.distance < vs.RATIO * c.distance]
    obj = np.float32([p3[g.queryIdx] for g in good]); img = np.float32([pts2[g.trainIdx] for g in good])
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=2.0, iterationsCount=200, confidence=0.999)
    inl = set(inl.ravel().tolist())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.imshow(L2, cmap='gray')
    for n_, g in enumerate(good):
        a = pts[idx[g.queryIdx]]; c = pts2[g.trainIdx]
        ax.plot([a[0], c[0]], [a[1], c[1]], color='#2e7d32' if n_ in inl else '#c62828', lw=1.0)
    ax.plot([], [], color='#2e7d32', label=f'kept by RANSAC: {len(inl)}')
    ax.plot([], [], color='#c62828', label=f'rejected: {len(good) - len(inl)}')
    ax.legend(loc='lower left', fontsize=11)
    ax.set_title(f'frame {FRAME} to frame {FRAME + 1}: each line joins a feature to its match'); ax.axis('off')
    save(fig, 'visual_matches')
    m['figure_frame'].update({'stereo_points': int(len(idx)), 'temporal_matches': len(good), 'ransac_inliers': len(inl)})

    # 5. trajectory
    r = np.load(HERE / 'visual_slam_results.npz')
    truth, vo = r['truth'], r['vo']
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(truth[:, 0], truth[:, 1], color='#222222', lw=2.2, label='truth')
    ax.plot(vo[:, 0], vo[:, 1], color='#c62828', lw=1.3,
            label=f"stereo visual odometry: {m['tracking']['final_error_m']:.2f} m off at the end")
    ax.set_aspect('equal'); ax.set_xlabel('x east (m)'); ax.set_ylabel('y north (m)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.13), fontsize=11, frameon=False)
    ax.set_title('Stereo visual odometry over two laps (914 m)')
    save(fig, 'visual_trajectory')

    # 6. bag-of-words similarity: recompute on keyframes from the same features
    keys = r['keys']
    sim = np.load(HERE / 'bow_similarity.npy') if (HERE / 'bow_similarity.npy').exists() else None
    if sim is not None:
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        imh = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('keyframe'); ax.set_ylabel('keyframe')
        ax.set_title('word similarity between keyframes')
        fig.colorbar(imh, ax=ax, fraction=0.046)
        save(fig, 'visual_bow_similarity')

    # 7. graph: before, plain, robust
    before, after, robust = r['before'], r['after'], r['after_robust']
    loops = r['loops']
    tk = truth[keys]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True, sharey=True)
    pg = m['pose_graph']
    for ax, Pn, t in ((axes[0], before, f"before: RMSE {pg['ape_rmse_before_m']:.2f} m"),
                      (axes[1], after, f"plain least squares: {pg['plain']['ape_rmse_after_m']:.2f} m"),
                      (axes[2], robust, f"with a robust kernel: {pg['cauchy']['ape_rmse_after_m']:.2f} m")):
        ax.plot(tk[:, 0], tk[:, 1], color='#bbbbbb', lw=4, zorder=0)
        for j, i, okl, ni in loops:
            if not okl:
                ax.plot([Pn[j, 0], Pn[i, 0]], [Pn[j, 1], Pn[i, 1]], color='#ff6f00', lw=2.2, zorder=3)
        ax.plot(Pn[:, 0], Pn[:, 1], '.-', color='#1565c0', ms=2.5, lw=0.8, zorder=2)
        ax.set_title(t); ax.set_aspect('equal'); ax.set_xlabel('x east (m)')
    axes[0].set_ylabel('y north (m)')
    axes[0].plot([], [], color='#bbbbbb', lw=4, label='truth')
    axes[0].plot([], [], '.-', color='#1565c0', label='keyframes')
    axes[0].plot([], [], color='#ff6f00', lw=2.2, label=f"the {pg['wrong_loop_edges']} wrong loop edges")
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -0.12), fontsize=11, frameon=False)
    save(fig, 'visual_graph')
    (HERE / 'measured_visual.json').write_text(json.dumps(m, indent=1))
    print('done', m['figure_frame'])


if __name__ == '__main__':
    main()
