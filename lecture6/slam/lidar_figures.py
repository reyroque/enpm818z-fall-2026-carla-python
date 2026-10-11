#!/usr/bin/env python3
"""
lidar_figures.py: the figures of the reading page "SLAM with a LiDAR".

Runs after lidar_slam.py. Adds one measurement (the same pose graph, built
on the poorer scan-to-scan odometry) to measured_lidar.json, then draws:

  lidar_odometry_compare   truth, scan-to-scan and scan-to-submap odometry
  lidar_graph_before_after the pose graph's nodes and loop edges, before and after
  lidar_maps               occupancy grids from drifting and optimized poses
  lidar_scan_context       one place on lap 1 and lap 2, as Scan Context grids
  lidar_sparsity           the nonzero pattern of the normal-equation matrix H
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import lidar_slam as ls

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figs'
OUT.mkdir(exist_ok=True)
DPI = 200
TRUTH, ODO, OPT, LOOP = '#222222', '#c62828', '#1565c0', '#2e7d32'
plt.rcParams.update({'font.size': 12, 'figure.facecolor': 'white'})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    m = json.loads((HERE / 'measured_lidar.json').read_text())
    r = np.load(HERE / 'lidar_slam_results.npz')
    truth, keys, before, after = r['truth'], r['keys'], r['before'], r['after']
    scan_odo, sub_odo, path, loops = r['scan_odo'], r['submap_odo'], r['path'], r['loops']
    st, scans = ls.l6common.load_scans(ls.L6 / 'l6_scans.npz')

    # The same graph on the scan-to-scan odometry: rebuild the loop edges by ICP
    nodes_s = scan_odo[keys].copy()
    Om_odo = np.diag([1 / 0.05 ** 2, 1 / 0.05 ** 2, 1 / np.radians(0.5) ** 2])
    edges = [(i, i + 1, ls.between(nodes_s[i], nodes_s[i + 1]), Om_odo) for i in range(len(nodes_s) - 1)]
    Om_loop = np.diag([1 / 0.1 ** 2, 1 / 0.1 ** 2, 1 / np.radians(1.0) ** 2])
    for j, i, _ in loops:
        R, t, h = ls.icp_mod.icp(scans[keys[i]], scans[keys[j]], R0=ls.rot(ls.l6common.wrap(after[i, 2] - after[j, 2])),
                                 t0=ls.between(after[j], after[i])[:2], max_dist=2.0)
        edges.append((j, i, np.array([t[0], t[1], ls.ang(R)]), Om_loop))
    opt_s, costs_s, _ = ls.optimize(nodes_s, edges)
    tk = truth[keys]
    eb = np.hypot(*(nodes_s[:, :2] - tk[:, :2]).T)
    ea = np.hypot(*(opt_s[:, :2] - tk[:, :2]).T)
    m['pose_graph_on_scan_to_scan'] = {'ape_rmse_before_m': round(float(np.sqrt(np.mean(eb ** 2))), 2),
                                      'ape_rmse_after_m': round(float(np.sqrt(np.mean(ea ** 2))), 2),
                                      'max_error_before_m': round(float(eb.max()), 2),
                                      'max_error_after_m': round(float(ea.max()), 2),
                                      'iterations': len(costs_s) - 1}
    print('graph on scan-to-scan', m['pose_graph_on_scan_to_scan'])
    (HERE / 'measured_lidar.json').write_text(json.dumps(m, indent=1))

    # 1. odometry comparison
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(truth[:, 0], truth[:, 1], color=TRUTH, lw=2.2, label='truth')
    ax.plot(scan_odo[:, 0], scan_odo[:, 1], color=ODO, lw=1.4,
            label=f"scan to scan: {m['odometry_scan']['final_error_m']:.2f} m off at the end")
    ax.plot(sub_odo[:, 0], sub_odo[:, 1], color=OPT, lw=1.4,
            label=f"scan to submap: {m['odometry_submap']['final_error_m']:.2f} m off at the end")
    ax.plot(*truth[0, :2], 'o', color=TRUTH, ms=8)
    ax.set_aspect('equal'); ax.set_xlabel('x east (m)'); ax.set_ylabel('y north (m)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.13), fontsize=11, frameon=False)
    ax.set_title('LiDAR odometry over two laps (916 m)')
    save(fig, 'lidar_odometry_compare')

    # 2. graph before / after, on the scan-to-scan odometry (the clearer case)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    for ax, P, title in ((axes[0], nodes_s, f"before: RMSE {m['pose_graph_on_scan_to_scan']['ape_rmse_before_m']:.2f} m"),
                         (axes[1], opt_s, f"after: RMSE {m['pose_graph_on_scan_to_scan']['ape_rmse_after_m']:.2f} m")):
        ax.plot(truth[:, 0], truth[:, 1], color='#bbbbbb', lw=4, label='truth', zorder=0)
        for j, i, _ in loops[::4]:
            ax.plot([P[j, 0], P[i, 0]], [P[j, 1], P[i, 1]], color=LOOP, lw=0.8, zorder=1)
        ax.plot(P[:, 0], P[:, 1], '.-', color=OPT if P is opt_s else ODO, ms=3, lw=0.8, zorder=2)
        ax.set_title(title); ax.set_aspect('equal'); ax.set_xlabel('x east (m)')
    axes[0].set_ylabel('y north (m)')
    axes[0].plot([], [], color=LOOP, lw=1, label='loop edge (every 4th drawn)')
    axes[0].plot([], [], '.-', color=ODO, label='keyframes, scan-to-scan odometry')
    axes[1].plot([], [], '.-', color=OPT, label='keyframes, optimized')
    for ax in axes:
        ax.legend(loc='upper left', bbox_to_anchor=(0, -0.12), fontsize=10, frameon=False)
    save(fig, 'lidar_graph_before_after')

    # 3. maps: odometry poses against optimized poses (every 5th scan)
    sel = keys
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, P, title in ((axes[0], nodes_s, 'map from scan-to-scan odometry'),
                         (axes[1], opt_s, 'map from the optimized poses')):
        L, lo, cell = ls.occupancy(P, [scans[k] for k in sel])
        img = 1 - 1 / (1 + np.exp(-np.clip(L, -10, 10)))
        ax.imshow(img.T, origin='lower', cmap='gray', vmin=0, vmax=1,
                  extent=[lo[0], lo[0] + L.shape[0] * cell, lo[1], lo[1] + L.shape[1] * cell])
        ax.set_title(title); ax.set_xlabel('x east (m)')
    axes[0].set_ylabel('y north (m)')
    save(fig, 'lidar_maps')

    # 4. scan context: the first keyframe on lap 2 that closed a loop, and its lap-1 match
    j, i, _ = loops[0]
    sc_i, sc_j = ls.scan_context(scans[keys[i]]), ls.scan_context(scans[keys[j]])
    dist, shift = ls.sc_distance(sc_i, sc_j)
    m['scan_context_example'] = {'lap1_key': int(j), 'lap2_key': int(i), 'distance': round(float(dist), 3),
                                 'shift_sectors': int(shift), 'shift_deg': round(shift * 360 / ls.SC_SECTORS, 1),
                                 'true_separation_m': round(float(np.hypot(*(truth[keys[i], :2] - truth[keys[j], :2]))), 2)}
    (HERE / 'measured_lidar.json').write_text(json.dumps(m, indent=1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for ax, s, t in ((axes[0], sc_j, f'lap 1, keyframe {j}'), (axes[1], sc_i, f'lap 2, keyframe {i}')):
        ax.imshow(np.log1p(s), aspect='auto', cmap='magma', origin='lower',
                  extent=[0, 360, 0, ls.SC_MAX_R])
        ax.set_title(t); ax.set_ylabel('range (m)')
    axes[1].set_xlabel('bearing around the car (deg), 6 deg per sector')
    save(fig, 'lidar_scan_context')

    # 5. sparsity of H, on a 60-node slice so the blocks show
    n = len(after)
    rows, cols = [], []
    for i_, j_ in [(e[0], e[1]) for e in edges]:
        for p in (i_, j_):
            for q in (i_, j_):
                rows.append(p); cols.append(q)
    S = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.spy(S, markersize=1.2, color='#222222')
    ax.set_title(f'{n} nodes: nonzero blocks of H')
    ax.set_xlabel('node'); ax.set_ylabel('node')
    save(fig, 'lidar_sparsity')
    print('figures in', OUT)


if __name__ == '__main__':
    main()
