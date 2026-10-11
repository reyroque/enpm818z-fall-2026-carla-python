# SLAM on the L6 drive

The code behind the two reading pages
[SLAM with a LiDAR](https://enpm818z-fall-2026-docs.readthedocs.io/en/latest/preread/slam-lidar.html)
and
[Visual SLAM](https://enpm818z-fall-2026-docs.readthedocs.io/en/latest/preread/slam-visual.html).
Run it to get every number and figure on those pages, then change a value
and see what happens.

Needs `numpy`, `scipy`, `matplotlib` and, for the camera part,
`opencv-python`. No ROS.

## LiDAR: no CARLA needed

Uses the L6 recording in the folder above (`../l6_drive.csv`,
`../l6_scans.npz`) and L6's ICP (`../icp.py`).

```bash
python3 lidar_slam.py      # odometry, Scan Context loops, the pose graph
python3 lidar_figures.py   # the figures, in figs/
```

`lidar_slam.py` writes its numbers to `measured_lidar.json`;
`lidar_figures.py` adds two more. It takes about 10 s.

| What | Where in `lidar_slam.py` |
|---|---|
| ICP odometry, scan to scan and scan to submap | `odometry()` |
| Scan Context and its distance | `scan_context()`, `sc_distance()` |
| The SE(2) pose graph, Gauss-Newton | `optimize()` |
| Occupancy grid | `occupancy()` |

Values to try: `SUBMAP_KEYS` (10), `KEY_EVERY_M` (2 m), the Scan Context
distance gate (0.35), `LOOP_ICP_RMSE` (0.25 m), and the odometry and loop
information matrices in `main()`.

## Camera: record once with CARLA, then no CARLA

The L6 recording has no camera, so record the same route with a stereo
camera first. This needs a running CARLA 0.9.16 server and about 900 MB of
disk:

```bash
python3 record_stereo.py --route ../route.json --out stereo --port 2000 --tm-port 8000
```

Use your server's ports (CARLA's defaults are 2000 and 8000). Then:

```bash
python3 visual_slam.py      # ORB, stereo depth, PnP odometry, bag of words, pose graph
python3 visual_figures.py   # the figures, in figs/
```

`visual_slam.py` takes about 3 minutes and writes `measured_visual.json`.
It reuses the pose graph of `lidar_slam.py`.

Values to try: `N_FEATURES` (4000), `MAX_DEPTH_M` (40 m), `VOCAB` (500
words), `LOOP_MIN_INLIERS` (60), and the robust kernel's `robust_c` (3.0):
set it very large and the graph behaves like plain least squares.

Every value marked "chosen" in the code is a choice for this example, not a
recommendation.
