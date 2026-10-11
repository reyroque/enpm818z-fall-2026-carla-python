"""Shared pieces for the L6 hands-on: the recorded drive, dead reckoning, APE.

Frames: the map frame is local meters, x east, y north (CARLA's y flipped so
the frame is right-handed); the vehicle frame is x forward, y left.
"""
import csv
from pathlib import Path

import numpy as np

DRIVE_COLUMNS = ("t", "x", "y", "theta", "v", "gyro_z", "wheel_v", "gnss_x", "gnss_y")


def wrap(a):
    """Angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def load_drive(path):
    """Read l6_drive.csv into arrays. Empty cells (no GNSS fix) become NaN."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in DRIVE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s): {', '.join(missing)}")
        rows = list(reader)
    return {c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows]) for c in DRIVE_COLUMNS}


def _hold_last_valid(a):
    a = np.asarray(a, float).copy()
    last = 0.0
    for i, v in enumerate(a):
        if np.isnan(v):
            a[i] = last
        else:
            last = v
    return a


def dead_reckon(t, wheel_v, gyro_z, x0, y0, theta0):
    """Integrate wheel speed and yaw rate: drive v*dt along the heading, then turn.

    NaN samples hold the last valid value (a dropped message is not a stop).
    """
    v = _hold_last_valid(wheel_v)
    w = _hold_last_valid(gyro_z)
    n = len(t)
    x, y, th = np.empty(n), np.empty(n), np.empty(n)
    x[0], y[0], th[0] = x0, y0, theta0
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        x[k] = x[k - 1] + v[k - 1] * np.cos(th[k - 1]) * dt
        y[k] = y[k - 1] + v[k - 1] * np.sin(th[k - 1]) * dt
        th[k] = wrap(th[k - 1] + w[k - 1] * dt)
    return x, y, th


def ape(est_xy, true_xy):
    """Absolute pose error (position part): distance from estimate to truth per row."""
    est_xy, true_xy = np.asarray(est_xy, float), np.asarray(true_xy, float)
    err = np.linalg.norm(est_xy - true_xy, axis=1)
    ok = ~np.isnan(err)
    if not ok.any():
        raise ValueError("no valid rows")
    e = err[ok]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mean": float(e.mean()),
            "max": float(e.max()), "errors": err}


def save_scans(path, times, scans):
    """Ragged scans (each an (N_i, 2) array, vehicle frame) as one npz."""
    counts = np.array([len(s) for s in scans], dtype=np.int64)
    points = np.concatenate([np.asarray(s, float).reshape(-1, 2) for s in scans]) if len(scans) else np.zeros((0, 2))
    np.savez_compressed(path, t=np.asarray(times, float), counts=counts, points=points)


def load_scans(path):
    d = np.load(path)
    # Read each array once: an NpzFile does not cache, so d["points"] inside the loop
    # would decompress the whole file again for every scan (42 s instead of under 1 s).
    t, counts, points = d["t"], d["counts"], d["points"]
    offs = np.concatenate([[0], np.cumsum(counts)])
    return t, [points[offs[i]:offs[i + 1]] for i in range(len(counts))]
