"""Monte Carlo localization: L3's particle filter, given a map.

The map is a 2-D point set. A scan point that lands near a map point is
likely; the likelihood field (Thrun, Burgard and Fox) turns distance to the
nearest map point into a probability.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from l6common import wrap


class LikelihoodField:
    # sigma_hit is wide on purpose: with a few thousand particles over a whole
    # map, the particle nearest the truth is often a meter or a few degrees
    # off, and a narrow likelihood would score it as badly as a wrong one.
    def __init__(self, map_points, sigma_hit=1.0, z_hit=0.9, z_rand=0.1, max_dist=3.0):
        self.tree = cKDTree(np.asarray(map_points, float))
        self.sigma_hit, self.z_hit, self.z_rand, self.max_dist = sigma_hit, z_hit, z_rand, max_dist

    def log_prob(self, points_world):
        d, _ = self.tree.query(points_world, distance_upper_bound=self.max_dist)
        hit = np.where(np.isinf(d), 0.0, self.z_hit * np.exp(-0.5 * (d / self.sigma_hit) ** 2))
        with np.errstate(divide="ignore"):
            return np.log(hit + self.z_rand / self.max_dist)


def predict(particles, v, w, dt, rng, sigma_v, sigma_w):
    n = len(particles)
    vn = v + rng.normal(0, sigma_v, n)
    wn = w + rng.normal(0, sigma_w, n)
    out = particles.copy()
    out[:, 0] += vn * np.cos(out[:, 2]) * dt
    out[:, 1] += vn * np.sin(out[:, 2]) * dt
    out[:, 2] = wrap(out[:, 2] + wn * dt)
    return out


def weigh(particles, scan_vehicle, field, max_beams=60):
    """Weights from one scan. Only max_beams evenly spaced points are used:
    nearby beams are not independent, and multiplying hundreds of them makes
    the weights collapse onto one particle after a single scan."""
    scan = np.asarray(scan_vehicle, float)
    n = len(particles)
    if len(scan) == 0:
        return np.full(n, 1.0 / n), False
    if len(scan) > max_beams:
        scan = scan[np.linspace(0, len(scan) - 1, max_beams).astype(int)]
    c, s = np.cos(particles[:, 2])[:, None], np.sin(particles[:, 2])[:, None]
    wx = particles[:, 0:1] + c * scan[:, 0] - s * scan[:, 1]          # (n, beams)
    wy = particles[:, 1:2] + s * scan[:, 0] + c * scan[:, 1]
    logp = field.log_prob(np.column_stack([wx.ravel(), wy.ravel()])).reshape(n, -1)
    logw = logp.sum(axis=1)
    if not np.isfinite(logw).any():
        return np.full(n, 1.0 / n), True
    logw = np.where(np.isfinite(logw), logw, -np.inf)
    return np.exp(logw - logsumexp(logw)), False


def systematic_resample(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(weights), positions), n - 1)


def neff(weights):
    return float(1.0 / np.sum(np.asarray(weights) ** 2))


def run_mcl(t, wheel_v, gyro_z, scan_t, scans, field, n, rng, init, bounds=None,
            init_pose=None, radius=20.0, sigma_v=0.3, sigma_w=0.03, max_beams=60):
    """Run MCL over a log. init is "lost" (uniform over bounds, any heading),
    "coarse" (uniform within radius of init_pose's x, y, any heading: a cold
    start with only a rough GNSS fix), or "known" (Gaussian around init_pose).
    Resamples only when the effective particle count falls below n / 2.
    sigma_v (m per square root of s) and sigma_w (rad per square root of s) set
    how fast the particles spread: after T seconds, about sigma_v * sqrt(T)."""
    if init == "lost":
        x0, x1, y0, y1 = bounds
        parts = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(-np.pi, np.pi, n)])
    elif init == "coarse":
        r = radius * np.sqrt(rng.random(n))
        a = rng.uniform(-np.pi, np.pi, n)
        parts = np.column_stack([init_pose[0] + r * np.cos(a), init_pose[1] + r * np.sin(a),
                                 rng.uniform(-np.pi, np.pi, n)])
    else:
        parts = np.asarray(init_pose, float) + rng.normal(0, [1.0, 1.0, 0.05], (n, 3))
    wts = np.full(n, 1.0 / n)
    means, spreads, neffs, losts = [], [], [], []
    k = 0
    for j, ts in enumerate(scan_t):
        while k + 1 < len(t) and t[k + 1] <= ts + 1e-9:
            dt = t[k + 1] - t[k]
            v = wheel_v[k] if np.isfinite(wheel_v[k]) else 0.0
            w = gyro_z[k] if np.isfinite(gyro_z[k]) else 0.0
            parts = predict(parts, v, w, dt, rng, sigma_v / np.sqrt(dt), sigma_w / np.sqrt(dt))
            k += 1
        new, lost = weigh(parts, scans[j], field, max_beams)
        wts = wts * new
        wts = wts / wts.sum() if wts.sum() > 0 and np.isfinite(wts.sum()) else new
        mean_xy = wts @ parts[:, :2]
        mean_th = np.arctan2(wts @ np.sin(parts[:, 2]), wts @ np.cos(parts[:, 2]))
        means.append([*mean_xy, mean_th])
        spreads.append(float(np.sqrt(wts @ np.sum((parts[:, :2] - mean_xy) ** 2, axis=1))))
        neffs.append(neff(wts))
        losts.append(lost)
        if neffs[-1] < n / 2:
            parts = parts[systematic_resample(wts, rng)]
            wts = np.full(n, 1.0 / n)
    return {"mean": np.array(means), "spread": np.array(spreads), "neff": np.array(neffs), "lost": np.array(losts)}
