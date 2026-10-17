"""L11: imitation learning on one lap of Town01, in numpy.

An expert keeps the AV in its lane. A clone learns to copy it from a recording
(behavior cloning), then from the states it reaches itself (DAgger). Everything
here runs on a laptop: numpy only, no CARLA, no GPU.

The road (l11_road.npz) is one lap of Town01's outer loop, read from the town's
OpenDRIVE map: the lane centerline, its heading and its curvature, every 0.5 m.

Who knows what:
  * the expert sees the true offset and heading error (a privileged expert, like
    CARLA's autopilot, which reads the simulator's state);
  * the clone sees what a lane detector reports: the same two numbers plus
    detector noise. Both read the road's curvature ahead from the map.

Every value in CHOSEN is chosen for this example; the slides say so.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

CHOSEN = {
    "speed_mps": 8.0,            # constant speed, about 29 km/h
    "wheelbase_m": 2.9,          # a mid-size car
    "dt_s": 0.05,                # 20 steps per second
    "max_steer_deg": 30.0,
    "av_width_m": 2.0,           # so the AV leaves its 4.0 m lane at |offset| > 1.0 m
    "preview_s": 0.3,            # the curvature is read this far ahead
    # the expert: steer = atan(L * curvature ahead) - K_E * offset - K_PSI * heading error
    "expert_k_offset": 0.06,     # rad per m of offset
    "expert_k_heading": 0.8,     # rad per rad of heading error
    # disturbance on the executed steering (gusts, road crown): AR(1), every driver gets it
    "gust_sigma_deg": 0.4,
    "gust_tau_s": 2.0,
    # the lane detector's noise on what the clone sees
    "detector_sigma_offset_m": 0.35,
    "detector_sigma_heading_deg": 0.3,
    # how far off the road an episode may go before it stops
    "stop_offset_m": 1.6,
}


@dataclass
class Road:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    lane_width: float
    length: float

    def kappa(self, s):
        return np.interp(np.mod(s, self.length), self.s, self.curvature)

    def pose(self, s, e):
        """Map-frame position of a point at arc length s and offset e (left positive)."""
        sm = np.mod(s, self.length)
        x = np.interp(sm, self.s, self.x)
        y = np.interp(sm, self.s, self.y)
        h = np.interp(sm, self.s, self.heading)
        return x - e * np.sin(h), y + e * np.cos(h), h


def load_road(path=HERE / "l11_road.npz"):
    d = np.load(path)
    return Road(d["s"], d["x"], d["y"], d["heading"], d["curvature"],
                float(d["lane_width"]), float(d["length"]))


def room(road, c=CHOSEN):
    """How far the AV's center may move from the lane center before a wheel leaves it."""
    return (road.lane_width - c["av_width_m"]) / 2


# ---------------------------------------------------------------- the drivers

def expert_steer(e, psi, kappa_ahead, c=CHOSEN):
    """The expert: steer for the curve ahead, then steer back toward the center."""
    return np.arctan(c["wheelbase_m"] * kappa_ahead) - c["expert_k_offset"] * e - c["expert_k_heading"] * psi


def features(e_seen, psi_seen, kappa_ahead):
    """What the clone's linear policy multiplies by its weights."""
    return np.column_stack([np.atleast_1d(e_seen), np.atleast_1d(psi_seen),
                            np.atleast_1d(kappa_ahead), np.ones(np.size(e_seen))])


FEATURE_NAMES = ("offset seen (m)", "heading error seen (rad)", "curvature ahead (1/m)", "constant")


@dataclass
class LinearPolicy:
    w: np.ndarray                     # 4 weights, see FEATURE_NAMES

    def __call__(self, e_seen, psi_seen, kappa_ahead):
        return features(e_seen, psi_seen, kappa_ahead) @ self.w


def fit(X, y):
    """Behavior cloning for a linear policy: least squares, steering against features."""
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return LinearPolicy(w)


def expert_weights(c=CHOSEN):
    """The expert written as weights on the same features (small-angle: atan(L k) ~ L k)."""
    return np.array([-c["expert_k_offset"], -c["expert_k_heading"], c["wheelbase_m"], 0.0])


# ---------------------------------------------------------------- one drive

@dataclass
class Drive:
    t: np.ndarray
    s: np.ndarray
    e: np.ndarray                     # true offset, left positive (m)
    psi: np.ndarray                   # true heading error (rad)
    e_seen: np.ndarray                # what the detector reports
    psi_seen: np.ndarray
    kappa_ahead: np.ndarray
    steer_cmd: np.ndarray             # what the driver commanded (rad)
    steer_expert: np.ndarray          # what the expert would command in that state (rad)
    left_lane_at_m: float             # distance driven when |e| first passed the room; nan if never
    distance_m: float
    info: dict = field(default_factory=dict)


def drive(road, policy, rng, laps=1.0, c=CHOSEN, start_offset=0.0, beta=0.0):
    """Drive `laps` laps. policy(e_seen, psi_seen, kappa_ahead) -> steering (rad), or the
    string "expert". With beta > 0, each step the expert's command is executed with
    probability beta (DAgger's mixture); the recorded labels are always the expert's.
    The drive stops early if the AV is stop_offset_m off the lane center."""
    v, L, dt = c["speed_mps"], c["wheelbase_m"], c["dt_s"]
    dmax = np.radians(c["max_steer_deg"])
    n = int(round(laps * road.length / v / dt))
    a = np.exp(-dt / c["gust_tau_s"])
    gs = np.radians(c["gust_sigma_deg"])
    so, sh = c["detector_sigma_offset_m"], np.radians(c["detector_sigma_heading_deg"])
    lim = room(road, c)
    s, e, psi, gust = 0.0, start_offset, 0.0, 0.0
    rec = {k: np.full(n, np.nan) for k in ("t", "s", "e", "psi", "es", "ps", "ka", "u", "ue")}
    left_at, travelled, k_end = np.nan, 0.0, n
    for k in range(n):
        ka = float(road.kappa(s + v * c["preview_s"]))
        es = e + so * rng.standard_normal()
        ps = psi + sh * rng.standard_normal()
        ue = float(expert_steer(e, psi, ka, c))
        if isinstance(policy, str) and policy == "expert":
            u = ue
        elif beta > 0 and rng.random() < beta:
            u = ue
        else:
            u = float(np.asarray(policy(es, ps, ka)).ravel()[0])
        for key, val in zip(("t", "s", "e", "psi", "es", "ps", "ka", "u", "ue"),
                            (k * dt, s, e, psi, es, ps, ka, u, ue)):
            rec[key][k] = val
        delta = np.clip(u + gust, -dmax, dmax)
        kap = float(road.kappa(s))
        sdot = v * np.cos(psi) / (1.0 - kap * e)
        s += dt * sdot
        e += dt * v * np.sin(psi)
        psi += dt * (v / L * np.tan(delta) - kap * sdot)
        gust = a * gust + np.sqrt(1 - a * a) * gs * rng.standard_normal()
        travelled += dt * v
        if np.isnan(left_at) and abs(e) > lim:
            left_at = travelled
        if abs(e) > c["stop_offset_m"]:
            k_end = k + 1
            break
    cut = {k: v_[:k_end] for k, v_ in rec.items()}
    return Drive(cut["t"], cut["s"], cut["e"], cut["psi"], cut["es"], cut["ps"], cut["ka"],
                 cut["u"], cut["ue"], left_at, travelled)


def dataset(drives, labels="expert"):
    """Stack drives into (X, y). labels="expert": the expert's command in each state (what
    DAgger asks for); labels="commanded": what was executed (the same for expert drives)."""
    X = np.vstack([features(d.e_seen, d.psi_seen, d.kappa_ahead) for d in drives])
    y = np.concatenate([d.steer_expert if labels == "expert" else d.steer_cmd for d in drives])
    return X, y


def open_loop_error_deg(policy, drives):
    """Mean |clone - expert| steering on recorded states: an open-loop score."""
    X, y = dataset(drives)
    return float(np.degrees(np.mean(np.abs(X @ policy.w - y))))


def completion(road, d):
    """Share of the lap driven before a wheel left the lane (1.0 if it never did)."""
    lap = road.length
    if np.isnan(d.left_lane_at_m):
        return 1.0
    return min(d.left_lane_at_m / lap, 1.0)


def behavior_cloning(road, rng, expert_laps=1):
    """Record the expert for `expert_laps` laps and fit the clone to it."""
    demos = [drive(road, "expert", rng) for _ in range(expert_laps)]
    return fit(*dataset(demos)), demos


def dagger(road, rng, iterations=5, expert_laps=1, beta1=1.0):
    """DAgger (Ross, Gordon and Bagnell, 2011), with beta_1 = 1 and beta_i = 0 after:
    iteration 1 is behavior cloning on the expert's own drive; every later iteration
    drives the newest clone for one lap, asks the expert what it would have done in
    every state the clone reached, adds those pairs to the data, and refits on all of it."""
    policies, drives = [], []
    pol, demos = behavior_cloning(road, rng, expert_laps)
    drives += demos
    policies.append(pol)
    for _ in range(iterations - 1):
        d = drive(road, pol, rng)
        drives.append(d)
        pol = fit(*dataset(drives))
        policies.append(pol)
    return policies, drives


# ---------------------------------------------------------------- the lecture's experiment

TRAIN_SEED = 11                   # the expert's recorded lap and DAgger's drives
EVAL_SEEDS = tuple(range(100, 120))   # 20 test drives for every policy: new gusts, new noise


def evaluate(road, policy, seeds=EVAL_SEEDS, c=CHOSEN):
    """Closed loop: drive one lap per seed. Returns the drives and three scores."""
    ds = [drive(road, policy, np.random.default_rng(s), c=c) for s in seeds]
    finished = [bool(np.isnan(d.left_lane_at_m)) for d in ds]
    return ds, {"finished": int(sum(finished)), "drives": len(ds),
                "completion_mean": float(np.mean([completion(road, d) for d in ds])),
                "left_at_median_m": float(np.nanmedian([d.left_lane_at_m for d in ds]))
                if not all(finished) else float("nan"),
                "mean_abs_offset_m": float(np.mean([np.abs(d.e).mean() for d in ds]))}


def experiment(road, iterations=6, train_seed=TRAIN_SEED, eval_seeds=EVAL_SEEDS, c=CHOSEN):
    """Behavior cloning, then DAgger, on one training seed; every policy tested closed loop
    on the same test drives, and open loop on one held-out expert lap."""
    rng = np.random.default_rng(train_seed)
    policies, drives = dagger(road, rng, iterations=iterations)
    held_out = [drive(road, "expert", np.random.default_rng(10_000 + train_seed), c=c)]
    rows = []
    for i, p in enumerate(policies):
        _, score = evaluate(road, p, eval_seeds, c)
        X, _ = dataset(drives[:i + 1])
        rows.append({"iteration": i + 1, "weights": p.w.tolist(), "samples": int(len(X)),
                     "open_loop_deg": open_loop_error_deg(p, held_out), **score})
    _, expert_score = evaluate(road, "expert", eval_seeds, c)
    return {"policies": policies, "drives": drives, "rows": rows, "expert": expert_score,
            "held_out": held_out}
