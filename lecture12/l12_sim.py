"""L12: a pedestrian crosses mid-block in front of the AV. A small 2-D simulation.

The scene is a straight two-lane street of CARLA's Town01 (lanes 4.0 m wide,
shoulders 0.3 m, speed limit 25 mph: all read from Town01.xodr, see town01.json).
The AV drives along +x in the right-hand lane. A pedestrian steps off the far
(left) curb and walks straight across, first through the oncoming lane, then
through the AV's lane.

Frames: x along the road, in the AV's direction of travel; y to the AV's left.
The AV's lane center is y = 0. The AV's position is its base link, the middle of
the rear axle, 0.9 m in from the rear bumper of the 4.5 m car.

The logical scenario has three parameters (the ranges are in LOGICAL):
  v0    the AV's speed when the pedestrian steps off the curb (m/s)
  d0    the gap from the AV's front bumper to the pedestrian's crossing line, then (m)
  vped  the pedestrian's walking speed (m/s)
One concrete scenario is one value of each. simulate() runs many at once.

The system under test is a TOY stand-in for a driving stack, nothing more:
  * it sees the pedestrian when the crossing line is between 0 and `sense_range`
    meters ahead of its front bumper;
  * it treats the pedestrian as a hazard when the pedestrian is inside the AV's
    lane, widened by `margin` on each side;
  * `latency` seconds after the first hazard it brakes at `decel` until it stops,
    and stays stopped. Otherwise it holds its speed.
Every value of the toy stack is chosen for the lecture (STACKS below).
A run fails if the pedestrian (a circle) touches the AV (a rectangle).
"""
from dataclasses import dataclass, replace

import numpy as np

# --- the street, from Town01.xodr (town01.json holds the measured values) -------
LANE_W = 4.0            # m, every driving lane in Town01
SHOULDER = 0.3          # m, between the driving lane and the sidewalk
SPEED_LIMIT = 25 * 0.44704   # 25 mph in m/s = 11.176
Y_FAR_CURB = 1.5 * LANE_W + SHOULDER     # 6.3 m: where the pedestrian starts
Y_NEAR_CURB = -0.5 * LANE_W - SHOULDER   # -2.3 m: the AV's side of the street

# --- the AV's body (the course's 4.5 m car, base link 0.9 m from the rear) -----
AV_LENGTH, AV_WIDTH, REAR_TO_BASE = 4.5, 1.9, 0.9
BASE_TO_FRONT = AV_LENGTH - REAR_TO_BASE           # 3.6 m
PED_RADIUS = 0.25                                  # m, chosen

# --- the logical scenario: parameter ranges (chosen; v0 tops out at the limit) -
LOGICAL = {"v0": (2.0, SPEED_LIMIT), "d0": (5.0, 60.0), "vped": (0.8, 2.0)}
DEFAULT_VPED = 1.4       # m/s, chosen: a typical walk, used when vped is held fixed


@dataclass(frozen=True)
class Stack:
    """The toy system under test. Every number is chosen for the lecture."""
    sense_range: float = 30.0   # m ahead of the front bumper
    margin: float = 0.5         # m added to each side of the AV's lane
    latency: float = 0.5        # s from the first hazard to the brakes
    decel: float = 6.0          # m/s^2 while braking


# The ODD cells of the lecture: a wet road brakes softer, night shortens the range.
STACKS = {
    "dry_day": Stack(),
    "wet_day": Stack(decel=4.0),
    "dry_night": Stack(sense_range=12.0),
    "wet_night": Stack(sense_range=12.0, decel=4.0),
}


def simulate(v0, d0, vped, stack=Stack(), dt=0.01, record=False):
    """Run every concrete scenario given by the broadcast arrays v0, d0, vped.

    Returns a dict of arrays shaped like the broadcast inputs:
      collided   True if the pedestrian touched the AV
      impact_v   the AV's speed at first contact (m/s), 0 where none
      clearance  the closest the pedestrian came to the AV's body (m), 0 if touched
      braked     True if the toy stack braked at all
      t_brake    when the brakes came on (s), nan if never
      stop_gap   for runs that stopped: gap from the front bumper to the crossing
                 line when stopped (m); nan otherwise
    With record=True (one scenario only) it also returns the time series.
    """
    v0, d0, vped = np.broadcast_arrays(*(np.asarray(a, float) for a in (v0, d0, vped)))
    shape = v0.shape
    v0, d0, vped = v0.ravel(), d0.ravel(), vped.ravel()
    n = v0.size
    xc = d0 + BASE_TO_FRONT          # crossing line, with the AV's base link at x = 0
    xb = np.zeros(n)                 # base link
    v = v0.copy()
    t_end = (Y_FAR_CURB - Y_NEAR_CURB) / vped + 1.0   # the pedestrian is across, plus 1 s
    hazard_t = np.full(n, np.nan)
    collided = np.zeros(n, bool)
    impact_v = np.zeros(n)
    clearance = np.full(n, np.inf)
    t_brake = np.full(n, np.nan)
    t_col = np.full(n, np.inf)          # when the run collided: everything stops there
    half_w = LANE_W / 2 + stack.margin
    steps = int(np.ceil(t_end.max() / dt)) + 1
    rec = {k: [] for k in ("t", "xb", "v", "yp", "hazard", "braking")} if record else None
    for k in range(steps):
        t = k * dt
        live = t <= t_end
        yp = np.maximum(Y_FAR_CURB - vped * np.minimum(t, t_col), Y_NEAR_CURB)
        gap = xc - (xb + BASE_TO_FRONT)                       # bumper to crossing line
        seen = (gap >= -AV_LENGTH) & (gap <= stack.sense_range)
        hazard = seen & (np.abs(yp) < half_w) & live
        new = hazard & np.isnan(hazard_t)
        hazard_t[new] = t
        braking = np.isfinite(hazard_t) & (t >= hazard_t + stack.latency)
        t_brake[braking & np.isnan(t_brake)] = t
        # distance from the pedestrian's center to the AV's rectangle
        dx = np.maximum(np.maximum((xb - REAR_TO_BASE) - xc, xc - (xb + BASE_TO_FRONT)), 0.0)
        dy = np.maximum(np.abs(yp) - AV_WIDTH / 2, 0.0)
        dist = np.hypot(dx, dy) - PED_RADIUS
        touch = (dist <= 0) & live & ~collided
        impact_v[touch] = v[touch]
        t_col[touch] = t
        collided |= touch
        clearance = np.where(live & ~collided, np.minimum(clearance, dist), clearance)
        if record:
            for key, val in (("t", t), ("xb", xb[0]), ("v", v[0]), ("yp", yp[0]),
                             ("hazard", hazard[0]), ("braking", braking[0])):
                rec[key].append(float(val))
        # move: brake or hold speed; a collided run stops where it touched
        v = np.where(braking, np.maximum(v - stack.decel * dt, 0.0), v)
        v = np.where(collided, 0.0, v)
        xb = xb + v * dt
    clearance[collided] = 0.0
    stopped = (v == 0) & ~collided
    stop_gap = np.where(stopped, xc - (xb + BASE_TO_FRONT), np.nan)
    out = {"collided": collided, "impact_v": impact_v, "clearance": clearance,
           "braked": np.isfinite(t_brake), "t_brake": t_brake, "stop_gap": stop_gap}
    out = {k: a.reshape(shape) for k, a in out.items()}
    if record:
        out["series"] = {k: np.array(a) for k, a in rec.items()}
    return out


def stopping_distance(v0, stack=Stack()):
    """Distance covered from the first hazard to standstill: latency, then braking."""
    v0 = np.asarray(v0, float)
    return v0 * stack.latency + v0 ** 2 / (2 * stack.decel)


def grid(n_v, n_d, ranges=LOGICAL):
    """An n_v by n_d grid over v0 and d0, cell centers, as two 2-D arrays."""
    (v_lo, v_hi), (d_lo, d_hi) = ranges["v0"], ranges["d0"]
    vs = v_lo + (np.arange(n_v) + 0.5) * (v_hi - v_lo) / n_v
    ds = d_lo + (np.arange(n_d) + 0.5) * (d_hi - d_lo) / n_d
    return np.meshgrid(vs, ds, indexing="ij")


def grid_edges(n_v, n_d, ranges=LOGICAL):
    """A grid that includes the range ends (what an engineer usually types)."""
    (v_lo, v_hi), (d_lo, d_hi) = ranges["v0"], ranges["d0"]
    return np.meshgrid(np.linspace(v_lo, v_hi, n_v), np.linspace(d_lo, d_hi, n_d), indexing="ij")


def random_runs(n, seed, ranges=LOGICAL, vped=None):
    """n concrete scenarios drawn uniformly from the logical scenario's ranges."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(*ranges["v0"], n)
    d = rng.uniform(*ranges["d0"], n)
    p = np.full(n, DEFAULT_VPED) if vped is None else rng.uniform(*ranges["vped"], n)
    return v, d, p


def zero_failure_bound(n, confidence=0.95):
    """Largest failure probability per run still consistent, at this confidence,
    with n runs and no failure: the p with (1 - p)^n = 1 - confidence."""
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def miles_needed(rate_per_mile, confidence=0.95):
    """Kalra and Paddock's zero-failure miles: (1 - r)^n = 1 - C, solved for n."""
    return np.log(1.0 - confidence) / np.log(1.0 - rate_per_mile)


def boundary_search(stack=Stack(), vped=DEFAULT_VPED, n0=5, levels=3, ranges=LOGICAL):
    """Criticality-guided search, the simple kind: run a coarse n0 x n0 grid that
    includes the range ends, then split every cell whose four corners disagree
    (some pass, some fail) into four, run the new corners, and repeat `levels` times.
    Returns the runs (v0, d0, failed) in the order they were run."""
    (v_lo, v_hi), (d_lo, d_hi) = ranges["v0"], ranges["d0"]
    done = {}

    def run(points):
        new = [p for p in points if p not in done]
        if new:
            vv = np.array([p[0] for p in new]); dd = np.array([p[1] for p in new])
            res = simulate(vv, dd, vped, stack)["collided"]
            for p, f in zip(new, res):
                done[p] = bool(f)

    vs = np.linspace(v_lo, v_hi, n0); ds = np.linspace(d_lo, d_hi, n0)
    key = lambda v, d: (round(float(v), 6), round(float(d), 6))
    run([key(v, d) for v in vs for d in ds])
    cells = [(vs[i], vs[i + 1], ds[j], ds[j + 1]) for i in range(n0 - 1) for j in range(n0 - 1)]
    for _ in range(levels):
        nxt = []
        for (a, b, c, e) in cells:
            corners = [done[key(a, c)], done[key(a, e)], done[key(b, c)], done[key(b, e)]]
            if any(corners) and not all(corners):
                m, h = (a + b) / 2, (c + e) / 2
                run([key(x, y) for x in (a, m, b) for y in (c, h, e)])
                nxt += [(a, m, c, h), (a, m, h, e), (m, b, c, h), (m, b, h, e)]
        cells = nxt
    pts = np.array([(v, d) for (v, d) in done])
    return pts[:, 0], pts[:, 1], np.array(list(done.values()))


# --- scoring a run: the CARLA Leaderboard's published formulas ------------------
# leaderboard.carla.org/evaluation_v2_1/ (read 2026-10-14): driving score = route
# completion R (percent) times the infraction penalty P = 1 / (1 + sum c_j n_j).
LB21_COEF = {"pedestrian": 1.0, "vehicle": 0.70, "static": 0.60, "red_light": 0.40,
             "emergency_yield": 0.40, "timeout": 0.40, "stop_sign": 0.25}
# leaderboard.carla.org/evaluation_v2_0/: P = product of p_j to the power n_j.
LB20_FACTOR = {"pedestrian": 0.50, "vehicle": 0.60, "static": 0.65, "red_light": 0.70,
               "emergency_yield": 0.70, "timeout": 0.70, "stop_sign": 0.80}


def driving_score(route_completion, infractions, version="2.1"):
    """Driving score of one route. route_completion in percent (0 to 100);
    infractions: {kind: count}. Returns (score, penalty)."""
    if version == "2.1":
        p = 1.0 / (1.0 + sum(LB21_COEF[k] * n for k, n in infractions.items()))
    elif version == "2.0":
        p = float(np.prod([LB20_FACTOR[k] ** n for k, n in infractions.items()]))
    else:
        raise ValueError(f"unknown version {version}")
    return route_completion * p, p
