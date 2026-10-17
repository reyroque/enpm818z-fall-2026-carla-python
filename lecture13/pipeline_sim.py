"""A timing model of the course pipeline (GP1 to GP4), with no ROS and no CARLA.

It follows one camera frame through the graph and records when each node handles
it. Only times are modeled: no images, no boxes, no driving.

  sensor_manager  ticks the world every 50 ms (GP1's fixed_delta_seconds); each
                  tick captures a camera frame stamped with the tick time and
                  applies the newest control command that has arrived.
  detector_node   runs when a frame arrives (event-driven). One frame at a time;
                  frames that arrive while it is busy wait in its subscription
                  queue, which keeps the newest `depth` frames (ROS 2 "keep
                  last"). Its time per frame is drawn from detector_times.json,
                  measured on a real GPU or CPU.
  fusion_node     runs when detections arrive (event-driven).
  behavior_node   runs on a timer (GP4: "call update() at the FSM tick rate
                  (e.g. 10 Hz)") and reads the newest fused objects.
  controller_node runs on a timer and reads the newest behavior output; the
                  helper node passes its command on to CARLA.

Every time in ms. Every value in Config that is not measured is chosen for this
example, and says so.

  python3 pipeline_sim.py            # prints a summary of the default runs
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    tick_ms: float = 50.0            # GP1: fixed_delta_seconds = 0.05
    cam_delay_ms: float = 3.0        # chosen: publish and deliver one image
    detector: str = "yolov8s_gpu"    # a run in detector_times.json
    depth: int = 10                  # queue depth 10, as in the course nodes
    fusion_ms: float = 4.0           # chosen
    behavior_period_ms: float = 100.0  # GP4: e.g. 10 Hz
    behavior_ms: float = 1.0         # chosen
    control_period_ms: float = 50.0  # chosen: one command per world tick
    control_ms: float = 1.0          # chosen
    helper_ms: float = 1.0           # chosen: carla_vehicle_control.py passes it on
    behavior_phase_ms: float = 0.0   # when the behavior timer first fires, after tick 0
    control_phase_ms: float = 0.0    # when the controller timer first fires
    duration_s: float = 60.0
    freeze_at_ms: float = float("inf")  # the detector stops answering at this time


def load_times(path=HERE / "detector_times.json"):
    d = json.loads(Path(path).read_text())
    return {k: np.asarray(v, float) for k, v in d["runs_ms"].items()}, d["meta"]


def detector_queue(arrive, service, depth, freeze_at=np.inf):
    """One worker, a keep-last queue of `depth`, oldest message first.

    arrive[k]  when frame k reaches the detector's queue (sorted)
    service[k] how long the detector takes on frame k, if it gets it
    Returns (start, done), NaN for frames dropped from the full queue or never
    finished. A frame is dropped when a newer one arrives while the queue
    already holds `depth` frames: keep last drops the oldest.
    """
    n = len(arrive)
    start = np.full(n, np.nan)
    done = np.full(n, np.nan)
    queue = []                       # frame indices, oldest first
    busy_until = -np.inf
    k = 0
    while k < n or queue:
        # next event: the worker frees up (and takes the oldest), or a frame arrives
        if queue and (k >= n or busy_until <= arrive[k]):
            j = queue.pop(0)
            t0 = max(busy_until, arrive[j])
            if t0 >= freeze_at:      # the node hangs: nothing more comes out
                break
            start[j] = t0
            busy_until = t0 + service[j]
            if busy_until > freeze_at:
                break
            done[j] = busy_until
        else:
            queue.append(k)
            if len(queue) > depth:
                queue.pop(0)         # keep last: the oldest frame is lost
            k += 1
    return start, done


def run(cfg=Config(), times=None, seed=0):
    """Simulate cfg.duration_s of driving. Returns a dict of arrays (ms)."""
    if times is None:
        times, _ = load_times()
    rng = np.random.default_rng(seed)
    n = int(round(cfg.duration_s * 1000 / cfg.tick_ms))
    ticks = np.arange(n) * cfg.tick_ms                    # world ticks = camera stamps
    arrive = ticks + cfg.cam_delay_ms
    service = rng.choice(times[cfg.detector], size=n, replace=True)
    det_start, det_done = detector_queue(arrive, service, cfg.depth, cfg.freeze_at_ms)

    # fusion: event-driven, one at a time, in order
    ok = np.flatnonzero(np.isfinite(det_done))
    fus_done = np.full(n, np.nan)
    free = -np.inf
    for j in ok:
        free = max(free, det_done[j]) + cfg.fusion_ms
        fus_done[j] = free
    ok = ok[np.argsort(fus_done[ok])]
    fus_t, fus_stamp = fus_done[ok], ticks[ok]            # when out, which frame

    # behavior timer: reads the newest fused message that has arrived
    beh_fire = np.arange(cfg.behavior_phase_ms, cfg.duration_s * 1000, cfg.behavior_period_ms)
    i = np.searchsorted(fus_t, beh_fire, side="right") - 1
    beh_stamp = np.where(i >= 0, fus_stamp[np.maximum(i, 0)], np.nan)
    beh_out = beh_fire + cfg.behavior_ms

    # controller timer: reads the newest behavior output, then the helper passes it on
    ctl_fire = np.arange(cfg.control_phase_ms, cfg.duration_s * 1000, cfg.control_period_ms)
    i = np.searchsorted(beh_out, ctl_fire, side="right") - 1
    ctl_stamp = np.where(i >= 0, beh_stamp[np.maximum(i, 0)], np.nan)
    ctl_out = ctl_fire + cfg.control_ms + cfg.helper_ms

    # world tick: applies the newest command that has arrived
    i = np.searchsorted(ctl_out, ticks, side="right") - 1
    applied_stamp = np.where(i >= 0, ctl_stamp[np.maximum(i, 0)], np.nan)

    return {"ticks": ticks, "arrive": arrive, "service": service,
            "det_start": det_start, "det_done": det_done, "fus_done": fus_done,
            "beh_fire": beh_fire, "beh_stamp": beh_stamp,
            "ctl_fire": ctl_fire, "ctl_stamp": ctl_stamp, "applied_stamp": applied_stamp}


def reaction_ms(r):
    """For each frame: the time from its stamp to the first world tick that applies a
    command based on that frame or a newer one. This is how long the AV takes to act
    on something that first shows in that frame. NaN when no such tick comes."""
    ticks, applied = r["ticks"], np.nan_to_num(r["applied_stamp"], nan=-np.inf)
    applied = np.maximum.accumulate(applied)
    idx = np.searchsorted(applied, ticks, side="left")
    out = np.full(len(ticks), np.nan)
    ok = idx < len(ticks)
    out[ok] = ticks[idx[ok]] - ticks[ok]
    return out


def ages_at_behavior(r):
    """Age of the data the behavior node holds each time its timer fires: fire time
    minus the camera stamp of the newest fused message (header.stamp)."""
    return r["beh_fire"] - r["beh_stamp"]


def first_alarm_after(r, timeout_ms, t_from):
    """First behavior check at or after t_from whose data is older than timeout_ms."""
    age = ages_at_behavior(r)
    m = (r["beh_fire"] >= t_from) & (age > timeout_ms)
    k = np.flatnonzero(m)
    return r["beh_fire"][k[0]] if len(k) else np.nan


def stopping_distance_m(v_mps, reaction_s, decel_mps2):
    """Distance from the frame that shows the hazard to standing still:
    driven during the reaction, then braking at a constant rate."""
    return v_mps * reaction_s + v_mps ** 2 / (2 * decel_mps2)


def random_phases(cfg, rng):
    return replace(cfg, behavior_phase_ms=float(rng.uniform(0, cfg.behavior_period_ms)),
                   control_phase_ms=float(rng.uniform(0, cfg.control_period_ms)))


if __name__ == "__main__":
    times, meta = load_times()
    rng = np.random.default_rng(818)
    for det in times:
        for depth in (10, 1):
            lat = []
            for s in range(20):
                cfg = random_phases(Config(detector=det, depth=depth), rng)
                r = run(cfg, times, seed=s)
                lat.append(reaction_ms(r)[200:-40])
            lat = np.concatenate(lat)
            lat = lat[np.isfinite(lat)]
            print(f"{det:14s} depth {depth:2d}: reaction median {np.median(lat):6.0f} ms,"
                  f" max {lat.max():6.0f} ms")
