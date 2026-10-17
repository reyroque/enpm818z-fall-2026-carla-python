"""L13 hands-on: how long does the pipeline take to brake, and what if a node hangs?

Five parts on one timing model of the course pipeline (pipeline_sim.py). No ROS, no
CARLA: only the times at which each node handles a camera frame.

  python3 l13_handson.py --part chain     # one frame, step by step, through every node
  python3 l13_handson.py --part launches  # many launches: how the reaction time spreads
  python3 l13_handson.py --part queue     # a detector slower than the camera, queue depth 10 or 1
  python3 l13_handson.py --part watchdog  # the detector hangs: a timeout notices it
  python3 l13_handson.py --part stop      # reaction time and braking, in meters
Add --snapshot out.png to save a still, --html out.html for a page (needs mpld3).
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

import pipeline_sim as ps

HERE = Path(__file__).resolve().parent
DETECTORS = ["yolov8s_gpu", "rtdetr_l_gpu", "yolov8s_cpu"]
NAMES = {"yolov8s_gpu": "YOLOv8s, GPU", "rtdetr_l_gpu": "RT-DETR-L, GPU", "yolov8s_cpu": "YOLOv8s, CPU"}
COLOR = {"yolov8s_gpu": "#2D6CA2", "rtdetr_l_gpu": "#6A4C93", "yolov8s_cpu": "#A0522D"}
ORANGE, GREY, RED, GREEN = "#B5651D", "#7A828C", "#C0392B", "#1E7A5A"


def one_frame(cfg, times, k=20):
    """The times one frame (number k) reaches each node, from a run with the detector's
    median time. Returns a dict of times relative to the frame's stamp (ms)."""
    med = {cfg.detector: np.array([float(np.median(times[cfg.detector]))])}
    r = ps.run(replace(cfg, duration_s=4.0, depth=10), med, seed=0)
    s = r["ticks"][k]
    fus = r["fus_done"][k]
    b = r["beh_fire"][np.searchsorted(r["beh_fire"], fus)]
    c = r["ctl_fire"][np.searchsorted(r["ctl_fire"], b + cfg.behavior_ms)]
    out = c + cfg.control_ms + cfg.helper_ms
    tick = r["ticks"][np.searchsorted(r["ticks"], out)]
    return {"camera": r["arrive"][k] - s, "detector": r["det_done"][k] - s, "fusion": fus - s,
            "behavior_fires": b - s, "behavior": b + cfg.behavior_ms - s, "controller_fires": c - s,
            "controller": out - s, "tick": tick - s, "median_ms": float(med[cfg.detector][0])}


def part_chain(fig, times, a):
    ax = fig.add_axes([0.14, 0.38, 0.82, 0.55])
    s_b = Slider(fig.add_axes([0.25, 0.20, 0.5, 0.03]), "behavior timer phase (ms)", 0, 99, valinit=30, valstep=1)
    s_c = Slider(fig.add_axes([0.25, 0.14, 0.5, 0.03]), "controller timer phase (ms)", 0, 49, valinit=15, valstep=1)
    radio = RadioButtons(fig.add_axes([0.02, 0.02, 0.2, 0.12]), [NAMES[d] for d in DETECTORS])
    rows = ["camera", "detector", "fusion", "behavior", "controller", "world tick"]

    def draw(_=None):
        det = DETECTORS[[NAMES[d] for d in DETECTORS].index(radio.value_selected)]
        cfg = ps.Config(detector=det, behavior_phase_ms=s_b.val, control_phase_ms=s_c.val)
        f = one_frame(cfg, times)
        ax.clear()
        y = {n: len(rows) - 1 - i for i, n in enumerate(rows)}
        segs = [("camera", 0, f["camera"], GREY, False), ("detector", f["camera"], f["detector"], COLOR[det], False),
                ("fusion", f["detector"], f["fusion"], COLOR[det], False),
                ("behavior", f["fusion"], f["behavior_fires"], ORANGE, True),
                ("behavior", f["behavior_fires"], f["behavior"], ORANGE, False),
                ("controller", f["behavior"], f["controller_fires"], ORANGE, True),
                ("controller", f["controller_fires"], f["controller"], ORANGE, False),
                ("world tick", f["controller"], f["tick"], GREY, True)]
        for n, t0, t1, col, wait in segs:
            if wait:
                ax.barh(y[n], t1 - t0, left=t0, height=0.42, color="white", edgecolor=col, hatch="///")
            else:
                ax.barh(y[n], t1 - t0, left=t0, height=0.42, color=col)
        compute = f["tick"] - sum(t1 - t0 for _, t0, t1, _, w in segs if w)
        ax.set_yticks([y[n] for n in rows]); ax.set_yticklabels(rows)
        ax.set_xlim(0, 300); ax.set_xlabel("time since the camera frame (ms)")
        ax.set_title(f"{NAMES[det]} at its median {f['median_ms']:.1f} ms: the brake acts after "
                     f"{f['tick']:.0f} ms ({compute:.0f} ms computing, solid; "
                     f"{f['tick'] - compute:.0f} ms waiting, hatched)", fontsize=10)
        fig.canvas.draw_idle()

    for w in (s_b, s_c):
        w.on_changed(draw)
    radio.on_clicked(draw)
    draw()
    return [s_b, s_c, radio]


def launch_runs(det, depth, n, seed, times, duration_s=30.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cfg = ps.random_phases(ps.Config(detector=det, depth=depth, duration_s=duration_s), rng)
        out.append(ps.run(cfg, times, seed=seed + i))
    return out


def part_launches(fig, times, a):
    ax = fig.add_axes([0.1, 0.32, 0.85, 0.6])
    s_n = Slider(fig.add_axes([0.3, 0.16, 0.45, 0.03]), "launches", 5, 100, valinit=40, valstep=5)
    radio = RadioButtons(fig.add_axes([0.02, 0.02, 0.2, 0.12]), [NAMES[d] for d in DETECTORS])

    def draw(_=None):
        det = DETECTORS[[NAMES[d] for d in DETECTORS].index(radio.value_selected)]
        lat = np.concatenate([ps.reaction_ms(r)[40:-40] for r in
                              launch_runs(det, 10, int(s_n.val), a.seed, times)])
        lat = lat[np.isfinite(lat)]
        ax.clear()
        v, c = np.unique(lat, return_counts=True)
        ax.bar(v, 100 * c / len(lat), width=30, color=COLOR[det])
        ax.set_xlabel("time from camera frame to the brake acting (ms)"); ax.set_ylabel("share of frames (%)")
        ax.set_title(f"{NAMES[det]}, {int(s_n.val)} launches with random timer phases: "
                     f"median {np.median(lat):.0f} ms, longest {lat.max():.0f} ms", fontsize=10)
        fig.canvas.draw_idle()

    s_n.on_changed(draw)
    radio.on_clicked(draw)
    draw()
    return [s_n, radio]


def part_queue(fig, times, a):
    ax = fig.add_axes([0.12, 0.3, 0.83, 0.6])
    s_d = Slider(fig.add_axes([0.3, 0.14, 0.45, 0.03]), "queue depth", 1, 10, valinit=10, valstep=1)

    def draw(_=None):
        depth = int(s_d.val)
        cfg = ps.Config(detector="yolov8s_cpu", depth=depth, duration_s=8.0,
                        behavior_phase_ms=30, control_phase_ms=15)
        r = ps.run(cfg, times, seed=a.seed)
        # frames finished after the last tick never meet a newer frame: leave the end out
        ok = np.isfinite(r["det_done"]) & (r["det_done"] < r["ticks"][-1] - 500)
        age = r["det_done"][ok] - r["ticks"][ok]
        lat = ps.reaction_ms(r)
        ax.clear()
        ax.plot(r["det_done"][ok] / 1000, age, ".-", color=COLOR["yolov8s_cpu"])
        ax.set_xlabel("time since the pipeline started (s)")
        ax.set_ylabel("age of a detection when it is ready (ms)")
        ax.set_ylim(0, 700)
        ax.set_title(f"YOLOv8s on the CPU, queue depth {depth}: {100 * np.isfinite(r["det_done"]).mean():.0f}% of frames detected; "
                     f"reaction after 3 s: median {np.nanmedian(lat[60:-20]):.0f} ms", fontsize=10)
        fig.canvas.draw_idle()

    s_d.on_changed(draw)
    draw()
    return [s_d]


def part_watchdog(fig, times, a):
    ax = fig.add_axes([0.12, 0.32, 0.83, 0.58])
    s_t = Slider(fig.add_axes([0.3, 0.16, 0.45, 0.03]), "timeout (ms)", 40, 300, valinit=70, valstep=5)
    radio = RadioButtons(fig.add_axes([0.02, 0.02, 0.2, 0.12]), [NAMES[d] for d in DETECTORS[:2]])
    cache = {}

    def runs(det):
        if det not in cache:
            normal = launch_runs(det, 10, 40, a.seed, times)
            cfg = ps.Config(detector=det, duration_s=30.0, freeze_at_ms=20000.0,
                            behavior_phase_ms=37.0, control_phase_ms=15.0)
            cache[det] = (normal, ps.run(cfg, times, seed=a.seed))
        return cache[det]

    def draw(_=None):
        det = DETECTORS[[NAMES[d] for d in DETECTORS].index(radio.value_selected)]
        normal, frz = runs(det)
        T = s_t.val
        ages = np.concatenate([ps.ages_at_behavior(r)[20:] for r in normal])
        false_alarms = int((ages > T).sum())
        hours = len(normal) * 30.0 / 3600
        alarm = ps.first_alarm_after(frz, T, 20000.0)
        ax.clear()
        t = frz["beh_fire"] / 1000
        m = (t > 19) & (t < 21)
        ax.plot(t[m], ps.ages_at_behavior(frz)[m], "o-", color=ORANGE, ms=4)
        ax.axhline(T, color=RED, ls="--")
        ax.axvline(20.0, color=GREY, ls=":")
        if np.isfinite(alarm):
            ax.plot([alarm / 1000], [T], "s", color=GREEN, ms=10)
        ax.set_ylim(0, 400); ax.set_xlabel("time (s); the detector hangs at 20 s (dotted)")
        ax.set_ylabel("age of the newest fused objects (ms)")
        ax.set_title(f"{NAMES[det]}, timeout {T:.0f} ms (dashed): {false_alarms} false alarms in "
                     f"{hours:.2f} h of normal driving; the hang is noticed after "
                     f"{alarm - 20000:.0f} ms (square)", fontsize=10)
        fig.canvas.draw_idle()

    s_t.on_changed(draw)
    radio.on_clicked(draw)
    draw()
    return [s_t, radio]


def part_stop(fig, times, a):
    ax = fig.add_axes([0.1, 0.45, 0.85, 0.45])
    s_v = Slider(fig.add_axes([0.3, 0.30, 0.45, 0.03]), "speed (m/s)", 2, 15, valinit=8, valstep=0.5)
    s_r = Slider(fig.add_axes([0.3, 0.23, 0.45, 0.03]), "reaction time (ms)", 50, 800, valinit=200, valstep=50)
    s_a = Slider(fig.add_axes([0.3, 0.16, 0.45, 0.03]), "braking (m/s^2)", 2, 9, valinit=6, valstep=0.5)
    s_g = Slider(fig.add_axes([0.3, 0.09, 0.45, 0.03]), "pedestrian ahead (m)", 3, 30, valinit=15, valstep=0.5)

    def draw(_=None):
        v, r, dec, gap = s_v.val, s_r.val / 1000, s_a.val, s_g.val
        d1, d2 = v * r, v * v / (2 * dec)
        ax.clear()
        ax.barh(0, d1, color="#2D6CA2", height=0.4)
        ax.barh(0, d2, left=d1, color="white", edgecolor="#2D6CA2", hatch="///", height=0.4)
        ax.axvline(gap, color=RED, ls="--")
        ax.set_xlim(0, 35); ax.set_yticks([])
        ax.set_xlabel("meters from where the AV is when the frame is taken")
        verdict = "stops in time" if d1 + d2 < gap else "does NOT stop in time"
        ax.set_title(f"{v * r:.2f} m during the reaction (solid) + {d2:.2f} m braking (hatched) = "
                     f"{d1 + d2:.2f} m; pedestrian at {gap:.1f} m (dashed): {verdict}", fontsize=10)
        fig.canvas.draw_idle()

    for w in (s_v, s_r, s_a, s_g):
        w.on_changed(draw)
    draw()
    return [s_v, s_r, s_a, s_g]


PARTS = {"chain": part_chain, "launches": part_launches, "queue": part_queue,
         "watchdog": part_watchdog, "stop": part_stop}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=list(PARTS), required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--html")
    ap.add_argument("--seed", type=int, default=818)
    ap.add_argument("--times", default=str(HERE / "detector_times.json"),
                    help="detector times (measure_detector.py writes your own)")
    a = ap.parse_args(argv)
    times, _ = ps.load_times(a.times)
    fig = plt.figure(figsize=(11, 6.5))
    keep = PARTS[a.part](fig, times, a)          # noqa: F841  (widgets must stay alive)
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=120, facecolor="white")
    elif a.html:
        try:
            import mpld3
        except ImportError:
            sys.exit("--html needs mpld3 (pip install mpld3); or use --snapshot")
        Path(a.html).write_text(mpld3.fig_to_html(fig))
    else:
        plt.show()


if __name__ == "__main__":
    main()
