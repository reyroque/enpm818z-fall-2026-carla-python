"""L11 hands-on: an expert drives one lap of Town01; a clone learns to copy it.

  python3 l11_handson.py --part road     # the lap, and the expert driving it
  python3 l11_handson.py --part bc       # behavior cloning: fit, weights, the clone drives
  python3 l11_handson.py --part shift    # why the clone drifts: its error grows off center
  python3 l11_handson.py --part dagger   # DAgger, iteration by iteration (slider)
  python3 l11_handson.py --part eval     # open loop against closed loop, every policy
Add --snapshot out.png to save a still, --html out.html for a page (needs mpld3).
Change the example with --detector-noise, --gust, --seed and --iterations.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import l11common as c

HERE = Path(__file__).resolve().parent
EXPERT, CLONE, DAGGER, LANE = "#1E2939", "#D9694A", "#1E9E74", "#7A828C"


def lane_edges(ax, road, x):
    r = c.room(road)
    for y in (-r, r):
        ax.axhline(y, color=LANE, lw=1, ls="--")
    ax.text(x, r, " a wheel leaves the lane", va="bottom", fontsize=8, color=LANE)


def part_road(fig, road, a):
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.4]})
    ax1.plot(road.x, road.y, color=LANE)
    ax1.plot(road.x[0], road.y[0], "o", color=EXPERT)
    ax1.annotate("start", (road.x[0], road.y[0]), textcoords="offset points", xytext=(6, -14))
    ax1.set_aspect("equal"); ax1.set_xlabel("x east (m)"); ax1.set_ylabel("y north (m)")
    ax1.set_title(f"Town01, one lap: {road.length:.0f} m, counterclockwise")
    d = c.drive(road, "expert", np.random.default_rng(a.seed))
    ax2.plot(d.s, d.e, color=EXPERT, lw=1, label="the expert's offset")
    lane_edges(ax2, road, 0)
    ax2.set_xlabel("distance along the lap (m)"); ax2.set_ylabel("offset from lane center (m), left +")
    ax2.set_ylim(-1.7, 1.7); ax2.legend(loc="lower right")
    ax2.set_title(f"expert: mean |offset| {np.abs(d.e).mean():.2f} m, largest {np.abs(d.e).max():.2f} m")


def fit_bc(road, a):
    rng = np.random.default_rng(a.seed)
    pol, demos = c.behavior_cloning(road, rng)
    return pol, demos


def part_bc(fig, road, a):
    pol, demos = fit_bc(road, a)
    d0 = demos[0]
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(d0.e_seen, np.degrees(d0.steer_cmd), ".", ms=2, color="0.75", label="recorded: offset seen, steering")
    xs = np.linspace(-1.5, 1.5, 2)
    we, wc = c.expert_weights()[0], pol.w[0]
    ax1.plot(xs, np.degrees(we * xs), color=EXPERT, lw=2, label=f"expert: {np.degrees(we):.2f} deg per m")
    ax1.plot(xs, np.degrees(wc * xs), color=CLONE, lw=2, label=f"clone learned: {np.degrees(wc):.2f} deg per m")
    ax1.set_xlabel("offset the detector reports (m)"); ax1.set_ylabel("steering (deg)")
    ax1.set_title("how hard to steer back, per meter of offset"); ax1.legend(fontsize=8, loc="upper right")
    held = [c.drive(road, "expert", np.random.default_rng(10_000 + a.seed))]
    test = c.drive(road, pol, np.random.default_rng(c.EVAL_SEEDS[0]))
    ax2.plot(test.s, test.e, color=CLONE, lw=1.2, label="the clone's offset")
    lane_edges(ax2, road, 0)
    ax2.set_ylim(-1.7, 1.7); ax2.legend(loc="lower right")
    ax2.set_xlabel("distance along the lap (m)"); ax2.set_ylabel("offset from lane center (m), left +")
    left = test.left_lane_at_m
    ax2.set_title(f"open-loop error {c.open_loop_error_deg(pol, held):.2f} deg; "
                  + ("finished the lap" if np.isnan(left) else f"left the lane after {left:.0f} m"))
    print("weights (offset, heading, curvature, constant):")
    print("  expert", np.round(c.expert_weights(), 4))
    print("  clone ", np.round(pol.w, 4))


def per_step_error(road, pol, drives, bins):
    """Mean |clone - expert| steering (deg) in each |true offset| bin, on the clone's own drives."""
    e = np.concatenate([np.abs(d.e) for d in drives])
    X, y = c.dataset(drives)
    err = np.degrees(np.abs(X @ pol.w - y))
    idx = np.digitize(e, bins) - 1
    return np.array([err[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(bins) - 1)]), \
        np.array([np.sum(idx == i) for i in range(len(bins) - 1)])


def part_shift(fig, road, a):
    pol, demos = fit_bc(road, a)
    ds, _ = c.evaluate(road, pol, c.EVAL_SEEDS[:5])
    bins = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.6])
    err, n = per_step_error(road, pol, ds, bins)
    ax1, ax2 = fig.subplots(1, 2)
    mid = (bins[:-1] + bins[1:]) / 2
    ax1.bar(mid, err, width=np.diff(bins) * 0.9, color=CLONE)
    ax1.set_xlabel("|true offset| (m)"); ax1.set_ylabel("mean |clone - expert| steering (deg)")
    ax1.set_title("the clone's error at each step, on its own drives")
    hb = np.linspace(-1.6, 1.6, 33)
    ax2.hist(demos[0].e, hb, color=EXPERT, alpha=0.6, density=True, label="expert's lap (training data)")
    ax2.hist(np.concatenate([d.e for d in ds]), hb, color=CLONE, alpha=0.6, density=True, label="the clone's drives")
    ax2.set_xlabel("true offset (m)"); ax2.set_ylabel("share of steps (density)")
    ax2.set_title("where each one drives"); ax2.legend(fontsize=8)
    for lo, hi, e_, k in zip(bins[:-1], bins[1:], err, n):
        print(f"|offset| {lo:.2f} to {hi:.2f} m: {k:5d} steps, mean error {e_:.2f} deg")


def part_dagger(fig, road, a):
    print("running DAgger and testing every iteration on 20 laps (about 15 s)...")
    r = c.experiment(road, iterations=a.iterations, train_seed=a.seed)
    rows, drives = r["rows"], r["drives"]
    ax1 = fig.add_axes([0.07, 0.25, 0.4, 0.65])
    ax2 = fig.add_axes([0.57, 0.25, 0.38, 0.65])
    sl = Slider(fig.add_axes([0.2, 0.06, 0.6, 0.04]), "iteration", 1, len(rows), valinit=len(rows), valstep=1)
    hb = np.linspace(-1.6, 1.6, 33)
    for row in rows:
        print(f"iteration {row['iteration']}: {row['samples']:6d} samples, offset weight "
              f"{np.degrees(row['weights'][0]):6.2f} deg/m, finished {row['finished']}/{row['drives']}, "
              f"open-loop error {row['open_loop_deg']:.2f} deg")
    print(f"expert: finished {r['expert']['finished']}/{r['expert']['drives']}")

    def draw(i):
        i = int(i)
        ax1.clear(); ax2.clear()
        ax1.hist(np.concatenate([d.e for d in drives[:i]]), hb, color=DAGGER, label=f"data after iteration {i}")
        ax1.set_xlabel("true offset in the data (m)"); ax1.set_ylabel("steps")
        ax1.set_title(f"{rows[i - 1]['samples']} labeled states"); ax1.legend(fontsize=8)
        it = [row["iteration"] for row in rows[:i]]
        ax2.plot(it, [row["finished"] for row in rows[:i]], "o-", color=DAGGER, label="laps finished of 20")
        ax2.axhline(r["expert"]["finished"], color=EXPERT, ls="--", lw=1, label="expert")
        ax2.set_xlim(0.5, len(rows) + 0.5); ax2.set_ylim(-1, 21)
        ax2.set_xlabel("DAgger iteration (1 = behavior cloning)"); ax2.legend(loc="lower right")
        ax2.set_title(f"offset weight {np.degrees(rows[i - 1]['weights'][0]):.2f} deg/m "
                      f"(expert {np.degrees(c.expert_weights()[0]):.2f})")
        fig.canvas.draw_idle()

    sl.on_changed(draw)
    draw(len(rows))
    fig._keep = sl


def part_eval(fig, road, a):
    print("testing the expert, the BC clone and the last DAgger clone on 20 laps (about 15 s)...")
    r = c.experiment(road, iterations=a.iterations, train_seed=a.seed)
    bc, dg = r["rows"][0], r["rows"][-1]
    names = ["behavior cloning", f"DAgger, iteration {dg['iteration']}"]
    ax1, ax2 = fig.subplots(1, 2)
    ax1.bar(names, [bc["open_loop_deg"], dg["open_loop_deg"]], color=[CLONE, DAGGER])
    ax1.set_ylabel("mean |clone - expert| steering (deg)"); ax1.set_title("open loop: on the expert's recorded lap")
    ax2.bar(names + ["expert"], [bc["finished"], dg["finished"], r["expert"]["finished"]],
            color=[CLONE, DAGGER, EXPERT])
    ax2.set_ylabel("laps finished, of 20"); ax2.set_ylim(0, 21)
    ax2.set_title("closed loop: each one drives")
    for name, row in zip(names, (bc, dg)):
        print(f"{name}: open-loop {row['open_loop_deg']:.2f} deg, finished {row['finished']}/{row['drives']}, "
              f"mean route completion {100 * row['completion_mean']:.0f} %")
    print(f"expert: finished {r['expert']['finished']}/{r['expert']['drives']}")


PARTS = {"road": part_road, "bc": part_bc, "shift": part_shift, "dagger": part_dagger, "eval": part_eval}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=list(PARTS), required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--html")
    ap.add_argument("--seed", type=int, default=c.TRAIN_SEED, help="the expert's recorded lap and DAgger's drives")
    ap.add_argument("--iterations", type=int, default=6, help="DAgger iterations, the first is behavior cloning")
    ap.add_argument("--detector-noise", type=float, default=c.CHOSEN["detector_sigma_offset_m"],
                    help="sigma of the detector's offset reading, m (default 0.35)")
    ap.add_argument("--gust", type=float, default=c.CHOSEN["gust_sigma_deg"],
                    help="sigma of the steering disturbance, deg (default 0.4)")
    a = ap.parse_args(argv)
    c.CHOSEN["detector_sigma_offset_m"] = a.detector_noise
    c.CHOSEN["gust_sigma_deg"] = a.gust
    road = c.load_road()
    fig = plt.figure(figsize=(12, 5.5))
    PARTS[a.part](fig, road, a)
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=120, facecolor="white", bbox_inches="tight")
    elif a.html:
        try:
            import mpld3
        except ImportError:
            sys.exit("--html needs mpld3 (pip install mpld3); or use --snapshot")
        Path(a.html).write_text(mpld3.fig_to_html(fig))
    else:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
