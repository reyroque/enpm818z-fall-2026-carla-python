"""L12 hands-on: test a toy stack on a pedestrian crossing, in a small simulation.

  python3 l12_handson.py --part run      # one concrete run, played from above
  python3 l12_handson.py --part sweep    # the pass/fail map of the logical scenario
  python3 l12_handson.py --part sample   # 25 runs: a grid, a random sample, a search
  python3 l12_handson.py --part score    # GP4-style numbers and the Leaderboard score
Add --snapshot out.png to save a still, --html out.html for an animation (part run).

Everything runs on a laptop: numpy and matplotlib, no CARLA, no GPU. The street is
a Town01 street (town01.json); the scenario and the toy stack are in l12_sim.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider

import l12_sim as s

HERE = Path(__file__).resolve().parent
PED, FAIL, PASS, CLOSE, INK, HAZ = "#2D6CA2", "#C0392B", "#CFE8DA", "#E08A1E", "#1E2939", "#9A7B00"
CELLS = list(s.STACKS)


# ------------------------------------------------------------------ part run

def draw_street(ax, x0, x1):
    yn, yf = s.Y_NEAR_CURB, s.Y_FAR_CURB
    ax.add_patch(Rectangle((x0, yn - 1.5), x1 - x0, 1.5, fc="#E6E6E6", ec="none"))
    ax.add_patch(Rectangle((x0, yf), x1 - x0, 1.5, fc="#E6E6E6", ec="none"))
    for y in (yn, yf):
        ax.plot([x0, x1], [y, y], color=INK, lw=2)
    ax.plot([x0, x1], [s.LANE_W / 2] * 2, color="0.5", lw=1.2, ls=(0, (6, 5)))
    ax.set_xlim(x0, x1); ax.set_ylim(yn - 1.5, yf + 1.5); ax.set_aspect("equal")
    ax.set_xlabel("x along the road (m)"); ax.set_ylabel("y (m)")


def av_box(xb):
    """The AV's footprint from its base link (rear axle middle), driving to +x."""
    return Rectangle((xb - s.REAR_TO_BASE, -s.AV_WIDTH / 2), s.AV_LENGTH, s.AV_WIDTH,
                     fc=INK, ec="none", zorder=5)


def run_summary(q):
    if q["collided"]:
        return f"COLLISION at {q['impact_v'] * 3.6:.1f} km/h"
    if np.isfinite(q["stop_gap"]):
        return f"pass: stopped {q['stop_gap']:.2f} m before the crossing line"
    return f"pass: closest {q['clearance']:.2f} m"


def part_run(fig, v0, d0, vped, cell, anim_out=None):
    """One concrete run from above, and the gap against time. Play or drag the slider."""
    st = s.STACKS[cell]
    q = s.simulate(v0, d0, vped, st, record=True)
    S = q["series"]
    xc = d0 + s.BASE_TO_FRONT
    ax = fig.add_axes([0.06, 0.52, 0.9, 0.4])
    ax2 = fig.add_axes([0.08, 0.14, 0.86, 0.28])
    draw_street(ax, -6, xc + 10)
    hw = s.LANE_W / 2 + st.margin
    ax.add_patch(Rectangle((-6, -hw), xc + 16, 2 * hw, fc=HAZ, alpha=0.08, ec="none"))
    ax.plot([xc, xc], [s.Y_NEAR_CURB, s.Y_FAR_CURB], ls=":", color=PED)
    box = av_box(0.0); ax.add_patch(box)
    ped = Circle((xc, s.Y_FAR_CURB), s.PED_RADIUS, fc=PED, zorder=6); ax.add_patch(ped)
    ax.set_title(f"{cell}: v0 = {v0:g} m/s, d0 = {d0:g} m, vped = {vped:g} m/s. {run_summary(q)}")
    gap = d0 - S["xb"]
    ax2.plot(S["t"], gap, color=INK, lw=2)
    ax2.axhline(0, color="0.5", lw=0.8)
    if np.isfinite(q["t_brake"]):
        ax2.axvline(q["t_brake"], color="0.5", ls=":", label=f"brakes at {q['t_brake']:.2f} s")
        ax2.legend(loc="upper right")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("gap to the crossing line (m)")
    now = ax2.axvline(0, color=FAIL, lw=1)
    n = len(S["t"])
    sl = Slider(fig.add_axes([0.2, 0.03, 0.5, 0.03]), "time (s)", 0, float(S["t"][-1]), valinit=0)
    state = {"playing": False}

    def show(k):
        k = int(min(max(k, 0), n - 1))
        box.set_x(S["xb"][k] - s.REAR_TO_BASE)
        ped.center = (xc, S["yp"][k])
        now.set_xdata([S["t"][k]] * 2)
        fig.canvas.draw_idle()
        return box, ped, now

    sl.on_changed(lambda t: show(round(t / 0.01)))
    b = Button(fig.add_axes([0.78, 0.025, 0.1, 0.04]), "Play")
    timer = fig.canvas.new_timer(interval=40)

    def tick():
        t = sl.val + 0.04
        if t > S["t"][-1]:
            t = 0.0
        sl.set_val(t)

    timer.add_callback(tick)

    def toggle(_):
        state["playing"] = not state["playing"]
        (timer.start if state["playing"] else timer.stop)()

    b.on_clicked(toggle)
    fig._keep = (sl, b, timer)
    if anim_out:
        from matplotlib.animation import FuncAnimation
        frames = range(0, n, 4)
        an = FuncAnimation(fig, lambda k: show(k), frames=frames, interval=40, blit=False)
        Path(anim_out).write_text(an.to_jshtml())
    sl.set_val(float(S["t"][-1]))
    return q


# ------------------------------------------------------------------ part sweep

def part_sweep(fig, cell, vped, n_v=60, n_d=110):
    """The pass/fail map. Pick an ODD cell, move the walking speed, click a cell to run it."""
    ax = fig.add_axes([0.08, 0.12, 0.6, 0.8])
    info = fig.text(0.72, 0.55, "", fontsize=11, va="top")
    radio = RadioButtons(fig.add_axes([0.72, 0.68, 0.22, 0.22]), CELLS, active=CELLS.index(cell))
    sl = Slider(fig.add_axes([0.2, 0.02, 0.45, 0.03]), "vped (m/s)", *s.LOGICAL["vped"], valinit=vped)
    V, D = s.grid(n_v, n_d)
    cm = ListedColormap([PASS, CLOSE, FAIL])
    (vlo, vhi), (dlo, dhi) = s.LOGICAL["v0"], s.LOGICAL["d0"]
    state = {"cell": cell}

    def redraw(*_):
        r = s.simulate(V, D, sl.val, s.STACKS[state["cell"]])
        f, cl = r["collided"], r["clearance"]
        img = np.where(f, 1.0, np.where(cl < 0.5, 0.5, 0.0))
        ax.clear()
        ax.imshow(img.T, origin="lower", extent=(vlo, vhi, dlo, dhi), aspect="auto", cmap=cm,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlabel("AV speed v0 (m/s)"); ax.set_ylabel("gap d0 (m)")
        ax.set_title(f"{state['cell']}, vped {sl.val:.2f} m/s: {100 * f.mean():.1f} % collide "
                     f"(red), {100 * ((~f) & (cl < 0.5)).mean():.1f} % pass within 0.5 m (orange)")
        fig.canvas.draw_idle()

    def pick(name):
        state["cell"] = name
        redraw()

    def click(ev):
        if ev.inaxes is not ax:
            return
        q = s.simulate(ev.xdata, ev.ydata, sl.val, s.STACKS[state["cell"]])
        info.set_text(f"v0 = {ev.xdata:.2f} m/s\nd0 = {ev.ydata:.1f} m\n{run_summary(q)}\n\n"
                      f"python3 l12_handson.py --part run\n  --v0 {ev.xdata:.2f} --d0 {ev.ydata:.1f}\n"
                      f"  --vped {sl.val:.2f} --cell {state['cell']}")
        fig.canvas.draw_idle()

    radio.on_clicked(pick)
    sl.on_changed(redraw)
    fig.canvas.mpl_connect("button_press_event", click)
    fig._keep = (radio, sl)
    redraw()


# ------------------------------------------------------------------ part sample

def part_sample(fig, budget, seed):
    """The same budget spent three ways, over the true band (light red, from a fine sweep)."""
    V, D = s.grid(120, 220)
    f = s.simulate(V, D, s.DEFAULT_VPED)["collided"]
    (vlo, vhi), (dlo, dhi) = s.LOGICAL["v0"], s.LOGICAL["d0"]
    axs = [fig.add_axes([0.06 + 0.32 * i, 0.16, 0.27, 0.72]) for i in range(3)]
    sl = Slider(fig.add_axes([0.25, 0.03, 0.4, 0.03]), "budget", 4, 400, valinit=budget, valstep=1)
    b = Button(fig.add_axes([0.75, 0.02, 0.15, 0.045]), "new random seed")
    state = {"seed": seed}

    def draw(*_):
        n = int(sl.val)
        k = max(2, int(round(np.sqrt(n))))
        vg, dg = s.grid_edges(k, k)
        fg = s.simulate(vg, dg, s.DEFAULT_VPED)["collided"]
        v, d, p = s.random_runs(n, state["seed"])
        fr = s.simulate(v, d, p)["collided"]
        bv, bd, bf = s.boundary_search(levels=3)
        for ax, (x, y, ff, title) in zip(axs, (
                (vg.ravel(), dg.ravel(), fg.ravel(), f"grid {k} x {k}: {fg.sum()} of {fg.size} fail"),
                (v, d, fr, f"random {n} (seed {state['seed']}): {fr.sum()} fail"),
                (bv, bd, bf, f"search: {bf.sum()} of {bv.size} fail"))):
            ax.clear()
            ax.imshow(f.T.astype(float), origin="lower", extent=(vlo, vhi, dlo, dhi), aspect="auto",
                      cmap=ListedColormap(["white", "#F5D3CF"]), vmin=0, vmax=1, interpolation="nearest")
            ax.scatter(x[~ff], y[~ff], s=14, facecolor="white", edgecolor=INK, lw=0.8)
            ax.scatter(x[ff], y[ff], s=14, color=FAIL)
            ax.set_title(title, fontsize=10); ax.set_xlabel("AV speed v0 (m/s)")
        axs[0].set_ylabel("gap d0 (m)")
        ph = fr.mean()
        fig.suptitle(f"true share that collides: {100 * f.mean():.1f} %;  random estimate "
                     f"{100 * ph:.1f} % +/- {196 * np.sqrt(ph * (1 - ph) / n):.1f} (95 %)")
        fig.canvas.draw_idle()

    def reseed(_):
        state["seed"] += 1
        draw()

    sl.on_changed(draw)
    b.on_clicked(reseed)
    fig._keep = (sl, b)
    draw()


# ------------------------------------------------------------------ part score

def part_score(fig, rc, counts):
    """Leaderboard 2.0 and 2.1 scores of one route, and what n clean runs can claim."""
    ax = fig.add_axes([0.1, 0.55, 0.5, 0.35])
    ax2 = fig.add_axes([0.1, 0.1, 0.5, 0.3])
    kinds = ("pedestrian", "vehicle", "static", "red_light", "stop_sign")
    sls = {"rc": Slider(fig.add_axes([0.72, 0.85, 0.2, 0.03]), "route %", 0, 100, valinit=rc)}
    for i, k in enumerate(kinds):
        sls[k] = Slider(fig.add_axes([0.72, 0.78 - 0.06 * i, 0.2, 0.03]), k, 0, 6,
                        valinit=counts.get(k, 0), valstep=1)
    n_sl = Slider(fig.add_axes([0.72, 0.3, 0.2, 0.03]), "clean runs n", 1, 200, valinit=3, valstep=1)

    def draw(*_):
        inf = {k: int(sls[k].val) for k in kinds}
        sc21, p21 = s.driving_score(sls["rc"].val, inf, "2.1")
        sc20, p20 = s.driving_score(sls["rc"].val, inf, "2.0")
        ax.clear()
        ax.barh(["2.0", "2.1"], [sc20, sc21], color=["0.6", INK])
        ax.set_xlim(0, 100)
        ax.set_xlabel("driving score = route completion x penalty")
        ax.set_title(f"penalty: 2.0 gives {p20:.3f}, 2.1 gives {p21:.3f}")
        for y, v in enumerate((sc20, sc21)):
            ax.text(v + 1, y, f"{v:.1f}", va="center")
        n = int(n_sl.val)
        nn = np.arange(1, 201)
        ax2.clear()
        ax2.plot(nn, 100 * s.zero_failure_bound(nn), color=INK)
        b = 100 * s.zero_failure_bound(n)
        ax2.plot([n], [b], "o", color=FAIL)
        ax2.set_xscale("log"); ax2.set_ylim(0, 100)
        ax2.set_xlabel("runs in a row with no collision")
        ax2.set_ylabel("largest collision\nchance per run (%)")
        ax2.set_title(f"{n} clean runs: the chance per run is below {b:.1f} % (95 % confidence)")
        fig.canvas.draw_idle()

    for w in list(sls.values()) + [n_sl]:
        w.on_changed(draw)
    fig._keep = (sls, n_sl)
    draw()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["run", "sweep", "sample", "score"], required=True)
    ap.add_argument("--v0", type=float, default=10.0, help="run: the AV's speed (m/s)")
    ap.add_argument("--d0", type=float, default=36.0, help="run: the gap to the crossing line (m)")
    ap.add_argument("--vped", type=float, default=s.DEFAULT_VPED, help="the pedestrian's speed (m/s)")
    ap.add_argument("--cell", choices=CELLS, default="dry_day", help="the ODD cell (the toy stack's values)")
    ap.add_argument("--budget", type=int, default=25, help="sample: number of runs")
    ap.add_argument("--seed", type=int, default=1, help="sample: random seed")
    ap.add_argument("--snapshot")
    ap.add_argument("--html", help="run: write an animation to this HTML file")
    a = ap.parse_args(argv)
    if a.snapshot or a.html:
        matplotlib.use("Agg", force=True)
    fig = plt.figure(figsize=(13, 7.5))
    if a.part == "run":
        part_run(fig, a.v0, a.d0, a.vped, a.cell, anim_out=a.html)
    elif a.part == "sweep":
        part_sweep(fig, a.cell, a.vped)
    elif a.part == "sample":
        part_sample(fig, a.budget, a.seed)
    else:
        part_score(fig, 80, {"vehicle": 1, "red_light": 1})
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=110, facecolor="white")
    elif a.html and a.part != "run":
        sys.exit("--html makes an animation of part run only; use --snapshot for the others")
    elif not a.html:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
