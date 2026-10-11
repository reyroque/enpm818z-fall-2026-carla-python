"""L6 hands-on: where is the AV? Five parts on one recorded drive.

  python3 l6_handson.py --part gnss   # the GNSS fixes: scatter, bias and the outage
  python3 l6_handson.py --part dr     # dead reckoning: wheels + gyro, no GNSS
  python3 l6_handson.py --part ekf    # the fused filter's results (8 runs)
  python3 l6_handson.py --part mcl    # particle filter on the map, lost start
  python3 l6_handson.py --part icp    # scan matching, one iteration at a time
Add --snapshot out.png to save a still, --html out.html for an animation.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider

import icp as icpmod
import l6common as c
from l6common import draw_av
import mcl as mclmod

HERE = Path(__file__).resolve().parent
RESULT_KEYS = ("t", "xy", "sigma_xy", "nis_t", "nis", "nis_source", "ape_rmse", "ape_max")


def load_results(folder, outage, wheels, map_):
    name = f"run_gps{int(not outage)}_wheels{int(wheels)}_map{int(map_)}.npz"
    path = Path(folder) / name
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    d = np.load(path)
    for k in RESULT_KEYS:
        if k not in d.files:
            raise KeyError(f"{name} has no '{k}'")
    return {k: d[k] for k in RESULT_KEYS}


def data():
    d = c.load_drive(HERE / "l6_drive.csv")
    st, scans = c.load_scans(HERE / "l6_scans.npz")
    mp = np.load(HERE / "l6_map.npz")["points"]
    return d, st, scans, mp


def outage_rows(d):
    """First and last row of the GNSS outage: the longest stretch with no fix."""
    fixes = np.flatnonzero(np.isfinite(d["gnss_x"]))
    j = int(np.argmax(np.diff(fixes)))
    return fixes[j], fixes[j + 1]            # the last fix before it, the first fix after


def gnss_stats(d):
    """What the fixes say against the truth: how many, the error of each, and the outage."""
    m = np.isfinite(d["gnss_x"])
    k0, k1 = outage_rows(d)
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(d["x"]), np.diff(d["y"])))])
    ex, ey = d["gnss_x"][m] - d["x"][m], d["gnss_y"][m] - d["y"][m]
    return {"mask": m, "k0": k0, "k1": k1, "dist": dist, "ex": ex, "ey": ey,
            "n": int(m.sum()), "mean": (float(ex.mean()), float(ey.mean())),
            "std": (float(ex.std()), float(ey.std())),
            "outage_s": float(d["t"][k1] - d["t"][k0]), "outage_m": float(dist[k1] - dist[k0])}


def _gnss_path(d, upto):
    """The GNSS-only path: the fixes up to row upto joined in order, broken where fixes
    are more than 1.5 s apart (the outage), so no line crosses the gap."""
    k = np.flatnonzero(np.isfinite(d["gnss_x"][:upto + 1]))
    x, y, t = d["gnss_x"][k], d["gnss_y"][k], d["t"][k]
    br = np.flatnonzero(np.diff(t) > 1.5) + 1
    return np.insert(x, br, np.nan), np.insert(y, br, np.nan)


def part_gnss(fig, d, *_, t_open=None):
    """The GNSS fixes as they arrive. Play runs the drive from start to finish, one second
    per tick; Step moves one second (one fix) at a time; Reset goes back to the start. A fix every second, then none for
    the outage. Opens at t_open (s), or at the start.
      top left   the whole route; the AV drawn 3x size so it shows at this zoom
      top right  a 40 m close-up that follows the AV, at true size, with a 1-sigma circle
                 (the chosen noise) around the newest fix
      bottom left  each fix's error, east and north, against time
      bottom right the same errors as points, so far in blue, later ones in gray: their
                   center is the bias, their spread the noise
    The two laps cover the same road, so on the map the gap is hidden; it shows in time."""
    import json
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    meta = json.loads((HERE / "l6_drive_meta.json").read_text())["chosen"]
    sigma, bias = meta["gnss_sigma_m"], meta["gnss_bias_m"]
    s = gnss_stats(d)
    m, k0, k1 = s["mask"], s["k0"], s["k1"]
    t = d["t"]
    ax_map = fig.add_axes([0.05, 0.56, 0.42, 0.38])
    ax_zoom = fig.add_axes([0.55, 0.56, 0.40, 0.38])
    ax_err = fig.add_axes([0.07, 0.13, 0.42, 0.33])
    ax_sc = fig.add_axes([0.55, 0.13, 0.22, 0.33])   # legend to its right
    t_start, t_end = float(t[0]), float(t[-1])
    now_t = [t_start if t_open is None else min(max(float(t_open), t_start), t_end)]
    b_play = Button(fig.add_axes([0.28, 0.02, 0.12, 0.045]), "Play")
    b_step = Button(fig.add_axes([0.44, 0.02, 0.12, 0.045]), "Step: +1 s")
    b_reset = Button(fig.add_axes([0.60, 0.02, 0.12, 0.045]), "Reset")

    # bottom left: error against time (fixed), with a line for the current time
    ax_err.axvspan(t[k0], t[k1], color="#c0392b", alpha=0.12,
                   label=f"last fix to first fix: {s['outage_s']:.1f} s, {s['outage_m']:.0f} m")
    ax_err.plot(t[m], s["ex"], ".", color="#1f6fb4", label="east error")
    ax_err.plot(t[m], s["ey"], ".", color="#c9682f", label="north error")
    ax_err.axhline(s["mean"][0], color="#1f6fb4", ls="--", lw=1)
    ax_err.axhline(s["mean"][1], color="#c9682f", ls="--", lw=1)
    ax_err.axhline(0, color="0.5", lw=0.5)
    lo = min(s["ex"].min(), s["ey"].min())
    ax_err.set_ylim(lo - 1.6, max(s["ex"].max(), s["ey"].max()) + 0.4)
    ax_err.set_xlabel("time (s)"); ax_err.set_ylabel("fix minus truth (m)")
    ax_err.legend(loc="lower center", ncol=3, fontsize=8)
    ax_err.set_title(f"{s['n']} fixes; dashed: their means")
    now = ax_err.axvline(now_t[0], color="k", lw=1)

    # bottom right: bias against noise; the fixes so far in blue, the later ones in gray
    ax_sc.plot(s["ex"], s["ey"], ".", color="0.8", label="later fixes")
    sc_sofar, = ax_sc.plot([], [], ".", color="#1f6fb4", label="one fix's error, so far")
    sc_last, = ax_sc.plot([], [], "o", mfc="none", mec="#1f6fb4", ms=10, label="last fix")
    ax_sc.add_patch(Circle(bias, sigma, fill=False, ls="--", color="#c9682f",
                           label=f"{sigma:g} m circle (noise we chose)"))
    ax_sc.plot(*bias, "+", color="#c9682f", ms=12, mew=2,
               label=f"bias we chose ({bias[0]:+g} E, {bias[1]:+g} N)")
    ax_sc.plot(*s["mean"], "x", color="k", ms=10, mew=2,
               label=f"mean of all {s['n']}, measured ({s['mean'][0]:+.2f} E, {s['mean'][1]:+.2f} N)")
    ax_sc.axhline(0, color="0.6", lw=0.5); ax_sc.axvline(0, color="0.6", lw=0.5)
    r = 1.15 * max(np.abs(s["ex"]).max(), np.abs(s["ey"]).max())
    ax_sc.set_xlim(-r, r); ax_sc.set_ylim(-r, r); ax_sc.set_aspect("equal")
    ax_sc.set_xlabel("east error (m)"); ax_sc.set_ylabel("north error (m)")
    ax_sc.set_title("bias (center) against noise (spread)")
    ax_sc.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=7, borderaxespad=0)

    proxy_av = Line2D([], [], marker="s", color="k", ls="", ms=7)

    def draw_map(ax, tc, k, zoom):
        got = m & (t <= tc)
        last = int(np.flatnonzero(got)[-1]) if got.any() else None
        ax.clear()
        ax.plot(d["x"], d["y"], "-", color="0.85", lw=0.8)
        ax.plot(d["x"][:k + 1], d["y"][:k + 1], "k-", lw=1, label="truth so far")
        px, py = _gnss_path(d, k)
        ax.plot(px, py, "-", color="#1f6fb4", lw=0.8, alpha=0.7, label="GNSS-only path")
        ax.plot(d["gnss_x"][got], d["gnss_y"][got], ".", color="#1f6fb4", ms=6, label="fixes so far")
        if last is not None:
            ax.plot(d["gnss_x"][last], d["gnss_y"][last], "o", mfc="none", mec="#1f6fb4",
                    ms=11 if not zoom else 8, label="last fix")
        draw_av(ax, d["x"][k], d["y"][k], d["theta"][k], scale=1.0 if zoom else 3.0)
        ax.set_aspect("equal")
        if zoom:
            if last is not None:
                ax.add_patch(Circle((d["gnss_x"][last], d["gnss_y"][last]), sigma, fill=False,
                                    ls="--", color="#1f6fb4", lw=1.2))
            ax.set_xlim(d["x"][k] - 20, d["x"][k] + 20); ax.set_ylim(d["y"][k] - 20, d["y"][k] + 20)
            ax.set_title(f"close-up, 40 m across, true size; dashed: {sigma:g} m around the last fix",
                         fontsize=9)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles + [proxy_av], labels + ["the AV (true size)"], loc="upper left", fontsize=7)
        else:
            # imshow (the AV) resets the limits to its own extent: fix them to the route
            pad = 8.0
            ax.set_xlim(np.nanmin(d["x"]) - pad, np.nanmax(d["x"]) + pad)
            ax.set_ylim(np.nanmin(d["y"]) - pad, np.nanmax(d["y"]) + pad)
            gap = tc - t[last] if last is not None else tc - t[0]
            if last is not None and gap > 1.5:
                ax.set_title(f"t = {tc:.0f} s: no fix for {gap:.0f} s; "
                             f"the AV drove {s['dist'][k] - s['dist'][last]:.0f} m since the last one", fontsize=9)
            else:
                ax.set_title(f"t = {tc:.0f} s: a fix every second", fontsize=9)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles + [proxy_av], labels + ["the AV (drawn 3x size)"], loc="center", fontsize=7)
        ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)")

    t_fix = t[m]                              # the time of each fix, in the order of ex, ey

    def draw(tc):
        now_t[0] = tc
        k = int(np.searchsorted(t, tc, side="right")) - 1
        draw_map(ax_map, tc, k, zoom=False)
        draw_map(ax_zoom, tc, k, zoom=True)
        now.set_xdata([tc, tc])
        n = int(np.searchsorted(t_fix, tc, side="right"))   # fixes so far
        sc_sofar.set_data(s["ex"][:n], s["ey"][:n])
        sc_last.set_data(s["ex"][n - 1:n], s["ey"][n - 1:n])
        fig.canvas.draw_idle()

    # Play: one second of the drive per 100 ms tick, so the 126 s drive takes about 13 s
    timer = fig.canvas.new_timer(interval=100)
    playing = [False]

    def stop():
        timer.stop()
        playing[0] = False
        b_play.label.set_text("Play")
        fig.canvas.draw_idle()

    def tick():
        if now_t[0] >= t_end:
            stop()
        else:
            draw(min(now_t[0] + 1.0, t_end))

    def on_play(_):
        if playing[0]:
            stop()
            return
        if now_t[0] >= t_end:                 # at the end: play again from the start
            draw(t_start)
        playing[0] = True
        b_play.label.set_text("Pause")
        timer.start()

    def on_step(_):
        if playing[0]:
            stop()
        draw(min(now_t[0] + 1.0, t_end))

    def on_reset(_):                          # stop and go back to the start of the drive
        if playing[0]:
            stop()
        draw(t_start)

    timer.add_callback(tick)
    b_play.on_clicked(on_play)
    b_step.on_clicked(on_step)
    b_reset.on_clicked(on_reset)
    draw(now_t[0])
    fig._keep = (b_play, b_step, b_reset, timer)


def part_dr(fig, d, *_):
    """Dead reckoning through the outage, from the true pose at the last fix
    before it (the best start it could have), as on the slide."""
    k0, k1 = outage_rows(d)
    sl = slice(k0, k1 + 1)
    x, y, _ = c.dead_reckon(d["t"][sl], d["wheel_v"][sl], d["gyro_z"][sl], d["x"][k0], d["y"][k0], d["theta"][k0])
    err = np.hypot(x - d["x"][sl], y - d["y"][sl])
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(d["x"][sl]), np.diff(d["y"][sl])))])
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(d["x"][sl], d["y"][sl], "k-", lw=1, label="truth")
    ax1.plot(x, y, color="#c9682f", label="dead reckoning")
    ax1.plot(d["x"][k0], d["y"][k0], "ko"); ax1.annotate("last fix", (d["x"][k0], d["y"][k0]), xytext=(6, -14),
                                                       textcoords="offset points")
    ax1.set_aspect("equal"); ax1.legend(); ax1.set_xlabel("x east (m)"); ax1.set_ylabel("y north (m)")
    ax2.plot(dist, err, color="#c9682f")
    ax2.set_xlabel("distance since the last fix (m)"); ax2.set_ylabel("position error (m)")
    ax2.set_title(f"after {dist[-1]:.0f} m: {err[-1]:.2f} m off; worst {err.max():.2f} m")


def part_ekf(fig, d, *_):
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    state = {"outage": True, "wheels": True, "map": False}

    def draw():
        r = load_results(HERE / "ekf_results", state["outage"], state["wheels"], state["map"])
        err = c.ape(r["xy"], np.column_stack([d["x"], d["y"]]))["errors"]
        ax1.clear(); ax2.clear()
        ax1.plot(r["t"], err, color="#1f6fb4", label="position error")
        ax1.fill_between(r["t"], 0, np.hypot(*r["sigma_xy"].T), color="#1f6fb4", alpha=0.2, label="filter's 1 sigma")
        ax1.set_ylabel("m"); ax1.legend(loc="upper right")
        ax1.set_title(f"APE RMSE {float(r['ape_rmse']):.2f} m, max {float(r['ape_max']):.2f} m")
        for s, col in (("gnss", "#1f6fb4"), ("map", "#2e7d32")):
            m = r["nis_source"] == s
            ax2.plot(r["nis_t"][m], r["nis"][m], ".", color=col, label=f"NIS {s}")
        ax2.set_ylabel("NIS"); ax2.set_xlabel("time (s)"); ax2.legend(loc="upper right")
        fig.canvas.draw_idle()

    cax = fig.add_axes([0.01, 0.4, 0.12, 0.15])
    cb = CheckButtons(cax, ["GNSS outage", "wheels", "map matching"], [True, True, False])

    def on(label):
        key = {"GNSS outage": "outage", "wheels": "wheels", "map matching": "map"}[label]
        state[key] = not state[key]
        draw()

    cb.on_clicked(on)
    fig.subplots_adjust(left=0.2)
    draw()
    fig._keep = cb


def part_mcl(fig, d, st, scans, mp, n=2000, seed=818):
    # Cold start: the first GNSS fix says "within about 20 m", heading unknown.
    first = int(np.flatnonzero(np.isfinite(d["gnss_x"]))[0])
    st, scans = st[::5], scans[::5]                     # one scan every 0.5 s
    out = mclmod.run_mcl(d["t"], d["wheel_v"], d["gyro_z"], st, scans, mclmod.LikelihoodField(mp), n=n,
                         rng=np.random.default_rng(seed), init="coarse",
                         init_pose=(d["gnss_x"][first], d["gnss_y"][first]), radius=20.0)
    k = [int(np.argmin(np.abs(d["t"] - s))) for s in st]
    err = np.hypot(out["mean"][:, 0] - d["x"][k], out["mean"][:, 1] - d["y"][k])
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(mp[:, 0], mp[:, 1], ",", color="0.6")
    ax1.plot(out["mean"][:, 0], out["mean"][:, 1], color="#2e7d32", label="MCL mean")
    ax1.plot(d["x"], d["y"], "k-", lw=0.8, label="truth"); ax1.set_aspect("equal"); ax1.legend()
    ax2.plot(st, err, color="#2e7d32", label="error (m)"); ax2.plot(st, out["spread"], "--", color="0.4", label="spread (m)")
    ax2.set_xlabel("time (s)"); ax2.legend()


def part_icp(fig, d, st, scans, mp, start=(1.5, -1.0, 5.0)):
    """ICP on the middle scan, started off the truth by start = (east m, north m, deg)."""
    j = len(scans) // 2
    k = int(np.argmin(np.abs(d["t"] - st[j])))
    th0 = d["theta"][k] + np.radians(start[2])
    t0 = np.array([d["x"][k] + start[0], d["y"][k] + start[1]])
    R0 = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
    _, _, hist = icpmod.icp(scans[j], mp, R0=R0, t0=t0, max_dist=3.0)
    ax = fig.add_axes([0.08, 0.2, 0.84, 0.75])
    sax = fig.add_axes([0.2, 0.05, 0.6, 0.04])
    sl = Slider(sax, "iteration", 0, len(hist), valinit=len(hist), valstep=1)
    near = mp[np.hypot(mp[:, 0] - d["x"][k], mp[:, 1] - d["y"][k]) < 40]

    def draw(i):
        ax.clear()
        R, t = (R0, t0) if i == 0 else (hist[int(i) - 1]["R"], hist[int(i) - 1]["t"])
        ax.plot(near[:, 0], near[:, 1], ".", ms=2, color="0.6", label="map")
        p = icpmod.apply(R, t, scans[j])
        ax.plot(p[:, 0], p[:, 1], ".", ms=3, color="#c9682f", label="scan")
        rmse = "" if i == 0 else f", RMSE {hist[int(i) - 1]['rmse']:.3f} m"
        ax.set_title(f"iteration {int(i)}{rmse}"); ax.set_aspect("equal"); ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    sl.on_changed(draw)
    draw(len(hist))
    fig._keep = sl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["gnss", "dr", "ekf", "mcl", "icp"], required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--html")
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=818)
    ap.add_argument("--time", type=float, help="gnss: open at this time (s) instead of the start")
    ap.add_argument("--icp-start", type=float, nargs=3, default=(1.5, -1.0, 5.0), metavar=("EAST", "NORTH", "DEG"),
                    help="ICP's starting guess, off the truth by EAST m, NORTH m and DEG degrees")
    a = ap.parse_args(argv)
    d, st, scans, mp = data()
    fig = plt.figure(figsize=(13, 10) if a.part == "gnss" else (12, 6))
    if a.part == "gnss":
        part_gnss(fig, d, t_open=a.time)
    elif a.part in ("dr", "ekf"):
        {"dr": part_dr, "ekf": part_ekf}[a.part](fig, d)
    elif a.part == "icp":
        part_icp(fig, d, st, scans, mp, start=a.icp_start)
    else:
        part_mcl(fig, d, st, scans, mp, n=a.particles, seed=a.seed)
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
    main(sys.argv[1:])
