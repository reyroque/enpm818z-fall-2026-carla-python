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
from matplotlib.widgets import CheckButtons, Slider

import icp as icpmod
import l6common as c
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


def part_gnss(fig, d, *_):
    """The GNSS fixes as they arrive. Drag the slider: a fix every second, then none for
    the outage. The two laps cover the same road, so the gap shows in time, not on the map.
    Opens in the middle of the outage."""
    s = gnss_stats(d)
    m, k0, k1, dist = s["mask"], s["k0"], s["k1"], s["dist"]
    ax1 = fig.add_axes([0.06, 0.2, 0.40, 0.68])
    ax2 = fig.add_axes([0.56, 0.2, 0.40, 0.68])
    sax = fig.add_axes([0.2, 0.06, 0.6, 0.035])
    t = d["t"]
    sl = Slider(sax, "time (s)", float(t[0]), float(t[-1]), valinit=float((t[k0] + t[k1]) / 2))
    ax2.axvspan(t[k0], t[k1], color="#c0392b", alpha=0.12,
                label=f"last fix to first fix: {s['outage_s']:.1f} s, {s['outage_m']:.0f} m")
    ax2.plot(t[m], s["ex"], ".", color="#1f6fb4", label="east error")
    ax2.plot(t[m], s["ey"], ".", color="#c9682f", label="north error")
    ax2.axhline(s["mean"][0], color="#1f6fb4", ls="--", lw=1)
    ax2.axhline(s["mean"][1], color="#c9682f", ls="--", lw=1)
    ax2.axhline(0, color="0.5", lw=0.5)
    lo = min(s["ex"].min(), s["ey"].min())
    ax2.set_ylim(lo - 1.6, max(s["ex"].max(), s["ey"].max()) + 0.4)   # room for the legend below the dots
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("fix minus truth (m)")
    ax2.legend(loc="lower center", ncol=3, fontsize=8)
    ax2.set_title(f"{s['n']} fixes. Mean (dashed) {s['mean'][0]:+.2f} m east, {s['mean'][1]:+.2f} m north;"
                  f"\nspread {s['std'][0]:.2f} m east, {s['std'][1]:.2f} m north")
    now = ax2.axvline(sl.valinit, color="k", lw=1)

    def draw(tc):
        k = int(np.searchsorted(t, tc, side="right")) - 1
        got = m & (t <= tc)
        ax1.clear()
        ax1.plot(d["x"], d["y"], "-", color="0.85", lw=0.8)
        ax1.plot(d["x"][:k + 1], d["y"][:k + 1], "k-", lw=1, label="truth so far")
        ax1.plot(d["gnss_x"][got], d["gnss_y"][got], ".", color="#1f6fb4", ms=6, label="fixes so far")
        ax1.plot(d["x"][k], d["y"][k], "o", color="k", ms=7, label="the AV now")
        last = int(np.flatnonzero(got)[-1]) if got.any() else None
        if last is not None:
            ax1.plot(d["gnss_x"][last], d["gnss_y"][last], "o", mfc="none", mec="#1f6fb4", ms=11,
                     label="last fix")
        gap = tc - t[last] if last is not None else tc - t[0]
        if last is not None and gap > 1.5:
            ax1.set_title(f"t = {tc:.0f} s: no fix for {gap:.0f} s;\n"
                          f"the AV drove {dist[k] - dist[last]:.0f} m since the last one")
        else:
            ax1.set_title(f"t = {tc:.0f} s: a fix every second")
        ax1.set_aspect("equal"); ax1.legend(loc="center", fontsize=8)
        ax1.set_xlabel("x east (m)"); ax1.set_ylabel("y north (m)")
        now.set_xdata([tc, tc])
        fig.canvas.draw_idle()

    sl.on_changed(draw)
    draw(sl.valinit)
    fig._keep = sl


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
    ap.add_argument("--icp-start", type=float, nargs=3, default=(1.5, -1.0, 5.0), metavar=("EAST", "NORTH", "DEG"),
                    help="ICP's starting guess, off the truth by EAST m, NORTH m and DEG degrees")
    a = ap.parse_args(argv)
    d, st, scans, mp = data()
    fig = plt.figure(figsize=(12, 6))
    if a.part in ("gnss", "dr", "ekf"):
        {"gnss": part_gnss, "dr": part_dr, "ekf": part_ekf}[a.part](fig, d)
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
