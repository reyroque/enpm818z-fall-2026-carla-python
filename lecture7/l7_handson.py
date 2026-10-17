"""L7 hands-on: one Town01 trip, from the map file to a route. Five parts.

  python3 l7_handson.py --part map      # Town01's map file, read with CARLA's Python API
  python3 l7_handson.py --part graph    # Town01's waypoint graph; click a node to see it
  python3 l7_handson.py --part costs    # the trip's route under three edge costs
  python3 l7_handson.py --part search   # Dijkstra and A*, replayed step by step (slider)
  python3 l7_handson.py --part replan   # an obstacle on the route, and the new route
Add --snapshot out.png to save a still, --html out.html for a page (needs mpld3).

The routes and search orders were computed in advance and saved in l7_results/.
This script replays and draws them; it has no search code. GP4 asks you to write it.
The map part needs CARLA's Python package (pip install carla==0.9.16), no server.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import l7common as c

HERE = Path(__file__).resolve().parent
LANE, ROUTE, FAST, EXPANDED, BLOCK, NEW, INK = "#C3C9D2", "#2D6CA2", "#C2552F", "#E8A33D", "#C0392B", "#1E9E74", "#1E2939"


def draw_lanes(ax, g, lw=1.0):
    xy = g["node_xy"]
    seg = np.stack([xy[g["edge_from"]], xy[g["edge_to"]]], axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(seg, colors=LANE, linewidths=lw, zorder=1))


def draw_route(ax, g, route, color, lw=3.0, **kw):
    xy = g["node_xy"][route]
    return ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, zorder=4, **kw)[0]


def start_goal(ax, g, s):
    xy = g["node_xy"]
    for k, name in ((s["trip"]["start_node"], "S"), (s["trip"]["goal_node"], "G")):
        ax.plot(*xy[k], "o", ms=9, color=INK, zorder=6)
        ax.annotate(name, xy[k], xytext=(6, 8), textcoords="offset points", fontsize=13, weight="bold")


def axes_map(ax, xlim=(-15, 410), ylim=(-345, 25)):
    ax.set_aspect("equal")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("x, east (m)"); ax.set_ylabel("y, north (m)")


def part_graph(fig, g, r, s):
    ax = fig.add_subplot(111)
    draw_lanes(ax, g)
    xy = g["node_xy"]
    two = np.bincount(g["edge_from"], minlength=len(xy)) == 2
    ax.scatter(*xy.T, s=4, color=INK, zorder=2)
    ax.scatter(*xy[two].T, s=30, color=EXPANDED, zorder=3, label="a node with two successors")
    gs = s["graph"]
    ax.set_title(f"Town01 at 2 m: {gs['nodes']} nodes, {gs['edges']} directed edges, "
                 f"{gs['total_lane_length_m'] / 1000:.1f} km of lane", fontsize=12)
    ax.legend(loc="upper right")
    axes_map(ax)
    info = ax.text(0.01, 0.01, "click a node", transform=ax.transAxes, fontsize=10, va="bottom",
                   bbox=dict(fc="white", ec=LANE))

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        k = int(np.argmin(np.hypot(xy[:, 0] - ev.xdata, xy[:, 1] - ev.ydata)))
        succ = g["edge_to"][g["edge_from"] == k]
        lens = g["edge_len"][g["edge_from"] == k]
        info.set_text(f"node {k}: ({xy[k][0]:.1f}, {xy[k][1]:.1f}) m, heading {g['node_yaw_deg'][k]:.0f} deg, "
                      f"road {g['node_road'][k]}, lane {g['node_lane'][k]}"
                      + (" (in an intersection)" if g["node_junction"][k] else "")
                      + "\nsuccessors: " + ", ".join(f"{int(a)} ({b:.2f} m)" for a, b in zip(succ, lens)))
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("button_press_event", on_click)


def part_costs(fig, g, r, s):
    ax = fig.add_subplot(111)
    draw_lanes(ax, g, lw=1.4)
    t = s["trip"]["routes"]
    draw_route(ax, g, r["route_distance"], ROUTE, lw=6,
               label=f"cost = distance: {t['distance']['length_m']} m, {t['distance']['time_s']} s")
    draw_route(ax, g, r["route_time"], FAST, lw=3, ls="--",
               label=f"cost = travel time: {t['time']['length_m']} m, {t['time']['time_s']} s")
    draw_route(ax, g, r["route_left_penalty"], INK, lw=1.2, ls=":",
               label=f"cost = distance + {s['chosen']['left_penalty_m']:.0f} m per left turn: same as travel time")
    start_goal(ax, g, s)
    ax.legend(loc="lower right", fontsize=10)
    axes_map(ax, (60, 225), (-215, 25))


def part_search(fig, g, r, s):
    """Replays the saved expansion orders: the first k nodes each search took from its open set."""
    xy = g["node_xy"]
    orders = {"Dijkstra": r["order_dijkstra"], "A*": r["order_a_star"]}
    axs = [fig.add_axes([0.08 + 0.48 * i, 0.14, 0.42, 0.78]) for i in range(2)]
    dots = {}
    for ax, (name, order) in zip(axs, orders.items()):
        draw_lanes(ax, g)
        start_goal(ax, g, s)
        dots[name] = ax.scatter(*xy[order[:1]].T, s=10, color=EXPANDED, zorder=3)
        ax.route = draw_route(ax, g, r["route_distance"], ROUTE)
        ax.route.set_visible(False)
        axes_map(ax)
    n = max(len(o) for o in orders.values())
    sax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    sl = Slider(sax, "step", 1, n, valinit=n, valstep=1)

    def update(k):
        k = int(k)
        for ax, (name, order) in zip(axs, orders.items()):
            m = min(k, len(order))
            dots[name].set_offsets(xy[order[:m]])
            done = k >= len(order)
            ax.route.set_visible(done)
            ax.set_title(f"{name}: {m} of {len(order)} nodes taken" + (" (goal reached)" if done else ""), fontsize=12)
        fig.canvas.draw_idle()
    sl.on_changed(update)
    update(n)
    fig._keep = sl


def part_replan(fig, g, r, s):
    ax = fig.add_subplot(111)
    draw_lanes(ax, g, lw=1.4)
    xy = g["node_xy"]
    rp = s["replan"]
    old = r["route_distance"]
    k0 = int(np.flatnonzero(old == rp["from_node"])[0])
    draw_route(ax, g, old[k0:], ROUTE, lw=6, label=f"old route, from the AV: {rp['old_rest_m']} m")
    draw_route(ax, g, r["replan_route"], NEW, lw=3, ls="--", label=f"new route: {rp['new_m']} m")
    o = s["chosen"]["obstacle_xy"]
    ax.add_patch(plt.Circle(o, s["chosen"]["clearance_m"], fill=False, color=BLOCK, lw=2))
    ax.plot(*o, "X", ms=12, color=BLOCK, label=f"obstacle, {s['chosen']['clearance_m']} m clearance")
    ax.plot(*xy[rp["from_node"]], "^", ms=11, color=INK, label="the AV when it sees the obstacle")
    start_goal(ax, g, s)
    ax.legend(loc="lower right", fontsize=10)
    axes_map(ax, (70, 215), (-95, 18))


def part_map(fig, g, r, s):
    """Read maps/Town01.xodr offline: what the file holds, and the waypoints CARLA makes of it."""
    try:
        import carla
    except ImportError:
        sys.exit("--part map needs CARLA's Python package: pip install carla==0.9.16 (no server needed)")
    import xml.etree.ElementTree as ET
    text = (HERE / "maps" / "Town01.xodr").read_text()
    root = ET.fromstring(text)
    roads = root.findall("road")
    in_j = [d for d in roads if d.get("junction") != "-1"]
    drive_j = [d for d in in_j if any(l.get("type") == "driving" for l in d.iter("lane"))]
    lights = [x for x in root.iter("signal") if x.get("type") == "1000001"]
    m = carla.Map("Town01", text)                       # no CARLA server: the file is enough
    wps = m.generate_waypoints(2.0)
    topo = m.get_topology()
    print(f"Town01.xodr: {len(text.splitlines())} lines; {len(roads)} roads, {len(in_j)} of them inside the "
          f"{len(root.findall('junction'))} junctions ({len(drive_j)} with a driving lane); "
          f"{len(lights)} traffic lights")
    print(f"carla.Map: {len(wps)} waypoints at 2 m, {len(topo)} topology segments")
    xy = np.array([[w.transform.location.x, -w.transform.location.y] for w in wps])
    jn = np.array([w.is_junction for w in wps])
    ax = fig.add_subplot(111)
    ax.scatter(*xy[~jn].T, s=3, color=INK, zorder=2, label="a waypoint on a road")
    ax.scatter(*xy[jn].T, s=3, color=EXPANDED, zorder=2, label="a waypoint inside an intersection")
    ax.set_title(f"Town01.xodr read offline: {len(roads)} roads, {len(wps)} waypoints at 2 m", fontsize=12)
    ax.legend(loc="center", markerscale=4)
    axes_map(ax)
    line_of = {d.get("id"): next(i for i, l in enumerate(text.splitlines(), 1) if f'id="{d.get("id")}"' in l
                                 and "<road" in l) for d in roads}
    info = ax.text(0.01, 0.01, "click a waypoint", transform=ax.transAxes, fontsize=10, va="bottom",
                   bbox=dict(fc="white", ec=LANE))

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        w = wps[int(np.argmin(np.hypot(xy[:, 0] - ev.xdata, xy[:, 1] - ev.ydata)))]
        rid = str(w.road_id)
        info.set_text(f"road {w.road_id}, lane {w.lane_id}, s = {w.s:.1f} m along the road, lane width "
                      f"{w.lane_width:.1f} m" + (f", intersection {w.junction_id}" if w.is_junction else "")
                      + f"\nin Town01.xodr: line {line_of.get(rid, '?')}, the road with id=\"{rid}\"")
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("button_press_event", on_click)


PARTS = {"map": part_map, "graph": part_graph, "costs": part_costs, "search": part_search, "replan": part_replan}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=list(PARTS), required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--html")
    a = ap.parse_args(argv)
    g = c.load_graph()
    r, s = c.load_results()
    fig = plt.figure(figsize=(12, 8) if a.part == "search" else (10, 8))
    PARTS[a.part](fig, g, r, s)
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=110, facecolor="white")
        plt.close(fig)
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
