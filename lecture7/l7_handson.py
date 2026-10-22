"""L7 hands-on: one Town01 trip, from the map file to a route. Five parts.

  python3 l7_handson.py --part map      # Town01's map file: zoom, search a road, lane, junction, segment
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


FOUND = "#D81B60"   # what a search found, in the map window only


def load_map():
    """Town01.xodr read offline: the parsed XML, CARLA's map, its waypoints and segments."""
    try:
        import carla
    except ImportError:
        sys.exit("--part map needs CARLA's Python package: pip install carla==0.9.16 (no server needed)")
    import xml.etree.ElementTree as ET
    text = (HERE / "maps" / "Town01.xodr").read_text()
    root = ET.fromstring(text)
    m = carla.Map("Town01", text)                       # no CARLA server: the file is enough
    wps = m.generate_waypoints(2.0)
    lines = text.splitlines()
    roads = {d.get("id"): d for d in root.findall("road")}
    line_of = {}
    for i, l in enumerate(lines, 1):
        if "<road " in l:
            for rid in roads:
                if f'id="{rid}"' in l and rid not in line_of:
                    line_of[rid] = i
    return {"text": text, "root": root, "map": m, "wps": wps, "topo": m.get_topology(),
            "roads": roads, "line_of": line_of}


def map_point(x, y, M):
    """The lane point nearest to (x, y) in the map frame (x east, y north), from CARLA's
    Map.get_waypoint: the lane centre the AV at (x, y) would be on. Returns (the lane point
    as a 1 x 2 array, a one-line description)."""
    import carla
    w = M["map"].get_waypoint(carla.Location(x=float(x), y=-float(y), z=0.0))   # CARLA's y is flipped
    px, py = w.transform.location.x, -w.transform.location.y
    rid = str(w.road_id)
    return (np.array([[px, py]]),
            f"({x:g}, {y:g}): nearest lane point {np.hypot(px - x, py - y):.2f} m away, road {w.road_id}, "
            f"lane {w.lane_id}, s = {w.s:.1f} m along the road"
            + (f", junction {w.junction_id}" if w.is_junction else "")
            + f"; Town01.xodr line {M['line_of'].get(rid, '?')}")


def map_search(query, M):
    """A search typed in the map window. Returns (points as an N x 2 array in the map frame,
    a one-line description), or (None, why nothing matched).

        road 25        every waypoint of road 25, and what the file says about it
        lane 25 -1     the waypoints of one lane of one road
        junction 332   the waypoints inside one intersection
        segment 12     one of get_topology()'s segments, start to end
        waypoint 100   one waypoint of generate_waypoints(2.0)
        25             the same as "road 25"
        133.6 -129.4   the lane point nearest to x = 133.6, y = -129.4 m (also "xy 133.6 -129.4")
    """
    words = query.strip().lower().split()
    if not words:
        return None, "type a search: road 25, lane 25 -1, junction 332, segment 12, waypoint 100"
    def number(s_):
        try:
            float(s_)
            return True
        except ValueError:
            return False
    if words[0] == "xy" or (len(words) == 2 and all(number(w_) for w_ in words)
                            and any("." in w_ for w_ in words)):
        vals = [float(w_.rstrip(",")) for w_ in (words[1:] if words[0] == "xy" else words)]
        if len(vals) != 2:
            return None, "type x and y in meters, e.g. 133.6 -129.4"
        return map_point(vals[0], vals[1], M)
    if words[0].lstrip("-").isdigit():
        words = ["road"] + words
    kind, args = words[0], words[1:]
    xy = lambda ws: np.array([[w.transform.location.x, -w.transform.location.y] for w in ws])
    try:
        nums = [int(x) for x in args]
    except ValueError:
        return None, f"not a number in: {query}"
    wps = M["wps"]
    if kind == "road" and len(nums) == 1:
        rid = str(nums[0])
        if rid not in M["roads"]:
            return None, f"no road {rid} in Town01.xodr ({len(M['roads'])} roads)"
        d = M["roads"][rid]
        ws = [w for w in wps if w.road_id == nums[0]]
        link = d.find("link")
        ends = [f"{e.tag} {e.get('elementType')} {e.get('elementId')}" for e in link] if link is not None else []
        lanes = sorted({w.lane_id for w in ws})
        where = "inside junction " + d.get("junction") if d.get("junction") != "-1" else "between intersections"
        return (xy(ws) if ws else None,
                f"road {rid}: {float(d.get('length')):.2f} m, {where}; driving lanes {lanes}; "
                f"{', '.join(ends)}; line {M['line_of'].get(rid, '?')} of Town01.xodr")
    if kind == "lane" and len(nums) == 2:
        ws = [w for w in wps if w.road_id == nums[0] and w.lane_id == nums[1]]
        if not ws:
            return None, f"no driving lane {nums[1]} on road {nums[0]}"
        th = (-ws[0].transform.rotation.yaw) % 360            # map frame: CARLA's y flipped
        way = ["east", "north", "west", "south"][int(((th + 45) % 360) // 90)]
        return xy(ws), (f"road {nums[0]}, lane {nums[1]}: {len(ws)} waypoints at 2 m, "
                        f"{ws[0].lane_width:.1f} m wide, heading {th:.0f} deg from east ({way})")
    if kind == "junction" and len(nums) == 1:
        ws = [w for w in wps if w.is_junction and w.junction_id == nums[0]]
        if not ws:
            ids = sorted({w.junction_id for w in wps if w.is_junction})
            return None, f"no junction {nums[0]}; Town01's are {ids}"
        inside = sorted({w.road_id for w in ws})
        return xy(ws), f"junction {nums[0]}: {len(inside)} short roads with a driving lane inside, {inside}"
    if kind == "segment" and len(nums) == 1:
        topo = M["topo"]
        if not 0 <= nums[0] < len(topo):
            return None, f"segments are numbered 0 to {len(topo) - 1}"
        a, b = topo[nums[0]]
        ws = [a] + a.next_until_lane_end(2.0)
        pts = np.vstack([xy(ws), xy([b])])
        length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        kind_ = "through junction %d" % a.junction_id if a.is_junction else "between intersections"
        return pts, (f"segment {nums[0]} of {len(topo)}: road {a.road_id} lane {a.lane_id}, {kind_}, "
                     f"about {length:.1f} m")
    if kind == "waypoint" and len(nums) == 1:
        if not 0 <= nums[0] < len(wps):
            return None, f"waypoints are numbered 0 to {len(wps) - 1}"
        w = wps[nums[0]]
        return xy([w]), (f"waypoint {nums[0]}: road {w.road_id}, lane {w.lane_id}, s = {w.s:.1f} m"
                         + (f", junction {w.junction_id}" if w.is_junction else ""))
    return None, f"cannot read: {query}  (try road 25, lane 25 -1, junction 332, segment 12, waypoint 100)"


def part_map(fig, g, r, s):
    """Town01.xodr read offline, in a window you can zoom (mouse wheel, or the buttons) and
    search (road, lane, junction, segment, waypoint). Click a waypoint to see its road."""
    from matplotlib.widgets import Button, TextBox
    M = load_map()
    roads, wps, text = M["roads"], M["wps"], M["text"]
    root = M["root"]
    in_j = [d for d in roads.values() if d.get("junction") != "-1"]
    drive_j = [d for d in in_j if any(l.get("type") == "driving" for l in d.iter("lane"))]
    lights = [x for x in root.iter("signal") if x.get("type") == "1000001"]
    print(f"Town01.xodr: {len(text.splitlines())} lines; {len(roads)} roads, {len(in_j)} of them inside the "
          f"{len(root.findall('junction'))} junctions ({len(drive_j)} with a driving lane); "
          f"{len(lights)} traffic lights")
    print(f"carla.Map: {len(wps)} waypoints at 2 m, {len(M['topo'])} topology segments")
    xy = np.array([[w.transform.location.x, -w.transform.location.y] for w in wps])
    jn = np.array([w.is_junction for w in wps])
    ax = fig.add_axes([0.08, 0.2, 0.9, 0.74])
    ax.scatter(*xy[~jn].T, s=3, color=INK, zorder=2, label="a waypoint on a road")
    ax.scatter(*xy[jn].T, s=3, color=EXPANDED, zorder=2, label="a waypoint inside an intersection")
    ax.set_title(f"Town01.xodr read offline: {len(roads)} roads, {len(wps)} waypoints at 2 m", fontsize=12)
    ax.legend(loc="upper right", markerscale=4, fontsize=8, framealpha=0.9)
    axes_map(ax)
    home = (ax.get_xlim(), ax.get_ylim())
    found = ax.scatter([], [], s=28, color=FOUND, zorder=5)
    typed = ax.scatter([], [], s=80, marker="x", color=INK, zorder=6)
    info = fig.text(0.08, 0.012, "click a waypoint, or search below", fontsize=9, va="bottom", wrap=True)

    def zoom(f, cx=None, cy=None):
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        cx = 0.5 * (x0 + x1) if cx is None else cx
        cy = 0.5 * (y0 + y1) if cy is None else cy
        ax.set_xlim(cx - (cx - x0) * f, cx + (x1 - cx) * f)
        ax.set_ylim(cy - (cy - y0) * f, cy + (y1 - cy) * f)
        fig.canvas.draw_idle()

    def on_scroll(ev):
        if ev.inaxes is ax:
            zoom(0.8 if ev.button == "up" else 1.25, ev.xdata, ev.ydata)

    def show(pts, msg):
        info.set_text(msg)
        if pts is None:
            found.set_offsets(np.empty((0, 2)))
        else:
            found.set_offsets(pts)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            pad = max(15.0, 0.25 * float(np.max(hi - lo)))
            c_ = 0.5 * (lo + hi)
            half = 0.5 * float(np.max(hi - lo)) + pad
            ax.set_xlim(c_[0] - half, c_[0] + half); ax.set_ylim(c_[1] - half, c_[1] + half)
        fig.canvas.draw_idle()

    def on_search(q):
        typed.set_offsets(np.empty((0, 2)))
        show(*map_search(q, M))

    def on_xy(q):
        try:
            x_, y_ = [float(v) for v in q.replace(",", " ").split()]
        except ValueError:
            show(None, "type x and y in meters, e.g. 133.6, -129.4")
            return
        typed.set_offsets([[x_, y_]])
        show(*map_point(x_, y_, M))

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or fig.canvas.toolbar is not None and \
                getattr(fig.canvas.toolbar, "mode", ""):
            return
        k = int(np.argmin(np.hypot(xy[:, 0] - ev.xdata, xy[:, 1] - ev.ydata)))
        w = wps[k]
        rid = str(w.road_id)
        found.set_offsets(xy[[k]])
        info.set_text(f"waypoint {k}: road {w.road_id}, lane {w.lane_id}, s = {w.s:.1f} m along the road, "
                      f"lane width {w.lane_width:.1f} m" + (f", junction {w.junction_id}" if w.is_junction else "")
                      + f"; Town01.xodr line {M['line_of'].get(rid, '?')}")
        fig.canvas.draw_idle()

    box = TextBox(fig.add_axes([0.16, 0.075, 0.24, 0.05]), "Search ", initial="road 25")
    box.on_submit(on_search)
    xybox = TextBox(fig.add_axes([0.47, 0.075, 0.13, 0.05]), "x, y (m) ", initial="133.6, -129.4")
    xybox.on_submit(on_xy)
    b_in = Button(fig.add_axes([0.62, 0.075, 0.1, 0.05]), "Zoom in")
    b_out = Button(fig.add_axes([0.73, 0.075, 0.1, 0.05]), "Zoom out")
    b_all = Button(fig.add_axes([0.84, 0.075, 0.12, 0.05]), "Whole map")
    b_in.on_clicked(lambda _: zoom(0.67))
    b_out.on_clicked(lambda _: zoom(1.5))
    b_all.on_clicked(lambda _: (ax.set_xlim(*home[0]), ax.set_ylim(*home[1]), fig.canvas.draw_idle()))
    fig.text(0.16, 0.135, "search: road 25 | lane 25 -1 | junction 332 | segment 12 | waypoint 100;"
             "  x, y: nearest lane point;  wheel: zoom", fontsize=9, color="#555555")
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig._l7_widgets = [box, xybox, b_in, b_out, b_all]          # keep the widgets alive


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
