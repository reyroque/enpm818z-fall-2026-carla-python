"""Optional: --part search in the Rerun viewer instead of a matplotlib window.

It replays the same saved search orders as `python3 l7_handson.py --part search`: at step k,
each view shows the first k nodes Dijkstra and A* took, and the route once the search
reaches G. Drag the time slider at the bottom of the viewer, or press play. Scroll to
zoom and drag to pan in each view.

  pip install rerun-sdk                    # once; the viewer comes with it
  python3 rerun_search.py                  # opens the viewer
  python3 rerun_search.py --save x.rrd     # writes a recording; open it with: rerun x.rrd

Rerun's 2D views point y down, so the map's y (north) is drawn flipped: north stays up,
and the axis shows 130 where the map frame has y = -130 m.
"""
import argparse
import sys

import numpy as np

import l7common as c

LANE, TAKEN, ROUTE, INK = (190, 196, 204), (232, 163, 61), (46, 109, 164), (30, 41, 57)


def to_view(xy):
    """Map frame (x east, y north) to Rerun's 2D view, whose y axis points down."""
    return np.column_stack([xy[:, 0], -xy[:, 1]])


def main(argv=None):
    """Read the options, load the graph and the saved results, and log the replay."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", help="write a .rrd recording instead of opening the viewer")
    a = ap.parse_args(argv)
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError:
        sys.exit("rerun_search.py needs Rerun: pip install rerun-sdk")

    g = c.load_graph()
    r, s = c.load_results()
    xy = to_view(g["node_xy"])
    orders = {"dijkstra": ("Dijkstra", r["order_dijkstra"]), "a_star": ("A*", r["order_a_star"])}

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*[rrb.Spatial2DView(origin=f"/{k}", name=name) for k, (name, _) in orders.items()]),
            rrb.TimeSeriesView(origin="/nodes_taken", name="Nodes taken so far"),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
    rr.init("l7_search", spawn=a.save is None, default_blueprint=blueprint)
    if a.save:
        rr.save(a.save, default_blueprint=blueprint)

    lanes = np.stack([xy[g["edge_from"]], xy[g["edge_to"]]], axis=1)   # every edge: every lane
    sg = xy[[s["trip"]["start_node"], s["trip"]["goal_node"]]]
    route = xy[r["route_distance"]]
    for k in orders:                                   # drawn once, shown at every step
        rr.log(f"{k}/lanes", rr.LineStrips2D(lanes, colors=LANE, radii=0.4), static=True)
        rr.log(f"{k}/start_goal", rr.Points2D(sg, colors=INK, radii=3.0, labels=["S", "G"]), static=True)

    n = max(len(o) for _, o in orders.values())
    for step in range(1, n + 1):
        rr.set_time("step", sequence=step)
        for k, (name, order) in orders.items():
            if step <= len(order):                     # after its last step, a view keeps that frame
                rr.log(f"{k}/taken", rr.Points2D(xy[order[:step]], colors=TAKEN, radii=1.2))
                rr.log(f"nodes_taken/{k}", rr.Scalars(step))
            if step == len(order):                     # the route appears when the search reaches G
                rr.log(f"{k}/route", rr.LineStrips2D([route], colors=ROUTE, radii=1.0))


if __name__ == "__main__":
    main()
