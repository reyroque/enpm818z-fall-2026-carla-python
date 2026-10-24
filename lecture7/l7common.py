"""L7 shared helpers: load the Town01 graph and the saved results, and turn a route
into a reference path.

Nothing here searches a graph. The routes and the search orders were computed in
advance and saved in l7_results/; this file only reads and draws them. You write the
search yourself in GP4.

Frames, as in L6: the map frame is in meters, x east and y north (CARLA's y is
flipped). Headings are in degrees, counterclockwise from +x (east).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRAPH = HERE / "l7_town01_graph.npz"
RESULTS = HERE / "l7_results"


def load_graph(path=GRAPH):
    """Town01's waypoint graph as a dict of arrays. The ones the scripts use:
    node_xy (N x 2, m), node_yaw_deg (heading of each node), node_road and node_lane (where
    each node sits in Town01.xodr), node_junction (True inside an intersection), and the
    edges as three parallel arrays: edge_from[i] -> edge_to[i], edge_len[i] in meters."""
    g = np.load(path)
    return {k: g[k] for k in g.files}     # copy out of the .npz so the file can close


def load_results(folder=RESULTS):
    """routes.npz (arrays) and summary.json (numbers and the intersection graph's steps)."""
    folder = Path(folder)
    r = np.load(folder / "routes.npz")
    with open(folder / "summary.json") as f:
        s = json.load(f)
    return {k: r[k] for k in r.files}, s


def wrap_deg(a):
    """An angle in degrees, brought into [-180, 180): 350 becomes -10. Needed before
    comparing headings, or a turn from 179 to -179 degrees would look like 358 degrees."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def edge_lookup(g):
    """{(from, to): edge index}, to read an edge's length along a saved route."""
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(g["edge_from"], g["edge_to"]))}


def check_route(g, route):
    """True when every step of the route is an edge of the graph (a connected path)."""
    lut = edge_lookup(g)
    return all((int(a), int(b)) in lut for a, b in zip(route[:-1], route[1:]))


def route_length(g, route):
    """The route's length in meters: the sum of its edges' lengths, read from the graph."""
    lut = edge_lookup(g)
    return float(sum(g["edge_len"][lut[(int(a), int(b))]] for a, b in zip(route[:-1], route[1:])))


def reference_path(g, route):
    """A route (node indices) as a reference path: arc length s, x, y, heading, curvature.

    Curvature at each node: the change of heading over the next edge, divided by its
    length (1/m; left turns positive). The last node repeats the one before it.
    """
    xy = g["node_xy"][route]
    yaw = g["node_yaw_deg"][route]
    ds = np.hypot(*np.diff(xy, axis=0).T)              # length of each step between nodes
    s = np.concatenate([[0.0], np.cumsum(ds)])         # distance along the route, 0 at the start
    # curvature = change of heading (rad) per meter: 0 on a straight, about 1/r in a turn
    kappa = np.radians(wrap_deg(np.diff(yaw))) / ds
    kappa = np.concatenate([kappa, kappa[-1:]])        # one value per node, like s, x and y
    return {"s": s, "x": xy[:, 0], "y": xy[:, 1], "yaw_deg": yaw, "kappa": kappa}


