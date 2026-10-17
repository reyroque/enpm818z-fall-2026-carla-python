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
    g = np.load(path)
    return {k: g[k] for k in g.files}


def load_results(folder=RESULTS):
    """routes.npz (arrays) and summary.json (numbers and the toy graph's steps)."""
    folder = Path(folder)
    r = np.load(folder / "routes.npz")
    with open(folder / "summary.json") as f:
        s = json.load(f)
    return {k: r[k] for k in r.files}, s


def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def edge_lookup(g):
    """{(from, to): edge index}, to read an edge's length along a saved route."""
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(g["edge_from"], g["edge_to"]))}


def check_route(g, route):
    """True when every step of the route is an edge of the graph (a connected path)."""
    lut = edge_lookup(g)
    return all((int(a), int(b)) in lut for a, b in zip(route[:-1], route[1:]))


def route_length(g, route):
    lut = edge_lookup(g)
    return float(sum(g["edge_len"][lut[(int(a), int(b))]] for a, b in zip(route[:-1], route[1:])))


def reference_path(g, route):
    """A route (node indices) as a reference path: arc length s, x, y, heading, curvature.

    Curvature at each node: the change of heading over the next edge, divided by its
    length (1/m; left turns positive). The last node repeats the one before it.
    """
    xy = g["node_xy"][route]
    yaw = g["node_yaw_deg"][route]
    ds = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    kappa = np.radians(wrap_deg(np.diff(yaw))) / ds
    kappa = np.concatenate([kappa, kappa[-1:]])
    return {"s": s, "x": xy[:, 0], "y": xy[:, 1], "yaw_deg": yaw, "kappa": kappa}


