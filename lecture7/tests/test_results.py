"""The saved results agree with the graph and with each other."""
import numpy as np
import pytest

import l7common as c


@pytest.fixture(scope="module")
def data():
    g = c.load_graph()
    r, s = c.load_results()
    return g, r, s


@pytest.mark.parametrize("key", ["route_distance", "route_time", "route_left_penalty", "route_a_star",
                                 "replan_route", "route_time_wrong_h"])
def test_routes_are_connected_paths(data, key):
    g, r, _ = data
    assert c.check_route(g, r[key])


def test_routes_start_and_end_at_the_trip(data):
    g, r, s = data
    for key in ("route_distance", "route_time", "route_left_penalty"):
        assert r[key][0] == s["trip"]["start_node"] and r[key][-1] == s["trip"]["goal_node"]
    assert r["replan_route"][0] == s["replan"]["from_node"]


def test_lengths_match_summary(data):
    g, r, s = data
    t = s["trip"]["routes"]
    assert abs(c.route_length(g, r["route_distance"]) - t["distance"]["length_m"]) < 0.1
    assert abs(c.route_length(g, r["route_time"]) - t["time"]["length_m"]) < 0.1
    assert abs(c.route_length(g, r["replan_route"]) - s["replan"]["new_m"]) < 0.1


def test_dijkstra_and_a_star_agree(data):
    g, r, s = data
    sc = s["scale"]["distance"]
    assert np.array_equal(r["route_a_star"], r["route_distance"])
    assert len(r["order_dijkstra"]) == sc["dijkstra_expanded"] and len(r["order_a_star"]) == sc["a_star_expanded"]
    assert sc["a_star_expanded"] < sc["dijkstra_expanded"]
    # each search ends when it takes the goal, and takes no node twice
    for k in ("order_dijkstra", "order_a_star"):
        assert r[k][-1] == s["trip"]["goal_node"] and len(set(r[k].tolist())) == len(r[k])


def test_no_route_beats_the_distance_route_on_length(data):
    g, r, s = data
    L = c.route_length(g, r["route_distance"])
    for k in ("route_time", "route_left_penalty", "route_time_wrong_h"):
        assert c.route_length(g, r[k]) >= L - 1e-6


def test_replanned_route_avoids_the_obstacle(data):
    g, r, s = data
    xy = g["node_xy"][r["replan_route"]]
    d = np.hypot(*(xy - np.asarray(s["chosen"]["obstacle_xy"])).T)
    assert d.min() > s["chosen"]["clearance_m"]


def test_toy_graph_steps(data):
    _, _, s = data
    t = s["toy"]
    assert t["admissible"]
    assert t["dijkstra"]["cost"] == t["a_star"]["cost"]
    assert len(t["a_star"]["expanded"]) < len(t["dijkstra"]["expanded"])
    assert t["wrong"]["cost"] > t["a_star"]["cost"] and t["wrong"]["h"] > t["wrong"]["true"]
    # the route's cost is the sum of its edges
    E = {}
    for a, b, L in t["edges"]:
        E[(a, b)] = E[(b, a)] = L
    for k in ("dijkstra", "a_star", "wrong"):
        p = t[k]["path"]
        assert sum(E[(a, b)] for a, b in zip(p[:-1], p[1:])) == t[k]["cost"]
