"""Tests for the L12 simulation and hands-on. Run: python3 -m pytest tests"""
import json
from pathlib import Path

import numpy as np
import pytest

import l12_handson as h
import l12_sim as s

HERE = Path(__file__).resolve().parents[1]


def test_street_matches_town01_json():
    town = json.loads((HERE / "town01.json").read_text())
    assert list(town["driving_lane_widths_m"]) == [str(s.LANE_W)]
    assert ["Shoulder", s.SHOULDER] in [list(x) for x in town["beside_the_right_lane"]]
    assert "25 mph" in town["speed_limits"]
    assert abs(s.SPEED_LIMIT - 11.176) < 1e-3


def test_far_run_stops_short_of_the_line():
    q = s.simulate(10.0, 50.0, 1.4)
    assert not q["collided"] and q["stop_gap"] > 0


def test_stopping_distance_by_hand():
    # 10 m/s: 0.5 s of latency is 5 m, braking at 6 m/s^2 is 100 / 12 = 8.33 m
    assert s.stopping_distance(10.0) == pytest.approx(5.0 + 100 / 12)


def test_upper_edge_of_the_band_matches_the_formula():
    # the pedestrian enters the hazard zone at (6.3 - 2.5) / 1.4 s; the AV must stop
    # 0.25 m (the pedestrian's radius) short of the line
    t_h = (s.Y_FAR_CURB - 2.5) / 1.4
    edge = 10 * t_h + s.stopping_distance(10.0) + s.PED_RADIUS
    d = np.round(np.arange(30, 45, 0.01), 2)
    f = s.simulate(10.0, d, 1.4)["collided"]
    assert abs(d[np.flatnonzero(f)[-1]] - edge) < 0.05


def test_passing_ahead_is_a_pass():
    q = s.simulate(10.0, 20.0, 1.4)
    assert not q["collided"] and not q["braked"]


def test_zero_failure_bound():
    assert s.zero_failure_bound(3) == pytest.approx(1 - 0.05 ** (1 / 3))
    assert 100 * s.zero_failure_bound(100) == pytest.approx(2.95, abs=0.01)


def test_rand_miles():
    assert s.miles_needed(1.09e-8) / 1e6 == pytest.approx(274.8, abs=0.1)


def test_driving_score_versions():
    assert s.driving_score(80, {"vehicle": 1, "red_light": 1})[0] == pytest.approx(80 / 2.1)
    assert s.driving_score(80, {"vehicle": 1, "red_light": 1}, "2.0")[0] == pytest.approx(80 * 0.6 * 0.7)


def test_random_runs_are_inside_the_logical_scenario():
    v, d, p = s.random_runs(500, 3, vped=True)
    for x, key in ((v, "v0"), (d, "d0"), (p, "vped")):
        lo, hi = s.LOGICAL[key]
        assert x.min() >= lo and x.max() <= hi


@pytest.mark.parametrize("part", ["run", "sweep", "sample", "score"])
def test_each_part_writes_a_snapshot(part, tmp_path):
    out = tmp_path / f"{part}.png"
    h.main(["--part", part, "--snapshot", str(out)])
    assert out.exists() and out.stat().st_size > 10000


def test_run_html(tmp_path):
    out = tmp_path / "run.html"
    h.main(["--part", "run", "--html", str(out), "--d0", "50"])
    assert out.stat().st_size > 10000


def test_no_gp4_solution_code():
    banned = ("heapq", "astar", "dijkstra(", "pure_pursuit", "BehaviorFSM", "LANE_FOLLOW",
              "EMERGENCY_STOP", "OBSTACLE_AVOIDANCE", "rclpy")
    for f in HERE.glob("*.py"):
        text = f.read_text()
        for b in banned:
            assert b not in text, f"{f.name} contains {b}"
