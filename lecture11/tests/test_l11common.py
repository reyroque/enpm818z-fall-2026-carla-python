import numpy as np
import pytest

import l11common as c


@pytest.fixture(scope="module")
def road():
    return c.load_road()


@pytest.fixture(autouse=True)
def chosen_values():
    """Every test starts from the chosen values and leaves them unchanged."""
    keep = dict(c.CHOSEN)
    yield
    c.CHOSEN.clear()
    c.CHOSEN.update(keep)


def test_road_is_one_closed_lap_of_town01(road):
    assert 1440 < road.length < 1445
    assert road.lane_width == pytest.approx(4.0)
    assert c.room(road) == pytest.approx(1.0)
    dx, dy = road.x[-1] - road.x[0], road.y[-1] - road.y[0]
    assert np.hypot(dx, dy) < 1.0                          # it closes
    turned = np.degrees(road.heading[-1] - road.heading[0])
    assert turned == pytest.approx(360, abs=2)            # four left turns


def test_expert_finishes_every_test_lap(road):
    _, score = c.evaluate(road, "expert")
    assert score["finished"] == score["drives"] == 20


def test_bc_recovers_the_expert_without_detector_noise(road):
    # With a perfect detector, least squares sees the expert's rule exactly (small-angle
    # curvature term aside): the offset and heading weights come back.
    c.CHOSEN["detector_sigma_offset_m"] = 0.0
    c.CHOSEN["detector_sigma_heading_deg"] = 0.0
    pol, _ = c.behavior_cloning(road, np.random.default_rng(1))
    we = c.expert_weights()
    assert pol.w[0] == pytest.approx(we[0], rel=0.05)
    assert pol.w[1] == pytest.approx(we[1], rel=0.05)


def test_the_lecture_numbers(road):
    # The slides quote these (measured.json in the slides repo); seed 11, test laps 100 to 119.
    r = c.experiment(road)
    bc, last = r["rows"][0], r["rows"][-1]
    assert bc["finished"] == 1 and last["finished"] == 19
    assert np.degrees(-bc["weights"][0]) == pytest.approx(0.38, abs=0.01)
    assert np.degrees(-last["weights"][0]) == pytest.approx(1.20, abs=0.01)
    assert bc["open_loop_deg"] == pytest.approx(0.35, abs=0.01)
    assert last["open_loop_deg"] == pytest.approx(0.43, abs=0.01)
    assert bc["open_loop_deg"] < last["open_loop_deg"]      # better open loop, worse closed loop


def test_dagger_labels_are_the_experts(road):
    pols, drives = c.dagger(road, np.random.default_rng(3), iterations=2)
    d = drives[1]                                          # driven by the BC clone
    assert np.allclose(d.steer_expert, c.expert_steer(d.e, d.psi, d.kappa_ahead))
    assert not np.allclose(d.steer_cmd, d.steer_expert)   # the clone steered, not the expert
