import pytest
import record_drive as rd


def test_preflight_refuses_user_carla():
    with pytest.raises(RuntimeError, match="carla"):
        rd.preflight(["carla", "postgres"])


def test_preflight_allows_only_our_container():
    rd.preflight(["carla-test", "postgres"])
    rd.preflight([])
