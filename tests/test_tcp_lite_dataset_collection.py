from __future__ import annotations

import carla
import pytest

from scripts.collect_tcp_lite_dataset import control_from_diagnostics, local_xy, parse_distances


def test_parse_distances_requires_at_least_one_value():
    assert parse_distances("2, 4,6") == [2.0, 4.0, 6.0]
    with pytest.raises(ValueError):
        parse_distances("")


def test_local_xy_uses_vehicle_heading():
    transform = carla.Transform(carla.Location(x=10.0, y=20.0), carla.Rotation(yaw=90.0))
    local = local_xy(transform, carla.Location(x=10.0, y=24.0))

    assert local[0] == pytest.approx(4.0, abs=1e-6)
    assert local[1] == pytest.approx(0.0, abs=1e-6)


def test_control_from_diagnostics_requires_control_fields():
    assert control_from_diagnostics({"steer": 0.1, "throttle": 0.2, "brake": 0.0}) == {
        "steer": 0.1,
        "throttle": 0.2,
        "brake": 0.0,
    }
    assert control_from_diagnostics({"steer": 0.1}) is None
    assert control_from_diagnostics({"steer": "bad", "throttle": 0.2, "brake": 0.0}) is None
