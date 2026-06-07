from __future__ import annotations

import numpy as np
import carla
import pytest

from scripts.collect_tcp_lite_dataset import (
    control_from_diagnostics,
    driver_decision_observation,
    future_route_trajectory,
    local_xy,
    parse_distances,
    should_keep_tcp_lite_sample,
)


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


def test_driver_decision_observation_prefers_policy_input_frame():
    policy_rgb = np.full((2, 3, 3), 7, dtype=np.uint8)
    fresh_rgb = np.full((2, 3, 3), 19, dtype=np.uint8)

    class _Rig:
        def snapshot(self):
            return {"rgb": fresh_rgb}

    class _Driver:
        last_observation = {"rgb": policy_rgb, "speed_mps": 2.5, "navigation_command": "left"}
        sensor_rig = _Rig()

    observation = driver_decision_observation(_Driver())

    assert np.array_equal(observation["rgb"], policy_rgb)
    assert observation["speed_mps"] == 2.5
    assert observation["navigation_command"] == "left"


def test_should_keep_tcp_lite_sample_rejects_recovery_and_backward_trajectory():
    assert should_keep_tcp_lite_sample(
        {"reverse": False, "recovery_active": False},
        [[2.0, 0.0], [4.0, 0.0]],
    )
    assert not should_keep_tcp_lite_sample(
        {"reverse": True, "recovery_active": False},
        [[2.0, 0.0], [4.0, 0.0]],
    )
    assert not should_keep_tcp_lite_sample(
        {"reverse": False, "recovery_active": True},
        [[2.0, 0.0], [4.0, 0.0]],
    )
    assert not should_keep_tcp_lite_sample(
        {"reverse": False, "recovery_active": False},
        [[2.0, 0.0], [-1.0, 0.0]],
    )


def test_future_route_trajectory_prefers_heading_continuity_at_branch():
    class _Rotation:
        def __init__(self, yaw):
            self.yaw = yaw

    class _Transform:
        def __init__(self, x, y, yaw):
            self.location = carla.Location(x=x, y=y, z=0.0)
            self.rotation = _Rotation(yaw)

    class _Waypoint:
        def __init__(self, name, x, y, yaw):
            self.name = name
            self.transform = _Transform(x, y, yaw)

        def next(self, distance):
            if self.name == "root":
                return [
                    _Waypoint("sharp_turn", 2.0, 0.2, 80.0),
                    _Waypoint("straight", 2.0, 0.6, 0.0),
                ]
            return [self]

    class _Map:
        def get_waypoint(self, location, project_to_road=True, lane_type=None):
            return _Waypoint("root", 0.0, 0.0, 0.0)

    class _World:
        def get_map(self):
            return _Map()

    class _Vehicle:
        def get_location(self):
            return carla.Location(x=0.0, y=0.0, z=0.0)

        def get_transform(self):
            return carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.0),
                carla.Rotation(yaw=0.0),
            )

    trajectory = future_route_trajectory(_World(), _Vehicle(), [2.0])

    assert trajectory == [[2.0, 0.6]]


def test_future_route_trajectory_selects_commanded_turn_branch():
    class _Rotation:
        def __init__(self, yaw):
            self.yaw = yaw

    class _Transform:
        def __init__(self, x, y, yaw):
            self.location = carla.Location(x=x, y=y, z=0.0)
            self.rotation = _Rotation(yaw)

    class _Waypoint:
        def __init__(self, name, x, y, yaw):
            self.name = name
            self.transform = _Transform(x, y, yaw)

        def next(self, distance):
            if self.name == "root":
                return [
                    _Waypoint("left", 2.0, -1.0, -70.0),
                    _Waypoint("straight", 2.0, 0.0, 0.0),
                    _Waypoint("right", 2.0, 1.0, 70.0),
                ]
            return [self]

    class _Map:
        def get_waypoint(self, location, project_to_road=True, lane_type=None):
            return _Waypoint("root", 0.0, 0.0, 0.0)

    class _World:
        def get_map(self):
            return _Map()

    class _Vehicle:
        def get_location(self):
            return carla.Location(x=0.0, y=0.0, z=0.0)

        def get_transform(self):
            return carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.0),
                carla.Rotation(yaw=0.0),
            )

    assert future_route_trajectory(_World(), _Vehicle(), [2.0], command="left") == [[2.0, -1.0]]
    assert future_route_trajectory(_World(), _Vehicle(), [2.0], command="right") == [[2.0, 1.0]]
