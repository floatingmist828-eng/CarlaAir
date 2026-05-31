from __future__ import annotations

import numpy as np

from carlaair_active_world.ego_driver import EgoDriveConfig, RouteFollowingDriver


class _FakeLocation:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeRotation:
    def __init__(self, yaw: float = 0.0):
        self.yaw = yaw


class _FakeTransform:
    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self.location = _FakeLocation(x, y, 0.0)
        self.rotation = _FakeRotation(yaw)


class _FakeWaypoint:
    def __init__(self, yaw: float = 0.0, x: float = 8.0, y: float = 0.0, is_junction: bool = False):
        self.transform = _FakeTransform(yaw=yaw)
        self.transform.location = _FakeLocation(x, y, 0.0)
        self.is_junction = is_junction

    def next(self, lookahead: float):
        return [self]


class _FakeMap:
    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        return _FakeWaypoint(yaw=0.0)


class _FakeActorList:
    def __init__(self, actors):
        self._actors = list(actors)

    def filter(self, pattern: str):
        if pattern == "vehicle.*":
            return list(self._actors)
        return []


class _FakeWorld:
    def get_map(self):
        return _FakeMap()

    def get_actors(self):
        return _FakeActorList([])


class _FakeVector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeVehicle:
    def __init__(self, actor_id: int = 1):
        self._transform = _FakeTransform()
        self._velocity = _FakeVector()
        self._actor_id = actor_id

    def get_location(self):
        return self._transform.location

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    @property
    def bounding_box(self):
        class _BoundingBox:
            extent = _FakeVector(2.5, 1.0, 1.0)

        return _BoundingBox()

    @property
    def id(self):
        return self._actor_id

    def is_at_traffic_light(self):
        return False


def test_route_follow_drives_forward_on_clear_straight_road():
    driver = RouteFollowingDriver(EgoDriveConfig())
    control = driver.predict(_FakeVehicle(), _FakeWorld())

    assert control.throttle > 0
    assert control.brake == 0.0
    assert abs(control.steer) < 0.2


def test_route_follow_brakes_for_vehicle_ahead():
    driver = RouteFollowingDriver(EgoDriveConfig())
    front_vehicle = _FakeVehicle(actor_id=2)
    front_vehicle._transform = _FakeTransform(x=8.0, y=0.0, yaw=0.0)

    class _TrafficWorld(_FakeWorld):
        def get_actors(self):
            return _FakeActorList([front_vehicle])

    control = driver.predict(_FakeVehicle(actor_id=1), _TrafficWorld())

    assert control.brake == 1.0
    assert control.throttle == 0.0


def test_route_follow_steers_right_for_rightward_target():
    driver = RouteFollowingDriver(EgoDriveConfig())

    class _RightMap(_FakeMap):
        def get_waypoint(self, location, project_to_road=True, lane_type=None):
            return _FakeWaypoint(yaw=0.0, x=8.0, y=3.0)

    class _RightWorld(_FakeWorld):
        def get_map(self):
            return _RightMap()

    control = driver.predict(_FakeVehicle(actor_id=1), _RightWorld())

    assert control.steer > 0
