from __future__ import annotations

import math

import carla

from carlaair_active_world import traffic
from carlaair_active_world.traffic import _crossing_source_target


def test_crossing_source_target_applies_longitudinal_offset():
    spawn_tf = carla.Transform(
        carla.Location(x=-45.0, y=115.0, z=0.6),
        carla.Rotation(yaw=-90.0),
    )

    source, target = _crossing_source_target(spawn_tf, side_offset_m=7.0, longitudinal_offset_m=-2.0)

    assert math.isclose(source.x, -38.0, abs_tol=1e-3)
    assert math.isclose(target.x, -52.0, abs_tol=1e-3)
    assert math.isclose(source.y, 117.0, abs_tol=1e-3)
    assert math.isclose(target.y, 117.0, abs_tol=1e-3)


def test_spawn_traffic_vehicles_sets_route_commands(monkeypatch):
    class _Blueprint:
        def __init__(self):
            self.attributes = {}

        def has_attribute(self, name):
            return name in {"role_name", "color"}

        def set_attribute(self, name, value):
            self.attributes[name] = value

    class _BlueprintLibrary:
        def find(self, _blueprint_id):
            return _Blueprint()

        def filter(self, _pattern):
            return [_Blueprint()]

    class _Actor:
        def __init__(self):
            self.id = 10

        def set_simulate_physics(self, _enabled):
            pass

        def set_autopilot(self, *_args):
            pass

    class _Map:
        def get_spawn_points(self):
            return [carla.Transform(carla.Location(x=1.0), carla.Rotation())]

    class _World:
        def __init__(self):
            self.actor = _Actor()

        def get_blueprint_library(self):
            return _BlueprintLibrary()

        def get_map(self):
            return _Map()

        def try_spawn_actor(self, _blueprint, _transform):
            return self.actor

        def get_settings(self):
            return type("Settings", (), {"synchronous_mode": False})()

    class _TrafficManager:
        def __init__(self):
            self.routes = []

        def global_percentage_speed_difference(self, _percentage):
            pass

        def set_synchronous_mode(self, _enabled):
            pass

        def set_route(self, actor, route):
            self.routes.append((actor.id, route))

    class _Client:
        def __init__(self):
            self.tm = _TrafficManager()

        def get_trafficmanager(self, _port):
            return self.tm

    monkeypatch.setattr(traffic, "configure_autopilot", lambda *_args, **_kwargs: None)
    client = _Client()
    world = _World()

    spawned = traffic.spawn_traffic_vehicles(
        client,
        world,
        count=1,
        route_commands=["Right", "Straight"],
    )

    assert [item.actor for item in spawned] == [world.actor]
    assert client.tm.routes == [(10, ["Right", "Straight"])]


def test_spawn_static_obstacle_vehicles_offsets_and_holds_position():
    class _Blueprint:
        def __init__(self):
            self.attributes = {}

        def has_attribute(self, name):
            return name in {"role_name", "color"}

        def set_attribute(self, name, value):
            self.attributes[name] = value

    class _BlueprintLibrary:
        def find(self, _blueprint_id):
            return _Blueprint()

        def filter(self, _pattern):
            return [_Blueprint()]

    class _Actor:
        def __init__(self, actor_id):
            self.id = actor_id
            self.physics = None
            self.autopilot = None
            self.control = None

        def set_simulate_physics(self, enabled):
            self.physics = enabled

        def set_autopilot(self, enabled, *_args):
            self.autopilot = enabled

        def apply_control(self, control):
            self.control = control

    class _World:
        def __init__(self):
            self.transforms = []
            self.actors = []

        def get_blueprint_library(self):
            return _BlueprintLibrary()

        def try_spawn_actor(self, _blueprint, transform):
            actor = _Actor(len(self.actors) + 10)
            self.transforms.append(transform)
            self.actors.append(actor)
            return actor

    world = _World()
    anchor = carla.Transform(carla.Location(x=-41.6, y=58.0, z=0.6), carla.Rotation(yaw=-90.0))

    spawned = traffic.spawn_static_obstacle_vehicles(
        world,
        count=2,
        anchor_transform=anchor,
        forward_offsets_m=[0.0, 4.2],
        lateral_offsets_m=[0.0, 0.8],
        yaw_offsets_deg=[25.0, -35.0],
    )

    assert [item.actor for item in spawned] == world.actors
    assert world.transforms[0].location.x == -41.6
    assert world.transforms[0].location.y == 58.0
    assert world.transforms[0].rotation.yaw == -65.0
    assert round(world.transforms[1].location.x, 2) == -40.8
    assert round(world.transforms[1].location.y, 2) == 53.8
    assert world.transforms[1].rotation.yaw == -125.0
    assert all(actor.physics is False for actor in world.actors)
    assert all(actor.autopilot is False for actor in world.actors)
    assert all(getattr(actor.control, "hand_brake", False) is True for actor in world.actors)
