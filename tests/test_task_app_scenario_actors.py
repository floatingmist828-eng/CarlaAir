from __future__ import annotations

from pathlib import Path

import carla

from carlaair_active_world import task_app
from carlaair_active_world.scenario import ScenarioConfig


class _Actor:
    def __init__(self, actor_id: int, type_id: str = "vehicle.test") -> None:
        self.id = actor_id
        self.type_id = type_id
        self.destroyed = False
        self.autopilot = None
        self.control = None

    def destroy(self) -> None:
        self.destroyed = True

    def set_autopilot(self, value) -> None:
        self.autopilot = value

    def get_location(self):
        return carla.Location(x=0.0, y=0.0, z=0.0)

    def get_transform(self):
        return carla.Transform(carla.Location(), carla.Rotation())

    def apply_control(self, control) -> None:
        self.control = control


class _Spawned:
    def __init__(self, actor, controller=None, target=None, speed_mps=0.0) -> None:
        self.actor = actor
        self.controller = controller
        self.target = target
        self.speed_mps = speed_mps


class _World:
    def get_settings(self):
        return type("Settings", (), {"synchronous_mode": False})()


def test_task_app_spawns_configured_traffic_and_walkers(monkeypatch):
    calls = {"vehicles": None, "walkers": None}
    traffic = _Actor(20)
    walker = _Actor(30, type_id="walker.pedestrian.test")
    controller = _Actor(31, type_id="controller.ai.walker")

    monkeypatch.setattr(task_app, "cleanup_actors_by_role", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_app, "cleanup_old_vehicles", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_app, "spawn_ego_vehicle", lambda *_args, **_kwargs: _Actor(10))
    monkeypatch.setattr(task_app, "configure_autopilot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_app.ActiveUAVTaskApp, "start_ego_driver", lambda self: None)
    monkeypatch.setattr(task_app.ActiveUAVTaskApp, "_attach_vehicle_sensors", lambda self: None)
    monkeypatch.setattr(task_app.ActiveUAVTaskApp, "start_vehicle_viewer", lambda self: None)
    monkeypatch.setattr(
        task_app,
        "spawn_traffic_vehicles",
        lambda *_args, **kwargs: calls.__setitem__("vehicles", kwargs) or [_Spawned(traffic)],
    )
    monkeypatch.setattr(
        task_app,
        "spawn_traffic_walkers",
        lambda *_args, **kwargs: calls.__setitem__("walkers", kwargs)
        or [_Spawned(walker, controller, carla.Location(x=3.0, y=4.0, z=0.0), 1.2)],
    )

    scenario = ScenarioConfig.from_dict(
        {
            "name": "task_app_actors",
            "uav_enabled": False,
            "ego_spawn_index": 86,
            "traffic_vehicles": 4,
            "traffic_spawn_indices": [25, 24, 32, 31],
            "traffic_route_commands": ["Right", "Straight"],
            "traffic_spawn_delay_sec": 0.0,
            "traffic_speed_difference": 45.0,
            "traffic_walkers": 4,
            "walker_spawn_indices": [146, 146, 146, 146],
            "walker_spawn_delay_sec": 0.0,
            "walker_crossing_distance_m": 14.0,
            "walker_crossing_offsets_m": [-2.0, -0.7, 0.7, 2.0],
            "walker_speed_mps": 0.85,
        }
    )
    app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_task_app_actors"))
    app.client = object()
    app.world = _World()

    app.setup()

    assert calls["vehicles"]["count"] == 4
    assert calls["vehicles"]["spawn_indices"] == [25, 24, 32, 31]
    assert calls["vehicles"]["route_commands"] == ["Right", "Straight"]
    assert calls["vehicles"]["speed_difference"] == 45.0
    assert calls["walkers"]["count"] == 4
    assert calls["walkers"]["spawn_indices"] == [146, 146, 146, 146]
    assert calls["walkers"]["crossing_distance_m"] == 14.0
    assert calls["walkers"]["crossing_offsets_m"] == [-2.0, -0.7, 0.7, 2.0]
    assert calls["walkers"]["use_ai_controller"] is False
    assert calls["walkers"]["speed_mps"] == 0.85
    assert app.traffic_actors == [traffic]
    assert app.walker_actors == [walker]
    assert app.walker_controllers == [controller]


def test_task_app_drives_scripted_walkers_until_target(monkeypatch):
    walker = _Actor(30, type_id="walker.pedestrian.test")
    scenario = ScenarioConfig.from_dict({"name": "scripted_task_walker", "uav_enabled": False})
    app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_scripted_task_walker"))
    app._walker_targets = [(walker, carla.Location(x=3.0, y=4.0, z=0.0), 1.2)]

    app._drive_scripted_walkers()

    assert walker.control is not None
    assert walker.control.speed == 1.2
    assert round(walker.control.direction.x, 3) == 0.6
    assert round(walker.control.direction.y, 3) == 0.8
    assert app._walker_targets == [(walker, app._walker_targets[0][1], 1.2)]


def test_task_app_keeps_scripted_walker_moving_when_not_fully_crossed(monkeypatch):
    class _Walker(_Actor):
        def get_location(self):
            return carla.Location(x=2.7, y=3.6, z=0.0)

    walker = _Walker(30, type_id="walker.pedestrian.test")
    target = carla.Location(x=3.0, y=4.0, z=0.0)
    scenario = ScenarioConfig.from_dict({"name": "scripted_task_walker_short", "uav_enabled": False})
    app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_scripted_task_walker_short"))
    app._walker_targets = [(walker, target, 1.2)]

    app._drive_scripted_walkers()

    assert walker.control is not None
    assert walker.control.speed == 1.2
    assert app._walker_targets == [(walker, target, 1.2)]


def test_task_app_delays_configured_traffic_and_walkers(monkeypatch):
    calls = {"vehicles": 0, "walkers": 0}

    monkeypatch.setattr(
        task_app,
        "spawn_traffic_vehicles",
        lambda *_args, **_kwargs: calls.__setitem__("vehicles", calls["vehicles"] + 1) or [],
    )
    monkeypatch.setattr(
        task_app,
        "spawn_traffic_walkers",
        lambda *_args, **_kwargs: calls.__setitem__("walkers", calls["walkers"] + 1) or [],
    )

    now = [100.0]
    monkeypatch.setattr(task_app.time, "time", lambda: now[0])
    scenario = ScenarioConfig.from_dict(
        {
            "name": "delayed_task_actors",
            "uav_enabled": False,
            "traffic_vehicles": 1,
            "traffic_spawn_delay_sec": 3.0,
            "traffic_walkers": 1,
            "walker_spawn_delay_sec": 5.0,
        }
    )
    app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_delayed_task_actors"))
    app.client = object()
    app.world = object()
    app.start_time = now[0]
    app._traffic_spawned = False
    app._walkers_spawned = False

    now[0] = 102.0
    app._maybe_spawn_delayed_actors()
    assert calls == {"vehicles": 0, "walkers": 0}

    now[0] = 103.1
    app._maybe_spawn_delayed_actors()
    assert calls == {"vehicles": 1, "walkers": 0}

    now[0] = 105.1
    app._maybe_spawn_delayed_actors()
    assert calls == {"vehicles": 1, "walkers": 1}

    now[0] = 110.0
    app._maybe_spawn_delayed_actors()
    assert calls == {"vehicles": 1, "walkers": 1}


def test_task_app_uav_patrol_anchor_follows_latest_ego_transform():
    class _Ego:
        def __init__(self):
            self.transform = carla.Transform(
                carla.Location(x=10.0, y=20.0, z=0.0),
                carla.Rotation(yaw=45.0),
            )

        def get_transform(self):
            return self.transform

    scenario = ScenarioConfig.from_dict({"name": "patrol_anchor", "uav_enabled": True})
    app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_patrol_anchor"))
    app.ego_vehicle = _Ego()
    app._patrol_anchor = carla.Transform(carla.Location(x=-1.0, y=-2.0, z=0.0), carla.Rotation(yaw=-90.0))

    anchor = app._uav_patrol_anchor_transform()

    assert anchor.location.x == 10.0
    assert anchor.location.y == 20.0
    assert anchor.rotation.yaw == 45.0
