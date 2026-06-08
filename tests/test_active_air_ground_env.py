from __future__ import annotations

import carla

from carlaair_active_world import env as env_module
from carlaair_active_world.geometry import ActorState, Pose, Vector3
from carlaair_active_world.scenario import ScenarioConfig


class _Actor:
    def __init__(self, actor_id: int = 1, type_id: str = "vehicle.test", role_name: str = "") -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role_name} if role_name else {}
        self.destroyed = False
        self.autopilot = None

    def destroy(self) -> None:
        self.destroyed = True

    def set_autopilot(self, value, *_args) -> None:
        self.autopilot = value

    def get_transform(self):
        return carla.Transform(carla.Location(), carla.Rotation())

    def get_velocity(self):
        return Vector3(0.0, 0.0, 0.0)


class _Spawned:
    def __init__(self, actor, controller=None) -> None:
        self.actor = actor
        self.controller = controller


class _World:
    def __init__(self) -> None:
        self.ticks = 0

    def get_settings(self):
        return type("Settings", (), {"synchronous_mode": True})()

    def tick(self):
        self.ticks += 1


def test_active_env_spawns_configured_traffic_and_walkers(monkeypatch):
    calls = {"vehicles": None, "walkers": None}
    ego = _Actor(actor_id=10, role_name="ego")
    traffic = _Actor(actor_id=20, role_name="task_traffic")
    walker = _Actor(actor_id=30, type_id="walker.pedestrian.test", role_name="task_walker")
    controller = _Actor(actor_id=31, type_id="controller.ai.walker")

    monkeypatch.setattr(env_module, "spawn_ego_vehicle", lambda *_args, **_kwargs: ego)
    monkeypatch.setattr(env_module, "configure_autopilot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "set_traffic_manager_speed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module.ActiveAirGroundEnv, "_park_disabled_uav", lambda self: None)
    monkeypatch.setattr(
        env_module,
        "spawn_traffic_vehicles",
        lambda *_args, **kwargs: calls.__setitem__("vehicles", kwargs) or [_Spawned(traffic)],
    )
    monkeypatch.setattr(
        env_module,
        "spawn_traffic_walkers",
        lambda *_args, **kwargs: calls.__setitem__("walkers", kwargs) or [_Spawned(walker, controller)],
    )

    scenario = ScenarioConfig.from_dict(
        {
            "name": "traffic_env",
            "uav_enabled": False,
            "ego_spawn_index": 7,
            "ego_spawn_forward_m": 20.0,
            "traffic_vehicles": 2,
            "traffic_spawn_indices": [42, 43],
            "traffic_route_commands": ["Right", "Straight"],
            "traffic_walkers": 1,
            "traffic_spawn_start_index": 42,
            "traffic_speed_difference": 65.0,
            "walker_spawn_start_index": 18,
            "walker_spawn_indices": [18, 19],
            "walker_crossing_distance_m": 8.0,
            "walker_crossing_offsets_m": [-1.0, 1.0],
            "walker_speed_mps": 1.2,
        }
    )
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.world = _World()
    app.observe = lambda: {"time": 0.0}

    app.reset()

    # spawn_ego_vehicle is monkeypatched above; traffic assertions verify the rest of reset.
    assert calls["vehicles"]["count"] == 2
    assert calls["vehicles"]["start_index"] == 42
    assert calls["vehicles"]["spawn_indices"] == [42, 43]
    assert calls["vehicles"]["route_commands"] == ["Right", "Straight"]
    assert calls["vehicles"]["speed_difference"] == 65.0
    assert calls["walkers"]["count"] == 1
    assert calls["walkers"]["start_index"] == 18
    assert calls["walkers"]["spawn_indices"] == [18, 19]
    assert calls["walkers"]["crossing_distance_m"] == 8.0
    assert calls["walkers"]["crossing_offsets_m"] == [-1.0, 1.0]
    assert calls["walkers"]["use_ai_controller"] is False
    assert calls["walkers"]["speed_mps"] == 1.2
    assert app.traffic_actors == [traffic]
    assert app.walker_actors == [walker]
    assert app.walker_controllers == [controller]

    app.close()

    assert traffic.destroyed is True
    assert walker.destroyed is True
    assert controller.destroyed is True


def test_active_env_delays_configured_traffic_and_walkers(monkeypatch):
    calls = {"vehicles": 0, "walkers": 0}
    ego = _Actor(actor_id=10, role_name="ego")
    traffic = _Actor(actor_id=20, role_name="task_traffic")
    walker = _Actor(actor_id=30, type_id="walker.pedestrian.test", role_name="task_walker")
    controller = _Actor(actor_id=31, type_id="controller.ai.walker")

    monkeypatch.setattr(env_module, "spawn_ego_vehicle", lambda *_args, **_kwargs: ego)
    monkeypatch.setattr(env_module, "configure_autopilot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "set_traffic_manager_speed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module.ActiveAirGroundEnv, "_park_disabled_uav", lambda self: None)
    monkeypatch.setattr(env_module, "build_labels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        env_module,
        "spawn_traffic_vehicles",
        lambda *_args, **kwargs: calls.__setitem__("vehicles", calls["vehicles"] + 1) or [_Spawned(traffic)],
    )
    monkeypatch.setattr(
        env_module,
        "spawn_traffic_walkers",
        lambda *_args, **kwargs: calls.__setitem__("walkers", calls["walkers"] + 1) or [_Spawned(walker, controller)],
    )

    now = [100.0]
    monkeypatch.setattr(env_module.time, "time", lambda: now[0])
    scenario = ScenarioConfig.from_dict(
        {
            "name": "delayed_actors",
            "uav_enabled": False,
            "traffic_vehicles": 1,
            "traffic_spawn_delay_sec": 3.0,
            "traffic_spawn_start_index": 138,
            "traffic_walkers": 1,
            "walker_spawn_delay_sec": 5.0,
            "walker_spawn_start_index": 27,
        }
    )
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.world = _World()
    app.observe = lambda: {"time": now[0] - app.start_time}

    app.reset()
    assert calls == {"vehicles": 0, "walkers": 0}

    now[0] = 102.0
    app.step(0)
    assert calls == {"vehicles": 0, "walkers": 0}

    now[0] = 103.1
    app.step(0)
    assert calls == {"vehicles": 1, "walkers": 0}
    assert app.traffic_actors == [traffic]

    now[0] = 105.1
    app.step(0)
    assert calls == {"vehicles": 1, "walkers": 1}
    assert app.walker_actors == [walker]
    assert app.walker_controllers == [controller]

    now[0] = 110.0
    app.step(0)
    assert calls == {"vehicles": 1, "walkers": 1}

    app.close()


def test_active_env_observe_records_walker_states(monkeypatch):
    ego = _Actor(actor_id=10, role_name="ego")
    walker_state = ActorState(
        actor_id=30,
        type_id="walker.pedestrian.test",
        role_name="task_walker",
        pose=Pose(Vector3(1.0, 2.0, 0.0)),
        velocity=Vector3(0.5, 0.0, 0.0),
    )

    monkeypatch.setattr(
        env_module,
        "get_actor_state",
        lambda actor: ActorState(
            actor_id=actor.id,
            type_id=actor.type_id,
            role_name=actor.attributes.get("role_name", ""),
            pose=Pose(Vector3(0.0, 0.0, 0.0)),
            velocity=Vector3(0.0, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(env_module, "collect_vehicle_states", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(env_module, "collect_walker_states", lambda *_args, **_kwargs: [walker_state])

    scenario = ScenarioConfig.from_dict({"name": "walker_observe", "uav_enabled": False})
    app = env_module.ActiveAirGroundEnv(scenario)
    app.world = object()
    app.ego_vehicle = ego
    app.start_time = 0.0
    app.build_candidates = lambda: []

    observation = app.observe()

    assert observation["walkers"] == [walker_state.to_dict()]


def test_active_env_collision_labels_come_from_collision_sensor_events():
    scenario = ScenarioConfig.from_dict({"name": "collision_labels", "uav_enabled": False})
    app = env_module.ActiveAirGroundEnv(scenario)
    app.collision_events = [{"other_actor_id": 123}]
    label = {
        "collision_proxy": False,
        "collision_proxy_count": 0,
    }

    app._apply_collision_labels(label)

    assert label["collision"] is True
    assert label["collision_count"] == 1
    assert label["collision_events"] == [{"other_actor_id": 123}]


def test_active_env_ignores_airsim_drone_collision_sensor_events():
    scenario = ScenarioConfig.from_dict({"name": "ignore_drone_collision", "uav_enabled": True})
    app = env_module.ActiveAirGroundEnv(scenario)
    app.collision_events = [
        {"other_actor_id": 24, "other_type_id": "airsim.drone"},
        {"other_actor_id": 123, "other_type_id": "vehicle.test"},
    ]
    label = {
        "collision_proxy": False,
        "collision_proxy_count": 0,
    }

    app._apply_collision_labels(label)

    assert label["collision"] is True
    assert label["collision_count"] == 1
    assert label["collision_events"] == [{"other_actor_id": 123, "other_type_id": "vehicle.test"}]


def test_active_env_passes_forward_spawn_offset(monkeypatch):
    captured = {}
    ego = _Actor(actor_id=10, role_name="ego")

    monkeypatch.setattr(
        env_module,
        "spawn_ego_vehicle",
        lambda *_args, **kwargs: captured.update(kwargs) or ego,
    )
    monkeypatch.setattr(env_module, "configure_autopilot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "set_traffic_manager_speed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "spawn_traffic_vehicles", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(env_module, "spawn_traffic_walkers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(env_module.ActiveAirGroundEnv, "_park_disabled_uav", lambda self: None)

    scenario = ScenarioConfig.from_dict(
        {
            "name": "forward_spawn_env",
            "uav_enabled": False,
            "ego_spawn_index": 20,
            "ego_spawn_forward_m": 18.0,
        }
    )
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.world = _World()
    app.observe = lambda: {"time": 0.0}

    app.reset()

    assert captured["spawn_index"] == 20
    assert captured["forward_m"] == 18.0


def test_active_env_parks_disabled_uav(monkeypatch):
    calls = {}
    air_client = object()

    def _connect(*_args, **kwargs):
        calls["connect"] = kwargs
        return air_client

    monkeypatch.setattr(env_module, "connect_airsim", _connect)

    def _move(client, pose, ox, oy, oz, vehicle_name=None):
        calls["move"] = {
            "client": client,
            "pose": pose,
            "origin": (ox, oy, oz),
            "vehicle_name": vehicle_name,
        }

    monkeypatch.setattr(env_module, "move_uav_to", _move)
    monkeypatch.setattr(env_module, "set_uav_hover", lambda client, vehicle_name=None: calls.setdefault("hover", vehicle_name))

    scenario = ScenarioConfig.from_dict({"name": "no_uav_parking", "uav_enabled": False, "uav_name": "Drone1"})
    app = env_module.ActiveAirGroundEnv(scenario)

    app._park_disabled_uav()

    assert calls["connect"]["vehicle_name"] == "Drone1"
    assert calls["move"]["client"] is air_client
    assert calls["move"]["pose"].position.x == -1000.0
    assert calls["move"]["pose"].position.y == -1000.0
    assert calls["move"]["pose"].position.z == 120.0
    assert calls["move"]["vehicle_name"] == "Drone1"
    assert calls["hover"] == "Drone1"


def test_active_env_drives_scripted_walkers_toward_targets():
    class WalkerActor:
        def __init__(self) -> None:
            self.control = None

        def get_location(self):
            return carla.Location(x=0.0, y=0.0, z=0.0)

        def apply_control(self, control) -> None:
            self.control = control

    walker = WalkerActor()
    scenario = ScenarioConfig.from_dict({"name": "scripted_walker", "uav_enabled": False})
    app = env_module.ActiveAirGroundEnv(scenario)
    app._walker_targets = [(walker, carla.Location(x=3.0, y=4.0, z=0.0), 1.2)]

    app._drive_scripted_walkers()

    assert walker.control is not None
    assert walker.control.speed == 1.2
    assert round(walker.control.direction.x, 3) == 0.6
    assert round(walker.control.direction.y, 3) == 0.8


def test_active_env_keeps_scripted_walker_moving_when_not_fully_crossed():
    class WalkerActor:
        def __init__(self) -> None:
            self.control = None

        def get_location(self):
            return carla.Location(x=2.7, y=3.6, z=0.0)

        def apply_control(self, control) -> None:
            self.control = control

    walker = WalkerActor()
    target = carla.Location(x=3.0, y=4.0, z=0.0)
    scenario = ScenarioConfig.from_dict({"name": "scripted_walker_short", "uav_enabled": False})
    app = env_module.ActiveAirGroundEnv(scenario)
    app._walker_targets = [(walker, target, 1.2)]

    app._drive_scripted_walkers()

    assert walker.control is not None
    assert walker.control.speed == 1.2
    assert app._walker_targets == [(walker, target, 1.2)]
