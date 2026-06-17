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
        self.controls = []

    def destroy(self) -> None:
        self.destroyed = True

    def set_autopilot(self, value, *_args) -> None:
        self.autopilot = value

    def apply_control(self, control) -> None:
        self.controls.append(control)

    def get_transform(self):
        return carla.Transform(carla.Location(), carla.Rotation())

    def get_velocity(self):
        return Vector3(0.0, 0.0, 0.0)


class _Spawned:
    def __init__(self, actor, controller=None) -> None:
        self.actor = actor
        self.controller = controller


class _ActorList(list):
    def filter(self, _pattern):
        return []


class _World:
    def __init__(self) -> None:
        self.ticks = 0

    def get_settings(self):
        return type("Settings", (), {"synchronous_mode": True})()

    def get_actors(self):
        return _ActorList()

    def tick(self):
        self.ticks += 1


class _TimedSettings:
    def __init__(self, synchronous_mode=False, fixed_delta_seconds=None) -> None:
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds


class _TimedWorld(_World):
    def __init__(self, synchronous_mode=False, fixed_delta_seconds=None) -> None:
        super().__init__()
        self.settings = _TimedSettings(synchronous_mode, fixed_delta_seconds)
        self.applied_settings = []

    def get_settings(self):
        return self.settings

    def apply_settings(self, settings):
        self.settings = settings
        self.applied_settings.append((settings.synchronous_mode, settings.fixed_delta_seconds))


def test_active_env_clears_existing_scene_before_spawn_in_synchronous_world(monkeypatch):
    order = []
    ego = _Actor(actor_id=10, role_name="ego")

    monkeypatch.setattr(
        env_module,
        "cleanup_actors_by_role",
        lambda *_args, **_kwargs: order.append("cleanup_roles") or 2,
    )
    monkeypatch.setattr(
        env_module,
        "cleanup_old_vehicles",
        lambda *_args, **_kwargs: order.append("cleanup_ego") or 1,
    )
    monkeypatch.setattr(
        env_module,
        "spawn_ego_vehicle",
        lambda *_args, **_kwargs: order.append("spawn_ego") or ego,
    )
    monkeypatch.setattr(env_module, "configure_autopilot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "set_traffic_manager_speed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "spawn_traffic_vehicles", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(env_module, "spawn_static_obstacle_vehicles", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(env_module, "spawn_traffic_walkers", lambda *_args, **_kwargs: [])

    scenario = ScenarioConfig.from_dict({"name": "clean_spawn", "uav_enabled": False})
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.world = _World()
    app.observe = lambda: {"time": 0.0}

    app.reset()

    assert order[:3] == ["cleanup_roles", "cleanup_ego", "spawn_ego"]
    assert app.world.ticks >= 2


def test_active_env_spawns_configured_traffic_and_walkers(monkeypatch):
    calls = {"vehicles": None, "walkers": None, "obstacles": None}
    ego = _Actor(actor_id=10, role_name="ego")
    traffic = _Actor(actor_id=20, role_name="task_traffic")
    obstacle = _Actor(actor_id=21, role_name="task_obstacle")
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
    monkeypatch.setattr(
        env_module,
        "spawn_static_obstacle_vehicles",
        lambda *_args, **kwargs: calls.__setitem__("obstacles", kwargs) or [_Spawned(obstacle)],
        raising=False,
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
            "obstacle_vehicles": 1,
            "obstacle_anchor_x": -41.6,
            "obstacle_anchor_y": 58.0,
            "obstacle_anchor_yaw_deg": -90.0,
            "obstacle_forward_offsets_m": [0.0],
            "obstacle_lateral_offsets_m": [0.8],
            "obstacle_yaw_offsets_deg": [25.0],
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
    assert calls["obstacles"]["count"] == 1
    assert calls["obstacles"]["anchor_transform"].location.x == -41.6
    assert calls["obstacles"]["anchor_transform"].location.y == 58.0
    assert calls["obstacles"]["anchor_transform"].rotation.yaw == -90.0
    assert calls["obstacles"]["forward_offsets_m"] == [0.0]
    assert calls["obstacles"]["lateral_offsets_m"] == [0.8]
    assert calls["obstacles"]["yaw_offsets_deg"] == [25.0]
    assert app.traffic_actors == [traffic]
    assert app.obstacle_actors == [obstacle]
    assert app.walker_actors == [walker]
    assert app.walker_controllers == [controller]

    app.close()

    assert traffic.destroyed is True
    assert obstacle.destroyed is True
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
    app.observe = lambda: {"time": app._episode_elapsed_sec}

    app.reset()
    assert calls == {"vehicles": 0, "walkers": 0}

    for _ in range(6):
        app.step(0)
    assert calls == {"vehicles": 0, "walkers": 0}

    app.step(0)
    assert calls == {"vehicles": 1, "walkers": 0}
    assert app.traffic_actors == [traffic]

    for _ in range(3):
        app.step(0)
    assert calls == {"vehicles": 1, "walkers": 0}

    app.step(0)
    assert calls == {"vehicles": 1, "walkers": 1}
    assert app.walker_actors == [walker]
    assert app.walker_controllers == [controller]

    app.step(0)
    assert calls == {"vehicles": 1, "walkers": 1}

    app.close()


def test_active_env_configures_sync_timing_and_restores_on_close(monkeypatch):
    ego = _Actor(actor_id=10, role_name="ego")

    monkeypatch.setattr(env_module, "spawn_ego_vehicle", lambda *_args, **_kwargs: ego)
    monkeypatch.setattr(env_module, "configure_autopilot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "set_traffic_manager_speed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module.ActiveAirGroundEnv, "_park_disabled_uav", lambda self: None)

    scenario = ScenarioConfig.from_dict({"name": "timed_world", "uav_enabled": False, "step_sec": 0.05})
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.world = _TimedWorld(synchronous_mode=False, fixed_delta_seconds=None)
    app.observe = lambda: {"time": 0.0}

    app.reset()

    assert app.world.applied_settings[0] == (True, 0.05)
    assert app.world.get_settings().synchronous_mode is True
    assert app.world.get_settings().fixed_delta_seconds == 0.05

    app.close()

    assert app.world.applied_settings[-1] == (False, None)
    assert app.world.get_settings().synchronous_mode is False
    assert app.world.get_settings().fixed_delta_seconds is None


def test_active_env_uses_inline_ego_control_in_synchronous_mode(monkeypatch):
    ego = _Actor(actor_id=10, role_name="ego")
    calls = {"predict": 0}

    class _Driver:
        def __init__(self, *_args, **_kwargs) -> None:
            self.last_diagnostics = {}

        def predict(self, *_args, **_kwargs):
            calls["predict"] += 1
            return "control"

    monkeypatch.setattr(env_module, "spawn_ego_vehicle", lambda *_args, **_kwargs: ego)
    monkeypatch.setattr(env_module, "set_traffic_manager_speed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "build_labels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(env_module.ActiveAirGroundEnv, "_park_disabled_uav", lambda self: None)
    monkeypatch.setattr(env_module, "RouteFollowingDriver", _Driver)

    scenario = ScenarioConfig.from_dict(
        {"name": "sync_inline_control", "uav_enabled": False, "ego_control_mode": "route_follow", "step_sec": 0.05}
    )
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.world = _TimedWorld(synchronous_mode=False, fixed_delta_seconds=None)
    app.observe = lambda: {"time": app._episode_elapsed_sec}

    app.reset()

    assert app._ego_control_inline is True
    assert app._ego_driver_thread is None

    app.step(0)

    assert calls["predict"] == 1
    assert ego.controls == ["control"]
    assert app.world.ticks == 2

    app.close()


def test_active_env_starts_episode_clock_after_uav_setup(monkeypatch):
    calls = {"vehicles": 0, "uav_observation_times": []}
    ego = _Actor(actor_id=10, role_name="ego")
    traffic = _Actor(actor_id=20, role_name="task_traffic")

    monkeypatch.setattr(env_module, "spawn_ego_vehicle", lambda *_args, **_kwargs: ego)
    monkeypatch.setattr(env_module, "configure_autopilot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "set_traffic_manager_speed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "move_uav_to", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "set_uav_hover", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env_module, "build_labels", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        env_module,
        "spawn_traffic_vehicles",
        lambda *_args, **kwargs: calls.__setitem__("vehicles", calls["vehicles"] + 1) or [_Spawned(traffic)],
    )

    now = [100.0]
    monkeypatch.setattr(env_module.time, "time", lambda: now[0])

    def _place_initial_uav(self, observation):
        calls["uav_observation_times"].append(observation["time"])
        now[0] += 30.0

    monkeypatch.setattr(env_module.ActiveAirGroundEnv, "_place_initial_uav", _place_initial_uav)

    scenario = ScenarioConfig.from_dict(
        {
            "name": "delayed_after_uav_setup",
            "uav_enabled": True,
            "traffic_vehicles": 1,
            "traffic_spawn_delay_sec": 3.0,
            "traffic_spawn_start_index": 138,
            "candidate_offsets": [{"name": "front", "x": 10.0, "y": 0.0, "z": 12.0}],
        }
    )
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.air_client = object()
    app.world = _World()
    app.observe = lambda: {"time": app._episode_elapsed_sec}

    observation = app.reset()
    assert observation["time"] == 0.0
    assert calls == {"vehicles": 0, "uav_observation_times": [0.0]}
    assert app.start_time == 130.0

    for _ in range(6):
        app.step(0)
    assert calls["vehicles"] == 0

    app.step(0)
    assert calls["vehicles"] == 1

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


def test_active_env_passes_explicit_spawn_pose(monkeypatch):
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
            "name": "explicit_spawn_env",
            "uav_enabled": False,
            "ego_spawn_index": 86,
            "ego_spawn_x": -28.3,
            "ego_spawn_y": 130.15,
            "ego_spawn_z": 0.6,
            "ego_spawn_yaw_deg": -179.65,
        }
    )
    app = env_module.ActiveAirGroundEnv(scenario)
    app.client = object()
    app.world = _World()
    app.observe = lambda: {"time": 0.0}

    app.reset()

    spawn_transform = captured["spawn_transform"]
    assert spawn_transform.location.x == -28.3
    assert spawn_transform.location.y == 130.15
    assert spawn_transform.location.z == 0.6
    assert spawn_transform.rotation.yaw == -179.65


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


def test_active_env_freezes_scripted_walker_at_crossing_target():
    class WalkerActor:
        def __init__(self) -> None:
            self.control = None
            self.location = carla.Location(x=2.92, y=3.94, z=0.0)
            self.transforms = []

        def get_location(self):
            return self.location

        def get_transform(self):
            return carla.Transform(self.location, carla.Rotation(yaw=10.0))

        def set_transform(self, transform) -> None:
            self.transforms.append(transform)
            self.location = transform.location

        def apply_control(self, control) -> None:
            self.control = control

    walker = WalkerActor()
    target = carla.Location(x=3.0, y=4.0, z=0.0)
    scenario = ScenarioConfig.from_dict({"name": "scripted_walker_arrived", "uav_enabled": False})
    app = env_module.ActiveAirGroundEnv(scenario)
    app._walker_targets = [(walker, target, 1.2)]

    app._drive_scripted_walkers()

    assert walker.control is not None
    assert walker.control.speed == 0.0
    assert walker.location.x == target.x
    assert walker.location.y == target.y
    assert app._walker_targets == []

    walker.location = carla.Location(x=2.5, y=3.5, z=0.0)
    app._drive_scripted_walkers()

    assert walker.control.speed == 0.0
    assert walker.location.x == target.x
    assert walker.location.y == target.y


def test_active_env_freezes_overshot_scripted_walker_without_reverse_snap():
    class WalkerActor:
        def __init__(self) -> None:
            self.control = None
            self.location = carla.Location(x=-11.0, y=0.0, z=0.0)

        def get_location(self):
            return self.location

        def get_velocity(self):
            return carla.Vector3D(x=-0.8, y=0.0, z=0.0)

        def get_transform(self):
            return carla.Transform(self.location, carla.Rotation(yaw=180.0))

        def set_transform(self, transform) -> None:
            self.location = transform.location

        def apply_control(self, control) -> None:
            self.control = control

    walker = WalkerActor()
    scenario = ScenarioConfig.from_dict({"name": "scripted_walker_overshot", "uav_enabled": False})
    app = env_module.ActiveAirGroundEnv(scenario)
    app._walker_targets = [(walker, carla.Location(x=-10.0, y=0.0, z=0.0), 0.85)]

    app._drive_scripted_walkers()

    assert walker.control is not None
    assert walker.control.speed == 0.0
    assert walker.location.x == -11.0
    assert app._walker_targets == []
