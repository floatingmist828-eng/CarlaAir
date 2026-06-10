from __future__ import annotations

import numpy as np

from carlaair_active_world.adversarial import apply_vision_attack, apply_weather_preset, build_weather_parameters
from carlaair_active_world.scenario import ScenarioConfig
from carlaair_active_world.sensors import VehicleSensorRig
from carlaair_active_world.vision_driver import VisionEgoDriver
from carlaair_active_world.vision_models.simple_lane import SimpleLaneVisionPolicy
from carlaair_active_world.vision_models.yolo11_obstacle import UltralyticsObstacleDetector


def _lane_image(width: int = 160, height: int = 90, lane_x: int = 80) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[height // 2 :, max(0, lane_x - 2) : min(width, lane_x + 3), :] = 255
    return image


def _yellow_line_image(width: int = 160, height: int = 90, lane_x: int = 58) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[height // 2 :, max(0, lane_x - 2) : min(width, lane_x + 3), :] = (255, 210, 0)
    return image


def _clear_depth(width: int = 160, height: int = 90) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _semantic(width: int = 160, height: int = 90, road_x0: int = 0, road_x1: int = 160) -> np.ndarray:
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[height // 2 :, road_x0:road_x1] = 7
    return semantic


def test_vision_simple_scenario_config_loads_project_file():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_simple.json")

    assert scenario.ego_control_mode == "vision_simple"
    assert scenario.ego_target_speed_mps == 4.0
    assert scenario.ego_drive_hz == 8.0
    assert scenario.ego_spawn_index == 20
    assert scenario.vehicle_sensor_limit == 1
    assert scenario.traffic_vehicles == 0
    assert scenario.uav_enabled is False


def test_vision_attack_scenario_config_loads_project_file():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_texture_attack.json")

    assert scenario.ego_control_mode == "vision_rgb_only"
    assert scenario.traffic_vehicles == 0
    assert scenario.vision_attack == "texture"
    assert scenario.vision_attack_intensity == 1.0
    assert scenario.weather_preset == "none"
    assert scenario.uav_enabled is False


def test_weather_attack_scenario_config_loads_project_file():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_weather_attack.json")

    assert scenario.ego_control_mode == "vision_rgb_only"
    assert scenario.traffic_vehicles == 0
    assert scenario.vision_attack == "weather"
    assert scenario.weather_preset == "hard_rain_fog"
    assert scenario.uav_enabled is False


def test_vision_detector_config_round_trips():
    scenario = ScenarioConfig.from_dict(
        {
            "name": "detector_config",
            "vision_detector_model_path": "models/yolo11n.pt",
            "vision_detector_confidence": 0.4,
        }
    )

    assert scenario.vision_detector_model_path == "models/yolo11n.pt"
    assert scenario.vision_detector_confidence == 0.4
    assert scenario.to_dict()["vision_detector_model_path"] == "models/yolo11n.pt"
    assert scenario.to_dict()["vision_detector_confidence"] == 0.4


def test_hard_rain_fog_weather_preset_is_adverse():
    weather = build_weather_parameters("hard_rain_fog")

    assert weather is not None
    assert weather.precipitation >= 80.0
    assert weather.fog_density >= 70.0
    assert weather.wetness >= 80.0


def test_none_weather_preset_resets_world_to_default(monkeypatch):
    import carla

    calls = []
    monkeypatch.setattr(carla.WeatherParameters, "Default", "DefaultWeather", raising=False)

    class _World:
        def set_weather(self, weather):
            calls.append(weather)

    apply_weather_preset(_World(), "none")

    assert calls == ["DefaultWeather"]


def test_texture_attack_overlays_rgb_and_can_remove_semantic():
    rgb = np.zeros((90, 160, 3), dtype=np.uint8)
    semantic = _semantic()

    attacked = apply_vision_attack(
        {"rgb": rgb, "semantic": semantic, "depth": _clear_depth()},
        attack="texture",
        intensity=1.0,
        disable_semantic=True,
    )

    assert attacked["rgb"].shape == rgb.shape
    assert np.count_nonzero(attacked["rgb"]) > 0
    assert attacked["semantic"] is None


def test_weather_attack_degrades_rgb_contrast_and_can_remove_semantic():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)
    rgb = _lane_image()
    semantic = _semantic()

    policy.predict({"rgb": rgb, "depth": _clear_depth(), "speed_mps": 1.0})
    clean_confidence = policy.last_diagnostics["lane_confidence"]
    policy.reset()
    attacked = apply_vision_attack(
        {"rgb": rgb, "semantic": semantic},
        attack="weather",
        intensity=1.0,
        disable_semantic=True,
    )
    policy.predict({"rgb": attacked["rgb"], "depth": _clear_depth(), "speed_mps": 1.0})

    assert attacked["rgb"].shape == rgb.shape
    assert policy.last_diagnostics["lane_confidence"] < clean_confidence
    assert attacked["semantic"] is None


def test_texture_attack_biases_rgb_policy_away_from_clean_yellow_boundary():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)
    clean = _yellow_line_image(lane_x=58)
    attacked = apply_vision_attack({"rgb": clean}, attack="texture", intensity=1.0)["rgb"]

    clean_control = policy.predict({"rgb": clean, "depth": _clear_depth(), "speed_mps": 1.0})
    policy.reset()
    attack_control = policy.predict({"rgb": attacked, "depth": _clear_depth(), "speed_mps": 1.0})

    assert clean_control.steer > 0.0
    assert attack_control.steer > clean_control.steer + 0.10


def test_vision_driver_rgb_only_disables_semantic_for_policy():
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.obs = None
            self.last_diagnostics = {}

        def predict(self, obs):
            self.obs = obs
            import carla

            return carla.VehicleControl()

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    policy = _Policy()
    original = VisionEgoDriver.sensor_rig_class
    VisionEgoDriver.sensor_rig_class = _Rig
    try:
        driver = VisionEgoDriver(object(), _Vehicle(), use_semantic=False, policy=policy)
        driver.predict()
    finally:
        VisionEgoDriver.sensor_rig_class = original

    assert policy.obs["semantic"] is None


def test_vision_driver_can_disable_depth_for_rgb_only_policy():
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.obs = None
            self.last_diagnostics = {}

        def predict(self, obs):
            self.obs = obs
            import carla

            return carla.VehicleControl()

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    policy = _Policy()
    original = VisionEgoDriver.sensor_rig_class
    VisionEgoDriver.sensor_rig_class = _Rig
    try:
        driver = VisionEgoDriver(object(), _Vehicle(), use_semantic=False, use_depth=False, policy=policy)
        driver.predict()
    finally:
        VisionEgoDriver.sensor_rig_class = original

    assert driver.sensor_rig.kwargs["disable_depth"] is True
    assert policy.obs["depth"] is None


def test_vision_driver_forwards_detector_obstacle_to_policy():
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Detector:
        def predict(self, rgb):
            return {"obstacle": True, "label": "car", "confidence": 0.8}

    class _Policy:
        def __init__(self) -> None:
            self.obs = None
            self.last_diagnostics = {"policy": "ok"}

        def predict(self, obs):
            self.obs = obs
            import carla

            return carla.VehicleControl()

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    policy = _Policy()
    original = VisionEgoDriver.sensor_rig_class
    VisionEgoDriver.sensor_rig_class = _Rig
    try:
        driver = VisionEgoDriver(object(), _Vehicle(), policy=policy, detector=_Detector())
        driver.predict()
    finally:
        VisionEgoDriver.sensor_rig_class = original

    assert policy.obs["vision_obstacle"] is True
    assert policy.obs["vision_detector"]["label"] == "car"
    assert driver.last_diagnostics["vision_obstacle"] is True


def test_vision_driver_forwards_uav_bev_context_to_policy():
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.obs = None
            self.last_diagnostics = {"policy": "ok"}

        def predict(self, obs):
            self.obs = obs
            import carla

            return carla.VehicleControl()

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    policy = _Policy()
    original = VisionEgoDriver.sensor_rig_class
    VisionEgoDriver.sensor_rig_class = _Rig
    try:
        driver = VisionEgoDriver(
            object(),
            _Vehicle(),
            policy=policy,
            uav_bev_provider=lambda: {"available": True, "center_bias": 0.4},
        )
        driver.predict()
    finally:
        VisionEgoDriver.sensor_rig_class = original

    assert policy.obs["uav_bev"]["available"] is True
    assert policy.obs["uav_bev"]["center_bias"] == 0.4
    assert driver.last_diagnostics["uav_bev"]["available"] is True


def test_vision_driver_uses_configured_junction_command_sequence(monkeypatch):
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    now = [10.0]
    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)
    driver = VisionEgoDriver(
        object(),
        _Vehicle(),
        junction_command_sequence=["right", "straight"],
        junction_command_hold_sec=2.0,
        clock=lambda: now[0],
    )

    assert driver._navigation_command_for_lane({"in_junction": False}) == "lane_follow"
    assert driver._navigation_command_for_lane({"in_junction": True}) == "right"
    now[0] = 11.0
    assert driver._navigation_command_for_lane({"in_junction": True}) == "right"
    now[0] = 13.0
    assert driver._navigation_command_for_lane({"in_junction": False}) == "lane_follow"
    now[0] = 14.0
    assert driver._navigation_command_for_lane({"in_junction": True}) == "straight"


def test_vision_driver_can_hold_junction_command_until_exit(monkeypatch):
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    now = [10.0]
    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)
    driver = VisionEgoDriver(
        object(),
        _Vehicle(),
        junction_command_sequence=["straight"],
        junction_command_hold_sec=2.0,
        junction_command_hold_until_exit=True,
        clock=lambda: now[0],
    )

    assert driver._navigation_command_for_lane({"in_junction": True}) == "straight"
    now[0] = 20.0
    assert driver._navigation_command_for_lane({"in_junction": True}) == "straight"
    assert driver._navigation_command_for_lane({"in_junction": False}) == "lane_follow"


def test_vision_driver_does_not_hold_right_turn_until_exit(monkeypatch):
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    now = [10.0]
    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)
    driver = VisionEgoDriver(
        object(),
        _Vehicle(),
        junction_command_sequence=["right"],
        junction_command_hold_sec=2.0,
        junction_command_hold_until_exit=True,
        clock=lambda: now[0],
    )

    assert driver._navigation_command_for_lane({"in_junction": True}) == "right"
    now[0] = 20.0
    assert driver._navigation_command_for_lane({"in_junction": True}) == "lane_follow"


def test_vision_driver_adds_forward_interaction_hazard(monkeypatch):
    import carla

    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": _lane_image(), "semantic": _semantic(), "depth": _clear_depth()}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.obs = None
            self.last_diagnostics = {}

        def predict(self, obs):
            self.obs = obs
            return carla.VehicleControl()

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="") -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    pedestrian = _Actor(2, "walker.pedestrian.test", 4.0, 0.8, role_name="task_walker")
    side_vehicle = _Actor(3, "vehicle.test", 5.0, 8.0, role_name="task_traffic")
    policy = _Policy()
    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)

    driver = VisionEgoDriver(_World([ego, pedestrian, side_vehicle]), ego, policy=policy)
    driver.predict()

    hazard = policy.obs["interaction_hazard"]
    assert hazard["active"] is True
    assert hazard["action"] == "stop"
    assert hazard["actor_type"] == "walker"
    assert hazard["actor_id"] == 2


def test_vision_driver_stops_for_crossing_walker_before_turn_crosswalk():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D()

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    pedestrian = _Actor(2, "walker.pedestrian.test", 12.0, 2.2, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, pedestrian]))

    assert hazard["active"] is True
    assert hazard["action"] == "stop"
    assert hazard["target_speed_mps"] == 0.0


def test_vision_driver_clears_walker_outside_front_path():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D(y=-0.8)

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    pedestrian = _Actor(2, "walker.pedestrian.test", 12.0, 4.4, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, pedestrian]))

    assert hazard["active"] is False


def test_vision_driver_slows_for_walker_near_front_path():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D(y=-0.8)

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    pedestrian = _Actor(2, "walker.pedestrian.test", 12.0, 3.6, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, pedestrian]))

    assert hazard["active"] is True
    assert hazard["action"] == "slow"
    assert hazard["target_speed_mps"] > 0.0


def test_vision_driver_clears_stationary_walker_after_crossing():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D()

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    crossed = _Actor(2, "walker.pedestrian.test", 14.5, -3.6, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossed]))

    assert hazard["active"] is False


def test_vision_driver_clears_stationary_walker_just_past_ego_lane():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D()

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    crossed = _Actor(2, "walker.pedestrian.test", 12.0, -2.7, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossed]))

    assert hazard["active"] is False


def test_vision_driver_clears_task_walker_once_past_far_side_path():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D(y=-0.8)

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    crossed = _Actor(2, "walker.pedestrian.test", 8.5, -0.2, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossed]))

    assert hazard["active"] is False


def test_vision_driver_clears_task_walker_on_world_far_side_after_turn():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", yaw=0.0, velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=yaw))
            self._velocity = velocity or carla.Vector3D(x=-0.8)

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", -44.0, 121.0, role_name="ego", yaw=-150.0)
    crossed = _Actor(2, "walker.pedestrian.test", -54.0, 115.0, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossed]))

    assert hazard["active"] is False


def test_vision_driver_stops_for_close_crossing_walker_before_crosswalk():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D()

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    crossing = _Actor(2, "walker.pedestrian.test", 0.6, 3.8, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossing]))

    assert hazard["active"] is True
    assert hazard["action"] == "stop"


def test_vision_driver_prebrakes_for_turn_crosswalk_walker():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D(y=-0.8)

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    crossing = _Actor(2, "walker.pedestrian.test", 0.5, 5.8, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossing]))

    assert hazard["active"] is True
    assert hazard["action"] == "stop"


def test_vision_driver_prebrakes_for_fast_turn_crosswalk_approach():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D(y=-0.8)

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    crossing = _Actor(2, "walker.pedestrian.test", 9.0, 8.5, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossing]))

    assert hazard["active"] is True
    assert hazard["action"] == "stop"


def test_vision_driver_world_guards_first_turn_crosswalk():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", yaw=0.0, velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=yaw))
            self._velocity = velocity or carla.Vector3D(x=-0.8)

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", -33.0, 128.0, role_name="ego", yaw=-153.0)
    crossing = _Actor(2, "walker.pedestrian.test", -36.5, 116.9, role_name="task_walker")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, crossing]))

    assert hazard["active"] is True
    assert hazard["action"] == "stop"


def test_vision_driver_requests_left_avoidance_for_lane_obstacle():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D()

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    obstacle = _Actor(2, "vehicle.dodge.charger_police_2020", 18.0, 0.2, role_name="task_obstacle")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, obstacle]))

    assert hazard["active"] is True
    assert hazard["action"] == "avoid_left"
    assert hazard["target_speed_mps"] > 1.0
    assert hazard["avoid_lateral_m"] < 0.0


def test_vision_driver_requests_early_left_avoidance_for_static_lane_obstacle():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D()

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    obstacle = _Actor(2, "vehicle.dodge.charger_police_2020", 32.0, 0.5, role_name="task_obstacle")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, obstacle]))

    assert hazard["active"] is True
    assert hazard["action"] == "avoid_left"
    assert hazard["target_speed_mps"] <= 2.0


def test_vision_driver_clears_lane_obstacle_after_left_lane_change():
    import carla

    class _Actor:
        def __init__(self, actor_id, type_id, x, y, role_name="", velocity=None) -> None:
            self.id = actor_id
            self.type_id = type_id
            self.attributes = {"role_name": role_name}
            self._transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation())
            self._velocity = velocity or carla.Vector3D()

        def get_transform(self):
            return self._transform

        def get_location(self):
            return self._transform.location

        def get_velocity(self):
            return self._velocity

    class _Actors(list):
        def filter(self, pattern):
            if pattern == "vehicle.*":
                return [item for item in self if item.type_id.startswith("vehicle.")]
            if pattern == "walker.pedestrian.*":
                return [item for item in self if item.type_id.startswith("walker.pedestrian.")]
            return []

    class _World:
        def __init__(self, actors):
            self._actors = _Actors(actors)

        def get_actors(self):
            return self._actors

    ego = _Actor(1, "vehicle.ego", 0.0, 0.0, role_name="ego")
    obstacle_to_right = _Actor(2, "vehicle.dodge.charger_police_2020", 3.0, 2.45, role_name="task_obstacle")

    hazard = VisionEgoDriver._interaction_hazard(ego, _World([ego, obstacle_to_right]))

    assert hazard["active"] is False


def test_yolo_detector_reports_traffic_diagnostics_without_obstacle():
    class _Boxes:
        cls = np.asarray([9])
        conf = np.asarray([0.8])
        xyxy = np.asarray([[70.0, 10.0, 90.0, 40.0]])

    class _Result:
        boxes = _Boxes()

    class _Model:
        names = {9: "traffic light"}

        def predict(self, image, conf, verbose, device):
            return [_Result()]

    detector = object.__new__(UltralyticsObstacleDetector)
    detector.model = _Model()
    detector.confidence = 0.35

    diagnostics = detector.predict(np.zeros((90, 160, 3), dtype=np.uint8))

    assert diagnostics["available"] is True
    assert diagnostics["obstacle"] is False
    assert diagnostics["traffic"] is True
    assert diagnostics["traffic_label"] == "traffic light"
    assert diagnostics["traffic_detections"] == 1


def test_yolo_forward_obstacle_rejects_midfield_static_car_bbox():
    bbox = (270.0, 115.0, 397.0, 201.0)

    assert not UltralyticsObstacleDetector._is_forward_obstacle(640, 360, bbox)


def test_yolo_forward_obstacle_rejects_clean_scene_self_or_background_bbox():
    bbox = (127.0, 126.0, 316.0, 260.0)

    assert not UltralyticsObstacleDetector._is_forward_obstacle(640, 360, bbox)


def test_yolo_forward_obstacle_rejects_right_edge_ego_body_bbox():
    bbox = (251.0, 140.0, 639.0, 357.0)

    assert not UltralyticsObstacleDetector._is_forward_obstacle(640, 360, bbox)


def test_yolo_forward_obstacle_accepts_close_center_car_bbox():
    bbox = (220.0, 145.0, 430.0, 315.0)

    assert UltralyticsObstacleDetector._is_forward_obstacle(640, 360, bbox)


def test_vehicle_sensor_rig_spawns_rgb_depth_and_semantic_cameras():
    class _Blueprint:
        def __init__(self, type_id: str) -> None:
            self.type_id = type_id
            self.attributes = {}

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    class _BlueprintLibrary:
        def find(self, type_id: str):
            return _Blueprint(type_id)

    class _Sensor:
        def __init__(self, blueprint: _Blueprint) -> None:
            self.blueprint = blueprint
            self.callback = None

        def listen(self, callback) -> None:
            self.callback = callback

    class _World:
        def __init__(self) -> None:
            self.spawned = []

        def get_blueprint_library(self):
            return _BlueprintLibrary()

        def spawn_actor(self, blueprint, transform, attach_to=None):
            sensor = _Sensor(blueprint)
            self.spawned.append(sensor)
            return sensor

    world = _World()
    rig = VehicleSensorRig(world, object(), "ego", width=4, height=3)

    rig.spawn()

    assert [s.blueprint.type_id for s in world.spawned] == [
        "sensor.camera.rgb",
        "sensor.camera.depth",
        "sensor.camera.semantic_segmentation",
    ]
    assert set(rig.latest) == {"rgb", "depth", "semantic"}


def test_vehicle_sensor_rig_can_spawn_rgb_only_camera():
    class _Blueprint:
        def __init__(self, type_id: str) -> None:
            self.type_id = type_id
            self.attributes = {}

        def set_attribute(self, key: str, value: str) -> None:
            self.attributes[key] = value

    class _BlueprintLibrary:
        def find(self, type_id: str):
            return _Blueprint(type_id)

    class _Sensor:
        def __init__(self, blueprint: _Blueprint) -> None:
            self.blueprint = blueprint
            self.callback = None

        def listen(self, callback) -> None:
            self.callback = callback

    class _World:
        def __init__(self) -> None:
            self.spawned = []

        def get_blueprint_library(self):
            return _BlueprintLibrary()

        def spawn_actor(self, blueprint, transform, attach_to=None):
            sensor = _Sensor(blueprint)
            self.spawned.append(sensor)
            return sensor

    world = _World()
    rig = VehicleSensorRig(
        world,
        object(),
        "ego",
        width=4,
        height=3,
        disable_depth=True,
        disable_semantic=True,
    )

    rig.spawn()

    assert [s.blueprint.type_id for s in world.spawned] == ["sensor.camera.rgb"]
    assert set(rig.latest) == {"rgb"}


def test_vehicle_sensor_rig_camera_points_down_to_see_road():
    rig = VehicleSensorRig(object(), object(), "ego")

    assert rig.attach_transform.rotation.pitch < 0.0


def test_simple_lane_policy_prefers_semantic_road_center_over_rgb():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "depth": _clear_depth(),
            "semantic": _semantic(road_x0=90, road_x1=160),
            "speed_mps": 1.0,
        }
    )

    assert control.steer > 0.15
    assert control.throttle > 0.0
    assert policy.last_diagnostics["lane_source"] == "semantic"


def test_simple_lane_policy_targets_right_lane_with_full_semantic_road():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    control = policy.predict({"semantic": _semantic(road_x0=0, road_x1=160), "speed_mps": 1.0})

    assert control.steer > 0.08
    assert policy.last_diagnostics["lane_source"] == "semantic"
    assert policy.last_diagnostics["lane_target_x"] > 80.0


def test_simple_lane_policy_can_drive_from_semantic_without_rgb():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    control = policy.predict({"semantic": _semantic(road_x0=0, road_x1=70), "speed_mps": 1.0})

    assert control.steer < -0.15
    assert control.throttle > 0.0
    assert control.brake == 0.0
    assert policy.last_diagnostics["lane_source"] == "semantic"


def test_simple_lane_policy_steers_right_when_lane_center_is_right():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    control = policy.predict({"rgb": _lane_image(lane_x=120), "depth": _clear_depth(), "speed_mps": 1.0})

    assert control.steer > 0.15
    assert control.throttle > 0.0
    assert control.brake == 0.0
    assert policy.last_diagnostics["lane_confidence"] > 0.0


def test_simple_lane_policy_steers_left_when_lane_center_is_left():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    control = policy.predict({"rgb": _lane_image(lane_x=40), "depth": _clear_depth(), "speed_mps": 1.0})

    assert control.steer < -0.15
    assert control.throttle > 0.0
    assert control.brake == 0.0


def test_simple_lane_policy_treats_yellow_line_as_left_boundary():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    control = policy.predict({"rgb": _yellow_line_image(lane_x=58), "depth": _clear_depth(), "speed_mps": 1.0})

    assert control.steer > 0.05
    assert control.throttle > 0.0
    assert policy.last_diagnostics["lane_source"] == "rgb"
    assert policy.last_diagnostics["lane_target_x"] > 105.0


def test_simple_lane_policy_keeps_lane_target_during_brief_rgb_dropout():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    first = policy.predict({"rgb": _yellow_line_image(lane_x=58), "depth": _clear_depth(), "speed_mps": 1.0})
    second = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "depth": _clear_depth(), "speed_mps": 1.0})

    assert first.steer > 0.05
    assert second.steer > 0.0
    assert policy.last_diagnostics["lane_source"] == "memory"


def test_simple_lane_policy_keeps_lane_target_through_short_lane_gap():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    policy.predict({"rgb": _yellow_line_image(lane_x=58), "depth": _clear_depth(), "speed_mps": 1.0})
    control = None
    for _ in range(20):
        control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "depth": _clear_depth(), "speed_mps": 1.0})

    assert control is not None
    assert control.steer > 0.0
    assert policy.last_diagnostics["lane_source"] == "memory"


def test_simple_lane_policy_brakes_for_close_depth_obstacle():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)
    depth = _clear_depth()
    depth[50:82, 58:102, :] = 255

    control = policy.predict({"rgb": _lane_image(lane_x=80), "depth": depth, "speed_mps": 2.0})

    assert control.brake == 1.0
    assert control.throttle == 0.0
    assert policy.last_diagnostics["clearance"] < policy.config.min_clearance_signal


def test_simple_lane_policy_brakes_for_visual_detector_obstacle():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)

    control = policy.predict(
        {
            "semantic": _semantic(road_x0=20, road_x1=140),
            "depth": _clear_depth(),
            "speed_mps": 1.0,
            "vision_obstacle": True,
        }
    )

    assert control.brake == 1.0
    assert control.throttle == 0.0
    assert policy.last_diagnostics["vision_obstacle"] is True


def test_simple_lane_policy_reverses_briefly_when_stuck_with_valid_vision():
    policy = SimpleLaneVisionPolicy(target_speed_mps=4.0)
    control = None

    for _ in range(policy.config.stuck_frame_threshold + 1):
        control = policy.predict(
            {
                "semantic": _semantic(road_x0=20, road_x1=140),
                "depth": _clear_depth(),
                "speed_mps": 0.0,
            }
        )

    assert control is not None
    assert control.reverse is True
    assert control.throttle > 0.0
    assert control.brake == 0.0
    assert policy.last_diagnostics["recovery_active"] is True


def test_active_uav_task_starts_vehicle_viewer_before_uav_takeoff():
    from pathlib import Path

    import carla
    from carlaair_active_world import task_app

    order = []

    class _Actor:
        id = 42

        def set_autopilot(self, value):
            pass

        def get_transform(self):
            return carla.Transform(carla.Location(), carla.Rotation())

    class _World:
        def get_settings(self):
            return type("Settings", (), {"synchronous_mode": False})()

    class _Controller:
        def takeoff(self):
            order.append("takeoff")

        def hover(self):
            order.append("hover")

    original = {
        "cleanup_actors_by_role": task_app.cleanup_actors_by_role,
        "cleanup_old_vehicles": task_app.cleanup_old_vehicles,
        "spawn_ego_vehicle": task_app.spawn_ego_vehicle,
        "spawn_traffic_vehicles": task_app.spawn_traffic_vehicles,
        "move_uav_to": task_app.move_uav_to,
        "configure_autopilot": task_app.configure_autopilot,
    }
    try:
        task_app.cleanup_actors_by_role = lambda *args, **kwargs: None
        task_app.cleanup_old_vehicles = lambda *args, **kwargs: None
        task_app.spawn_ego_vehicle = lambda *args, **kwargs: _Actor()
        task_app.spawn_traffic_vehicles = lambda *args, **kwargs: []
        task_app.move_uav_to = lambda *args, **kwargs: order.append("move_uav")
        task_app.configure_autopilot = lambda *args, **kwargs: None

        scenario = ScenarioConfig.from_dict(
            {
                "name": "viewer_order",
                "ego_control_mode": "autopilot",
                "uav_enabled": True,
                "candidate_offsets": [],
            }
        )
        app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_viewer_order"))
        app.client = object()
        app.world = _World()
        app.air_client = object()
        app.controller = _Controller()
        app._attach_vehicle_sensors = lambda: order.append("attach")
        app.start_vehicle_viewer = lambda: order.append("viewer")

        app.setup()
    finally:
        for name, value in original.items():
            setattr(task_app, name, value)

    assert order.index("attach") < order.index("viewer") < order.index("takeoff")


def test_active_uav_task_defers_uav_connect_until_after_vehicle_viewer():
    from pathlib import Path

    import carla
    from carlaair_active_world import task_app

    order = []

    class _Map:
        name = "Town10HD"

    class _World:
        def get_map(self):
            return _Map()

        def get_settings(self):
            return type("Settings", (), {"synchronous_mode": False})()

    class _Actor:
        id = 42

        def set_autopilot(self, value):
            pass

        def get_transform(self):
            return carla.Transform(carla.Location(), carla.Rotation())

    class _Controller:
        def __init__(self, *args, **kwargs):
            order.append("controller")
            self.vehicle_name = kwargs.get("vehicle_name", "SimpleFlight")
            self.api_ready = False

        def resolve_vehicle_name(self):
            order.append("resolve_uav")
            return self.vehicle_name

        def takeoff(self):
            order.append("takeoff")

        def hover(self):
            order.append("hover")

    original = {
        "connect_carla": task_app.connect_carla,
        "connect_airsim": task_app.connect_airsim,
        "calibrate_offset": task_app.calibrate_offset,
        "UAVCommandController": task_app.UAVCommandController,
        "cleanup_actors_by_role": task_app.cleanup_actors_by_role,
        "cleanup_old_vehicles": task_app.cleanup_old_vehicles,
        "spawn_ego_vehicle": task_app.spawn_ego_vehicle,
        "spawn_traffic_vehicles": task_app.spawn_traffic_vehicles,
        "move_uav_to": task_app.move_uav_to,
        "configure_autopilot": task_app.configure_autopilot,
    }
    try:
        task_app.connect_carla = lambda *args, **kwargs: (object(), _World())
        task_app.connect_airsim = lambda *args, **kwargs: order.append("connect_uav") or object()
        task_app.calibrate_offset = lambda *args, **kwargs: order.append("calibrate") or (0.0, 0.0, 0.0)
        task_app.UAVCommandController = _Controller
        task_app.cleanup_actors_by_role = lambda *args, **kwargs: None
        task_app.cleanup_old_vehicles = lambda *args, **kwargs: None
        task_app.spawn_ego_vehicle = lambda *args, **kwargs: _Actor()
        task_app.spawn_traffic_vehicles = lambda *args, **kwargs: []
        task_app.move_uav_to = lambda *args, **kwargs: order.append("move_uav")
        task_app.configure_autopilot = lambda *args, **kwargs: None

        scenario = ScenarioConfig.from_dict(
            {
                "name": "viewer_order",
                "map_name": "Town10HD",
                "ego_control_mode": "autopilot",
                "uav_enabled": True,
                "candidate_offsets": [],
            }
        )
        app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_viewer_order"))

        app.connect()
        assert "connect_uav" not in order

        app._attach_vehicle_sensors = lambda: order.append("attach")
        app.start_vehicle_viewer = lambda: order.append("viewer")
        app.setup()
    finally:
        for name, value in original.items():
            setattr(task_app, name, value)

    assert order.index("viewer") < order.index("connect_uav") < order.index("takeoff")


def test_active_uav_task_reuses_ego_driver_sensor_rig_for_viewer():
    from pathlib import Path

    from carlaair_active_world import task_app

    created = []

    class _Actor:
        id = 42

    class _ExistingRig:
        pass

    class _NewRig:
        def __init__(self, *args, **kwargs):
            created.append(("init", args, kwargs))

        def spawn(self):
            created.append(("spawn",))

    original = task_app.VehicleSensorRig
    try:
        task_app.VehicleSensorRig = _NewRig
        scenario = ScenarioConfig.from_dict(
            {
                "name": "reuse_ego_rig",
                "ego_control_mode": "vision_simple",
                "vehicle_sensor_limit": 1,
            }
        )
        app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_reuse_ego_rig"))
        app.world = object()
        app.ego_vehicle = _Actor()
        existing = _ExistingRig()
        app.ego_driver = type("Driver", (), {"sensor_rig": existing})()

        app._attach_vehicle_sensors()
    finally:
        task_app.VehicleSensorRig = original

    assert created == []
    assert app.vehicle_sensors[42] is existing


def test_active_uav_task_cleanup_destroys_reused_ego_sensor_rig_once():
    from pathlib import Path

    from carlaair_active_world import task_app

    class _Rig:
        def __init__(self):
            self.destroy_calls = 0

        def destroy(self):
            self.destroy_calls += 1

    class _Driver:
        def __init__(self, rig):
            self.sensor_rig = rig

        def destroy(self):
            self.sensor_rig.destroy()

    class _Actor:
        def set_autopilot(self, value):
            pass

        def destroy(self):
            pass

    scenario = ScenarioConfig.from_dict({"name": "cleanup_reuse", "uav_enabled": False})
    app = task_app.ActiveUAVTaskApp(scenario=scenario, output_dir=Path("recordings/test_cleanup_reuse"))
    rig = _Rig()
    app.ego_driver = _Driver(rig)
    app.vehicle_sensors[42] = rig
    app.ego_vehicle = _Actor()

    app.cleanup()

    assert rig.destroy_calls == 1
