from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from carlaair_active_world.scenario import ScenarioConfig
from carlaair_active_world.vision_models.safety_gate import (
    VisionSafetyGateConfig,
    compute_attack_pattern_score,
    compute_visibility_score,
    evaluate_vision_safety_gate,
)
from carlaair_active_world.vision_models.tcp_lite_policy import TcpLiteVisionPolicy
from carlaair_active_world.vision_models.uav_bev import CachedUAVBEVProvider, extract_uav_bev_feature

ROOT = Path(__file__).resolve().parents[1]


class _MockTcpModel:
    def predict(self, rgb, speed_mps, command):
        return [[1.0, 0.0], [2.0, 0.5]], [1.5, 0.4, -0.2]


class _ConsistentTcpModel:
    def predict(self, rgb, speed_mps, command):
        return [[1.0, 0.0], [2.0, 0.5]], [0.3, 0.4, 0.0]


class _RaisingTcpModel:
    def predict(self, rgb, speed_mps, command):
        raise RuntimeError("boom")


class _MalformedControlTcpModel:
    def predict(self, rgb, speed_mps, command):
        return [], "bad"


class _StraightTcpModel:
    def predict(self, rgb, speed_mps, command):
        return [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]], [0.0, 0.3, 0.0]


class _AlternatingTcpModel:
    def __init__(self) -> None:
        self.sign = 1.0

    def predict(self, rgb, speed_mps, command):
        self.sign *= -1.0
        y = 4.0 * self.sign
        return [[2.0, y], [4.0, y], [6.0, y], [8.0, y]], [0.0, 0.3, 0.0]


class _FallbackPolicy:
    def __init__(self) -> None:
        self.last_diagnostics = {}

    def predict(self, obs):
        import carla

        self.last_diagnostics = {"lane_confidence": 0.25}
        control = carla.VehicleControl()
        control.steer = -0.5
        control.throttle = 0.3
        return control


class _AgreeingFallbackPolicy:
    def __init__(self) -> None:
        self.last_diagnostics = {}

    def predict(self, obs):
        import carla

        self.last_diagnostics = {"lane_confidence": 0.25}
        control = carla.VehicleControl()
        control.steer = 0.3
        control.throttle = 0.3
        return control


def _checkerboard(width=160, height=90):
    y, x = np.indices((height, width))
    board = ((x // 4 + y // 4) % 2 * 255).astype(np.uint8)
    return np.stack([board, board, board], axis=-1)


def _write_tiny_tcp_lite_dataset(root: Path):
    PIL_Image = pytest.importorskip("PIL.Image")
    image_dir = root / "images"
    image_dir.mkdir()
    for index in range(2):
        rgb = np.full((32, 48, 3), index * 64, dtype=np.uint8)
        PIL_Image.fromarray(rgb).save(image_dir / f"{index:06d}.png")

    samples = [
        {
            "rgb": f"images/{index:06d}.png",
            "speed_mps": float(index),
            "command": "lane_follow",
            "trajectory": [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            "control": {"steer": 0.1, "throttle": 0.2, "brake": 0.0},
        }
        for index in range(2)
    ]
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def _write_single_tcp_lite_sample(root: Path, sample: dict):
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(sample) + "\n")


def test_train_tcp_lite_module_imports_without_torch_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.train_tcp_lite import train_tcp_lite; print(callable(train_tcp_lite))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_train_tcp_lite_help_works_without_torch_dependency():
    result = subprocess.run(
        [sys.executable, "scripts/train_tcp_lite.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Train TCP-Lite" in result.stdout or "--dataset" in result.stdout


def test_vehicle_eval_launcher_uses_car_airsim_mode_without_city_traffic():
    script = ROOT / "scripts" / "launch_carla_vehicle_eval.sh"

    content = script.read_text(encoding="utf-8")

    assert '"SimMode": "Car"' in content
    assert '"VehicleType": "PhysXCar"' in content
    assert "CARLAAIR_DISPLAY" in content
    assert "XAUTHORITY" in content
    assert "-windowed" in content
    assert "CARLA process exited before RPC became ready." in content
    assert "--no-city-traffic" in content


def test_run_active_world_help_exposes_ego_viewer_flag():
    script = ROOT / "scripts" / "run_active_world.py"

    content = script.read_text(encoding="utf-8")

    assert "--viewer" in content
    assert "Show the ego vehicle RGB camera" in content


def test_run_active_world_help_exposes_artifact_dir_flag():
    script = ROOT / "scripts" / "run_active_world.py"

    content = script.read_text(encoding="utf-8")

    assert "--artifact-dir" in content
    assert "Save ego/UAV frames and control diagnostics" in content


def test_tcp_lite_helpers_import_without_carla_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from carlaair_active_world.vision_models.tcp_lite import command_to_index; "
            'print(command_to_index("left"))',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_tcp_lite_scenario_config_round_trips():
    scenario = ScenarioConfig.from_dict(
        {
            "name": "tcp_lite_config",
            "ego_control_mode": "vision_tcp_lite",
            "vision_model_path": "checkpoints/tcp_lite.pt",
            "vision_model_device": "cpu",
            "vision_model_control_mode": "direct",
            "vision_navigation_command": "lane_follow",
            "vision_safety_gate_enabled": False,
            "vision_attack_pattern_gate": True,
            "vision_attack_pattern_threshold": 0.12,
            "vision_low_visibility_gate": True,
            "vision_low_visibility_threshold": 0.09,
            "vision_first_junction_command": "left",
            "vision_junction_command_hold_sec": 4.0,
        }
    )

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == "checkpoints/tcp_lite.pt"
    assert scenario.vision_model_device == "cpu"
    assert scenario.vision_model_control_mode == "direct"
    assert scenario.vision_navigation_command == "lane_follow"
    assert scenario.vision_safety_gate_enabled is False
    assert scenario.vision_attack_pattern_gate is True
    assert scenario.vision_attack_pattern_threshold == 0.12
    assert scenario.vision_low_visibility_gate is True
    assert scenario.vision_low_visibility_threshold == 0.09
    assert scenario.vision_first_junction_command == "left"
    assert scenario.vision_junction_command_hold_sec == 4.0
    assert scenario.to_dict()["vision_model_path"] == "checkpoints/tcp_lite.pt"
    assert scenario.to_dict()["vision_model_device"] == "cpu"
    assert scenario.to_dict()["vision_model_control_mode"] == "direct"
    assert scenario.to_dict()["vision_navigation_command"] == "lane_follow"
    assert scenario.to_dict()["vision_safety_gate_enabled"] is False
    assert scenario.to_dict()["vision_attack_pattern_gate"] is True
    assert scenario.to_dict()["vision_attack_pattern_threshold"] == 0.12
    assert scenario.to_dict()["vision_low_visibility_gate"] is True
    assert scenario.to_dict()["vision_low_visibility_threshold"] == 0.09
    assert scenario.to_dict()["vision_first_junction_command"] == "left"
    assert scenario.to_dict()["vision_junction_command_hold_sec"] == 4.0


def test_tcp_lite_project_scenario_loads():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_tcp_lite.json")

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == "models/tcp_lite_combined_vehicle_traj10_control1_e40.pt"
    assert scenario.vision_model_device == "cpu"
    assert scenario.vision_model_control_mode == "trajectory_model"
    assert scenario.vision_navigation_command == "lane_follow"
    assert scenario.vision_safety_gate_enabled is True
    assert scenario.vision_attack_pattern_gate is False
    assert scenario.vehicle_sensor_limit == 1
    assert scenario.traffic_vehicles == 0
    assert scenario.uav_enabled is False


def test_tcp_lite_yolo_project_scenario_loads():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_tcp_lite_yolo.json")

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == "models/tcp_lite_combined_vehicle_traj10_control1_e40.pt"
    assert scenario.vision_model_control_mode == "trajectory_model"
    assert scenario.vision_detector_model_path == "models/yolo11n.pt"
    assert scenario.vision_detector_confidence == 0.35
    assert scenario.vision_safety_gate_enabled is True
    assert scenario.traffic_vehicles == 0
    assert scenario.uav_enabled is False


def test_tcp_lite_yolo_uav_bev_project_scenario_loads():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_tcp_lite_yolo_uav_bev.json")

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == "models/tcp_lite_combined_vehicle_traj10_control1_e40.pt"
    assert scenario.vision_detector_model_path == "models/yolo11n.pt"
    assert scenario.uav_enabled is True
    assert scenario.uav_control_enabled is False
    assert scenario.uav_bev_fusion_enabled is True
    assert scenario.uav_bev_camera_name == "front_center"
    assert scenario.uav_bev_refresh_hz == 2.0
    assert scenario.to_dict()["uav_bev_fusion_enabled"] is True


def test_tcp_lite_yolo_attack_project_scenarios_load():
    texture = ScenarioConfig.load("configs/scenarios/town10hd_vision_tcp_lite_yolo_texture_attack.json")
    weather = ScenarioConfig.load("configs/scenarios/town10hd_vision_tcp_lite_yolo_weather_attack.json")

    assert texture.ego_control_mode == "vision_tcp_lite"
    assert texture.vision_attack == "texture"
    assert texture.vision_attack_pattern_gate is False
    assert texture.vision_attack_pattern_threshold == 0.50
    assert texture.vision_detector_model_path == "models/yolo11n.pt"
    assert weather.weather_preset == "hard_rain_fog"
    assert weather.vision_attack == "none"
    assert weather.vision_low_visibility_gate is False
    assert weather.vision_low_visibility_threshold == 0.19
    assert weather.vision_detector_model_path == "models/yolo11n.pt"


def test_tcp_lite_policy_brakes_without_model_path():
    policy = TcpLiteVisionPolicy(model_path="")

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.brake == 1.0
    assert control.throttle == 0.0
    assert policy.last_diagnostics["model_ready"] is False
    assert policy.last_diagnostics["reason"] == "missing_model_path"
    assert policy.last_diagnostics["model_path"] == ""


def test_tcp_lite_policy_reports_missing_checkpoint_path():
    policy = TcpLiteVisionPolicy(model_path="missing/checkpoint.pt")

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.brake == 1.0
    assert policy.last_diagnostics["reason"] == "missing_model_path"
    assert policy.last_diagnostics["model_path"] == "missing/checkpoint.pt"


def test_tcp_lite_policy_uses_mock_model_and_clamps_control():
    policy = TcpLiteVisionPolicy(model=_MockTcpModel(), navigation_command="lane_follow", control_mode="direct")

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.steer == 1.0
    assert control.throttle == 0.4
    assert control.brake == 0.0
    assert policy.last_diagnostics["model_ready"] is True
    assert policy.last_diagnostics["command"] == "lane_follow"
    assert policy.last_diagnostics["trajectory"][0] == [1.0, 0.0]


def test_tcp_lite_policy_defaults_to_trajectory_tracking_control():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        target_speed_mps=4.0,
    )

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert 0.0 < control.steer < 0.5
    assert control.throttle > 0.0
    assert control.brake == 0.0
    assert policy.last_diagnostics["control_mode"] == "trajectory"
    assert policy.last_diagnostics["raw_control"] == [0.3, 0.4, 0.0]


def test_tcp_lite_policy_falls_back_when_trajectory_and_control_disagree():
    policy = TcpLiteVisionPolicy(
        model=_MockTcpModel(),
        navigation_command="lane_follow",
        target_speed_mps=4.0,
    )

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 0.1})

    assert control.throttle > 0.0
    assert policy.last_diagnostics["reason"] == "fallback_confidence_gate"
    assert policy.last_diagnostics["fallback"]["reason"] == "model_trajectory_control_disagreement"


def test_tcp_lite_policy_falls_back_when_rgb_lane_reference_disagrees():
    policy = TcpLiteVisionPolicy(model=_ConsistentTcpModel(), navigation_command="lane_follow")
    policy.fallback_policy = _FallbackPolicy()

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.steer == -0.5
    assert control.throttle == 0.3
    assert policy.last_diagnostics["reason"] == "fallback_confidence_gate"
    assert policy.last_diagnostics["fallback"]["reason"] == "rgb_lane_reference_available"


def test_tcp_lite_policy_prefers_confident_rgb_lane_reference():
    policy = TcpLiteVisionPolicy(model=_ConsistentTcpModel(), navigation_command="lane_follow")
    policy.fallback_policy = _AgreeingFallbackPolicy()

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.steer == 0.3
    assert control.throttle == 0.3
    assert policy.last_diagnostics["reason"] == "fallback_confidence_gate"
    assert policy.last_diagnostics["fallback"]["reason"] == "rgb_lane_reference_available"
    assert policy.last_diagnostics["raw_control"] is None
    assert policy.last_diagnostics["safety_gate"]["reason"] == "rgb_reference_shortcut"


def test_tcp_lite_policy_trajectory_model_mode_uses_rgb_reference_fallback_outside_junction():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
    )
    policy.fallback_policy = _FallbackPolicy()

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "in_junction": False,
        }
    )

    assert control.steer == -0.5
    assert control.throttle == 0.3
    assert policy.last_diagnostics["reason"] == "fallback_confidence_gate"
    assert policy.last_diagnostics["control_mode"] == "trajectory"
    assert policy.last_diagnostics["fallback"]["reason"] == "rgb_lane_reference_available"


def test_tcp_lite_policy_keeps_model_turn_control_inside_junction():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="right",
        control_mode="trajectory_model",
    )
    policy.fallback_policy = _FallbackPolicy()

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "in_junction": True,
        }
    )

    assert control.steer != -0.5
    assert policy.last_diagnostics["reason"] == "ok"
    assert policy.last_diagnostics["control_mode"] == "trajectory"
    assert policy.last_diagnostics["raw_control"] == [0.3, 0.4, 0.0]


def test_tcp_lite_policy_uses_junction_route_reference_for_turn_target():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="right",
        control_mode="trajectory_model",
        target_speed_mps=2.5,
    )
    policy.fallback_policy = _FallbackPolicy()

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "in_junction": True,
            "route_target_source": "junction_turn_reference",
            "route_target_local_x": 10.0,
            "route_target_local_y": 2.0,
        }
    )

    assert 0.0 < control.steer <= 0.25
    assert control.throttle > 0.0
    assert policy.last_diagnostics["reason"] == "fallback_confidence_gate"
    assert policy.last_diagnostics["fallback"]["reason"] == "route_reference_available"
    assert policy.last_diagnostics["fallback"]["diagnostics"]["route_target_source"] == "junction_turn_reference"
    assert policy.last_diagnostics["raw_control"] is None


def test_tcp_lite_policy_uses_waypoint_reference_for_lane_follow_inside_junction():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        target_speed_mps=2.5,
    )
    policy.fallback_policy = _FallbackPolicy()

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "navigation_command": "lane_follow",
            "in_junction": True,
            "route_target_source": "waypoint_next",
            "route_target_local_x": 10.0,
            "route_target_local_y": 1.0,
        }
    )

    assert control.steer > 0.0
    assert policy.last_diagnostics["reason"] == "fallback_confidence_gate"
    assert policy.last_diagnostics["fallback"]["diagnostics"]["route_target_source"] == "waypoint_next"


def test_tcp_lite_policy_prefers_route_reference_fallback_outside_junction():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        target_speed_mps=2.5,
    )
    policy.fallback_policy = _FallbackPolicy()

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "in_junction": False,
            "route_target_local_x": 10.0,
            "route_target_local_y": 2.0,
        }
    )

    assert 0.0 < control.steer <= 0.25
    assert control.throttle > 0.0
    assert policy.last_diagnostics["reason"] == "fallback_confidence_gate"
    assert policy.last_diagnostics["fallback"]["reason"] == "route_reference_available"


def test_tcp_lite_policy_route_reference_applies_lane_centering():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        target_speed_mps=2.5,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "in_junction": False,
            "route_target_local_x": 10.0,
            "route_target_local_y": 0.0,
            "lane_center_offset_m": 1.0,
        }
    )

    assert control.steer < 0.0
    assert policy.last_diagnostics["fallback"]["diagnostics"]["lane_centering_correction"] < 0.0


def test_tcp_lite_policy_route_reference_applies_uav_residual():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        target_speed_mps=2.5,
        uav_bev_fusion_enabled=True,
        uav_bev_steer_gain=0.10,
        uav_bev_max_steer_correction=0.10,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "in_junction": False,
            "route_target_local_x": 10.0,
            "route_target_local_y": 0.0,
            "uav_bev": {
                "available": True,
                "road_confidence": 0.8,
                "center_bias": 1.0,
            },
        }
    )

    assert control.steer > 0.0
    assert policy.last_diagnostics["uav_bev_fusion"]["applied"] is True


def test_tcp_lite_policy_slows_for_forward_interaction_hazard():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        target_speed_mps=3.0,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 2.4,
            "in_junction": False,
            "route_target_local_x": 12.0,
            "route_target_local_y": 0.0,
            "interaction_hazard": {
                "active": True,
                "action": "slow",
                "target_speed_mps": 1.0,
                "distance_m": 14.0,
                "actor_type": "vehicle",
            },
        }
    )

    diagnostics = policy.last_diagnostics["fallback"]["diagnostics"]
    assert control.throttle == 0.0
    assert control.brake > 0.0
    assert diagnostics["target_speed_mps"] == 1.0
    assert diagnostics["interaction_yield"]["action"] == "slow"


def test_tcp_lite_policy_applies_throttle_for_slow_hazard_from_stop():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        target_speed_mps=3.0,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 0.0,
            "in_junction": False,
            "route_target_local_x": 12.0,
            "route_target_local_y": 0.0,
            "interaction_hazard": {
                "active": True,
                "action": "slow",
                "target_speed_mps": 1.0,
                "distance_m": 14.0,
                "actor_type": "walker",
            },
        }
    )

    diagnostics = policy.last_diagnostics["fallback"]["diagnostics"]
    assert control.throttle > 0.0
    assert control.brake == 0.0
    assert diagnostics["interaction_yield"]["action"] == "slow"


def test_tcp_lite_policy_brakes_for_close_pedestrian_hazard():
    policy = TcpLiteVisionPolicy(
        model=_ConsistentTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        target_speed_mps=3.0,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.6,
            "in_junction": False,
            "route_target_local_x": 12.0,
            "route_target_local_y": 0.0,
            "interaction_hazard": {
                "active": True,
                "action": "stop",
                "target_speed_mps": 0.0,
                "distance_m": 4.0,
                "actor_type": "walker",
            },
        }
    )

    diagnostics = policy.last_diagnostics["fallback"]["diagnostics"]
    assert control.throttle == 0.0
    assert control.brake >= 0.5
    assert diagnostics["target_speed_mps"] == 0.0
    assert diagnostics["interaction_yield"]["action"] == "stop"


def test_tcp_lite_policy_applies_lane_centering_correction():
    policy = TcpLiteVisionPolicy(
        model=_StraightTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "lane_center_offset_m": -1.0,
            "in_junction": False,
        }
    )

    assert control.steer > 0.0
    assert policy.last_diagnostics["stabilization"]["lane_centering_correction"] > 0.0


def test_tcp_lite_policy_fuses_uav_bev_center_bias():
    policy = TcpLiteVisionPolicy(
        model=_StraightTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        uav_bev_fusion_enabled=True,
        uav_bev_steer_gain=0.10,
        uav_bev_max_steer_correction=0.10,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "lane_center_offset_m": 0.0,
            "in_junction": False,
            "uav_bev": {
                "available": True,
                "road_confidence": 0.8,
                "center_bias": 1.0,
            },
        }
    )

    assert control.steer > 0.02
    assert policy.last_diagnostics["uav_bev_fusion"]["enabled"] is True
    assert policy.last_diagnostics["uav_bev_fusion"]["applied"] is True
    assert policy.last_diagnostics["uav_bev_fusion"]["steer_correction"] > 0.0


def test_tcp_lite_policy_ignores_uav_bev_when_disabled():
    policy = TcpLiteVisionPolicy(
        model=_StraightTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        uav_bev_fusion_enabled=False,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "lane_center_offset_m": 0.0,
            "in_junction": False,
            "uav_bev": {
                "available": True,
                "road_confidence": 0.8,
                "center_bias": 1.0,
            },
        }
    )

    assert abs(control.steer) < 0.02
    assert policy.last_diagnostics["uav_bev_fusion"]["enabled"] is False


def test_tcp_lite_policy_ignores_uav_bev_in_none_fusion_mode():
    policy = TcpLiteVisionPolicy(
        model=_StraightTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        uav_bev_fusion_enabled=True,
        uav_fusion_mode="none",
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "lane_center_offset_m": 0.0,
            "in_junction": False,
            "uav_bev": {
                "available": True,
                "road_confidence": 0.8,
                "center_bias": 1.0,
            },
        }
    )

    assert abs(control.steer) < 0.02
    assert policy.last_diagnostics["uav_bev_fusion"]["mode"] == "none"
    assert policy.last_diagnostics["uav_bev_fusion"]["applied"] is False


def test_tcp_lite_policy_uses_learned_uav_fusion_residual(tmp_path):
    planner_path = tmp_path / "fusion.json"
    planner_path.write_text(
        json.dumps(
            {
                "weights": {
                    "center_bias": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    policy = TcpLiteVisionPolicy(
        model=_StraightTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
        uav_bev_fusion_enabled=True,
        uav_fusion_mode="learned",
        uav_fusion_planner_path=str(planner_path),
        uav_fusion_planner_gain=0.5,
        uav_fusion_max_steer_correction=0.08,
        uav_fusion_min_confidence=0.1,
    )

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "lane_center_offset_m": 0.0,
            "in_junction": False,
            "uav_bev": {
                "available": True,
                "road_confidence": 0.8,
                "center_bias": 1.0,
                "forward_density": 0.5,
                "left_right_balance": 0.0,
            },
        }
    )

    assert control.steer > 0.02
    assert policy.last_diagnostics["uav_bev_fusion"]["mode"] == "learned"
    assert policy.last_diagnostics["uav_bev_fusion"]["applied"] is True
    assert policy.last_diagnostics["uav_bev_fusion"]["steer_correction"] == 0.08


def test_tcp_lite_policy_rate_limits_alternating_junction_steer():
    policy = TcpLiteVisionPolicy(
        model=_AlternatingTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
    )
    obs = {"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0, "in_junction": True}

    first = policy.predict(obs)
    second = policy.predict(obs)

    assert abs(second.steer - first.steer) <= 0.091
    assert abs(second.steer) <= 0.34
    assert policy.last_diagnostics["stabilization"]["in_junction"] is True


def test_tcp_lite_policy_rate_limits_first_steer_from_rest():
    policy = TcpLiteVisionPolicy(
        model=_AlternatingTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
    )
    obs = {"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 0.0, "in_junction": False}

    control = policy.predict(obs)

    assert abs(control.steer) <= 0.061
    assert abs(policy.last_diagnostics["stabilization"]["target_steer"]) > 0.30


def test_tcp_lite_policy_recovers_low_speed_in_junction():
    policy = TcpLiteVisionPolicy(
        model=_StraightTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
    )
    obs = {"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 0.02, "in_junction": True}

    control = policy.predict(obs)

    assert control.throttle >= 0.32
    assert control.brake == 0.0
    assert policy.last_diagnostics["stabilization"]["junction_low_speed_recovery"] is True


def test_tcp_lite_policy_keeps_junction_throttle_cap_at_normal_speed():
    policy = TcpLiteVisionPolicy(
        model=_StraightTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
    )
    obs = {"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0, "in_junction": True}

    control = policy.predict(obs)

    assert control.throttle <= 0.24
    assert policy.last_diagnostics["stabilization"]["junction_low_speed_recovery"] is False


def test_tcp_lite_policy_limits_lane_follow_steer_outside_junction():
    policy = TcpLiteVisionPolicy(
        model=_AlternatingTcpModel(),
        navigation_command="lane_follow",
        control_mode="trajectory_model",
    )
    obs = {"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0, "in_junction": False}

    first = policy.predict(obs)
    second = policy.predict(obs)

    assert abs(first.steer) <= 0.38
    assert abs(second.steer - first.steer) <= 0.061
    assert abs(second.steer) <= 0.38


def test_tcp_lite_policy_brakes_when_model_predict_raises():
    policy = TcpLiteVisionPolicy(model=_RaisingTcpModel(), navigation_command="lane_follow")

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.brake == 1.0
    assert control.throttle == 0.0
    assert "RuntimeError" in policy.last_diagnostics["reason"] or "boom" in policy.last_diagnostics["reason"]
    assert policy.last_diagnostics["model_ready"] is True


def test_tcp_lite_policy_brakes_when_model_control_is_malformed():
    policy = TcpLiteVisionPolicy(model=_MalformedControlTcpModel(), navigation_command="lane_follow")

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.brake == 1.0
    assert control.throttle == 0.0
    reason = policy.last_diagnostics["reason"]
    assert "ValueError" in reason or "invalid control" in reason
    assert policy.last_diagnostics["model_ready"] is True


def test_tcp_lite_policy_safety_gate_brakes_for_obstacle():
    policy = TcpLiteVisionPolicy(model=_MockTcpModel(), safety_gate_enabled=True)

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 1.0,
            "vision_detector": {"obstacle": True, "label": "car"},
        }
    )

    assert control.brake == 1.0
    assert control.throttle == 0.0
    assert policy.last_diagnostics["safety_gate"]["blocked"] is True
    assert policy.last_diagnostics["reason"] == "vision_obstacle"


def test_safety_gate_blocks_detector_obstacle():
    result = evaluate_vision_safety_gate(
        np.zeros((90, 160, 3), dtype=np.uint8),
        {"obstacle": True, "label": "car"},
        VisionSafetyGateConfig(enabled=True),
    )

    assert result["blocked"] is True
    assert result["reason"] == "vision_obstacle"
    assert result["detector_obstacle"] is True


def test_safety_gate_does_not_block_when_disabled():
    result = evaluate_vision_safety_gate(
        np.zeros((90, 160, 3), dtype=np.uint8),
        {"obstacle": True},
        VisionSafetyGateConfig(enabled=False),
    )

    assert result["blocked"] is False
    assert result["reason"] == "disabled"


def test_extract_uav_bev_feature_reports_center_bias_from_road_region():
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[:, 60:, :] = 90

    feature = extract_uav_bev_feature(rgb)

    assert feature["available"] is True
    assert feature["road_confidence"] > 0.4
    assert feature["center_bias"] > 0.25
    assert feature["feature_dim"] == 4


def test_cached_uav_bev_provider_reuses_recent_feature():
    rgb = np.full((20, 30, 3), 80, dtype=np.uint8)

    class _Rig:
        def __init__(self) -> None:
            self.calls = 0

        def snapshot(self):
            self.calls += 1
            return {"rgb": rgb}

    rig = _Rig()
    now = [0.0]
    provider = CachedUAVBEVProvider(lambda: rig, refresh_hz=2.0, clock=lambda: now[0])

    first = provider.snapshot()
    second = provider.snapshot()
    now[0] = 0.6
    third = provider.snapshot()

    assert first["available"] is True
    assert second["available"] is True
    assert third["available"] is True
    assert rig.calls == 2


def test_attack_pattern_score_is_higher_for_repeated_high_contrast_texture():
    clean_score = compute_attack_pattern_score(np.zeros((90, 160, 3), dtype=np.uint8))
    noisy_score = compute_attack_pattern_score(_checkerboard())

    assert noisy_score > clean_score + 0.2


def test_attack_pattern_score_detects_project_texture_attack():
    from carlaair_active_world.adversarial import apply_vision_attack

    clean = np.full((90, 160, 3), 35, dtype=np.uint8)
    attacked = apply_vision_attack({"rgb": clean}, attack="texture", intensity=1.5)["rgb"]

    assert compute_attack_pattern_score(clean) < 0.04
    assert compute_attack_pattern_score(attacked) > 0.08


def test_attack_pattern_score_returns_zero_for_invalid_rgb():
    assert compute_attack_pattern_score([[[1], [2, 3]]]) == 0.0


def test_attack_pattern_gate_blocks_only_when_enabled():
    rgb = _checkerboard()
    disabled_gate = evaluate_vision_safety_gate(
        rgb,
        {},
        VisionSafetyGateConfig(attack_pattern_gate=False, attack_pattern_threshold=0.2),
    )
    enabled_gate = evaluate_vision_safety_gate(
        rgb,
        {},
        VisionSafetyGateConfig(attack_pattern_gate=True, attack_pattern_threshold=0.2),
    )

    assert disabled_gate["blocked"] is False
    assert disabled_gate["attack_pattern_score"] > 0.2
    assert enabled_gate["blocked"] is True
    assert enabled_gate["reason"] == "attack_pattern"


def test_low_visibility_gate_blocks_flat_foggy_frame():
    clean = _checkerboard()
    foggy = np.full((90, 160, 3), 115, dtype=np.uint8)

    assert compute_visibility_score(clean) > compute_visibility_score(foggy)

    result = evaluate_vision_safety_gate(
        foggy,
        {},
        VisionSafetyGateConfig(low_visibility_gate=True, low_visibility_threshold=0.12),
    )

    assert result["blocked"] is True
    assert result["reason"] == "low_visibility"


def test_command_to_index_accepts_known_and_unknown_commands():
    from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, command_to_index

    assert command_to_index("lane_follow") == COMMAND_TO_INDEX["lane_follow"]
    assert command_to_index("left") == COMMAND_TO_INDEX["left"]
    assert command_to_index("unknown") == COMMAND_TO_INDEX["lane_follow"]


def test_tcp_lite_model_outputs_trajectory_and_control_shapes():
    torch = pytest.importorskip("torch")
    from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, TcpLiteModel

    model = TcpLiteModel(
        image_channels=3,
        command_count=len(COMMAND_TO_INDEX),
        trajectory_points=4,
    )
    rgb = torch.zeros((2, 3, 96, 160))
    speed = torch.zeros((2, 1))
    command = torch.zeros((2,), dtype=torch.long)

    output = model(rgb, speed, command)

    assert output["trajectory"].shape == (2, 4, 2)
    assert output["control"].shape == (2, 3)


def test_tcp_lite_model_keeps_spatial_image_features():
    torch = pytest.importorskip("torch")
    from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, TcpLiteModel

    model = TcpLiteModel(
        image_channels=3,
        command_count=len(COMMAND_TO_INDEX),
        trajectory_points=4,
    )

    features = model.image_encoder(torch.zeros((1, 3, 96, 160)))

    assert features.shape == (1, 64 * 4 * 4)


def test_tcp_lite_dataset_reads_jsonl_samples(tmp_path):
    pytest.importorskip("torch")
    _write_tiny_tcp_lite_dataset(tmp_path)
    from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset

    dataset = TcpLiteImitationDataset(tmp_path, image_size=(32, 48), trajectory_points=4)
    item = dataset[0]

    assert len(dataset) == 2
    assert item["rgb"].shape == (3, 32, 48)
    assert item["speed"].shape == (1,)
    assert item["trajectory"].shape == (4, 2)
    assert item["control"].shape == (3,)


def test_tcp_lite_dataset_rejects_rgb_path_outside_root(tmp_path):
    pytest.importorskip("torch")
    PIL_Image = pytest.importorskip("PIL.Image")
    outside_path = tmp_path.parent / "outside.png"
    PIL_Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(outside_path)
    _write_single_tcp_lite_sample(
        tmp_path,
        {
            "rgb": "../outside.png",
            "speed_mps": 0.0,
            "command": "lane_follow",
            "trajectory": [[0.0, 0.0]],
            "control": {"steer": 0.0, "throttle": 0.0, "brake": 0.0},
        },
    )
    from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset

    dataset = TcpLiteImitationDataset(tmp_path)

    with pytest.raises(ValueError, match="outside dataset root"):
        dataset[0]


@pytest.mark.parametrize(
    "trajectory",
    [
        [1.0, 2.0, 3.0],
        [[0.0, 0.0], [1.0]],
        [[0.0, "bad"]],
    ],
)
def test_tcp_lite_dataset_rejects_malformed_trajectory(tmp_path, trajectory):
    pytest.importorskip("torch")
    PIL_Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "image.png"
    PIL_Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
    _write_single_tcp_lite_sample(
        tmp_path,
        {
            "rgb": "image.png",
            "speed_mps": 0.0,
            "command": "lane_follow",
            "trajectory": trajectory,
            "control": {"steer": 0.0, "throttle": 0.0, "brake": 0.0},
        },
    )
    from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset

    dataset = TcpLiteImitationDataset(tmp_path)

    with pytest.raises(ValueError, match="trajectory must be a sequence of \\[x, y\\] points"):
        dataset[0]


def test_train_tcp_lite_saves_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    _write_tiny_tcp_lite_dataset(tmp_path)
    from scripts.train_tcp_lite import train_tcp_lite

    output_path = tmp_path / "tcp_lite.pt"
    train_tcp_lite(
        dataset_root=tmp_path,
        output_path=output_path,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        image_height=32,
        image_width=48,
        trajectory_points=4,
        trajectory_loss_weight=0.5,
        control_loss_weight=3.0,
        trajectory_smoothness_loss_weight=0.2,
        straight_lateral_loss_weight=0.05,
    )

    checkpoint = torch.load(output_path, map_location="cpu")
    assert output_path.exists()
    assert "model_state_dict" in checkpoint
    assert checkpoint["image_size"] == [32, 48]
    assert checkpoint["trajectory_points"] == 4
    assert checkpoint["trajectory_loss_weight"] == 0.5
    assert checkpoint["control_loss_weight"] == 3.0
    assert checkpoint["trajectory_smoothness_loss_weight"] == 0.2
    assert checkpoint["straight_lateral_loss_weight"] == 0.05


def test_vision_driver_forwards_navigation_command_to_policy(monkeypatch):
    from carlaair_active_world.vision_driver import VisionEgoDriver

    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": np.zeros((90, 160, 3), dtype=np.uint8)}

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
    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)

    driver = VisionEgoDriver(object(), _Vehicle(), policy=policy, navigation_command="left")
    driver.predict()

    assert policy.obs["navigation_command"] == "left"
    assert driver.last_observation["navigation_command"] == "left"
    assert driver.last_observation["rgb"].shape == (90, 160, 3)


def test_vision_driver_applies_first_junction_turn_command_once(monkeypatch):
    import carla
    from carlaair_active_world.vision_driver import VisionEgoDriver

    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": np.zeros((90, 160, 3), dtype=np.uint8)}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.commands = []
            self.in_junction = []
            self.last_diagnostics = {}

        def predict(self, obs):
            self.commands.append(obs["navigation_command"])
            self.in_junction.append(obs.get("in_junction"))
            import carla

            return carla.VehicleControl()

    class _Waypoint:
        def __init__(self, is_junction: bool) -> None:
            self.is_junction = is_junction
            self.lane_width = 3.5
            self.road_id = 1
            self.lane_id = 1
            self.transform = carla.Transform(carla.Location(), carla.Rotation())

    class _Map:
        def __init__(self) -> None:
            self.values = [False, True, True, False]
            self.calls = 0

        def get_waypoint(self, *_args, **_kwargs):
            value = self.values[min(self.calls, len(self.values) - 1)]
            self.calls += 1
            return _Waypoint(value)

    class _World:
        def __init__(self) -> None:
            self.map = _Map()

        def get_map(self):
            return self.map

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

        def get_location(self):
            return carla.Location()

    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)
    policy = _Policy()
    now = [0.0]
    driver = VisionEgoDriver(
        _World(),
        _Vehicle(),
        policy=policy,
        navigation_command="lane_follow",
        first_junction_command="left",
        junction_command_hold_sec=4.0,
        clock=lambda: now[0],
    )

    driver.predict()
    now[0] = 1.0
    driver.predict()
    now[0] = 2.0
    driver.predict()
    now[0] = 6.0
    driver.predict()

    assert policy.in_junction == [False, True, True, False]
    assert policy.commands == ["lane_follow", "left", "left", "lane_follow"]


def test_vision_driver_caps_first_junction_command_hold(monkeypatch):
    import carla
    from carlaair_active_world.vision_driver import VisionEgoDriver

    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": np.zeros((90, 160, 3), dtype=np.uint8)}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.commands = []
            self.last_diagnostics = {}

        def predict(self, obs):
            self.commands.append(obs["navigation_command"])
            return carla.VehicleControl()

    class _Waypoint:
        def __init__(self, is_junction: bool) -> None:
            self.is_junction = is_junction
            self.lane_width = 3.5
            self.road_id = 1
            self.lane_id = 1
            self.transform = carla.Transform(carla.Location(), carla.Rotation())

    class _Map:
        def __init__(self) -> None:
            self.values = [False, True, True, True, False]
            self.calls = 0

        def get_waypoint(self, *_args, **_kwargs):
            value = self.values[min(self.calls, len(self.values) - 1)]
            self.calls += 1
            return _Waypoint(value)

    class _World:
        def __init__(self) -> None:
            self.map = _Map()

        def get_map(self):
            return self.map

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

        def get_location(self):
            return carla.Location()

    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)
    now = [0.0]
    policy = _Policy()
    driver = VisionEgoDriver(
        _World(),
        _Vehicle(),
        policy=policy,
        navigation_command="lane_follow",
        first_junction_command="right",
        junction_command_hold_sec=4.0,
        clock=lambda: now[0],
    )

    for value in [0.0, 1.0, 6.0, 10.0, 11.0]:
        now[0] = value
        driver.predict()

    assert policy.commands == ["lane_follow", "right", "lane_follow", "lane_follow", "lane_follow"]


def test_vision_driver_adds_junction_turn_route_reference(monkeypatch):
    import carla
    from carlaair_active_world.vision_driver import VisionEgoDriver

    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": np.zeros((90, 160, 3), dtype=np.uint8)}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.obs = {}
            self.last_diagnostics = {}

        def predict(self, obs):
            self.obs = dict(obs)
            return carla.VehicleControl()

    class _Junction:
        def get_waypoints(self, _lane_type):
            entry = carla.Transform(carla.Location(x=-28.5, y=130.0), carla.Rotation(yaw=-180.0))
            straight_exit = carla.Transform(carla.Location(x=-70.0, y=129.0), carla.Rotation(yaw=-170.0))
            right_exit = carla.Transform(carla.Location(x=-45.0, y=115.0), carla.Rotation(yaw=-90.0))
            return [
                (type("Waypoint", (), {"transform": entry})(), type("Waypoint", (), {"transform": straight_exit})()),
                (type("Waypoint", (), {"transform": entry})(), type("Waypoint", (), {"transform": right_exit})()),
            ]

    class _Waypoint:
        is_junction = True
        lane_width = 3.5
        road_id = 515
        lane_id = -2
        transform = carla.Transform(carla.Location(x=-29.0, y=130.0), carla.Rotation(yaw=-180.0))

        def next(self, _distance):
            return []

        def get_junction(self):
            return _Junction()

    class _Map:
        def get_waypoint(self, *_args, **_kwargs):
            return _Waypoint()

    class _World:
        def get_map(self):
            return _Map()

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

        def get_location(self):
            return carla.Location(x=-29.0, y=130.0)

        def get_transform(self):
            return carla.Transform(carla.Location(x=-29.0, y=130.0), carla.Rotation(yaw=-180.0))

    monkeypatch.setattr(VisionEgoDriver, "sensor_rig_class", _Rig)
    policy = _Policy()
    driver = VisionEgoDriver(
        _World(),
        _Vehicle(),
        policy=policy,
        navigation_command="lane_follow",
        first_junction_command="right",
        junction_command_hold_sec=4.0,
        clock=lambda: 1.0,
    )

    driver.predict()

    assert policy.obs["navigation_command"] == "right"
    assert policy.obs["route_target_source"] == "junction_turn_reference"
    assert policy.obs["route_target_local_x"] > 10.0
    assert policy.obs["route_target_local_y"] > 10.0


def test_env_uses_tcp_lite_policy_for_tcp_lite_mode(monkeypatch):
    from carlaair_active_world import env

    captured = {}

    class _Policy:
        def __init__(self, **kwargs) -> None:
            captured["policy_kwargs"] = kwargs

    class _Driver:
        def __init__(self, world, ego_vehicle, **kwargs) -> None:
            captured["driver_policy"] = kwargs.get("policy")
            captured["driver_navigation_command"] = kwargs.get("navigation_command")
            captured["driver_first_junction_command"] = kwargs.get("first_junction_command")
            captured["driver_junction_command_hold_sec"] = kwargs.get("junction_command_hold_sec")
            captured["driver_junction_command_hold_until_exit"] = kwargs.get("junction_command_hold_until_exit")
            captured["driver_use_depth"] = kwargs.get("use_depth")

        def predict(self, *args, **kwargs):
            import carla

            return carla.VehicleControl()

    class _Vehicle:
        def __init__(self) -> None:
            self.autopilot = None

        def set_autopilot(self, value):
            self.autopilot = value

    monkeypatch.setattr(env, "TcpLiteVisionPolicy", _Policy)
    monkeypatch.setattr(env, "VisionEgoDriver", _Driver)

    scenario = ScenarioConfig.from_dict(
        {
            "name": "tcp_lite_env",
            "ego_control_mode": "vision_tcp_lite",
            "vision_model_path": "checkpoints/tcp.pt",
            "vision_model_device": "cpu",
            "vision_model_control_mode": "direct",
            "vision_navigation_command": "right",
            "vision_first_junction_command": "left",
            "vision_junction_command_hold_sec": 4.0,
            "vision_junction_command_hold_until_exit": True,
            "ego_target_speed_mps": 3.5,
        }
    )
    app = env.ActiveAirGroundEnv(scenario=scenario)
    app.world = object()
    app.ego_vehicle = _Vehicle()

    app._start_ego_control()
    app._closed = True
    if app._ego_driver_thread is not None:
        app._ego_driver_thread.join(timeout=1.0)

    assert app.ego_vehicle.autopilot is False
    assert captured["policy_kwargs"]["model_path"] == "checkpoints/tcp.pt"
    assert captured["policy_kwargs"]["control_mode"] == "direct"
    assert captured["policy_kwargs"]["navigation_command"] == "right"
    assert captured["policy_kwargs"]["target_speed_mps"] == 3.5
    assert captured["policy_kwargs"]["uav_fusion_mode"] == "none"
    assert captured["driver_policy"] is not None
    assert captured["driver_navigation_command"] == "right"
    assert captured["driver_first_junction_command"] == "left"
    assert captured["driver_junction_command_hold_sec"] == 4.0
    assert captured["driver_junction_command_hold_until_exit"] is True
    assert captured["driver_use_depth"] is False


def test_task_app_uses_tcp_lite_policy_for_tcp_lite_mode(monkeypatch):
    import carla
    from carlaair_active_world import task_app

    captured = {"configure_autopilot_calls": 0, "move_uav_calls": 0}

    class _Actor:
        id = 42

        def __init__(self) -> None:
            self.autopilot = None

        def set_autopilot(self, value):
            self.autopilot = value

        def get_transform(self):
            return carla.Transform(carla.Location(), carla.Rotation())

    class _World:
        def get_settings(self):
            return type("Settings", (), {"synchronous_mode": False})()

    class _Policy:
        def __init__(self, **kwargs) -> None:
            captured["policy"] = self
            captured["policy_kwargs"] = kwargs

    class _Driver:
        def __init__(self, world, ego_vehicle, **kwargs) -> None:
            captured["driver_policy"] = kwargs.get("policy")
            captured["driver_navigation_command"] = kwargs.get("navigation_command")
            captured["driver_first_junction_command"] = kwargs.get("first_junction_command")
            captured["driver_junction_command_hold_sec"] = kwargs.get("junction_command_hold_sec")
            captured["driver_junction_command_hold_until_exit"] = kwargs.get("junction_command_hold_until_exit")
            captured["driver_use_depth"] = kwargs.get("use_depth")

    actor = _Actor()
    monkeypatch.setattr(task_app, "TcpLiteVisionPolicy", _Policy)
    monkeypatch.setattr(task_app, "VisionEgoDriver", _Driver)
    monkeypatch.setattr(task_app, "cleanup_actors_by_role", lambda *args, **kwargs: None)
    monkeypatch.setattr(task_app, "cleanup_old_vehicles", lambda *args, **kwargs: None)
    monkeypatch.setattr(task_app, "spawn_ego_vehicle", lambda *args, **kwargs: actor)
    monkeypatch.setattr(task_app, "spawn_traffic_vehicles", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        task_app,
        "configure_autopilot",
        lambda *args, **kwargs: captured.__setitem__(
            "configure_autopilot_calls",
            captured["configure_autopilot_calls"] + 1,
        ),
    )
    monkeypatch.setattr(
        task_app,
        "move_uav_to",
        lambda *args, **kwargs: captured.__setitem__("move_uav_calls", captured["move_uav_calls"] + 1),
    )

    scenario = ScenarioConfig.from_dict(
        {
            "name": "tcp_lite_task_app",
            "ego_control_mode": "vision_tcp_lite",
            "uav_enabled": False,
            "vision_model_path": "checkpoints/tcp.pt",
            "vision_model_device": "cpu",
            "vision_model_control_mode": "direct",
            "vision_navigation_command": "left",
            "vision_first_junction_command": "right",
            "vision_junction_command_hold_sec": 3.5,
            "vision_junction_command_hold_until_exit": True,
            "vision_safety_gate_enabled": False,
            "vision_attack_pattern_gate": True,
            "ego_target_speed_mps": 3.25,
        }
    )
    app = task_app.ActiveUAVTaskApp(scenario, output_dir=Path("recordings/test_tcp_lite_task_app"))
    app.client = object()
    app.world = _World()
    app.start_ego_driver = lambda: None
    app._attach_vehicle_sensors = lambda: None
    app.start_vehicle_viewer = lambda: None

    app.setup()

    assert actor.autopilot is False
    assert captured["policy_kwargs"]["model_path"] == "checkpoints/tcp.pt"
    assert captured["policy_kwargs"]["device"] == "cpu"
    assert captured["policy_kwargs"]["control_mode"] == "direct"
    assert captured["policy_kwargs"]["navigation_command"] == "left"
    assert captured["policy_kwargs"]["safety_gate_enabled"] is False
    assert captured["policy_kwargs"]["attack_pattern_gate"] is True
    assert captured["driver_use_depth"] is False
    assert captured["policy_kwargs"]["target_speed_mps"] == 3.25
    assert captured["policy_kwargs"]["uav_fusion_mode"] == "none"
    assert captured["driver_policy"] is captured["policy"]
    assert captured["driver_navigation_command"] == "left"
    assert captured["driver_first_junction_command"] == "right"
    assert captured["driver_junction_command_hold_sec"] == 3.5
    assert captured["driver_junction_command_hold_until_exit"] is True
    assert captured["configure_autopilot_calls"] == 0
    assert captured["move_uav_calls"] == 0
