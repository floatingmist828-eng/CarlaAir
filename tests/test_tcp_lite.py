from __future__ import annotations

import numpy as np
import pytest

from carlaair_active_world.scenario import ScenarioConfig
from carlaair_active_world.vision_models.safety_gate import (
    VisionSafetyGateConfig,
    compute_attack_pattern_score,
    evaluate_vision_safety_gate,
)


def _checkerboard(width=160, height=90):
    y, x = np.indices((height, width))
    board = ((x // 4 + y // 4) % 2 * 255).astype(np.uint8)
    return np.stack([board, board, board], axis=-1)


def test_tcp_lite_scenario_config_round_trips():
    scenario = ScenarioConfig.from_dict(
        {
            "name": "tcp_lite_config",
            "ego_control_mode": "vision_tcp_lite",
            "vision_model_path": "checkpoints/tcp_lite.pt",
            "vision_model_device": "cpu",
            "vision_navigation_command": "lane_follow",
            "vision_safety_gate_enabled": False,
            "vision_attack_pattern_gate": True,
        }
    )

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == "checkpoints/tcp_lite.pt"
    assert scenario.vision_model_device == "cpu"
    assert scenario.vision_navigation_command == "lane_follow"
    assert scenario.vision_safety_gate_enabled is False
    assert scenario.vision_attack_pattern_gate is True
    assert scenario.to_dict()["vision_model_path"] == "checkpoints/tcp_lite.pt"
    assert scenario.to_dict()["vision_model_device"] == "cpu"
    assert scenario.to_dict()["vision_navigation_command"] == "lane_follow"
    assert scenario.to_dict()["vision_safety_gate_enabled"] is False
    assert scenario.to_dict()["vision_attack_pattern_gate"] is True


def test_tcp_lite_project_scenario_loads():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_tcp_lite.json")

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == ""
    assert scenario.vision_model_device == "cpu"
    assert scenario.vision_navigation_command == "lane_follow"
    assert scenario.vision_safety_gate_enabled is True
    assert scenario.vision_attack_pattern_gate is False
    assert scenario.vehicle_sensor_limit == 1
    assert scenario.traffic_vehicles == 0


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


def test_attack_pattern_score_is_higher_for_repeated_high_contrast_texture():
    clean_score = compute_attack_pattern_score(np.zeros((90, 160, 3), dtype=np.uint8))
    noisy_score = compute_attack_pattern_score(_checkerboard())

    assert noisy_score > clean_score + 0.2


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

    trajectory, control = model(rgb, speed, command)

    assert trajectory.shape == (2, 4, 2)
    assert control.shape == (2, 3)
