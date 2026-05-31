from __future__ import annotations

from carlaair_active_world.scenario import ScenarioConfig


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
