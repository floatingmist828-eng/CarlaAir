from pathlib import Path

from carlaair_active_world.scenario import ScenarioConfig
from scripts import run_uav_task

ROOT = Path(__file__).resolve().parents[1]


def test_scenario_config_defaults_to_autopilot():
    config = ScenarioConfig.from_dict(
        {
            "name": "demo",
            "ego_control_mode": "route_follow",
            "ego_drive_hz": 6.0,
            "ego_target_speed_mps": 7.5,
            "ego_lookahead_m": 12.0,
        }
    )

    assert config.ego_control_mode == "route_follow"
    assert config.ego_drive_hz == 6.0
    assert config.ego_target_speed_mps == 7.5
    assert config.ego_lookahead_m == 12.0


def test_run_uav_task_defaults_are_project_relative():
    assert run_uav_task.DEFAULT_SCENARIO.exists()
    assert run_uav_task.DEFAULT_SCENARIO.is_relative_to(run_uav_task.ROOT)
    assert run_uav_task.DEFAULT_OUTPUT_DIR.is_relative_to(run_uav_task.ROOT)


def test_stable_uav_bev_scenario_loads():
    config = ScenarioConfig.load(ROOT / "configs/scenarios/town10hd_vision_tcp_lite_yolo_uav_bev_stable.json")

    assert config.duration_sec == 140.0
    assert config.ego_spawn_index == 86
    assert config.ego_spawn_forward_m == 0.0
    assert config.ego_target_speed_mps == 2.6
    assert config.vision_navigation_command == "lane_follow"
    assert config.vision_first_junction_command == "right"
    assert config.vision_junction_command_sequence == ["right", "straight"]
    assert config.vision_junction_command_hold_sec == 8.0
    assert config.traffic_walkers == 4
    assert config.walker_spawn_indices == [146, 146, 146, 146]
    assert config.walker_spawn_delay_sec == 8.0
    assert config.walker_crossing_offsets_m == [-2.0, -0.7, 0.7, 2.0]
    assert config.traffic_vehicles == 4
    assert config.traffic_spawn_indices == [25, 24, 32, 31]
    assert config.traffic_spawn_delay_sec == 23.0
    assert config.vision_detector_confidence == 0.45
    assert config.vision_safety_gate_enabled is True
    assert config.uav_enabled is True
    assert config.uav_control_enabled is True
    assert config.uav_bev_fusion_enabled is True
    assert config.candidate_offsets[0].name == "front_lead_high"
    assert config.candidate_offsets[0].local_offset.x >= 26.0
    assert config.candidate_offsets[0].local_offset.z >= 30.0


def test_scenario_config_round_trips_uav_fusion_mode_and_planner():
    config = ScenarioConfig.from_dict(
        {
            "name": "fusion_config",
            "uav_bev_fusion_enabled": True,
            "uav_fusion_mode": "learned",
            "uav_fusion_planner_path": "models/fusion_planner.json",
            "uav_fusion_planner_gain": 0.5,
            "uav_fusion_max_steer_correction": 0.04,
            "uav_fusion_min_confidence": 0.3,
            "experiment_group": "learned_fusion",
            "scenario_stage": "clean",
            "scenario_complexity": ["clean"],
            "traffic_spawn_start_index": 12,
            "traffic_spawn_indices": [12, 18, 24],
            "traffic_spawn_delay_sec": 3.5,
            "traffic_speed_difference": 65.0,
            "walker_spawn_start_index": 18,
            "walker_spawn_indices": [18, 27, 95],
            "walker_spawn_delay_sec": 7.5,
            "walker_crossing_distance_m": 8.0,
            "walker_crossing_offsets_m": [-1.0, 0.5, 2.0],
            "walker_speed_mps": 1.2,
            "ego_spawn_forward_m": 20.0,
        }
    )

    assert config.uav_fusion_mode == "learned"
    assert config.uav_fusion_planner_path == "models/fusion_planner.json"
    assert config.uav_fusion_planner_gain == 0.5
    assert config.uav_fusion_max_steer_correction == 0.04
    assert config.uav_fusion_min_confidence == 0.3
    assert config.experiment_group == "learned_fusion"
    assert config.scenario_stage == "clean"
    assert config.scenario_complexity == ["clean"]
    assert config.traffic_spawn_start_index == 12
    assert config.traffic_spawn_indices == [12, 18, 24]
    assert config.traffic_spawn_delay_sec == 3.5
    assert config.traffic_speed_difference == 65.0
    assert config.walker_spawn_start_index == 18
    assert config.walker_spawn_indices == [18, 27, 95]
    assert config.walker_spawn_delay_sec == 7.5
    assert config.walker_crossing_distance_m == 8.0
    assert config.walker_crossing_offsets_m == [-1.0, 0.5, 2.0]
    assert config.walker_speed_mps == 1.2
    assert config.ego_spawn_forward_m == 20.0
    assert config.to_dict()["uav_fusion_mode"] == "learned"
    assert config.to_dict()["uav_fusion_planner_path"] == "models/fusion_planner.json"
    assert config.to_dict()["traffic_speed_difference"] == 65.0
    assert config.to_dict()["traffic_spawn_indices"] == [12, 18, 24]
    assert config.to_dict()["walker_spawn_start_index"] == 18
    assert config.to_dict()["walker_spawn_indices"] == [18, 27, 95]
    assert config.to_dict()["walker_crossing_offsets_m"] == [-1.0, 0.5, 2.0]
    assert config.to_dict()["traffic_spawn_delay_sec"] == 3.5
    assert config.to_dict()["walker_spawn_delay_sec"] == 7.5
    assert config.to_dict()["ego_spawn_forward_m"] == 20.0


def test_uav_fusion_mode_preserves_legacy_boolean_default():
    disabled = ScenarioConfig.from_dict({"name": "disabled", "uav_bev_fusion_enabled": False})
    enabled = ScenarioConfig.from_dict({"name": "enabled", "uav_bev_fusion_enabled": True})

    assert disabled.uav_fusion_mode == "none"
    assert enabled.uav_fusion_mode == "rule"


def test_scenario_config_round_trips_junction_command_sequence():
    config = ScenarioConfig.from_dict(
        {
            "name": "junction_sequence",
            "vision_first_junction_command": "right",
            "vision_junction_command_sequence": ["right", "straight"],
        }
    )

    assert config.vision_first_junction_command == "right"
    assert config.vision_junction_command_sequence == ["right", "straight"]
    assert config.to_dict()["vision_junction_command_sequence"] == ["right", "straight"]


def test_experiment_scenario_ladder_configs_load():
    base = ROOT / "configs/scenarios/experiments"
    expected = [
        "clean_no_uav",
        "clean_rule_uav_bev",
        "clean_learned_fusion",
        "meeting_no_uav",
        "meeting_rule_uav_bev",
        "meeting_learned_fusion",
        "multi_vehicle_meeting_no_uav",
        "multi_vehicle_meeting_rule_uav_bev",
        "multi_vehicle_meeting_learned_fusion",
        "slow_lead_no_uav",
        "slow_lead_rule_uav_bev",
        "slow_lead_learned_fusion",
        "pedestrian_crossing_no_uav",
        "pedestrian_crossing_rule_uav_bev",
        "pedestrian_crossing_learned_fusion",
        "multi_pedestrian_crossing_no_uav",
        "multi_pedestrian_crossing_rule_uav_bev",
        "multi_pedestrian_crossing_learned_fusion",
        "junction_no_uav",
        "junction_rule_uav_bev",
        "junction_learned_fusion",
        "occlusion_no_uav",
        "occlusion_rule_uav_bev",
        "occlusion_learned_fusion",
        "rain_fog_no_uav",
        "rain_fog_rule_uav_bev",
        "rain_fog_learned_fusion",
        "texture_attack_no_uav",
        "texture_attack_rule_uav_bev",
        "texture_attack_learned_fusion",
    ]

    for name in expected:
        config = ScenarioConfig.load(base / f"{name}.json")
        assert config.experiment_group
        assert config.scenario_stage
        assert config.scenario_complexity
        assert config.vision_navigation_command == "lane_follow"
        assert config.vision_first_junction_command == "right"
        assert config.vision_junction_command_hold_sec == 8.0
        assert config.candidate_offsets[0].local_offset.z >= 18.0


def test_experiment_ladder_has_three_way_comparison_groups_per_stage():
    base = ROOT / "configs/scenarios/experiments"
    expected_groups = {"no_uav", "rule_uav_bev", "learned_fusion"}
    groups_by_stage = {}

    for path in base.glob("*.json"):
        config = ScenarioConfig.load(path)
        groups_by_stage.setdefault(config.scenario_stage, set()).add(config.experiment_group)

    for stage in [
        "clean",
        "junction_meeting",
        "junction_multi_vehicle_meeting",
        "slow_lead",
        "junction_pedestrian",
        "junction_multi_pedestrian",
        "junction",
        "occlusion",
        "rain_fog",
        "texture_attack",
    ]:
        assert groups_by_stage[stage] == expected_groups


def test_clean_configs_use_tcp_lite_known_spawn_and_high_uav_viewpoint():
    base = ROOT / "configs/scenarios/experiments"
    for name in ["clean_no_uav", "clean_rule_uav_bev", "clean_learned_fusion"]:
        config = ScenarioConfig.load(base / f"{name}.json")

        assert config.ego_spawn_index == 20
        assert config.ego_spawn_forward_m == 20.0
        assert config.ego_target_speed_mps == 3.0
        assert config.duration_sec == 90.0
        assert config.scenario_stage == "clean"
        assert config.scenario_complexity == ["tcp_lite_known_clean"]
        assert config.candidate_offsets[0].local_offset.z == 18.0


def test_complex_configs_use_junction_interaction_spawns():
    base = ROOT / "configs/scenarios/experiments"
    for name in ["meeting_no_uav", "meeting_rule_uav_bev", "meeting_learned_fusion"]:
        config = ScenarioConfig.load(base / f"{name}.json")

        assert config.ego_spawn_index == 86
        assert config.ego_spawn_forward_m == 0.0
        assert config.duration_sec == 75.0
        assert config.traffic_spawn_start_index == 138
        assert config.traffic_spawn_delay_sec == 28.0
        assert config.scenario_stage == "junction_meeting"
        assert "junction" in config.scenario_complexity

    for name in [
        "multi_vehicle_meeting_no_uav",
        "multi_vehicle_meeting_rule_uav_bev",
        "multi_vehicle_meeting_learned_fusion",
    ]:
        config = ScenarioConfig.load(base / f"{name}.json")

        assert config.ego_spawn_index == 86
        assert config.duration_sec == 90.0
        assert config.traffic_vehicles == 4
        assert config.traffic_spawn_indices == [25, 24, 32, 31]
        assert config.traffic_spawn_delay_sec == 23.0
        assert config.scenario_stage == "junction_multi_vehicle_meeting"
        assert "multi_vehicle_meeting" in config.scenario_complexity

    slow_lead = ScenarioConfig.load(base / "slow_lead_rule_uav_bev.json")
    assert slow_lead.ego_spawn_index == 86
    assert slow_lead.ego_spawn_forward_m == 0.0
    assert slow_lead.traffic_spawn_start_index > 0
    assert slow_lead.traffic_speed_difference >= 60.0
    assert slow_lead.vision_first_junction_command == "right"
    assert slow_lead.candidate_offsets[0].local_offset.z == 18.0

    for name in [
        "junction_rule_uav_bev",
        "occlusion_rule_uav_bev",
        "rain_fog_rule_uav_bev",
        "texture_attack_rule_uav_bev",
    ]:
        config = ScenarioConfig.load(base / f"{name}.json")
        assert config.ego_spawn_index == 86
        assert config.ego_spawn_forward_m == 0.0
        assert "junction" in config.scenario_complexity

    for name in [
        "pedestrian_crossing_no_uav",
        "pedestrian_crossing_rule_uav_bev",
        "pedestrian_crossing_learned_fusion",
    ]:
        config = ScenarioConfig.load(base / f"{name}.json")

        assert config.ego_spawn_index == 86
        assert config.ego_spawn_forward_m == 0.0
        assert config.duration_sec == 90.0
        assert config.walker_spawn_start_index == 27
        assert config.walker_spawn_delay_sec == 38.0
        assert config.walker_crossing_distance_m == 14.0
        assert config.scenario_stage == "junction_pedestrian"
        assert "junction" in config.scenario_complexity

    for name in [
        "multi_pedestrian_crossing_no_uav",
        "multi_pedestrian_crossing_rule_uav_bev",
        "multi_pedestrian_crossing_learned_fusion",
    ]:
        config = ScenarioConfig.load(base / f"{name}.json")

        assert config.ego_spawn_index == 86
        assert config.duration_sec == 95.0
        assert config.traffic_walkers == 4
        assert config.walker_spawn_indices == [146, 146, 146, 146]
        assert config.walker_spawn_delay_sec == 8.0
        assert config.walker_crossing_distance_m == 14.0
        assert config.walker_crossing_offsets_m == [-2.0, -0.7, 0.7, 2.0]
        assert config.walker_speed_mps == 0.85
        assert config.scenario_stage == "junction_multi_pedestrian"
        assert "multi_pedestrian_crossing" in config.scenario_complexity


def test_remote_sync_script_uses_rsync_and_excludes_heavy_outputs():
    script = ROOT / "scripts/sync_remote_code.sh"

    content = script.read_text(encoding="utf-8")

    assert "rsync" in content
    assert "CARLAAIR_REMOTE_TARGET" in content
    assert "/home/fp/CARLA/CarlaAir-v0.1.7/code/" in content
    assert "--exclude 'recordings/'" in content
    assert "--exclude 'models/*.pt'" in content
