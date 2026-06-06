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

    assert config.duration_sec == 120.0
    assert config.ego_target_speed_mps == 2.5
    assert config.uav_enabled is True
    assert config.uav_control_enabled is True
    assert config.uav_bev_fusion_enabled is True


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
    assert config.to_dict()["uav_fusion_mode"] == "learned"
    assert config.to_dict()["uav_fusion_planner_path"] == "models/fusion_planner.json"


def test_uav_fusion_mode_preserves_legacy_boolean_default():
    disabled = ScenarioConfig.from_dict({"name": "disabled", "uav_bev_fusion_enabled": False})
    enabled = ScenarioConfig.from_dict({"name": "enabled", "uav_bev_fusion_enabled": True})

    assert disabled.uav_fusion_mode == "none"
    assert enabled.uav_fusion_mode == "rule"


def test_experiment_scenario_ladder_configs_load():
    base = ROOT / "configs/scenarios/experiments"
    expected = [
        "clean_no_uav",
        "clean_rule_uav_bev",
        "clean_learned_fusion",
        "meeting_rule_uav_bev",
        "slow_lead_rule_uav_bev",
        "pedestrian_crossing_rule_uav_bev",
        "junction_rule_uav_bev",
        "occlusion_rule_uav_bev",
        "rain_fog_rule_uav_bev",
        "texture_attack_rule_uav_bev",
    ]

    for name in expected:
        config = ScenarioConfig.load(base / f"{name}.json")
        assert config.experiment_group
        assert config.scenario_stage
        assert config.scenario_complexity


def test_remote_sync_script_uses_rsync_and_excludes_heavy_outputs():
    script = ROOT / "scripts/sync_remote_code.sh"

    content = script.read_text(encoding="utf-8")

    assert "rsync" in content
    assert "CARLAAIR_REMOTE_TARGET" in content
    assert "/home/fp/CARLA/CarlaAir-v0.1.7/code/" in content
    assert "--exclude 'recordings/'" in content
    assert "--exclude 'models/*.pt'" in content
