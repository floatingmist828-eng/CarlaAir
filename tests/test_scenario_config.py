from carlaair_active_world.scenario import ScenarioConfig
from scripts import run_uav_task


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
