from __future__ import annotations

import csv
import json

from scripts.evaluate_fusion_benchmark import (
    _write_svg_bar_chart,
    _write_svg_line_chart,
    evaluate_directory,
    evaluate_episode,
)


def _step(
    t,
    x,
    y,
    speed,
    steer,
    brake,
    lane_offset,
    blocked=False,
    collision=False,
    junction=False,
    nested_control=False,
    uav_fusion=None,
):
    control_values = {
        "steer": steer,
        "throttle": 0.2 if brake <= 0.0 else 0.0,
        "brake": brake,
    }
    ego_control = {
        "safety_gate": {
            "blocked": blocked,
        },
        "stabilization": {
            "lane_center_offset_m": lane_offset,
            "in_junction": junction,
        },
    }
    if uav_fusion is not None:
        ego_control["stabilization"]["uav_bev_fusion"] = uav_fusion
    if nested_control:
        ego_control["control"] = control_values
    else:
        ego_control.update(control_values)
    return {
        "observation": {
            "time": t,
            "ego": {
                "pose": {
                    "position": {
                        "x": x,
                        "y": y,
                        "z": 0.0,
                    }
                },
                "velocity": {
                    "x": speed,
                    "y": 0.0,
                    "z": 0.0,
                },
            },
            "waypoint": {
                "is_junction": junction,
            },
            "ego_control": ego_control,
        },
        "label": {
            "collision": collision,
            "ego": {
                "junction": junction,
            },
        },
    }


def test_evaluate_episode_computes_fusion_benchmark_metrics(tmp_path):
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "scenario": {
                        "name": "clean_rule_uav_bev",
                        "duration_sec": 60.0,
                        "ego_target_speed_mps": 2.0,
                        "experiment_group": "rule_uav_bev",
                        "scenario_stage": "clean",
                    }
                },
                "steps": [
                    _step(0.0, 0.0, 0.0, 2.0, 0.10, 0.0, 0.10),
                    _step(30.0, 30.0, 0.0, 2.0, -0.10, 0.7, -0.20, blocked=True),
                    _step(60.0, 70.0, 0.0, 2.0, 0.20, 0.0, 0.30, collision=True),
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_episode(path)

    assert metrics["scenario"] == "clean_rule_uav_bev"
    assert metrics["experiment_group"] == "rule_uav_bev"
    assert metrics["scenario_stage"] == "clean"
    assert metrics["collision_rate"] == 1.0
    assert metrics["lane_offset_mean_abs"] == 0.2
    assert metrics["lane_offset_max_abs"] == 0.3
    assert metrics["path_distance_m"] == 70.0
    assert metrics["net_displacement_m"] == 70.0
    assert metrics["hard_brake_count"] == 1
    assert metrics["steer_oscillation_count"] == 2
    assert metrics["safety_gate_count"] == 1
    assert metrics["scene_success"] == 0.0
    assert metrics["segments"][0]["path_distance_m"] == 30.0
    assert metrics["segments"][1]["path_distance_m"] == 40.0


def test_evaluate_directory_writes_summary_and_segments_csv(tmp_path):
    input_dir = tmp_path / "runs"
    output_dir = tmp_path / "eval"
    input_dir.mkdir()
    for index, group in enumerate(["no_uav", "learned_fusion"]):
        path = input_dir / f"episode_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "meta": {
                        "scenario": {
                            "name": f"clean_{group}",
                            "experiment_group": group,
                            "scenario_stage": "clean",
                        }
                    },
                    "steps": [
                        _step(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                        _step(10.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.1),
                    ],
                }
            ),
            encoding="utf-8",
        )

    results = evaluate_directory(input_dir, output_dir, make_plots=False)

    assert len(results) == 2
    summary_path = output_dir / "summary.csv"
    segments_path = output_dir / "segments.csv"
    timeseries_path = output_dir / "timeseries.csv"
    assert summary_path.exists()
    assert segments_path.exists()
    assert timeseries_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment_group"] for row in rows] == ["no_uav", "learned_fusion"]
    with timeseries_path.open("r", encoding="utf-8", newline="") as handle:
        series_rows = list(csv.DictReader(handle))
    assert series_rows[0]["time_sec"] == "0.0"
    assert series_rows[0]["steer"] == "0.0"
    assert series_rows[1]["lane_offset_m"] == "0.1"


def test_svg_plot_fallback_writes_bar_and_line_charts(tmp_path):
    bar_path = tmp_path / "plots" / "collision_rate.svg"
    line_path = tmp_path / "plots" / "lane_offset_m_curve.svg"

    _write_svg_bar_chart(
        [
            {"scenario_stage": "clean", "experiment_group": "no_uav", "collision_rate": 0.0},
            {"scenario_stage": "clean", "experiment_group": "rule_uav_bev", "collision_rate": 1.0},
        ],
        bar_path,
        "collision_rate",
        "Collision Rate",
    )
    _write_svg_line_chart(
        [
            {"label": "clean:no_uav", "xs": [0.0, 1.0], "ys": [0.0, 0.2]},
            {"label": "clean:rule_uav_bev", "xs": [0.0, 1.0], "ys": [0.1, 0.3]},
        ],
        line_path,
        "Lane Offset Curve",
        "time_sec",
        "lane_offset_m",
    )

    assert "<svg" in bar_path.read_text(encoding="utf-8")
    assert "Collision Rate" in bar_path.read_text(encoding="utf-8")
    assert "<polyline" in line_path.read_text(encoding="utf-8")
    assert "clean:no_uav" in line_path.read_text(encoding="utf-8")


def test_evaluate_episode_reports_nested_controls_and_junction_metrics(tmp_path):
    path = tmp_path / "junction_episode.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "scenario": {
                        "name": "junction_meeting_rule",
                        "duration_sec": 6.0,
                        "ego_target_speed_mps": 2.0,
                        "experiment_group": "rule_uav_bev",
                        "scenario_stage": "junction_meeting",
                        "scenario_complexity": ["junction", "meeting"],
                    }
                },
                "steps": [
                    _step(
                        0.0,
                        0.0,
                        0.0,
                        2.0,
                        0.12,
                        0.0,
                        0.2,
                        nested_control=True,
                        uav_fusion={"mode": "rule", "available": True, "applied": True, "steer_correction": 0.02},
                    ),
                    _step(
                        1.0,
                        2.0,
                        0.0,
                        2.0,
                        -0.14,
                        0.6,
                        -0.4,
                        blocked=True,
                        junction=True,
                        nested_control=True,
                        uav_fusion={"mode": "rule", "available": True, "applied": True, "steer_correction": -0.03},
                    ),
                    _step(
                        2.0,
                        4.0,
                        0.0,
                        2.0,
                        0.15,
                        0.0,
                        4.2,
                        junction=True,
                        nested_control=True,
                        uav_fusion={"mode": "rule", "available": True, "applied": False, "steer_correction": 0.0},
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_episode(path)

    assert metrics["hard_brake_count"] == 1
    assert metrics["safety_gate_count"] == 1
    assert metrics["off_road_rate"] == 0.333333
    assert metrics["junction_lane_offset_mean_abs"] == 2.3
    assert metrics["junction_lane_offset_max_abs"] == 4.2
    assert metrics["junction_steer_oscillation_count"] == 1
    assert metrics["meeting_scene_success"] == 0.0
    assert metrics["pedestrian_scene_success"] == ""
    assert metrics["uav_available_rate"] == 1.0
    assert metrics["uav_applied_rate"] == 0.666667
    assert metrics["uav_mean_abs_steer_correction"] == 0.016667


def test_evaluate_directory_writes_uav_comparison_csv(tmp_path):
    input_dir = tmp_path / "runs"
    output_dir = tmp_path / "eval"
    input_dir.mkdir()
    episodes = [
        (
            "meeting_no_uav.json",
            "no_uav",
            [
                _step(0.0, 0.0, 0.0, 2.0, 0.20, 0.0, 0.8, junction=True),
                _step(1.0, 1.0, 0.0, 1.5, -0.20, 0.0, 2.0, collision=True, junction=True),
            ],
        ),
        (
            "meeting_rule_uav_bev.json",
            "rule_uav_bev",
            [
                _step(0.0, 0.0, 0.0, 2.0, 0.08, 0.0, 0.2, junction=True),
                _step(1.0, 2.0, 0.0, 2.0, 0.07, 0.0, 0.4, junction=True),
            ],
        ),
    ]
    for file_name, group, steps in episodes:
        (input_dir / file_name).write_text(
            json.dumps(
                {
                    "meta": {
                        "scenario": {
                            "name": file_name.removesuffix(".json"),
                            "duration_sec": 2.0,
                            "ego_target_speed_mps": 1.0,
                            "experiment_group": group,
                            "scenario_stage": "junction_meeting",
                            "scenario_complexity": ["junction", "meeting"],
                        }
                    },
                    "steps": steps,
                }
            ),
            encoding="utf-8",
        )

    evaluate_directory(input_dir, output_dir, make_plots=False)

    comparison_path = output_dir / "comparisons.csv"
    assert comparison_path.exists()
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["scenario_stage"] == "junction_meeting"
    assert row["baseline_group"] == "no_uav"
    assert row["comparison_group"] == "rule_uav_bev"
    assert row["collision_rate_reduction"] == "1.0"
    assert row["lane_offset_mean_abs_reduction"] == "1.1"
    assert row["junction_steer_oscillation_reduction"] == "1"
    assert row["meeting_scene_success_gain"] == "1.0"
    assert row["uav_helped"] == "true"
