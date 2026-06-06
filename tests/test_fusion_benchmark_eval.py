from __future__ import annotations

import csv
import json

from scripts.evaluate_fusion_benchmark import evaluate_directory, evaluate_episode


def _step(t, x, y, speed, steer, brake, lane_offset, blocked=False, collision=False):
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
            "ego_control": {
                "steer": steer,
                "brake": brake,
                "safety_gate": {
                    "blocked": blocked,
                },
                "stabilization": {
                    "lane_center_offset_m": lane_offset,
                },
            },
        },
        "label": {
            "collision": collision,
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
    assert summary_path.exists()
    assert segments_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment_group"] for row in rows] == ["no_uav", "learned_fusion"]
