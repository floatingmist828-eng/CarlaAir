from __future__ import annotations

import json

from carlaair_active_world.vision_models.fusion_planner import (
    FusionPlannerAdapter,
    FusionPlannerConfig,
    LinearFusionPlanner,
)


def test_linear_fusion_planner_applies_bounded_json_weights(tmp_path):
    path = tmp_path / "fusion.json"
    path.write_text(
        json.dumps(
            {
                "bias": 0.0,
                "weights": {
                    "center_bias": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = FusionPlannerAdapter(
        FusionPlannerConfig(
            mode="learned",
            checkpoint_path=str(path),
            gain=0.5,
            max_correction=0.2,
            min_confidence=0.1,
        )
    )

    correction, diagnostics = adapter.predict(
        {
            "speed_mps": 1.5,
            "lane_center_offset_m": 0.0,
            "uav_bev": {
                "available": True,
                "road_confidence": 0.8,
                "center_bias": 1.0,
                "forward_density": 0.5,
                "left_right_balance": 0.0,
            },
        }
    )

    assert correction == 0.2
    assert diagnostics["enabled"] is True
    assert diagnostics["applied"] is True
    assert diagnostics["reason"] == "ok"
    assert diagnostics["raw_correction"] == 1.0


def test_fusion_planner_reports_missing_checkpoint_without_correction(tmp_path):
    adapter = FusionPlannerAdapter(
        FusionPlannerConfig(
            mode="learned",
            checkpoint_path=str(tmp_path / "missing.json"),
            gain=1.0,
            max_correction=0.1,
        )
    )

    correction, diagnostics = adapter.predict(
        {
            "uav_bev": {
                "available": True,
                "road_confidence": 1.0,
                "center_bias": 1.0,
            }
        }
    )

    assert correction == 0.0
    assert diagnostics["applied"] is False
    assert diagnostics["reason"] == "missing_checkpoint"


def test_fusion_planner_gates_low_uav_confidence(tmp_path):
    path = tmp_path / "fusion.json"
    path.write_text(json.dumps({"weights": {"center_bias": 1.0}}), encoding="utf-8")
    adapter = FusionPlannerAdapter(
        FusionPlannerConfig(
            mode="learned",
            checkpoint_path=str(path),
            min_confidence=0.5,
        )
    )

    correction, diagnostics = adapter.predict(
        {
            "uav_bev": {
                "available": True,
                "road_confidence": 0.2,
                "center_bias": 1.0,
            }
        }
    )

    assert correction == 0.0
    assert diagnostics["reason"] == "low_confidence"


def test_linear_fusion_planner_uses_named_features():
    planner = LinearFusionPlanner(weights={"center_bias": 0.4, "lane_center_offset_m": -0.2}, bias=0.1)

    assert planner.predict({"center_bias": 1.0, "lane_center_offset_m": 0.5}) == 0.4
