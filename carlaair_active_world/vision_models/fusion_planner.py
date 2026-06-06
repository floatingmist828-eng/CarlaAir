from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


@dataclass
class FusionPlannerConfig:
    mode: str = "none"
    checkpoint_path: str = ""
    gain: float = 1.0
    max_correction: float = 0.08
    min_confidence: float = 0.20


class LinearFusionPlanner:
    def __init__(self, weights: Mapping[str, float], bias: float = 0.0) -> None:
        self.weights = {str(name): float(value) for name, value in weights.items()}
        self.bias = float(bias)

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearFusionPlanner":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        weights = payload.get("weights", {})
        if not isinstance(weights, dict) or not weights:
            raise ValueError("fusion planner checkpoint must contain non-empty weights")
        return cls(weights=weights, bias=float(payload.get("bias", 0.0)))

    def predict(self, features: Mapping[str, float]) -> float:
        value = self.bias
        for name, weight in self.weights.items():
            value += weight * float(features.get(name, 0.0) or 0.0)
        return float(value)


class FusionPlannerAdapter:
    def __init__(self, config: FusionPlannerConfig, planner: Optional[LinearFusionPlanner] = None) -> None:
        self.config = config
        self.mode = str(config.mode or "none").lower()
        self.planner = planner
        self.load_reason = "ok" if planner is not None else "disabled"
        if self.mode == "learned" and self.planner is None:
            self._load_checkpoint(config.checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not checkpoint_path:
            self.load_reason = "missing_checkpoint"
            return
        path = Path(checkpoint_path)
        if not path.exists():
            self.load_reason = "missing_checkpoint"
            return
        try:
            self.planner = LinearFusionPlanner.from_json(path)
            self.load_reason = "ok"
        except Exception as exc:
            self.planner = None
            self.load_reason = f"load_failed:{exc.__class__.__name__}"

    @staticmethod
    def _feature_map(obs: Mapping[str, Any]) -> Dict[str, float]:
        uav_bev = obs.get("uav_bev") or {}
        features = {
            "road_confidence": float(uav_bev.get("road_confidence", 0.0) or 0.0),
            "center_bias": float(uav_bev.get("center_bias", 0.0) or 0.0),
            "forward_density": float(uav_bev.get("forward_density", 0.0) or 0.0),
            "left_right_balance": float(uav_bev.get("left_right_balance", 0.0) or 0.0),
            "speed_mps": float(obs.get("speed_mps", 0.0) or 0.0),
            "lane_center_offset_m": float(obs.get("lane_center_offset_m", 0.0) or 0.0),
            "in_junction": 1.0 if bool(obs.get("in_junction", False)) else 0.0,
        }
        vector = uav_bev.get("feature")
        if isinstance(vector, (list, tuple)):
            for index, value in enumerate(vector):
                try:
                    features[f"feature_{index}"] = float(value)
                except (TypeError, ValueError):
                    features[f"feature_{index}"] = 0.0
        return features

    def predict(self, obs: Mapping[str, Any]) -> tuple[float, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "enabled": self.mode == "learned",
            "mode": self.mode,
            "applied": False,
            "steer_correction": 0.0,
            "max_steer_correction": float(self.config.max_correction),
        }
        if self.mode != "learned":
            diagnostics["reason"] = "disabled"
            return 0.0, diagnostics
        if self.planner is None:
            diagnostics["reason"] = self.load_reason
            return 0.0, diagnostics

        uav_bev = obs.get("uav_bev") or {}
        diagnostics["available"] = bool(uav_bev.get("available", False))
        if not diagnostics["available"]:
            diagnostics["reason"] = str(uav_bev.get("reason", "unavailable"))
            return 0.0, diagnostics

        try:
            features = self._feature_map(obs)
        except (TypeError, ValueError):
            diagnostics["reason"] = "invalid_feature"
            return 0.0, diagnostics

        confidence = float(features.get("road_confidence", 0.0))
        diagnostics["road_confidence"] = confidence
        diagnostics["min_confidence"] = float(self.config.min_confidence)
        if confidence < float(self.config.min_confidence):
            diagnostics["reason"] = "low_confidence"
            return 0.0, diagnostics

        raw = float(self.planner.predict(features))
        correction = _clamp(
            raw * float(self.config.gain),
            -float(self.config.max_correction),
            float(self.config.max_correction),
        )
        if abs(correction) < 1.0e-4:
            diagnostics["reason"] = "zero_correction"
            diagnostics["raw_correction"] = raw
            return 0.0, diagnostics

        diagnostics["applied"] = True
        diagnostics["reason"] = "ok"
        diagnostics["raw_correction"] = raw
        diagnostics["steer_correction"] = float(correction)
        return float(correction), diagnostics
