from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class VisionSafetyGateConfig:
    enabled: bool = True
    attack_pattern_gate: bool = False
    attack_pattern_threshold: float = 0.35


def compute_attack_pattern_score(rgb: Any) -> float:
    try:
        array = np.asarray(rgb)
    except (TypeError, ValueError):
        return 0.0

    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        return 0.0

    try:
        gray = array.astype(np.float32).mean(axis=2) / 255.0
    except (TypeError, ValueError):
        return 0.0

    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0

    vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
    horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
    score = (vertical_edges + horizontal_edges) / 2.0
    return float(np.clip(score, 0.0, 1.0))


def evaluate_vision_safety_gate(
    rgb: Any,
    detector_diagnostics: Mapping[str, Any] | None,
    config: VisionSafetyGateConfig | None = None,
) -> dict[str, Any]:
    gate_config = config or VisionSafetyGateConfig()
    diagnostics = detector_diagnostics or {}
    detector_obstacle = bool(diagnostics.get("obstacle", False))
    attack_pattern_score = compute_attack_pattern_score(rgb)

    result = {
        "enabled": gate_config.enabled,
        "blocked": False,
        "reason": "clear",
        "detector_obstacle": detector_obstacle,
        "attack_pattern_score": attack_pattern_score,
        "attack_pattern_gate": gate_config.attack_pattern_gate,
        "attack_pattern_threshold": gate_config.attack_pattern_threshold,
    }

    if not gate_config.enabled:
        result["reason"] = "disabled"
        return result

    if detector_obstacle:
        result["blocked"] = True
        result["reason"] = "vision_obstacle"
        return result

    if (
        gate_config.attack_pattern_gate
        and attack_pattern_score >= gate_config.attack_pattern_threshold
    ):
        result["blocked"] = True
        result["reason"] = "attack_pattern"

    return result
