from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def extract_uav_bev_feature(rgb: Any) -> Dict[str, Any]:
    """Build a compact BEV-like global context feature from the UAV RGB view."""
    try:
        image = np.asarray(rgb)
    except Exception:
        return {"available": False, "reason": "invalid_rgb"}
    if image.ndim != 3 or image.shape[2] < 3 or image.shape[0] < 4 or image.shape[1] < 4:
        return {"available": False, "reason": "invalid_rgb"}

    image = image[:, :, :3].astype(np.float32)
    height, width = image.shape[:2]
    max_channel = image.max(axis=2)
    min_channel = image.min(axis=2)
    brightness = image.mean(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1.0)

    road_mask = (brightness >= 35.0) & (brightness <= 210.0) & (saturation <= 0.42)
    road_confidence = float(road_mask.mean())
    if np.count_nonzero(road_mask) == 0:
        center_bias = 0.0
        forward_density = 0.0
        left_right_balance = 0.0
    else:
        ys, xs = np.nonzero(road_mask)
        center_x = float(xs.mean())
        center_bias = _clamp((center_x - (width - 1) * 0.5) / max(1.0, width * 0.5), -1.0, 1.0)
        forward_density = float(road_mask[: max(1, height // 2), :].mean())
        left_density = float(road_mask[:, : max(1, width // 2)].mean())
        right_density = float(road_mask[:, max(1, width // 2) :].mean())
        left_right_balance = _clamp(right_density - left_density, -1.0, 1.0)

    feature = [
        float(road_confidence),
        float(center_bias),
        float(forward_density),
        float(left_right_balance),
    ]
    return {
        "available": True,
        "source": "uav_rgb_bev_lite",
        "feature_dim": len(feature),
        "feature": feature,
        "road_confidence": float(road_confidence),
        "center_bias": float(center_bias),
        "forward_density": float(forward_density),
        "left_right_balance": float(left_right_balance),
        "image_shape": [int(height), int(width), int(image.shape[2])],
    }


@dataclass
class CachedUAVBEVProvider:
    sensor_getter: Callable[[], Any]
    refresh_hz: float = 2.0
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._cache: Optional[Dict[str, Any]] = None
        self._last_refresh_at = -1.0e9

    def snapshot(self) -> Dict[str, Any]:
        now = float(self.clock())
        interval = 1.0 / max(0.1, float(self.refresh_hz))
        if self._cache is not None and now - self._last_refresh_at < interval:
            cached = dict(self._cache)
            cached["age_sec"] = float(max(0.0, now - self._last_refresh_at))
            return cached

        sensor = self.sensor_getter()
        if sensor is None:
            feature = {"available": False, "reason": "missing_uav_sensor"}
        else:
            try:
                frames = sensor.snapshot() or {}
                feature = extract_uav_bev_feature(frames.get("rgb"))
                feature["rgb_available"] = frames.get("rgb") is not None
            except Exception as exc:
                feature = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

        feature["age_sec"] = 0.0
        self._cache = dict(feature)
        self._last_refresh_at = now
        return feature
