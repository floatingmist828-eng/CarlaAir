from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import carla
import numpy as np

from .base import VisionPolicy


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


@dataclass
class SimpleLaneVisionConfig:
    target_speed_mps: float = 4.0
    low_confidence_speed_mps: float = 1.2
    curve_speed_mps: float = 1.8
    curve_error_threshold: float = 0.35
    speed_kp: float = 0.18
    speed_kd: float = 0.05
    max_throttle: float = 0.35
    steer_gain: float = 0.75
    steer_smoothing: float = 0.65
    min_lane_confidence: float = 0.01
    semantic_min_confidence: float = 0.0
    min_clearance_signal: float = 0.005
    stuck_speed_mps: float = 0.18
    stuck_frame_threshold: int = 24
    recovery_frames: int = 28
    recovery_throttle: float = 0.42
    recovery_steer: float = 0.62
    semantic_right_lane_offset_ratio: float = 0.18
    rgb_yellow_lane_offset_ratio: float = 0.30
    lane_target_margin_ratio: float = 0.08
    lane_memory_frames: int = 32


class SimpleLaneVisionPolicy(VisionPolicy):
    """Lightweight RGB/depth visual baseline for low-speed closed-loop driving."""

    def __init__(self, target_speed_mps: float = 4.0, config: Optional[SimpleLaneVisionConfig] = None) -> None:
        self.config = config or SimpleLaneVisionConfig(target_speed_mps=target_speed_mps)
        self._steer_history: deque[float] = deque(maxlen=5)
        self._speed_error_history: deque[float] = deque(maxlen=5)
        self._stuck_frames = 0
        self._recovery_frames_left = 0
        self._recovery_direction = 1.0
        self._lane_memory_frames_left = 0
        self._last_lane_error = 0.0
        self._last_lane_confidence = 0.0
        self._last_lane_target_x = 0.0
        self.last_diagnostics: Dict[str, Any] = {}

    def reset(self) -> None:
        self._steer_history.clear()
        self._speed_error_history.clear()
        self._stuck_frames = 0
        self._recovery_frames_left = 0
        self._recovery_direction = 1.0
        self._lane_memory_frames_left = 0
        self._last_lane_error = 0.0
        self._last_lane_confidence = 0.0
        self._last_lane_target_x = 0.0
        self.last_diagnostics = {}

    @staticmethod
    def _weighted_center(mask: np.ndarray) -> Optional[float]:
        if not np.any(mask):
            return None
        ys, xs = np.nonzero(mask)
        weights = (ys + 1).astype(np.float32)
        return float(np.average(xs.astype(np.float32), weights=weights))

    def _error_from_target_x(self, target_x: float, width: int) -> float:
        return _clamp((float(target_x) - (width * 0.5)) / max(1.0, width * 0.5), -1.0, 1.0)

    def _estimate_lane_error(self, rgb: np.ndarray) -> tuple[float, float, float]:
        if rgb.ndim != 3 or rgb.shape[0] < 8 or rgb.shape[1] < 8:
            return 0.0, 0.0, 0.0

        height, width = rgb.shape[:2]
        roi = rgb[height // 2 :, :, :3].astype(np.uint8)
        red = roi[:, :, 0]
        green = roi[:, :, 1]
        blue = roi[:, :, 2]

        white = (red > 180) & (green > 180) & (blue > 180)
        yellow = (red > 150) & (green > 120) & (blue < 130)
        mask = white | yellow
        confidence = float(np.count_nonzero(mask)) / float(mask.size)
        if confidence <= 0.0:
            return 0.0, 0.0, 0.0

        yellow_center = self._weighted_center(yellow)
        white_center = self._weighted_center(white)
        margin = width * self.config.lane_target_margin_ratio
        if yellow_center is not None and white_center is not None and white_center > yellow_center:
            target_x = 0.5 * (yellow_center + white_center)
        elif yellow_center is not None:
            target_x = yellow_center + width * self.config.rgb_yellow_lane_offset_ratio
        else:
            target_x = self._weighted_center(white)
        target_x = _clamp(float(target_x), margin, width - margin)
        return self._error_from_target_x(target_x, width), confidence, target_x

    def _estimate_semantic_lane_error(self, semantic: np.ndarray) -> tuple[float, float, float]:
        if semantic.ndim == 3:
            semantic = semantic[:, :, 0]
        if semantic.ndim != 2 or semantic.shape[0] < 8 or semantic.shape[1] < 8:
            return 0.0, 0.0, 0.0

        height, width = semantic.shape[:2]
        roi = semantic[height // 2 :, :]
        drivable = (roi == 6) | (roi == 7)
        confidence = float(np.count_nonzero(drivable)) / float(drivable.size)
        if confidence <= 0.0:
            return 0.0, 0.0, 0.0

        rows = drivable.any(axis=1)
        left_edges = []
        right_edges = []
        weights = []
        for row in np.nonzero(rows)[0]:
            xs = np.nonzero(drivable[row])[0]
            if xs.size:
                left_edges.append(float(xs[0]))
                right_edges.append(float(xs[-1]))
                weights.append(float(row + 1))
        if not left_edges:
            return 0.0, 0.0, 0.0

        left_edge = float(np.average(left_edges, weights=weights))
        right_edge = float(np.average(right_edges, weights=weights))
        road_width = max(1.0, right_edge - left_edge)
        target_x = 0.5 * (left_edge + right_edge) + road_width * self.config.semantic_right_lane_offset_ratio
        margin = width * self.config.lane_target_margin_ratio
        target_x = _clamp(target_x, left_edge + margin, right_edge - margin)
        return self._error_from_target_x(target_x, width), confidence, target_x

    def _estimate_clearance(self, depth: Optional[np.ndarray]) -> float:
        if depth is None or depth.ndim != 3 or depth.size == 0:
            return 1.0

        height, width = depth.shape[:2]
        y0 = int(height * 0.55)
        y1 = int(height * 0.92)
        x0 = int(width * 0.36)
        x1 = int(width * 0.64)
        roi = depth[y0:y1, x0:x1, :3].astype(np.float32)
        if roi.size == 0:
            return 1.0

        brightness = float(np.percentile(roi, 98.0) / 255.0)
        return _clamp(1.0 - brightness, 0.0, 1.0)

    def _control_speed(self, speed_mps: float, target_speed: float) -> tuple[float, float]:
        error = float(target_speed - speed_mps)
        self._speed_error_history.append(error)
        derivative = 0.0
        if len(self._speed_error_history) >= 2:
            derivative = self._speed_error_history[-1] - self._speed_error_history[-2]

        throttle = self.config.speed_kp * error + self.config.speed_kd * derivative
        throttle = _clamp(throttle, 0.0, self.config.max_throttle)
        brake = 0.0
        if error < -0.8:
            brake = _clamp((-error) / max(1.0, self.config.target_speed_mps), 0.0, 0.6)
            throttle = 0.0
        return throttle, brake

    def predict(self, obs: Dict[str, Any]) -> carla.VehicleControl:
        control = carla.VehicleControl()
        rgb = obs.get("rgb")
        depth = obs.get("depth")
        semantic = obs.get("semantic")
        speed_mps = float(obs.get("speed_mps", 0.0))
        vision_obstacle = bool(obs.get("vision_obstacle", False))

        if rgb is None and semantic is None:
            control.brake = 1.0
            self.last_diagnostics = {
                "frame_ready": False,
                "lane_source": "none",
                "lane_confidence": 0.0,
                "clearance": 0.0,
                "vision_obstacle": bool(vision_obstacle),
                "speed_mps": speed_mps,
                "reason": "missing_vision",
            }
            return control

        lane_source = "rgb"
        lane_target_x = 0.0
        if semantic is not None:
            lane_error, lane_confidence, lane_target_x = self._estimate_semantic_lane_error(np.asarray(semantic))
            if lane_confidence > self.config.semantic_min_confidence:
                lane_source = "semantic"
            elif rgb is not None:
                lane_error, lane_confidence, lane_target_x = self._estimate_lane_error(np.asarray(rgb))
        else:
            lane_error, lane_confidence, lane_target_x = self._estimate_lane_error(np.asarray(rgb))
        if lane_confidence >= self.config.min_lane_confidence:
            self._last_lane_error = float(lane_error)
            self._last_lane_confidence = float(lane_confidence)
            self._last_lane_target_x = float(lane_target_x)
            self._lane_memory_frames_left = int(self.config.lane_memory_frames)
        elif self._lane_memory_frames_left > 0:
            self._lane_memory_frames_left -= 1
            lane_error = self._last_lane_error
            lane_confidence = self._last_lane_confidence
            lane_target_x = self._last_lane_target_x
            lane_source = "memory"
        clearance = self._estimate_clearance(None if depth is None else np.asarray(depth))
        close_obstacle = clearance < self.config.min_clearance_signal or vision_obstacle

        if lane_confidence < self.config.min_lane_confidence:
            target_speed = self.config.low_confidence_speed_mps
            steer_raw = 0.0
        else:
            target_speed = self.config.target_speed_mps
            if abs(float(lane_error)) >= self.config.curve_error_threshold:
                target_speed = min(target_speed, self.config.curve_speed_mps)
            steer_raw = _clamp(self.config.steer_gain * lane_error, -0.55, 0.55)

        self._steer_history.append(steer_raw)
        steer = float(np.mean(self._steer_history)) if self._steer_history else steer_raw
        steer = _clamp(
            self.config.steer_smoothing * steer + (1.0 - self.config.steer_smoothing) * steer_raw,
            -1.0,
            1.0,
        )

        throttle, brake = self._control_speed(speed_mps, target_speed)
        if close_obstacle:
            throttle = 0.0
            brake = 1.0

        recovery_active = False
        if (
            not close_obstacle
            and lane_confidence >= self.config.min_lane_confidence
            and speed_mps < self.config.stuck_speed_mps
            and throttle > 0.0
        ):
            self._stuck_frames += 1
        else:
            self._stuck_frames = 0

        if self._recovery_frames_left <= 0 and self._stuck_frames >= self.config.stuck_frame_threshold:
            self._recovery_frames_left = int(self.config.recovery_frames)
            self._recovery_direction = -1.0 if lane_error >= 0.0 else 1.0
            self._stuck_frames = 0

        if self._recovery_frames_left > 0:
            recovery_active = True
            self._recovery_frames_left -= 1
            throttle = float(self.config.recovery_throttle)
            brake = 0.0
            steer = _clamp(self._recovery_direction * self.config.recovery_steer, -1.0, 1.0)

        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.reverse = bool(recovery_active)
        self.last_diagnostics = {
            "frame_ready": True,
            "lane_source": lane_source,
            "lane_error": float(lane_error),
            "lane_confidence": float(lane_confidence),
            "lane_target_x": float(lane_target_x),
            "lane_memory_frames_left": int(self._lane_memory_frames_left),
            "clearance": float(clearance),
            "vision_obstacle": bool(vision_obstacle),
            "close_obstacle": bool(close_obstacle),
            "speed_mps": float(speed_mps),
            "target_speed_mps": float(target_speed),
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "reverse": bool(control.reverse),
            "recovery_active": bool(recovery_active),
            "stuck_frames": int(self._stuck_frames),
            "recovery_frames_left": int(self._recovery_frames_left),
        }
        return control
