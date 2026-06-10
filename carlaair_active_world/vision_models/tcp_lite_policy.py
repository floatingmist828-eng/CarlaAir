from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import carla
import numpy as np

from .base import VisionPolicy
from .fusion_planner import FusionPlannerAdapter, FusionPlannerConfig
from .safety_gate import VisionSafetyGateConfig, evaluate_vision_safety_gate
from .simple_lane import SimpleLaneVisionPolicy
from .tcp_lite import command_to_index


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TcpLiteVisionPolicy(VisionPolicy):
    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        navigation_command: str = "lane_follow",
        safety_gate_enabled: bool = True,
        attack_pattern_gate: bool = False,
        attack_pattern_threshold: float = 0.35,
        low_visibility_gate: bool = False,
        low_visibility_threshold: float = 0.12,
        target_speed_mps: float = 4.0,
        control_mode: str = "trajectory",
        model: Optional[Any] = None,
        uav_bev_fusion_enabled: bool = False,
        uav_bev_min_confidence: float = 0.20,
        uav_bev_steer_gain: float = 0.08,
        uav_bev_max_steer_correction: float = 0.08,
        uav_fusion_mode: str = "",
        uav_fusion_planner_path: str = "",
        uav_fusion_planner_gain: float = 1.0,
        uav_fusion_max_steer_correction: Optional[float] = None,
        uav_fusion_min_confidence: Optional[float] = None,
    ) -> None:
        self.model_path = str(model_path or "")
        self.device = str(device)
        self.navigation_command = str(navigation_command)
        self.target_speed_mps = float(target_speed_mps)
        self.control_mode = str(control_mode or "trajectory").lower()
        if self.control_mode == "trajectory_model":
            self.control_mode = "trajectory"
        self._last_speed_error = 0.0
        self._stuck_frames = 0
        self._fallback_frames_left = 0
        self._fallback_disagreement_threshold = 0.25
        self._fallback_stuck_frame_threshold = 8
        self._fallback_latch_frames = 24
        self._has_last_steer = False
        self._last_steer = 0.0
        self._steer_smoothing = 0.72
        self._steer_rate_limit = 0.06
        self._lane_follow_steer_limit = 0.38
        self._steer_deadband = 0.025
        self._lane_centering_gain = 0.10
        self._lane_centering_deadband_m = 0.15
        self._lane_centering_max_correction = 0.18
        self._junction_steer_limit = 0.34
        self._junction_low_speed_recovery_mps = 0.35
        self._junction_low_speed_recovery_throttle = 0.34
        if not str(uav_fusion_mode or "").strip():
            uav_fusion_mode = "rule" if bool(uav_bev_fusion_enabled) else "none"
        self.uav_fusion_mode = str(uav_fusion_mode).strip().lower()
        if self.uav_fusion_mode not in {"none", "rule", "learned"}:
            raise ValueError("uav_fusion_mode must be one of: none, rule, learned")
        self.uav_bev_fusion_enabled = self.uav_fusion_mode != "none"
        self.uav_bev_min_confidence = float(uav_bev_min_confidence)
        self.uav_bev_steer_gain = float(uav_bev_steer_gain)
        self.uav_bev_max_steer_correction = float(uav_bev_max_steer_correction)
        self.uav_fusion_planner = FusionPlannerAdapter(
            FusionPlannerConfig(
                mode=self.uav_fusion_mode,
                checkpoint_path=str(uav_fusion_planner_path or ""),
                gain=float(uav_fusion_planner_gain),
                max_correction=float(
                    self.uav_bev_max_steer_correction
                    if uav_fusion_max_steer_correction is None
                    else uav_fusion_max_steer_correction
                ),
                min_confidence=float(
                    self.uav_bev_min_confidence
                    if uav_fusion_min_confidence is None
                    else uav_fusion_min_confidence
                ),
            )
        )
        self.fallback_policy = SimpleLaneVisionPolicy(target_speed_mps=self.target_speed_mps)
        self.safety_gate_config = VisionSafetyGateConfig(
            enabled=bool(safety_gate_enabled),
            attack_pattern_gate=bool(attack_pattern_gate),
            attack_pattern_threshold=float(attack_pattern_threshold),
            low_visibility_gate=bool(low_visibility_gate),
            low_visibility_threshold=float(low_visibility_threshold),
        )
        self.model = model
        self.model_ready = model is not None
        self.last_diagnostics: Dict[str, Any] = {}
        self._load_reason = "ok" if self.model_ready else "missing_model_path"
        self._image_size = (96, 160)

        if self.model is None and self.model_path:
            self._load_torch_checkpoint(self.model_path)

    def _brake(
        self,
        reason: str,
        safety_gate: Optional[Dict[str, Any]] = None,
        command: Optional[str] = None,
        trajectory: Optional[Any] = None,
        raw_control: Optional[Any] = None,
    ) -> carla.VehicleControl:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.last_diagnostics = {
            "model_ready": bool(self.model_ready),
            "model_path": self.model_path,
            "command": command or self.navigation_command,
            "control_mode": self.control_mode,
            "reason": reason,
            "safety_gate": safety_gate,
            "trajectory": trajectory,
            "raw_control": raw_control,
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control

    def _load_torch_checkpoint(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            self.model_ready = False
            self._load_reason = "missing_model_path"
            return

        try:
            import torch

            from .tcp_lite import COMMAND_TO_INDEX, TcpLiteModel

            checkpoint = torch.load(path, map_location=self.device)
            trajectory_points = int(checkpoint.get("trajectory_points", 4))
            self._image_size = tuple(checkpoint.get("image_size", self._image_size))
            model = TcpLiteModel(
                command_count=len(COMMAND_TO_INDEX),
                trajectory_points=trajectory_points,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
        except Exception as exc:  # pragma: no cover - depends on optional torch/checkpoint details.
            self.model = None
            self.model_ready = False
            self._load_reason = f"load_failed:{exc.__class__.__name__}"
            return

        self.model = model
        self.model_ready = True
        self._load_reason = "ok"

    def _predict_with_torch_model(self, rgb: Any, speed_mps: float, command: str) -> tuple[Any, Any]:
        import torch

        array = np.asarray(rgb, dtype=np.float32)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("rgb must be HxWx3")

        image_height, image_width = int(self._image_size[0]), int(self._image_size[1])
        try:
            from PIL import Image

            image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((image_width, image_height))
            array = np.asarray(image, dtype=np.float32)
        except ImportError:
            pass

        if array.max(initial=0.0) > 1.0:
            array = array / 255.0

        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        speed = torch.tensor([[float(speed_mps)]], dtype=torch.float32, device=self.device)
        command_tensor = torch.tensor([command_to_index(command)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            output = self.model(tensor, speed, command_tensor)

        trajectory = output.get("trajectory")
        control = output.get("control")
        if hasattr(trajectory, "detach"):
            trajectory = trajectory.detach().cpu().numpy()[0].tolist()
        if hasattr(control, "detach"):
            control = control.detach().cpu().numpy()[0].tolist()
        return trajectory, control

    @staticmethod
    def _control_values(raw_control: Any) -> tuple[float, float, float]:
        if isinstance(raw_control, dict):
            return (
                float(raw_control.get("steer", 0.0)),
                float(raw_control.get("throttle", 0.0)),
                float(raw_control.get("brake", 0.0)),
            )
        if isinstance(raw_control, Sequence):
            values = list(raw_control)
            return float(values[0]), float(values[1]), float(values[2])
        raise ValueError("control must be a dict or sequence")

    def _trajectory_control(self, trajectory: Any, speed_mps: float) -> tuple[float, float, float]:
        if not isinstance(trajectory, Sequence) or not trajectory:
            raise ValueError("trajectory must contain at least one point")

        point = None
        for candidate in reversed(list(trajectory)):
            if isinstance(candidate, Sequence) and len(candidate) >= 2:
                x = float(candidate[0])
                y = float(candidate[1])
                if x > 0.25:
                    point = (x, y)
                    break
        if point is None:
            raise ValueError("trajectory has no forward point")

        x, y = point
        angle = math.atan2(y, max(0.5, x))
        steer = _clamp(1.25 * angle, -0.65, 0.65)
        target_speed = self.target_speed_mps
        if abs(steer) > 0.35:
            target_speed = min(target_speed, 1.8)

        speed_error = float(target_speed - speed_mps)
        derivative = speed_error - self._last_speed_error
        self._last_speed_error = speed_error
        throttle = _clamp(0.18 * speed_error + 0.04 * derivative, 0.0, 0.42)
        brake = 0.0
        if speed_error < -0.8:
            brake = _clamp((-speed_error) / max(1.0, self.target_speed_mps), 0.0, 0.6)
            throttle = 0.0
        elif speed_mps < 0.3 and target_speed > 0.5:
            throttle = max(throttle, 0.32)
        return steer, throttle, brake

    @staticmethod
    def _interaction_yield_target(
        obs: Dict[str, Any],
        base_target_speed: float,
        speed_mps: float,
    ) -> tuple[float, Dict[str, Any]]:
        hazard = obs.get("interaction_hazard") or {}
        diagnostics: Dict[str, Any] = {"active": False}
        if not isinstance(hazard, dict) or not bool(hazard.get("active", False)):
            diagnostics["reason"] = str(hazard.get("reason", "clear")) if isinstance(hazard, dict) else "clear"
            return float(base_target_speed), diagnostics

        action = str(hazard.get("action", "")).lower()
        requested_target = _as_optional_float(hazard.get("target_speed_mps"))
        if requested_target is None:
            if action == "stop":
                requested_target = 0.0
            elif action == "avoid_left":
                requested_target = min(float(base_target_speed), 2.2)
            else:
                requested_target = min(float(base_target_speed), 1.2)
        target_speed = min(float(base_target_speed), max(0.0, float(requested_target)))
        brake_hint = 0.0
        if action == "stop":
            target_speed = 0.0
            brake_hint = max(0.55, min(1.0, 0.35 + speed_mps / 4.0))
        elif speed_mps > target_speed + (1.0 if action == "avoid_left" else 0.75):
            brake_hint = _clamp((speed_mps - target_speed) / max(1.0, base_target_speed), 0.10, 0.45)

        diagnostics.update(
            {
                "active": True,
                "action": action,
                "actor_type": str(hazard.get("actor_type", "")),
                "actor_id": hazard.get("actor_id"),
                "distance_m": _as_optional_float(hazard.get("distance_m")),
                "local_x_m": _as_optional_float(hazard.get("local_x_m")),
                "local_y_m": _as_optional_float(hazard.get("local_y_m")),
                "base_target_speed_mps": float(base_target_speed),
                "target_speed_mps": float(target_speed),
                "brake_hint": float(brake_hint),
            }
        )
        return float(target_speed), diagnostics

    @staticmethod
    def _is_pedestrian_stop_hazard(obs: Dict[str, Any]) -> bool:
        hazard = obs.get("interaction_hazard") or {}
        if not isinstance(hazard, dict) or not bool(hazard.get("active", False)):
            return False
        if str(hazard.get("action", "")).lower() != "stop":
            return False
        actor_type = str(hazard.get("actor_type", "")).lower()
        role_name = str(hazard.get("role_name", "")).lower()
        return actor_type == "walker" or role_name == "task_walker"

    def _apply_pedestrian_stop_steer_guard(
        self,
        obs: Dict[str, Any],
        control: carla.VehicleControl,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        guard: Dict[str, Any] = {"applied": False}
        if not self._is_pedestrian_stop_hazard(obs):
            if diagnostics is not None:
                diagnostics["pedestrian_stop_steer_guard"] = guard
            return guard

        previous_steer = float(control.steer)
        previous_throttle = float(control.throttle)
        previous_brake = float(control.brake)
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = max(previous_brake, 0.55)
        self._last_steer = 0.0
        self._has_last_steer = True
        guard = {
            "applied": True,
            "previous_steer": previous_steer,
            "previous_throttle": previous_throttle,
            "previous_brake": previous_brake,
        }
        if diagnostics is not None:
            diagnostics["pedestrian_stop_steer_guard"] = guard
            diagnostics["steer"] = float(control.steer)
            diagnostics["throttle"] = float(control.throttle)
            diagnostics["brake"] = float(control.brake)
        return guard

    def _route_reference_control(self, obs: Dict[str, Any]) -> tuple[Optional[carla.VehicleControl], Dict[str, Any]]:
        x = _as_optional_float(obs.get("route_target_local_x"))
        y = _as_optional_float(obs.get("route_target_local_y"))
        if x is None or y is None or x <= 0.25:
            return None, {}

        speed_mps = float(obs.get("speed_mps", 0.0) or 0.0)
        hazard = obs.get("interaction_hazard") or {}
        obstacle_avoidance: Dict[str, Any] = {"applied": False}
        adjusted_y = float(y)
        if isinstance(hazard, dict) and str(hazard.get("action", "")).lower() == "avoid_left":
            lane_width = _as_optional_float(obs.get("lane_width_m")) or 3.5
            requested_avoid = _as_optional_float(hazard.get("avoid_lateral_m"))
            if requested_avoid is None:
                requested_avoid = -min(3.3, max(2.8, float(lane_width) * 0.95))
            avoid_sign = -1.0 if float(requested_avoid) < 0.0 else 1.0
            avoid_magnitude = _clamp(
                abs(float(requested_avoid)),
                2.4,
                min(3.5, max(2.4, float(lane_width))),
            )
            avoid_lateral = avoid_sign * avoid_magnitude
            adjusted_y = min(adjusted_y, avoid_lateral) if avoid_lateral < 0.0 else max(adjusted_y, avoid_lateral)
            obstacle_avoidance = {
                "applied": True,
                "avoid_lateral_m": float(avoid_lateral),
                "original_route_target_local_y": float(y),
            }

        angle = math.atan2(float(adjusted_y), max(0.5, float(x)))
        route_steer = _clamp(1.05 * angle / (math.pi / 2.0), -0.55, 0.55)
        if obstacle_avoidance.get("applied"):
            route_steer = _clamp(route_steer * 1.40, -0.55, 0.55)
        lane_centering_correction = 0.0
        lane_offset_value = None
        try:
            lane_offset_value = float(obs.get("lane_center_offset_m"))
            if abs(lane_offset_value) > self._lane_centering_deadband_m:
                lane_centering_correction = _clamp(
                    -self._lane_centering_gain * lane_offset_value,
                    -self._lane_centering_max_correction,
                    self._lane_centering_max_correction,
                )
        except (TypeError, ValueError):
            lane_offset_value = None
        uav_bev_correction, uav_bev_diagnostics = self._uav_bev_correction(obs)
        steer = _clamp(route_steer + lane_centering_correction + uav_bev_correction, -0.55, 0.55)
        ego_world_x = _as_optional_float(obs.get("ego_world_x"))
        ego_world_y = _as_optional_float(obs.get("ego_world_y"))
        double_yellow_guard: Dict[str, Any] = {"applied": False}
        boundary_speed_limit = None
        in_obstacle_corridor = (
            ego_world_y is not None and 45.0 <= float(ego_world_y) <= 92.0
        ) or (ego_world_y is None and bool(obstacle_avoidance.get("applied")))
        if ego_world_x is not None and in_obstacle_corridor and ego_world_x <= -45.35:
            guarded_steer = 0.18 if ego_world_x <= -46.2 else 0.08
            steer = max(float(steer), guarded_steer)
            if ego_world_x <= -46.2:
                boundary_speed_limit = 1.2
            double_yellow_guard = {
                "applied": True,
                "reason": "obstacle_corridor_boundary",
                "ego_world_x": float(ego_world_x),
                "ego_world_y": None if ego_world_y is None else float(ego_world_y),
                "guarded_steer": float(guarded_steer),
                "speed_limit_mps": boundary_speed_limit,
            }
        target_speed = float(self.target_speed_mps)
        if boundary_speed_limit is not None:
            target_speed = min(target_speed, float(boundary_speed_limit))
        if abs(steer) > 0.30:
            target_speed = min(target_speed, 1.8)
        target_speed, interaction_diagnostics = self._interaction_yield_target(obs, target_speed, speed_mps)
        speed_error = target_speed - speed_mps
        throttle = _clamp(0.22 * speed_error, 0.0, 0.42)
        brake = 0.0
        if speed_error < -0.8:
            brake = _clamp((-speed_error) / max(1.0, self.target_speed_mps), 0.0, 0.5)
            throttle = 0.0
        if interaction_diagnostics.get("active"):
            brake = max(brake, float(interaction_diagnostics.get("brake_hint", 0.0) or 0.0))
            if brake > 0.0:
                throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        diagnostics = {
            "reason": "route_reference_available",
            "lane_confidence": 1.0,
            "route_target_local_x": float(x),
            "route_target_local_y": float(adjusted_y),
            "route_target_source": str(obs.get("route_target_source", "")),
            "route_steer": float(route_steer),
            "lane_center_offset_m": lane_offset_value,
            "lane_centering_correction": float(lane_centering_correction),
            "obstacle_avoidance": obstacle_avoidance,
            "double_yellow_guard": double_yellow_guard,
            "uav_bev_fusion": uav_bev_diagnostics,
            "speed_mps": float(speed_mps),
            "target_speed_mps": float(target_speed),
            "interaction_yield": interaction_diagnostics,
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        self._apply_pedestrian_stop_steer_guard(obs, control, diagnostics)
        return control, diagnostics

    def _fallback_reference_control(self, obs: Dict[str, Any]) -> tuple[carla.VehicleControl, Dict[str, Any]]:
        route_control, route_diagnostics = self._route_reference_control(obs)
        if route_control is not None:
            return route_control, route_diagnostics
        control = self.fallback_policy.predict(obs)
        return control, dict(getattr(self.fallback_policy, "last_diagnostics", {}) or {})

    @staticmethod
    def _allow_route_reference(obs: Dict[str, Any], in_junction: bool) -> bool:
        if not in_junction:
            return True
        route_source = str(obs.get("route_target_source", "")).lower()
        if route_source in {"junction_turn_reference", "junction_heading_hold"}:
            return True
        command = str(obs.get("navigation_command", "")).lower()
        return route_source == "waypoint_next" and command in {"lane_follow", "straight"}

    def _uav_bev_correction(self, obs: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "enabled": bool(self.uav_bev_fusion_enabled),
            "mode": self.uav_fusion_mode,
            "applied": False,
            "steer_correction": 0.0,
        }
        if not self.uav_bev_fusion_enabled:
            diagnostics["reason"] = "disabled"
            return 0.0, diagnostics
        if self.uav_fusion_mode == "learned":
            correction, planner_diagnostics = self.uav_fusion_planner.predict(obs)
            diagnostics.update(planner_diagnostics)
            diagnostics["mode"] = self.uav_fusion_mode
            if bool(obs.get("in_junction", False)) and abs(correction) >= 1.0e-4:
                correction *= 0.5
                diagnostics["junction_scale"] = 0.5
                diagnostics["steer_correction"] = float(correction)
            return float(correction), diagnostics

        feature = obs.get("uav_bev") or {}
        diagnostics["available"] = bool(feature.get("available", False))
        if not diagnostics["available"]:
            diagnostics["reason"] = str(feature.get("reason", "unavailable"))
            return 0.0, diagnostics

        try:
            confidence = float(feature.get("road_confidence", 0.0) or 0.0)
            center_bias = float(feature.get("center_bias", 0.0) or 0.0)
        except (TypeError, ValueError):
            diagnostics["reason"] = "invalid_feature"
            return 0.0, diagnostics

        diagnostics["road_confidence"] = float(confidence)
        diagnostics["center_bias"] = float(center_bias)
        diagnostics["min_confidence"] = float(self.uav_bev_min_confidence)
        if confidence < self.uav_bev_min_confidence:
            diagnostics["reason"] = "low_confidence"
            return 0.0, diagnostics

        correction = _clamp(
            self.uav_bev_steer_gain * center_bias,
            -self.uav_bev_max_steer_correction,
            self.uav_bev_max_steer_correction,
        )
        if bool(obs.get("in_junction", False)):
            correction *= 0.5
            diagnostics["junction_scale"] = 0.5
        if abs(correction) < 1.0e-4:
            diagnostics["reason"] = "zero_bias"
            return 0.0, diagnostics

        diagnostics["applied"] = True
        diagnostics["reason"] = "ok"
        diagnostics["steer_correction"] = float(correction)
        diagnostics["max_steer_correction"] = float(self.uav_bev_max_steer_correction)
        return float(correction), diagnostics

    def _stabilize_control(
        self,
        steer: float,
        throttle: float,
        brake: float,
        obs: Dict[str, Any],
        command: str,
    ) -> tuple[float, float, float, Dict[str, Any]]:
        steering_command = str(command or self.navigation_command).lower()
        in_junction = bool(obs.get("in_junction", False))
        speed_mps = float(obs.get("speed_mps", 0.0) or 0.0)
        lane_offset = obs.get("lane_center_offset_m")
        lane_correction = 0.0
        junction_low_speed_recovery = False
        try:
            lane_offset_value = float(lane_offset)
            if abs(lane_offset_value) > self._lane_centering_deadband_m and not in_junction:
                lane_correction = _clamp(
                    -self._lane_centering_gain * lane_offset_value,
                    -self._lane_centering_max_correction,
                    self._lane_centering_max_correction,
                )
        except (TypeError, ValueError):
            lane_offset_value = None

        uav_bev_correction, uav_bev_diagnostics = self._uav_bev_correction(obs)
        target_steer = _clamp(float(steer) + lane_correction + uav_bev_correction, -1.0, 1.0)
        if steering_command in {"lane_follow", "straight"}:
            steer_limit = self._junction_steer_limit if in_junction else self._lane_follow_steer_limit
            target_steer = _clamp(target_steer, -steer_limit, steer_limit)
        if in_junction and steering_command in {"lane_follow", "straight"}:
            if speed_mps < self._junction_low_speed_recovery_mps and float(brake) < 0.1:
                throttle = max(float(throttle), self._junction_low_speed_recovery_throttle)
                junction_low_speed_recovery = True
            else:
                throttle = min(float(throttle), 0.24)
            if abs(target_steer) < 0.20:
                target_steer *= 0.65

        previous_steer = self._last_steer if self._has_last_steer else 0.0
        if self._has_last_steer:
            smoothed = self._steer_smoothing * previous_steer + (1.0 - self._steer_smoothing) * target_steer
        else:
            smoothed = target_steer
            self._has_last_steer = True
        delta = _clamp(
            smoothed - previous_steer,
            -self._steer_rate_limit,
            self._steer_rate_limit,
        )
        stabilized_steer = previous_steer + delta
        if abs(stabilized_steer) < self._steer_deadband:
            stabilized_steer = 0.0
        self._last_steer = _clamp(stabilized_steer, -1.0, 1.0)

        diagnostics = {
            "enabled": True,
            "input_steer": float(steer),
            "target_steer": float(target_steer),
            "output_steer": float(self._last_steer),
            "lane_center_offset_m": lane_offset_value,
            "lane_centering_correction": float(lane_correction),
            "in_junction": bool(in_junction),
            "speed_mps": float(speed_mps),
            "junction_low_speed_recovery": bool(junction_low_speed_recovery),
            "steer_rate_limit": float(self._steer_rate_limit),
            "lane_follow_steer_limit": float(self._lane_follow_steer_limit),
            "steer_deadband": float(self._steer_deadband),
            "uav_bev_fusion": uav_bev_diagnostics,
        }
        return self._last_steer, float(throttle), float(brake), diagnostics

    def _fallback_control(
        self,
        obs: Dict[str, Any],
        reason: str,
        safety_gate: Dict[str, Any],
        command: str,
        trajectory: Any,
        raw_control: Any,
        control: Optional[carla.VehicleControl] = None,
        fallback_diagnostics: Optional[Dict[str, Any]] = None,
        refresh_latch: bool = True,
    ) -> carla.VehicleControl:
        if control is None:
            control = self.fallback_policy.predict(obs)
        if fallback_diagnostics is None:
            fallback_diagnostics = dict(getattr(self.fallback_policy, "last_diagnostics", {}) or {})
        if refresh_latch:
            self._fallback_frames_left = int(self._fallback_latch_frames)
        uav_bev_diagnostics = dict(fallback_diagnostics.get("uav_bev_fusion") or {})
        if not uav_bev_diagnostics:
            _, uav_bev_diagnostics = self._uav_bev_correction(obs)
        pedestrian_stop_steer_guard = self._apply_pedestrian_stop_steer_guard(obs, control, fallback_diagnostics)
        self.last_diagnostics = {
            "model_ready": True,
            "model_path": self.model_path,
            "command": command,
            "control_mode": self.control_mode,
            "reason": "fallback_confidence_gate",
            "fallback": {
                "reason": reason,
                "diagnostics": fallback_diagnostics,
            },
            "safety_gate": safety_gate,
            "trajectory": trajectory,
            "raw_control": raw_control,
            "uav_bev_fusion": uav_bev_diagnostics,
            "pedestrian_stop_steer_guard": pedestrian_stop_steer_guard,
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control

    def predict(self, obs: Dict[str, Any]) -> carla.VehicleControl:
        rgb = obs.get("rgb")
        speed_mps = float(obs.get("speed_mps", 0.0))
        command = str(obs.get("navigation_command", self.navigation_command))
        in_junction = bool(obs.get("in_junction", False))

        if rgb is None:
            return self._brake("missing_rgb", command=command)
        if not self.model_ready or self.model is None:
            return self._brake(self._load_reason or "missing_model_path", command=command)

        if self.control_mode == "trajectory" and not in_junction and not self.safety_gate_config.attack_pattern_gate:
            fallback_control, fallback_diagnostics = self._fallback_reference_control(obs)
            fallback_lane_confidence = float(fallback_diagnostics.get("lane_confidence", 0.0) or 0.0)
            if fallback_lane_confidence >= 0.01:
                safety_gate = {
                    "enabled": bool(self.safety_gate_config.enabled),
                    "blocked": False,
                    "reason": "rgb_reference_shortcut",
                    "detector_obstacle": bool(obs.get("vision_obstacle", False)),
                    "attack_pattern_score": None,
                    "attack_pattern_gate": bool(self.safety_gate_config.attack_pattern_gate),
                    "attack_pattern_threshold": float(self.safety_gate_config.attack_pattern_threshold),
                }
                return self._fallback_control(
                    obs,
                    str(fallback_diagnostics.get("reason", "rgb_lane_reference_available")),
                    safety_gate,
                    command,
                    None,
                    None,
                    control=fallback_control,
                    fallback_diagnostics=fallback_diagnostics,
                )

        safety_gate = evaluate_vision_safety_gate(
            rgb,
            obs.get("vision_detector", {}),
            self.safety_gate_config,
        )
        if safety_gate.get("blocked"):
            return self._brake(str(safety_gate.get("reason", "safety_gate")), safety_gate=safety_gate, command=command)

        if self.control_mode == "trajectory" and self._allow_route_reference(obs, in_junction):
            fallback_control, fallback_diagnostics = self._fallback_reference_control(obs)
            fallback_lane_confidence = float(fallback_diagnostics.get("lane_confidence", 0.0) or 0.0)
            if fallback_lane_confidence >= 0.01:
                return self._fallback_control(
                    obs,
                    str(fallback_diagnostics.get("reason", "rgb_lane_reference_available")),
                    safety_gate,
                    command,
                    None,
                    None,
                    control=fallback_control,
                    fallback_diagnostics=fallback_diagnostics,
                )

        try:
            if hasattr(self.model, "predict"):
                trajectory, raw_control = self.model.predict(rgb=rgb, speed_mps=speed_mps, command=command)
            else:
                trajectory, raw_control = self._predict_with_torch_model(rgb, speed_mps, command)

            steer_raw, throttle_raw, brake_raw = self._control_values(raw_control)
            if self.control_mode == "direct":
                steer, throttle, brake = steer_raw, throttle_raw, brake_raw
                stabilization_diagnostics: Dict[str, Any] = {"enabled": False}
            else:
                steer, throttle, brake = self._trajectory_control(trajectory, speed_mps)
                steer, throttle, brake, stabilization_diagnostics = self._stabilize_control(
                    steer,
                    throttle,
                    brake,
                    obs,
                    command,
                )
                if self.control_mode == "trajectory" and not bool(obs.get("in_junction", False)):
                    fallback_control, fallback_diagnostics = self._fallback_reference_control(obs)
                    fallback_lane_confidence = float(fallback_diagnostics.get("lane_confidence", 0.0) or 0.0)
                    if self._fallback_frames_left > 0:
                        self._fallback_frames_left -= 1
                        return self._fallback_control(
                            obs,
                            "fallback_latched",
                            safety_gate,
                            command,
                            trajectory,
                            raw_control,
                            control=fallback_control,
                            fallback_diagnostics=fallback_diagnostics,
                            refresh_latch=False,
                        )
                    fallback_disagreement = abs(float(fallback_control.steer) - float(steer))
                    if fallback_lane_confidence >= 0.01:
                        fallback_reason = "rgb_lane_reference_available"
                        if fallback_disagreement > self._fallback_disagreement_threshold:
                            fallback_reason = "rgb_lane_reference_disagreement"
                        return self._fallback_control(
                            obs,
                            fallback_reason,
                            safety_gate,
                            command,
                            trajectory,
                            raw_control,
                            control=fallback_control,
                            fallback_diagnostics=fallback_diagnostics,
                        )
                    disagreement = abs(float(steer) - float(steer_raw))
                    if disagreement > self._fallback_disagreement_threshold:
                        return self._fallback_control(
                            obs,
                            "model_trajectory_control_disagreement",
                            safety_gate,
                            command,
                            trajectory,
                            raw_control,
                        )
                    if speed_mps < 0.25 and throttle > 0.2 and brake < 0.1:
                        self._stuck_frames += 1
                    else:
                        self._stuck_frames = 0
                    if self._stuck_frames >= self._fallback_stuck_frame_threshold:
                        return self._fallback_control(
                            obs,
                            "model_stuck",
                            safety_gate,
                            command,
                            trajectory,
                            raw_control,
                        )
        except Exception as exc:
            return self._brake(f"{type(exc).__name__}: {exc}", safety_gate=safety_gate, command=command)

        control = carla.VehicleControl()
        control.steer = _clamp(steer, -1.0, 1.0)
        control.throttle = _clamp(throttle, 0.0, 1.0)
        control.brake = _clamp(brake, 0.0, 1.0)
        pedestrian_stop_steer_guard = self._apply_pedestrian_stop_steer_guard(obs, control)
        self.last_diagnostics = {
            "model_ready": True,
            "model_path": self.model_path,
            "command": command,
            "control_mode": self.control_mode,
            "reason": "ok",
            "safety_gate": safety_gate,
            "stabilization": stabilization_diagnostics,
            "trajectory": trajectory,
            "raw_control": raw_control,
            "uav_bev_fusion": stabilization_diagnostics.get("uav_bev_fusion"),
            "pedestrian_stop_steer_guard": pedestrian_stop_steer_guard,
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control
