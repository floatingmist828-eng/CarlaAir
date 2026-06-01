from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import carla
import numpy as np

from .base import VisionPolicy
from .safety_gate import VisionSafetyGateConfig, evaluate_vision_safety_gate
from .simple_lane import SimpleLaneVisionPolicy
from .tcp_lite import command_to_index


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


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
    ) -> None:
        self.model_path = str(model_path or "")
        self.device = str(device)
        self.navigation_command = str(navigation_command)
        self.target_speed_mps = float(target_speed_mps)
        self.control_mode = str(control_mode or "trajectory").lower()
        self._last_speed_error = 0.0
        self._stuck_frames = 0
        self._fallback_frames_left = 0
        self._fallback_disagreement_threshold = 0.25
        self._fallback_stuck_frame_threshold = 8
        self._fallback_latch_frames = 24
        self._has_last_steer = False
        self._last_steer = 0.0
        self._steer_smoothing = 0.58
        self._steer_rate_limit = 0.09
        self._lane_centering_gain = 0.10
        self._lane_centering_deadband_m = 0.15
        self._lane_centering_max_correction = 0.18
        self._junction_steer_limit = 0.34
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
        lane_offset = obs.get("lane_center_offset_m")
        lane_correction = 0.0
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

        target_steer = _clamp(float(steer) + lane_correction, -1.0, 1.0)
        if in_junction and steering_command in {"lane_follow", "straight"}:
            target_steer = _clamp(target_steer, -self._junction_steer_limit, self._junction_steer_limit)
            throttle = min(float(throttle), 0.24)
            if abs(target_steer) < 0.20:
                target_steer *= 0.65

        if self._has_last_steer:
            smoothed = self._steer_smoothing * self._last_steer + (1.0 - self._steer_smoothing) * target_steer
            delta = _clamp(
                smoothed - self._last_steer,
                -self._steer_rate_limit,
                self._steer_rate_limit,
            )
            stabilized_steer = self._last_steer + delta
        else:
            stabilized_steer = target_steer
            self._has_last_steer = True
        self._last_steer = _clamp(stabilized_steer, -1.0, 1.0)

        diagnostics = {
            "enabled": True,
            "input_steer": float(steer),
            "target_steer": float(target_steer),
            "output_steer": float(self._last_steer),
            "lane_center_offset_m": lane_offset_value,
            "lane_centering_correction": float(lane_correction),
            "in_junction": bool(in_junction),
            "steer_rate_limit": float(self._steer_rate_limit),
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
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control

    def predict(self, obs: Dict[str, Any]) -> carla.VehicleControl:
        rgb = obs.get("rgb")
        speed_mps = float(obs.get("speed_mps", 0.0))
        command = str(obs.get("navigation_command", self.navigation_command))

        if rgb is None:
            return self._brake("missing_rgb", command=command)
        if not self.model_ready or self.model is None:
            return self._brake(self._load_reason or "missing_model_path", command=command)

        if self.control_mode == "trajectory" and not self.safety_gate_config.attack_pattern_gate:
            fallback_control = self.fallback_policy.predict(obs)
            fallback_diagnostics = dict(getattr(self.fallback_policy, "last_diagnostics", {}) or {})
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
                    "rgb_lane_reference_available",
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

        if self.control_mode == "trajectory":
            fallback_control = self.fallback_policy.predict(obs)
            fallback_diagnostics = dict(getattr(self.fallback_policy, "last_diagnostics", {}) or {})
            fallback_lane_confidence = float(fallback_diagnostics.get("lane_confidence", 0.0) or 0.0)
            if fallback_lane_confidence >= 0.01:
                return self._fallback_control(
                    obs,
                    "rgb_lane_reference_available",
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
                if self.control_mode == "trajectory":
                    fallback_control = self.fallback_policy.predict(obs)
                    fallback_diagnostics = dict(getattr(self.fallback_policy, "last_diagnostics", {}) or {})
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
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control
