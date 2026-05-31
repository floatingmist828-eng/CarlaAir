from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import carla
import numpy as np

from .base import VisionPolicy
from .safety_gate import VisionSafetyGateConfig, evaluate_vision_safety_gate
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
        model: Optional[Any] = None,
    ) -> None:
        self.model_path = str(model_path or "")
        self.device = str(device)
        self.navigation_command = str(navigation_command)
        self.safety_gate_config = VisionSafetyGateConfig(
            enabled=bool(safety_gate_enabled),
            attack_pattern_gate=bool(attack_pattern_gate),
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

    def predict(self, obs: Dict[str, Any]) -> carla.VehicleControl:
        rgb = obs.get("rgb")
        speed_mps = float(obs.get("speed_mps", 0.0))
        command = str(obs.get("navigation_command", self.navigation_command))

        if rgb is None:
            return self._brake("missing_rgb", command=command)
        if not self.model_ready or self.model is None:
            return self._brake(self._load_reason or "missing_model_path", command=command)

        safety_gate = evaluate_vision_safety_gate(
            rgb,
            obs.get("vision_detector", {}),
            self.safety_gate_config,
        )
        if safety_gate.get("blocked"):
            return self._brake(str(safety_gate.get("reason", "safety_gate")), safety_gate=safety_gate, command=command)

        if hasattr(self.model, "predict"):
            trajectory, raw_control = self.model.predict(rgb=rgb, speed_mps=speed_mps, command=command)
        else:
            trajectory, raw_control = self._predict_with_torch_model(rgb, speed_mps, command)

        steer_raw, throttle_raw, brake_raw = self._control_values(raw_control)
        control = carla.VehicleControl()
        control.steer = _clamp(steer_raw, -1.0, 1.0)
        control.throttle = _clamp(throttle_raw, 0.0, 1.0)
        control.brake = _clamp(brake_raw, 0.0, 1.0)
        self.last_diagnostics = {
            "model_ready": True,
            "model_path": self.model_path,
            "command": command,
            "reason": "ok",
            "safety_gate": safety_gate,
            "trajectory": trajectory,
            "raw_control": raw_control,
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control
