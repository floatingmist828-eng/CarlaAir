from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import carla
import numpy as np

from .sensors import VehicleSensorRig
from .vision_models import SimpleLaneVisionPolicy, VisionPolicy


class VisionEgoDriver:
    sensor_rig_class = VehicleSensorRig

    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Actor,
        target_speed_mps: float = 4.0,
        policy: Optional[VisionPolicy] = None,
        use_semantic: bool = True,
        vision_attack: str = "none",
        vision_attack_intensity: float = 1.0,
        vision_detector_model_path: str = "",
        vision_detector_confidence: float = 0.35,
        detector: Optional[Any] = None,
        navigation_command: str = "lane_follow",
    ) -> None:
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.use_semantic = bool(use_semantic)
        self.navigation_command = navigation_command
        self.sensor_rig = self.sensor_rig_class(
            world,
            ego_vehicle,
            "ego_vision",
            vision_attack=vision_attack,
            vision_attack_intensity=vision_attack_intensity,
            disable_semantic=not bool(use_semantic),
        )
        self.sensor_rig.spawn()
        self.policy = policy or SimpleLaneVisionPolicy(target_speed_mps=target_speed_mps)
        self.detector = detector
        self._detector_diagnostics: Dict[str, Any] = {}
        if self.detector is None and vision_detector_model_path:
            self.detector = self._load_detector(vision_detector_model_path, vision_detector_confidence)
        self.last_diagnostics: Dict[str, Any] = {}

    def _load_detector(self, model_path: str, confidence: float) -> Optional[Any]:
        path = Path(model_path)
        if not path.exists():
            self._detector_diagnostics = {
                "available": False,
                "reason": "missing_model_path",
                "model_path": str(path),
            }
            return None
        try:
            from .vision_models.yolo11_obstacle import UltralyticsObstacleDetector

            detector = UltralyticsObstacleDetector(str(path), confidence=confidence)
            self._detector_diagnostics = {
                "available": True,
                "model_path": str(path),
            }
            return detector
        except Exception as exc:
            self._detector_diagnostics = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "model_path": str(path),
            }
            return None

    @staticmethod
    def _vehicle_speed_mps(vehicle: carla.Actor) -> float:
        velocity = vehicle.get_velocity()
        return float(np.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z))

    def predict(self, ego_vehicle: Optional[carla.Actor] = None, world: Optional[carla.World] = None) -> carla.VehicleControl:
        vehicle = ego_vehicle or self.ego_vehicle
        frames = self.sensor_rig.snapshot()
        obs = {
            "rgb": frames.get("rgb"),
            "depth": frames.get("depth"),
            "semantic": frames.get("semantic") if self.use_semantic else None,
            "speed_mps": self._vehicle_speed_mps(vehicle),
            "navigation_command": self.navigation_command,
        }
        detector_diagnostics = dict(self._detector_diagnostics)
        vision_obstacle = False
        if self.detector is not None and obs["rgb"] is not None:
            try:
                detector_diagnostics = dict(self.detector.predict(obs["rgb"]) or {})
            except Exception as exc:
                detector_diagnostics = {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            vision_obstacle = bool(detector_diagnostics.get("obstacle", False))
        obs["vision_detector"] = detector_diagnostics
        obs["vision_obstacle"] = bool(vision_obstacle)
        control = self.policy.predict(obs)
        self.last_diagnostics = dict(getattr(self.policy, "last_diagnostics", {}))
        self.last_diagnostics["vision_detector"] = detector_diagnostics
        self.last_diagnostics["vision_obstacle"] = bool(vision_obstacle)
        return control

    def destroy(self) -> None:
        self.sensor_rig.destroy()
