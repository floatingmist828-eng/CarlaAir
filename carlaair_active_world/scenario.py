from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .geometry import CandidateViewpoint, Vector3


@dataclass
class ScenarioConfig:
    name: str
    map_name: str = "Town10HD"
    ego_blueprint: str = "vehicle.tesla.model3"
    ego_spawn_index: int = 0
    ego_sensor_mode: str = "autopilot"
    ego_control_mode: str = "autopilot"
    ego_drive_hz: float = 8.0
    ego_target_speed_mps: float = 8.0
    ego_lookahead_m: float = 10.0
    traffic_vehicles: int = 0
    traffic_walkers: int = 0
    duration_sec: float = 30.0
    step_sec: float = 0.5
    future_horizon_sec: float = 3.0
    sample_only_near_hotspot: bool = True
    sample_hotspot_radius_m: float = 70.0
    sample_min_interval_sec: float = 0.5
    vehicle_sensor_limit: int = 6
    weather_preset: str = "none"
    vision_attack: str = "none"
    vision_attack_intensity: float = 1.0
    vision_detector_model_path: str = ""
    vision_detector_confidence: float = 0.35
    uav_enabled: bool = True
    uav_name: str = "SimpleFlight"
    uav_altitude: float = 18.0
    uav_back_distance: float = 8.0
    uav_auto_patrol_enabled: bool = False
    uav_patrol_interval_sec: float = 4.0
    candidate_offsets: List[CandidateViewpoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        candidates = []
        for item in data.get("candidate_offsets", []):
            candidates.append(
                CandidateViewpoint(
                    name=item["name"],
                    local_offset=Vector3(
                        float(item["x"]),
                        float(item["y"]),
                        float(item.get("z", 0.0)),
                    ),
                    weight=float(item.get("weight", 1.0)),
                )
            )
        return cls(
            name=str(data["name"]),
            map_name=str(data.get("map_name", "Town10HD")),
            ego_blueprint=str(data.get("ego_blueprint", "vehicle.tesla.model3")),
            ego_spawn_index=int(data.get("ego_spawn_index", 0)),
            ego_sensor_mode=str(data.get("ego_sensor_mode", "autopilot")),
            ego_control_mode=str(data.get("ego_control_mode", data.get("ego_sensor_mode", "autopilot"))),
            ego_drive_hz=float(data.get("ego_drive_hz", 8.0)),
            ego_target_speed_mps=float(data.get("ego_target_speed_mps", 8.0)),
            ego_lookahead_m=float(data.get("ego_lookahead_m", 10.0)),
            traffic_vehicles=int(data.get("traffic_vehicles", 0)),
            traffic_walkers=int(data.get("traffic_walkers", 0)),
            duration_sec=float(data.get("duration_sec", 30.0)),
            step_sec=float(data.get("step_sec", 0.5)),
            future_horizon_sec=float(data.get("future_horizon_sec", 3.0)),
            sample_only_near_hotspot=bool(data.get("sample_only_near_hotspot", True)),
            sample_hotspot_radius_m=float(data.get("sample_hotspot_radius_m", 70.0)),
            sample_min_interval_sec=float(data.get("sample_min_interval_sec", 0.5)),
            vehicle_sensor_limit=int(data.get("vehicle_sensor_limit", 6)),
            weather_preset=str(data.get("weather_preset", "none")),
            vision_attack=str(data.get("vision_attack", "none")),
            vision_attack_intensity=float(data.get("vision_attack_intensity", 1.0)),
            vision_detector_model_path=str(data.get("vision_detector_model_path", "")),
            vision_detector_confidence=float(data.get("vision_detector_confidence", 0.35)),
            uav_enabled=bool(data.get("uav_enabled", True)),
            uav_name=str(data.get("uav_name", "SimpleFlight")),
            uav_altitude=float(data.get("uav_altitude", 18.0)),
            uav_back_distance=float(data.get("uav_back_distance", 8.0)),
            uav_auto_patrol_enabled=bool(data.get("uav_auto_patrol_enabled", False)),
            uav_patrol_interval_sec=float(data.get("uav_patrol_interval_sec", 4.0)),
            candidate_offsets=candidates,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "map_name": self.map_name,
            "ego_blueprint": self.ego_blueprint,
            "ego_spawn_index": self.ego_spawn_index,
            "ego_sensor_mode": self.ego_sensor_mode,
            "ego_control_mode": self.ego_control_mode,
            "ego_drive_hz": self.ego_drive_hz,
            "ego_target_speed_mps": self.ego_target_speed_mps,
            "ego_lookahead_m": self.ego_lookahead_m,
            "traffic_vehicles": self.traffic_vehicles,
            "traffic_walkers": self.traffic_walkers,
            "duration_sec": self.duration_sec,
            "step_sec": self.step_sec,
            "future_horizon_sec": self.future_horizon_sec,
            "sample_only_near_hotspot": self.sample_only_near_hotspot,
            "sample_hotspot_radius_m": self.sample_hotspot_radius_m,
            "sample_min_interval_sec": self.sample_min_interval_sec,
            "vehicle_sensor_limit": self.vehicle_sensor_limit,
            "weather_preset": self.weather_preset,
            "vision_attack": self.vision_attack,
            "vision_attack_intensity": self.vision_attack_intensity,
            "vision_detector_model_path": self.vision_detector_model_path,
            "vision_detector_confidence": self.vision_detector_confidence,
            "uav_enabled": self.uav_enabled,
            "uav_name": self.uav_name,
            "uav_altitude": self.uav_altitude,
            "uav_back_distance": self.uav_back_distance,
            "uav_auto_patrol_enabled": self.uav_auto_patrol_enabled,
            "uav_patrol_interval_sec": self.uav_patrol_interval_sec,
            "candidate_offsets": [c.to_dict() for c in self.candidate_offsets],
        }

    @classmethod
    def with_default_candidates(
        cls,
        name: str,
            map_name: str = "Town10HD",
    ) -> "ScenarioConfig":
        candidates = [
            CandidateViewpoint("front_high", Vector3(18.0, 0.0, 10.0), 1.0),
            CandidateViewpoint("front_left", Vector3(15.0, 8.0, 10.0), 1.0),
            CandidateViewpoint("front_right", Vector3(15.0, -8.0, 10.0), 1.0),
            CandidateViewpoint("top", Vector3(0.0, 0.0, 22.0), 0.8),
            CandidateViewpoint("rear_high", Vector3(-10.0, 0.0, 12.0), 0.7),
            CandidateViewpoint("left_high", Vector3(0.0, 14.0, 12.0), 0.8),
            CandidateViewpoint("right_high", Vector3(0.0, -14.0, 12.0), 0.8),
        ]
        return cls(name=name, map_name=map_name, candidate_offsets=candidates)


def ensure_scenario_dir(path: str | Path) -> Path:
    scenario_path = Path(path)
    scenario_path.mkdir(parents=True, exist_ok=True)
    return scenario_path
