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
    ego_spawn_forward_m: float = 0.0
    ego_sensor_mode: str = "autopilot"
    ego_control_mode: str = "autopilot"
    ego_drive_hz: float = 8.0
    ego_target_speed_mps: float = 8.0
    ego_lookahead_m: float = 10.0
    traffic_vehicles: int = 0
    traffic_walkers: int = 0
    traffic_spawn_start_index: int = -1
    traffic_spawn_indices: List[int] = field(default_factory=list)
    traffic_route_commands: List[str] = field(default_factory=list)
    traffic_spawn_delay_sec: float = 0.0
    traffic_speed_difference: float = 25.0
    obstacle_vehicles: int = 0
    obstacle_spawn_delay_sec: float = 0.0
    obstacle_blueprint: str = "vehicle.dodge.charger_police_2020"
    obstacle_anchor_x: float = 0.0
    obstacle_anchor_y: float = 0.0
    obstacle_anchor_z: float = 0.6
    obstacle_anchor_yaw_deg: float = 0.0
    obstacle_forward_offsets_m: List[float] = field(default_factory=list)
    obstacle_lateral_offsets_m: List[float] = field(default_factory=list)
    obstacle_yaw_offsets_deg: List[float] = field(default_factory=list)
    walker_spawn_start_index: int = -1
    walker_spawn_indices: List[int] = field(default_factory=list)
    walker_spawn_delay_sec: float = 0.0
    walker_crossing_distance_m: float = 18.0
    walker_crossing_offsets_m: List[float] = field(default_factory=list)
    walker_speed_mps: float = 1.4
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
    vision_model_path: str = ""
    vision_model_device: str = "cpu"
    vision_model_control_mode: str = "trajectory"
    vision_navigation_command: str = "lane_follow"
    vision_safety_gate_enabled: bool = True
    vision_attack_pattern_gate: bool = False
    vision_attack_pattern_threshold: float = 0.35
    vision_low_visibility_gate: bool = False
    vision_low_visibility_threshold: float = 0.12
    vision_first_junction_command: str = ""
    vision_junction_command_sequence: List[str] = field(default_factory=list)
    vision_junction_command_hold_sec: float = 4.0
    vision_junction_command_hold_until_exit: bool = False
    uav_enabled: bool = True
    uav_control_enabled: bool = True
    uav_name: str = "SimpleFlight"
    uav_altitude: float = 18.0
    uav_back_distance: float = 8.0
    uav_auto_patrol_enabled: bool = False
    uav_patrol_interval_sec: float = 4.0
    uav_bev_fusion_enabled: bool = False
    uav_bev_camera_name: str = "front_center"
    uav_bev_refresh_hz: float = 2.0
    uav_bev_min_confidence: float = 0.20
    uav_bev_steer_gain: float = 0.08
    uav_bev_max_steer_correction: float = 0.08
    uav_fusion_mode: str = "none"
    uav_fusion_planner_path: str = ""
    uav_fusion_planner_gain: float = 1.0
    uav_fusion_max_steer_correction: float = 0.08
    uav_fusion_min_confidence: float = 0.20
    experiment_group: str = ""
    scenario_stage: str = ""
    scenario_complexity: List[str] = field(default_factory=list)
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
        raw_mode = data.get("uav_fusion_mode")
        if raw_mode is None or str(raw_mode).strip() == "":
            uav_fusion_mode = "rule" if bool(data.get("uav_bev_fusion_enabled", False)) else "none"
        else:
            uav_fusion_mode = str(raw_mode).strip().lower()
        if uav_fusion_mode not in {"none", "rule", "learned"}:
            raise ValueError("uav_fusion_mode must be one of: none, rule, learned")
        complexity = data.get("scenario_complexity", [])
        if isinstance(complexity, str):
            complexity = [complexity]
        traffic_spawn_indices = data.get("traffic_spawn_indices", [])
        if traffic_spawn_indices is None:
            traffic_spawn_indices = []
        traffic_route_commands = data.get("traffic_route_commands", [])
        if traffic_route_commands is None:
            traffic_route_commands = []
        if isinstance(traffic_route_commands, str):
            traffic_route_commands = [traffic_route_commands]
        obstacle_forward_offsets = data.get("obstacle_forward_offsets_m", [])
        if obstacle_forward_offsets is None:
            obstacle_forward_offsets = []
        obstacle_lateral_offsets = data.get("obstacle_lateral_offsets_m", [])
        if obstacle_lateral_offsets is None:
            obstacle_lateral_offsets = []
        obstacle_yaw_offsets = data.get("obstacle_yaw_offsets_deg", [])
        if obstacle_yaw_offsets is None:
            obstacle_yaw_offsets = []
        walker_spawn_indices = data.get("walker_spawn_indices", [])
        if walker_spawn_indices is None:
            walker_spawn_indices = []
        walker_crossing_offsets = data.get("walker_crossing_offsets_m", [])
        if walker_crossing_offsets is None:
            walker_crossing_offsets = []
        junction_command_sequence = data.get("vision_junction_command_sequence", [])
        if junction_command_sequence is None:
            junction_command_sequence = []
        if isinstance(junction_command_sequence, str):
            junction_command_sequence = [junction_command_sequence]
        return cls(
            name=str(data["name"]),
            map_name=str(data.get("map_name", "Town10HD")),
            ego_blueprint=str(data.get("ego_blueprint", "vehicle.tesla.model3")),
            ego_spawn_index=int(data.get("ego_spawn_index", 0)),
            ego_spawn_forward_m=float(data.get("ego_spawn_forward_m", 0.0)),
            ego_sensor_mode=str(data.get("ego_sensor_mode", "autopilot")),
            ego_control_mode=str(data.get("ego_control_mode", data.get("ego_sensor_mode", "autopilot"))),
            ego_drive_hz=float(data.get("ego_drive_hz", 8.0)),
            ego_target_speed_mps=float(data.get("ego_target_speed_mps", 8.0)),
            ego_lookahead_m=float(data.get("ego_lookahead_m", 10.0)),
            traffic_vehicles=int(data.get("traffic_vehicles", 0)),
            traffic_walkers=int(data.get("traffic_walkers", 0)),
            traffic_spawn_start_index=int(data.get("traffic_spawn_start_index", -1)),
            traffic_spawn_indices=[int(item) for item in traffic_spawn_indices],
            traffic_route_commands=[str(item) for item in traffic_route_commands],
            traffic_spawn_delay_sec=float(data.get("traffic_spawn_delay_sec", 0.0)),
            traffic_speed_difference=float(data.get("traffic_speed_difference", 25.0)),
            obstacle_vehicles=int(data.get("obstacle_vehicles", 0)),
            obstacle_spawn_delay_sec=float(data.get("obstacle_spawn_delay_sec", 0.0)),
            obstacle_blueprint=str(data.get("obstacle_blueprint", "vehicle.dodge.charger_police_2020")),
            obstacle_anchor_x=float(data.get("obstacle_anchor_x", 0.0)),
            obstacle_anchor_y=float(data.get("obstacle_anchor_y", 0.0)),
            obstacle_anchor_z=float(data.get("obstacle_anchor_z", 0.6)),
            obstacle_anchor_yaw_deg=float(data.get("obstacle_anchor_yaw_deg", 0.0)),
            obstacle_forward_offsets_m=[float(item) for item in obstacle_forward_offsets],
            obstacle_lateral_offsets_m=[float(item) for item in obstacle_lateral_offsets],
            obstacle_yaw_offsets_deg=[float(item) for item in obstacle_yaw_offsets],
            walker_spawn_start_index=int(data.get("walker_spawn_start_index", -1)),
            walker_spawn_indices=[int(item) for item in walker_spawn_indices],
            walker_spawn_delay_sec=float(data.get("walker_spawn_delay_sec", 0.0)),
            walker_crossing_distance_m=float(data.get("walker_crossing_distance_m", 18.0)),
            walker_crossing_offsets_m=[float(item) for item in walker_crossing_offsets],
            walker_speed_mps=float(data.get("walker_speed_mps", 1.4)),
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
            vision_model_path=str(data.get("vision_model_path", "")),
            vision_model_device=str(data.get("vision_model_device", "cpu")),
            vision_model_control_mode=str(data.get("vision_model_control_mode", "trajectory")),
            vision_navigation_command=str(data.get("vision_navigation_command", "lane_follow")),
            vision_safety_gate_enabled=bool(data.get("vision_safety_gate_enabled", True)),
            vision_attack_pattern_gate=bool(data.get("vision_attack_pattern_gate", False)),
            vision_attack_pattern_threshold=float(data.get("vision_attack_pattern_threshold", 0.35)),
            vision_low_visibility_gate=bool(data.get("vision_low_visibility_gate", False)),
            vision_low_visibility_threshold=float(data.get("vision_low_visibility_threshold", 0.12)),
            vision_first_junction_command=str(data.get("vision_first_junction_command", "")),
            vision_junction_command_sequence=[str(item) for item in junction_command_sequence],
            vision_junction_command_hold_sec=float(data.get("vision_junction_command_hold_sec", 4.0)),
            vision_junction_command_hold_until_exit=bool(
                data.get("vision_junction_command_hold_until_exit", False)
            ),
            uav_enabled=bool(data.get("uav_enabled", True)),
            uav_control_enabled=bool(data.get("uav_control_enabled", True)),
            uav_name=str(data.get("uav_name", "SimpleFlight")),
            uav_altitude=float(data.get("uav_altitude", 18.0)),
            uav_back_distance=float(data.get("uav_back_distance", 8.0)),
            uav_auto_patrol_enabled=bool(data.get("uav_auto_patrol_enabled", False)),
            uav_patrol_interval_sec=float(data.get("uav_patrol_interval_sec", 4.0)),
            uav_bev_fusion_enabled=bool(data.get("uav_bev_fusion_enabled", uav_fusion_mode != "none")),
            uav_bev_camera_name=str(data.get("uav_bev_camera_name", "front_center")),
            uav_bev_refresh_hz=float(data.get("uav_bev_refresh_hz", 2.0)),
            uav_bev_min_confidence=float(data.get("uav_bev_min_confidence", 0.20)),
            uav_bev_steer_gain=float(data.get("uav_bev_steer_gain", 0.08)),
            uav_bev_max_steer_correction=float(data.get("uav_bev_max_steer_correction", 0.08)),
            uav_fusion_mode=uav_fusion_mode,
            uav_fusion_planner_path=str(data.get("uav_fusion_planner_path", "")),
            uav_fusion_planner_gain=float(data.get("uav_fusion_planner_gain", 1.0)),
            uav_fusion_max_steer_correction=float(
                data.get(
                    "uav_fusion_max_steer_correction",
                    data.get("uav_bev_max_steer_correction", 0.08),
                )
            ),
            uav_fusion_min_confidence=float(
                data.get(
                    "uav_fusion_min_confidence",
                    data.get("uav_bev_min_confidence", 0.20),
                )
            ),
            experiment_group=str(data.get("experiment_group", "")),
            scenario_stage=str(data.get("scenario_stage", "")),
            scenario_complexity=[str(item) for item in complexity],
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
            "ego_spawn_forward_m": self.ego_spawn_forward_m,
            "ego_sensor_mode": self.ego_sensor_mode,
            "ego_control_mode": self.ego_control_mode,
            "ego_drive_hz": self.ego_drive_hz,
            "ego_target_speed_mps": self.ego_target_speed_mps,
            "ego_lookahead_m": self.ego_lookahead_m,
            "traffic_vehicles": self.traffic_vehicles,
            "traffic_walkers": self.traffic_walkers,
            "traffic_spawn_start_index": self.traffic_spawn_start_index,
            "traffic_spawn_indices": list(self.traffic_spawn_indices),
            "traffic_route_commands": list(self.traffic_route_commands),
            "traffic_spawn_delay_sec": self.traffic_spawn_delay_sec,
            "traffic_speed_difference": self.traffic_speed_difference,
            "obstacle_vehicles": self.obstacle_vehicles,
            "obstacle_spawn_delay_sec": self.obstacle_spawn_delay_sec,
            "obstacle_blueprint": self.obstacle_blueprint,
            "obstacle_anchor_x": self.obstacle_anchor_x,
            "obstacle_anchor_y": self.obstacle_anchor_y,
            "obstacle_anchor_z": self.obstacle_anchor_z,
            "obstacle_anchor_yaw_deg": self.obstacle_anchor_yaw_deg,
            "obstacle_forward_offsets_m": list(self.obstacle_forward_offsets_m),
            "obstacle_lateral_offsets_m": list(self.obstacle_lateral_offsets_m),
            "obstacle_yaw_offsets_deg": list(self.obstacle_yaw_offsets_deg),
            "walker_spawn_start_index": self.walker_spawn_start_index,
            "walker_spawn_indices": list(self.walker_spawn_indices),
            "walker_spawn_delay_sec": self.walker_spawn_delay_sec,
            "walker_crossing_distance_m": self.walker_crossing_distance_m,
            "walker_crossing_offsets_m": list(self.walker_crossing_offsets_m),
            "walker_speed_mps": self.walker_speed_mps,
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
            "vision_model_path": self.vision_model_path,
            "vision_model_device": self.vision_model_device,
            "vision_model_control_mode": self.vision_model_control_mode,
            "vision_navigation_command": self.vision_navigation_command,
            "vision_safety_gate_enabled": self.vision_safety_gate_enabled,
            "vision_attack_pattern_gate": self.vision_attack_pattern_gate,
            "vision_attack_pattern_threshold": self.vision_attack_pattern_threshold,
            "vision_low_visibility_gate": self.vision_low_visibility_gate,
            "vision_low_visibility_threshold": self.vision_low_visibility_threshold,
            "vision_first_junction_command": self.vision_first_junction_command,
            "vision_junction_command_sequence": list(self.vision_junction_command_sequence),
            "vision_junction_command_hold_sec": self.vision_junction_command_hold_sec,
            "vision_junction_command_hold_until_exit": self.vision_junction_command_hold_until_exit,
            "uav_enabled": self.uav_enabled,
            "uav_control_enabled": self.uav_control_enabled,
            "uav_name": self.uav_name,
            "uav_altitude": self.uav_altitude,
            "uav_back_distance": self.uav_back_distance,
            "uav_auto_patrol_enabled": self.uav_auto_patrol_enabled,
            "uav_patrol_interval_sec": self.uav_patrol_interval_sec,
            "uav_bev_fusion_enabled": self.uav_bev_fusion_enabled,
            "uav_bev_camera_name": self.uav_bev_camera_name,
            "uav_bev_refresh_hz": self.uav_bev_refresh_hz,
            "uav_bev_min_confidence": self.uav_bev_min_confidence,
            "uav_bev_steer_gain": self.uav_bev_steer_gain,
            "uav_bev_max_steer_correction": self.uav_bev_max_steer_correction,
            "uav_fusion_mode": self.uav_fusion_mode,
            "uav_fusion_planner_path": self.uav_fusion_planner_path,
            "uav_fusion_planner_gain": self.uav_fusion_planner_gain,
            "uav_fusion_max_steer_correction": self.uav_fusion_max_steer_correction,
            "uav_fusion_min_confidence": self.uav_fusion_min_confidence,
            "experiment_group": self.experiment_group,
            "scenario_stage": self.scenario_stage,
            "scenario_complexity": list(self.scenario_complexity),
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
