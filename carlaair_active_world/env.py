from __future__ import annotations

import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import carla

from .core import (
    AIRSIM_PORT,
    CARLA_HOST,
    CARLA_PORT,
    cleanup_old_vehicles,
    calibrate_offset,
    connect_airsim,
    connect_carla,
    cleanup_actors_by_role,
    configure_autopilot,
    get_actor_state,
    local_candidate_to_world,
    move_uav_to,
    set_traffic_manager_speed,
    set_uav_hover,
    spawn_ego_vehicle,
    collect_vehicle_states,
)
from .ego_driver import EgoDriveConfig, RouteFollowingDriver
from .adversarial import apply_weather_preset
from .geometry import CandidateViewpoint, Pose, ScenarioResult, Vector3
from .labels import build_labels
from .scenario import ScenarioConfig
from .vision_driver import VisionEgoDriver
from .vision_models import TcpLiteVisionPolicy


class ActiveAirGroundEnv:
    def __init__(
        self,
        scenario: ScenarioConfig,
        carla_host: str = CARLA_HOST,
        carla_port: int = CARLA_PORT,
        airsim_port: int = AIRSIM_PORT,
        destroy_old_vehicles: bool = True,
    ) -> None:
        self.scenario = scenario
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.airsim_port = airsim_port
        self.destroy_old_vehicles = destroy_old_vehicles
        self.client = None
        self.world = None
        self.air_client = None
        self.ego_vehicle = None
        self.ego_driver = None
        self._ego_driver_thread = None
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.start_time = 0.0
        self.last_action = 0
        self._closed = False

    def connect(self) -> None:
        self.client, self.world = connect_carla(self.carla_host, self.carla_port)
        settings = self.world.get_settings()
        apply_weather_preset(self.world, getattr(self.scenario, "weather_preset", "none"))
        if self.destroy_old_vehicles:
            cleanup_actors_by_role(self.world, {"ego", "task_ego", "task_traffic"})
            cleanup_old_vehicles(self.world)
        if self.scenario.uav_enabled:
            self.air_client = connect_airsim(self.airsim_port, vehicle_name=self.scenario.uav_name)
            self.ox, self.oy, self.oz = calibrate_offset(
                self.world,
                self.air_client,
                preferred_name=self.scenario.uav_name,
            )

    def reset(self) -> Dict[str, Any]:
        if self.client is None or self.world is None:
            self.connect()
        self._closed = False
        settings = self.world.get_settings()
        self.ego_vehicle = spawn_ego_vehicle(
            self.world,
            blueprint_id=self.scenario.ego_blueprint,
            spawn_index=self.scenario.ego_spawn_index,
        )
        self._start_ego_control()
        set_traffic_manager_speed(self.client, 40.0)
        if settings.synchronous_mode:
            self.world.tick()
        self.start_time = time.time()
        observation = self.observe()
        if self.scenario.uav_enabled and self.air_client is not None:
            self._place_initial_uav(observation)
        return observation

    def _start_ego_control(self) -> None:
        mode = str(getattr(self.scenario, "ego_control_mode", "autopilot")).lower()
        if mode in {"route_follow", "route", "behavior", "vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
            try:
                self.ego_vehicle.set_autopilot(False)
            except Exception:
                pass
            if mode in {"vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
                policy = None
                if mode == "vision_tcp_lite":
                    policy = TcpLiteVisionPolicy(
                        model_path=self.scenario.vision_model_path,
                        device=self.scenario.vision_model_device,
                        control_mode=self.scenario.vision_model_control_mode,
                        navigation_command=self.scenario.vision_navigation_command,
                        safety_gate_enabled=self.scenario.vision_safety_gate_enabled,
                        attack_pattern_gate=self.scenario.vision_attack_pattern_gate,
                        target_speed_mps=float(getattr(self.scenario, "ego_target_speed_mps", 4.0)),
                    )
                self.ego_driver = VisionEgoDriver(
                    self.world,
                    self.ego_vehicle,
                    target_speed_mps=float(getattr(self.scenario, "ego_target_speed_mps", 4.0)),
                    policy=policy,
                    use_semantic=mode == "vision_simple",
                    use_depth=mode != "vision_tcp_lite",
                    navigation_command=self.scenario.vision_navigation_command,
                    vision_attack=str(getattr(self.scenario, "vision_attack", "none")),
                    vision_attack_intensity=float(getattr(self.scenario, "vision_attack_intensity", 1.0)),
                    vision_detector_model_path=str(getattr(self.scenario, "vision_detector_model_path", "")),
                    vision_detector_confidence=float(getattr(self.scenario, "vision_detector_confidence", 0.35)),
                )
            else:
                self.ego_driver = RouteFollowingDriver(
                    EgoDriveConfig(
                        target_speed_mps=float(getattr(self.scenario, "ego_target_speed_mps", 8.0)),
                        lookahead_m=float(getattr(self.scenario, "ego_lookahead_m", 10.0)),
                    )
                )

            def _loop():
                interval = 1.0 / max(0.1, float(getattr(self.scenario, "ego_drive_hz", 8.0)))
                while not self._closed and self.ego_vehicle is not None and self.world is not None:
                    try:
                        control = self.ego_driver.predict(self.ego_vehicle, self.world)
                        self.ego_vehicle.apply_control(control)
                    except Exception:
                        try:
                            brake = carla.VehicleControl()
                            brake.brake = 1.0
                            self.ego_vehicle.apply_control(brake)
                        except Exception:
                            pass
                    time.sleep(interval)

            self._ego_driver_thread = threading.Thread(target=_loop, daemon=True)
            self._ego_driver_thread.start()
            return

        configure_autopilot(self.client, self.world, self.ego_vehicle)

    def _place_initial_uav(self, observation: Dict[str, Any]) -> None:
        candidate = self.scenario.candidate_offsets[0]
        ego_transform = self.ego_vehicle.get_transform()
        pose = local_candidate_to_world(ego_transform, candidate)
        move_uav_to(
            self.air_client,
            pose=pose,
            ox=self.ox,
            oy=self.oy,
            oz=self.oz,
            vehicle_name=self.scenario.uav_name,
        )

    def build_candidates(self) -> List[Dict[str, Any]]:
        ego_transform = self.ego_vehicle.get_transform()
        candidates = []
        for idx, cand in enumerate(self.scenario.candidate_offsets):
            pose = local_candidate_to_world(ego_transform, cand)
            candidates.append(
                {
                    "index": idx,
                    "name": cand.name,
                    "weight": cand.weight,
                    "local_offset": cand.local_offset.to_dict(),
                    "pose": pose.to_dict(),
                }
            )
        return candidates

    def observe(self) -> Dict[str, Any]:
        ego_state = get_actor_state(self.ego_vehicle)
        vehicle_states = collect_vehicle_states(self.world, include_ego=False)
        waypoint = None
        try:
            map_waypoint = self.world.get_map().get_waypoint(
                self.ego_vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            waypoint = {
                "road_id": int(map_waypoint.road_id),
                "lane_id": int(map_waypoint.lane_id),
                "is_junction": bool(map_waypoint.is_junction),
            }
        except Exception:
            waypoint = None

        drone_state = None
        if self.scenario.uav_enabled and self.air_client is not None:
            try:
                if self.scenario.uav_name:
                    state = self.air_client.getMultirotorState(vehicle_name=self.scenario.uav_name)
                else:
                    state = self.air_client.getMultirotorState()
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                drone_state = {
                    "position": {"x": pos.x_val, "y": pos.y_val, "z": pos.z_val},
                    "velocity": {"x": vel.x_val, "y": vel.y_val, "z": vel.z_val},
                }
            except Exception:
                drone_state = None

        observation = {
            "time": float(time.time() - self.start_time),
            "scenario": self.scenario.to_dict(),
            "ego": ego_state.to_dict(),
            "vehicles": [v.to_dict() for v in vehicle_states],
            "drone": drone_state,
            "waypoint": waypoint,
            "candidates": self.build_candidates() if self.ego_vehicle is not None else [],
            "last_action": self.last_action,
            "ego_control": dict(getattr(self.ego_driver, "last_diagnostics", {}) or {}),
        }
        return observation

    def step(self, action_index: int) -> ScenarioResult:
        self.last_action = int(action_index)
        if self.ego_vehicle is None:
            raise RuntimeError("Call reset() before step().")
        if self.scenario.uav_enabled and self.air_client is not None:
            candidates = self.scenario.candidate_offsets
            idx = max(0, min(int(action_index), len(candidates) - 1))
            candidate = candidates[idx]
            ego_transform = self.ego_vehicle.get_transform()
            pose = local_candidate_to_world(ego_transform, candidate)
            move_uav_to(
                self.air_client,
                pose=pose,
                ox=self.ox,
                oy=self.oy,
                oz=self.oz,
                vehicle_name=self.scenario.uav_name,
            )
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            time.sleep(self.scenario.step_sec)
        observation = self.observe()
        label = build_labels(
            self.world,
            self.ego_vehicle,
            horizon_sec=self.scenario.future_horizon_sec,
            step_sec=self.scenario.step_sec,
        )
        done = observation["time"] >= self.scenario.duration_sec
        info = {
            "candidate_count": len(self.scenario.candidate_offsets),
        }
        return ScenarioResult(observation=observation, label=label, info=info, done=done)

    def close(self) -> None:
        if self._closed:
            return
        if self._ego_driver_thread is not None:
            self._closed = True
            self._ego_driver_thread.join(timeout=2.0)
            self._ego_driver_thread = None
        if self.ego_driver is not None and hasattr(self.ego_driver, "destroy"):
            try:
                self.ego_driver.destroy()
            except Exception:
                pass
        self.ego_driver = None
        if self.air_client is not None:
            set_uav_hover(self.air_client, vehicle_name=self.scenario.uav_name)
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.set_autopilot(False)
            except Exception:
                pass
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
        self._closed = True
