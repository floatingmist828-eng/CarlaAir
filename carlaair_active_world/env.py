from __future__ import annotations

import threading
import time
import math
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
    collect_walker_states,
)
from .ego_driver import EgoDriveConfig, RouteFollowingDriver
from .adversarial import apply_weather_preset
from .geometry import CandidateViewpoint, Pose, ScenarioResult, Vector3
from .labels import build_labels
from .scenario import ScenarioConfig
from .sensors import UAVSensorRig
from .traffic import spawn_traffic_vehicles, spawn_traffic_walkers
from .vision_driver import VisionEgoDriver
from .vision_models import TcpLiteVisionPolicy
from .vision_models.uav_bev import CachedUAVBEVProvider


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
        self.traffic_actors = []
        self.walker_actors = []
        self.walker_controllers = []
        self._walker_targets = []
        self.collision_sensor = None
        self.collision_events = []
        self.ego_driver = None
        self.uav_sensors = None
        self.uav_bev_provider = CachedUAVBEVProvider(
            lambda: self.uav_sensors,
            refresh_hz=float(getattr(self.scenario, "uav_bev_refresh_hz", 2.0)),
        )
        self._air_rpc_lock = threading.RLock()
        self._ego_driver_thread = None
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.start_time = 0.0
        self.last_action = 0
        self._closed = False
        self._traffic_spawned = False
        self._walkers_spawned = False

    def connect(self) -> None:
        self.client, self.world = connect_carla(self.carla_host, self.carla_port)
        settings = self.world.get_settings()
        apply_weather_preset(self.world, getattr(self.scenario, "weather_preset", "none"))
        if self.destroy_old_vehicles:
            cleanup_actors_by_role(self.world, {"ego", "task_ego", "task_traffic", "task_walker"})
            cleanup_old_vehicles(self.world)
        if self.scenario.uav_enabled:
            self.air_client = connect_airsim(self.airsim_port, vehicle_name=self.scenario.uav_name)
            if bool(getattr(self.scenario, "uav_control_enabled", True)):
                with self._air_rpc_lock:
                    self.ox, self.oy, self.oz = calibrate_offset(
                        self.world,
                        self.air_client,
                        preferred_name=self.scenario.uav_name,
                    )
        else:
            self._park_disabled_uav()

    def _park_disabled_uav(self) -> None:
        try:
            air_client = connect_airsim(self.airsim_port, vehicle_name=self.scenario.uav_name)
        except Exception:
            return

        park_pose = Pose(position=Vector3(-1000.0, -1000.0, 120.0), roll=0.0, pitch=0.0, yaw=0.0)
        names = [self.scenario.uav_name, None] if self.scenario.uav_name else [None]
        for name in names:
            try:
                move_uav_to(
                    air_client,
                    park_pose,
                    self.ox,
                    self.oy,
                    self.oz,
                    vehicle_name=name,
                )
                set_uav_hover(air_client, vehicle_name=name)
                return
            except Exception:
                continue

    def reset(self) -> Dict[str, Any]:
        if self.client is None or self.world is None:
            self.connect()
        self._closed = False
        settings = self.world.get_settings()
        self.ego_vehicle = spawn_ego_vehicle(
            self.world,
            blueprint_id=self.scenario.ego_blueprint,
            spawn_index=self.scenario.ego_spawn_index,
            forward_m=float(getattr(self.scenario, "ego_spawn_forward_m", 0.0)),
        )
        self.collision_events = []
        self._attach_collision_sensor()
        self._start_ego_control()
        self.traffic_actors = []
        self.walker_actors = []
        self.walker_controllers = []
        self._walker_targets = []
        self._traffic_spawned = max(0, int(self.scenario.traffic_vehicles)) <= 0
        self._walkers_spawned = max(0, int(self.scenario.traffic_walkers)) <= 0
        self.start_time = time.time()
        if float(getattr(self.scenario, "traffic_spawn_delay_sec", 0.0)) <= 0.0:
            self._spawn_configured_traffic()
        if float(getattr(self.scenario, "walker_spawn_delay_sec", 0.0)) <= 0.0:
            self._spawn_configured_walkers()
        traffic_speed_difference = float(getattr(self.scenario, "traffic_speed_difference", 25.0))
        set_traffic_manager_speed(self.client, traffic_speed_difference)
        if settings.synchronous_mode:
            self.world.tick()
        observation = self.observe()
        if self.scenario.uav_enabled and self.air_client is not None:
            if bool(getattr(self.scenario, "uav_control_enabled", True)):
                self._place_initial_uav(observation)
            if str(getattr(self.scenario, "uav_fusion_mode", "none")).lower() != "none":
                self.uav_sensors = UAVSensorRig(
                    self.air_client,
                    camera_name=str(getattr(self.scenario, "uav_bev_camera_name", "front_center")),
                    record_depth=False,
                    rpc_lock=self._air_rpc_lock,
                )
        return observation

    def _spawn_configured_traffic(self) -> None:
        if self._traffic_spawned or self.client is None or self.world is None:
            return
        traffic_start_index = int(getattr(self.scenario, "traffic_spawn_start_index", -1))
        if traffic_start_index < 0:
            traffic_start_index = max(1, int(self.scenario.ego_spawn_index) + 1)
        traffic_speed_difference = float(getattr(self.scenario, "traffic_speed_difference", 25.0))
        spawned = spawn_traffic_vehicles(
            self.client,
            self.world,
            count=max(0, int(self.scenario.traffic_vehicles)),
            start_index=traffic_start_index,
            spawn_indices=list(getattr(self.scenario, "traffic_spawn_indices", []) or []),
            speed_difference=traffic_speed_difference,
        )
        self.traffic_actors.extend(item.actor for item in spawned)
        self._traffic_spawned = True

    def _spawn_configured_walkers(self) -> None:
        if self._walkers_spawned or self.client is None or self.world is None:
            return
        walker_start_index = int(getattr(self.scenario, "walker_spawn_start_index", -1))
        if walker_start_index < 0:
            walker_start_index = max(1, int(self.scenario.ego_spawn_index) + 5)
        spawned_walkers = spawn_traffic_walkers(
            self.client,
            self.world,
            count=max(0, int(self.scenario.traffic_walkers)),
            start_index=walker_start_index,
            spawn_indices=list(getattr(self.scenario, "walker_spawn_indices", []) or []),
            crossing_distance_m=float(getattr(self.scenario, "walker_crossing_distance_m", 18.0)),
            crossing_offsets_m=list(getattr(self.scenario, "walker_crossing_offsets_m", []) or []),
            use_ai_controller=False,
            speed_mps=float(getattr(self.scenario, "walker_speed_mps", 1.4)),
        )
        self.walker_actors.extend(item.actor for item in spawned_walkers)
        self.walker_controllers.extend(item.controller for item in spawned_walkers if item.controller is not None)
        for item in spawned_walkers:
            target = getattr(item, "target", None)
            speed_mps = float(getattr(item, "speed_mps", 0.0) or 0.0)
            if target is not None and speed_mps > 0.0:
                self._walker_targets.append((item.actor, target, speed_mps))
        self._walkers_spawned = True

    def _maybe_spawn_delayed_actors(self) -> None:
        elapsed = float(time.time() - self.start_time) if self.start_time else 0.0
        if elapsed >= float(getattr(self.scenario, "traffic_spawn_delay_sec", 0.0)):
            self._spawn_configured_traffic()
        if elapsed >= float(getattr(self.scenario, "walker_spawn_delay_sec", 0.0)):
            self._spawn_configured_walkers()

    def _drive_scripted_walkers(self) -> None:
        remaining = []
        for actor, target, speed_mps in list(self._walker_targets):
            try:
                loc = actor.get_location()
                dx = float(target.x - loc.x)
                dy = float(target.y - loc.y)
                distance = math.hypot(dx, dy)
                if distance <= 0.6:
                    control = carla.WalkerControl()
                    control.speed = 0.0
                    actor.apply_control(control)
                    continue
                control = carla.WalkerControl()
                control.direction = carla.Vector3D(dx / distance, dy / distance, 0.0)
                control.speed = float(speed_mps)
                actor.apply_control(control)
                remaining.append((actor, target, speed_mps))
            except Exception:
                continue
        self._walker_targets = remaining

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
                        attack_pattern_threshold=self.scenario.vision_attack_pattern_threshold,
                        low_visibility_gate=self.scenario.vision_low_visibility_gate,
                        low_visibility_threshold=self.scenario.vision_low_visibility_threshold,
                        target_speed_mps=float(getattr(self.scenario, "ego_target_speed_mps", 4.0)),
                        uav_bev_fusion_enabled=bool(getattr(self.scenario, "uav_bev_fusion_enabled", False)),
                        uav_bev_min_confidence=float(getattr(self.scenario, "uav_bev_min_confidence", 0.20)),
                        uav_bev_steer_gain=float(getattr(self.scenario, "uav_bev_steer_gain", 0.08)),
                        uav_bev_max_steer_correction=float(
                            getattr(self.scenario, "uav_bev_max_steer_correction", 0.08)
                        ),
                        uav_fusion_mode=str(getattr(self.scenario, "uav_fusion_mode", "none")),
                        uav_fusion_planner_path=str(getattr(self.scenario, "uav_fusion_planner_path", "")),
                        uav_fusion_planner_gain=float(getattr(self.scenario, "uav_fusion_planner_gain", 1.0)),
                        uav_fusion_max_steer_correction=float(
                            getattr(self.scenario, "uav_fusion_max_steer_correction", 0.08)
                        ),
                        uav_fusion_min_confidence=float(getattr(self.scenario, "uav_fusion_min_confidence", 0.20)),
                    )
                self.ego_driver = VisionEgoDriver(
                    self.world,
                    self.ego_vehicle,
                    target_speed_mps=float(getattr(self.scenario, "ego_target_speed_mps", 4.0)),
                    policy=policy,
                    use_semantic=mode == "vision_simple",
                    use_depth=mode != "vision_tcp_lite",
                    navigation_command=self.scenario.vision_navigation_command,
                    first_junction_command=str(getattr(self.scenario, "vision_first_junction_command", "")),
                    junction_command_sequence=list(
                        getattr(self.scenario, "vision_junction_command_sequence", []) or []
                    ),
                    junction_command_hold_sec=float(getattr(self.scenario, "vision_junction_command_hold_sec", 4.0)),
                    vision_attack=str(getattr(self.scenario, "vision_attack", "none")),
                    vision_attack_intensity=float(getattr(self.scenario, "vision_attack_intensity", 1.0)),
                    vision_detector_model_path=str(getattr(self.scenario, "vision_detector_model_path", "")),
                    vision_detector_confidence=float(getattr(self.scenario, "vision_detector_confidence", 0.35)),
                    uav_bev_provider=(
                        self.uav_bev_provider.snapshot
                        if str(getattr(self.scenario, "uav_fusion_mode", "none")).lower() != "none"
                        else None
                    ),
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

    def _attach_collision_sensor(self) -> None:
        if self.world is None or self.ego_vehicle is None:
            return
        try:
            bp = self.world.get_blueprint_library().find("sensor.other.collision")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego_vehicle)
        except Exception:
            self.collision_sensor = None
            return

        def _on_collision(event) -> None:
            other = getattr(event, "other_actor", None)
            impulse = getattr(event, "normal_impulse", None)
            self.collision_events.append(
                {
                    "time": float(time.time() - self.start_time) if self.start_time else 0.0,
                    "other_actor_id": int(getattr(other, "id", -1)),
                    "other_type_id": str(getattr(other, "type_id", "")),
                    "normal_impulse": {
                        "x": float(getattr(impulse, "x", 0.0)),
                        "y": float(getattr(impulse, "y", 0.0)),
                        "z": float(getattr(impulse, "z", 0.0)),
                    },
                }
            )

        try:
            sensor.listen(_on_collision)
            self.collision_sensor = sensor
        except Exception:
            try:
                sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None

    def _apply_collision_labels(self, label: Dict[str, Any]) -> None:
        count = len(self.collision_events)
        label["collision"] = count > 0
        label["collision_count"] = count
        if count:
            label["collision_events"] = list(self.collision_events)

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
        walker_states = collect_walker_states(self.world)
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
        if (
            self.scenario.uav_enabled
            and self.air_client is not None
            and bool(getattr(self.scenario, "uav_control_enabled", True))
        ):
            try:
                with self._air_rpc_lock:
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
        elif self.scenario.uav_enabled and self.air_client is not None:
            drone_state = {
                "mode": "airsim_camera_only",
                "camera_name": str(getattr(self.scenario, "uav_bev_camera_name", "front_center")),
            }

        observation = {
            "time": float(time.time() - self.start_time),
            "scenario": self.scenario.to_dict(),
            "ego": ego_state.to_dict(),
            "vehicles": [v.to_dict() for v in vehicle_states],
            "walkers": [w.to_dict() for w in walker_states],
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
        self._maybe_spawn_delayed_actors()
        self._drive_scripted_walkers()
        if (
            self.scenario.uav_enabled
            and self.air_client is not None
            and bool(getattr(self.scenario, "uav_control_enabled", True))
        ):
            candidates = self.scenario.candidate_offsets
            idx = max(0, min(int(action_index), len(candidates) - 1))
            candidate = candidates[idx]
            ego_transform = self.ego_vehicle.get_transform()
            pose = local_candidate_to_world(ego_transform, candidate)
            with self._air_rpc_lock:
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
        self._apply_collision_labels(label)
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
        if self.air_client is not None and bool(getattr(self.scenario, "uav_control_enabled", True)):
            with self._air_rpc_lock:
                set_uav_hover(self.air_client, vehicle_name=self.scenario.uav_name)
        self.uav_sensors = None
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except Exception:
                pass
            try:
                self.collision_sensor.destroy()
            except Exception:
                pass
        self.collision_sensor = None
        for controller in self.walker_controllers:
            try:
                controller.stop()
            except Exception:
                pass
            try:
                controller.destroy()
            except Exception:
                pass
        self.walker_controllers = []
        self._walker_targets = []
        for actor in self.walker_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self.walker_actors = []
        for actor in self.traffic_actors:
            try:
                actor.set_autopilot(False)
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self.traffic_actors = []
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
