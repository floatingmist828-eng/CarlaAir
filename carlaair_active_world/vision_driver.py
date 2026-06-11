from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
        use_depth: bool = True,
        vision_attack: str = "none",
        vision_attack_intensity: float = 1.0,
        vision_detector_model_path: str = "",
        vision_detector_confidence: float = 0.35,
        detector: Optional[Any] = None,
        navigation_command: str = "lane_follow",
        uav_bev_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        first_junction_command: str = "",
        junction_command_sequence: Optional[list[str]] = None,
        junction_command_hold_sec: float = 4.0,
        junction_command_hold_until_exit: bool = False,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.use_semantic = bool(use_semantic)
        self.use_depth = bool(use_depth)
        self.navigation_command = navigation_command
        self.first_junction_command = self._normalize_turn_command(first_junction_command)
        sequence = junction_command_sequence if junction_command_sequence is not None else []
        self.junction_command_sequence = [
            command for command in (self._normalize_turn_command(item) for item in sequence) if command
        ]
        if not self.junction_command_sequence and self.first_junction_command:
            self.junction_command_sequence = [self.first_junction_command]
        self.junction_command_hold_sec = max(0.0, float(junction_command_hold_sec))
        self.junction_command_hold_until_exit = bool(junction_command_hold_until_exit)
        self._clock = clock or time.monotonic
        self._junction_command_until: Optional[float] = None
        self._junction_active_command = ""
        self._junction_command_index = 0
        self._was_in_junction = False
        self._junction_cached_turn_target: Optional[tuple[str, carla.Location]] = None
        self._obstacle_corridor_active = False
        self.sensor_rig = self.sensor_rig_class(
            world,
            ego_vehicle,
            "ego_vision",
            vision_attack=vision_attack,
            vision_attack_intensity=vision_attack_intensity,
            disable_depth=not bool(use_depth),
            disable_semantic=not bool(use_semantic),
        )
        self.sensor_rig.spawn()
        self.policy = policy or SimpleLaneVisionPolicy(target_speed_mps=target_speed_mps)
        self.detector = detector
        self.uav_bev_provider = uav_bev_provider
        self._detector_diagnostics: Dict[str, Any] = {}
        if self.detector is None and vision_detector_model_path:
            self.detector = self._load_detector(vision_detector_model_path, vision_detector_confidence)
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_observation: Dict[str, Any] = {}

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

    @staticmethod
    def _normalize_turn_command(command: str) -> str:
        value = str(command or "").strip().lower()
        return value if value in {"left", "right", "straight"} else ""

    @staticmethod
    def _angle_delta_deg(target: float, source: float) -> float:
        return (float(target) - float(source) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    @staticmethod
    def _local_target(vehicle_transform: carla.Transform, target: carla.Location) -> tuple[float, float]:
        ego_loc = vehicle_transform.location
        ego_yaw = np.deg2rad(float(vehicle_transform.rotation.yaw))
        target_dx = float(target.x - ego_loc.x)
        target_dy = float(target.y - ego_loc.y)
        local_x = target_dx * np.cos(-ego_yaw) - target_dy * np.sin(-ego_yaw)
        local_y = target_dx * np.sin(-ego_yaw) + target_dy * np.cos(-ego_yaw)
        return float(local_x), float(local_y)

    def _navigation_command_for_lane(self, lane_reference: Dict[str, Any]) -> str:
        base_command = str(self.navigation_command or "lane_follow")
        if not self.junction_command_sequence:
            return base_command

        now = float(self._clock())
        in_junction = bool(lane_reference.get("in_junction", False))
        if not in_junction:
            if self._was_in_junction and self._junction_active_command:
                self._junction_command_index += 1
            self._was_in_junction = False
            self._junction_command_until = None
            self._junction_active_command = ""
            return base_command

        if not self._was_in_junction:
            self._was_in_junction = True
            if self._junction_command_index < len(self.junction_command_sequence):
                self._junction_active_command = self.junction_command_sequence[self._junction_command_index]
                self._junction_command_until = now + self.junction_command_hold_sec

        if not self._junction_active_command:
            return base_command

        if self.junction_command_hold_until_exit and self._junction_active_command == "straight":
            return self._junction_active_command

        if self._junction_command_until is None or now <= self._junction_command_until:
            return self._junction_active_command

        return base_command

    @staticmethod
    def _lane_reference(vehicle: carla.Actor, world: Optional[carla.World]) -> Dict[str, Any]:
        if world is None:
            return {}
        try:
            waypoint = world.get_map().get_waypoint(
                vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            loc = vehicle.get_location()
            center = waypoint.transform.location
            yaw = np.deg2rad(float(waypoint.transform.rotation.yaw))
            dx = float(loc.x - center.x)
            dy = float(loc.y - center.y)
            lateral_offset = -np.sin(yaw) * dx + np.cos(yaw) * dy
            result = {
                "lane_center_offset_m": float(lateral_offset),
                "lane_width_m": float(getattr(waypoint, "lane_width", 0.0) or 0.0),
                "lane_road_id": int(getattr(waypoint, "road_id", 0) or 0),
                "lane_id": int(getattr(waypoint, "lane_id", 0) or 0),
                "in_junction": bool(getattr(waypoint, "is_junction", False)),
                "_route_waypoint": waypoint,
            }
            try:
                ego_transform = vehicle.get_transform()
                best = None
                best_cost = None
                for candidate in waypoint.next(10.0):
                    target = candidate.transform.location
                    local_x, local_y = VisionEgoDriver._local_target(ego_transform, target)
                    if local_x <= 0.25:
                        continue
                    heading_delta = abs(
                        ((float(candidate.transform.rotation.yaw - ego_transform.rotation.yaw) + 180.0) % 360.0)
                        - 180.0
                    )
                    cost = heading_delta + 0.10 * abs(local_y)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best = (local_x, local_y)
                if best is not None:
                    result["route_target_local_x"] = float(best[0])
                    result["route_target_local_y"] = float(best[1])
                    result["route_target_source"] = "waypoint_next"
            except Exception:
                pass
            return result
        except Exception:
            return {}

    @staticmethod
    def _interaction_hazard(vehicle: carla.Actor, world: Optional[carla.World]) -> Dict[str, Any]:
        if world is None:
            return {"active": False, "reason": "no_world"}
        try:
            vehicle_transform = vehicle.get_transform()
            ego_id = int(getattr(vehicle, "id", -1))
            actors = world.get_actors()
            candidates = list(actors.filter("vehicle.*")) + list(actors.filter("walker.pedestrian.*"))
        except Exception:
            return {"active": False, "reason": "actor_query_failed"}

        best: Optional[tuple[int, float, Dict[str, Any]]] = None
        for actor in candidates:
            actor_id = int(getattr(actor, "id", -1))
            if actor_id == ego_id:
                continue
            role_name = str(getattr(actor, "attributes", {}).get("role_name", ""))
            if role_name in {"ego", "task_ego"}:
                continue
            type_id = str(getattr(actor, "type_id", ""))
            try:
                local_x, local_y = VisionEgoDriver._local_target(vehicle_transform, actor.get_location())
            except Exception:
                continue
            if local_x < -1.5:
                continue

            actor_type = "walker" if type_id.startswith("walker.") else "vehicle"
            abs_y = abs(float(local_y))
            distance = math.hypot(float(local_x), float(local_y))
            actor_speed = 0.0
            local_velocity_y = 0.0
            try:
                velocity = actor.get_velocity()
                actor_speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
                yaw = math.radians(float(vehicle_transform.rotation.yaw))
                local_velocity_y = float(velocity.x) * math.sin(-yaw) + float(velocity.y) * math.cos(-yaw)
            except Exception:
                actor_speed = 0.0
                local_velocity_y = 0.0
            action = ""
            target_speed = 0.0
            priority = 0
            if actor_type == "walker":
                actor_location = actor.get_location()
                if local_y <= -2.4 and local_velocity_y <= 0.1:
                    continue
                if role_name == "task_walker" and local_x >= 6.0 and local_y <= -0.15:
                    continue
                if role_name == "task_walker" and local_x >= 6.0 and actor_location.x <= vehicle_transform.location.x - 5.0:
                    continue
                crosswalk_prebrake = (
                    role_name == "task_walker" and actor_speed > 0.15 and local_x <= 10.0 and abs_y <= 9.0
                )
                if not crosswalk_prebrake and abs_y >= 4.2:
                    continue
                if actor_speed <= 0.15 and local_x >= 8.0 and abs_y >= 3.2:
                    continue
                if crosswalk_prebrake or (local_x <= 6.0 and abs_y <= 4.2) or (local_x <= 15.0 and abs_y <= 3.0):
                    action = "stop"
                    target_speed = 0.0
                    priority = 2
                elif local_x <= 18.0 and abs_y <= 4.2:
                    action = "slow"
                    target_speed = 0.8
                    priority = 1
            else:
                if role_name == "task_obstacle":
                    if local_x <= 8.0 and local_y >= 2.25:
                        continue
                    if local_x <= 5.0 and abs_y <= 2.1:
                        action = "stop"
                        target_speed = 0.0
                        priority = 3
                    elif local_x <= 30.0 and abs_y <= 5.5:
                        action = "avoid_left"
                        target_speed = 1.8
                        priority = 3
                elif local_x <= 7.5 and abs_y <= 4.8:
                    action = "stop"
                    target_speed = 0.0
                    priority = 2
                elif local_x <= 22.0 and abs_y <= 5.5:
                    action = "slow"
                    target_speed = 1.2
                    priority = 1
            if not action:
                continue

            hazard = {
                "active": True,
                "action": action,
                "target_speed_mps": float(target_speed),
                "distance_m": float(distance),
                "local_x_m": float(local_x),
                "local_y_m": float(local_y),
                "actor_speed_mps": float(actor_speed),
                "actor_id": actor_id,
                "actor_type": actor_type,
                "type_id": type_id,
                "role_name": role_name,
                "source": "world_actor_proximity",
            }
            if action == "avoid_left":
                hazard["avoid_lateral_m"] = -2.7
            score = (priority, -distance)
            if best is None or score > (best[0], best[1]):
                best = (priority, -distance, hazard)

        if best is None:
            return {"active": False, "reason": "clear"}
        return best[2]

    def _obstacle_corridor_reference(self, vehicle: carla.Actor, hazard: Dict[str, Any]) -> Dict[str, Any]:
        hazard_active = (
            isinstance(hazard, dict)
            and bool(hazard.get("active", False))
            and str(hazard.get("action", "")).lower() == "avoid_left"
        )
        try:
            vehicle_transform = vehicle.get_transform()
            ego_x = float(vehicle_transform.location.x)
            ego_y = float(vehicle_transform.location.y)
        except Exception:
            return {}

        corridor_active = bool(getattr(self, "_obstacle_corridor_active", False) or hazard_active)
        if not corridor_active:
            return {}

        if ego_y <= 45.0 or ego_y > 90.0 or ego_x < -56.0 or ego_x > -35.0:
            self._obstacle_corridor_active = False
            return {}

        self._obstacle_corridor_active = True
        if ego_y >= 58.0:
            target_x = -45.0
            lookahead_y = 18.0
            target_speed = 1.8
        elif ego_y >= 50.0:
            target_x = -45.0
            lookahead_y = 14.0
            target_speed = 1.5
        else:
            target_x = -44.2
            lookahead_y = 12.0
            target_speed = 1.5

        target = carla.Location(x=target_x, y=ego_y - lookahead_y, z=float(vehicle_transform.location.z))
        local_x, local_y = self._local_target(vehicle_transform, target)
        local_x = self._clamp(local_x, 8.0, 22.0)
        local_y = self._clamp(local_y, -2.6, 2.6)
        if ego_x <= -47.0:
            local_y = max(local_y, 2.0)
        elif ego_x <= -45.4:
            local_y = max(local_y, 1.2)
        elif ego_x >= -40.5 and ego_y >= 58.0:
            local_y = min(local_y, -1.4)

        return {
            "route_target_local_x": float(local_x),
            "route_target_local_y": float(local_y),
            "route_target_source": "obstacle_corridor_reference",
            "obstacle_corridor_target_speed_mps": float(target_speed),
            "obstacle_corridor": {
                "active": True,
                "hazard_latched": bool(hazard_active),
                "target_world_x": float(target_x),
                "target_world_y": float(target.y),
                "ego_world_x": float(ego_x),
                "ego_world_y": float(ego_y),
            },
        }

    def _post_turn_straight_reference(self, vehicle: carla.Actor) -> Dict[str, Any]:
        try:
            vehicle_transform = vehicle.get_transform()
            ego_x = float(vehicle_transform.location.x)
            ego_y = float(vehicle_transform.location.y)
        except Exception:
            return {}

        if not (88.0 <= ego_y <= 112.0 and -52.5 <= ego_x <= -33.0):
            return {}

        target_x = -43.5
        target = carla.Location(x=target_x, y=ego_y - 18.0, z=float(vehicle_transform.location.z))
        local_x, local_y = self._local_target(vehicle_transform, target)
        local_x = self._clamp(local_x, 8.0, 22.0)
        local_y = self._clamp(local_y, -2.4, 2.4)
        if ego_x <= -48.0:
            local_y = max(local_y, 2.2)
        elif ego_x <= -46.0:
            local_y = max(local_y, 1.4)
        elif ego_x >= -36.5:
            local_y = min(local_y, -2.2)
        elif ego_x >= -39.0:
            local_y = min(local_y, -1.4)

        return {
            "route_target_local_x": float(local_x),
            "route_target_local_y": float(local_y),
            "route_target_source": "post_turn_straight_reference",
            "post_turn_corridor_target_speed_mps": 2.2,
            "post_turn_corridor": {
                "active": True,
                "target_world_x": float(target_x),
                "target_world_y": float(target.y),
                "ego_world_x": float(ego_x),
                "ego_world_y": float(ego_y),
            },
        }

    @staticmethod
    def _junction_turn_reference(
        vehicle: carla.Actor,
        world: Optional[carla.World],
        command: str,
        waypoint: Optional[carla.Waypoint] = None,
    ) -> Dict[str, Any]:
        turn_command = VisionEgoDriver._normalize_turn_command(command)
        if not turn_command or world is None:
            return {}
        try:
            active_waypoint = waypoint
            if active_waypoint is None:
                active_waypoint = world.get_map().get_waypoint(
                    vehicle.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            if active_waypoint is None or not bool(getattr(active_waypoint, "is_junction", False)):
                return {}

            vehicle_transform = vehicle.get_transform()
            ego_loc = vehicle_transform.location
            waypoint_yaw = float(active_waypoint.transform.rotation.yaw)
            junction = active_waypoint.get_junction()
            best: Optional[tuple[float, Dict[str, Any]]] = None
            for entry, exit_waypoint in junction.get_waypoints(carla.LaneType.Driving):
                entry_transform = entry.transform
                exit_transform = exit_waypoint.transform
                entry_yaw = float(entry_transform.rotation.yaw)
                exit_yaw = float(exit_transform.rotation.yaw)
                entry_heading_error = abs(VisionEgoDriver._angle_delta_deg(entry_yaw, waypoint_yaw))
                if entry_heading_error > 45.0:
                    continue

                turn_delta = VisionEgoDriver._angle_delta_deg(exit_yaw, entry_yaw)
                if turn_command == "right" and not (35.0 <= turn_delta <= 140.0):
                    continue
                if turn_command == "left" and not (-140.0 <= turn_delta <= -35.0):
                    continue
                if turn_command == "straight" and abs(turn_delta) > 35.0:
                    continue

                entry_loc = entry_transform.location
                entry_distance = math.hypot(float(entry_loc.x - ego_loc.x), float(entry_loc.y - ego_loc.y))
                if entry_distance > 35.0:
                    continue

                local_x, local_y = VisionEgoDriver._local_target(vehicle_transform, exit_transform.location)
                if local_x <= 2.0:
                    continue

                score = entry_heading_error + 0.20 * entry_distance + 0.02 * abs(local_y)
                reference = {
                    "route_target_local_x": float(local_x),
                    "route_target_local_y": float(local_y),
                    "route_target_source": "junction_turn_reference",
                    "route_target_turn_command": turn_command,
                    "route_target_turn_delta_deg": float(turn_delta),
                    "route_target_entry_distance_m": float(entry_distance),
                    "route_target_world_x": float(exit_transform.location.x),
                    "route_target_world_y": float(exit_transform.location.y),
                    "route_target_world_z": float(exit_transform.location.z),
                }
                if best is None or score < best[0]:
                    best = (score, reference)
            return best[1] if best is not None else {}
        except Exception:
            return {}

    def predict(self, ego_vehicle: Optional[carla.Actor] = None, world: Optional[carla.World] = None) -> carla.VehicleControl:
        vehicle = ego_vehicle or self.ego_vehicle
        active_world = world or self.world
        frames = self.sensor_rig.snapshot()
        lane_reference = self._lane_reference(vehicle, active_world)
        navigation_command = self._navigation_command_for_lane(lane_reference)
        route_waypoint = lane_reference.pop("_route_waypoint", None)
        if bool(lane_reference.get("in_junction", False)):
            turn_reference = self._junction_turn_reference(
                vehicle,
                active_world,
                navigation_command,
                waypoint=route_waypoint,
            )
            if turn_reference:
                self._junction_cached_turn_target = (
                    navigation_command,
                    carla.Location(
                        x=float(turn_reference["route_target_world_x"]),
                        y=float(turn_reference["route_target_world_y"]),
                        z=float(turn_reference.get("route_target_world_z", 0.0) or 0.0),
                    ),
                )
                lane_reference.update(turn_reference)
            else:
                cached_command, cached_target = (
                    self._junction_cached_turn_target
                    if self._junction_cached_turn_target is not None
                    else ("", None)
                )
                if cached_target is not None and cached_command == navigation_command:
                    try:
                        local_x, local_y = self._local_target(vehicle.get_transform(), cached_target)
                    except Exception:
                        local_x, local_y = 0.0, 0.0
                    if local_x > 2.0:
                        lane_reference.update(
                            {
                                "route_target_local_x": float(local_x),
                                "route_target_local_y": float(local_y),
                                "route_target_source": "junction_turn_reference_cached",
                                "route_target_turn_command": navigation_command,
                            }
                        )
            if (
                bool(lane_reference.get("in_junction", False))
                and (
                    navigation_command in {"left", "right"}
                    or "route_target_local_x" not in lane_reference
                    or "route_target_local_y" not in lane_reference
                )
                and lane_reference.get("route_target_source") not in {"junction_turn_reference", "junction_turn_reference_cached"}
            ):
                lane_reference.update(
                    {
                        "route_target_local_x": 10.0,
                        "route_target_local_y": 0.0,
                        "route_target_source": "junction_heading_hold",
                        "route_target_turn_command": navigation_command,
                    }
                )
        else:
            self._junction_cached_turn_target = None
        obs = {
            "rgb": frames.get("rgb"),
            "depth": frames.get("depth") if self.use_depth else None,
            "semantic": frames.get("semantic") if self.use_semantic else None,
            "speed_mps": self._vehicle_speed_mps(vehicle),
            "navigation_command": navigation_command,
            "base_navigation_command": self.navigation_command,
            "first_junction_command": self.first_junction_command,
        }
        try:
            vehicle_transform = vehicle.get_transform()
            obs["ego_world_x"] = float(vehicle_transform.location.x)
            obs["ego_world_y"] = float(vehicle_transform.location.y)
            obs["ego_world_yaw_deg"] = float(vehicle_transform.rotation.yaw)
        except Exception:
            pass
        obs.update(lane_reference)
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
        obs["interaction_hazard"] = self._interaction_hazard(vehicle, active_world)
        obstacle_corridor_reference = self._obstacle_corridor_reference(vehicle, obs["interaction_hazard"])
        if obstacle_corridor_reference:
            self._junction_cached_turn_target = None
            obs.update(obstacle_corridor_reference)
        else:
            post_turn_reference = self._post_turn_straight_reference(vehicle)
            if post_turn_reference:
                self._junction_cached_turn_target = None
                obs.update(post_turn_reference)
        uav_bev: Dict[str, Any] = {"available": False, "reason": "disabled"}
        if self.uav_bev_provider is not None:
            try:
                uav_bev = dict(self.uav_bev_provider() or {})
            except Exception as exc:
                uav_bev = {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        obs["uav_bev"] = uav_bev
        self.last_observation = dict(obs)
        control = self.policy.predict(obs)
        self.last_diagnostics = dict(getattr(self.policy, "last_diagnostics", {}))
        self.last_diagnostics["vision_detector"] = detector_diagnostics
        self.last_diagnostics["vision_obstacle"] = bool(vision_obstacle)
        self.last_diagnostics["uav_bev"] = uav_bev
        return control

    def destroy(self) -> None:
        self.sensor_rig.destroy()
