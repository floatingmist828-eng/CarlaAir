from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import carla
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return float(angle)


@dataclass
class EgoDriveConfig:
    target_speed_mps: float = 8.0
    lookahead_m: float = 10.0
    control_gain: float = 1.05
    speed_kp: float = 0.18
    speed_kd: float = 0.06
    max_throttle: float = 0.45
    brake_distance_m: float = 12.0
    min_clearance_m: float = 7.0
    steer_smoothing: float = 0.70
    junction_speed_mps: float = 3.0
    follow_distance_m: float = 16.0
    emergency_vehicle_clearance_m: float = 7.0


class RouteFollowingDriver:
    def __init__(self, config: Optional[EgoDriveConfig] = None) -> None:
        self.config = config or EgoDriveConfig()
        self._steer_history: deque[float] = deque(maxlen=5)
        self._speed_error_history: deque[float] = deque(maxlen=5)

    @staticmethod
    def _vehicle_speed_mps(vehicle: carla.Actor) -> float:
        velocity = vehicle.get_velocity()
        return float(np.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z))

    def _route_target_point(self, vehicle: carla.Actor, world: carla.World) -> Optional[np.ndarray]:
        try:
            waypoint = world.get_map().get_waypoint(
                vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            candidates = waypoint.next(max(2.0, float(self.config.lookahead_m)))
            if not candidates:
                return None
            ego_transform = vehicle.get_transform()
            ego_loc = ego_transform.location
            ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
            best = None
            best_cost = None
            for candidate in candidates:
                loc = candidate.transform.location
                dx = float(loc.x - ego_loc.x)
                dy = float(loc.y - ego_loc.y)
                local_x = dx * np.cos(-ego_yaw) - dy * np.sin(-ego_yaw)
                local_y = dx * np.sin(-ego_yaw) + dy * np.cos(-ego_yaw)
                if local_x <= 0.1:
                    continue
                heading_delta = abs(
                    _normalize_angle_deg(candidate.transform.rotation.yaw - ego_transform.rotation.yaw)
                )
                cost = heading_delta + 0.15 * abs(local_y)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = np.array([local_x, local_y], dtype=np.float32)
            return best
        except Exception:
            return None

    def _vehicle_ahead_clearance(self, vehicle: carla.Actor, world: carla.World) -> Optional[float]:
        try:
            ego_transform = vehicle.get_transform()
            ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
            ego_loc = ego_transform.location
            ego_extent_x = 2.5
            try:
                ego_extent_x = float(getattr(getattr(vehicle.bounding_box, "extent", None), "x", ego_extent_x))
            except Exception:
                pass
            best: Optional[float] = None
            for actor in world.get_actors().filter("vehicle.*"):
                try:
                    if int(actor.id) == int(vehicle.id):
                        continue
                except Exception:
                    pass
                try:
                    actor_tf = actor.get_transform()
                    dx = float(actor_tf.location.x - ego_loc.x)
                    dy = float(actor_tf.location.y - ego_loc.y)
                    local_x = dx * np.cos(-ego_yaw) - dy * np.sin(-ego_yaw)
                    local_y = dx * np.sin(-ego_yaw) + dy * np.cos(-ego_yaw)
                    if local_x <= 0.0 or local_x > 35.0:
                        continue
                    if abs(local_y) > 4.5:
                        continue
                    heading_delta = abs(
                        _normalize_angle_deg(actor_tf.rotation.yaw - ego_transform.rotation.yaw)
                    )
                    if heading_delta > 80.0:
                        continue
                    actor_extent_x = 2.5
                    try:
                        actor_extent_x = float(
                            getattr(getattr(actor.bounding_box, "extent", None), "x", actor_extent_x)
                        )
                    except Exception:
                        pass
                    clearance = local_x - ego_extent_x - actor_extent_x
                    if best is None or clearance < best:
                        best = float(clearance)
                except Exception:
                    continue
            return best
        except Exception:
            return None

    def predict(self, ego_vehicle: carla.Actor, world: carla.World) -> carla.VehicleControl:
        route_target = self._route_target_point(ego_vehicle, world)
        speed_mps = self._vehicle_speed_mps(ego_vehicle)
        clearance_m = self._vehicle_ahead_clearance(ego_vehicle, world)

        try:
            waypoint = world.get_map().get_waypoint(
                ego_vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            in_junction = bool(getattr(waypoint, "is_junction", False))
        except Exception:
            in_junction = False

        try:
            red_light = bool(ego_vehicle.is_at_traffic_light())
        except Exception:
            red_light = False

        if route_target is not None:
            route_steer = np.arctan2(float(route_target[1]), float(route_target[0])) / (np.pi / 2.0)
        else:
            route_steer = 0.0

        steer_raw = _clamp(self.config.control_gain * route_steer, -0.85, 0.85)
        self._steer_history.append(steer_raw)
        steer = float(np.mean(self._steer_history)) if self._steer_history else steer_raw
        steer = _clamp(self.config.steer_smoothing * steer + (1.0 - self.config.steer_smoothing) * steer_raw, -1.0, 1.0)

        curvature = min(1.0, abs(float(route_steer)) * 1.15)
        obstacle_factor = 1.0
        if clearance_m is not None:
            if clearance_m < self.config.min_clearance_m:
                obstacle_factor = 0.0
            elif clearance_m < self.config.brake_distance_m:
                obstacle_factor = (clearance_m - self.config.min_clearance_m) / max(
                    1e-3, self.config.brake_distance_m - self.config.min_clearance_m
                )

        lane_factor = 1.0
        curvature_factor = _clamp(1.0 - curvature * 0.80, 0.18, 1.0)
        target_speed = self.config.target_speed_mps * obstacle_factor * lane_factor * curvature_factor
        if in_junction:
            target_speed = min(target_speed, self.config.junction_speed_mps)
        if clearance_m is not None and clearance_m < self.config.follow_distance_m:
            follow_factor = _clamp(clearance_m / max(1.0, self.config.follow_distance_m), 0.0, 1.0)
            target_speed = min(target_speed, self.config.target_speed_mps * follow_factor)
        if clearance_m is not None and clearance_m < self.config.emergency_vehicle_clearance_m:
            target_speed = 0.0
        if red_light:
            target_speed = 0.0
        target_speed = max(0.0, target_speed)

        speed_error = target_speed - speed_mps
        self._speed_error_history.append(speed_error)
        speed_error_derivative = 0.0
        if len(self._speed_error_history) >= 2:
            speed_error_derivative = self._speed_error_history[-1] - self._speed_error_history[-2]

        throttle = self.config.speed_kp * speed_error + self.config.speed_kd * speed_error_derivative
        throttle = _clamp(throttle, 0.0, self.config.max_throttle)
        if in_junction or curvature > 0.7:
            throttle = min(throttle, 0.18)
        if red_light:
            throttle = 0.0

        brake = 0.0
        if red_light:
            brake = 1.0
        elif clearance_m is not None and clearance_m < self.config.min_clearance_m + 2.0:
            brake = 1.0
            throttle = 0.0
        elif speed_error < -0.8 and target_speed < 2.0:
            brake = _clamp((-speed_error) / max(1.0, self.config.target_speed_mps), 0.0, 1.0)
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        return control
