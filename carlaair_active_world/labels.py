from __future__ import annotations

import math
from typing import Any, Dict, List

import carla

from .geometry import Vector3
from .core import vector_from_carla_location, vector_from_carla_vector


def project_constant_velocity(
    location: carla.Location,
    velocity: carla.Vector3D,
    horizon_sec: float,
    step_sec: float,
) -> List[Dict[str, float]]:
    states: List[Dict[str, float]] = []
    steps = max(1, int(round(horizon_sec / step_sec)))
    for i in range(1, steps + 1):
        dt = i * step_sec
        states.append(
            {
                "t": float(dt),
                "x": float(location.x + velocity.x * dt),
                "y": float(location.y + velocity.y * dt),
                "z": float(location.z + velocity.z * dt),
            }
        )
    return states


def ego_risk_proxy(ego_transform: carla.Transform, nearby_actors: List[carla.Actor]) -> float:
    ego_loc = ego_transform.location
    risk = 0.0
    for actor in nearby_actors:
        loc = actor.get_location()
        dist = math.sqrt(
            (loc.x - ego_loc.x) ** 2 +
            (loc.y - ego_loc.y) ** 2 +
            (loc.z - ego_loc.z) ** 2
        )
        if dist < 25.0:
            risk += max(0.0, 25.0 - dist) / 25.0
    return float(risk)


def _collision_proxy(ego_vehicle: carla.Actor, nearby_actors: List[carla.Actor]) -> int:
    ego_loc = ego_vehicle.get_location()
    try:
        ego_extent = max(float(ego_vehicle.bounding_box.extent.x), float(ego_vehicle.bounding_box.extent.y))
    except Exception:
        ego_extent = 1.2
    collision_count = 0
    for actor in nearby_actors:
        try:
            loc = actor.get_location()
            dist = math.sqrt((loc.x - ego_loc.x) ** 2 + (loc.y - ego_loc.y) ** 2)
            try:
                extent = max(float(actor.bounding_box.extent.x), float(actor.bounding_box.extent.y))
            except Exception:
                extent = 0.6 if str(actor.type_id).startswith("walker.") else 1.2
            if dist <= ego_extent + extent + 0.25:
                collision_count += 1
        except Exception:
            continue
    return collision_count


def build_labels(
    world: carla.World,
    ego_vehicle: carla.Actor,
    horizon_sec: float,
    step_sec: float,
) -> Dict[str, Any]:
    ego_transform = ego_vehicle.get_transform()
    vehicle_actors = [a for a in world.get_actors().filter("vehicle.*") if a.id != ego_vehicle.id]
    walker_actors = list(world.get_actors().filter("walker.pedestrian.*"))
    vehicle_labels = []
    for actor in vehicle_actors:
        try:
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            vehicle_labels.append(
                {
                    "actor_id": int(actor.id),
                    "type_id": str(actor.type_id),
                    "role_name": str(actor.attributes.get("role_name", "")),
                    "current": {
                        "x": float(transform.location.x),
                        "y": float(transform.location.y),
                        "z": float(transform.location.z),
                    },
                    "future": project_constant_velocity(
                        transform.location,
                        velocity,
                        horizon_sec=horizon_sec,
                        step_sec=step_sec,
                    ),
                }
            )
        except Exception:
            continue

    walker_labels = []
    for actor in walker_actors:
        try:
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            walker_labels.append(
                {
                    "actor_id": int(actor.id),
                    "type_id": str(actor.type_id),
                    "role_name": str(actor.attributes.get("role_name", "")),
                    "current": {
                        "x": float(transform.location.x),
                        "y": float(transform.location.y),
                        "z": float(transform.location.z),
                    },
                    "future": project_constant_velocity(
                        transform.location,
                        velocity,
                        horizon_sec=horizon_sec,
                        step_sec=step_sec,
                    ),
                }
            )
        except Exception:
            continue

    junction = None
    try:
        waypoint = world.get_map().get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        junction = bool(waypoint.is_junction)
    except Exception:
        junction = None

    nearby_actors = vehicle_actors + walker_actors
    collision_count = _collision_proxy(ego_vehicle, nearby_actors)
    return {
        "ego": {
            "actor_id": int(ego_vehicle.id),
            "junction": junction,
        },
        "vehicles": vehicle_labels,
        "walkers": walker_labels,
        "risk_proxy": ego_risk_proxy(ego_transform, nearby_actors),
        "collision_proxy": collision_count > 0,
        "collision_proxy_count": collision_count,
    }
