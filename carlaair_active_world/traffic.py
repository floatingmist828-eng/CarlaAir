from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import carla

from .core import TRAFFIC_MANAGER_PORT, configure_autopilot, set_traffic_manager_speed


@dataclass
class SpawnedVehicle:
    actor: carla.Actor
    spawn_index: int


def _choose_blueprint(bp_lib: carla.BlueprintLibrary, blueprint_id: Optional[str]) -> carla.ActorBlueprint:
    if blueprint_id:
        bp = bp_lib.find(blueprint_id)
        if bp is not None:
            return bp
    candidates = bp_lib.filter("vehicle.*")
    if not candidates:
        raise RuntimeError("No vehicle blueprints available.")
    return candidates[0]


def spawn_traffic_vehicles(
    client: carla.Client,
    world: carla.World,
    count: int,
    blueprint_id: Optional[str] = None,
    start_index: int = 1,
    speed_difference: float = 25.0,
    role_name: str = "task_traffic",
) -> List[SpawnedVehicle]:
    if count <= 0:
        return []

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    chosen_bp = _choose_blueprint(bp_lib, blueprint_id)
    if chosen_bp.has_attribute("role_name"):
        chosen_bp.set_attribute("role_name", role_name)
    if chosen_bp.has_attribute("color"):
        try:
            chosen_bp.set_attribute("color", "0,120,255")
        except Exception:
            pass

    vehicles: List[SpawnedVehicle] = []
    for i in range(count):
        spawn_idx = (start_index + i) % len(spawn_points)
        spawn_tf = spawn_points[spawn_idx]
        actor = world.try_spawn_actor(chosen_bp, spawn_tf)
        if actor is None:
            continue
        try:
            actor.set_simulate_physics(True)
        except Exception:
            pass
        try:
            actor.set_autopilot(True, TRAFFIC_MANAGER_PORT)
        except Exception:
            pass
        vehicles.append(SpawnedVehicle(actor=actor, spawn_index=spawn_idx))

    set_traffic_manager_speed(client, speed_difference)
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    try:
        tm.set_synchronous_mode(world.get_settings().synchronous_mode)
    except Exception:
        pass
    for item in vehicles:
        try:
            configure_autopilot(client, world, item.actor, speed_percentage=speed_difference)
        except Exception:
            pass
    return vehicles
