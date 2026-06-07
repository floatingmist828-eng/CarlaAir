from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import carla

from .core import TRAFFIC_MANAGER_PORT, configure_autopilot, set_traffic_manager_speed


@dataclass
class SpawnedVehicle:
    actor: carla.Actor
    spawn_index: int


@dataclass
class SpawnedWalker:
    actor: carla.Actor
    spawn_index: int
    controller: Optional[carla.Actor] = None
    target: Optional[carla.Location] = None
    speed_mps: float = 0.0


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
    spawn_indices: Optional[Sequence[int]] = None,
    speed_difference: float = 25.0,
    role_name: str = "task_traffic",
) -> List[SpawnedVehicle]:
    explicit_indices = [int(item) for item in (spawn_indices or [])]
    if count <= 0 and not explicit_indices:
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
    requested_indices = explicit_indices[: max(0, int(count))] if explicit_indices else []
    if not requested_indices:
        requested_indices = [int(start_index) + i for i in range(max(0, int(count)))]
    for requested_idx in requested_indices:
        spawn_idx = int(requested_idx) % len(spawn_points)
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


def _choose_walker_blueprint(bp_lib: carla.BlueprintLibrary, blueprint_id: Optional[str]) -> carla.ActorBlueprint:
    if blueprint_id:
        bp = bp_lib.find(blueprint_id)
        if bp is not None:
            return bp
    candidates = bp_lib.filter("walker.pedestrian.*")
    if not candidates:
        raise RuntimeError("No pedestrian blueprints available.")
    return candidates[0]


def _crossing_source_target(
    spawn_tf: carla.Transform,
    side_offset_m: float,
    longitudinal_offset_m: float = 0.0,
) -> tuple[carla.Location, carla.Location]:
    yaw = math.radians(float(spawn_tf.rotation.yaw))
    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)
    loc = spawn_tf.location
    base_x = float(loc.x + forward_x * longitudinal_offset_m)
    base_y = float(loc.y + forward_y * longitudinal_offset_m)
    source = carla.Location(
        x=float(base_x + right_x * side_offset_m),
        y=float(base_y + right_y * side_offset_m),
        z=float(loc.z + 0.6),
    )
    target = carla.Location(
        x=float(base_x - right_x * side_offset_m),
        y=float(base_y - right_y * side_offset_m),
        z=float(loc.z + 0.6),
    )
    return source, target


def spawn_traffic_walkers(
    client: carla.Client,
    world: carla.World,
    count: int,
    blueprint_id: Optional[str] = None,
    start_index: int = 5,
    spawn_indices: Optional[Sequence[int]] = None,
    role_name: str = "task_walker",
    crossing_distance_m: float = 18.0,
    crossing_offsets_m: Optional[Sequence[float]] = None,
    use_ai_controller: bool = True,
    speed_mps: float = 1.4,
) -> List[SpawnedWalker]:
    explicit_indices = [int(item) for item in (spawn_indices or [])]
    crossing_offsets = [float(item) for item in (crossing_offsets_m or [])]
    if count <= 0 and not explicit_indices:
        return []

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found.")

    walker_bp = _choose_walker_blueprint(bp_lib, blueprint_id)
    if walker_bp.has_attribute("role_name"):
        walker_bp.set_attribute("role_name", role_name)
    if walker_bp.has_attribute("is_invincible"):
        try:
            walker_bp.set_attribute("is_invincible", "false")
        except Exception:
            pass

    try:
        controller_bp = bp_lib.find("controller.ai.walker")
    except Exception:
        controller_bp = None

    walkers: List[SpawnedWalker] = []
    side_offset = max(2.0, float(crossing_distance_m) * 0.5)
    requested_indices = explicit_indices[: max(0, int(count))] if explicit_indices else []
    if not requested_indices:
        requested_indices = [int(start_index) + i for i in range(max(0, int(count)))]
    for item_idx, requested_idx in enumerate(requested_indices):
        spawn_idx = int(requested_idx) % len(spawn_points)
        longitudinal_offset = crossing_offsets[item_idx % len(crossing_offsets)] if crossing_offsets else 0.0
        source, target = _crossing_source_target(spawn_points[spawn_idx], side_offset, longitudinal_offset)
        walker = world.try_spawn_actor(walker_bp, carla.Transform(source, carla.Rotation()))
        if walker is None:
            continue

        controller = None
        if use_ai_controller and controller_bp is not None:
            try:
                controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
                controller.start()
                controller.go_to_location(target)
                controller.set_max_speed(float(speed_mps))
            except Exception:
                if controller is not None:
                    try:
                        controller.destroy()
                    except Exception:
                        pass
                controller = None
        walkers.append(
            SpawnedWalker(
                actor=walker,
                spawn_index=spawn_idx,
                controller=controller,
                target=target,
                speed_mps=float(speed_mps),
            )
        )

    return walkers
