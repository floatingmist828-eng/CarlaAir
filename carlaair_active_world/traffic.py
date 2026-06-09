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
    route_commands: Optional[Sequence[str]] = None,
    speed_difference: float = 25.0,
    role_name: str = "task_traffic",
) -> List[SpawnedVehicle]:
    explicit_indices = [int(item) for item in (spawn_indices or [])]
    route = [str(item) for item in (route_commands or [])]
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
        if route:
            try:
                tm.set_route(item.actor, route)
            except Exception:
                pass
    return vehicles


def _offset_transform(
    anchor: carla.Transform,
    forward_offset_m: float,
    lateral_offset_m: float,
    yaw_offset_deg: float,
) -> carla.Transform:
    yaw = math.radians(float(anchor.rotation.yaw))
    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    lateral_x = -math.sin(yaw)
    lateral_y = math.cos(yaw)
    location = carla.Location(
        x=float(anchor.location.x + forward_x * forward_offset_m + lateral_x * lateral_offset_m),
        y=float(anchor.location.y + forward_y * forward_offset_m + lateral_y * lateral_offset_m),
        z=float(anchor.location.z),
    )
    rotation = carla.Rotation(
        pitch=float(anchor.rotation.pitch),
        yaw=float(anchor.rotation.yaw + yaw_offset_deg),
        roll=float(anchor.rotation.roll),
    )
    return carla.Transform(location, rotation)


def spawn_static_obstacle_vehicles(
    world: carla.World,
    count: int,
    blueprint_id: Optional[str] = "vehicle.dodge.charger_police_2020",
    anchor_transform: Optional[carla.Transform] = None,
    forward_offsets_m: Optional[Sequence[float]] = None,
    lateral_offsets_m: Optional[Sequence[float]] = None,
    yaw_offsets_deg: Optional[Sequence[float]] = None,
    role_name: str = "task_obstacle",
) -> List[SpawnedVehicle]:
    if count <= 0:
        return []
    if anchor_transform is None:
        raise ValueError("anchor_transform is required for static obstacle vehicles.")

    bp_lib = world.get_blueprint_library()
    chosen_bp = _choose_blueprint(bp_lib, blueprint_id)
    if chosen_bp.has_attribute("role_name"):
        chosen_bp.set_attribute("role_name", role_name)
    if chosen_bp.has_attribute("color"):
        try:
            chosen_bp.set_attribute("color", "220,40,30")
        except Exception:
            pass

    forward_offsets = [float(item) for item in (forward_offsets_m or [])]
    lateral_offsets = [float(item) for item in (lateral_offsets_m or [])]
    yaw_offsets = [float(item) for item in (yaw_offsets_deg or [])]
    obstacles: List[SpawnedVehicle] = []
    for idx in range(max(0, int(count))):
        transform = _offset_transform(
            anchor_transform,
            forward_offsets[idx % len(forward_offsets)] if forward_offsets else float(idx) * 4.0,
            lateral_offsets[idx % len(lateral_offsets)] if lateral_offsets else 0.0,
            yaw_offsets[idx % len(yaw_offsets)] if yaw_offsets else 0.0,
        )
        actor = world.try_spawn_actor(chosen_bp, transform)
        if actor is None:
            continue
        try:
            actor.set_autopilot(False, TRAFFIC_MANAGER_PORT)
        except Exception:
            try:
                actor.set_autopilot(False)
            except Exception:
                pass
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass
        try:
            control = carla.VehicleControl()
            control.brake = 1.0
            control.throttle = 0.0
            control.hand_brake = True
            actor.apply_control(control)
        except Exception:
            pass
        obstacles.append(SpawnedVehicle(actor=actor, spawn_index=-1))
    return obstacles


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
