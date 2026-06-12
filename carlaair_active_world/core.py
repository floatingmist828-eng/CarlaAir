from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import airsim
import carla

from .geometry import ActorState, CandidateViewpoint, Pose, Vector3


CARLA_HOST = "localhost"
CARLA_PORT = 2000
AIRSIM_PORT = 41451
TRAFFIC_MANAGER_PORT = int(os.environ.get("CARLAAIR_TASK_TRAFFIC_MANAGER_PORT", "8001"))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def vector_from_carla_location(loc: carla.Location) -> Vector3:
    return Vector3(float(loc.x), float(loc.y), float(loc.z))


def vector_from_carla_vector(vec: carla.Vector3D) -> Vector3:
    return Vector3(float(vec.x), float(vec.y), float(vec.z))


def pose_from_carla_transform(transform: carla.Transform) -> Pose:
    rotation = transform.rotation
    return Pose(
        position=vector_from_carla_location(transform.location),
        roll=float(rotation.roll),
        pitch=float(rotation.pitch),
        yaw=float(rotation.yaw),
    )


def world_to_local_offset(base: carla.Transform, offset: Vector3) -> Vector3:
    yaw = math.radians(base.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    dx = offset.x * cos_yaw - offset.y * sin_yaw
    dy = offset.x * sin_yaw + offset.y * cos_yaw
    dz = offset.z
    return Vector3(
        base.location.x + dx,
        base.location.y + dy,
        base.location.z + dz,
    )


def heading_towards(src: Vector3, dst: Vector3) -> float:
    return math.degrees(math.atan2(dst.y - src.y, dst.x - src.x))


def connect_carla(host: str = CARLA_HOST, port: int = CARLA_PORT, timeout: float = 60.0):
    timeout = float(os.environ.get("CARLAAIR_CARLA_TIMEOUT_SEC", timeout))
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


def connect_airsim(port: int = AIRSIM_PORT, vehicle_name: Optional[str] = None):
    client = airsim.MultirotorClient(port=port)
    client.confirmConnection()
    resolved_name = vehicle_name
    try:
        names = list(client.listVehicles())
    except Exception:
        names = []
    if resolved_name and names and resolved_name not in names:
        resolved_name = names[0]
    try:
        if resolved_name:
            client.enableApiControl(True, vehicle_name=resolved_name)
            client.armDisarm(True, vehicle_name=resolved_name)
        else:
            client.enableApiControl(True)
            client.armDisarm(True)
    except Exception:
        if resolved_name:
            client.enableApiControl(True)
            client.armDisarm(True)
    return client


def cleanup_old_vehicles(world: carla.World) -> int:
    destroyed = 0
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            role_name = ""
            try:
                role_name = str(actor.attributes.get("role_name", ""))
            except Exception:
                role_name = ""
            if role_name != "ego":
                continue
            actor.destroy()
            destroyed += 1
        except Exception:
            pass
    if destroyed:
        time.sleep(0.25)
    return destroyed


def cleanup_actors_by_role(world: carla.World, role_names: set[str]) -> int:
    destroyed = 0
    for actor in world.get_actors():
        try:
            role_name = str(actor.attributes.get("role_name", ""))
        except Exception:
            role_name = ""
        if role_name not in role_names:
            continue
        try:
            actor.destroy()
            destroyed += 1
        except Exception:
            pass
    if destroyed:
        time.sleep(0.25)
    return destroyed


def spawn_ego_vehicle(
    world: carla.World,
    blueprint_id: str,
    spawn_index: int = 0,
    forward_m: float = 0.0,
) -> carla.Actor:
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(blueprint_id)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "ego")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on the current map.")
    spawn_index = max(0, min(spawn_index, len(spawn_points) - 1))
    spawn_point = spawn_points[spawn_index]
    if forward_m > 0.0:
        try:
            waypoint = world.get_map().get_waypoint(
                spawn_point.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            candidates = waypoint.next(float(forward_m))
            if candidates:
                spawn_point = candidates[0].transform
                spawn_point.location.z += 0.2
        except Exception:
            pass
    actor = world.try_spawn_actor(blueprint, spawn_point)
    if actor is None:
        for point in spawn_points:
            actor = world.try_spawn_actor(blueprint, point)
            if actor is not None:
                break
    if actor is None:
        raise RuntimeError("Failed to spawn ego vehicle.")
    return actor


def configure_autopilot(
    client: carla.Client,
    world: carla.World,
    vehicle: carla.Actor,
    speed_percentage: float = 40.0,
) -> None:
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.set_synchronous_mode(world.get_settings().synchronous_mode)
    tm.global_percentage_speed_difference(float(speed_percentage))
    vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)


def get_actor_state(actor: carla.Actor) -> ActorState:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    extent = None
    try:
        bbox = actor.bounding_box
        extent = vector_from_carla_vector(bbox.extent)
    except Exception:
        extent = None
    role_name = ""
    try:
        role_name = str(actor.attributes.get("role_name", ""))
    except Exception:
        role_name = ""
    return ActorState(
        actor_id=int(actor.id),
        type_id=str(actor.type_id),
        role_name=role_name,
        pose=pose_from_carla_transform(transform),
        velocity=vector_from_carla_vector(velocity),
        extent=extent,
    )


def collect_vehicle_states(world: carla.World, include_ego: bool = True) -> List[ActorState]:
    vehicles: List[ActorState] = []
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            if not include_ego:
                role = str(actor.attributes.get("role_name", ""))
                if role == "ego":
                    continue
            vehicles.append(get_actor_state(actor))
        except Exception:
            pass
    return vehicles


def collect_walker_states(world: carla.World) -> List[ActorState]:
    walkers: List[ActorState] = []
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        try:
            walkers.append(get_actor_state(actor))
        except Exception:
            pass
    return walkers


def find_drone_actor(world: carla.World, preferred_name: Optional[str] = None):
    keywords = ("drone", "uav", "multirotor", "quadcopter", "airsim")
    for actor in world.get_actors():
        type_id = actor.type_id.lower()
        if preferred_name:
            role_name = str(actor.attributes.get("role_name", "")).lower()
            if role_name == preferred_name.lower():
                return actor
        if any(key in type_id for key in keywords):
            return actor
    return None


def calibrate_offset(
    world: carla.World,
    air_client: airsim.MultirotorClient,
    preferred_name: Optional[str] = None,
) -> Tuple[float, float, float]:
    drone_actor = find_drone_actor(world, preferred_name=preferred_name)
    if drone_actor is None:
        return 0.0, 0.0, 0.0
    carla_loc = drone_actor.get_location()
    if preferred_name:
        air_pos = air_client.getMultirotorState(vehicle_name=preferred_name).kinematics_estimated.position
    else:
        air_pos = air_client.getMultirotorState().kinematics_estimated.position
    ox = air_pos.x_val - carla_loc.x
    oy = air_pos.y_val - carla_loc.y
    oz = air_pos.z_val - (-carla_loc.z)
    return float(ox), float(oy), float(oz)


def carla_to_airsim_ned(x: float, y: float, z: float, ox: float, oy: float, oz: float) -> Tuple[float, float, float]:
    return float(x + ox), float(y + oy), float(-z + oz)


def local_candidate_to_world(
    ego_transform: carla.Transform,
    candidate: CandidateViewpoint,
) -> Pose:
    yaw = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local = candidate.local_offset
    dx = local.x * cos_yaw - local.y * sin_yaw
    dy = local.x * sin_yaw + local.y * cos_yaw
    dz = local.z
    ego = ego_transform.location
    world_pos = Vector3(float(ego.x + dx), float(ego.y + dy), float(ego.z + dz))
    face_yaw = heading_towards(world_pos, vector_from_carla_location(ego))
    return Pose(position=world_pos, roll=0.0, pitch=0.0, yaw=face_yaw)


def carla_pose_to_airsim_pose(
    pose: Pose,
    ox: float,
    oy: float,
    oz: float,
) -> airsim.Pose:
    x, y, z = carla_to_airsim_ned(pose.position.x, pose.position.y, pose.position.z, ox, oy, oz)
    return airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(
            math.radians(pose.pitch),
            math.radians(pose.roll),
            math.radians(pose.yaw),
        ),
    )


def move_uav_to(
    air_client: airsim.MultirotorClient,
    pose: Pose,
    ox: float,
    oy: float,
    oz: float,
    vehicle_name: Optional[str] = None,
    duration: float = 0.2,
) -> None:
    air_pose = carla_pose_to_airsim_pose(pose, ox, oy, oz)
    if vehicle_name:
        air_client.simSetVehiclePose(air_pose, True, vehicle_name=vehicle_name)
    else:
        air_client.simSetVehiclePose(air_pose, True)
    try:
        kinematics_cls = getattr(airsim, "KinematicsState", None)
        vector_cls = getattr(airsim, "Vector3r", None)
        set_kinematics = getattr(air_client, "simSetKinematics", None)
        if kinematics_cls is None or vector_cls is None or set_kinematics is None:
            return
        kinematics = kinematics_cls()
        kinematics.position = air_pose.position
        kinematics.orientation = air_pose.orientation
        kinematics.linear_velocity = vector_cls(0.0, 0.0, 0.0)
        kinematics.angular_velocity = vector_cls(0.0, 0.0, 0.0)
        kinematics.linear_acceleration = vector_cls(0.0, 0.0, 0.0)
        kinematics.angular_acceleration = vector_cls(0.0, 0.0, 0.0)
        if vehicle_name:
            set_kinematics(kinematics, True, vehicle_name=vehicle_name)
        else:
            set_kinematics(kinematics, True)
    except Exception:
        pass


def set_uav_hover(
    air_client: airsim.MultirotorClient,
    vehicle_name: Optional[str] = None,
) -> None:
    try:
        if vehicle_name:
            air_client.hoverAsync(vehicle_name=vehicle_name)
            air_client.enableApiControl(False, vehicle_name=vehicle_name)
            air_client.armDisarm(False, vehicle_name=vehicle_name)
        else:
            air_client.hoverAsync()
            air_client.enableApiControl(False)
            air_client.armDisarm(False)
    except Exception:
        pass


def set_traffic_manager_speed(client: carla.Client, percentage: float) -> None:
    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    tm.global_percentage_speed_difference(float(percentage))
