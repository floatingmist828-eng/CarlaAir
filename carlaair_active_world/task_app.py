from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import airsim
import carla
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .control import UAVCommandController
from .adversarial import apply_weather_preset
from .core import (
    AIRSIM_PORT,
    CARLA_HOST,
    CARLA_PORT,
    calibrate_offset,
    cleanup_old_vehicles,
    cleanup_actors_by_role,
    connect_airsim,
    connect_carla,
    configure_autopilot,
    carla_pose_to_airsim_pose,
    find_drone_actor,
    get_actor_state,
    local_candidate_to_world,
    move_uav_to,
    spawn_ego_vehicle,
)
from .geometry import CandidateViewpoint, Pose, Vector3
from .ego_driver import EgoDriveConfig, RouteFollowingDriver
from .labels import build_labels
from .recorder import EpisodeRecorder
from .scenario import ScenarioConfig
from .sensors import UAVSensorRig, VehicleSensorRig, save_numpy_image
from .traffic import spawn_traffic_vehicles, spawn_traffic_walkers
from .vision_driver import VisionEgoDriver
from .vision_models import TcpLiteVisionPolicy
from .vision_models.uav_bev import CachedUAVBEVProvider


def _safe_actor_role(actor: carla.Actor) -> str:
    try:
        return str(actor.attributes.get("role_name", ""))
    except Exception:
        return ""


class ActiveUAVTaskApp:
    def __init__(
        self,
        scenario: ScenarioConfig,
        output_dir: Path,
        sample_hz: float = 2.0,
        carla_host: str = CARLA_HOST,
        carla_port: int = CARLA_PORT,
        airsim_port: int = AIRSIM_PORT,
    ) -> None:
        self.scenario = scenario
        self.output_dir = output_dir
        self.sample_hz = sample_hz
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.airsim_port = airsim_port
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.air_client: Optional[airsim.MultirotorClient] = None
        self.controller: Optional[UAVCommandController] = None
        self.ego_vehicle: Optional[carla.Actor] = None
        self.traffic_actors: List[carla.Actor] = []
        self.walker_actors: List[carla.Actor] = []
        self.walker_controllers: List[carla.Actor] = []
        self._walker_targets: List[tuple[carla.Actor, carla.Location, float]] = []
        self._frozen_walker_targets: List[tuple[carla.Actor, carla.Location]] = []
        self.vehicle_sensors: Dict[int, VehicleSensorRig] = {}
        self.uav_sensors: Optional[UAVSensorRig] = None
        self.uav_bev_provider = CachedUAVBEVProvider(
            lambda: self.uav_sensors,
            refresh_hz=float(getattr(self.scenario, "uav_bev_refresh_hz", 2.0)),
        )
        self.ego_driver: Optional[Any] = None
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.start_time = 0.0
        self.collection_center = None
        self.collection_radius = float(self.scenario.sample_hotspot_radius_m)
        self.last_capture_time = 0.0
        self._stop = threading.Event()
        self._sampler: Optional[threading.Thread] = None
        self._viewer: Optional[threading.Thread] = None
        self._patrol: Optional[threading.Thread] = None
        self._recorder: Optional[EpisodeRecorder] = None
        self._sample_index = 0
        self._patrol_index = 0
        self._patrol_anchor: Optional[carla.Transform] = None
        self._current_uav_view: Optional[Dict[str, Any]] = None
        self._air_rpc_lock = threading.RLock()
        self._ego_driver_thread: Optional[threading.Thread] = None
        self._ego_control_mode = "autopilot"
        self._traffic_spawned = False
        self._walkers_spawned = False
        self.enable_viewer = os.environ.get("CARLAAIR_ENABLE_VIEWER", "0") == "1"

    def connect(self) -> None:
        self.client, self.world = connect_carla(self.carla_host, self.carla_port)
        current_map = ""
        try:
            current_map = self.world.get_map().name.split("/")[-1]
        except Exception:
            current_map = ""
        if self.scenario.map_name and current_map and current_map != self.scenario.map_name:
            raise RuntimeError(
                f"Scenario expects CARLA map '{self.scenario.map_name}', but the current map is '{current_map}'."
            )
        apply_weather_preset(self.world, getattr(self.scenario, "weather_preset", "none"))

    def _ensure_uav_connected(self) -> None:
        if not self.scenario.uav_enabled:
            return
        control_enabled = bool(getattr(self.scenario, "uav_control_enabled", True))
        if self.air_client is not None and not control_enabled:
            return
        if self.air_client is not None and self.controller is not None:
            return
        self.air_client = connect_airsim(self.airsim_port, vehicle_name=self.scenario.uav_name)
        if not control_enabled:
            return
        self.controller = UAVCommandController(
            self.air_client,
            vehicle_name=self.scenario.uav_name,
            rpc_lock=self._air_rpc_lock,
        )
        self.controller.vehicle_name = self.controller.resolve_vehicle_name()
        self.controller.api_ready = True
        self.scenario.uav_name = self.controller.vehicle_name
        self.ox, self.oy, self.oz = calibrate_offset(
            self.world,
            self.air_client,
            preferred_name=self.scenario.uav_name,
        )

    def cleanup(self) -> None:
        self._stop.set()
        if self._ego_driver_thread is not None:
            self._ego_driver_thread.join(timeout=2.0)
            self._ego_driver_thread = None
        destroyed_rig_ids = set()
        if self.ego_driver is not None and hasattr(self.ego_driver, "destroy"):
            driver_rig = getattr(self.ego_driver, "sensor_rig", None)
            try:
                self.ego_driver.destroy()
                if driver_rig is not None:
                    destroyed_rig_ids.add(id(driver_rig))
            except Exception:
                pass
        self.ego_driver = None
        for rig in self.vehicle_sensors.values():
            if id(rig) in destroyed_rig_ids:
                continue
            rig.destroy()
        self.vehicle_sensors.clear()
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.set_autopilot(False)
            except Exception:
                pass
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
            self.ego_vehicle = None
        for actor in self.traffic_actors:
            try:
                actor.set_autopilot(False)
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self.traffic_actors.clear()
        for controller in self.walker_controllers:
            try:
                controller.stop()
            except Exception:
                pass
            try:
                controller.destroy()
            except Exception:
                pass
        self.walker_controllers.clear()
        self._walker_targets.clear()
        for actor in self.walker_actors:
            try:
                actor.destroy()
            except Exception:
                pass
        self.walker_actors.clear()
        if self.air_client is not None and bool(getattr(self.scenario, "uav_control_enabled", True)):
            try:
                with self._air_rpc_lock:
                    self.air_client.hoverAsync(vehicle_name=self.scenario.uav_name)
            except Exception:
                pass
            try:
                with self._air_rpc_lock:
                    self.air_client.armDisarm(False, vehicle_name=self.scenario.uav_name)
                    self.air_client.enableApiControl(False, vehicle_name=self.scenario.uav_name)
            except Exception:
                pass
        if self.enable_viewer and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def setup(self) -> None:
        if self.client is None or self.world is None:
            self.connect()
        cleanup_actors_by_role(self.world, {"ego", "task_traffic", "task_walker", "task_ego", "task_uav"})
        cleanup_old_vehicles(self.world)
        self._ego_control_mode = str(self.scenario.ego_control_mode).lower()
        self.ego_vehicle = spawn_ego_vehicle(
            self.world,
            blueprint_id=self.scenario.ego_blueprint,
            spawn_index=self.scenario.ego_spawn_index,
            forward_m=float(getattr(self.scenario, "ego_spawn_forward_m", 0.0)),
        )
        if self._ego_control_mode in {"route_follow", "route", "behavior", "vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
            try:
                self.ego_vehicle.set_autopilot(False)
            except Exception:
                pass
            if self._ego_control_mode in {"vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
                policy = None
                if self._ego_control_mode == "vision_tcp_lite":
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
                        target_speed_mps=float(self.scenario.ego_target_speed_mps),
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
                    target_speed_mps=float(self.scenario.ego_target_speed_mps),
                    policy=policy,
                    use_semantic=self._ego_control_mode == "vision_simple",
                    use_depth=self._ego_control_mode != "vision_tcp_lite",
                    navigation_command=self.scenario.vision_navigation_command,
                    first_junction_command=str(getattr(self.scenario, "vision_first_junction_command", "")),
                    junction_command_sequence=list(
                        getattr(self.scenario, "vision_junction_command_sequence", []) or []
                    ),
                    junction_command_hold_sec=float(getattr(self.scenario, "vision_junction_command_hold_sec", 4.0)),
                    junction_command_hold_until_exit=bool(
                        getattr(self.scenario, "vision_junction_command_hold_until_exit", False)
                    ),
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
                        target_speed_mps=float(self.scenario.ego_target_speed_mps),
                        lookahead_m=float(self.scenario.ego_lookahead_m),
                    )
                )
            self.start_ego_driver()
        else:
            configure_autopilot(self.client, self.world, self.ego_vehicle, speed_percentage=35.0)
        ego_tf = self.ego_vehicle.get_transform()
        self.collection_center = carla.Location(
            x=float(ego_tf.location.x),
            y=float(ego_tf.location.y),
            z=float(ego_tf.location.z),
        )
        self.traffic_actors = []
        self.walker_actors = []
        self.walker_controllers = []
        self._walker_targets = []
        self._frozen_walker_targets = []
        self._traffic_spawned = max(0, int(self.scenario.traffic_vehicles)) <= 0
        self._walkers_spawned = max(0, int(self.scenario.traffic_walkers)) <= 0
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        self.start_time = time.time()
        if float(getattr(self.scenario, "traffic_spawn_delay_sec", 0.0)) <= 0.0:
            self._spawn_configured_traffic()
        if float(getattr(self.scenario, "walker_spawn_delay_sec", 0.0)) <= 0.0:
            self._spawn_configured_walkers()
        self._attach_vehicle_sensors()
        self.start_vehicle_viewer()
        if self.scenario.uav_enabled:
            self._ensure_uav_connected()
        if (
            self.scenario.uav_enabled
            and self.controller is not None
            and bool(getattr(self.scenario, "uav_control_enabled", True))
        ):
            self.controller.takeoff()
            start_candidate = (
                self.scenario.candidate_offsets[0]
                if self.scenario.candidate_offsets
                else CandidateViewpoint(
                    "front_lead_close",
                    Vector3(24.0, 0.0, float(max(self.scenario.uav_altitude, 22.0))),
                )
            )
            start_pose = local_candidate_to_world(ego_tf, start_candidate)
            with self._air_rpc_lock:
                move_uav_to(
                    self.air_client,
                    pose=start_pose,
                    ox=self.ox,
                    oy=self.oy,
                    oz=self.oz,
                    vehicle_name=self.scenario.uav_name,
                )
            self.controller.hover()
            self._current_uav_view = {
                "name": start_candidate.name,
                "pose": start_pose.to_dict(),
                "index": -1,
            }
            self.uav_sensors = UAVSensorRig(
                self.air_client,
                camera_name=str(getattr(self.scenario, "uav_bev_camera_name", "front_center")),
                rpc_lock=self._air_rpc_lock,
                )
            self._patrol_anchor = carla.Transform(
                location=carla.Location(
                    x=float(ego_tf.location.x),
                    y=float(ego_tf.location.y),
                    z=float(ego_tf.location.z),
                ),
                rotation=carla.Rotation(
                    pitch=0.0,
                    yaw=float(ego_tf.rotation.yaw),
                    roll=0.0,
                ),
            )
        elif self.scenario.uav_enabled and self.air_client is not None:
            self._current_uav_view = {
                "name": "airsim_camera_only",
                "pose": None,
                "index": -1,
            }
            self.uav_sensors = UAVSensorRig(
                self.air_client,
                camera_name=str(getattr(self.scenario, "uav_bev_camera_name", "front_center")),
                record_depth=False,
                rpc_lock=self._air_rpc_lock,
            )
        if (
            self.scenario.uav_enabled
            and self.scenario.uav_auto_patrol_enabled
            and bool(getattr(self.scenario, "uav_control_enabled", True))
        ):
            self.start_uav_patrol()
        print(
            f"[Setup] ego=1, traffic_spawned={len(self.traffic_actors)}, "
            f"vehicle_sensors={len(self.vehicle_sensors)}, uav={self.scenario.uav_name if self.scenario.uav_enabled else 'disabled'}"
        )

    def _spawn_configured_traffic(self) -> None:
        if self._traffic_spawned or self.client is None or self.world is None:
            return
        start_index = int(getattr(self.scenario, "traffic_spawn_start_index", -1))
        if start_index < 0:
            start_index = max(1, int(self.scenario.ego_spawn_index) + 1)
        spawned = spawn_traffic_vehicles(
            self.client,
            self.world,
            count=max(0, int(self.scenario.traffic_vehicles)),
            start_index=start_index,
            spawn_indices=list(getattr(self.scenario, "traffic_spawn_indices", []) or []),
            route_commands=list(getattr(self.scenario, "traffic_route_commands", []) or []),
            speed_difference=float(getattr(self.scenario, "traffic_speed_difference", 25.0)),
        )
        self.traffic_actors.extend(item.actor for item in spawned)
        self._traffic_spawned = True

    def _spawn_configured_walkers(self) -> None:
        if self._walkers_spawned or self.client is None or self.world is None:
            return
        start_index = int(getattr(self.scenario, "walker_spawn_start_index", -1))
        if start_index < 0:
            start_index = max(1, int(self.scenario.ego_spawn_index) + 5)
        spawned = spawn_traffic_walkers(
            self.client,
            self.world,
            count=max(0, int(self.scenario.traffic_walkers)),
            start_index=start_index,
            spawn_indices=list(getattr(self.scenario, "walker_spawn_indices", []) or []),
            crossing_distance_m=float(getattr(self.scenario, "walker_crossing_distance_m", 18.0)),
            crossing_offsets_m=list(getattr(self.scenario, "walker_crossing_offsets_m", []) or []),
            use_ai_controller=False,
            speed_mps=float(getattr(self.scenario, "walker_speed_mps", 1.4)),
        )
        self.walker_actors.extend(item.actor for item in spawned)
        self.walker_controllers.extend(item.controller for item in spawned if item.controller is not None)
        for item in spawned:
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
        frozen = []
        for actor, target in list(self._frozen_walker_targets):
            try:
                self._freeze_scripted_walker(actor, target)
                frozen.append((actor, target))
            except Exception:
                continue
        self._frozen_walker_targets = frozen

        remaining = []
        for actor, target, speed_mps in list(self._walker_targets):
            try:
                loc = actor.get_location()
                dx = float(target.x - loc.x)
                dy = float(target.y - loc.y)
                distance = math.hypot(dx, dy)
                if distance <= 0.25:
                    self._freeze_scripted_walker(actor, target)
                    self._frozen_walker_targets.append((actor, target))
                    continue
                control = carla.WalkerControl()
                control.direction = carla.Vector3D(dx / distance, dy / distance, 0.0)
                control.speed = float(speed_mps)
                actor.apply_control(control)
                remaining.append((actor, target, speed_mps))
            except Exception:
                continue
        self._walker_targets = remaining

    @staticmethod
    def _freeze_scripted_walker(actor: carla.Actor, target: carla.Location) -> None:
        try:
            rotation = actor.get_transform().rotation
            actor.set_transform(
                carla.Transform(
                    carla.Location(x=float(target.x), y=float(target.y), z=float(target.z)),
                    rotation,
                )
            )
        except Exception:
            pass
        control = carla.WalkerControl()
        control.speed = 0.0
        try:
            control.direction = carla.Vector3D(0.0, 0.0, 0.0)
        except Exception:
            pass
        actor.apply_control(control)

    def _attach_vehicle_sensors(self) -> None:
        if self.world is None:
            return
        vehicles = [self.ego_vehicle] + self.traffic_actors
        sensor_limit = max(1, int(self.scenario.vehicle_sensor_limit))
        for idx, actor in enumerate([v for v in vehicles if v is not None][:sensor_limit]):
            role = _safe_actor_role(actor) or f"vehicle_{idx}"
            existing_rig = getattr(getattr(self, "ego_driver", None), "sensor_rig", None)
            if actor is self.ego_vehicle and existing_rig is not None:
                self.vehicle_sensors[int(actor.id)] = existing_rig
                continue
            rig = VehicleSensorRig(
                self.world,
                actor,
                role,
                vision_attack=str(getattr(self.scenario, "vision_attack", "none")) if actor is self.ego_vehicle else "none",
                vision_attack_intensity=float(getattr(self.scenario, "vision_attack_intensity", 1.0)),
                disable_depth=self._ego_control_mode == "vision_tcp_lite" if actor is self.ego_vehicle else False,
                disable_semantic=self._ego_control_mode in {"vision_rgb_only", "vision_tcp_lite"} if actor is self.ego_vehicle else False,
            )
            rig.spawn()
            self.vehicle_sensors[int(actor.id)] = rig

    def _is_near_hotspot(self) -> bool:
        if self.collection_center is None or self.ego_vehicle is None:
            return True
        if not self.scenario.sample_only_near_hotspot:
            return True
        actors = [self.ego_vehicle] + self.traffic_actors
        radius = float(self.collection_radius)
        radius_sq = radius * radius
        for actor in actors:
            try:
                loc = actor.get_location()
            except Exception:
                continue
            dx = float(loc.x - self.collection_center.x)
            dy = float(loc.y - self.collection_center.y)
            dz = float(loc.z - self.collection_center.z)
            if dx * dx + dy * dy + dz * dz <= radius_sq:
                return True
        return False

    def sample_once(self) -> Dict[str, Any]:
        if self.world is None or self.ego_vehicle is None:
            raise RuntimeError("Call setup() before sampling.")
        self._sample_index += 1
        ts = time.time() - self.start_time
        near_hotspot = self._is_near_hotspot()
        min_interval = max(0.0, float(self.scenario.sample_min_interval_sec))
        if not near_hotspot:
            return {
                "sample_index": self._sample_index,
                "time": float(ts),
                "captured": False,
                "reason": "outside_hotspot",
            }
        if (time.time() - self.last_capture_time) < min_interval:
            return {
                "sample_index": self._sample_index,
                "time": float(ts),
                "captured": False,
                "reason": "interval_gate",
            }
        self.last_capture_time = time.time()
        obs = {
            "sample_index": self._sample_index,
            "time": float(ts),
            "captured": True,
            "reason": "captured",
            "uav_view": self._current_uav_view,
            "ego": get_actor_state(self.ego_vehicle).to_dict(),
            "traffic": [get_actor_state(a).to_dict() for a in self.traffic_actors],
            "uav": None,
            "vehicle_sensors": {},
            "ego_control": dict(getattr(self.ego_driver, "last_diagnostics", {}) or {}),
            "data_files": {
                "uav": {},
                "vehicles": {},
            },
        }
        if self.uav_sensors is not None:
            try:
                with self._air_rpc_lock:
                    uav_obs = self.uav_sensors.snapshot()
                obs["uav"] = {k: (v.shape if v is not None else None) for k, v in uav_obs.items()}
                obs["data_files"]["uav"] = self._persist_uav_sample(self._sample_index, uav_obs)
            except Exception:
                obs["uav"] = None
        for actor_id, rig in self.vehicle_sensors.items():
            try:
                vehicle_obs = rig.snapshot()
                obs["vehicle_sensors"][str(actor_id)] = {
                    k: (v.shape if v is not None else None) for k, v in vehicle_obs.items()
                }
                obs["data_files"]["vehicles"][str(actor_id)] = self._persist_vehicle_sample(
                    self._sample_index,
                    actor_id,
                    vehicle_obs,
                )
            except Exception:
                continue
        obs["labels"] = build_labels(
            self.world,
            self.ego_vehicle,
            horizon_sec=self.scenario.future_horizon_sec,
            step_sec=self.scenario.step_sec,
        )
        return obs

    def _persist_uav_sample(self, sample_index: int, data: Dict[str, np.ndarray]) -> Dict[str, str]:
        base = self.output_dir / "samples" / f"step_{sample_index:06d}" / "uav"
        saved: Dict[str, str] = {}
        for key, image in data.items():
            if image is None:
                continue
            path = base / f"{key}.png"
            save_numpy_image(path, image)
            saved[key] = str(path)
        return saved

    def _persist_vehicle_sample(self, sample_index: int, actor_id: int, data: Dict[str, np.ndarray]) -> Dict[str, str]:
        base = self.output_dir / "samples" / f"step_{sample_index:06d}" / f"vehicle_{actor_id}"
        saved: Dict[str, str] = {}
        for key, image in data.items():
            if image is None:
                continue
            path = base / f"{key}.png"
            save_numpy_image(path, image)
            saved[key] = str(path)
        return saved

    def start_sampler(self) -> None:
        if self._sampler is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._recorder = EpisodeRecorder(
            output_path=self.output_dir / "episode.json",
            meta={
                "scenario": self.scenario.to_dict(),
                "sample_hz": self.sample_hz,
            },
        )

        def _loop():
            interval = 1.0 / max(0.1, self.sample_hz)
            while not self._stop.is_set():
                try:
                    snapshot = self.sample_once()
                    if self._recorder is not None and snapshot.get("captured"):
                        self._recorder.append(snapshot)
                except Exception:
                    pass
                time.sleep(interval)

        self._sampler = threading.Thread(target=_loop, daemon=True)
        self._sampler.start()

    def start_vehicle_viewer(self) -> None:
        if self._viewer is not None:
            return
        if not self.vehicle_sensors:
            return
        if cv2 is None:
            return
        if not self.enable_viewer:
            return

        def _loop():
            interval = 0.08
            while not self._stop.is_set():
                for actor_id, rig in list(self.vehicle_sensors.items()):
                    try:
                        frames = rig.snapshot()
                        rgb = frames.get("rgb")
                        if rgb is None:
                            continue
                        window_name = f"VehicleView-{actor_id}"
                        cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    except Exception:
                        continue
                try:
                    cv2.waitKey(1)
                except Exception:
                    pass
                time.sleep(interval)

        self._viewer = threading.Thread(target=_loop, daemon=True)
        self._viewer.start()

    def start_ego_driver(self) -> None:
        if self._ego_driver_thread is not None:
            return
        if self.ego_driver is None or self.ego_vehicle is None or self.world is None:
            return

        def _loop():
            interval = 1.0 / max(0.1, float(self.scenario.ego_drive_hz))
            while not self._stop.is_set():
                try:
                    self._maybe_spawn_delayed_actors()
                    self._drive_scripted_walkers()
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

    def _set_patrol_enabled(self, enabled: bool) -> str:
        self.scenario.uav_auto_patrol_enabled = bool(enabled)
        if enabled:
            self.start_uav_patrol()
            return "UAV patrol enabled."
        return "UAV patrol disabled."

    def _uav_patrol_anchor_transform(self) -> Optional[carla.Transform]:
        if self.ego_vehicle is not None:
            try:
                ego_tf = self.ego_vehicle.get_transform()
                return carla.Transform(
                    location=carla.Location(
                        x=float(ego_tf.location.x),
                        y=float(ego_tf.location.y),
                        z=float(ego_tf.location.z),
                    ),
                    rotation=carla.Rotation(
                        pitch=0.0,
                        yaw=float(ego_tf.rotation.yaw),
                        roll=0.0,
                    ),
                )
            except Exception:
                pass
        return self._patrol_anchor

    def start_uav_patrol(self) -> None:
        if self._patrol is not None:
            return
        if self.air_client is None or self.controller is None:
            return
        if not self.scenario.candidate_offsets:
            return

        def _loop():
            interval = max(0.5, float(self.scenario.uav_patrol_interval_sec))
            while not self._stop.is_set():
                if not self.scenario.uav_auto_patrol_enabled:
                    time.sleep(0.2)
                    continue
                anchor = self._uav_patrol_anchor_transform()
                if self.air_client is None or self.controller is None or anchor is None:
                    time.sleep(0.5)
                    continue
                candidate = self.scenario.candidate_offsets[self._patrol_index % len(self.scenario.candidate_offsets)]
                try:
                    pose = local_candidate_to_world(anchor, candidate)
                    with self._air_rpc_lock:
                        move_uav_to(
                            self.air_client,
                            pose=pose,
                            ox=self.ox,
                            oy=self.oy,
                            oz=self.oz,
                            vehicle_name=self.scenario.uav_name,
                        )
                    self._current_uav_view = {
                        "name": candidate.name,
                        "pose": pose.to_dict(),
                        "index": int(self._patrol_index % len(self.scenario.candidate_offsets)),
                    }
                    self._patrol_index += 1
                except Exception:
                    pass
                time.sleep(interval)

        self._patrol = threading.Thread(target=_loop, daemon=True)
        self._patrol.start()

    def _disable_manual_patrol_override(self) -> None:
        if self.scenario.uav_auto_patrol_enabled:
            self.scenario.uav_auto_patrol_enabled = False

    def stop_sampler(self) -> Path:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2.0)
            self._sampler = None
        if self._recorder is None:
            return self.output_dir / "episode.json"
        return self._recorder.save()

    def handle_command(self, line: str) -> str:
        if self.controller is None:
            return "UAV controller not available."
        parts = line.strip().split()
        if not parts:
            return ""
        cmd = parts[0].lower()
        args = parts[1:]
        try:
            if cmd in {"help", "?"}:
                return (
                    "Commands: takeoff, hover, up N, down N, forward N, back N, left N, right N, "
                    "goto X Y Z, yaw DEG, status, sample, patrol on/off, quit"
                )
            if cmd == "takeoff":
                self._disable_manual_patrol_override()
                self.controller.takeoff()
                return "Takeoff started."
            if cmd == "hover":
                self._disable_manual_patrol_override()
                self.controller.hover()
                return "Hover."
            if cmd == "up" and len(args) == 1:
                self._disable_manual_patrol_override()
                self.controller.move_relative(0.0, 0.0, -float(args[0]), speed=3.0)
                return f"Move up {args[0]} m."
            if cmd == "down" and len(args) == 1:
                self._disable_manual_patrol_override()
                self.controller.move_relative(0.0, 0.0, float(args[0]), speed=3.0)
                return f"Move down {args[0]} m."
            if cmd == "forward" and len(args) == 1:
                self._disable_manual_patrol_override()
                self.controller.move_body_relative(float(args[0]), 0.0, 0.0, speed=3.0)
                return f"Move forward {args[0]} m."
            if cmd == "back" and len(args) == 1:
                self._disable_manual_patrol_override()
                self.controller.move_body_relative(-float(args[0]), 0.0, 0.0, speed=3.0)
                return f"Move back {args[0]} m."
            if cmd == "left" and len(args) == 1:
                self._disable_manual_patrol_override()
                self.controller.move_body_relative(0.0, float(args[0]), 0.0, speed=3.0)
                return f"Move left {args[0]} m."
            if cmd == "right" and len(args) == 1:
                self._disable_manual_patrol_override()
                self.controller.move_body_relative(0.0, -float(args[0]), 0.0, speed=3.0)
                return f"Move right {args[0]} m."
            if cmd == "goto" and len(args) == 3:
                self._disable_manual_patrol_override()
                self.controller.goto(float(args[0]), float(args[1]), float(args[2]), speed=3.0)
                return f"Goto {args[0]} {args[1]} {args[2]}."
            if cmd == "yaw" and len(args) == 1:
                self._disable_manual_patrol_override()
                self.controller.rotate_yaw(float(args[0]), duration=1.0)
                return f"Yaw {args[0]} deg/s."
            if cmd == "status":
                return self.describe()
            if cmd == "sample":
                snap = self.sample_once()
                if self._recorder is not None and snap.get("captured"):
                    self._recorder.append(snap)
                if snap.get("captured"):
                    return f"Sampled step {snap['sample_index']}."
                return f"Skipped step {snap['sample_index']}: {snap.get('reason')}"
            if cmd == "patrol" and len(args) == 1:
                choice = args[0].lower()
                if choice in {"on", "start", "enable"}:
                    return self._set_patrol_enabled(True)
                if choice in {"off", "stop", "disable"}:
                    return self._set_patrol_enabled(False)
                return "Usage: patrol on|off"
            if cmd == "quit":
                self._stop.set()
                return "quit"
            return f"Unknown command: {cmd}"
        except Exception as exc:
            return f"Command failed: {type(exc).__name__}: {exc}"

    def describe(self) -> str:
        ego = get_actor_state(self.ego_vehicle).to_dict() if self.ego_vehicle else {}
        lines = [
            f"Scenario: {self.scenario.name}",
            f"Ego: {ego.get('actor_id')} at {ego.get('pose', {}).get('position', {})}",
            f"Ego control: {self._ego_control_mode}",
            f"Traffic spawned: {len(self.traffic_actors)} / requested {self.scenario.traffic_vehicles}",
            f"Vehicle sensors: {len(self.vehicle_sensors)}",
            f"UAV: {self.scenario.uav_name if self.scenario.uav_enabled else 'disabled'}",
            f"UAV patrol: {'on' if self.scenario.uav_auto_patrol_enabled else 'off'}",
            f"UAV view: {self._current_uav_view['name'] if self._current_uav_view else 'static'}",
            f"Hotspot radius: {self.collection_radius:.1f} m",
            f"Samples: {self._sample_index}",
        ]
        return "\n".join(lines)
