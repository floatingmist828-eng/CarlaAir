from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import airsim
import carla
import numpy as np
from PIL import Image

from .adversarial import apply_vision_attack


def _carla_image_to_rgb_array(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()


def _carla_semantic_to_tag_array(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, 2].copy()


def _depth_to_vis(depth_resp) -> np.ndarray:
    # AirSim depth image in float meters.
    depth = np.array(depth_resp.image_data_float, dtype=np.float32)
    if depth.size == 0:
        return np.zeros((depth_resp.height, depth_resp.width, 3), dtype=np.uint8)
    depth = depth.reshape((depth_resp.height, depth_resp.width))
    depth_norm = np.clip(depth / 80.0, 0.0, 1.0)
    vis = (255.0 * (1.0 - depth_norm))[..., None].repeat(3, axis=2)
    return vis.astype(np.uint8)


@dataclass
class VehicleSensorRig:
    world: carla.World
    parent: carla.Actor
    name: str
    width: int = 640
    height: int = 360
    attach_transform: carla.Transform = field(
        default_factory=lambda: carla.Transform(
            carla.Location(x=1.8, z=1.7),
            carla.Rotation(pitch=-12.0),
        )
    )
    sensors: List[carla.Sensor] = field(default_factory=list)
    latest: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    _locks: Dict[str, threading.Lock] = field(default_factory=dict)
    vision_attack: str = "none"
    vision_attack_intensity: float = 1.0
    disable_depth: bool = False
    disable_semantic: bool = False

    def spawn(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        sensor_specs = [("rgb", "sensor.camera.rgb")]
        if not self.disable_depth:
            sensor_specs.append(("depth", "sensor.camera.depth"))
        if not self.disable_semantic:
            sensor_specs.append(("semantic", "sensor.camera.semantic_segmentation"))
        for sensor_name, sensor_type in sensor_specs:
            bp = bp_lib.find(sensor_type)
            bp.set_attribute("image_size_x", str(self.width))
            bp.set_attribute("image_size_y", str(self.height))
            bp.set_attribute("fov", "90")
            sensor = self.world.spawn_actor(bp, self.attach_transform, attach_to=self.parent)
            self.sensors.append(sensor)
            self.latest[sensor_name] = None
            self._locks[sensor_name] = threading.Lock()

            def make_callback(kind: str):
                def _callback(image):
                    if kind == "rgb":
                        arr = _carla_image_to_rgb_array(image)
                    elif kind == "semantic":
                        arr = _carla_semantic_to_tag_array(image)
                    else:
                        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
                        raw = raw.reshape((image.height, image.width, 4))[:, :, :3]
                        r = raw[:, :, 2].astype(np.float32)
                        g = raw[:, :, 1].astype(np.float32)
                        b = raw[:, :, 0].astype(np.float32)
                        depth = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0
                        depth_norm = np.clip(depth / 80.0, 0.0, 1.0)
                        arr = (255.0 * (1.0 - depth_norm))[..., None].repeat(3, axis=2).astype(np.uint8)
                    with self._locks[kind]:
                        self.latest[kind] = arr

                return _callback

            sensor.listen(make_callback(sensor_name))

    def snapshot(self) -> Dict[str, Optional[np.ndarray]]:
        out: Dict[str, Optional[np.ndarray]] = {}
        for key in self.latest:
            with self._locks[key]:
                value = self.latest[key]
                out[key] = None if value is None else value.copy()
        return apply_vision_attack(
            out,
            attack=self.vision_attack,
            intensity=self.vision_attack_intensity,
            disable_semantic=self.disable_semantic,
        )

    def destroy(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
        self.sensors.clear()
        self.latest.clear()
        self._locks.clear()


@dataclass
class UAVSensorRig:
    client: airsim.MultirotorClient
    camera_name: str = "front_center"
    record_depth: bool = True
    width: int = 1280
    height: int = 960
    rpc_lock: Optional[threading.Lock] = None

    def snapshot(self) -> Dict[str, np.ndarray]:
        camera_candidates = [self.camera_name, "front_center", "0"]
        seen = set()
        for camera in camera_candidates:
            if camera in seen:
                continue
            seen.add(camera)
            try:
                requests = [
                    airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False),
                ]
                if self.record_depth:
                    requests.append(
                        airsim.ImageRequest(camera, airsim.ImageType.DepthPerspective, True, False)
                    )
                if self.rpc_lock is None:
                    responses = self.client.simGetImages(requests)
                else:
                    with self.rpc_lock:
                        responses = self.client.simGetImages(requests)
                result: Dict[str, np.ndarray] = {}
                if not responses:
                    continue
                scene = responses[0]
                rgb = np.frombuffer(scene.image_data_uint8, dtype=np.uint8)
                rgb = rgb.reshape(scene.height, scene.width, 3).copy()
                result["rgb"] = rgb[:, :, ::-1].copy()
                if self.record_depth and len(responses) > 1:
                    result["depth"] = _depth_to_vis(responses[1])
                self.camera_name = camera
                return result
            except Exception:
                continue
        return {}


def save_numpy_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
