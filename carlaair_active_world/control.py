from __future__ import annotations

from contextlib import nullcontext
import math
import threading
from dataclasses import dataclass
from typing import Optional

import airsim


@dataclass
class UAVCommandController:
    client: airsim.MultirotorClient
    vehicle_name: str = "SimpleFlight"
    rpc_lock: Optional[threading.RLock] = None
    api_ready: bool = False

    def _rpc_scope(self):
        if self.rpc_lock is None:
            return nullcontext()
        return self.rpc_lock

    def resolve_vehicle_name(self) -> str:
        try:
            with self._rpc_scope():
                names = list(self.client.listVehicles())
        except Exception:
            names = []
        if self.vehicle_name and names and self.vehicle_name in names:
            return self.vehicle_name
        if names:
            return names[0]
        return self.vehicle_name

    def ensure_api(self) -> None:
        if self.api_ready:
            return
        name = self.resolve_vehicle_name()
        try:
            with self._rpc_scope():
                self.client.enableApiControl(True, vehicle_name=name)
                self.client.armDisarm(True, vehicle_name=name)
        except Exception:
            with self._rpc_scope():
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
        self.vehicle_name = name
        self.api_ready = True

    def takeoff(self, timeout_sec: float = 8.0) -> None:
        self.ensure_api()
        with self._rpc_scope():
            self.client.takeoffAsync(timeout_sec=timeout_sec, vehicle_name=self.vehicle_name).join()

    def hover(self) -> None:
        try:
            with self._rpc_scope():
                self.client.hoverAsync(vehicle_name=self.vehicle_name)
        except Exception:
            pass

    def move_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float = 1.0,
        yaw_deg: Optional[float] = None,
    ) -> None:
        self.ensure_api()
        yaw_mode = airsim.YawMode(False, yaw_deg if yaw_deg is not None else 0.0)
        with self._rpc_scope():
            self.client.moveByVelocityAsync(
                float(vx),
                float(vy),
                float(vz),
                duration=float(duration),
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=self.vehicle_name,
            ).join()

    def move_relative(
        self,
        dx: float,
        dy: float,
        dz: float,
        speed: float = 2.0,
    ) -> None:
        self.ensure_api()
        with self._rpc_scope():
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            self.client.moveToPositionAsync(
                pos.x_val + float(dx),
                pos.y_val + float(dy),
                pos.z_val + float(dz),
                velocity=float(speed),
                vehicle_name=self.vehicle_name,
            ).join()

    def move_body_relative(
        self,
        forward_m: float,
        right_m: float,
        down_m: float,
        speed: float = 2.0,
    ) -> None:
        self.ensure_api()
        with self._rpc_scope():
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            q = state.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(q)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            dx = forward_m * cos_yaw - right_m * sin_yaw
            dy = forward_m * sin_yaw + right_m * cos_yaw
            self.client.moveToPositionAsync(
                pos.x_val + float(dx),
                pos.y_val + float(dy),
                pos.z_val + float(down_m),
                velocity=float(speed),
                vehicle_name=self.vehicle_name,
            ).join()

    def goto(self, x: float, y: float, z: float, speed: float = 2.0) -> None:
        self.ensure_api()
        with self._rpc_scope():
            self.client.moveToPositionAsync(
                float(x),
                float(y),
                float(z),
                velocity=float(speed),
                vehicle_name=self.vehicle_name,
            ).join()

    def rotate_yaw(self, yaw_rate_deg_s: float, duration: float = 1.0) -> None:
        self.ensure_api()
        with self._rpc_scope():
            self.client.rotateByYawRateAsync(
                float(yaw_rate_deg_s),
                duration=float(duration),
                vehicle_name=self.vehicle_name,
            ).join()

    def move_body_velocity(
        self,
        forward_mps: float,
        right_mps: float,
        down_mps: float,
        duration: float = 1.0,
    ) -> None:
        self.ensure_api()
        with self._rpc_scope():
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            q = state.kinematics_estimated.orientation
            _, _, yaw = airsim.to_eularian_angles(q)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            vx = forward_mps * cos_yaw - right_mps * sin_yaw
            vy = forward_mps * sin_yaw + right_mps * cos_yaw
            self.client.moveByVelocityAsync(
                float(vx),
                float(vy),
                float(down_mps),
                duration=float(duration),
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0.0),
                vehicle_name=self.vehicle_name,
            ).join()
