from __future__ import annotations

import airsim

from carlaair_active_world.core import move_uav_to
from carlaair_active_world.geometry import Pose, Vector3


class _FakeVector3r:
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0):
        self.x_val = float(x_val)
        self.y_val = float(y_val)
        self.z_val = float(z_val)


class _FakePose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class _FakeKinematicsState:
    def __init__(self):
        self.position = None
        self.orientation = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.linear_acceleration = None
        self.angular_acceleration = None


class _FakeAirClient:
    def __init__(self):
        self.pose_calls = []
        self.kinematics_calls = []

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name=None):
        self.pose_calls.append((pose, ignore_collision, vehicle_name))

    def simSetKinematics(self, state, ignore_collision, vehicle_name=None):
        self.kinematics_calls.append((state, ignore_collision, vehicle_name))


def test_move_uav_to_zeroes_kinematics_after_pose_set(monkeypatch):
    monkeypatch.setattr(airsim, "Vector3r", _FakeVector3r, raising=False)
    monkeypatch.setattr(airsim, "Pose", _FakePose, raising=False)
    monkeypatch.setattr(airsim, "KinematicsState", _FakeKinematicsState, raising=False)
    monkeypatch.setattr(airsim, "to_quaternion", lambda pitch, roll, yaw: (pitch, roll, yaw), raising=False)
    client = _FakeAirClient()

    move_uav_to(
        client,
        pose=Pose(position=Vector3(10.0, 20.0, 16.0), roll=0.0, pitch=0.0, yaw=90.0),
        ox=1.0,
        oy=2.0,
        oz=3.0,
        vehicle_name="SimpleFlight",
    )

    assert len(client.pose_calls) == 1
    assert len(client.kinematics_calls) == 1
    state, ignore_collision, vehicle_name = client.kinematics_calls[0]
    assert ignore_collision is True
    assert vehicle_name == "SimpleFlight"
    assert (state.position.x_val, state.position.y_val, state.position.z_val) == (11.0, 22.0, -13.0)
    assert state.linear_velocity.x_val == 0.0
    assert state.linear_velocity.y_val == 0.0
    assert state.linear_velocity.z_val == 0.0
    assert state.angular_velocity.x_val == 0.0
    assert state.angular_velocity.y_val == 0.0
    assert state.angular_velocity.z_val == 0.0
