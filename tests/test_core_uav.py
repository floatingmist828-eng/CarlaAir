import pytest
import airsim
import carla

from carlaair_active_world.core import move_uav_to, spawn_ego_vehicle
from carlaair_active_world.geometry import Pose, Vector3


class _FakeVector3r:
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0) -> None:
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val


class _FakePose:
    def __init__(self, position, orientation) -> None:
        self.position = position
        self.orientation = orientation


class _FakeKinematicsState:
    pass


class _FakeAirClient:
    def __init__(self) -> None:
        self.pose_calls = []
        self.kinematics_calls = []

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name=None) -> None:
        self.pose_calls.append((pose, ignore_collision, vehicle_name))

    def simSetKinematics(self, kinematics, ignore_collision, vehicle_name=None) -> None:
        self.kinematics_calls.append((kinematics, ignore_collision, vehicle_name))


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
    air_pose, pose_ignore_collision, pose_vehicle_name = client.pose_calls[0]
    kinematics, kinematics_ignore_collision, kinematics_vehicle_name = client.kinematics_calls[0]
    assert pose_ignore_collision is True
    assert pose_vehicle_name == "SimpleFlight"
    assert kinematics_ignore_collision is True
    assert kinematics_vehicle_name == "SimpleFlight"
    assert kinematics.position is air_pose.position
    assert kinematics.orientation is air_pose.orientation
    assert kinematics.linear_velocity.x_val == 0.0
    assert kinematics.linear_velocity.y_val == 0.0
    assert kinematics.linear_velocity.z_val == 0.0
    assert kinematics.angular_velocity.x_val == 0.0
    assert kinematics.angular_velocity.y_val == 0.0
    assert kinematics.angular_velocity.z_val == 0.0


def test_spawn_ego_vehicle_rejects_distant_fallback_spawn():
    class _Blueprint:
        def has_attribute(self, name):
            return name == "role_name"

        def set_attribute(self, _name, _value):
            pass

    class _BlueprintLibrary:
        def find(self, _blueprint_id):
            return _Blueprint()

    class _Map:
        def get_spawn_points(self):
            return [
                carla.Transform(carla.Location(x=-28.0, y=130.0), carla.Rotation(yaw=-180.0)),
                carla.Transform(carla.Location(x=-20.0, y=130.0), carla.Rotation(yaw=-180.0)),
            ]

    class _World:
        def get_blueprint_library(self):
            return _BlueprintLibrary()

        def get_map(self):
            return _Map()

        def try_spawn_actor(self, _blueprint, transform):
            if transform.location.x == -20.0:
                return object()
            return None

    with pytest.raises(RuntimeError, match="near requested spawn"):
        spawn_ego_vehicle(_World(), "vehicle.tesla.model3", spawn_index=0)


def test_spawn_ego_vehicle_uses_explicit_transform_before_spawn_index():
    requested = carla.Transform(
        carla.Location(x=-28.3, y=130.15, z=0.6),
        carla.Rotation(yaw=-179.65),
    )
    spawned_at = []
    actor = object()

    class _Blueprint:
        def has_attribute(self, name):
            return name == "role_name"

        def set_attribute(self, _name, _value):
            pass

    class _BlueprintLibrary:
        def find(self, _blueprint_id):
            return _Blueprint()

    class _Map:
        def get_spawn_points(self):
            return [
                carla.Transform(carla.Location(x=-18.0, y=130.0), carla.Rotation(yaw=-180.0)),
                carla.Transform(carla.Location(x=-20.0, y=130.0), carla.Rotation(yaw=-180.0)),
            ]

    class _World:
        def get_blueprint_library(self):
            return _BlueprintLibrary()

        def get_map(self):
            return _Map()

        def try_spawn_actor(self, _blueprint, transform):
            spawned_at.append(transform)
            return actor

    result = spawn_ego_vehicle(
        _World(),
        "vehicle.tesla.model3",
        spawn_index=1,
        spawn_transform=requested,
    )

    assert result is actor
    assert spawned_at[0].location.x == -28.3
    assert spawned_at[0].location.y == 130.15
    assert spawned_at[0].rotation.yaw == -179.65
