import airsim

from carlaair_active_world.core import move_uav_to
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
