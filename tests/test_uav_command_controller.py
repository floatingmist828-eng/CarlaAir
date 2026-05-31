from __future__ import annotations

import math

import airsim

from carlaair_active_world.control import UAVCommandController


class _FakePosition:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x_val = x
        self.y_val = y
        self.z_val = z


class _FakeOrientation:
    pass


class _FakeKinematics:
    def __init__(self):
        self.position = _FakePosition(10.0, 20.0, -5.0)
        self.orientation = _FakeOrientation()


class _FakeState:
    def __init__(self):
        self.kinematics_estimated = _FakeKinematics()


class _FakeAsyncAction:
    def __init__(self, client):
        self.client = client

    def join(self):
        self.client.joined = True


class _FakeClient:
    def __init__(self):
        self.joined = False
        self.move_calls = []
        self.api_enabled = []
        self.armed = []

    def listVehicles(self):
        return ["SimpleFlight"]

    def enableApiControl(self, enabled, vehicle_name=None):
        self.api_enabled.append((enabled, vehicle_name))

    def armDisarm(self, armed, vehicle_name=None):
        self.armed.append((armed, vehicle_name))

    def getMultirotorState(self, vehicle_name=None):
        return _FakeState()

    def moveToPositionAsync(self, x, y, z, velocity, vehicle_name=None):
        self.move_calls.append((x, y, z, velocity, vehicle_name))
        return _FakeAsyncAction(self)


def test_move_relative_uses_position_target():
    client = _FakeClient()
    controller = UAVCommandController(client, vehicle_name="SimpleFlight")
    controller.api_ready = True

    controller.move_relative(3.0, -2.0, 4.0, speed=2.5)

    assert client.move_calls == [(13.0, 18.0, -1.0, 2.5, "SimpleFlight")]
    assert client.joined is True


def test_move_body_relative_converts_body_frame_to_world_frame(monkeypatch):
    client = _FakeClient()
    controller = UAVCommandController(client, vehicle_name="SimpleFlight")
    controller.api_ready = True

    monkeypatch.setattr(airsim, "to_eularian_angles", lambda q: (0.0, 0.0, math.pi / 2.0))

    controller.move_body_relative(10.0, 5.0, -3.0, speed=3.0)

    assert len(client.move_calls) == 1
    x, y, z, velocity, vehicle_name = client.move_calls[0]
    assert round(x, 6) == 5.0
    assert round(y, 6) == 30.0
    assert round(z, 6) == -8.0
    assert velocity == 3.0
    assert vehicle_name == "SimpleFlight"
    assert client.joined is True
