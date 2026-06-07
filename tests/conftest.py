from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if "carla" not in sys.modules:
    carla_stub = types.ModuleType("carla")

    class _Location:
        def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

    class _Transform:
        def __init__(self, location=None, rotation=None) -> None:
            self.location = location
            self.rotation = rotation

    class _Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0) -> None:
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

    class _VehicleControl:
        def __init__(self) -> None:
            self.throttle = 0.0
            self.steer = 0.0
            self.brake = 0.0
            self.reverse = False

    class _Vector3D:
        def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

    class _WalkerControl:
        def __init__(self) -> None:
            self.direction = _Vector3D()
            self.speed = 0.0
            self.jump = False

    class _WeatherParameters:
        ClearNoon = "ClearNoon"

        def __init__(
            self,
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            fog_distance=0.0,
            wetness=0.0,
            sun_altitude_angle=0.0,
        ) -> None:
            self.cloudiness = cloudiness
            self.precipitation = precipitation
            self.precipitation_deposits = precipitation_deposits
            self.wind_intensity = wind_intensity
            self.fog_density = fog_density
            self.fog_distance = fog_distance
            self.wetness = wetness
            self.sun_altitude_angle = sun_altitude_angle

    class _LaneType:
        Driving = "Driving"

    carla_stub.VehicleControl = _VehicleControl
    carla_stub.Vector3D = _Vector3D
    carla_stub.WalkerControl = _WalkerControl
    carla_stub.WeatherParameters = _WeatherParameters
    carla_stub.LaneType = _LaneType
    carla_stub.Actor = object
    carla_stub.World = object
    carla_stub.Sensor = object
    carla_stub.Transform = _Transform
    carla_stub.Location = _Location
    carla_stub.Rotation = _Rotation
    sys.modules["carla"] = carla_stub


if "airsim" not in sys.modules:
    airsim_stub = types.ModuleType("airsim")

    class _ImageType:
        Scene = 0
        DepthPerspective = 1

    class _ImageRequest:
        def __init__(self, camera_name, image_type, pixels_as_float, compress) -> None:
            self.camera_name = camera_name
            self.image_type = image_type
            self.pixels_as_float = pixels_as_float
            self.compress = compress

    airsim_stub.ImageType = _ImageType
    airsim_stub.ImageRequest = _ImageRequest
    airsim_stub.MultirotorClient = object
    airsim_stub.to_eularian_angles = lambda orientation: (0.0, 0.0, 0.0)
    sys.modules["airsim"] = airsim_stub
