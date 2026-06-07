from __future__ import annotations

import math

import carla

from carlaair_active_world.traffic import _crossing_source_target


def test_crossing_source_target_applies_longitudinal_offset():
    spawn_tf = carla.Transform(
        carla.Location(x=-45.0, y=115.0, z=0.6),
        carla.Rotation(yaw=-90.0),
    )

    source, target = _crossing_source_target(spawn_tf, side_offset_m=7.0, longitudinal_offset_m=-2.0)

    assert math.isclose(source.x, -38.0, abs_tol=1e-3)
    assert math.isclose(target.x, -52.0, abs_tol=1e-3)
    assert math.isclose(source.y, 117.0, abs_tol=1e-3)
    assert math.isclose(target.y, 117.0, abs_tol=1e-3)
