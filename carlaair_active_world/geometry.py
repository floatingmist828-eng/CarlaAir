from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Pose:
    position: Vector3
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }


@dataclass
class ActorState:
    actor_id: int
    type_id: str
    role_name: str
    pose: Pose
    velocity: Vector3
    extent: Optional[Vector3] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "actor_id": self.actor_id,
            "type_id": self.type_id,
            "role_name": self.role_name,
            "pose": self.pose.to_dict(),
            "velocity": self.velocity.to_dict(),
        }
        if self.extent is not None:
            payload["extent"] = self.extent.to_dict()
        return payload


@dataclass
class CandidateViewpoint:
    name: str
    local_offset: Vector3
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "local_offset": self.local_offset.to_dict(),
            "weight": self.weight,
        }


@dataclass
class ScenarioResult:
    observation: Dict[str, Any]
    label: Dict[str, Any]
    info: Dict[str, Any]
    done: bool = False

