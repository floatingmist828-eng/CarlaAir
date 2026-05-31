from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import carla


class VisionPolicy(ABC):
    @abstractmethod
    def predict(self, obs: Dict[str, Any]) -> carla.VehicleControl:
        raise NotImplementedError
