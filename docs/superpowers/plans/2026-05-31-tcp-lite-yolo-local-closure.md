# TCP-Lite YOLO Local Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, testable TCP-Lite learning-control path plus optional YOLO/safety-gate perception while preserving the existing rule-based vision baseline.

**Architecture:** Keep `VisionEgoDriver` as the sensor and diagnostics coordinator. Add `vision_tcp_lite` as a new control mode that uses `TcpLiteVisionPolicy`, a small PyTorch TCP-Lite model, JSONL imitation training data, and a shared safety gate for detector obstacles and optional attack-pattern diagnostics.

**Tech Stack:** Python, NumPy, CARLA control stubs in tests, optional PyTorch, optional Pillow, optional Ultralytics, pytest.

---

## File Structure

- Modify `E:\a2\CarlaAir\carlaair_active_world\scenario.py`: add TCP-Lite and safety-gate scenario fields.
- Create `E:\a2\CarlaAir\configs\scenarios\town10hd_vision_tcp_lite.json`: safe default TCP-Lite scenario.
- Create `E:\a2\CarlaAir\carlaair_active_world\vision_models\safety_gate.py`: detector obstacle and attack-pattern gate helpers.
- Create `E:\a2\CarlaAir\carlaair_active_world\vision_models\tcp_lite.py`: lightweight TCP-Lite PyTorch model and command encoding helpers.
- Create `E:\a2\CarlaAir\carlaair_active_world\vision_models\tcp_lite_dataset.py`: JSONL imitation dataset loader.
- Create `E:\a2\CarlaAir\carlaair_active_world\vision_models\tcp_lite_policy.py`: inference policy returning `carla.VehicleControl`.
- Modify `E:\a2\CarlaAir\carlaair_active_world\vision_models\__init__.py`: export TCP-Lite policy/model helpers.
- Modify `E:\a2\CarlaAir\carlaair_active_world\vision_driver.py`: carry navigation command in observations.
- Modify `E:\a2\CarlaAir\carlaair_active_world\env.py`: instantiate TCP-Lite policy for `vision_tcp_lite`.
- Modify `E:\a2\CarlaAir\carlaair_active_world\task_app.py`: instantiate TCP-Lite policy for viewer/task runs.
- Create `E:\a2\CarlaAir\scripts\train_tcp_lite.py`: train from JSONL dataset and save checkpoint.
- Create `E:\a2\CarlaAir\tests\test_tcp_lite.py`: scenario, model, policy, gate, driver, and training smoke tests.

---

### Task 1: Scenario Fields And TCP-Lite Scenario

**Files:**
- Modify: `E:\a2\CarlaAir\carlaair_active_world\scenario.py`
- Create: `E:\a2\CarlaAir\configs\scenarios\town10hd_vision_tcp_lite.json`
- Test: `E:\a2\CarlaAir\tests\test_tcp_lite.py`

- [ ] **Step 1: Write failing scenario tests**

Add this to a new file `tests/test_tcp_lite.py`:

```python
from __future__ import annotations

from carlaair_active_world.scenario import ScenarioConfig


def test_tcp_lite_scenario_config_round_trips():
    scenario = ScenarioConfig.from_dict(
        {
            "name": "tcp_lite_config",
            "ego_control_mode": "vision_tcp_lite",
            "vision_model_path": "checkpoints/tcp_lite.pt",
            "vision_model_device": "cpu",
            "vision_navigation_command": "lane_follow",
            "vision_safety_gate_enabled": False,
            "vision_attack_pattern_gate": True,
        }
    )

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == "checkpoints/tcp_lite.pt"
    assert scenario.vision_model_device == "cpu"
    assert scenario.vision_navigation_command == "lane_follow"
    assert scenario.vision_safety_gate_enabled is False
    assert scenario.vision_attack_pattern_gate is True
    assert scenario.to_dict()["vision_model_path"] == "checkpoints/tcp_lite.pt"
    assert scenario.to_dict()["vision_model_device"] == "cpu"
    assert scenario.to_dict()["vision_navigation_command"] == "lane_follow"
    assert scenario.to_dict()["vision_safety_gate_enabled"] is False
    assert scenario.to_dict()["vision_attack_pattern_gate"] is True


def test_tcp_lite_project_scenario_loads():
    scenario = ScenarioConfig.load("configs/scenarios/town10hd_vision_tcp_lite.json")

    assert scenario.ego_control_mode == "vision_tcp_lite"
    assert scenario.vision_model_path == ""
    assert scenario.vision_model_device == "cpu"
    assert scenario.vision_navigation_command == "lane_follow"
    assert scenario.vision_safety_gate_enabled is True
    assert scenario.vision_attack_pattern_gate is False
    assert scenario.vehicle_sensor_limit == 1
    assert scenario.traffic_vehicles == 0
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_tcp_lite_scenario_config_round_trips tests/test_tcp_lite.py::test_tcp_lite_project_scenario_loads -q
```

Expected: FAIL because `ScenarioConfig` has no TCP-Lite fields and `configs/scenarios/town10hd_vision_tcp_lite.json` does not exist.

- [ ] **Step 3: Add scenario dataclass fields**

In `carlaair_active_world/scenario.py`, add fields after `vision_detector_confidence`:

```python
    vision_model_path: str = ""
    vision_model_device: str = "cpu"
    vision_navigation_command: str = "lane_follow"
    vision_safety_gate_enabled: bool = True
    vision_attack_pattern_gate: bool = False
```

In `from_dict`, add constructor arguments after `vision_detector_confidence`:

```python
            vision_model_path=str(data.get("vision_model_path", "")),
            vision_model_device=str(data.get("vision_model_device", "cpu")),
            vision_navigation_command=str(data.get("vision_navigation_command", "lane_follow")),
            vision_safety_gate_enabled=bool(data.get("vision_safety_gate_enabled", True)),
            vision_attack_pattern_gate=bool(data.get("vision_attack_pattern_gate", False)),
```

In `to_dict`, add keys after `vision_detector_confidence`:

```python
            "vision_model_path": self.vision_model_path,
            "vision_model_device": self.vision_model_device,
            "vision_navigation_command": self.vision_navigation_command,
            "vision_safety_gate_enabled": self.vision_safety_gate_enabled,
            "vision_attack_pattern_gate": self.vision_attack_pattern_gate,
```

- [ ] **Step 4: Add safe default TCP-Lite scenario**

Create `configs/scenarios/town10hd_vision_tcp_lite.json`:

```json
{
  "name": "town10hd_vision_tcp_lite",
  "map_name": "Town10HD",
  "ego_blueprint": "vehicle.tesla.model3",
  "ego_spawn_index": 20,
  "ego_sensor_mode": "vision",
  "ego_control_mode": "vision_tcp_lite",
  "ego_drive_hz": 8.0,
  "ego_target_speed_mps": 4.0,
  "ego_lookahead_m": 10.0,
  "traffic_vehicles": 0,
  "traffic_walkers": 0,
  "duration_sec": 60.0,
  "step_sec": 0.5,
  "future_horizon_sec": 3.0,
  "sample_only_near_hotspot": true,
  "sample_hotspot_radius_m": 70.0,
  "sample_min_interval_sec": 0.5,
  "vehicle_sensor_limit": 1,
  "vision_model_path": "",
  "vision_model_device": "cpu",
  "vision_navigation_command": "lane_follow",
  "vision_safety_gate_enabled": true,
  "vision_attack_pattern_gate": false,
  "uav_enabled": true,
  "uav_name": "SimpleFlight",
  "uav_altitude": 18.0,
  "uav_back_distance": 8.0,
  "uav_auto_patrol_enabled": false,
  "uav_patrol_interval_sec": 4.0,
  "candidate_offsets": [
    {"name": "front_high", "x": 18.0, "y": 0.0, "z": 10.0, "weight": 1.0},
    {"name": "front_left", "x": 15.0, "y": 8.0, "z": 10.0, "weight": 1.0},
    {"name": "front_right", "x": 15.0, "y": -8.0, "z": 10.0, "weight": 1.0},
    {"name": "top", "x": 0.0, "y": 0.0, "z": 22.0, "weight": 0.8},
    {"name": "rear_high", "x": -10.0, "y": 0.0, "z": 12.0, "weight": 0.7},
    {"name": "left_high", "x": 0.0, "y": 14.0, "z": 12.0, "weight": 0.8},
    {"name": "right_high", "x": 0.0, "y": -14.0, "z": 12.0, "weight": 0.8}
  ]
}
```

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_tcp_lite_scenario_config_round_trips tests/test_tcp_lite.py::test_tcp_lite_project_scenario_loads -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add carlaair_active_world/scenario.py configs/scenarios/town10hd_vision_tcp_lite.json tests/test_tcp_lite.py
git commit -m "feat: add tcp lite scenario configuration"
```

---

### Task 2: Safety Gate Helpers

**Files:**
- Create: `E:\a2\CarlaAir\carlaair_active_world\vision_models\safety_gate.py`
- Test: `E:\a2\CarlaAir\tests\test_tcp_lite.py`

- [ ] **Step 1: Write failing safety-gate tests**

Append to `tests/test_tcp_lite.py`:

```python
import numpy as np

from carlaair_active_world.vision_models.safety_gate import (
    VisionSafetyGateConfig,
    compute_attack_pattern_score,
    evaluate_vision_safety_gate,
)


def _checkerboard(width: int = 160, height: int = 90) -> np.ndarray:
    yy, xx = np.indices((height, width))
    pattern = ((xx // 4 + yy // 4) % 2).astype(np.uint8) * 255
    return np.stack([pattern, pattern, pattern], axis=2)


def test_safety_gate_blocks_detector_obstacle():
    decision = evaluate_vision_safety_gate(
        rgb=np.zeros((90, 160, 3), dtype=np.uint8),
        detector_diagnostics={"available": True, "obstacle": True, "label": "car"},
        config=VisionSafetyGateConfig(enabled=True),
    )

    assert decision["blocked"] is True
    assert decision["reason"] == "vision_obstacle"
    assert decision["detector_obstacle"] is True


def test_safety_gate_does_not_block_when_disabled():
    decision = evaluate_vision_safety_gate(
        rgb=np.zeros((90, 160, 3), dtype=np.uint8),
        detector_diagnostics={"available": True, "obstacle": True},
        config=VisionSafetyGateConfig(enabled=False),
    )

    assert decision["blocked"] is False
    assert decision["reason"] == "disabled"


def test_attack_pattern_score_is_higher_for_repeated_high_contrast_texture():
    clean = np.zeros((90, 160, 3), dtype=np.uint8)
    noisy = _checkerboard()

    assert compute_attack_pattern_score(noisy) > compute_attack_pattern_score(clean) + 0.2


def test_attack_pattern_gate_blocks_only_when_enabled():
    rgb = _checkerboard()

    diagnostics_only = evaluate_vision_safety_gate(
        rgb=rgb,
        detector_diagnostics={},
        config=VisionSafetyGateConfig(enabled=True, attack_pattern_gate=False, attack_pattern_threshold=0.2),
    )
    blocking = evaluate_vision_safety_gate(
        rgb=rgb,
        detector_diagnostics={},
        config=VisionSafetyGateConfig(enabled=True, attack_pattern_gate=True, attack_pattern_threshold=0.2),
    )

    assert diagnostics_only["blocked"] is False
    assert diagnostics_only["attack_pattern_score"] > 0.2
    assert blocking["blocked"] is True
    assert blocking["reason"] == "attack_pattern"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_safety_gate_blocks_detector_obstacle tests/test_tcp_lite.py::test_safety_gate_does_not_block_when_disabled tests/test_tcp_lite.py::test_attack_pattern_score_is_higher_for_repeated_high_contrast_texture tests/test_tcp_lite.py::test_attack_pattern_gate_blocks_only_when_enabled -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'carlaair_active_world.vision_models.safety_gate'`.

- [ ] **Step 3: Implement safety gate**

Create `carlaair_active_world/vision_models/safety_gate.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class VisionSafetyGateConfig:
    enabled: bool = True
    attack_pattern_gate: bool = False
    attack_pattern_threshold: float = 0.35


def _valid_rgb(rgb: Optional[np.ndarray]) -> bool:
    return rgb is not None and np.asarray(rgb).ndim == 3 and np.asarray(rgb).shape[0] >= 8 and np.asarray(rgb).shape[1] >= 8


def compute_attack_pattern_score(rgb: Optional[np.ndarray]) -> float:
    if not _valid_rgb(rgb):
        return 0.0
    image = np.asarray(rgb).astype(np.float32)
    gray = image[:, :, :3].mean(axis=2) / 255.0
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_strength = 0.5 * (float(dx.mean()) + float(dy.mean()))
    contrast = float(gray.std())
    return float(max(0.0, min(1.0, edge_strength * 1.8 + contrast * 0.6)))


def evaluate_vision_safety_gate(
    rgb: Optional[np.ndarray],
    detector_diagnostics: Optional[Dict[str, Any]],
    config: Optional[VisionSafetyGateConfig] = None,
) -> Dict[str, Any]:
    cfg = config or VisionSafetyGateConfig()
    detector = dict(detector_diagnostics or {})
    detector_obstacle = bool(detector.get("obstacle", False))
    attack_score = compute_attack_pattern_score(rgb)
    decision: Dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "blocked": False,
        "reason": "clear",
        "detector_obstacle": detector_obstacle,
        "attack_pattern_score": float(attack_score),
        "attack_pattern_gate": bool(cfg.attack_pattern_gate),
        "attack_pattern_threshold": float(cfg.attack_pattern_threshold),
    }
    if not cfg.enabled:
        decision["reason"] = "disabled"
        return decision
    if detector_obstacle:
        decision["blocked"] = True
        decision["reason"] = "vision_obstacle"
        return decision
    if cfg.attack_pattern_gate and attack_score >= float(cfg.attack_pattern_threshold):
        decision["blocked"] = True
        decision["reason"] = "attack_pattern"
        return decision
    return decision
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_safety_gate_blocks_detector_obstacle tests/test_tcp_lite.py::test_safety_gate_does_not_block_when_disabled tests/test_tcp_lite.py::test_attack_pattern_score_is_higher_for_repeated_high_contrast_texture tests/test_tcp_lite.py::test_attack_pattern_gate_blocks_only_when_enabled -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add carlaair_active_world/vision_models/safety_gate.py tests/test_tcp_lite.py
git commit -m "feat: add vision safety gate"
```

---

### Task 3: TCP-Lite Model

**Files:**
- Create: `E:\a2\CarlaAir\carlaair_active_world\vision_models\tcp_lite.py`
- Modify: `E:\a2\CarlaAir\carlaair_active_world\vision_models\__init__.py`
- Test: `E:\a2\CarlaAir\tests\test_tcp_lite.py`

- [ ] **Step 1: Write failing model tests**

Append to `tests/test_tcp_lite.py`:

```python
import pytest


def test_command_to_index_accepts_known_and_unknown_commands():
    from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, command_to_index

    assert command_to_index("lane_follow") == COMMAND_TO_INDEX["lane_follow"]
    assert command_to_index("left") == COMMAND_TO_INDEX["left"]
    assert command_to_index("unrecognized") == COMMAND_TO_INDEX["lane_follow"]


def test_tcp_lite_model_outputs_trajectory_and_control_shapes():
    torch = pytest.importorskip("torch")
    from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, TcpLiteModel

    model = TcpLiteModel(image_channels=3, command_count=len(COMMAND_TO_INDEX), trajectory_points=4)
    rgb = torch.zeros((2, 3, 96, 160), dtype=torch.float32)
    speed = torch.zeros((2, 1), dtype=torch.float32)
    command = torch.zeros((2,), dtype=torch.long)

    output = model(rgb, speed, command)

    assert output["trajectory"].shape == (2, 4, 2)
    assert output["control"].shape == (2, 3)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_command_to_index_accepts_known_and_unknown_commands tests/test_tcp_lite.py::test_tcp_lite_model_outputs_trajectory_and_control_shapes -q
```

Expected: FAIL with `ModuleNotFoundError` for `tcp_lite`. If PyTorch is missing, expected result is SKIPPED and implementation still proceeds because the remote training environment needs this module.

- [ ] **Step 3: Implement TCP-Lite model**

Create `carlaair_active_world/vision_models/tcp_lite.py`:

```python
from __future__ import annotations

from typing import Dict

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None


COMMAND_TO_INDEX = {
    "lane_follow": 0,
    "left": 1,
    "right": 2,
    "straight": 3,
}


def command_to_index(command: str) -> int:
    return int(COMMAND_TO_INDEX.get(str(command), COMMAND_TO_INDEX["lane_follow"]))


if torch is not None:

    class TcpLiteModel(nn.Module):
        def __init__(
            self,
            image_channels: int = 3,
            command_count: int = len(COMMAND_TO_INDEX),
            trajectory_points: int = 4,
        ) -> None:
            super().__init__()
            self.trajectory_points = int(trajectory_points)
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.speed_encoder = nn.Sequential(nn.Linear(1, 16), nn.ReLU(inplace=True))
            self.command_embedding = nn.Embedding(command_count, 16)
            self.fusion = nn.Sequential(nn.Linear(96, 128), nn.ReLU(inplace=True))
            self.trajectory_head = nn.Linear(128, self.trajectory_points * 2)
            self.control_head = nn.Linear(128, 3)

        def forward(self, rgb, speed, command) -> Dict[str, torch.Tensor]:
            visual = self.encoder(rgb).flatten(1)
            speed_feature = self.speed_encoder(speed.float().view(speed.shape[0], 1))
            command_feature = self.command_embedding(command.long().view(command.shape[0]))
            fused = self.fusion(torch.cat([visual, speed_feature, command_feature], dim=1))
            trajectory = self.trajectory_head(fused).view(rgb.shape[0], self.trajectory_points, 2)
            control = self.control_head(fused)
            return {"trajectory": trajectory, "control": control}

else:

    class TcpLiteModel:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TcpLiteModel")
```

Modify `carlaair_active_world/vision_models/__init__.py`:

```python
from .base import VisionPolicy
from .simple_lane import SimpleLaneVisionPolicy

try:
    from .tcp_lite import COMMAND_TO_INDEX, TcpLiteModel, command_to_index
except Exception:
    COMMAND_TO_INDEX = {"lane_follow": 0, "left": 1, "right": 2, "straight": 3}
    TcpLiteModel = None

    def command_to_index(command: str) -> int:
        return int(COMMAND_TO_INDEX.get(str(command), COMMAND_TO_INDEX["lane_follow"]))

__all__ = [
    "VisionPolicy",
    "SimpleLaneVisionPolicy",
    "COMMAND_TO_INDEX",
    "TcpLiteModel",
    "command_to_index",
]
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_command_to_index_accepts_known_and_unknown_commands tests/test_tcp_lite.py::test_tcp_lite_model_outputs_trajectory_and_control_shapes -q
```

Expected: PASS when PyTorch is installed, SKIPPED when PyTorch is absent.

- [ ] **Step 5: Commit**

Run:

```powershell
git add carlaair_active_world/vision_models/tcp_lite.py carlaair_active_world/vision_models/__init__.py tests/test_tcp_lite.py
git commit -m "feat: add tcp lite model"
```

---

### Task 4: TCP-Lite Dataset And Training Script

**Files:**
- Create: `E:\a2\CarlaAir\carlaair_active_world\vision_models\tcp_lite_dataset.py`
- Create: `E:\a2\CarlaAir\scripts\train_tcp_lite.py`
- Test: `E:\a2\CarlaAir\tests\test_tcp_lite.py`

- [ ] **Step 1: Write failing dataset and training smoke tests**

Append to `tests/test_tcp_lite.py`:

```python
import json
from pathlib import Path


def _write_tiny_tcp_lite_dataset(root: Path) -> None:
    PIL_Image = pytest.importorskip("PIL.Image")
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    samples = []
    for idx in range(2):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[:, :, 0] = 40 + idx * 20
        image_path = image_dir / f"{idx:06d}.png"
        PIL_Image.fromarray(image).save(image_path)
        samples.append(
            {
                "rgb": f"images/{idx:06d}.png",
                "speed_mps": float(idx),
                "command": "lane_follow",
                "trajectory": [[1.0, 0.0], [2.0, 0.1], [3.0, 0.1], [4.0, 0.0]],
                "control": {"steer": 0.0, "throttle": 0.2, "brake": 0.0},
            }
        )
    with (root / "samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def test_tcp_lite_dataset_reads_jsonl_samples(tmp_path):
    pytest.importorskip("torch")
    _write_tiny_tcp_lite_dataset(tmp_path)

    from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset

    dataset = TcpLiteImitationDataset(tmp_path, image_size=(32, 48), trajectory_points=4)
    item = dataset[0]

    assert len(dataset) == 2
    assert item["rgb"].shape == (3, 32, 48)
    assert item["speed"].shape == (1,)
    assert item["trajectory"].shape == (4, 2)
    assert item["control"].shape == (3,)


def test_train_tcp_lite_saves_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    _write_tiny_tcp_lite_dataset(tmp_path)
    from scripts.train_tcp_lite import train_tcp_lite

    output = tmp_path / "tcp_lite.pt"

    train_tcp_lite(
        dataset=tmp_path,
        output=output,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        image_height=32,
        image_width=48,
        trajectory_points=4,
    )

    assert output.exists()
    checkpoint = torch.load(output, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert checkpoint["image_size"] == [32, 48]
    assert checkpoint["trajectory_points"] == 4
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_tcp_lite_dataset_reads_jsonl_samples tests/test_tcp_lite.py::test_train_tcp_lite_saves_checkpoint -q
```

Expected: FAIL with `ModuleNotFoundError` for `tcp_lite_dataset` or `ImportError` for `train_tcp_lite`. If PyTorch or Pillow is missing, expected result is SKIPPED.

- [ ] **Step 3: Implement dataset loader**

Create `carlaair_active_world/vision_models/tcp_lite_dataset.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .tcp_lite import command_to_index


class TcpLiteImitationDataset(Dataset):
    def __init__(self, root: str | Path, image_size: tuple[int, int] = (96, 160), trajectory_points: int = 4) -> None:
        self.root = Path(root)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.trajectory_points = int(trajectory_points)
        samples_path = self.root / "samples.jsonl"
        with samples_path.open("r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = [json.loads(line) for line in f if line.strip()]
        if not self.samples:
            raise ValueError(f"No samples found in {samples_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, relative_path: str) -> torch.Tensor:
        image = Image.open(self.root / relative_path).convert("RGB")
        height, width = self.image_size
        image = image.resize((width, height))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def _trajectory(self, values: Sequence[Sequence[float]]) -> torch.Tensor:
        arr = np.zeros((self.trajectory_points, 2), dtype=np.float32)
        source = np.asarray(values, dtype=np.float32).reshape(-1, 2)
        count = min(self.trajectory_points, source.shape[0])
        arr[:count, :] = source[:count, :]
        return torch.from_numpy(arr)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[int(index)]
        control = sample.get("control", {})
        return {
            "rgb": self._load_rgb(str(sample["rgb"])),
            "speed": torch.tensor([float(sample.get("speed_mps", 0.0))], dtype=torch.float32),
            "command": torch.tensor(command_to_index(str(sample.get("command", "lane_follow"))), dtype=torch.long),
            "trajectory": self._trajectory(sample.get("trajectory", [])),
            "control": torch.tensor(
                [
                    float(control.get("steer", 0.0)),
                    float(control.get("throttle", 0.0)),
                    float(control.get("brake", 0.0)),
                ],
                dtype=torch.float32,
            ),
        }
```

- [ ] **Step 4: Implement training script**

Create `scripts/train_tcp_lite.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, TcpLiteModel
from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset


def train_tcp_lite(
    dataset: Path,
    output: Path,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    image_height: int = 96,
    image_width: int = 160,
    trajectory_points: int = 4,
) -> Path:
    torch_device = torch.device(device)
    train_dataset = TcpLiteImitationDataset(
        dataset,
        image_size=(int(image_height), int(image_width)),
        trajectory_points=int(trajectory_points),
    )
    loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    model = TcpLiteModel(command_count=len(COMMAND_TO_INDEX), trajectory_points=int(trajectory_points)).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _epoch in range(int(epochs)):
        for batch in loader:
            rgb = batch["rgb"].to(torch_device)
            speed = batch["speed"].to(torch_device)
            command = batch["command"].to(torch_device)
            trajectory = batch["trajectory"].to(torch_device)
            control = batch["control"].to(torch_device)
            output_dict = model(rgb, speed, command)
            loss = loss_fn(output_dict["trajectory"], trajectory) + loss_fn(output_dict["control"], control)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": [int(image_height), int(image_width)],
            "trajectory_points": int(trajectory_points),
            "commands": dict(COMMAND_TO_INDEX),
        },
        output,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCP-Lite imitation model from JSONL samples.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--trajectory-points", type=int, default=4)
    args = parser.parse_args()
    saved = train_tcp_lite(
        dataset=args.dataset,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        image_height=args.image_height,
        image_width=args.image_width,
        trajectory_points=args.trajectory_points,
    )
    print(f"Saved checkpoint: {saved}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_tcp_lite_dataset_reads_jsonl_samples tests/test_tcp_lite.py::test_train_tcp_lite_saves_checkpoint -q
```

Expected: PASS when PyTorch and Pillow are installed, SKIPPED when a dependency is absent.

- [ ] **Step 6: Commit**

Run:

```powershell
git add carlaair_active_world/vision_models/tcp_lite_dataset.py scripts/train_tcp_lite.py tests/test_tcp_lite.py
git commit -m "feat: add tcp lite training pipeline"
```

---

### Task 5: TCP-Lite Vision Policy

**Files:**
- Create: `E:\a2\CarlaAir\carlaair_active_world\vision_models\tcp_lite_policy.py`
- Modify: `E:\a2\CarlaAir\carlaair_active_world\vision_models\__init__.py`
- Test: `E:\a2\CarlaAir\tests\test_tcp_lite.py`

- [ ] **Step 1: Write failing policy tests**

Append to `tests/test_tcp_lite.py`:

```python
from carlaair_active_world.vision_models.tcp_lite_policy import TcpLiteVisionPolicy


class _MockTcpModel:
    def predict(self, rgb, speed_mps, command):
        return {
            "trajectory": [[1.0, 0.0], [2.0, 0.1], [3.0, 0.1], [4.0, 0.0]],
            "control": [1.5, 0.4, -0.2],
        }


def test_tcp_lite_policy_brakes_without_model_path():
    policy = TcpLiteVisionPolicy(model_path="")

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 1.0})

    assert control.brake == 1.0
    assert control.throttle == 0.0
    assert policy.last_diagnostics["model_ready"] is False
    assert policy.last_diagnostics["reason"] == "missing_model_path"


def test_tcp_lite_policy_uses_mock_model_and_clamps_control():
    policy = TcpLiteVisionPolicy(model=_MockTcpModel(), navigation_command="lane_follow")

    control = policy.predict({"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "speed_mps": 2.0})

    assert control.steer == 1.0
    assert control.throttle == 0.4
    assert control.brake == 0.0
    assert policy.last_diagnostics["model_ready"] is True
    assert policy.last_diagnostics["command"] == "lane_follow"
    assert policy.last_diagnostics["trajectory"][0] == [1.0, 0.0]


def test_tcp_lite_policy_safety_gate_brakes_for_obstacle():
    policy = TcpLiteVisionPolicy(model=_MockTcpModel(), safety_gate_enabled=True)

    control = policy.predict(
        {
            "rgb": np.zeros((90, 160, 3), dtype=np.uint8),
            "speed_mps": 2.0,
            "vision_detector": {"available": True, "obstacle": True, "label": "car"},
        }
    )

    assert control.brake == 1.0
    assert control.throttle == 0.0
    assert policy.last_diagnostics["safety_gate"]["blocked"] is True
    assert policy.last_diagnostics["reason"] == "vision_obstacle"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_tcp_lite_policy_brakes_without_model_path tests/test_tcp_lite.py::test_tcp_lite_policy_uses_mock_model_and_clamps_control tests/test_tcp_lite.py::test_tcp_lite_policy_safety_gate_brakes_for_obstacle -q
```

Expected: FAIL with `ModuleNotFoundError` for `tcp_lite_policy`.

- [ ] **Step 3: Implement TCP-Lite policy**

Create `carlaair_active_world/vision_models/tcp_lite_policy.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import carla
import numpy as np

from .base import VisionPolicy
from .safety_gate import VisionSafetyGateConfig, evaluate_vision_safety_gate


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


class TcpLiteVisionPolicy(VisionPolicy):
    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        navigation_command: str = "lane_follow",
        safety_gate_enabled: bool = True,
        attack_pattern_gate: bool = False,
        model: Optional[Any] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.navigation_command = str(navigation_command)
        self.safety_gate_config = VisionSafetyGateConfig(
            enabled=bool(safety_gate_enabled),
            attack_pattern_gate=bool(attack_pattern_gate),
        )
        self.model = model
        self.model_ready = model is not None
        self.last_diagnostics: Dict[str, Any] = {}
        if self.model is None and self.model_path:
            self._load_torch_checkpoint(self.model_path)

    def _brake(self, reason: str, safety_gate: Optional[Dict[str, Any]] = None) -> carla.VehicleControl:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        self.last_diagnostics = {
            "model_ready": bool(self.model_ready),
            "command": self.navigation_command,
            "reason": reason,
            "safety_gate": dict(safety_gate or {}),
            "trajectory": [],
            "raw_control": [],
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control

    def _load_torch_checkpoint(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            self.model_ready = False
            self.last_diagnostics = {"model_ready": False, "reason": "missing_model_path", "model_path": str(path)}
            return
        try:
            import torch

            from .tcp_lite import COMMAND_TO_INDEX, TcpLiteModel

            checkpoint = torch.load(path, map_location=self.device)
            trajectory_points = int(checkpoint.get("trajectory_points", 4))
            model = TcpLiteModel(command_count=len(COMMAND_TO_INDEX), trajectory_points=trajectory_points)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(torch.device(self.device))
            model.eval()
            self.model = model
            self.model_ready = True
        except Exception as exc:
            self.model = None
            self.model_ready = False
            self.last_diagnostics = {
                "model_ready": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "model_path": str(path),
            }

    def _predict_model(self, rgb: np.ndarray, speed_mps: float, command: str) -> Dict[str, Any]:
        if hasattr(self.model, "predict"):
            return dict(self.model.predict(rgb=rgb, speed_mps=float(speed_mps), command=command))
        import torch
        from PIL import Image

        from .tcp_lite import command_to_index

        image = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGB").resize((160, 96))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(torch.device(self.device))
        speed = torch.tensor([[float(speed_mps)]], dtype=torch.float32, device=torch.device(self.device))
        command_tensor = torch.tensor([command_to_index(command)], dtype=torch.long, device=torch.device(self.device))
        with torch.no_grad():
            output = self.model(tensor, speed, command_tensor)
        return {
            "trajectory": output["trajectory"].detach().cpu().numpy()[0].tolist(),
            "control": output["control"].detach().cpu().numpy()[0].tolist(),
        }

    def predict(self, obs: Dict[str, Any]) -> carla.VehicleControl:
        rgb = obs.get("rgb")
        if rgb is None:
            return self._brake("missing_rgb")
        if not self.model_ready or self.model is None:
            return self._brake("missing_model_path")
        safety_gate = evaluate_vision_safety_gate(
            np.asarray(rgb),
            dict(obs.get("vision_detector", {}) or {}),
            config=self.safety_gate_config,
        )
        if bool(safety_gate.get("blocked", False)):
            return self._brake(str(safety_gate.get("reason", "safety_gate")), safety_gate=safety_gate)
        speed_mps = float(obs.get("speed_mps", 0.0))
        command = str(obs.get("navigation_command", self.navigation_command))
        try:
            output = self._predict_model(np.asarray(rgb), speed_mps, command)
        except Exception as exc:
            return self._brake(f"{type(exc).__name__}: {exc}", safety_gate=safety_gate)
        raw_control = list(output.get("control", [0.0, 0.0, 1.0]))
        while len(raw_control) < 3:
            raw_control.append(0.0)
        control = carla.VehicleControl()
        control.steer = _clamp(float(raw_control[0]), -1.0, 1.0)
        control.throttle = _clamp(float(raw_control[1]), 0.0, 1.0)
        control.brake = _clamp(float(raw_control[2]), 0.0, 1.0)
        self.last_diagnostics = {
            "model_ready": True,
            "command": command,
            "reason": "ok",
            "safety_gate": safety_gate,
            "trajectory": output.get("trajectory", []),
            "raw_control": [float(v) for v in raw_control[:3]],
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control
```

- [ ] **Step 4: Export policy**

Modify `carlaair_active_world/vision_models/__init__.py` to include:

```python
from .tcp_lite_policy import TcpLiteVisionPolicy
```

and add `"TcpLiteVisionPolicy"` to `__all__`.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_tcp_lite_policy_brakes_without_model_path tests/test_tcp_lite.py::test_tcp_lite_policy_uses_mock_model_and_clamps_control tests/test_tcp_lite.py::test_tcp_lite_policy_safety_gate_brakes_for_obstacle -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add carlaair_active_world/vision_models/tcp_lite_policy.py carlaair_active_world/vision_models/__init__.py tests/test_tcp_lite.py
git commit -m "feat: add tcp lite vision policy"
```

---

### Task 6: Driver And App Integration

**Files:**
- Modify: `E:\a2\CarlaAir\carlaair_active_world\vision_driver.py`
- Modify: `E:\a2\CarlaAir\carlaair_active_world\env.py`
- Modify: `E:\a2\CarlaAir\carlaair_active_world\task_app.py`
- Test: `E:\a2\CarlaAir\tests\test_tcp_lite.py`

- [ ] **Step 1: Write failing integration tests**

Append to `tests/test_tcp_lite.py`:

```python
from carlaair_active_world.vision_driver import VisionEgoDriver


def test_vision_driver_forwards_navigation_command_to_policy():
    class _Rig:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def spawn(self) -> None:
            pass

        def snapshot(self):
            return {"rgb": np.zeros((90, 160, 3), dtype=np.uint8), "semantic": None, "depth": None}

        def destroy(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.obs = None
            self.last_diagnostics = {}

        def predict(self, obs):
            self.obs = obs
            import carla

            return carla.VehicleControl()

    class _Vehicle:
        def get_velocity(self):
            return type("Velocity", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    policy = _Policy()
    original = VisionEgoDriver.sensor_rig_class
    VisionEgoDriver.sensor_rig_class = _Rig
    try:
        driver = VisionEgoDriver(object(), _Vehicle(), policy=policy, navigation_command="left")
        driver.predict()
    finally:
        VisionEgoDriver.sensor_rig_class = original

    assert policy.obs["navigation_command"] == "left"


def test_env_uses_tcp_lite_policy_for_tcp_lite_mode():
    from carlaair_active_world import env as env_module

    created = {}

    class _Policy:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

    class _Driver:
        def __init__(self, *args, **kwargs) -> None:
            created["driver_policy"] = kwargs.get("policy")
            created["driver_navigation_command"] = kwargs.get("navigation_command")

    class _Vehicle:
        def set_autopilot(self, value):
            created["autopilot"] = value

    original_policy = env_module.TcpLiteVisionPolicy
    original_driver = env_module.VisionEgoDriver
    try:
        env_module.TcpLiteVisionPolicy = _Policy
        env_module.VisionEgoDriver = _Driver
        scenario = ScenarioConfig.from_dict(
            {
                "name": "tcp_lite_env",
                "ego_control_mode": "vision_tcp_lite",
                "vision_model_path": "checkpoints/tcp.pt",
                "vision_model_device": "cpu",
                "vision_navigation_command": "right",
            }
        )
        app = env_module.ActiveAirGroundEnv(scenario)
        app.world = object()
        app.ego_vehicle = _Vehicle()
        app._start_ego_control()
    finally:
        env_module.TcpLiteVisionPolicy = original_policy
        env_module.VisionEgoDriver = original_driver

    assert created["autopilot"] is False
    assert created["model_path"] == "checkpoints/tcp.pt"
    assert created["navigation_command"] == "right"
    assert created["driver_policy"] is not None
    assert created["driver_navigation_command"] == "right"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_vision_driver_forwards_navigation_command_to_policy tests/test_tcp_lite.py::test_env_uses_tcp_lite_policy_for_tcp_lite_mode -q
```

Expected: FAIL because `VisionEgoDriver` does not accept `navigation_command`, and `env.py` does not import or instantiate `TcpLiteVisionPolicy`.

- [ ] **Step 3: Update `VisionEgoDriver` observations**

In `carlaair_active_world/vision_driver.py`, add a constructor parameter after `vision_detector_confidence`:

```python
        navigation_command: str = "lane_follow",
```

Store it:

```python
        self.navigation_command = str(navigation_command)
```

Add it to `obs` in `predict`:

```python
            "navigation_command": self.navigation_command,
```

- [ ] **Step 4: Update `env.py` control-mode routing**

Modify imports:

```python
from .vision_models import TcpLiteVisionPolicy
```

Extend the control mode set:

```python
        if mode in {"route_follow", "route", "behavior", "vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
```

Replace the vision driver branch with:

```python
            if mode in {"vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
                policy = None
                if mode == "vision_tcp_lite":
                    policy = TcpLiteVisionPolicy(
                        model_path=str(getattr(self.scenario, "vision_model_path", "")),
                        device=str(getattr(self.scenario, "vision_model_device", "cpu")),
                        navigation_command=str(getattr(self.scenario, "vision_navigation_command", "lane_follow")),
                        safety_gate_enabled=bool(getattr(self.scenario, "vision_safety_gate_enabled", True)),
                        attack_pattern_gate=bool(getattr(self.scenario, "vision_attack_pattern_gate", False)),
                    )
                self.ego_driver = VisionEgoDriver(
                    self.world,
                    self.ego_vehicle,
                    target_speed_mps=float(getattr(self.scenario, "ego_target_speed_mps", 4.0)),
                    policy=policy,
                    use_semantic=mode == "vision_simple",
                    vision_attack=str(getattr(self.scenario, "vision_attack", "none")),
                    vision_attack_intensity=float(getattr(self.scenario, "vision_attack_intensity", 1.0)),
                    vision_detector_model_path=str(getattr(self.scenario, "vision_detector_model_path", "")),
                    vision_detector_confidence=float(getattr(self.scenario, "vision_detector_confidence", 0.35)),
                    navigation_command=str(getattr(self.scenario, "vision_navigation_command", "lane_follow")),
                )
```

- [ ] **Step 5: Update `task_app.py` control-mode routing**

Modify imports:

```python
from .vision_models import TcpLiteVisionPolicy
```

Extend both relevant control-mode sets:

```python
        if self._ego_control_mode in {"route_follow", "route", "behavior", "vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
```

and:

```python
            if self._ego_control_mode in {"vision_simple", "vision_rgb_only", "vision_tcp_lite"}:
```

Replace the `VisionEgoDriver` construction inside that branch with:

```python
                policy = None
                if self._ego_control_mode == "vision_tcp_lite":
                    policy = TcpLiteVisionPolicy(
                        model_path=str(self.scenario.vision_model_path),
                        device=str(self.scenario.vision_model_device),
                        navigation_command=str(self.scenario.vision_navigation_command),
                        safety_gate_enabled=bool(self.scenario.vision_safety_gate_enabled),
                        attack_pattern_gate=bool(self.scenario.vision_attack_pattern_gate),
                    )
                self.ego_driver = VisionEgoDriver(
                    self.world,
                    self.ego_vehicle,
                    target_speed_mps=float(self.scenario.ego_target_speed_mps),
                    policy=policy,
                    use_semantic=self._ego_control_mode == "vision_simple",
                    vision_attack=str(getattr(self.scenario, "vision_attack", "none")),
                    vision_attack_intensity=float(getattr(self.scenario, "vision_attack_intensity", 1.0)),
                    vision_detector_model_path=str(getattr(self.scenario, "vision_detector_model_path", "")),
                    vision_detector_confidence=float(getattr(self.scenario, "vision_detector_confidence", 0.35)),
                    navigation_command=str(self.scenario.vision_navigation_command),
                )
```

- [ ] **Step 6: Verify GREEN**

Run:

```powershell
pytest tests/test_tcp_lite.py::test_vision_driver_forwards_navigation_command_to_policy tests/test_tcp_lite.py::test_env_uses_tcp_lite_policy_for_tcp_lite_mode -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add carlaair_active_world/vision_driver.py carlaair_active_world/env.py carlaair_active_world/task_app.py tests/test_tcp_lite.py
git commit -m "feat: wire tcp lite vision control mode"
```

---

### Task 7: Full Local Verification

**Files:**
- Modify only files needed for fixes discovered by verification.

- [ ] **Step 1: Run full pytest**

Run:

```powershell
pytest -q
```

Expected: all non-skipped tests PASS. PyTorch/Pillow-dependent tests may SKIP only when those optional dependencies are unavailable.

- [ ] **Step 2: Run training CLI smoke when dependencies are present**

Run this only if `pytest tests/test_tcp_lite.py::test_train_tcp_lite_saves_checkpoint -q` passed rather than skipped:

```powershell
$tmp = Join-Path $env:TEMP "carlaair_tcp_lite_smoke"
Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:CARLAAIR_TCP_LITE_SMOKE_DIR = $tmp
@'
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(os.environ["CARLAAIR_TCP_LITE_SMOKE_DIR"])
image_dir = root / "images"
image_dir.mkdir(parents=True, exist_ok=True)
samples = []
for idx in range(2):
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :, 0] = 40 + idx * 20
    image_path = image_dir / f"{idx:06d}.png"
    Image.fromarray(image).save(image_path)
    samples.append(
        {
            "rgb": f"images/{idx:06d}.png",
            "speed_mps": float(idx),
            "command": "lane_follow",
            "trajectory": [[1.0, 0.0], [2.0, 0.1], [3.0, 0.1], [4.0, 0.0]],
            "control": {"steer": 0.0, "throttle": 0.2, "brake": 0.0},
        }
    )
with (root / "samples.jsonl").open("w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
'@ | python -
python scripts/train_tcp_lite.py --dataset "$tmp" --output "$tmp\tcp_lite.pt" --epochs 1 --batch-size 2 --device cpu --image-height 32 --image-width 48 --trajectory-points 4
```

Expected: prints a line beginning with `Saved checkpoint:` and writes `tcp_lite.pt` under `%TEMP%\carlaair_tcp_lite_smoke`. Do not commit files under `%TEMP%`.

- [ ] **Step 3: Check git status**

Run:

```powershell
git status --short
```

Expected: no untracked training artifacts, no recordings, no checkpoints. Only intentional source/test/config files should appear before the final commit.

- [ ] **Step 4: Final commit for verification fixes**

If verification required small fixes after Task 6, commit them:

```powershell
git add carlaair_active_world tests scripts configs
git commit -m "test: verify tcp lite local closure"
```

If no fixes were needed after Task 6, do not create an empty commit.

---

## Self-Review

- Spec coverage: Tasks cover scenario config, `vision_tcp_lite` mode, TCP-Lite model, policy, dataset, train script, safety gate, YOLO obstacle gate behavior through detector diagnostics, attack-pattern diagnostics, and local tests.
- Scope control: No task adds UAV BEV fusion, remote sync, GitHub push, model weight download, or CARLA 60-second result claims.
- Type consistency: The plan uses `vision_model_path`, `vision_model_device`, `vision_navigation_command`, `vision_safety_gate_enabled`, and `vision_attack_pattern_gate` consistently across scenario, env, task app, and tests.
- Optional dependencies: PyTorch/Pillow tests are written to skip when unavailable; non-torch policy tests use a mock model and remain runnable in the local stubbed test environment.
