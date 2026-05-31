from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from carlaair_active_world.vision_models.tcp_lite import command_to_index


class TcpLiteImitationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (96, 160),
        trajectory_points: int = 4,
    ) -> None:
        self.root = Path(root)
        self.root_resolved = self.root.resolve()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.trajectory_points = int(trajectory_points)
        self.samples = []

        samples_path = self.root / "samples.jsonl"
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if not self.samples:
            raise ValueError(f"No TCP-Lite samples found in {samples_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, rgb_path: str) -> np.ndarray:
        candidate = (self.root / rgb_path).resolve()
        try:
            candidate.relative_to(self.root_resolved)
        except ValueError as exc:
            raise ValueError(f"RGB path is outside dataset root: {rgb_path}") from exc

        height, width = self.image_size
        with Image.open(candidate) as image:
            image = image.convert("RGB").resize((width, height))
            return np.asarray(image, dtype=np.float32) / 255.0

    def _trajectory(self, sample: dict) -> np.ndarray:
        trajectory = np.zeros((self.trajectory_points, 2), dtype=np.float32)
        points = np.asarray(sample.get("trajectory", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("trajectory must be a sequence of [x, y] points")

        point_count = min(len(points), self.trajectory_points)
        if point_count:
            trajectory[:point_count] = points[:point_count]
        return trajectory

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        rgb = self._load_rgb(sample["rgb"])
        trajectory = self._trajectory(sample)

        control = sample["control"]
        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1),
            "speed": torch.tensor([float(sample.get("speed_mps", 0.0))], dtype=torch.float32),
            "command": torch.tensor(command_to_index(sample.get("command", "lane_follow")), dtype=torch.long),
            "trajectory": torch.from_numpy(trajectory),
            "control": torch.tensor(
                [
                    float(control.get("steer", 0.0)),
                    float(control.get("throttle", 0.0)),
                    float(control.get("brake", 0.0)),
                ],
                dtype=torch.float32,
            ),
        }
