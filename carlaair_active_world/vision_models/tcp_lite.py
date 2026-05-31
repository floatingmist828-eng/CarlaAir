from __future__ import annotations

from typing import Any, Dict

COMMAND_TO_INDEX = {
    "lane_follow": 0,
    "left": 1,
    "right": 2,
    "straight": 3,
}


def command_to_index(command: str) -> int:
    return COMMAND_TO_INDEX.get(str(command), COMMAND_TO_INDEX["lane_follow"])


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in environments without torch.
    torch = None
    nn = None


if nn is not None:

    class TcpLiteModel(nn.Module):
        def __init__(
            self,
            image_channels: int = 3,
            command_count: int = len(COMMAND_TO_INDEX),
            trajectory_points: int = 4,
        ) -> None:
            super().__init__()
            self.trajectory_points = int(trajectory_points)
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.speed_encoder = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(inplace=True),
            )
            self.command_embedding = nn.Embedding(command_count, 16)
            self.fusion = nn.Sequential(
                nn.Linear(96, 64),
                nn.ReLU(inplace=True),
            )
            self.trajectory_head = nn.Linear(64, self.trajectory_points * 2)
            self.control_head = nn.Linear(64, 3)

        def forward(self, rgb: torch.Tensor, speed: torch.Tensor, command: torch.Tensor) -> Dict[str, torch.Tensor]:
            image_features = self.image_encoder(rgb)
            speed_features = self.speed_encoder(speed)
            command_features = self.command_embedding(command.long())
            fused = self.fusion(torch.cat([image_features, speed_features, command_features], dim=1))
            trajectory = self.trajectory_head(fused).view(rgb.shape[0], self.trajectory_points, 2)
            control = self.control_head(fused)
            return {"trajectory": trajectory, "control": control}

else:

    class TcpLiteModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("TcpLiteModel requires torch to be installed.")
