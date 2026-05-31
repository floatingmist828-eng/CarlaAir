from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, TcpLiteModel
from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset


def train_tcp_lite(
    dataset_root: str | Path,
    output_path: str | Path,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    image_height: int = 96,
    image_width: int = 160,
    trajectory_points: int = 4,
) -> Path:
    output_path = Path(output_path)
    dataset = TcpLiteImitationDataset(
        dataset_root,
        image_size=(image_height, image_width),
        trajectory_points=trajectory_points,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    torch_device = torch.device(device)

    model = TcpLiteModel(
        command_count=len(COMMAND_TO_INDEX),
        trajectory_points=trajectory_points,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        for batch in loader:
            rgb = batch["rgb"].to(torch_device)
            speed = batch["speed"].to(torch_device)
            command = batch["command"].to(torch_device)
            target_trajectory = batch["trajectory"].to(torch_device)
            target_control = batch["control"].to(torch_device)

            optimizer.zero_grad()
            outputs = model(rgb, speed, command)
            loss = criterion(outputs["trajectory"], target_trajectory) + criterion(
                outputs["control"],
                target_control,
            )
            loss.backward()
            optimizer.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": [int(image_height), int(image_width)],
            "trajectory_points": int(trajectory_points),
            "commands": COMMAND_TO_INDEX,
        },
        output_path,
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TCP-Lite imitation model.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-height", default=96, type=int)
    parser.add_argument("--image-width", default=160, type=int)
    parser.add_argument("--trajectory-points", default=4, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_tcp_lite(
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        image_height=args.image_height,
        image_width=args.image_width,
        trajectory_points=args.trajectory_points,
    )


if __name__ == "__main__":
    main()
