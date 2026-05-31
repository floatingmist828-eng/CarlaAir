from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from carlaair_active_world.scenario import ScenarioConfig
from carlaair_active_world.vision_models.safety_gate import (
    VisionSafetyGateConfig,
    compute_attack_pattern_score,
    evaluate_vision_safety_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def _checkerboard(width=160, height=90):
    y, x = np.indices((height, width))
    board = ((x // 4 + y // 4) % 2 * 255).astype(np.uint8)
    return np.stack([board, board, board], axis=-1)


def _write_tiny_tcp_lite_dataset(root: Path):
    PIL_Image = pytest.importorskip("PIL.Image")
    image_dir = root / "images"
    image_dir.mkdir()
    for index in range(2):
        rgb = np.full((32, 48, 3), index * 64, dtype=np.uint8)
        PIL_Image.fromarray(rgb).save(image_dir / f"{index:06d}.png")

    samples = [
        {
            "rgb": f"images/{index:06d}.png",
            "speed_mps": float(index),
            "command": "lane_follow",
            "trajectory": [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            "control": {"steer": 0.1, "throttle": 0.2, "brake": 0.0},
        }
        for index in range(2)
    ]
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def _write_single_tcp_lite_sample(root: Path, sample: dict):
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(sample) + "\n")


def test_train_tcp_lite_module_imports_without_torch_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.train_tcp_lite import train_tcp_lite; print(callable(train_tcp_lite))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_train_tcp_lite_help_works_without_torch_dependency():
    result = subprocess.run(
        [sys.executable, "scripts/train_tcp_lite.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Train TCP-Lite" in result.stdout or "--dataset" in result.stdout


def test_tcp_lite_helpers_import_without_carla_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from carlaair_active_world.vision_models.tcp_lite import command_to_index; "
            'print(command_to_index("left"))',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


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


def test_safety_gate_blocks_detector_obstacle():
    result = evaluate_vision_safety_gate(
        np.zeros((90, 160, 3), dtype=np.uint8),
        {"obstacle": True, "label": "car"},
        VisionSafetyGateConfig(enabled=True),
    )

    assert result["blocked"] is True
    assert result["reason"] == "vision_obstacle"
    assert result["detector_obstacle"] is True


def test_safety_gate_does_not_block_when_disabled():
    result = evaluate_vision_safety_gate(
        np.zeros((90, 160, 3), dtype=np.uint8),
        {"obstacle": True},
        VisionSafetyGateConfig(enabled=False),
    )

    assert result["blocked"] is False
    assert result["reason"] == "disabled"


def test_attack_pattern_score_is_higher_for_repeated_high_contrast_texture():
    clean_score = compute_attack_pattern_score(np.zeros((90, 160, 3), dtype=np.uint8))
    noisy_score = compute_attack_pattern_score(_checkerboard())

    assert noisy_score > clean_score + 0.2


def test_attack_pattern_score_returns_zero_for_invalid_rgb():
    assert compute_attack_pattern_score([[[1], [2, 3]]]) == 0.0


def test_attack_pattern_gate_blocks_only_when_enabled():
    rgb = _checkerboard()
    disabled_gate = evaluate_vision_safety_gate(
        rgb,
        {},
        VisionSafetyGateConfig(attack_pattern_gate=False, attack_pattern_threshold=0.2),
    )
    enabled_gate = evaluate_vision_safety_gate(
        rgb,
        {},
        VisionSafetyGateConfig(attack_pattern_gate=True, attack_pattern_threshold=0.2),
    )

    assert disabled_gate["blocked"] is False
    assert disabled_gate["attack_pattern_score"] > 0.2
    assert enabled_gate["blocked"] is True
    assert enabled_gate["reason"] == "attack_pattern"


def test_command_to_index_accepts_known_and_unknown_commands():
    from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, command_to_index

    assert command_to_index("lane_follow") == COMMAND_TO_INDEX["lane_follow"]
    assert command_to_index("left") == COMMAND_TO_INDEX["left"]
    assert command_to_index("unknown") == COMMAND_TO_INDEX["lane_follow"]


def test_tcp_lite_model_outputs_trajectory_and_control_shapes():
    torch = pytest.importorskip("torch")
    from carlaair_active_world.vision_models.tcp_lite import COMMAND_TO_INDEX, TcpLiteModel

    model = TcpLiteModel(
        image_channels=3,
        command_count=len(COMMAND_TO_INDEX),
        trajectory_points=4,
    )
    rgb = torch.zeros((2, 3, 96, 160))
    speed = torch.zeros((2, 1))
    command = torch.zeros((2,), dtype=torch.long)

    output = model(rgb, speed, command)

    assert output["trajectory"].shape == (2, 4, 2)
    assert output["control"].shape == (2, 3)


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


def test_tcp_lite_dataset_rejects_rgb_path_outside_root(tmp_path):
    pytest.importorskip("torch")
    PIL_Image = pytest.importorskip("PIL.Image")
    outside_path = tmp_path.parent / "outside.png"
    PIL_Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(outside_path)
    _write_single_tcp_lite_sample(
        tmp_path,
        {
            "rgb": "../outside.png",
            "speed_mps": 0.0,
            "command": "lane_follow",
            "trajectory": [[0.0, 0.0]],
            "control": {"steer": 0.0, "throttle": 0.0, "brake": 0.0},
        },
    )
    from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset

    dataset = TcpLiteImitationDataset(tmp_path)

    with pytest.raises(ValueError, match="outside dataset root"):
        dataset[0]


@pytest.mark.parametrize(
    "trajectory",
    [
        [1.0, 2.0, 3.0],
        [[0.0, 0.0], [1.0]],
        [[0.0, "bad"]],
    ],
)
def test_tcp_lite_dataset_rejects_malformed_trajectory(tmp_path, trajectory):
    pytest.importorskip("torch")
    PIL_Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "image.png"
    PIL_Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
    _write_single_tcp_lite_sample(
        tmp_path,
        {
            "rgb": "image.png",
            "speed_mps": 0.0,
            "command": "lane_follow",
            "trajectory": trajectory,
            "control": {"steer": 0.0, "throttle": 0.0, "brake": 0.0},
        },
    )
    from carlaair_active_world.vision_models.tcp_lite_dataset import TcpLiteImitationDataset

    dataset = TcpLiteImitationDataset(tmp_path)

    with pytest.raises(ValueError, match="trajectory must be a sequence of \\[x, y\\] points"):
        dataset[0]


def test_train_tcp_lite_saves_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    _write_tiny_tcp_lite_dataset(tmp_path)
    from scripts.train_tcp_lite import train_tcp_lite

    output_path = tmp_path / "tcp_lite.pt"
    train_tcp_lite(
        dataset_root=tmp_path,
        output_path=output_path,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        image_height=32,
        image_width=48,
        trajectory_points=4,
    )

    checkpoint = torch.load(output_path, map_location="cpu")
    assert output_path.exists()
    assert "model_state_dict" in checkpoint
    assert checkpoint["image_size"] == [32, 48]
    assert checkpoint["trajectory_points"] == 4
