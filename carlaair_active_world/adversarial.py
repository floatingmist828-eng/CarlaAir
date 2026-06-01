from __future__ import annotations

from typing import Dict, Optional

import carla
import numpy as np


def build_weather_parameters(preset: str) -> Optional[carla.WeatherParameters]:
    name = str(preset or "none").lower()
    if name in {"none", "off", "default"}:
        return None
    if name in {"clear", "clear_noon"}:
        return getattr(carla.WeatherParameters, "ClearNoon", None)
    if name in {"hard_rain_fog", "rain_fog", "storm_fog"}:
        return carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=95.0,
            precipitation_deposits=100.0,
            wind_intensity=70.0,
            fog_density=85.0,
            fog_distance=8.0,
            wetness=100.0,
            sun_altitude_angle=8.0,
        )
    raise ValueError(f"Unknown weather_preset: {preset}")


def apply_weather_preset(world: carla.World, preset: str) -> None:
    set_weather = getattr(world, "set_weather", None)
    if not callable(set_weather):
        return
    weather = build_weather_parameters(preset)
    if weather is None:
        clear_weather = getattr(
            carla.WeatherParameters,
            "Default",
            getattr(carla.WeatherParameters, "ClearNoon", None),
        )
        if clear_weather is not None:
            set_weather(clear_weather)
        return
    set_weather(weather)


def apply_vision_attack(
    frames: Dict[str, Optional[np.ndarray]],
    attack: str = "none",
    intensity: float = 1.0,
    disable_semantic: bool = False,
) -> Dict[str, Optional[np.ndarray]]:
    out = dict(frames)
    mode = str(attack or "none").lower()
    if mode in {"none", "off", "default"}:
        if disable_semantic:
            out["semantic"] = None
        return out
    if mode not in {"texture", "weather"}:
        raise ValueError(f"Unknown vision_attack: {attack}")

    rgb = out.get("rgb")
    if rgb is not None:
        if mode == "texture":
            out["rgb"] = _apply_texture_attack(np.asarray(rgb), intensity=float(intensity))
        else:
            out["rgb"] = _apply_weather_attack(np.asarray(rgb), intensity=float(intensity))
    if disable_semantic:
        out["semantic"] = None
    return out


def _apply_weather_attack(rgb: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[0] < 8 or rgb.shape[1] < 8:
        return rgb.copy()

    amount = float(max(0.1, min(2.0, intensity)))
    image = rgb[:, :, :3].astype(np.float32)
    gray = np.mean(image, axis=2, keepdims=True)
    fog = np.full_like(image, 125.0)
    image = gray * 0.35 + image * 0.15 + fog * (0.50 + 0.10 * amount)
    image *= max(0.30, 0.62 - 0.14 * amount)

    height, width = image.shape[:2]
    streak_step = max(5, int(width * 0.045 / amount))
    streak_width = max(1, int(width * 0.006 * amount))
    for x in range(-height, width, streak_step):
        for offset in range(streak_width):
            xs = x + offset + np.arange(height) // 3
            valid = (xs >= 0) & (xs < width)
            image[np.arange(height)[valid], xs[valid], :] = 150.0

    horizon = np.linspace(1.0, 0.55, height, dtype=np.float32).reshape(height, 1, 1)
    image = image * horizon + 115.0 * (1.0 - horizon)
    out = rgb.copy()
    out[:, :, :3] = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return out


def _apply_texture_attack(rgb: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[0] < 8 or rgb.shape[1] < 8:
        return rgb.copy()

    amount = float(max(0.1, min(2.0, intensity)))
    out = rgb.copy()
    height, width = out.shape[:2]
    y0 = int(height * 0.48)
    stripe_h = max(2, int(height * 0.045 * amount))
    gap = max(4, int(height * 0.09))
    stripe_w = max(4, int(width * 0.075 * amount))
    # Draw a fake right-shifted lane: yellow left boundary, white right boundary.
    fake_boundaries = (
        (int(width * 0.76), (255, 220, 0)),
        (int(width * 0.98), (255, 255, 255)),
    )
    for center, rgb_color in fake_boundaries:
        x0 = max(0, center - stripe_w)
        x1 = min(width, center + stripe_w)
        color = np.array(rgb_color, dtype=np.uint8)
        for y in range(y0, height, gap):
            out[y : min(height, y + stripe_h), x0:x1, :3] = color

    checker_y0 = int(height * 0.76)
    tile = max(4, int(width * 0.035))
    for y in range(checker_y0, height, tile):
        for x in range(int(width * 0.80), width, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                out[y : min(height, y + tile), x : min(width, x + tile), :3] = 255
    return out
