"""Small compatibility subset of xinshuo_visualization used by AB3DMOT."""

from __future__ import annotations

import colorsys


def random_colors(num_colors: int, bright: bool = True):
    """Return deterministic RGB float tuples in the format AB3DMOT expects."""
    if num_colors <= 0:
        return []
    brightness = 1.0 if bright else 0.7
    return [
        colorsys.hsv_to_rgb(index / num_colors, 1.0, brightness)
        for index in range(num_colors)
    ]
