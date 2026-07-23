"""Shared helpers for stable color-name / hex prompt text."""

from __future__ import annotations

from typing import Sequence, Tuple


Color = Tuple[int, int, int]


def rgb_to_hex(color: Sequence[int]) -> str:
    """Return one canonical uppercase `#RRGGBB` string."""
    if len(color) < 3:
        raise ValueError("rgb_to_hex requires three channels")
    red = max(0, min(255, int(color[0])))
    green = max(0, min(255, int(color[1])))
    blue = max(0, min(255, int(color[2])))
    return f"#{red:02X}{green:02X}{blue:02X}"


def format_named_color_with_hex(name: str, color: Sequence[int]) -> str:
    """Return one prompt-facing named color label with hex in square brackets."""
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("color name cannot be empty")
    return f"{clean_name} [{rgb_to_hex(color)}]"


__all__ = [
    "Color",
    "format_named_color_with_hex",
    "rgb_to_hex",
]
