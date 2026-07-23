"""Scene state and object catalogs for straight 3D conveyor scenes."""

from __future__ import annotations

from typing import Mapping, Tuple

from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.shared.named_colors import available_named_colors, sample_named_color_palette
from trace_tasks.tasks.three_d.shared.object_resources import (
    OBJECT_CLUSTER_DIMENSIONS,
    OBJECT_CLUSTER_NAME_BY_SHAPE_TYPE,
    OBJECT_CLUSTER_SHAPE_TYPES,
)
from trace_tasks.tasks.three_d.shared.semantic_colors import COLOR_CONFUSION_EXCLUSIONS


SCENE_ID = "conveyor"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "warehouse_line",
    "factory_line",
    "parcel_line",
)

SEMANTIC_COLOR_RGB: Mapping[str, Tuple[int, int, int]] = {
    str(name): (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    for name, rgb in available_named_colors()
}

HORIZONTAL_LANE_KEYS: Tuple[str, ...] = ("top", "middle", "bottom")
VERTICAL_LANE_KEYS: Tuple[str, ...] = ("left", "middle", "right")
LANE_HALF_WIDTH: float = 0.46
LANE_SLOT_JITTER_ACROSS: float = 0.07
LANE_SLOT_JITTER_ALONG: float = 0.10
HORIZONTAL_LANE_LENGTH: float = 7.65
VERTICAL_LANE_LENGTH: float = 7.65
HORIZONTAL_SLOT_LENGTH: float = 6.95
VERTICAL_SLOT_LENGTH: float = 6.95

LANE_LABELS: Mapping[str, str] = {
    "top": "TOP",
    "bottom": "BOTTOM",
    "left": "LEFT",
    "right": "RIGHT",
    "middle": "MIDDLE",
}

HORIZONTAL_LANE_CENTER_BY_KEY: Mapping[str, float] = {
    "top": 1.85,
    "middle": 0.0,
    "bottom": -1.85,
}

VERTICAL_LANE_CENTER_BY_KEY: Mapping[str, float] = {
    "left": -1.85,
    "middle": 0.0,
    "right": 1.85,
}

CONVEYOR_OBJECT_SHAPE_TYPES: Tuple[str, ...] = tuple(
    shape
    for shape in (
        "apple",
        "sphere",
        "bell",
        "open_book",
        "bottle",
        "bowl",
        "button",
        "cactus",
        "calculator",
        "candle",
        "card",
        "carrot",
        "chess_piece",
        "clock",
        "cone",
        "cube",
        "cup",
        "cylinder",
        "dice",
        "mail_envelope",
        "flower",
        "glove",
        "hat",
        "heart",
        "jar",
        "key",
        "lantern",
        "leaf",
        "mushroom",
        "plate",
        "plug",
        "puzzle_piece",
        "pyramid",
        "remote_control",
        "torus",
        "shield",
        "star_prism",
        "tray",
        "trophy",
        "umbrella",
    )
    if shape in set(OBJECT_CLUSTER_SHAPE_TYPES)
)

CONVEYOR_COLOR_READOUT_SHAPE_TYPES: Tuple[str, ...] = tuple(
    shape
    for shape in (
        "sphere",
        "cube",
        "cylinder",
        "cone",
        "torus",
        "pyramid",
        "star_prism",
        "heart",
        "puzzle_piece",
        "cup",
        "bowl",
        "flower",
        "tray",
        "chess_piece",
        "shield",
        "dice",
        "bell",
        "candle",
        "trophy",
        "hat",
        "umbrella",
        "button",
        "card",
        "mail_envelope",
        "glove",
        "key",
        "plate",
    )
    if shape in set(CONVEYOR_OBJECT_SHAPE_TYPES)
)


SEMANTIC_COLOR_SUPPORT: Tuple[str, ...] = tuple(SEMANTIC_COLOR_RGB.keys())


def public_object_name(shape_type: str) -> str:
    """Return the prompt-facing object name for one conveyor object."""

    return str(OBJECT_CLUSTER_NAME_BY_SHAPE_TYPE.get(str(shape_type), str(shape_type).replace("_", " ")))


def public_object_plural(shape_type: str) -> str:
    """Return the prompt-facing plural object name for one conveyor object."""

    raw = public_object_name(str(shape_type)).strip()
    if raw in {"fish", "dice"}:
        return raw
    if raw.endswith("y") and (len(raw) < 2 or raw[-2].lower() not in {"a", "e", "i", "o", "u"}):
        return f"{raw[:-1]}ies"
    if raw.endswith(("s", "x", "z", "ch", "sh")):
        return f"{raw}es"
    return f"{raw}s"


def object_dimensions(shape_type: str, *, scale: float) -> tuple[float, float, float]:
    """Return scaled base dimensions for one scene object."""

    base = OBJECT_CLUSTER_DIMENSIONS.get(str(shape_type), (0.52, 0.52, 0.52))
    return tuple(round(float(value) * float(scale), 4) for value in base)


def sample_visual_color_names(rng: object, *, palette_size: int = 4) -> tuple[str, ...]:
    """Sample canonical named colors for non-semantic visual variety."""

    palette = sample_named_color_palette(rng, palette_size=int(palette_size))
    return tuple(str(name) for name, _rgb in palette)


def semantic_color_label(color_name: str) -> str:
    """Return prompt-facing canonical named color text."""

    return format_named_color_with_hex(str(color_name), SEMANTIC_COLOR_RGB[str(color_name)])


__all__ = [
    "CONVEYOR_COLOR_READOUT_SHAPE_TYPES",
    "CONVEYOR_OBJECT_SHAPE_TYPES",
    "COLOR_CONFUSION_EXCLUSIONS",
    "HORIZONTAL_LANE_CENTER_BY_KEY",
    "HORIZONTAL_LANE_KEYS",
    "HORIZONTAL_LANE_LENGTH",
    "HORIZONTAL_SLOT_LENGTH",
    "LANE_HALF_WIDTH",
    "LANE_LABELS",
    "LANE_SLOT_JITTER_ACROSS",
    "LANE_SLOT_JITTER_ALONG",
    "SCENE_ID",
    "SEMANTIC_COLOR_RGB",
    "SEMANTIC_COLOR_SUPPORT",
    "SUPPORTED_SCENE_VARIANTS",
    "VERTICAL_LANE_CENTER_BY_KEY",
    "VERTICAL_LANE_KEYS",
    "VERTICAL_LANE_LENGTH",
    "VERTICAL_SLOT_LENGTH",
    "object_dimensions",
    "public_object_name",
    "public_object_plural",
    "sample_visual_color_names",
    "semantic_color_label",
]
