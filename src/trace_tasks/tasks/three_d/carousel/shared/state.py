"""Scene state and catalogs for synthetic 3D conveyor scenes."""

from __future__ import annotations

from typing import Mapping, Tuple

from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.shared.named_colors import available_named_colors
from trace_tasks.tasks.three_d.shared.object_resources import (
    OBJECT_CLUSTER_NAME_BY_SHAPE_TYPE,
    OBJECT_CLUSTER_SHAPE_TYPES,
)
from trace_tasks.tasks.three_d.shared.semantic_colors import COLOR_CONFUSION_EXCLUSIONS


SCENE_ID = "carousel"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "airport_carousel",
    "inspection_carousel",
    "warehouse_carousel",
)

BELT_KEYS: Tuple[str, ...] = ("inner", "outer")
BELT_LABELS: Mapping[str, str] = {
    "inner": "INNER",
    "outer": "OUTER",
}
BELT_GEOMETRY: Mapping[str, Mapping[str, float]] = {
    "inner": {
        "radius_x": 1.95,
        "radius_y": 0.98,
        "band_width": 0.42,
    },
    "outer": {
        "radius_x": 3.2,
        "radius_y": 1.68,
        "band_width": 0.52,
    },
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
        "diamond",
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

SEMANTIC_COLOR_RGB: Mapping[str, Tuple[int, int, int]] = {
    str(name): (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    for name, rgb in available_named_colors()
}
SEMANTIC_COLOR_SUPPORT: Tuple[str, ...] = tuple(SEMANTIC_COLOR_RGB.keys())

def semantic_color_label(color_name: str) -> str:
    """Return prompt-facing canonical named color text."""

    return format_named_color_with_hex(str(color_name), SEMANTIC_COLOR_RGB[str(color_name)])


def public_object_name(shape_type: str) -> str:
    """Return the short prompt-facing object name for one conveyor object."""

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


__all__ = [
    "BELT_GEOMETRY",
    "BELT_KEYS",
    "BELT_LABELS",
    "COLOR_CONFUSION_EXCLUSIONS",
    "CONVEYOR_COLOR_READOUT_SHAPE_TYPES",
    "CONVEYOR_OBJECT_SHAPE_TYPES",
    "SCENE_ID",
    "SEMANTIC_COLOR_RGB",
    "SEMANTIC_COLOR_SUPPORT",
    "SUPPORTED_SCENE_VARIANTS",
    "public_object_name",
    "public_object_plural",
    "semantic_color_label",
]
