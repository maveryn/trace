"""Passive state contracts for pinball-table games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SUPPORTED_PINBALL_SCENE_VARIANTS: Tuple[str, ...] = ("schematic_table",)
SUPPORTED_PINBALL_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "blueprint",
    "neon",
    "carnival",
    "paper",
)
SUPPORTED_PINBALL_OBJECT_KINDS: Tuple[str, ...] = (
    "bumper",
    "drop_target",
    "rollover_lane",
    "standup_target",
)


@dataclass(frozen=True)
class PinballObject:
    """One labeled object on a pinball table."""

    object_id: str
    label: str
    kind: str
    x_norm: float
    y_norm: float
    radius_norm: float
    width_norm: float
    height_norm: float
    color_index: int
    score_value: int | None = None
    show_label: bool = True


@dataclass(frozen=True)
class PinballSceneState:
    """Generated pinball table state before objective answer binding."""

    scene_variant: str
    style_variant: str
    ball_x_norm: float
    ball_y_norm: float
    cue_angle_rad: float
    cue_visible_fraction: float
    objects: Tuple[PinballObject, ...]
    construction_mode: str
    hidden_path_norm: Tuple[Tuple[float, float], ...]


def pinball_object_id(index: int) -> str:
    """Return stable entity id for one pinball object."""

    return f"object_{int(index)}"


def validate_pinball_scene_state(scene: PinballSceneState) -> None:
    """Validate passive scene consistency independent of one public objective."""

    if str(scene.scene_variant) not in SUPPORTED_PINBALL_SCENE_VARIANTS:
        raise ValueError(f"unsupported pinball scene variant: {scene.scene_variant}")
    if str(scene.style_variant) not in SUPPORTED_PINBALL_STYLE_VARIANTS:
        raise ValueError(f"unsupported pinball style variant: {scene.style_variant}")
    object_ids = [str(obj.object_id) for obj in scene.objects]
    object_labels = [str(obj.label) for obj in scene.objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("pinball object ids must be unique")
    if len(object_labels) != len(set(object_labels)):
        raise ValueError("pinball object labels must be unique")
    if any(str(obj.kind) not in SUPPORTED_PINBALL_OBJECT_KINDS for obj in scene.objects):
        raise ValueError("pinball objects must use supported object kinds")
    if len(scene.hidden_path_norm) < 2:
        raise ValueError("pinball hidden path must contain at least two points")


__all__ = [
    "SUPPORTED_PINBALL_OBJECT_KINDS",
    "SUPPORTED_PINBALL_SCENE_VARIANTS",
    "SUPPORTED_PINBALL_STYLE_VARIANTS",
    "PinballObject",
    "PinballSceneState",
    "pinball_object_id",
    "validate_pinball_scene_state",
]
