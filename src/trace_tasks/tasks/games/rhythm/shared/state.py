"""Passive state and constants for Rhythm scene-package tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


SCENE_ID = "rhythm"
SCENE_NAMESPACE = "games.rhythm"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("falling_notes",)
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = ("arcade", "neon", "paper", "dark", "pastel")
SUPPORTED_COLOR_KEYS: Tuple[str, ...] = ("yellow", "cyan", "magenta", "green")


@dataclass(frozen=True)
class RhythmNote:
    """One visible falling rhythm note.

    `bottom_row` is one-indexed from the hit line upward. A note with
    `bottom_row == 1` reaches the hit line after one beat.
    """

    note_id: str
    lane_index: int
    bottom_row: int
    length: int
    color_key: str
    kind: str


@dataclass(frozen=True)
class RhythmVisualAxes:
    """Resolved visual and timing axes shared by Rhythm public tasks."""

    scene_variant: str
    style_variant: str
    lane_count: int
    row_count: int
    beat_window: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    lane_count_probabilities: Dict[str, float]
    row_count_probabilities: Dict[str, float]
    beat_window_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RhythmCountTargetAxis:
    """Resolved integer target for count-style Rhythm objectives."""

    target_count: int
    target_count_support: Tuple[int, ...]
    target_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SampledRhythmScene:
    """One sampled Rhythm board plus task-bound symbolic witnesses."""

    lane_count: int
    row_count: int
    beat_window: int
    scene_variant: str
    selected_lane_index: int | None
    selected_lane_label: str | None
    target_color_key: str | None
    answer: int
    notes: Tuple[RhythmNote, ...]
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str
    score_values_by_color: Mapping[str, int] | None = None


__all__ = [
    "RhythmCountTargetAxis",
    "RhythmNote",
    "RhythmVisualAxes",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_COLOR_KEYS",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
    "SampledRhythmScene",
]
