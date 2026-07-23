"""Passive state types for Mancala-style pit-board tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image


SCENE_ID = "mancala_pit_board"
PITS_PER_ROW = 5
PIT_COUNT = PITS_PER_ROW * 2
LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(PIT_COUNT))
OPTION_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(4))
STYLE_VARIANTS: Tuple[str, ...] = ("wood_tray", "sand_stone", "slate_bowls", "cloth_pits", "arcade_pits")
SCENE_VARIANTS: Tuple[str, ...] = ("low_seed", "mixed_seed", "busy_seed")
PitBBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class MancalaDefaults:
    """Stable fallback defaults for one Mancala pit-board scene."""

    target_count_support: Tuple[int, ...] = tuple(range(9))
    target_landing_label_support: Tuple[str, ...] = LABELS
    landing_option_label_support: Tuple[str, ...] = OPTION_LABELS
    min_seed_count_per_pit: int = 0
    max_seed_count_per_pit: int = 8
    min_source_seed_count: int = 1
    max_source_seed_count: int = 8
    canvas_width: int = 980
    canvas_height: int = 440
    canvas_min_width_px: int = 760
    canvas_min_height_px: int = 340
    canvas_side_padding_px: int = 168
    canvas_vertical_padding_px: int = 132
    pit_width_px: int = 108
    pit_height_px: int = 68
    pit_gap_px: int = 20
    row_gap_px: int = 74
    board_padding_px: int = 48
    seed_diameter_min_px: int = 16
    seed_diameter_max_px: int = 20
    pit_outline_width_px: int = 4
    marker_width_px: int = 5


@dataclass(frozen=True)
class MancalaSceneAxes:
    """Resolved scene and style axes shared by both Mancala objectives."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Mapping[str, float]
    style_variant_probabilities: Mapping[str, float]
    min_seed_count_per_pit: int
    max_seed_count_per_pit: int
    min_source_seed_count: int
    max_source_seed_count: int


@dataclass(frozen=True)
class LabelAxisSelection:
    """Resolved label axis with support metadata for trace payloads."""

    value: str
    support: Tuple[str, ...]
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class IntegerAxisSelection:
    """Resolved integer axis with support metadata for trace payloads."""

    value: int
    support: Tuple[int, ...]
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class MancalaSample:
    """Symbolic pit-board state before and after one sowing move."""

    initial_counts: Tuple[int, ...]
    final_counts: Tuple[int, ...]
    source_index: int
    sowing_path_indices: Tuple[int, ...]
    landing_index: int
    target_index: int | None
    construction_mode: str
    option_pit_indices: Tuple[int, ...] = ()
    option_labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MancalaTheme:
    """Scene-local palette for a Mancala-style board."""

    tray_fill_rgb: Tuple[int, int, int]
    tray_border_rgb: Tuple[int, int, int]
    pit_fill_rgb: Tuple[int, int, int]
    pit_shadow_rgb: Tuple[int, int, int]
    pit_outline_rgb: Tuple[int, int, int]
    seed_rgbs: Tuple[Tuple[int, int, int], ...]
    seed_outline_rgb: Tuple[int, int, int]
    label_fill_rgb: Tuple[int, int, int]
    label_text_rgb: Tuple[int, int, int]
    arrow_rgb: Tuple[int, int, int]
    source_marker_rgb: Tuple[int, int, int]
    target_marker_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedMancalaScene:
    """Rendered board plus trace-friendly maps."""

    image: Image.Image
    entities: Tuple[Mapping[str, Any], ...]
    render_map: Mapping[str, Any]
    style_meta: Mapping[str, Any]
    background_meta: Mapping[str, Any]


DEFAULTS = MancalaDefaults()


__all__ = [
    "DEFAULTS",
    "LABELS",
    "OPTION_LABELS",
    "PIT_COUNT",
    "PITS_PER_ROW",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "STYLE_VARIANTS",
    "IntegerAxisSelection",
    "LabelAxisSelection",
    "MancalaDefaults",
    "MancalaSample",
    "MancalaSceneAxes",
    "MancalaTheme",
    "PitBBox",
    "RenderedMancalaScene",
]
