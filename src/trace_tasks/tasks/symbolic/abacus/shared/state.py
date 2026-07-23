"""Passive state objects for the symbolic abacus scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class AbacusColumnSpec:
    item_id: str
    role: str
    place_label: str
    place_value: int
    digit: int


@dataclass(frozen=True)
class AbacusReadoutOptionSpec:
    label: str
    text: str
    value: int
    is_correct: bool


@dataclass(frozen=True)
class AbacusReadoutRenderParams:
    canvas_width: int = 980
    canvas_height: int = 760
    panel_width_px: int = 800
    panel_height_px: int = 540
    panel_corner_radius_px: int = 24
    frame_width_px: int = 8
    rod_width_px: int = 5
    beam_height_px: int = 22
    bead_width_px: int = 58
    bead_height_px: int = 34
    title_font_size_px: int = 25
    label_font_size_px: int = 23
    small_font_size_px: int = 16
    readout_option_card_width_px: int = 130
    readout_option_card_height_px: int = 58
    readout_option_card_gap_px: int = 12
    readout_option_card_margin_top_px: int = 24
    readout_option_label_font_size_px: int = 22
    readout_option_value_font_size_px: int = 24


@dataclass(frozen=True)
class AbacusOptionPanelRenderParams:
    canvas_width: int = 1200
    canvas_height: int = 760
    option_card_width_px: int = 340
    option_card_height_px: int = 280
    option_card_gap_x_px: int = 44
    option_card_gap_y_px: int = 52
    option_card_corner_radius_px: int = 18
    option_label_font_size_px: int = 26
    option_place_label_font_size_px: int = 16
    option_bead_width_px: int = 32
    option_bead_height_px: int = 20
    option_rod_width_px: int = 3
    option_beam_height_px: int = 10


@dataclass(frozen=True)
class AbacusOptionSpec:
    label: str
    value: int
    is_correct: bool


@dataclass(frozen=True)
class RenderedAbacusReadoutScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    item_bboxes: dict[str, list[float]]
    bead_bboxes: dict[str, list[float]]
    active_bead_bboxes_by_column: dict[str, list[list[float]]]
    active_bead_points_by_column: dict[str, list[list[float]]]
    active_bead_ids_by_column: dict[str, list[str]]
    column_bboxes: dict[str, list[float]]
    label_bboxes: dict[str, list[float]]
    option_card_bboxes: dict[str, list[float]]
    option_values_by_label: dict[str, int]
    selected_option_card_bbox: list[float] | None
    scene_bbox_px: list[float]
    style_metadata: dict[str, Any]


@dataclass(frozen=True)
class RenderedAbacusOptionPanelScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    item_bboxes: dict[str, list[float]]
    option_card_bboxes: dict[str, list[float]]
    option_abacus_bboxes: dict[str, list[float]]
    option_values_by_label: dict[str, int]
    selected_option_card_bbox: list[float]
    selected_option_abacus_bbox: list[float]
    scene_bbox_px: list[float]
    style_metadata: dict[str, Any]


__all__ = [
    "AbacusColumnSpec",
    "AbacusOptionPanelRenderParams",
    "AbacusOptionSpec",
    "AbacusReadoutOptionSpec",
    "AbacusReadoutRenderParams",
    "RenderedAbacusOptionPanelScene",
    "RenderedAbacusReadoutScene",
]
