"""Passive state models for balance-scale puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image

SCENE_ID = "balance_scale"
OBJECT_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SCENE_VARIANTS: tuple[str, ...] = (
    "balance_sheet",
    "balance_card",
    "balance_outline",
)
TARGET_CUE_MODES: tuple[str, ...] = (
    "query_row_only",
    "query_row_and_highlight",
)
MISSING_WEIGHT_ROW_KIND = "missing_weight"
EQUIVALENT_COUNT_ROW_KIND = "equivalent_count"
WEIGHT_ORDER_ROW_KIND = "weight_order"
SIDE_RELATION_ROW_KIND = "query_side_relation"


@dataclass(frozen=True)
class BalanceScaleRenderParams:
    """Resolved render parameters for one balance-scale scene."""

    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    scale_panel_gap_px: int
    query_row_height_px: int
    beam_width_px: int
    pan_width_px: int
    pan_height_px: int
    token_size_px: int
    token_gap_px: int
    line_width_px: int
    value_font_size_px: int
    label_font_size_px: int
    query_font_size_px: int
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedBalanceScaleScene:
    """Rendered balance-scale image with traceable object geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    item_bbox_map: Dict[str, List[float]]


@dataclass(frozen=True)
class BalanceSceneAxes:
    """Resolved visual axes shared by all balance-scale objectives."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    target_cue_mode: str
    target_cue_mode_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedBalanceContext:
    """Rendered image plus metadata needed by public task output builders."""

    image: Image.Image
    rendered_scene: RenderedBalanceScaleScene
    render_meta: Dict[str, Any]
    scene_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    font_meta: Dict[str, Any]


__all__ = [
    "BalanceScaleRenderParams",
    "BalanceSceneAxes",
    "EQUIVALENT_COUNT_ROW_KIND",
    "MISSING_WEIGHT_ROW_KIND",
    "OBJECT_LABELS",
    "RenderedBalanceContext",
    "RenderedBalanceScaleScene",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "SIDE_RELATION_ROW_KIND",
    "TARGET_CUE_MODES",
    "WEIGHT_ORDER_ROW_KIND",
]
