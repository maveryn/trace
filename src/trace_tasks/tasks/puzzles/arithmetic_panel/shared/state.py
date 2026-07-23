"""State contracts for the arithmetic-constraint puzzle scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

SCENE_ID = "arithmetic_panel"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "constraint_sheet",
    "constraint_card",
    "constraint_outline",
)


@dataclass(frozen=True)
class ArithmeticRenderParams:
    """Resolved canvas and mark dimensions for one arithmetic panel."""

    canvas_width: int
    canvas_height: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    cell_width_px: int
    cell_height_px: int
    node_radius_px: int
    line_width_px: int
    value_font_size_px: int
    note_font_size_px: int
    symbol_font_size_px: int
    unit_size_jitter: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArithmeticCase:
    """Generated symbolic puzzle state before rendering."""

    kind: str
    layout_style: str
    answer_value: int
    answer_support: Tuple[int, ...]
    answer_range: Tuple[int, int]
    target_item_id: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedArithmeticScene:
    """Rendered arithmetic panel with item geometry."""

    image: Any
    entities: Tuple[Dict[str, Any], ...]
    scene_bbox_px: Tuple[float, float, float, float]
    item_bbox_map: Mapping[str, Tuple[float, float, float, float]]


def target_bbox(
    rendered_scene: RenderedArithmeticScene, item_id: str
) -> Tuple[float, float, float, float]:
    """Return the rendered bbox for the single target witness."""

    key = str(item_id)
    if key not in rendered_scene.item_bbox_map:
        raise RuntimeError(f"missing arithmetic target bbox: {key!r}")
    bbox = rendered_scene.item_bbox_map[key]
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


__all__ = [
    "ArithmeticCase",
    "ArithmeticRenderParams",
    "RenderedArithmeticScene",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "target_bbox",
]
