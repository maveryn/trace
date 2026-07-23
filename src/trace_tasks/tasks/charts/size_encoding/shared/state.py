"""Scene-local data structures for size-encoded chart rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "size_encoding"
SCENE_NAMESPACE = "charts.size_encoding"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "rect_word_cloud",
    "circle_word_cloud",
    "packed_bubble_cloud",
    "small_multiple_bubble_cloud",
)
SINGLE_PANEL_SCENE_VARIANTS: Tuple[str, ...] = (
    "rect_word_cloud",
    "circle_word_cloud",
    "packed_bubble_cloud",
)
PANEL_SCENE_VARIANTS: Tuple[str, ...] = ("small_multiple_bubble_cloud",)
SUPPORTED_EXTREMUM_DIRECTIONS: Tuple[str, ...] = ("largest", "smallest")

BBox = Tuple[float, float, float, float]
RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class SizeEncodingItem:
    item_id: str
    label: str
    category: str
    panel: str
    value: int


@dataclass(frozen=True)
class SizeEncodingDataset:
    items: Tuple[SizeEncodingItem, ...]
    categories: Tuple[str, ...]
    panels: Tuple[str, ...]
    trace: Dict[str, Any]


@dataclass(frozen=True)
class SizeEncodingSelection:
    answer: str
    annotation_item_ids: Tuple[str, ...]
    category_label: str
    panel_label: str
    reference_label: str
    direction: str
    trace: Dict[str, Any]


@dataclass(frozen=True)
class RenderedSizeEncodingScene:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    item_bboxes: Dict[str, List[float]]
    panel_title_bboxes: Dict[str, List[float]]
    category_legend_bboxes: Dict[str, List[float]]
    plot_bbox_px: List[float]
    render_meta: Dict[str, Any]
