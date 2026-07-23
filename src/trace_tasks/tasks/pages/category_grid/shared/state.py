"""Passive scene state for category-grid tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    header_height_px: int
    gap_px: int
    corner_radius_px: int
    outline_width_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    category_title_font_size_px: int
    subcategory_title_font_size_px: int
    item_font_size_px: int


@dataclass(frozen=True)
class CategoryItem:
    item_id: str
    label: str


@dataclass(frozen=True)
class Subcategory:
    subcategory_id: str
    label: str
    items: Tuple[CategoryItem, ...]


@dataclass(frozen=True)
class Category:
    category_id: str
    label: str
    accent_rgb: Tuple[int, int, int]
    subcategories: Tuple[Subcategory, ...]


@dataclass(frozen=True)
class CategoryGridSpec:
    title: str
    subtitle: str
    categories: Tuple[Category, ...]
    text_resource_metadata: Dict[str, Any]


@dataclass(frozen=True)
class CategoryGridCase:
    scene_variant: str
    category_count: int
    subcategory_count: int
    category_count_support: Tuple[int, ...]
    subcategory_count_support: Tuple[int, ...]
    item_count_support: Tuple[int, ...]
    spec: CategoryGridSpec
    target_category: Category
    target_subcategory: Subcategory
    target_slot_index: int | None
    target_item: CategoryItem | None
    scene_variant_probabilities: Dict[str, float]
    category_count_probabilities: Dict[str, float]
    subcategory_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedCategoryGrid:
    image: Image.Image
    entities: List[Dict[str, Any]]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    category_header_bboxes_px: Dict[str, List[float]]
    subcategory_header_bboxes_px: Dict[str, Dict[str, List[float]]]
    item_row_bboxes_px: Dict[str, Dict[str, Dict[str, List[float]]]]
    item_label_bboxes_px: Dict[str, Dict[str, Dict[str, List[float]]]]
    layout_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedCategoryGridBundle:
    image: Image.Image
    rendered_grid: RenderedCategoryGrid
    render_params: RenderParams
    background_meta: Dict[str, Any]
    style_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
