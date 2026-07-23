"""State objects for radial-progress chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "radial_progress"
SCENE_NAMESPACE = "charts.radial_progress"
PROMPT_BUNDLE_ID = "charts_radial_progress_v1"

FULL_PROGRESS_RINGS = "full_progress_rings"
SEMICIRCLE_GAUGES = "semicircle_gauges"
SEGMENTED_RADIAL_BARS = "segmented_radial_bars"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (
    FULL_PROGRESS_RINGS,
    SEMICIRCLE_GAUGES,
    SEGMENTED_RADIAL_BARS,
)

BBox = list[float]
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class ProgressItem:
    item_id: str
    label: str
    value: int
    color_rgb: RGB


@dataclass(frozen=True)
class ProgressFrame:
    scene_variant: str
    scene_probabilities: dict[str, float]
    item_count: int
    item_count_probabilities: dict[str, float]
    labels: tuple[str, ...]
    colors: tuple[RGB, ...]
    title: str


@dataclass(frozen=True)
class ProgressQuestion:
    branch_id: str
    answer: int | str
    answer_type: str
    annotation_type: str
    annotation_item_ids: tuple[str, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class ProgressDataset:
    items: tuple[ProgressItem, ...]
    scene_variant: str
    branch_id: str
    branch_probabilities: dict[str, float]
    question: ProgressQuestion
    title: str


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    title_band_height_px: int
    card_gap_px: int
    card_corner_radius_px: int
    card_outline_width_px: int
    title_font_size_px: int
    label_font_size_px: int
    tick_font_size_px: int
    ring_width_px: int
    gauge_width_px: int
    segment_width_px: int
    tick_length_px: int
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    card_fill_rgb: RGB
    card_alt_fill_rgb: RGB
    card_outline_rgb: RGB
    track_rgb: RGB
    tick_rgb: RGB
    needle_rgb: RGB


@dataclass(frozen=True)
class RenderedProgressScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    item_bboxes_px: dict[str, BBox]
    progress_bboxes_px: dict[str, BBox]
    render_meta: dict[str, Any]


@dataclass(frozen=True)
class RadialProgressRenderResult:
    rendered_scene: RenderedProgressScene
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str

    @property
    def image(self) -> Image.Image:
        return self.rendered_scene.image
