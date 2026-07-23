"""Scene-local state containers for annotated-series generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.tasks.charts.shared.chart_scene_types import RenderedChartScene
from trace_tasks.tasks.shared.visual_style.context_layer import ContextTextElement


@dataclass(frozen=True)
class SeriesSample:
    scene_variant: str
    labels: tuple[str, ...]
    values: tuple[int, ...]
    scene_variant_probabilities: dict[str, float]


@dataclass(frozen=True)
class RenderedBaseSeries:
    image: Image.Image
    rendered_scene: RenderedChartScene
    context_layout: dict[str, Any]
    render_params: Any
    mark_style: dict[str, Any]
    background_meta: dict[str, Any]
    information_style_meta: dict[str, Any]
    chart_font_family: str
    annotation_font_family: str


@dataclass(frozen=True)
class MarkupRender:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    annotation_bboxes: dict[str, list[float]]


@dataclass(frozen=True)
class FinalRender:
    image: Image.Image
    context_elements: tuple[ContextTextElement, ...]
    post_noise_meta: dict[str, Any]
