"""State containers for histogram chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.chart_scene_types import ChartRenderParams, HistogramBinSpec, RenderedChartScene
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts


@dataclass(frozen=True)
class HistogramRenderArtifacts:
    """Rendered histogram scene and neutral render metadata."""

    rendered_scene: RenderedChartScene
    render_params: ChartRenderParams
    background_style: Mapping[str, Any]
    information_scene_style: Mapping[str, Any]
    font_assets: Mapping[str, Any]
    mark_style: Mapping[str, Any]
    post_image_noise: Mapping[str, Any]


@dataclass(frozen=True)
class HistogramTaskPlan:
    """Task-owned semantic plan for one histogram instance."""

    bins: Sequence[HistogramBinSpec]
    params: Mapping[str, Any]
    mark_style: Mapping[str, Any]
    answer_gt: TypedValue
    answer_value: Any
    question_format: str
    annotation_type: str
    annotation_labels: Sequence[str]
    relations: Mapping[str, Any]
    prompt_artifacts: PromptTraceArtifacts


@dataclass(frozen=True)
class MaterializedHistogramTask:
    """Rendered payload assembled from a public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    branch: str
    prompt_variants: dict[str, Any]


__all__ = [
    "HistogramRenderArtifacts",
    "HistogramTaskPlan",
    "MaterializedHistogramTask",
]
