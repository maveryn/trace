"""State containers for violin chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.tasks.charts.shared.chart_scene_types import ChartRenderParams, RenderedChartScene


@dataclass(frozen=True)
class ViolinRenderArtifacts:
    """Rendered violin chart plus neutral render metadata."""

    image: Image.Image
    rendered_scene: RenderedChartScene
    render_params: ChartRenderParams
    background_style: Mapping[str, Any]
    post_image_noise: Mapping[str, Any]
    chart_font_family: str
    mark_style: Mapping[str, Any]


__all__ = ["ViolinRenderArtifacts"]
