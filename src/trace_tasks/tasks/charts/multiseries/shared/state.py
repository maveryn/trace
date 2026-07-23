"""State containers for the multiseries chart scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image

from ...shared.labeled_chart_defaults import LabeledChartDefaults


SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS: Tuple[str, ...] = (
    "grouped_bar",
    "grouped_horizontal_bar",
    "multi_line",
    "grouped_lollipop",
)


@dataclass(frozen=True)
class MultiseriesChartDefaults(LabeledChartDefaults):
    """Stable fallback defaults shared by multiseries chart tasks."""

    category_count_min: int = 5
    category_count_max: int = 15
    series_count_min: int = 2
    series_count_max: int = 4
    target_answer_min: int = 0
    target_answer_max: int = 8


@dataclass(frozen=True)
class MultiseriesRenderResult:
    """Rendered multiseries chart plus projection metadata."""

    image: Image.Image
    rendered_scene: Any
    render_params: Any
    mark_style: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    chart_font_family: str
    category_labels: list[str]
    series_labels: list[str]


__all__ = [
    "MultiseriesChartDefaults",
    "MultiseriesRenderResult",
    "SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS",
]
