"""State containers for single-series chart scene primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "single_series"
SCENE_NAMESPACE = "charts_single_series"


@dataclass(frozen=True)
class SingleSeriesDataset:
    """Task-bound symbolic chart values before rendering."""

    labels: tuple[str, ...]
    values: tuple[int, ...]
    answer_value: int | str
    answer_type: str
    annotation_labels: tuple[str, ...]
    ordered_annotation_labels: tuple[str, ...]
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class SingleSeriesRenderResult:
    """Rendered chart plus projection metadata used by task materialization."""

    image: Image.Image
    rendered_scene: Any
    render_params: Any
    chart_font_family: str
    mark_style: Mapping[str, Any]
    background_meta: Mapping[str, Any]
    information_style_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]


def as_int_values(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def as_label_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


__all__ = [
    "DOMAIN",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SingleSeriesDataset",
    "SingleSeriesRenderResult",
    "as_int_values",
    "as_label_tuple",
]
