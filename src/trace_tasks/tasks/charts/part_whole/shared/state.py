"""State containers for part-whole chart rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class CategorySpec:
    """One visible part-whole category and its integer share."""

    label: str
    value: int
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class PartWholeDataset:
    """Scene state after one public task has bound its objective."""

    categories: tuple[CategorySpec, ...]
    answer_value: int
    annotation_labels: tuple[str, ...]
    trace_extras: dict[str, Any]


@dataclass(frozen=True)
class TransferQuery:
    """Adjacent transfer operands selected by a public task."""

    source: CategorySpec
    target: CategorySpec
    delta: int
    extras: dict[str, Any]


@dataclass(frozen=True)
class RenderedShareChart:
    """Rendered chart geometry and projection maps."""

    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: tuple[int, int, int, int]
    table_bbox_px: tuple[int, int, int, int]
    chart_traces: tuple[dict[str, Any], ...]
    category_traces: tuple[dict[str, Any], ...]
    annotation_bbox_by_label: dict[str, list[float]]
    annotation_point_by_label: dict[str, list[float]]
    layout_jitter_meta: dict[str, Any]
