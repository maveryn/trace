"""State objects for similar-figure measurement scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image


SCENE_ID = "similar_figure_measure_transfer"

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
Side = Tuple[int, int]


@dataclass(frozen=True)
class SimilarMeasureCase:
    """Scene-neutral data for one numeric similar-figure measurement setup."""

    construction_family: str
    shape_kind: str
    layout_kind: str
    answer: int
    target_name: str
    relation: str
    scale_factor: int
    source_target_side_value: int | None = None
    target_target_side_value: int | None = None
    support_source_side_value: int | None = None
    support_target_side_value: int | None = None
    source_perimeter: int | None = None
    target_perimeter: int | None = None
    source_area: int | None = None
    target_area: int | None = None
    area_ratio_label: str | None = None
    target_side: Side = (0, 1)
    support_side: Side = (2, 3)
    annotation_labels: Sequence[str] = ()


@dataclass(frozen=True)
class SimilarEquationCase:
    """Scene-neutral data for one expression-labeled similar-figure setup."""

    construction_family: str
    shape_kind: str
    answer: float
    target_name: str
    variable_name: str
    relation: str
    source_target_label: str
    target_target_label: str
    support_source_label: str
    support_target_label: str
    scale_factor: float
    source_target_value: float
    target_target_value: float
    annotation_labels: Sequence[str] = ()


@dataclass(frozen=True)
class FigureGeometry:
    """Projected source/target figure geometry after final layout transforms."""

    source_vertices: Tuple[Point, ...]
    target_vertices: Tuple[Point, ...]
    source_labels: Tuple[str, ...]
    target_labels: Tuple[str, ...]
    transform_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedSimilarScene:
    """Rendered scene plus geometry needed by public task annotation binding."""

    image: Image.Image
    figure_geometry: FigureGeometry
    point_label_bboxes: Mapping[str, BBox]
    readout_bboxes: Mapping[str, BBox]
    construction_bboxes: Mapping[str, BBox]
    style_metadata: Mapping[str, Any]
    background_metadata: Mapping[str, Any]
    render_map: Mapping[str, Any]
