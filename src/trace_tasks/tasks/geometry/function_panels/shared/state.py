"""State records for function-panel scene primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

Point = tuple[float, float]
BBox = tuple[int, int, int, int]
Color = tuple[int, int, int]
LineSegment = tuple[Point, Point]
CircleSpec = tuple[Point, float]

SCENE_ID = "function_panels"
GRID_MIN = -5
GRID_MAX = 5
DEFAULT_LABEL_POOL_6 = ("A", "B", "C", "D", "E", "F")
DEFAULT_LABEL_POOL_9 = ("A", "B", "C", "D", "E", "F", "G", "H", "I")

RULE_FUNCTION_TEST = "function_test"
RULE_ONE_TO_ONE_TEST = "injective_function_test"
RULE_RANGE_MATCH = "range_interval_match"
RULE_X_AXIS_SYMMETRY = "horizontal_axis_symmetry"
RULE_SIGN_INTERVAL = "interval_sign_status"
SIGN_POSITIVE = "positive"
SIGN_NEGATIVE = "negative"

INTERSECTION_LINE_CIRCLE = "line_circle"
INTERSECTION_CIRCLE_CIRCLE = "circle_circle"
INTERSECTION_TANGENT = "tangent"
INTERSECTION_TWO_POINTS = "two_points"


@dataclass(frozen=True)
class RelationSpec:
    """One relation rendered inside a coordinate panel."""

    relation_id: str
    draw_kind: str
    domain: tuple[float, float]
    range: tuple[float, float]
    is_function: bool
    is_one_to_one: bool
    points: tuple[Point, ...] = ()
    center: Point | None = None
    radii: Point | None = None
    symmetry_axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PropertySelection:
    """Resolved label/options for one selected-panel property task."""

    selected_label: str
    label_pool: tuple[str, ...]
    label_probabilities: dict[str, float]
    panel_count_probabilities: dict[str, float]
    target_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class RenderedPropertyScene:
    """Rendered property-panel scene and projection metadata."""

    image: Image.Image
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    panel_style_meta: dict[str, Any]
    line_color_meta: dict[str, Any]
    line_colors: tuple[Color, ...]
    relations_by_label: dict[str, RelationSpec]
    panel_bboxes: dict[str, list[int]]
    plot_bboxes: dict[str, list[int]]
    panel_columns: int
    panel_rows: int
    panel_count_probabilities: dict[str, float]
    target_range: str
    target_interval: str


@dataclass(frozen=True)
class IntersectionPanelSpec:
    """One rendered primitive pair and its symbolic intersection properties."""

    pair_id: str
    pair_kind: str
    relation_class: str
    line_segments: tuple[LineSegment, ...]
    circles: tuple[CircleSpec, ...]
    intersection_points: tuple[Point, ...]
    intersection_quadrants: tuple[str, ...]


@dataclass(frozen=True)
class IntersectionSelection:
    """Resolved label/options for one intersection-panel task."""

    selected_label: str
    label_pool: tuple[str, ...]
    label_probabilities: dict[str, float]
    panel_count_probabilities: dict[str, float]


@dataclass(frozen=True)
class RenderedIntersectionScene:
    """Rendered intersection-panel scene and projection metadata."""

    image: Image.Image
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    panel_style_meta: dict[str, Any]
    panels_by_label: dict[str, IntersectionPanelSpec]
    panel_bboxes: dict[str, list[int]]
    plot_bboxes: dict[str, list[int]]
    intersection_point_bboxes: dict[str, list[list[int]]]
    panel_columns: int
    panel_rows: int
    panel_count_probabilities: dict[str, float]
    object_color_meta: dict[str, Any]
    object_colors: tuple[Color, ...]
    intersection_color_meta: dict[str, Any]
    intersection_color: Color
