"""Style-contrast contracts for composite-shape geometry scenes."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.composite_shape.house_outline_perimeter import GeometryMeasurementCompositePerimeterValueTask
from trace_tasks.tasks.geometry.composite_shape.l_profile_area import GeometryLProfileAreaTask
from trace_tasks.tasks.geometry.composite_shape.missing_width_from_semicircle_area import (
    GeometryMissingWidthFromSemicircleAreaTask,
)
from trace_tasks.tasks.geometry.composite_shape.rectangle_quarter_sector_cutout_area import (
    GeometryRectangleQuarterSectorCutoutAreaTask,
)
from trace_tasks.tasks.geometry.composite_shape.rectangle_quarter_sector_cutout_perimeter import (
    GeometryRectangleQuarterSectorCutoutPerimeterTask,
)
from trace_tasks.tasks.geometry.composite_shape.rectangle_semicircle_area import GeometryRectangleSemicircleAreaTask
from trace_tasks.tasks.geometry.composite_shape.rectangle_semicircle_perimeter import (
    GeometryRectangleSemicirclePerimeterTask,
)
from trace_tasks.tasks.geometry.composite_shape.rectangle_triangle_cutout_area import GeometryRectangleTriangleCutoutAreaTask
from trace_tasks.tasks.geometry.composite_shape.sector_angle_value import GeometrySectorAngleValueTask
from trace_tasks.tasks.geometry.composite_shape.tabbed_rectilinear_perimeter import (
    GeometryCompositeShapeTabbedRectilinearPerimeterTask,
)
from trace_tasks.tasks.shared.color_distance import color_distance


COMPOSITE_SHAPE_TASK_CLASSES = (
    GeometryRectangleTriangleCutoutAreaTask,
    GeometryLProfileAreaTask,
    GeometryMeasurementCompositePerimeterValueTask,
    GeometryCompositeShapeTabbedRectilinearPerimeterTask,
    GeometryRectangleSemicircleAreaTask,
    GeometryRectangleQuarterSectorCutoutAreaTask,
    GeometryRectangleSemicirclePerimeterTask,
    GeometryRectangleQuarterSectorCutoutPerimeterTask,
    GeometryMissingWidthFromSemicircleAreaTask,
    GeometrySectorAngleValueTask,
)


def _rgb(values) -> tuple[int, int, int]:
    return tuple(int(value) for value in values[:3])


@pytest.mark.parametrize("task_cls", COMPOSITE_SHAPE_TASK_CLASSES)
@pytest.mark.parametrize("seed", (54001, 54011, 70501177616888))
def test_composite_shape_semantic_fills_stay_visible_against_background(task_cls, seed) -> None:
    task = task_cls()
    out = task.generate(seed, params={}, max_attempts=20)
    render_spec = out.trace_payload["render_spec"]
    style_meta = render_spec["composite_fill_style"]
    anchors = tuple(_rgb(color) for color in style_meta["surface_anchor_rgb"])
    fill = _rgb(render_spec["fill_color"])
    secondary_fill = _rgb(render_spec["secondary_fill_color"])
    min_background_distance = float(style_meta["min_fill_background_lab_distance_required"])
    min_pairwise_distance = float(style_meta["min_fill_pairwise_lab_distance_required"])

    assert min(color_distance(fill, anchor, distance_space="lab") for anchor in anchors) >= min_background_distance
    assert min(color_distance(secondary_fill, anchor, distance_space="lab") for anchor in anchors) >= min_background_distance
    assert color_distance(fill, secondary_fill, distance_space="lab") >= min_pairwise_distance
    assert style_meta["policy"] == "composite_shape_semantic_fill_contrast_v1"
    assert "readout_text_style" in style_meta


def test_rectangle_semicircle_cutout_issue_seed_uses_visible_shaded_fill() -> None:
    task = GeometryRectangleSemicircleAreaTask()
    out = task.generate(70501177616888, params={"query_id": "cutout_area"}, max_attempts=20)
    style_meta = out.trace_payload["render_spec"]["composite_fill_style"]

    assert style_meta["fill_background_lab_distance_min"] >= style_meta["min_fill_background_lab_distance_required"]
    assert style_meta["secondary_fill_background_lab_distance_min"] >= style_meta["min_fill_background_lab_distance_required"]
