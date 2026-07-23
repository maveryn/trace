"""Contracts for curvilinear composite geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.composite_shape.missing_width_from_semicircle_area import (
    GeometryMissingWidthFromSemicircleAreaTask,
)
from trace_tasks.tasks.geometry.composite_shape.rectangle_quarter_sector_cutout_area import GeometryRectangleQuarterSectorCutoutAreaTask
from trace_tasks.tasks.geometry.composite_shape.shared.defaults import SCENE_ID
from trace_tasks.tasks.geometry.composite_shape.shared.measurements import round1, semicircle_arc_length
from trace_tasks.tasks.geometry.composite_shape.rectangle_quarter_sector_cutout_perimeter import GeometryRectangleQuarterSectorCutoutPerimeterTask
from trace_tasks.tasks.geometry.composite_shape.rectangle_semicircle_area import GeometryRectangleSemicircleAreaTask
from trace_tasks.tasks.geometry.composite_shape.rectangle_semicircle_perimeter import (
    GeometryRectangleSemicirclePerimeterTask,
)
from trace_tasks.tasks.geometry.composite_shape.sector_angle_value import GeometrySectorAngleValueTask


TASK_CLASSES = (
    GeometryRectangleSemicircleAreaTask,
    GeometryRectangleQuarterSectorCutoutAreaTask,
    GeometryRectangleSemicirclePerimeterTask,
    GeometryRectangleQuarterSectorCutoutPerimeterTask,
    GeometryMissingWidthFromSemicircleAreaTask,
    GeometrySectorAngleValueTask,
)

QUERY_IDS_BY_TASK = {
    GeometryRectangleSemicircleAreaTask: ("cap_area", "cutout_area"),
    GeometryRectangleQuarterSectorCutoutAreaTask: ("single",),
    GeometryRectangleSemicirclePerimeterTask: ("cap_perimeter", "cutout_perimeter"),
    GeometryRectangleQuarterSectorCutoutPerimeterTask: ("single",),
    GeometryMissingWidthFromSemicircleAreaTask: ("cap_from_total_area", "cutout_from_total_area"),
    GeometrySectorAngleValueTask: ("from_arc_length", "from_sector_area"),
}

EXPECTED_ANNOTATION_KEYS_BY_TASK = {
    GeometryRectangleSemicircleAreaTask: {"A", "B", "C", "D", "O"},
    GeometryRectangleQuarterSectorCutoutAreaTask: {"A", "B", "C", "D", "E", "F"},
    GeometryRectangleSemicirclePerimeterTask: {"A", "B", "C", "D", "E", "F", "O"},
    GeometryRectangleQuarterSectorCutoutPerimeterTask: {"A", "B", "C", "D", "E", "F"},
    GeometryMissingWidthFromSemicircleAreaTask: {"A", "B", "C", "D", "O"},
    GeometrySectorAngleValueTask: {"O", "A", "B"},
}


def _assert_bbox_inside(bbox, *, width: int, height: int) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0.0 <= x0 <= float(width)
    assert 0.0 <= y0 <= float(height)
    assert 0.0 <= x1 <= float(width)
    assert 0.0 <= y1 <= float(height)
    assert x0 <= x1
    assert y0 <= y1


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_curvilinear_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(54001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == EXPECTED_ANNOTATION_KEYS_BY_TASK[task_cls]
    assert '"annotation"' in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["projected_annotation"]["type"] == out.annotation_gt.type


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_curvilinear_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(54011, params=params, max_attempts=20)
    out_b = task.generate(54011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_curvilinear_tasks_support_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            54021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == "number"
        probabilities = out.trace_payload["query_spec"]["params"]["query_id_probabilities"]
        assert probabilities[query_id] == 1.0
        assert set(probabilities) == set(QUERY_IDS_BY_TASK[task_cls])
        assert sum(float(value) for value in probabilities.values()) == 1.0


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_curvilinear_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            54041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        assert out.annotation_gt.type == "point_map"
        assert set(out.annotation_gt.value) == EXPECTED_ANNOTATION_KEYS_BY_TASK[task_cls]
        for x, y in out.annotation_gt.value.values():
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_curvilinear_tasks_support_forced_scene_rotation(task_cls) -> None:
    task = task_cls()
    base_params = {
        "scene_rotation_degrees": 35,
        "width_units": 10,
        "height_units": 12,
        "radius_units": 3,
        "theta_degrees": 80,
    }
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            54051 + index,
            params={**base_params, "query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        rotation = out.trace_payload["render_spec"]["single_object_scene_rotation"]

        assert rotation["enabled"] is True
        assert rotation["applied"] is True
        assert rotation["angle_degrees"] == pytest.approx(35.0)
        assert rotation["applied_before_annotation_projection"] is True
        assert set(out.annotation_gt.value) == EXPECTED_ANNOTATION_KEYS_BY_TASK[task_cls]
        for x, y in out.annotation_gt.value.values():
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)

        render_map = out.trace_payload["render_map"]
        for bbox_key in ("target_bbox", "curved_component_bbox", "sector_bbox", "center_marker_bbox"):
            if bbox_key in render_map:
                _assert_bbox_inside(render_map[bbox_key], width=width, height=height)
        for bbox_list_key in ("support_bboxes",):
            for bbox in render_map.get(bbox_list_key, []):
                _assert_bbox_inside(bbox, width=width, height=height)
        for bbox_map_key in ("point_label_bboxes", "visual_notation_bboxes"):
            for bbox in render_map.get(bbox_map_key, {}).values():
                _assert_bbox_inside(bbox, width=width, height=height)


def test_curvilinear_tasks_reject_unknown_query_id() -> None:
    task = GeometryRectangleSemicircleAreaTask()
    with pytest.raises(ValueError):
        task.generate(54031, params={"query_id": "not_a_query"}, max_attempts=20)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_curvilinear_public_annotation_avoids_measurement_label_boxes(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(54061 + index, params={"query_id": query_id}, max_attempts=20)
        assert out.trace_payload["projected_annotation"]["type"] == out.annotation_gt.type
        assert out.trace_payload["projected_annotation"]["point_map"] == out.annotation_gt.value
        assert out.trace_payload["projected_annotation"]["pixel_point_map"] == out.annotation_gt.value
        assert set(out.annotation_gt.value) == set(out.trace_payload["execution_trace"]["annotation_roles"])
        assert set(out.annotation_gt.value) == EXPECTED_ANNOTATION_KEYS_BY_TASK[task_cls]
        assert all("label" not in str(role) for role in out.trace_payload["execution_trace"]["annotation_roles"])
        assert "support_roles" in out.trace_payload["render_map"]
        assert set(out.trace_payload["render_map"]["point_label_bboxes"]) == set(out.annotation_gt.value)


def test_quarter_sector_area_omits_obvious_right_angle_label() -> None:
    task = GeometryRectangleQuarterSectorCutoutAreaTask()
    out = task.generate(
        54091,
        params={"query_id": "single"},
        max_attempts=20,
    )

    assert "angle_label" not in out.trace_payload["render_map"]["support_roles"]
    assert set(out.trace_payload["render_map"]["visual_notation_bboxes"]) == {
        "top_left_right_angle",
        "bottom_left_right_angle",
        "bottom_right_right_angle",
        "quarter_sector_right_angle",
        "original_top_extension_guide",
        "original_right_extension_guide",
    }
    assert "quarter_sector_right_angle" not in out.annotation_gt.value


@pytest.mark.parametrize(
    "task_cls, query_id",
    (
        (GeometryRectangleSemicircleAreaTask, "cap_area"),
        (GeometryRectangleSemicircleAreaTask, "cutout_area"),
        (GeometryRectangleSemicirclePerimeterTask, "cap_perimeter"),
        (GeometryRectangleSemicirclePerimeterTask, "cutout_perimeter"),
        (GeometryMissingWidthFromSemicircleAreaTask, "cap_from_total_area"),
        (GeometryMissingWidthFromSemicircleAreaTask, "cutout_from_total_area"),
    ),
)
def test_semicircle_rectangle_variants_show_all_four_right_angle_markers(task_cls, query_id) -> None:
    task = task_cls()
    out = task.generate(
        54095,
        params={"query_id": query_id, "width_units": 12, "height_units": 14, "radius_units": 4},
        max_attempts=20,
    )

    assert set(out.trace_payload["render_map"]["visual_notation_bboxes"]) >= {
        "top_left_right_angle",
        "top_right_right_angle",
        "bottom_right_right_angle",
        "bottom_left_right_angle",
    }


@pytest.mark.parametrize(
    "query_id",
    (
        "cap_perimeter",
        "cutout_perimeter",
    ),
)
def test_semicircle_perimeter_includes_right_side_remainders(query_id) -> None:
    task = GeometryRectangleSemicirclePerimeterTask()
    width_units = 12
    height_units = 14
    radius_units = 4
    out = task.generate(
        54097,
        params={
            "query_id": query_id,
            "width_units": width_units,
            "height_units": height_units,
            "radius_units": radius_units,
        },
        max_attempts=20,
    )
    expected = round1(
        (2.0 * float(width_units))
        + (2.0 * float(height_units))
        - (2.0 * float(radius_units))
        + semicircle_arc_length(radius_units)
    )

    assert out.answer_gt.value == expected
    assert out.trace_payload["execution_trace"]["straight_boundary_length"] == round1(
        (2.0 * float(width_units))
        + (2.0 * float(height_units))
        - (2.0 * float(radius_units))
    )


def test_curvilinear_perimeter_omits_derived_boundary_total_labels() -> None:
    for task_cls in (
        GeometryRectangleSemicirclePerimeterTask,
        GeometryRectangleQuarterSectorCutoutPerimeterTask,
    ):
        task = task_cls()
        for query_id in QUERY_IDS_BY_TASK[task_cls]:
            out = task.generate(
                54101,
                params={"query_id": query_id},
                max_attempts=20,
            )
            support_roles = set(out.trace_payload["render_map"]["support_roles"])

            assert "arc_length_label" in support_roles
            assert "straight_boundary_length_label" not in support_roles
            assert "perimeter_total_label" not in support_roles


def test_curvilinear_perimeter_prompts_name_curve_type() -> None:
    expected_terms = {
        GeometryRectangleSemicirclePerimeterTask: ("semicircle",),
        GeometryRectangleQuarterSectorCutoutPerimeterTask: ("quarter", "circle"),
    }

    for task_cls, terms in expected_terms.items():
        task = task_cls()
        for query_id in QUERY_IDS_BY_TASK[task_cls]:
            out = task.generate(
                54111,
                params={"query_id": query_id},
                max_attempts=20,
            )
            prompt = str(out.prompt).lower()

            assert all(term in prompt for term in terms)
