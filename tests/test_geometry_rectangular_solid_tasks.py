"""Regression tests for rectangular-solid geometry tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import lookup_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.geometry.rectangular_solid.cube_edge_from_frame_length_value import (
    TASK_ID as TASK_ID_FRAME_EDGE,
    GeometryRectangularSolidCubeEdgeFromFrameLengthValueTask,
)
from trace_tasks.tasks.geometry.rectangular_solid.cuboid_surface_area_value import (
    QUERY_ID_SURFACE_AREA,
    TASK_ID as TASK_ID_SURFACE_AREA,
    GeometryRectangularSolidCuboidSurfaceAreaValueTask,
)
from trace_tasks.tasks.geometry.rectangular_solid.cuboid_volume_missing_dimension_value import (
    QUERY_ID_MISSING_HEIGHT,
    QUERY_ID_MISSING_LENGTH,
    QUERY_ID_MISSING_WIDTH,
    TASK_ID,
    TASK_ID as TASK_ID_MISSING_DIMENSION,
    GeometryRectangularSolidCuboidVolumeMissingDimensionValueTask,
)
from trace_tasks.tasks.geometry.rectangular_solid.open_box_net_dimension_value import (
    TASK_ID as TASK_ID_OPEN_BOX_NET,
    GeometryRectangularSolidOpenBoxNetDimensionValueTask,
)
from trace_tasks.tasks.geometry.rectangular_solid.shared.defaults import SCENE_ID


def _generate(seed: int, *, task_id: str = TASK_ID, **params):
    task = create_task(task_id)
    return task.generate(seed, params=dict(params), max_attempts=80)


def test_rectangular_solid_missing_dimension_registered() -> None:
    assert TASK_ID_MISSING_DIMENSION in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID_MISSING_DIMENSION] is GeometryRectangularSolidCuboidVolumeMissingDimensionValueTask
    assert TASK_ID_SURFACE_AREA in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID_SURFACE_AREA] is GeometryRectangularSolidCuboidSurfaceAreaValueTask
    assert TASK_ID_FRAME_EDGE in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID_FRAME_EDGE] is GeometryRectangularSolidCubeEdgeFromFrameLengthValueTask
    assert TASK_ID_OPEN_BOX_NET in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID_OPEN_BOX_NET] is GeometryRectangularSolidOpenBoxNetDimensionValueTask
    for task_id in (TASK_ID_MISSING_DIMENSION, TASK_ID_SURFACE_AREA, TASK_ID_FRAME_EDGE, TASK_ID_OPEN_BOX_NET):
        taxonomy = lookup_task_taxonomy(task_id)
        assert taxonomy.domain == "geometry"
        assert taxonomy.scene_id == SCENE_ID
        assert taxonomy.source_scene_id == "measurement"


@pytest.mark.parametrize(
    ("query_id", "expected_role", "expected_answer"),
    [
        (QUERY_ID_MISSING_LENGTH, "length", 3),
        (QUERY_ID_MISSING_WIDTH, "width", 4),
        (QUERY_ID_MISSING_HEIGHT, "height", 5),
    ],
)
def test_rectangular_solid_missing_dimension_formula_and_annotation(
    query_id: str,
    expected_role: str,
    expected_answer: int,
) -> None:
    out = _generate(
        20260625,
        query_id=query_id,
        dimensions=(3, 4, 5),
        volume_units=60,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == expected_answer == execution["answer"]
    assert execution["target_role"] == expected_role
    assert execution["length"] * execution["width"] * execution["height"] == execution["volume"] == 60
    assert execution["formula_family"] == "cuboid_volume_missing_dimension"

    assert out.annotation_gt.type == "segment"
    annotation = out.annotation_gt.value
    assert annotation == trace["render_map"]["dimension_segments"][expected_role]
    assert trace["projected_annotation"]["segment"] == annotation
    assert trace["projected_annotation"]["pixel_segment"] == annotation
    assert execution["annotation_roles"] == ["target_dimension"]
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "geometry_rectangular_solid_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_segment_inside_image(annotation, out.image.size)


def test_rectangular_solid_surface_area_formula_and_annotation() -> None:
    out = _generate(
        20260626,
        task_id=TASK_ID_SURFACE_AREA,
        query_id=QUERY_ID_SURFACE_AREA,
        dimensions=(3, 4, 5),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID_SURFACE_AREA
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 94 == execution["answer"]
    assert execution["length"] == 3
    assert execution["width"] == 4
    assert execution["height"] == 5
    assert execution["surface_area"] == 2 * ((3 * 4) + (3 * 5) + (4 * 5))
    assert execution["formula_family"] == "cuboid_surface_area"
    assert execution["target_role"] == "surface_area"

    assert out.annotation_gt.type == "bbox"
    annotation = out.annotation_gt.value
    assert annotation == trace["render_map"]["cuboid_bbox"]
    assert trace["projected_annotation"]["bbox"] == annotation
    assert trace["projected_annotation"]["pixel_bbox"] == annotation
    assert execution["annotation_roles"] == ["target_dimension"]
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "geometry_rectangular_solid_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_inside_image(annotation, out.image.size)


def test_rectangular_solid_cube_edge_from_frame_formula_and_annotation() -> None:
    out = _generate(
        20260627,
        task_id=TASK_ID_FRAME_EDGE,
        query_id=SINGLE_QUERY_ID,
        edge_units=8,
        highlighted_edge_count=5,
        highlighted_frame_length_units=40,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 8 == execution["answer"]
    assert execution["cube_edge"] == 8
    assert execution["visible_frame_edge_count"] == 5
    assert execution["frame_length"] == 40
    assert execution["frame_length"] == execution["cube_edge"] * execution["visible_frame_edge_count"]
    assert execution["formula_family"] == "cube_edge_from_frame_length"
    assert execution["target_role"] == "cube_edge"

    assert out.annotation_gt.type == "bbox"
    annotation = out.annotation_gt.value
    assert annotation == trace["render_map"]["annotation_bboxes"]["given_length_region_bbox"]
    assert trace["projected_annotation"]["bbox"] == annotation
    assert trace["projected_annotation"]["pixel_bbox"] == annotation
    assert execution["annotation_roles"] == ["highlighted_frame_region"]
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "geometry_rectangular_solid_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_inside_image(annotation, out.image.size)


@pytest.mark.parametrize(
    ("target_role", "expected_answer"),
    [
        ("base_length", 8),
        ("base_width", 4),
    ],
)
def test_rectangular_solid_open_box_dimension_formula_and_annotation(target_role: str, expected_answer: int) -> None:
    out = _generate(
        20260628,
        task_id=TASK_ID_OPEN_BOX_NET,
        query_id=SINGLE_QUERY_ID,
        sheet_length_units=14,
        sheet_width_units=10,
        cut_size_units=3,
        target_dimension_role=target_role,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == expected_answer == execution["answer"]
    assert execution["sheet_length"] == 14
    assert execution["sheet_width"] == 10
    assert execution["cut_size"] == 3
    assert execution["base_length"] == 8
    assert execution["base_width"] == 4
    assert execution["open_box_volume"] == 96
    assert execution["target_role"] == target_role
    assert execution["formula_family"] == "open_box_net_corner_cut"

    assert out.annotation_gt.type == "segment"
    annotation = out.annotation_gt.value
    assert annotation == trace["render_map"]["target_dimension_segment"]
    assert trace["projected_annotation"]["segment"] == annotation
    assert trace["projected_annotation"]["pixel_segment"] == annotation
    assert execution["annotation_roles"] == ["target_dimension"]
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "geometry_rectangular_solid_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_segment_inside_image(annotation, out.image.size)


def test_rectangular_solid_missing_dimension_generation_is_deterministic() -> None:
    params = {
        "query_id": QUERY_ID_MISSING_WIDTH,
        "length_units": 6,
        "width_units": 8,
        "height_units": 9,
        "volume_units": 432,
    }
    first = _generate(314161, **params)
    second = _generate(314161, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt.value == 8
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_rectangular_solid_surface_area_generation_is_deterministic() -> None:
    params = {
        "query_id": QUERY_ID_SURFACE_AREA,
        "length_units": 6,
        "width_units": 8,
        "height_units": 9,
        "surface_area_units": 348,
    }
    first = _generate(314162, task_id=TASK_ID_SURFACE_AREA, **params)
    second = _generate(314162, task_id=TASK_ID_SURFACE_AREA, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt.value == 348
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_rectangular_solid_cube_edge_from_frame_generation_is_deterministic() -> None:
    params = {
        "query_id": SINGLE_QUERY_ID,
        "edge_units": 9,
        "highlighted_edge_count": 6,
        "highlighted_frame_length_units": 54,
    }
    first = _generate(314163, task_id=TASK_ID_FRAME_EDGE, **params)
    second = _generate(314163, task_id=TASK_ID_FRAME_EDGE, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt.value == 9
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_rectangular_solid_open_box_generation_is_deterministic() -> None:
    params = {
        "query_id": SINGLE_QUERY_ID,
        "sheet_length_units": 15,
        "sheet_width_units": 11,
        "cut_size_units": 3,
        "target_dimension_role": "base_length",
    }
    first = _generate(314164, task_id=TASK_ID_OPEN_BOX_NET, **params)
    second = _generate(314164, task_id=TASK_ID_OPEN_BOX_NET, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt.value == 9
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_rectangular_solid_missing_dimension_rejects_invalid_params() -> None:
    for task_id in (TASK_ID_MISSING_DIMENSION, TASK_ID_SURFACE_AREA, TASK_ID_FRAME_EDGE, TASK_ID_OPEN_BOX_NET):
        task = create_task(task_id)
        with pytest.raises(ValueError):
            task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    for task_id in (TASK_ID_MISSING_DIMENSION, TASK_ID_SURFACE_AREA):
        task = create_task(task_id)
        with pytest.raises(ValueError):
            task.generate(1, params={"dimensions": (3, 4)}, max_attempts=1)
        with pytest.raises(ValueError):
            task.generate(1, params={"dimensions": (3, 4, 0)}, max_attempts=1)
        with pytest.raises(ValueError):
            task.generate(1, params={"length_units": 3, "width_units": 4}, max_attempts=1)
    with pytest.raises(ValueError):
        create_task(TASK_ID_MISSING_DIMENSION).generate(1, params={"dimensions": (3, 4, 5), "volume_units": 61}, max_attempts=1)
    with pytest.raises(ValueError):
        create_task(TASK_ID_SURFACE_AREA).generate(1, params={"dimensions": (3, 4, 5), "surface_area_units": 95}, max_attempts=1)
    with pytest.raises(ValueError):
        create_task(TASK_ID_FRAME_EDGE).generate(
            1,
            params={"query_id": "cube_edge_from_total_frame", "edge_units": 8, "total_frame_length_units": 95},
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        create_task(TASK_ID_FRAME_EDGE).generate(
            1,
            params={
                "query_id": SINGLE_QUERY_ID,
                "edge_units": 8,
                "highlighted_edge_count": 5,
                "highlighted_frame_length_units": 41,
            },
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        create_task(TASK_ID_FRAME_EDGE).generate(
            1,
            params={"query_id": SINGLE_QUERY_ID, "edge_units": 8, "highlighted_edge_count": 9},
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        create_task(TASK_ID_OPEN_BOX_NET).generate(
            1,
            params={
                "query_id": SINGLE_QUERY_ID,
                "sheet_length_units": 6,
                "sheet_width_units": 8,
                "cut_size_units": 3,
                "target_dimension_role": "base_length",
            },
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        create_task(TASK_ID_OPEN_BOX_NET).generate(
            1,
            params={
                "query_id": "open_box_volume_from_net",
                "sheet_length_units": 14,
                "sheet_width_units": 10,
                "cut_size_units": 3,
                "open_box_volume_units": 95,
            },
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        create_task(TASK_ID_OPEN_BOX_NET).generate(
            1,
            params={
                "query_id": SINGLE_QUERY_ID,
                "sheet_length_units": 14,
                "sheet_width_units": 10,
                "cut_size_units": 3,
                "target_dimension_role": "height",
            },
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        create_task(TASK_ID_OPEN_BOX_NET).generate(
            1,
            params={
                "query_id": "open_box_volume_from_net",
                "sheet_length_units": 14,
                "sheet_width_units": 10,
                "cut_size_units": 3,
                "target_dimension_role": "base_length",
            },
            max_attempts=1,
        )


def _assert_segment_inside_image(annotation: list[list[float]], image_size: tuple[int, int]) -> None:
    width, height = image_size
    assert isinstance(annotation, list)
    assert len(annotation) == 2
    for point in annotation:
        assert isinstance(point, list)
        assert len(point) == 2
        x, y = [float(value) for value in point]
        assert 0.0 <= x <= float(width)
        assert 0.0 <= y <= float(height)


def _assert_bbox_inside_image(annotation: list[float], image_size: tuple[int, int]) -> None:
    width, height = image_size
    assert isinstance(annotation, list)
    assert len(annotation) == 4
    x1, y1, x2, y2 = [float(value) for value in annotation]
    assert 0.0 <= x1 < x2 <= float(width)
    assert 0.0 <= y1 < y2 <= float(height)
