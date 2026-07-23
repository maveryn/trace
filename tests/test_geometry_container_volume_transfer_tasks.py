"""Regression tests for container-volume-transfer geometry tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.core.taxonomy import lookup_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.geometry.container_volume_transfer.fill_count_value import (
    ANNOTATION_KEYS,
    QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT,
    QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT,
    TASK_ID,
    TASK_ID_FILL_COUNT,
    GeometryContainerVolumeTransferFillCountValueTask,
)
from trace_tasks.tasks.geometry.container_volume_transfer.shared.defaults import SCENE_ID
from trace_tasks.tasks.geometry.container_volume_transfer.resulting_height_value import (
    GeometryContainerVolumeTransferResultingHeightValueTask,
    QUERY_ID_CONE_POURS_TO_CYLINDER_HEIGHT,
    QUERY_ID_CYLINDER_POURS_TO_CUBOID_HEIGHT,
    RESULTING_HEIGHT_ANNOTATION_KEYS,
    TASK_ID_RESULTING_HEIGHT,
)


def _generate(seed: int, *, task_id: str = TASK_ID_FILL_COUNT, **params):
    task = create_task(task_id)
    return task.generate(seed, params=dict(params), max_attempts=80)


def test_container_volume_transfer_fill_count_registered() -> None:
    assert TASK_ID_FILL_COUNT in TASK_REGISTRY
    assert TASK_ID_RESULTING_HEIGHT in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID_FILL_COUNT] is GeometryContainerVolumeTransferFillCountValueTask
    assert TASK_REGISTRY[TASK_ID_RESULTING_HEIGHT] is GeometryContainerVolumeTransferResultingHeightValueTask
    assert TASK_ID == TASK_ID_FILL_COUNT
    taxonomy = lookup_task_taxonomy(TASK_ID_FILL_COUNT)
    assert taxonomy is not None
    assert taxonomy.domain == "geometry"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == "measurement"
    height_taxonomy = lookup_task_taxonomy(TASK_ID_RESULTING_HEIGHT)
    assert height_taxonomy is not None
    assert height_taxonomy.domain == "geometry"
    assert height_taxonomy.scene_id == SCENE_ID
    assert height_taxonomy.source_scene_id == "measurement"


def test_cone_to_cylinder_fill_count_formula_and_annotation() -> None:
    out = _generate(
        20260630,
        query_id=QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT,
        transfer_case=(12, 6, 8, 6),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 2 == execution["answer"]
    assert execution["source_shape"] == "cone"
    assert execution["target_shape"] == "cylinder"
    assert execution["source_volume"] == (12 * 6) // 3
    assert execution["target_volume"] == 8 * 6
    assert execution["fill_count"] == execution["target_volume"] // execution["source_volume"]
    assert execution["formula_family"] == "container_volume_transfer_fill_count"

    assert out.annotation_gt.type == "bbox_map"
    annotation = out.annotation_gt.value
    assert tuple(annotation.keys()) == ANNOTATION_KEYS
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_map"] == annotation
    assert trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_container_volume_transfer_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_map_inside_image(annotation, out.image.size)


def test_cone_pours_to_cylinder_resulting_height_formula_and_annotation() -> None:
    out = _generate(
        20260632,
        task_id=TASK_ID_RESULTING_HEIGHT,
        query_id=QUERY_ID_CONE_POURS_TO_CYLINDER_HEIGHT,
        transfer_case=(15, 6, 12, 10, 3),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID_CONE_POURS_TO_CYLINDER_HEIGHT
    assert out.answer_gt.type == "number"
    assert out.answer_gt.value == 7.5 == execution["answer"]
    assert execution["source_shape"] == "cone"
    assert execution["target_shape"] == "cylinder"
    assert execution["source_volume"] == (15 * 6) // 3
    assert execution["target_base_area"] == 12
    assert execution["pour_count"] == 3
    assert execution["resulting_height"] == 7.5
    assert execution["formula_family"] == "container_volume_transfer_resulting_height"

    assert out.annotation_gt.type == "bbox_map"
    annotation = out.annotation_gt.value
    assert tuple(annotation.keys()) == RESULTING_HEIGHT_ANNOTATION_KEYS
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_map"] == annotation
    assert trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_container_volume_transfer_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_map_inside_image(annotation, out.image.size, keys=RESULTING_HEIGHT_ANNOTATION_KEYS)


def test_cylinder_pours_to_cuboid_resulting_height_formula_and_annotation() -> None:
    out = _generate(
        20260633,
        task_id=TASK_ID_RESULTING_HEIGHT,
        query_id=QUERY_ID_CYLINDER_POURS_TO_CUBOID_HEIGHT,
        transfer_case=(8, 5, 10, 5, 9, 3),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID_CYLINDER_POURS_TO_CUBOID_HEIGHT
    assert out.answer_gt.type == "number"
    assert out.answer_gt.value == 2.4 == execution["answer"]
    assert execution["source_shape"] == "cylinder"
    assert execution["target_shape"] == "cuboid"
    assert execution["source_volume"] == 8 * 5
    assert execution["target_base_area"] == 10 * 5
    assert execution["pour_count"] == 3
    assert execution["resulting_height"] == 2.4
    assert execution["formula_family"] == "container_volume_transfer_resulting_height"

    assert out.annotation_gt.type == "bbox_map"
    annotation = out.annotation_gt.value
    assert tuple(annotation.keys()) == RESULTING_HEIGHT_ANNOTATION_KEYS
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_container_volume_transfer_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_map_inside_image(annotation, out.image.size, keys=RESULTING_HEIGHT_ANNOTATION_KEYS)


def test_cylinder_to_cuboid_fill_count_formula_and_annotation() -> None:
    out = _generate(
        20260631,
        query_id=QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT,
        transfer_case=(8, 4, 8, 4, 3),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 3 == execution["answer"]
    assert execution["source_shape"] == "cylinder"
    assert execution["target_shape"] == "cuboid"
    assert execution["source_volume"] == 8 * 4
    assert execution["target_volume"] == 8 * 4 * 3
    assert execution["fill_count"] == execution["target_volume"] // execution["source_volume"]
    assert execution["formula_family"] == "container_volume_transfer_fill_count"

    assert out.annotation_gt.type == "bbox_map"
    annotation = out.annotation_gt.value
    assert tuple(annotation.keys()) == ANNOTATION_KEYS
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_container_volume_transfer_v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_map_inside_image(annotation, out.image.size)


def test_container_volume_transfer_generation_is_deterministic() -> None:
    params = {
        "query_id": QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT,
        "transfer_case": (10, 3, 10, 4, 3),
    }
    first = _generate(314170, **params)
    second = _generate(314170, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt.value == 4
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_container_volume_transfer_rejects_invalid_params() -> None:
    task = create_task(TASK_ID_FILL_COUNT)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT, "transfer_case": (12, 6, 8)}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT, "transfer_case": (8, 4, 8, 4)}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT, "transfer_case": (10, 5, 8, 6)}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": QUERY_ID_CYLINDER_TO_CUBOID_FILL_COUNT, "transfer_case": (8, 4, 7, 4, 3)}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(
            1,
            params={
                "query_id": QUERY_ID_CONE_TO_CYLINDER_FILL_COUNT,
                "transfer_case": (12, 6, 8, 6),
                "fill_count": 3,
            },
            max_attempts=1,
        )
    height_task = create_task(TASK_ID_RESULTING_HEIGHT)
    with pytest.raises(ValueError):
        height_task.generate(1, params={"query_id": QUERY_ID_CONE_POURS_TO_CYLINDER_HEIGHT, "transfer_case": (12, 6, 8, 9)}, max_attempts=1)
    with pytest.raises(ValueError):
        height_task.generate(
            1,
            params={"query_id": QUERY_ID_CYLINDER_POURS_TO_CUBOID_HEIGHT, "transfer_case": (8, 5, 10, 5, 9)},
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        height_task.generate(
            1,
            params={"query_id": QUERY_ID_CONE_POURS_TO_CYLINDER_HEIGHT, "transfer_case": (12, 6, 8, 9, 1), "pour_count": 2},
            max_attempts=1,
        )
def _assert_bbox_map_inside_image(
    annotation: dict[str, list[float]],
    image_size: tuple[int, int],
    *,
    keys: tuple[str, ...] = ANNOTATION_KEYS,
) -> None:
    width, height = image_size
    for key in keys:
        bbox = annotation[key]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)
