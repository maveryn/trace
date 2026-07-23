"""Regression tests for volume-equivalence-conversion geometry tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.core.taxonomy import lookup_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.geometry.volume_equivalence_conversion.equal_volume_option_label import (
    QUERY_ID_CONE_MATCHES_CYLINDER_OPTION,
    QUERY_ID_CUBOID_MATCHES_CYLINDER_OPTION,
    QUERY_ID_CYLINDER_MATCHES_CONE_OPTION,
    TASK_ID_EQUAL_VOLUME_OPTION,
    GeometryVolumeEquivalenceConversionEqualVolumeOptionLabelTask,
)
from trace_tasks.tasks.geometry.volume_equivalence_conversion.missing_dimension_value import (
    MISSING_DIMENSION_ANNOTATION_KEYS,
    QUERY_ID_CUBOID_TO_CYLINDER_LENGTH,
    SCENE_ID,
    TASK_ID_MISSING_DIMENSION,
    GeometryVolumeEquivalenceConversionMissingDimensionValueTask,
)


def _generate(seed: int, *, task_id: str = TASK_ID_MISSING_DIMENSION, **params):
    task = create_task(task_id)
    return task.generate(seed, params=dict(params), max_attempts=80)


def test_volume_equivalence_conversion_tasks_registered() -> None:
    assert TASK_ID_MISSING_DIMENSION in TASK_REGISTRY
    assert TASK_ID_EQUAL_VOLUME_OPTION in TASK_REGISTRY
    assert (
        TASK_REGISTRY[TASK_ID_MISSING_DIMENSION]
        is GeometryVolumeEquivalenceConversionMissingDimensionValueTask
    )
    assert (
        TASK_REGISTRY[TASK_ID_EQUAL_VOLUME_OPTION]
        is GeometryVolumeEquivalenceConversionEqualVolumeOptionLabelTask
    )
    for task_id in (TASK_ID_MISSING_DIMENSION, TASK_ID_EQUAL_VOLUME_OPTION):
        taxonomy = lookup_task_taxonomy(task_id)
        assert taxonomy is not None
        assert taxonomy.domain == "geometry"
        assert taxonomy.scene_id == SCENE_ID
        assert taxonomy.source_scene_id == "measurement"


def test_cuboid_to_cylinder_missing_dimension_formula_and_annotation() -> None:
    out = _generate(
        20260710,
        query_id=QUERY_ID_CUBOID_TO_CYLINDER_LENGTH,
        conversion_case=(6, 4, 5, 12),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID_CUBOID_TO_CYLINDER_LENGTH
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 10 == execution["answer"]
    assert execution["source_shape"] == "cuboid"
    assert execution["target_shape"] == "cylinder"
    assert execution["source_volume"] == 6 * 4 * 5
    assert execution["target_volume"] == execution["source_volume"]
    assert execution["formula_family"] == "volume_equivalence_missing_dimension"

    assert out.annotation_gt.type == "bbox_map"
    annotation = out.annotation_gt.value
    assert tuple(annotation.keys()) == MISSING_DIMENSION_ANNOTATION_KEYS
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_map"] == annotation
    assert (
        trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"]
        == "geometry_volume_equivalence_conversion_v1"
    )
    assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_map_inside_image(
        annotation, out.image.size, keys=MISSING_DIMENSION_ANNOTATION_KEYS
    )


def test_equal_volume_option_formula_and_annotation() -> None:
    out = _generate(
        20260711,
        task_id=TASK_ID_EQUAL_VOLUME_OPTION,
        query_id=QUERY_ID_CONE_MATCHES_CYLINDER_OPTION,
        conversion_case=(18, 6),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == QUERY_ID_CONE_MATCHES_CYLINDER_OPTION
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D", "E", "F"}
    assert execution["answer"] == out.answer_gt.value
    assert execution["source_shape"] == "cone"
    assert execution["target_shape"] == "cylinder"
    assert execution["target_volume"] == execution["source_volume"]
    assert execution["selected_option_label"] == out.answer_gt.value
    assert execution["formula_family"] == "volume_equivalence_option_match"

    assert out.annotation_gt.type == "bbox"
    annotation = out.annotation_gt.value
    assert trace["projected_annotation"]["bbox"] == annotation
    assert trace["projected_annotation"]["pixel_bbox"] == annotation
    assert (
        trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"]
        == "geometry_volume_equivalence_conversion_v1"
    )
    assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert "task_variant" not in json.dumps(trace)
    _assert_bbox_inside_image(annotation, out.image.size)


def test_equal_volume_option_has_unique_matching_option() -> None:
    task = create_task(TASK_ID_EQUAL_VOLUME_OPTION)
    for seed, query_id in enumerate(
        (
            QUERY_ID_CONE_MATCHES_CYLINDER_OPTION,
            QUERY_ID_CYLINDER_MATCHES_CONE_OPTION,
            QUERY_ID_CUBOID_MATCHES_CYLINDER_OPTION,
        ),
        start=91000,
    ):
        for offset in range(12):
            out = task.generate(
                seed + offset,
                params={"query_id": query_id, "option_count": 6},
                max_attempts=80,
            )
            trace = out.trace_payload
            source_volume = trace["execution_trace"]["source_volume"]
            matching = [
                option
                for option in trace["execution_trace"]["options"]
                if option["volume"] == source_volume
            ]
            assert len(matching) == 1
            assert matching[0]["label"] == out.answer_gt.value


def test_volume_equivalence_conversion_generation_is_deterministic() -> None:
    params = {
        "query_id": QUERY_ID_CUBOID_TO_CYLINDER_LENGTH,
        "conversion_case": (8, 5, 3, 10),
    }
    first = _generate(314190, **params)
    second = _generate(314190, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt.value == 12
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert (
        first.trace_payload["execution_trace"]
        == second.trace_payload["execution_trace"]
    )
    assert first.image.tobytes() == second.image.tobytes()


def test_volume_equivalence_conversion_rejects_invalid_params() -> None:
    task = create_task(TASK_ID_MISSING_DIMENSION)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(
            1,
            params={
                "query_id": QUERY_ID_CUBOID_TO_CYLINDER_LENGTH,
                "conversion_case": (6, 4, 5, 7),
            },
            max_attempts=1,
        )
    option_task = create_task(TASK_ID_EQUAL_VOLUME_OPTION)
    with pytest.raises(ValueError):
        option_task.generate(
            1, params={"query_id": QUERY_ID_CUBOID_TO_CYLINDER_LENGTH}, max_attempts=1
        )


def _assert_bbox_map_inside_image(
    annotation: dict[str, list[float]],
    image_size: tuple[int, int],
    *,
    keys: tuple[str, ...],
) -> None:
    width, height = image_size
    for key in keys:
        bbox = annotation[key]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)


def _assert_bbox_inside_image(
    bbox: list[float],
    image_size: tuple[int, int],
) -> None:
    width, height = image_size
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0.0 <= x0 < x1 <= float(width)
    assert 0.0 <= y0 < y1 <= float(height)
