"""Regression tests for survey-traverse geometry tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.core.taxonomy import lookup_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.geometry.survey_traverse.outgoing_bearing_from_turn_value import (
    TASK_ID as TASK_ID_OUTGOING_BEARING,
    GeometrySurveyTraverseOutgoingBearingFromTurnValueTask,
)
from trace_tasks.tasks.geometry.survey_traverse.shared.state import SCENE_ID
from trace_tasks.tasks.geometry.survey_traverse.station_elevation_value import (
    SUPPORTED_QUERY_IDS as ELEVATION_QUERY_IDS,
    TASK_ID as TASK_ID_STATION_ELEVATION,
    GeometrySurveyTraverseStationElevationValueTask,
)
from trace_tasks.tasks.geometry.survey_traverse.traverse_area_value import (
    SUPPORTED_QUERY_IDS as AREA_QUERY_IDS,
    TASK_ID as TASK_ID_TRAVERSE_AREA,
    GeometrySurveyTraverseTraverseAreaValueTask,
)


ACTIVE_TASK_IDS = (
    TASK_ID_OUTGOING_BEARING,
    TASK_ID_STATION_ELEVATION,
    TASK_ID_TRAVERSE_AREA,
)


def _generate(seed: int, *, task_id: str = TASK_ID_OUTGOING_BEARING, **params):
    task = create_task(task_id)
    return task.generate(seed, params=dict(params), max_attempts=20)


def test_survey_traverse_registered_public_tasks() -> None:
    assert TASK_ID_OUTGOING_BEARING in TASK_REGISTRY
    assert TASK_ID_STATION_ELEVATION in TASK_REGISTRY
    assert TASK_ID_TRAVERSE_AREA in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID_OUTGOING_BEARING] is GeometrySurveyTraverseOutgoingBearingFromTurnValueTask
    assert TASK_REGISTRY[TASK_ID_STATION_ELEVATION] is GeometrySurveyTraverseStationElevationValueTask
    assert TASK_REGISTRY[TASK_ID_TRAVERSE_AREA] is GeometrySurveyTraverseTraverseAreaValueTask
    assert "task_geometry__survey_traverse__forward_bearing_from_back_bearing_value" not in TASK_REGISTRY

    for task_id in ACTIVE_TASK_IDS:
        taxonomy = lookup_task_taxonomy(task_id)
        assert taxonomy is not None
        assert taxonomy.domain == "geometry"
        assert taxonomy.scene_id == SCENE_ID


def test_closed_traverse_contract_and_formula() -> None:
    out = _generate(
        20260612,
        task_id=TASK_ID_OUTGOING_BEARING,
        query_id="single",
        base_bearing=100,
        turn_angle=45,
        turn_direction="right",
        station_labels=("A", "B", "C"),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 145
    assert execution["formula_family"] == "survey_outgoing_bearing_from_turn"
    assert execution["known_bearing"] == 100
    assert execution["turn_angle"] == 45
    assert execution["turn_direction"] == "right"
    assert execution["target_bearing"] == 145

    assert out.annotation_gt.type == "bbox_map"
    assert tuple(out.annotation_gt.value) == ("turn_diagram", "field_note_region")
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
    _assert_bbox_map_inside_image(out.annotation_gt.value, out.image.size)
    assert "task_variant" not in json.dumps(trace)


def test_leveling_station_elevation_contract_and_formula() -> None:
    out = _generate(
        20260614,
        task_id=TASK_ID_STATION_ELEVATION,
        elevation_case=(120, 4, 7),
        station_labels=("A", "B", "C"),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert execution["internal_query_id"] == "leveling_station_elevation"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 117
    assert execution["reference_elevation"] == 120
    assert execution["backsight"] == 4
    assert execution["foresight"] == 7
    assert execution["height_of_instrument"] == 124
    assert execution["target_elevation"] == 117
    assert execution["formula_family"] == "survey_leveling_station_elevation"

    assert out.annotation_gt.type == "bbox_map"
    assert tuple(out.annotation_gt.value) == ("station_profile", "field_note_region")
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
    _assert_bbox_map_inside_image(out.annotation_gt.value, out.image.size)
    assert "task_variant" not in json.dumps(trace)


def test_offset_trapezoid_area_contract_and_formula() -> None:
    out = _generate(
        20260618,
        task_id=TASK_ID_TRAVERSE_AREA,
        area_case=(20, 40, 60, 3, 5, 4, 2),
        station_labels=("A", "B", "C"),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert execution["internal_query_id"] == "offset_trapezoid_area"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 230
    assert execution["formula_family"] == "survey_offset_trapezoid_area"
    assert execution["chainages"] == [0, 20, 40, 60]
    assert execution["offsets"] == [3, 5, 4, 2]
    assert execution["answer"] == 230

    assert out.annotation_gt.type == "bbox_map"
    assert tuple(out.annotation_gt.value) == ("traverse_region", "field_note_region")
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
    _assert_bbox_map_inside_image(out.annotation_gt.value, out.image.size)
    assert "task_variant" not in json.dumps(trace)


@pytest.mark.parametrize("task_id", ACTIVE_TASK_IDS)
def test_survey_traverse_generation_is_deterministic(task_id: str) -> None:
    params = {"query_id": create_task(task_id).supported_query_ids[0]}
    first = _generate(20260619, task_id=task_id, **params)
    second = _generate(20260619, task_id=task_id, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_survey_traverse_bearing_rejects_invalid_params() -> None:
    outgoing_task = create_task(TASK_ID_OUTGOING_BEARING)
    with pytest.raises(ValueError):
        outgoing_task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        outgoing_task.generate(
            1,
            params={"query_id": "single", "turn_direction": "clockwise"},
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        outgoing_task.generate(1, params={"station_labels": ("A", "A", "B")}, max_attempts=1)


def test_survey_traverse_station_elevation_rejects_invalid_params() -> None:
    task = create_task(TASK_ID_STATION_ELEVATION)
    assert ELEVATION_QUERY_IDS == ("leveling_station_elevation",)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(
            1,
            params={"query_id": "single", "elevation_case": (1, 2, 3)},
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "slope_distance_elevation_change"}, max_attempts=1)


def test_survey_traverse_area_rejects_invalid_params() -> None:
    task = create_task(TASK_ID_TRAVERSE_AREA)
    assert AREA_QUERY_IDS == ("offset_trapezoid_area",)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "coordinate_traverse_area"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(
            1,
            params={"query_id": "single", "area_case": "20,40,60,3,5,4,2"},
            max_attempts=1,
        )


def _assert_bbox_map_inside_image(annotation: dict[str, list[float]], image_size: tuple[int, int]) -> None:
    width, height = image_size
    for bbox in annotation.values():
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)
