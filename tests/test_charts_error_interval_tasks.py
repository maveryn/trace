"""Behavior tests for error interval chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.charts.error_interval.interval_width_rank_label import ChartsErrorIntervalRelationLabelTask
from trace_tasks.tasks.charts.error_interval.reference_containment_count import ChartsErrorIntervalReferenceContainmentCountTask
from trace_tasks.tasks.charts.error_interval.reference_exclusion_side_count import ChartsErrorIntervalReferenceExclusionSideCountTask
from trace_tasks.tasks.charts.error_interval.shared.defaults import SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID


TASK_CASES = (
    (ChartsErrorIntervalReferenceContainmentCountTask, ChartsErrorIntervalReferenceContainmentCountTask.supported_query_ids, "integer", "segment_set"),
    (ChartsErrorIntervalReferenceExclusionSideCountTask, ChartsErrorIntervalReferenceExclusionSideCountTask.supported_query_ids, "integer", "segment_set"),
    (ChartsErrorIntervalRelationLabelTask, ChartsErrorIntervalRelationLabelTask.supported_query_ids, "string", "segment"),
)


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    assert 0 <= float(point[0]) <= width
    assert 0 <= float(point[1]) <= height


def _assert_segment_inside_canvas(segment: list[list[float]], *, width: int, height: int) -> None:
    assert len(segment) == 2
    for point in segment:
        _assert_point_inside_canvas(list(point), width=width, height=height)


def _expected_answer(execution: dict, query_id: str) -> int | str:
    items = list(execution["items"])
    if query_id == DEFAULT_QUERY_ID:
        reference = int(execution["reference_value"])
        return sum(1 for item in items if int(item["lower"]) <= reference <= int(item["upper"]))
    if query_id == "entirely_above_reference_count":
        reference = int(execution["reference_value"])
        return sum(1 for item in items if int(item["lower"]) > reference)
    if query_id == "entirely_below_reference_count":
        reference = int(execution["reference_value"])
        return sum(1 for item in items if int(item["upper"]) < reference)
    if query_id == "widest_interval_label":
        return max(items, key=lambda item: int(item["upper"]) - int(item["lower"]))["label"]
    if query_id == "narrowest_interval_label":
        return min(items, key=lambda item: int(item["upper"]) - int(item["lower"]))["label"]
    if query_id == "second_widest_interval_label":
        return sorted(items, key=lambda item: int(item["upper"]) - int(item["lower"]), reverse=True)[1]["label"]
    if query_id == "second_narrowest_interval_label":
        return sorted(items, key=lambda item: int(item["upper"]) - int(item["lower"]))[1]["label"]
    raise AssertionError(f"unsupported query_id: {query_id}")


@pytest.mark.parametrize(("task_cls", "query_ids", "answer_type", "annotation_type"), TASK_CASES)
def test_charts_error_interval_tasks_match_contract(task_cls: type, query_ids: tuple[str, ...], answer_type: str, annotation_type: str) -> None:
    task = task_cls()
    for index, query_id in enumerate(query_ids):
        out = task.generate(117000 + index + len(task_cls.task_id), params={"query_id": query_id}, max_attempts=60)
        width, height = out.image.size
        assert out.query_id == query_id
        assert out.answer_gt.type == answer_type
        assert out.annotation_gt.type == annotation_type
        assert out.answer_gt.value == _expected_answer(out.trace_payload["execution_trace"], query_id)
        assert out.trace_payload["query_spec"]["query_id"] == query_id
        assert out.trace_payload["query_spec"]["params"]["query_id"] == query_id
        assert out.trace_payload["execution_trace"]["query_id"] == query_id
        annotation = out.annotation_gt.value
        if annotation_type == "segment_set":
            assert isinstance(annotation, list)
            assert annotation
            assert out.trace_payload["projected_annotation"]["segment_set"] == annotation
            assert out.trace_payload["projected_annotation"]["pixel_segment_set"] == annotation
            for segment in annotation:
                _assert_segment_inside_canvas([list(point) for point in segment], width=width, height=height)
            assert len(annotation) == len(out.trace_payload["execution_trace"]["annotation_item_ids"])
        elif annotation_type == "segment":
            assert isinstance(annotation, list)
            _assert_segment_inside_canvas([list(point) for point in annotation], width=width, height=height)
            assert out.trace_payload["projected_annotation"]["segment"] == annotation
            assert out.trace_payload["projected_annotation"]["pixel_segment"] == annotation
            assert len(out.trace_payload["execution_trace"]["annotation_item_ids"]) == 1
        else:
            raise AssertionError(f"unsupported annotation type in test: {annotation_type}")


def test_charts_error_interval_prompt_examples_match_contract() -> None:
    for task_cls, _query_ids, answer_type, annotation_type in TASK_CASES:
        out = task_cls().generate(118000 + len(task_cls.task_id), params={}, max_attempts=60)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        if answer_type == "integer":
            assert isinstance(answer_and_annotation["answer"], int)
            assert isinstance(answer_only["answer"], int)
        else:
            assert isinstance(answer_and_annotation["answer"], str)
            assert isinstance(answer_only["answer"], str)
        assert isinstance(answer_and_annotation["annotation"], list)
        if annotation_type == "segment":
            assert len(answer_and_annotation["annotation"]) == 2
            assert all(isinstance(point, list) and len(point) == 2 for point in answer_and_annotation["annotation"])
        elif annotation_type == "segment_set":
            assert all(
                isinstance(segment, list)
                and len(segment) == 2
                and all(isinstance(point, list) and len(point) == 2 for point in segment)
                for segment in answer_and_annotation["annotation"]
            )


def test_charts_error_interval_balanced_sampling_covers_scene_axis() -> None:
    scenes: Counter[str] = Counter()
    queries: Counter[str] = Counter()
    for index in range(64):
        out = ChartsErrorIntervalReferenceContainmentCountTask().generate(hash64(119000, "charts_error_interval", index), params={}, max_attempts=60)
        scenes[str(out.trace_payload["execution_trace"]["scene_variant"])] += 1
        queries[str(out.query_id)] += 1
    assert set(scenes) == set(SUPPORTED_SCENE_VARIANTS)
    assert set(queries) == {DEFAULT_QUERY_ID}


def test_charts_error_interval_is_deterministic() -> None:
    params = {"scene_variant": "bar_with_error", "query_id": "widest_interval_label"}
    out_a = ChartsErrorIntervalRelationLabelTask().generate(120000, params=params, max_attempts=60)
    out_b = ChartsErrorIntervalRelationLabelTask().generate(120000, params=params, max_attempts=60)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
