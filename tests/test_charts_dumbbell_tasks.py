"""Behavior tests for dumbbell chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import assert_counter_support_within, extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task


TASK_QUERY_IDS = {
    "task_charts__dumbbell__gap_rank_row_label": (
        "largest_gap_rank_row_label",
        "smallest_gap_rank_row_label",
    ),
    "task_charts__dumbbell__side_winner_count": (
        "series_a_greater_threshold_count",
        "series_b_greater_threshold_count",
    ),
    "task_charts__dumbbell__absolute_gap_threshold_count": (
        "absolute_gap_at_least_threshold_count",
        "absolute_gap_at_most_threshold_count",
    ),
}


def _bbox_center(bbox: list[float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    assert 0 <= float(point[0]) <= width
    assert 0 <= float(point[1]) <= height


def _expected_answer(execution: dict) -> str | int:
    rows = list(execution["rows"])
    query_id = str(execution["query_id"])
    params = dict(execution["query_params"])
    if query_id in {"largest_gap_rank_row_label", "smallest_gap_rank_row_label"}:
        reverse = str(params["rank_order"]) == "largest"
        rank = int(params["rank_n"])
        ranked_gaps = sorted({int(row["gap"]) for row in rows}, reverse=reverse)
        target_gap = int(ranked_gaps[rank - 1])
        winners = [row for row in rows if int(row["gap"]) == target_gap]
        assert len(winners) == 1
        return str(winners[0]["label"])
    if query_id in {"series_a_greater_threshold_count", "series_b_greater_threshold_count"}:
        threshold = int(params["threshold_value"])
        if str(params["side_direction"]) == "series_a_greater":
            return sum(1 for row in rows if int(row["signed_delta_a_minus_b"]) >= threshold)
        return sum(1 for row in rows if -int(row["signed_delta_a_minus_b"]) >= threshold)
    if query_id in {"absolute_gap_at_least_threshold_count", "absolute_gap_at_most_threshold_count"}:
        threshold = int(params["gap_threshold_value"])
        if str(params["gap_threshold_relation"]) == "at_least":
            return sum(1 for row in rows if int(row["gap"]) >= threshold)
        return sum(1 for row in rows if int(row["gap"]) <= threshold)
    raise AssertionError(f"unsupported query_id: {query_id}")


@pytest.mark.parametrize("task_id,query_ids", TASK_QUERY_IDS.items())
def test_chart_dumbbell_query_branches_match_contract(task_id: str, query_ids: tuple[str, ...]) -> None:
    task = create_task(task_id)
    for index, query_id in enumerate(query_ids):
        out = task.generate(
            hash64(20260503, task_id, index),
            params={"query_id": query_id},
            max_attempts=80,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        assert out.query_id == query_id
        assert str(execution["scene_variant"]) == "horizontal_dumbbell"
        assert str(execution["question_format"]).startswith("dumbbell_")
        expected_annotation_type = "segment" if task_id.endswith("__gap_rank_row_label") else "segment_set"
        assert out.annotation_gt.type == expected_annotation_type
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert 10 <= int(execution["row_count"]) <= 16
        assert len(execution["rows"]) == int(execution["row_count"])
        assert len(set(execution["row_labels"])) == int(execution["row_count"])
        expected_answer = _expected_answer(execution)
        assert out.answer_gt.value == expected_answer
        assert execution["answer"] == expected_answer
        if task_id.endswith("__gap_rank_row_label"):
            assert out.answer_gt.type == "string"
            assert str(out.answer_gt.value) in set(execution["row_labels"])
            assert len(execution["annotation_row_ids"]) == 1
            assert int(execution["query_params"]["rank_n"]) in {1, 2}
        else:
            assert out.answer_gt.type == "integer"
            assert 2 <= int(out.answer_gt.value) <= 10
            assert len(execution["annotation_row_ids"]) == int(out.answer_gt.value)
        expected_point_pairs = [
            [
                _bbox_center(trace["render_map"]["point_bboxes_px"][f"{row_id}:series_a"]),
                _bbox_center(trace["render_map"]["point_bboxes_px"][f"{row_id}:series_b"]),
            ]
            for row_id in execution["annotation_row_ids"]
        ]
        expected_annotation = expected_point_pairs[0] if expected_annotation_type == "segment" else expected_point_pairs
        assert out.annotation_gt.value == expected_annotation
        if expected_annotation_type == "segment":
            assert trace["projected_annotation"]["segment"] == out.annotation_gt.value
            assert trace["projected_annotation"]["pixel_segment"] == out.annotation_gt.value
        else:
            assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
            assert trace["projected_annotation"]["pixel_segment_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["row_ids"] == execution["annotation_row_ids"]
        annotation_segments = [out.annotation_gt.value] if expected_annotation_type == "segment" else out.annotation_gt.value
        for segment in annotation_segments:
            assert len(segment) == 2
            for point in segment:
                _assert_point_inside_canvas([float(value) for value in point], width=int(render["canvas_width"]), height=int(render["canvas_height"]))


def test_chart_dumbbell_prompt_examples_match_contract() -> None:
    for task_id, query_ids in TASK_QUERY_IDS.items():
        task = create_task(task_id)
        for index, query_id in enumerate(query_ids, start=92100):
            out = task.generate(index, params={"query_id": query_id}, max_attempts=80)
            answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
            answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
            assert "annotation" in answer_and_annotation
            if task_id.endswith("__gap_rank_row_label"):
                assert isinstance(answer_and_annotation["answer"], str)
                assert isinstance(answer_only["answer"], str)
            else:
                assert isinstance(answer_and_annotation["answer"], int)
                assert isinstance(answer_only["answer"], int)


def test_chart_dumbbell_balanced_sampling_covers_axes() -> None:
    for task_id, query_ids in TASK_QUERY_IDS.items():
        task = create_task(task_id)
        queries: Counter[str] = Counter()
        row_counts: Counter[int] = Counter()
        answers: Counter[str] = Counter()
        rank_orders: Counter[str] = Counter()
        side_directions: Counter[str] = Counter()
        gap_relations: Counter[str] = Counter()
        for index in range(72):
            out = task.generate(
                hash64(92200, task_id, index),
                params={"_sample_cursor": index},
                max_attempts=120,
            )
            execution = out.trace_payload["execution_trace"]
            params = dict(execution["query_params"])
            queries[str(execution["query_id"])] += 1
            row_counts[int(execution["row_count"])] += 1
            answers[str(execution["answer"])] += 1
            if task_id.endswith("__gap_rank_row_label"):
                rank_orders[str(params["rank_order"])] += 1
            if task_id.endswith("__side_winner_count"):
                side_directions[str(params["side_direction"])] += 1
            if task_id.endswith("__absolute_gap_threshold_count"):
                gap_relations[str(params["gap_threshold_relation"])] += 1
        assert_counter_support_within(queries, query_ids, expected_per_key=36, tolerance=2)
        assert set(row_counts).issubset(set(range(10, 17)))
        assert len(row_counts) >= 2
        assert len(answers) >= 4
        if task_id.endswith("__gap_rank_row_label"):
            assert set(rank_orders) == {"largest", "smallest"}
        if task_id.endswith("__side_winner_count"):
            assert set(side_directions) == {"series_a_greater", "series_b_greater"}
        if task_id.endswith("__absolute_gap_threshold_count"):
            assert set(gap_relations) == {"at_least", "at_most"}


def test_chart_dumbbell_is_deterministic() -> None:
    task = create_task("task_charts__dumbbell__gap_rank_row_label")
    params = {"query_id": "largest_gap_rank_row_label", "rank_n": 2}
    out_a = task.generate(92300, params=params, max_attempts=80)
    out_b = task.generate(92300, params=params, max_attempts=80)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


def test_chart_dumbbell_registered_and_scene_config_loaded() -> None:
    for task_id in TASK_QUERY_IDS:
        assert create_task(task_id).task_id == task_id
    cfg = get_scene_defaults("charts", "dumbbell")
    assert isinstance(cfg.get("generation"), dict)
    assert isinstance(cfg.get("rendering"), dict)
    assert isinstance(cfg.get("prompt"), dict)
    generation = cfg["generation"]["shared"]
    assert int(generation["row_count_min"]) == 10
    assert int(generation["row_count_max"]) == 16
    assert "query_id_weights" not in generation
    prompt = cfg["prompt"]["shared"]
    assert str(prompt["bundle_id"]) == "charts_dumbbell_v1"
    assert str(prompt["scene_key"]) == "dumbbell_pairwise_chart"
    assert str(prompt["task_key"]) == "dumbbell_pairwise_comparison_query"
