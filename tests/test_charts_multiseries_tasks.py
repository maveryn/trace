"""Behavior tests for migrated chart multiseries tasks."""

from __future__ import annotations

import json
from typing import Any

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.multiseries.category_total_extremum_label import (
    CATEGORY_TOTAL_QUERY_IDS,
    ChartsMultiseriesCategoryTotalExtremumLabelTask,
    LARGEST_CATEGORY_TOTAL_QUERY_ID,
    SMALLEST_CATEGORY_TOTAL_QUERY_ID,
)
from trace_tasks.tasks.charts.multiseries.pair_equality_label import ChartsMultiseriesPairEqualityLabelTask
from trace_tasks.tasks.charts.multiseries.ranked_change_extremum_label import (
    ChartsMultiseriesRankedChangeExtremumTask,
    LARGEST_ABSOLUTE_GAP_QUERY_ID,
    LARGEST_DECREASE_QUERY_ID,
    LARGEST_INCREASE_QUERY_ID,
    RANKED_CHANGE_QUERY_IDS,
    SMALLEST_ABSOLUTE_GAP_QUERY_ID,
)
from trace_tasks.tasks.charts.multiseries.ranked_pair_ratio_extremum_label import (
    ChartsMultiseriesRankedPairRatioExtremumTask,
    LARGEST_PAIR_RATIO_QUERY_ID,
    PAIR_RATIO_QUERY_IDS,
    SMALLEST_PAIR_RATIO_QUERY_ID,
)
from trace_tasks.tasks.charts.multiseries.ranked_series_share_extremum_label import (
    ChartsMultiseriesRankedSeriesShareExtremumTask,
    LARGEST_SERIES_SHARE_QUERY_ID,
    SERIES_SHARE_QUERY_IDS,
    SMALLEST_SERIES_SHARE_QUERY_ID,
)
from trace_tasks.tasks.charts.multiseries.series_rank_at_category_label import (
    ChartsMultiseriesSeriesRankAtCategoryLabelTask,
    LARGEST_SERIES_AT_CATEGORY_QUERY_ID,
    SERIES_RANK_AT_CATEGORY_QUERY_IDS,
    SMALLEST_SERIES_AT_CATEGORY_QUERY_ID,
)


SINGLE_QUERY_TASK_CLASSES = (
    ChartsMultiseriesPairEqualityLabelTask,
)


def _extract_prompt_json_example(prompt: str) -> dict[str, Any]:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def _execution(out: Any) -> dict[str, Any]:
    return dict(out.trace_payload["execution_trace"])


def _values_by_category(out: Any) -> dict[str, dict[str, int]]:
    return {
        str(category): {str(series): int(value) for series, value in series_values.items()}
        for category, series_values in _execution(out)["values_by_category"].items()
    }


def _assert_common_public_task_contract(
    out: Any,
    *,
    expected_answer_type: str,
    expected_query_id: str = SINGLE_QUERY_ID,
) -> None:
    trace = out.trace_payload
    render = trace["render_spec"]
    projected = trace["projected_annotation"]

    assert out.query_id == str(expected_query_id)
    assert out.answer_gt.type == expected_answer_type
    assert out.annotation_gt.type == "point_map"
    assert projected["type"] == "point_map"
    assert projected["point_map"] == out.annotation_gt.value
    assert projected["pixel_point_map"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert _extract_prompt_json_example(out.prompt_variants["answer_only"]).keys() == {"answer"}
    assert _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"]).keys() == {
        "annotation",
        "answer",
    }
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert trace["query_spec"]["query_id"] == str(expected_query_id)
    assert _execution(out)["query_id"] == str(expected_query_id)
    assert _execution(out)["internal_query_id"]
    assert _execution(out)["scene_variant"] == render["scene_variant"]

    for key, point in out.annotation_gt.value.items():
        assert ":" in str(key)
        x_coord, y_coord = [float(value) for value in point]
        assert 0 <= x_coord <= int(render["canvas_width"])
        assert 0 <= y_coord <= int(render["canvas_height"])


@pytest.mark.parametrize("task_cls", SINGLE_QUERY_TASK_CLASSES)
def test_chart_multiseries_public_tasks_use_single_query_id(task_cls: type) -> None:
    task = task_cls()
    out = task.generate(12000, params={"query_id": SINGLE_QUERY_ID}, max_attempts=100)

    assert task.supported_query_ids == (SINGLE_QUERY_ID,)
    _assert_common_public_task_contract(out, expected_answer_type=out.answer_gt.type)


@pytest.mark.parametrize(
    ("query_id", "expected_direction"),
    [
        (LARGEST_CATEGORY_TOTAL_QUERY_ID, "largest"),
        (SMALLEST_CATEGORY_TOTAL_QUERY_ID, "smallest"),
    ],
)
def test_chart_multiseries_category_total_extremum_matches_contract(query_id: str, expected_direction: str) -> None:
    task = ChartsMultiseriesCategoryTotalExtremumLabelTask()
    out = task.generate(12020, params={"query_id": str(query_id)}, max_attempts=100)
    execution = _execution(out)
    answer_label = str(out.answer_gt.value)
    series_labels = [str(label) for label in execution["series_labels"]]

    assert task.supported_query_ids == CATEGORY_TOTAL_QUERY_IDS
    _assert_common_public_task_contract(out, expected_answer_type="string", expected_query_id=str(query_id))
    assert execution["extremum_direction"] == str(expected_direction)
    assert int(execution["answer_rank"]) == 1
    assert answer_label == str(execution["ranked_category_labels"][int(execution["answer_rank"]) - 1])
    totals = {str(label): int(value) for label, value in execution["category_totals_by_category"].items()}
    if str(expected_direction) == "largest":
        assert totals[answer_label] == max(totals.values())
    else:
        assert totals[answer_label] == min(totals.values())
    assert set(out.annotation_gt.value.keys()) == {f"{answer_label}:{series_label}" for series_label in series_labels}


def test_chart_multiseries_pair_equality_matches_contract() -> None:
    task = ChartsMultiseriesPairEqualityLabelTask()
    out = task.generate(12030, params={}, max_attempts=100)
    execution = _execution(out)
    values_by_category = _values_by_category(out)
    answer_label = str(out.answer_gt.value)
    left_series, right_series = [str(label) for label in execution["queried_series_labels"]]

    _assert_common_public_task_contract(out, expected_answer_type="string")
    equality_labels = [
        str(category)
        for category, series_values in values_by_category.items()
        if int(series_values[left_series]) == int(series_values[right_series])
    ]
    assert equality_labels == [answer_label]
    assert set(out.annotation_gt.value.keys()) == {
        f"{answer_label}:{left_series}",
        f"{answer_label}:{right_series}",
    }


@pytest.mark.parametrize(
    ("query_id", "expected_internal_query_id", "expected_change_measure", "expected_direction_key", "expected_direction"),
    [
        (
            LARGEST_INCREASE_QUERY_ID,
            "ranked_largest_increase",
            "directional_change",
            "change_direction",
            "increase",
        ),
        (
            LARGEST_DECREASE_QUERY_ID,
            "ranked_largest_decrease",
            "directional_change",
            "change_direction",
            "decrease",
        ),
        (
            LARGEST_ABSOLUTE_GAP_QUERY_ID,
            "ranked_largest_gap",
            "absolute_gap",
            "extremum_direction",
            "largest",
        ),
        (
            SMALLEST_ABSOLUTE_GAP_QUERY_ID,
            "ranked_smallest_gap",
            "absolute_gap",
            "extremum_direction",
            "smallest",
        ),
    ],
)
def test_chart_multiseries_ranked_change_extremum_matches_contract(
    query_id: str,
    expected_internal_query_id: str,
    expected_change_measure: str,
    expected_direction_key: str,
    expected_direction: str,
) -> None:
    task = ChartsMultiseriesRankedChangeExtremumTask()
    out = task.generate(
        12040,
        params={"query_id": str(query_id)},
        max_attempts=100,
    )
    execution = _execution(out)
    answer_label = str(out.answer_gt.value)
    left_series, right_series = [str(label) for label in execution["queried_series_labels"]]

    assert task.supported_query_ids == RANKED_CHANGE_QUERY_IDS
    _assert_common_public_task_contract(out, expected_answer_type="string", expected_query_id=str(query_id))
    assert int(execution["answer_rank"]) == 1
    assert answer_label == str(execution["ranked_category_labels"][int(execution["answer_rank"]) - 1])
    assert execution["internal_query_id"] == str(expected_internal_query_id)
    assert execution["change_measure"] == str(expected_change_measure)
    assert execution[str(expected_direction_key)] == str(expected_direction)
    assert set(out.annotation_gt.value.keys()) == {
        f"{answer_label}:{left_series}",
        f"{answer_label}:{right_series}",
    }


@pytest.mark.parametrize(
    ("query_id", "expected_internal_query_id", "expected_direction"),
    [
        (LARGEST_PAIR_RATIO_QUERY_ID, "ranked_largest_pair_ratio", "largest"),
        (SMALLEST_PAIR_RATIO_QUERY_ID, "ranked_smallest_pair_ratio", "smallest"),
    ],
)
def test_chart_multiseries_ranked_pair_ratio_extremum_matches_contract(
    query_id: str,
    expected_internal_query_id: str,
    expected_direction: str,
) -> None:
    task = ChartsMultiseriesRankedPairRatioExtremumTask()
    out = task.generate(12050, params={"query_id": str(query_id)}, max_attempts=100)
    execution = _execution(out)
    answer_label = str(out.answer_gt.value)
    numerator = str(execution["numerator_series_label"])
    denominator = str(execution["denominator_series_label"])

    assert task.supported_query_ids == PAIR_RATIO_QUERY_IDS
    _assert_common_public_task_contract(out, expected_answer_type="string", expected_query_id=str(query_id))
    assert int(execution["answer_rank"]) == 1
    assert answer_label == str(execution["ranked_category_labels"][int(execution["answer_rank"]) - 1])
    assert execution["internal_query_id"] == str(expected_internal_query_id)
    assert execution["extremum_direction"] == str(expected_direction)
    assert set(out.annotation_gt.value.keys()) == {
        f"{answer_label}:{numerator}",
        f"{answer_label}:{denominator}",
    }


@pytest.mark.parametrize(
    ("query_id", "expected_internal_query_id", "expected_direction"),
    [
        (LARGEST_SERIES_SHARE_QUERY_ID, "ranked_largest_series_share", "largest"),
        (SMALLEST_SERIES_SHARE_QUERY_ID, "ranked_smallest_series_share", "smallest"),
    ],
)
def test_chart_multiseries_ranked_series_share_extremum_matches_contract(
    query_id: str,
    expected_internal_query_id: str,
    expected_direction: str,
) -> None:
    task = ChartsMultiseriesRankedSeriesShareExtremumTask()
    out = task.generate(12060, params={"query_id": str(query_id)}, max_attempts=100)
    execution = _execution(out)
    answer_label = str(out.answer_gt.value)
    series_labels = [str(label) for label in execution["series_labels"]]

    assert task.supported_query_ids == SERIES_SHARE_QUERY_IDS
    _assert_common_public_task_contract(out, expected_answer_type="string", expected_query_id=str(query_id))
    assert int(execution["answer_rank"]) == 1
    assert answer_label == str(execution["ranked_category_labels"][int(execution["answer_rank"]) - 1])
    assert execution["internal_query_id"] == str(expected_internal_query_id)
    assert execution["extremum_direction"] == str(expected_direction)
    assert set(out.annotation_gt.value.keys()) == {f"{answer_label}:{series_label}" for series_label in series_labels}


@pytest.mark.parametrize(
    ("query_id", "expected_direction"),
    [
        (LARGEST_SERIES_AT_CATEGORY_QUERY_ID, "largest"),
        (SMALLEST_SERIES_AT_CATEGORY_QUERY_ID, "smallest"),
    ],
)
def test_chart_multiseries_series_rank_at_category_matches_contract(query_id: str, expected_direction: str) -> None:
    task = ChartsMultiseriesSeriesRankAtCategoryLabelTask()
    out = task.generate(12070, params={"query_id": str(query_id)}, max_attempts=100)
    execution = _execution(out)
    answer_label = str(out.answer_gt.value)
    target_category = str(execution["target_category_label"])
    series_values = {str(label): int(value) for label, value in execution["values_by_series_at_target_category"].items()}
    if str(expected_direction) == "largest":
        ranked_series = sorted(series_values, key=lambda label: (-series_values[label], label))
    else:
        ranked_series = sorted(series_values, key=lambda label: (series_values[label], label))

    assert task.supported_query_ids == SERIES_RANK_AT_CATEGORY_QUERY_IDS
    _assert_common_public_task_contract(out, expected_answer_type="string", expected_query_id=str(query_id))
    assert execution["extremum_direction"] == str(expected_direction)
    assert int(execution["answer_rank"]) == 1
    assert answer_label == ranked_series[int(execution["answer_rank"]) - 1]
    assert set(out.annotation_gt.value.keys()) == {
        f"{target_category}:{series_label}"
        for series_label in execution["series_labels"]
    }


def test_chart_multiseries_task_is_deterministic() -> None:
    task = ChartsMultiseriesRankedChangeExtremumTask()
    params = {
        "scene_variant": "multi_line",
        "change_measure": "absolute_gap",
        "extremum_direction": "largest",
    }
    out_a = task.generate(12080, params=params, max_attempts=100)
    out_b = task.generate(12080, params=params, max_attempts=100)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
