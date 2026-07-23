"""Behavior tests for scatter-cluster chart tasks."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task
from trace_tasks.tasks.charts.scatter_cluster.centroid_option_selection_label import (
    ChartsScatterClusterCentroidOptionSelectionLabelTask,
    SUPPORTED_QUERY_IDS as CENTROID_QUERY_IDS,
)
from trace_tasks.tasks.charts.scatter_cluster.cluster_area_rank_label import (
    ChartsScatterClusterAreaRankLabelTask,
    SUPPORTED_QUERY_IDS as AREA_RANK_QUERY_IDS,
)
from trace_tasks.tasks.charts.scatter_cluster.cluster_spread_extremum_label import (
    ChartsScatterClusterSpreadExtremumLabelTask,
    SUPPORTED_QUERY_IDS as SPREAD_QUERY_IDS,
)
from trace_tasks.tasks.charts.scatter_cluster.cluster_trend_direction_label import (
    ChartsScatterClusterTrendDirectionLabelTask,
    SUPPORTED_QUERY_IDS as TREND_QUERY_IDS,
)
from trace_tasks.tasks.charts.scatter_cluster.shared.state import AREA_ENVELOPE_SCATTER, OPTION_LABELS, SUPPORTED_SCENE_VARIANTS


_TASK_CASES = (
    (ChartsScatterClusterTrendDirectionLabelTask, TREND_QUERY_IDS, "bbox"),
    (ChartsScatterClusterSpreadExtremumLabelTask, SPREAD_QUERY_IDS, "bbox"),
    (ChartsScatterClusterAreaRankLabelTask, AREA_RANK_QUERY_IDS, "bbox"),
    (ChartsScatterClusterCentroidOptionSelectionLabelTask, CENTROID_QUERY_IDS, "point"),
)
_OPTION_LABELS = set(OPTION_LABELS)


def _assert_bbox_inside_canvas(bbox: Sequence[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _assert_point_inside_canvas(point: Sequence[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _expected_answer(execution: dict) -> str:
    branch = str(execution["query_id"])
    labels = [str(label) for label in execution["cluster_labels"]]
    if branch in TREND_QUERY_IDS:
        slopes = {str(label): float(value) for label, value in execution["cluster_slopes"].items()}
        if str(execution["trend_direction"]) == "upward":
            return max(labels, key=lambda label: (slopes[label], label))
        return min(labels, key=lambda label: (slopes[label], label))
    if branch in SPREAD_QUERY_IDS:
        metrics = {str(label): float(value) for label, value in execution["cluster_spread_metrics"].items()}
        if str(execution["spread_extremum"]) == "largest":
            return max(labels, key=lambda label: (metrics[label], label))
        return min(labels, key=lambda label: (metrics[label], label))
    if branch in AREA_RANK_QUERY_IDS:
        metrics = {str(label): float(value) for label, value in execution["cluster_area_metrics"].items()}
        largest_to_smallest = sorted(labels, key=lambda label: (-metrics[label], label))
        if branch == "smallest_cluster_area_label":
            return largest_to_smallest[-1]
        return largest_to_smallest[0]
    if branch == SINGLE_QUERY_ID:
        distances = {str(label): float(value) for label, value in execution["option_distances_to_centroid"].items()}
        return min(sorted(distances), key=lambda label: (distances[label], label))
    raise AssertionError(f"unsupported branch: {branch}")


@pytest.mark.parametrize(("task_cls", "query_ids", "annotation_type"), _TASK_CASES)
def test_chart_scatter_cluster_public_tasks_match_contract(task_cls, query_ids: tuple[str, ...], annotation_type: str) -> None:
    for query_index, branch in enumerate(query_ids):
        task = task_cls()
        out = task.generate(91300 + query_index, params={"query_id": branch}, max_attempts=120)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        render_map = trace["render_map"]
        expected_answer = _expected_answer(execution)

        assert out.query_id == branch
        assert execution["query_id"] == branch
        assert trace["query_spec"]["query_id"] == branch
        assert str(execution["question_format"]).startswith("scatter_cluster_")
        assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
        assert out.answer_gt.type == ("option_letter" if branch == SINGLE_QUERY_ID else "string")
        assert str(out.answer_gt.value) == expected_answer
        assert str(execution["answer"]) == expected_answer
        assert out.annotation_gt.type == annotation_type
        assert trace["projected_annotation"]["type"] == annotation_type
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert 4 <= int(execution["cluster_count"]) <= 7
        assert 8 <= int(execution["points_per_cluster"]) <= 12
        assert int(execution["total_point_count"]) == int(execution["cluster_count"]) * int(execution["points_per_cluster"])
        assert len(execution["cluster_labels"]) == int(execution["cluster_count"])
        assert len(set(execution["cluster_labels"])) == int(execution["cluster_count"])
        assert set(execution["cluster_labels"]).isdisjoint(_OPTION_LABELS)
        assert str(render["font_assets"]["font_asset_version"])
        assert str(render["font_assets"]["chart_font_family"])

        if annotation_type == "bbox":
            _assert_bbox_inside_canvas(out.annotation_gt.value, width=int(render["canvas_width"]), height=int(render["canvas_height"]))
            assert out.annotation_gt.value == render_map["cluster_bboxes_px"][expected_answer]
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        elif annotation_type == "point":
            _assert_point_inside_canvas(out.annotation_gt.value, width=int(render["canvas_width"]), height=int(render["canvas_height"]))
            assert out.annotation_gt.value == render_map["option_centers_px"][expected_answer]
            assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        else:
            raise AssertionError(f"unsupported annotation type: {annotation_type}")

        if branch in AREA_RANK_QUERY_IDS:
            assert str(execution["scene_variant"]) == AREA_ENVELOPE_SCATTER
            assert set(render_map["cluster_envelope_bboxes_px"]) == set(execution["cluster_labels"])
            assert out.annotation_gt.value == render_map["cluster_envelope_bboxes_px"][expected_answer]
            assert 0.18 <= float(execution["cluster_area_nearest_relative_gap"]) <= 0.35
        elif branch == SINGLE_QUERY_ID:
            option_labels = {str(label) for label in execution["option_labels"]}
            assert int(execution["option_count"]) in {4, 6}
            assert len(option_labels) == int(execution["option_count"])
            assert option_labels.issubset(_OPTION_LABELS)
            assert expected_answer in option_labels
            assert out.annotation_gt.value == render_map["option_centers_px"][expected_answer]
            distances = {str(label): float(value) for label, value in execution["option_distances_to_centroid"].items()}
            answer_distance = distances[expected_answer]
            assert all(
                answer_distance + float(execution["minimum_distance_margin"]) < distance
                for label, distance in distances.items()
                if str(label) != expected_answer
            )


def test_chart_scatter_prompt_examples_match_annotation_contract() -> None:
    for task_cls, query_ids, annotation_type in _TASK_CASES:
        task = task_cls()
        for query_index, branch in enumerate(query_ids):
            out = task.generate(91400 + query_index, params={"query_id": branch}, max_attempts=120)
            answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
            answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
            assert isinstance(answer_only["answer"], str)
            if branch == SINGLE_QUERY_ID:
                assert answer_and_annotation["answer"] in _OPTION_LABELS
            if annotation_type == "bbox":
                assert isinstance(answer_and_annotation["annotation"], list)
                assert len(answer_and_annotation["annotation"]) == 4
            elif annotation_type == "point":
                assert isinstance(answer_and_annotation["annotation"], list)
                assert len(answer_and_annotation["annotation"]) == 2
            else:
                assert isinstance(answer_and_annotation["annotation"], dict)
            assert "from from" not in out.prompt
            assert "to to" not in out.prompt


def test_chart_scatter_sampling_covers_task_branches_and_axes() -> None:
    observed: dict[str, set[str]] = {
        "trend": set(),
        "spread_axis": set(),
        "spread_extremum": set(),
        "area_rank": set(),
        "centroid": set(),
    }
    for query_index, branch in enumerate(TREND_QUERY_IDS):
        execution = ChartsScatterClusterTrendDirectionLabelTask().generate(91500 + query_index, params={"query_id": branch}, max_attempts=120).trace_payload["execution_trace"]
        observed["trend"].add(str(execution["trend_direction"]))
    for query_index, branch in enumerate(SPREAD_QUERY_IDS):
        execution = ChartsScatterClusterSpreadExtremumLabelTask().generate(91540 + query_index, params={"query_id": branch}, max_attempts=120).trace_payload["execution_trace"]
        observed["spread_axis"].add(str(execution["spread_axis"]))
        observed["spread_extremum"].add(str(execution["spread_extremum"]))
    for query_index, branch in enumerate(AREA_RANK_QUERY_IDS):
        execution = ChartsScatterClusterAreaRankLabelTask().generate(91580 + query_index, params={"query_id": branch}, max_attempts=120).trace_payload["execution_trace"]
        observed["area_rank"].add(str(execution["area_rank"]))
    for index in range(12):
        execution = ChartsScatterClusterCentroidOptionSelectionLabelTask().generate(hash64(91600, "centroid", index), params={}, max_attempts=120).trace_payload["execution_trace"]
        observed["centroid"].add(str(execution["answer"]))

    assert observed["trend"] == {"upward", "downward"}
    assert observed["spread_axis"] == {"horizontal", "vertical", "overall"}
    assert observed["spread_extremum"] == {"largest", "smallest"}
    assert observed["area_rank"] == {"largest", "smallest"}
    assert observed["centroid"].issubset(_OPTION_LABELS)


def test_chart_scatter_is_deterministic() -> None:
    task = ChartsScatterClusterSpreadExtremumLabelTask()
    params = {"query_id": "largest_overall_spread_label"}
    out_a = task.generate(91650, params=params, max_attempts=120)
    out_b = task.generate(91650, params=params, max_attempts=120)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


def test_chart_scatter_registered_and_scene_config_loaded() -> None:
    assert create_task("task_charts__scatter_cluster__cluster_trend_direction_label").task_id == "task_charts__scatter_cluster__cluster_trend_direction_label"
    assert create_task("task_charts__scatter_cluster__centroid_option_selection_label").task_id == "task_charts__scatter_cluster__centroid_option_selection_label"
    assert create_task("task_charts__scatter_cluster__cluster_area_rank_label").task_id == "task_charts__scatter_cluster__cluster_area_rank_label"
    cfg = get_scene_defaults("charts", "scatter_cluster")
    assert isinstance(cfg.get("generation"), dict)
    assert isinstance(cfg.get("rendering"), dict)
    assert isinstance(cfg.get("prompt"), dict)
    generation = cfg["generation"]["shared"]
    assert int(generation["cluster_count_min"]) == 4
    assert int(generation["cluster_count_max"]) == 7
    assert "query_id_weights" not in generation
    prompt = cfg["prompt"]["shared"]
    assert str(prompt["bundle_id"]) == "charts_scatter_cluster_v1"
