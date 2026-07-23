"""Behavior tests for synthetic 3D chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task


PUBLIC_TASKS = {
    "task_charts__surface_3d__reference_nearest_label": {
        "queries": ("single",),
        "annotation": "point",
    },
    "task_charts__surface_3d__series_trend_label": {
        "queries": ("increase", "decrease"),
        "annotation": "segment",
    },
    "task_charts__surface_3d__panel_variation_label": {
        "queries": ("single",),
        "annotation": "bbox",
    },
}


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _expected_answer(execution: dict) -> str:
    question_format = str(execution["question_format"])
    branch = str(execution["query_id"])
    if question_format == "surface_3d_reference_nearest_label":
        target = float(execution["target_axis_value"])
        return min(
            (str(point["label"]) for point in execution["points"]),
            key=lambda label: (
                abs(
                    next(
                        float(point["y_value"])
                        for point in execution["points"]
                        if str(point["label"]) == label
                    )
                    - target
                ),
                label,
            ),
        )
    if question_format == "surface_3d_series_trend_label":
        deltas = {str(label): int(value) for label, value in execution["deltas_by_series"].items()}
        if branch == "increase":
            return max(deltas, key=lambda label: (deltas[label], label))
        return min(deltas, key=lambda label: (deltas[label], label))
    if question_format == "surface_3d_panel_variation_label":
        ranges = {str(label): int(value) for label, value in execution["ranges_by_panel"].items()}
        return max(ranges, key=lambda label: (ranges[label], label))
    raise AssertionError(f"unsupported question format: {question_format}")


def _bbox_center(bbox: list[float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _expected_annotation(trace_payload: dict) -> list[float] | list[list[float]]:
    execution = trace_payload["execution_trace"]
    render_map = trace_payload["render_map"]
    question_format = str(execution["question_format"])
    if question_format == "surface_3d_reference_nearest_label":
        bbox = render_map["point_bboxes_px"][str(execution["answer_point_id"])]
        return _bbox_center(bbox)
    if question_format == "surface_3d_series_trend_label":
        return [
            _bbox_center(render_map["point_bboxes_px"][str(execution["start_point_id"])]),
            _bbox_center(render_map["point_bboxes_px"][str(execution["end_point_id"])]),
        ]
    if question_format == "surface_3d_panel_variation_label":
        return render_map["panel_bboxes_px"][str(execution["answer_panel_label"])]
    raise AssertionError(f"unsupported question format: {question_format}")


@pytest.mark.parametrize("task_id,contract", sorted(PUBLIC_TASKS.items()))
def test_chart_three_d_public_task_queries_match_contract(task_id: str, contract: dict) -> None:
    task = create_task(task_id)
    for index, query_id in enumerate(contract["queries"]):
        out = task.generate(98200 + index, params={"query_id": query_id}, max_attempts=80)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]

        assert out.query_id == query_id
        assert trace["query_spec"]["query_id"] == query_id
        assert trace["query_spec"]["params"]["query_id"] == query_id
        assert trace["scene_ir"]["relations"]["query_id"] == query_id
        assert out.annotation_gt.type == contract["annotation"]
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert str(execution["scene_variant"]) in {
            "three_d_scatter",
            "three_d_surface",
            "three_d_small_multiples",
        }
        assert out.answer_gt.value == _expected_answer(execution)
        assert execution["answer"] == out.answer_gt.value
        assert trace["projected_annotation"]["type"] == out.annotation_gt.type
        assert out.annotation_gt.value == _expected_annotation(trace)
        assert trace["render_spec"]["font_assets"]

        if out.annotation_gt.type == "bbox":
            _assert_bbox_inside_canvas(
                [float(value) for value in out.annotation_gt.value],
                width=int(render["canvas_width"]),
                height=int(render["canvas_height"]),
            )
        elif out.annotation_gt.type == "point":
            _assert_point_inside_canvas(
                [float(value) for value in out.annotation_gt.value],
                width=int(render["canvas_width"]),
                height=int(render["canvas_height"]),
            )
        elif out.annotation_gt.type == "segment":
            for point in out.annotation_gt.value:
                _assert_point_inside_canvas(
                    [float(value) for value in point],
                    width=int(render["canvas_width"]),
                    height=int(render["canvas_height"]),
                )
        else:
            for bbox in out.annotation_gt.value.values():
                _assert_bbox_inside_canvas(
                    [float(value) for value in bbox],
                    width=int(render["canvas_width"]),
                    height=int(render["canvas_height"]),
                )


def test_chart_three_d_prompt_examples_match_annotation_contracts() -> None:
    for task_id, contract in sorted(PUBLIC_TASKS.items()):
        task = create_task(task_id)
        for query_id in contract["queries"]:
            out = task.generate(
                hash64(98400, f"{task_id}.{query_id}"),
                params={"query_id": query_id},
                max_attempts=80,
            )
            answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
            answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
            assert "Read this visual" not in out.prompt
            assert "Shown is" not in out.prompt
            assert isinstance(answer_only["answer"], str)
            assert isinstance(answer_and_annotation["answer"], str)
            if contract["annotation"] == "bbox":
                assert isinstance(answer_and_annotation["annotation"], list)
                assert len(answer_and_annotation["annotation"]) == 4
            elif contract["annotation"] == "point":
                assert isinstance(answer_and_annotation["annotation"], list)
                assert len(answer_and_annotation["annotation"]) == 2
            elif contract["annotation"] == "segment":
                assert isinstance(answer_and_annotation["annotation"], list)
                assert len(answer_and_annotation["annotation"]) == 2
                assert all(isinstance(point, list) and len(point) == 2 for point in answer_and_annotation["annotation"])
            else:
                assert sorted(answer_and_annotation["annotation"]) == ["end_point", "start_point"]


def test_chart_three_d_sampling_covers_branches_and_sizes() -> None:
    variants: Counter[str] = Counter()
    panel_counts: Counter[int] = Counter()
    for index in range(80):
        trend_out = create_task("task_charts__surface_3d__series_trend_label").generate(
            hash64(98500, "surface_3d_trend", index),
            params={},
            max_attempts=120,
        )
        panel_out = create_task("task_charts__surface_3d__panel_variation_label").generate(
            hash64(98500, "surface_3d_panel", index),
            params={},
            max_attempts=120,
        )
        variants[str(trend_out.query_id)] += 1
        panel_counts[int(panel_out.trace_payload["execution_trace"]["panel_count"])] += 1

    assert {"increase", "decrease"}.issubset(set(variants))
    assert min(panel_counts) >= 4
    assert max(panel_counts) <= 6
    assert set(panel_counts).issubset({4, 6})


def test_chart_three_d_config_is_scene_package_ready() -> None:
    defaults = get_scene_defaults("charts", "surface_3d")
    assert defaults["prompt"]["shared"]["bundle_id"] == "charts_surface_3d_v1"
    assert "query_id_weights" not in defaults["generation"]["shared"]
    assert "balanced_query_id_sampling" not in defaults["generation"]["shared"]
