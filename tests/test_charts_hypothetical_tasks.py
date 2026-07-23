"""Behavior tests for single-series hypothetical/counterfactual chart tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.tasks.charts.single_series.remaining_mean_after_removal import ChartsHypotheticalRemainingMeanAfterRemovalPublicTask
from trace_tasks.tasks.charts.single_series.target_share_after_removal import ChartsHypotheticalTargetShareAfterRemovalPublicTask


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _mean(values: list[int]) -> int:
    assert sum(values) % len(values) == 0
    return int(sum(values) // len(values))


def _expected_answer(trace: dict) -> int:
    values_by_label = {str(label): int(value) for label, value in trace["values_by_label"].items()}
    operation = str(trace["counterfactual_operation"])
    if operation == "remove_labels_then_mean":
        retained = [values_by_label[str(label)] for label in trace["retained_labels"]]
        return _mean(retained)
    if operation == "remove_labels_then_target_share_percent":
        retained_total = sum(values_by_label[str(label)] for label in trace["retained_labels"])
        target_value = values_by_label[str(trace["target_label"])]
        assert target_value * 100 % retained_total == 0
        return int(target_value * 100 // retained_total)
    raise AssertionError(f"unsupported operation: {operation}")


@pytest.mark.parametrize(
    ("task", "scene_variant"),
    [
        (ChartsHypotheticalRemainingMeanAfterRemovalPublicTask(), "bar"),
        (ChartsHypotheticalTargetShareAfterRemovalPublicTask(), "dot_plot"),
    ],
)
def test_chart_hypothetical_tasks_match_contract(task, scene_variant: str) -> None:
    out = task.generate(12100, params={"query_id": "single", "scene_variant": scene_variant}, max_attempts=10)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
    assert str(execution["scene_variant"]) == str(scene_variant)
    assert str(render["scene_variant"]) == str(scene_variant)
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    labels = [str(label) for label in execution["labels"]]
    annotation_labels = [str(label) for label in execution["annotation_labels"]]
    annotation_points = [list(point) for point in out.annotation_gt.value]
    assert annotation_labels == sorted(annotation_labels)
    assert set(annotation_labels).issubset(set(labels))
    assert trace["projected_annotation"]["point_set"] == annotation_points
    assert "label_set" not in trace["projected_annotation"]
    assert len(annotation_points) == len(annotation_labels)
    assert len(trace["projected_annotation"]["bbox_set"]) == len(annotation_labels)
    assert int(out.answer_gt.value) == int(execution["answer_value"])
    assert int(out.answer_gt.value) == _expected_answer(execution)
    assert str(trace["query_spec"]["query_id"]) == "single"
    assert str(trace["query_spec"]["params"]["scene_variant"]) == str(scene_variant)
    assert len(trace["scene_ir"]["entities"]) == int(execution["mark_count"])
    assert {str(entity["attrs"]["label"]) for entity in trace["scene_ir"]["entities"]} == set(labels)
    assert set(trace["render_map"]["label_centers_px"]) == set(labels)


def test_chart_hypothetical_prompt_examples_match_selected_task() -> None:
    cases = (
        (
            ChartsHypotheticalRemainingMeanAfterRemovalPublicTask(),
            {"annotation": [[180, 360], [320, 240], [460, 300]], "answer": 18},
        ),
        (
            ChartsHypotheticalTargetShareAfterRemovalPublicTask(),
            {"annotation": [[180, 260], [320, 340], [460, 220]], "answer": 25},
        ),
    )
    for index, (task, expected) in enumerate(cases, start=12300):
        out = task.generate(index, params={"query_id": "single"}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected
        assert answer_only == {"answer": expected["answer"]}


def test_chart_hypothetical_rejects_non_initial_scene_variants() -> None:
    task = ChartsHypotheticalRemainingMeanAfterRemovalPublicTask()
    for scene_variant in ("horizontal_bar", "line", "area", "scatter", "pie", "donut", "radar"):
        with pytest.raises((ValueError, RuntimeError)):
            task.generate(12400, params={"query_id": "single", "scene_variant": scene_variant}, max_attempts=10)


def test_chart_hypothetical_task_is_deterministic() -> None:
    task = ChartsHypotheticalTargetShareAfterRemovalPublicTask()
    params = {"query_id": "single", "scene_variant": "lollipop"}
    out_a = task.generate(12500, params=params, max_attempts=10)
    out_b = task.generate(12500, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_chart_hypothetical_rejects_unsupported_query_id() -> None:
    task = ChartsHypotheticalRemainingMeanAfterRemovalPublicTask()
    with pytest.raises(ValueError, match="query_id"):
        task.generate(12600, params={"query_id": "__unsupported_query_id__"}, max_attempts=10)
