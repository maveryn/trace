"""Tests for named-shape closer-to-reference icon counting."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.icons.named_field.closer_to_reference_count import QUERY_ID, QUERIED_REFERENCE_LABELS


TASK_ID = "task_icons__named_field__closer_to_reference_count"


def _distance(entity: dict[str, object], reference: dict[str, object]) -> float:
    ex, ey = entity["center_xy"]
    rx, ry = reference["center_xy"]
    return ((float(ex) - float(rx)) ** 2 + (float(ey) - float(ry)) ** 2) ** 0.5


def test_icons_counting_named_shape_closer_to_reference_contract_all_queries() -> None:
    task = create_task(TASK_ID)
    for index, reference_label in enumerate(QUERIED_REFERENCE_LABELS):
        out = task.generate(
            hash64(20260524, "named-shape-closer-reference-contract", index),
            params={
                "queried_reference_label": reference_label,
                "target_shape_id": "star",
                "reference_a_shape_id": "circle",
                "reference_b_shape_id": "square",
                "reference_a_color_name": "red",
                "reference_b_color_name": "blue",
                "target_answer": 3,
                "target_icon_count": 7,
                "reference_axis_degrees_selected": 0,
            },
            max_attempts=300,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        entities = trace["scene_ir"]["entities"]
        references = {str(entity["label"]): entity for entity in entities if str(entity["role"]) == "reference"}
        targets = [entity for entity in entities if str(entity["role"]) == "target"]
        queried = str(execution["queried_reference_label"])
        other = "B" if queried == "A" else "A"
        counted = [
            entity
            for entity in targets
            if _distance(entity, references[queried]) < _distance(entity, references[other])
        ]

        assert out.scene_id == "named_field"
        assert out.query_id == "single"
        assert trace["query_spec"]["internal_query_id"] == QUERY_ID
        assert out.answer_gt.type == "integer"
        assert out.answer_gt.value == 3
        assert out.annotation_gt.type == "bbox_set"
        assert len(references) == 2
        assert set(references) == {"A", "B"}
        assert len(targets) == 7
        assert all(str(entity["shape_id"]) == "star" for entity in targets)
        assert all(str(entity["shape_id"]) != "star" for entity in references.values())
        assert len(counted) == 3
        assert len(out.annotation_gt.value) == 3
        assert set(trace["render_map"]["counted_instance_ids"]) == {str(entity["instance_id"]) for entity in counted}
        assert sorted(out.annotation_gt.value) == sorted(entity["bbox_xyxy"] for entity in counted)
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert len(trace["projected_annotation"]["pixel_point_set"]) == len(out.annotation_gt.value)
        assert trace["render_spec"]["style"]["text_legibility"]["failure_count"] == 0
        assert "reference_label_stroke_rgb" not in trace["render_spec"]["style"]
        assert all(entity["label_bbox_xyxy"] is None for entity in references.values())
        assert trace["execution_trace"]["question_format"] == "count_named_shape_icons_closer_to_named_reference"
        assert '"circle" icon' in out.prompt
        assert '"square" icon' in out.prompt
        assert '"star" icons' in out.prompt
        assert "reference A" not in out.prompt
        assert "reference B" not in out.prompt


def test_icons_counting_named_shape_closer_to_reference_supports_zero_answer() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260524, "named-shape-closer-reference-zero", 0),
        params={
            "queried_reference_label": "A",
            "target_shape_id": "triangle",
            "target_answer": 0,
            "target_icon_count": 5,
            "reference_axis_degrees_selected": 90,
        },
        max_attempts=300,
    )
    trace = out.trace_payload
    assert out.answer_gt.value == 0
    assert len(out.annotation_gt.value) == 0
    assert trace["execution_trace"]["closer_count_by_reference"]["A"] == 0
    assert trace["execution_trace"]["target_icon_count"] == 5


def test_icons_counting_named_shape_closer_to_reference_sampling_distribution() -> None:
    task = create_task(TASK_ID)
    query_counts: Counter[str] = Counter()
    reference_label_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    axes: set[int] = set()
    for index in range(120):
        out = task.generate(
            hash64(20260524, "named-shape-closer-reference-sampling", index),
            params={},
            max_attempts=300,
        )
        execution = out.trace_payload["execution_trace"]
        query_counts[str(out.trace_payload["query_spec"]["internal_query_id"])] += 1
        reference_label_counts[str(execution["queried_reference_label"])] += 1
        answer_counts[int(out.answer_gt.value)] += 1
        axes.add(int(execution["reference_axis_degrees"]))
        assert 4 <= int(execution["target_icon_count"]) <= 8
        assert 0 <= int(out.answer_gt.value) <= 4
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)

    assert set(query_counts) == {QUERY_ID}
    assert set(reference_label_counts) == set(QUERIED_REFERENCE_LABELS)
    assert set(answer_counts).issubset(set(range(0, 5)))
    assert len(answer_counts) >= 5
    assert axes.issubset({0, 35, 90, 145})
    assert axes
