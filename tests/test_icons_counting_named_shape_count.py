"""Tests for procedural named-shape icon counting."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.icons.shared.procedural_named_icons import PROCEDURAL_NAMED_ICON_SHAPES


TASK_ID = "task_icons__named_field__single_attribute_membership_count"


def test_icons_counting_named_shape_count_contract() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260523, "named-shape-contract", 0),
        params={
            "shape_id": "star",
            "target_count": 3,
            "object_count": 22,
            "arrangement_mode": "ordered_grid",
        },
        max_attempts=200,
    )
    trace = out.trace_payload
    assert out.scene_id == "named_field"
    assert out.query_id == "single"
    assert trace["query_spec"]["internal_query_id"] == "named_shape_count"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 3
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 3
    assert '"star" icons' in out.prompt
    assert trace["execution_trace"]["target_shape_id"] == "star"
    assert trace["execution_trace"]["target_shape_name"] == "star"
    assert trace["execution_trace"]["shape_counts"]["star"] == 3
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    assert len(trace["projected_annotation"]["pixel_point_set"]) == len(out.annotation_gt.value)
    counted_ids = set(trace["render_map"]["counted_instance_ids"])
    entity_by_id = {str(entity["instance_id"]): entity for entity in trace["scene_ir"]["entities"]}
    assert len(entity_by_id) == 22
    assert all(entity_by_id[instance_id]["shape_id"] == "star" for instance_id in counted_ids)
    annotation_bboxes = sorted(entity_by_id[instance_id]["bbox_xyxy"] for instance_id in counted_ids)
    assert sorted(out.annotation_gt.value) == annotation_bboxes


def test_icons_counting_named_shape_count_targeted_generation_supports_all_shapes() -> None:
    task = create_task(TASK_ID)
    shape_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    for index, shape_id in enumerate(PROCEDURAL_NAMED_ICON_SHAPES):
        out = task.generate(
            hash64(20260523, "named-shape-sampling", index),
            params={"target_shape_id": shape_id, "target_count": (index % 5) + 1},
            max_attempts=200,
        )
        execution = out.trace_payload["execution_trace"]
        shape_counts[str(execution["target_shape_id"])] += 1
        answer_counts[int(out.answer_gt.value)] += 1
        assert int(execution["object_count"]) >= int(out.answer_gt.value)
        assert int(execution["object_count"]) <= 36
        assert 1 <= int(out.answer_gt.value) <= 14
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert set(shape_counts) == set(PROCEDURAL_NAMED_ICON_SHAPES)
    assert set(answer_counts).issubset(set(range(1, 26)))


def test_icons_counting_named_shape_count_all_arrangement_modes_generate() -> None:
    task = create_task(TASK_ID)
    modes = {
        "jittered_grid",
        "ordered_grid",
        "shelf_rows",
        "free_scatter",
        "clustered_by_shape",
        "shape_stacks",
        "target_stack_with_oddballs",
        "mixed_stacks",
    }
    observed: set[str] = set()
    for index, mode in enumerate(sorted(modes)):
        out = task.generate(
            hash64(20260523, f"named-shape-mode-{mode}", index),
            params={"arrangement_mode": mode},
            max_attempts=300,
        )
        execution = out.trace_payload["execution_trace"]
        observed.add(str(execution["arrangement_mode"]))
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert observed == modes


def test_icons_counting_named_shape_count_shape_stack_is_compact() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260523, "named-shape-compact-stack", 0),
        params={
            "arrangement_mode": "shape_stacks",
            "target_shape_id": "star",
            "target_count": 12,
            "object_count": 28,
        },
        max_attempts=300,
    )
    target_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["shape_id"] == "star"
    ]
    assert len(target_entities) == 12
    rows: dict[int, list[dict[str, object]]] = {}
    for entity in target_entities:
        rows.setdefault(int(entity["layout_row"]), []).append(entity)
    horizontal_gaps: list[int] = []
    for row_entities in rows.values():
        ordered = sorted(row_entities, key=lambda entity: int(entity["layout_col"]))
        for left, right in zip(ordered, ordered[1:]):
            horizontal_gaps.append(int(right["bbox_xyxy"][0]) - int(left["bbox_xyxy"][2]))
    assert horizontal_gaps
    assert max(horizontal_gaps) <= 2


def test_icons_counting_named_shape_count_target_stack_has_one_oddball_only() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260523, "named-shape-one-oddball-stack", 0),
        params={
            "arrangement_mode": "target_stack_with_oddballs",
            "target_shape_id": "capsule",
            "target_count": 12,
        },
        max_attempts=300,
    )
    entities = out.trace_payload["scene_ir"]["entities"]
    shape_counts: Counter[str] = Counter(str(entity["shape_id"]) for entity in entities)
    placement_groups: Counter[str] = Counter(str(entity["placement_group"]) for entity in entities)
    details = out.trace_payload["query_spec"]["params"]["arrangement_details"]

    assert out.answer_gt.value == 12
    assert len(entities) == 13
    assert shape_counts["capsule"] == 12
    assert len(shape_counts) == 2
    assert sorted(shape_counts.values()) == [1, 12]
    assert placement_groups == Counter({"target_stack:capsule": 13})
    assert details["oddball_count"] == 1
    assert details["target_stack_total"] == 13
    assert details["stack_distractor_shape_count"] == 1
