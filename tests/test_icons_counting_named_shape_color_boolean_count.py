"""Tests for named-shape / visual-attribute Boolean icon counting."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.icons.named_field.multi_attribute_and_count import (
    INTERNAL_QUERY_ID as AND_INTERNAL_QUERY_ID,
    SUPPORTED_QUERY_IDS as AND_SUPPORTED_QUERY_IDS,
    TASK_ID as AND_TASK_ID,
)
from trace_tasks.tasks.icons.named_field.multi_attribute_complement_count import (
    INTERNAL_QUERY_ID as COMPLEMENT_INTERNAL_QUERY_ID,
    SUPPORTED_QUERY_IDS as COMPLEMENT_SUPPORTED_QUERY_IDS,
    TASK_ID as COMPLEMENT_TASK_ID,
)
from trace_tasks.tasks.icons.named_field.multi_attribute_exclusion_count import (
    ATTRIBUTE_WITHOUT_SHAPE_QUERY_ID,
    SHAPE_WITHOUT_ATTRIBUTE_QUERY_ID,
    SUPPORTED_QUERY_IDS as EXCLUSION_SUPPORTED_QUERY_IDS,
    TASK_ID as EXCLUSION_TASK_ID,
)
from trace_tasks.tasks.icons.named_field.multi_attribute_or_count import (
    INTERNAL_QUERY_ID as OR_INTERNAL_QUERY_ID,
    SUPPORTED_QUERY_IDS as OR_SUPPORTED_QUERY_IDS,
    TASK_ID as OR_TASK_ID,
)
from trace_tasks.tasks.icons.named_field.multi_attribute_xor_count import (
    INTERNAL_QUERY_ID as XOR_INTERNAL_QUERY_ID,
    SUPPORTED_QUERY_IDS as XOR_SUPPORTED_QUERY_IDS,
    TASK_ID as XOR_TASK_ID,
)


TASK_CASES = (
    (AND_TASK_ID, AND_INTERNAL_QUERY_ID, AND_SUPPORTED_QUERY_IDS),
    (OR_TASK_ID, OR_INTERNAL_QUERY_ID, OR_SUPPORTED_QUERY_IDS),
    (EXCLUSION_TASK_ID, SHAPE_WITHOUT_ATTRIBUTE_QUERY_ID, EXCLUSION_SUPPORTED_QUERY_IDS),
    (EXCLUSION_TASK_ID, ATTRIBUTE_WITHOUT_SHAPE_QUERY_ID, EXCLUSION_SUPPORTED_QUERY_IDS),
    (COMPLEMENT_TASK_ID, COMPLEMENT_INTERNAL_QUERY_ID, COMPLEMENT_SUPPORTED_QUERY_IDS),
    (XOR_TASK_ID, XOR_INTERNAL_QUERY_ID, XOR_SUPPORTED_QUERY_IDS),
)
QUERY_IDS = tuple(internal_query_id for _task_id, internal_query_id, _supported in TASK_CASES)
TASK_ID_BY_QUERY_ID = {internal_query_id: task_id for task_id, internal_query_id, _supported in TASK_CASES}
SUPPORTED_QUERY_IDS_BY_TASK_ID = {
    AND_TASK_ID: AND_SUPPORTED_QUERY_IDS,
    OR_TASK_ID: OR_SUPPORTED_QUERY_IDS,
    EXCLUSION_TASK_ID: EXCLUSION_SUPPORTED_QUERY_IDS,
    COMPLEMENT_TASK_ID: COMPLEMENT_SUPPORTED_QUERY_IDS,
    XOR_TASK_ID: XOR_SUPPORTED_QUERY_IDS,
}


def _predicate(query_id: str, *, is_shape: bool, is_attribute: bool) -> bool:
    if query_id == "shape_and_color_count":
        return is_shape and is_attribute
    if query_id == "shape_or_color_count":
        return is_shape or is_attribute
    if query_id == "shape_and_not_color_count":
        return is_shape and not is_attribute
    if query_id == "color_and_not_shape_count":
        return is_attribute and not is_shape
    if query_id == "neither_shape_nor_color_count":
        return (not is_shape) and (not is_attribute)
    if query_id == "exactly_one_shape_or_color_count":
        return bool(is_shape) ^ bool(is_attribute)
    raise AssertionError(f"unexpected query_id: {query_id}")


def test_icons_counting_named_shape_color_boolean_contract_all_queries() -> None:
    for index, (task_id, query_id, public_query_ids) in enumerate(TASK_CASES):
        task = create_task(TASK_ID_BY_QUERY_ID[str(query_id)])
        params = {
            "attribute_axis": "color",
            "target_shape_id": "star",
            "target_color_name": "red",
            "target_count": 4,
            "object_count": 7,
            "arrangement_mode": "ordered_grid",
        }
        if public_query_ids != ("single",):
            params["query_id"] = query_id
        out = task.generate(
            hash64(20260523, "named-shape-color-boolean-contract", index),
            params=params,
            max_attempts=200,
        )
        trace = out.trace_payload
        entities = trace["scene_ir"]["entities"]
        target_shape = trace["query_spec"]["params"]["target_shape_id"]
        target_color = trace["query_spec"]["params"]["target_color_name"]
        counted_entities = [
            entity
            for entity in entities
            if _predicate(
                query_id,
                is_shape=str(entity["shape_id"]) == str(target_shape),
                is_attribute=str(entity["color_name"]) == str(target_color),
            )
        ]

        assert out.scene_id == "named_field"
        assert out.query_id == ("single" if public_query_ids == ("single",) else query_id)
        assert trace["query_spec"]["internal_query_id"] == query_id
        assert out.answer_gt.type == "integer"
        assert out.answer_gt.value == 4
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == 4
        assert len(counted_entities) == 4
        assert "star" in out.prompt
        assert "red [#E63232]" in out.prompt
        assert all("color_name" in entity for entity in entities)
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert len(trace["projected_annotation"]["pixel_point_set"]) == len(out.annotation_gt.value)
        assert set(trace["render_map"]["counted_instance_ids"]) == {str(entity["instance_id"]) for entity in counted_entities}
        assert sorted(out.annotation_gt.value) == sorted(entity["bbox_xyxy"] for entity in counted_entities)


def test_icons_counting_named_shape_color_boolean_rejects_fill_style_axis() -> None:
    query_id = "shape_and_color_count"
    task = create_task(TASK_ID_BY_QUERY_ID[str(query_id)])
    params = {
        "attribute_axis": "fill_style",
        "target_shape_id": "star",
        "target_count": 3,
        "object_count": 6,
        "arrangement_mode": "ordered_grid",
    }
    if tuple(getattr(task, "supported_query_ids", ())) != ("single",):
        params["query_id"] = query_id
    try:
        task.generate(
            hash64(20260523, "named-shape-fill-style-boolean-contract", 0),
            params=params,
            max_attempts=20,
        )
    except (RuntimeError, ValueError) as exc:
        assert "attribute_axis" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Boolean named-icon task accepted fill_style as a semantic axis")


def test_icons_counting_named_shape_color_boolean_sampling_distribution() -> None:
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    attribute_axes: Counter[str] = Counter()
    layouts: set[str] = set()
    for task_id, supported_queries in SUPPORTED_QUERY_IDS_BY_TASK_ID.items():
        task = create_task(str(task_id))
        task_query_counts: Counter[str] = Counter()
        for index in range(24):
            out = task.generate(
                hash64(20260523, f"named-shape-color-boolean-sampling.{task_id}", index),
                params={},
                max_attempts=200,
            )
            execution = out.trace_payload["execution_trace"]
            public_query_id = str(execution["query_id"])
            internal_query_id = str(execution.get("internal_query_id") or public_query_id)
            task_query_counts[public_query_id] += 1
            query_counts[internal_query_id] += 1
            answer_counts[int(out.answer_gt.value)] += 1
            attribute_axes[str(execution["target_attribute_axis"])] += 1
            layouts.add(str(execution["arrangement_mode"]))
            assert public_query_id in set(supported_queries)
            assert 4 <= int(execution["object_count"]) <= 10
            assert 1 <= int(out.answer_gt.value) <= 5
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert set(task_query_counts).issubset(set(supported_queries))

    assert set(query_counts) == set(QUERY_IDS)
    assert set(answer_counts).issubset(set(range(1, 6)))
    assert set(attribute_axes) == {"color"}
    assert layouts.issubset({"jittered_grid", "ordered_grid", "shelf_rows", "free_scatter", "clustered_by_shape"})
    assert layouts


def test_icons_counting_named_shape_color_boolean_rejects_stack_layouts() -> None:
    task = create_task("task_icons__named_field__multi_attribute_and_count")
    try:
        task.generate(
            hash64(20260523, "named-shape-color-boolean-stack-reject", 0),
            params={"arrangement_mode": "shape_stacks"},
            max_attempts=20,
        )
    except RuntimeError as exc:
        assert "non-stack layouts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Boolean named-icon task accepted a stack layout")
