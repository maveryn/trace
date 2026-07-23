"""Tests for named-field counterfactual icon counting tasks."""

from __future__ import annotations

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.icons.named_field.counterfactual_attribute_count import (
    INTERNAL_QUERY_ID as ATTRIBUTE_INTERNAL_QUERY_ID,
    SUPPORTED_QUERY_IDS as ATTRIBUTE_SUPPORTED_QUERY_IDS,
)
from trace_tasks.tasks.icons.named_field.counterfactual_total_count import (
    INTERNAL_QUERY_ID as TOTAL_INTERNAL_QUERY_ID,
    SUPPORTED_QUERY_IDS as TOTAL_SUPPORTED_QUERY_IDS,
)


ATTRIBUTE_TASK_ID = "task_icons__named_field__counterfactual_attribute_count"
TOTAL_TASK_ID = "task_icons__named_field__counterfactual_total_count"
REMOVED_QUERY_ID = "target_count_after_remove_and_replace"


def test_named_field_counterfactual_public_query_support() -> None:
    """Counterfactual tasks expose only their current public query contracts."""

    assert ATTRIBUTE_SUPPORTED_QUERY_IDS == ("single",)
    assert TOTAL_SUPPORTED_QUERY_IDS == ("single",)
    assert ATTRIBUTE_INTERNAL_QUERY_ID == "target_count_after_shape_replacement"
    assert TOTAL_INTERNAL_QUERY_ID == "total_count_after_shape_removal"
    assert REMOVED_QUERY_ID not in {ATTRIBUTE_INTERNAL_QUERY_ID, TOTAL_INTERNAL_QUERY_ID}


def test_named_field_counterfactual_attribute_rejects_removed_query() -> None:
    task = create_task(ATTRIBUTE_TASK_ID)

    with pytest.raises(ValueError, match="only supports query_id values"):
        task.generate(
            hash64(20260614, "named-field-counterfactual-removed-query"),
            params={"counterfactual_query_id": REMOVED_QUERY_ID},
            max_attempts=20,
        )


def test_named_field_counterfactual_current_queries_generate() -> None:
    attribute_task = create_task(ATTRIBUTE_TASK_ID)
    total_task = create_task(TOTAL_TASK_ID)

    attribute_out = attribute_task.generate(
        hash64(20260614, "named-field-counterfactual-attribute"),
        params={"counterfactual_query_id": "target_count_after_shape_replacement", "target_count": 3},
        max_attempts=200,
    )
    total_out = total_task.generate(
        hash64(20260614, "named-field-counterfactual-total"),
        params={"counterfactual_query_id": "total_count_after_shape_removal", "target_count": 4},
        max_attempts=200,
    )

    assert attribute_out.query_id == "single"
    assert attribute_out.trace_payload["query_spec"]["internal_query_id"] == "target_count_after_shape_replacement"
    assert attribute_out.answer_gt.type == "integer"
    assert attribute_out.answer_gt.value == 3
    assert attribute_out.annotation_gt.type == "bbox_set"
    assert len(attribute_out.annotation_gt.value) == 3

    assert total_out.query_id == "single"
    assert total_out.trace_payload["query_spec"]["internal_query_id"] == "total_count_after_shape_removal"
    assert total_out.answer_gt.type == "integer"
    assert total_out.answer_gt.value == 4
    assert total_out.annotation_gt.type == "bbox_set"
    assert len(total_out.annotation_gt.value) == 4
