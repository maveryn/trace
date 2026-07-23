"""Aggregate values for one repeated child label across treemap parents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.composition.values import int_sum
from trace_tasks.tasks.registry import register_task

from ._lifecycle import TreemapTaskPlan, run_treemap_task_from_public_class
from .shared.prompts import (
    ANNOTATION_HINT_REPEATED_LEAF,
    JSON_EXAMPLE_ANSWER_ONLY_REPEATED_LEAF,
    JSON_EXAMPLE_REPEATED_LEAF,
    render_prompt_artifacts,
)
from .shared.sampling import build_treemap_dataset
from .shared.state import DOMAIN


TASK_ID = "task_charts__treemap__repeated_leaf_aggregate_value"
SUM_QUERY_ID = "treemap_repeated_leaf_sum_value"
AVERAGE_QUERY_ID = "treemap_repeated_leaf_average_value"
SUPPORTED_QUERY_IDS = (SUM_QUERY_ID, AVERAGE_QUERY_ID)
QUERY_OPERATIONS = {
    SUM_QUERY_ID: "sum",
    AVERAGE_QUERY_ID: "average",
}
PROGRAM_CODE = "aggregate(value(child_label across parents), operation=sum_or_average); output=integer_value; annotation=bbox_set(repeated_child_rectangles); scene=treemap; scope=repeated_leaf_aggregate_value"


def _label_probability_map(labels: list[str]) -> dict[str, float]:
    if not labels:
        return {}
    probability = 1.0 / float(len(labels))
    return {str(label): float(probability) for label in labels}


def _build_repeated_leaf_plan(instance_seed: int, params: Mapping[str, Any], selected_branch: str) -> TreemapTaskPlan:
    """Bind all occurrences of one child label to a sum or average.

    The public task owns operation selection, the repeated child label, all
    matching leaf witnesses, prompt slots, and the integer aggregate answer.
    """

    operation = QUERY_OPERATIONS.get(str(selected_branch))
    if operation is None:
        raise ValueError(f"unsupported treemap repeated-leaf query_id: {selected_branch}")
    dataset = build_treemap_dataset(params, instance_seed=int(instance_seed))
    leaf_labels = sorted({str(leaf.label) for leaf in dataset.leaves})
    leaf_label = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{TASK_ID}.target_leaf_label"),
            tuple(str(label) for label in leaf_labels),
        )
    )
    label_index = int(leaf_labels.index(str(leaf_label)))
    matching = tuple(leaf for leaf in dataset.leaves if str(leaf.label) == leaf_label)
    if not matching:
        raise ValueError("treemap repeated-leaf target has no matching leaves")
    total = int_sum([int(leaf.value) for leaf in matching])
    if operation == "average":
        if total % len(matching) != 0:
            raise ValueError("treemap repeated-leaf average is not an integer")
        answer = int(total // len(matching))
    else:
        answer = int(total)
    prompt_artifacts = render_prompt_artifacts(
        prompt_key=str(selected_branch),
        annotation_hint=ANNOTATION_HINT_REPEATED_LEAF,
        json_example=JSON_EXAMPLE_REPEATED_LEAF,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY_REPEATED_LEAF,
        dynamic_slot_values={
            "parent_label": "",
            "leaf_label": str(leaf_label),
        },
        instance_seed=int(instance_seed),
    )
    annotation_leaf_ids = tuple(str(leaf.leaf_id) for leaf in matching)
    leaf_values = [int(leaf.value) for leaf in matching]
    parent_labels = [str(leaf.parent_label) for leaf in matching]
    relations = {
        "program_code": PROGRAM_CODE,
        "semantic_operation": "repeated_leaf_aggregate",
        "operation": str(operation),
        "leaf_label": str(leaf_label),
        "leaf_label_index": int(label_index),
        "leaf_label_probabilities": _label_probability_map(leaf_labels),
        "leaf_ids": [str(leaf_id) for leaf_id in annotation_leaf_ids],
        "leaf_values": [int(value) for value in leaf_values],
        "parent_labels": [str(label) for label in parent_labels],
    }
    return TreemapTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(answer)),
        answer_value=int(answer),
        annotation_leaf_ids=tuple(annotation_leaf_ids),
        prompt_artifacts=prompt_artifacts,
        relations=relations,
        witness_type="treemap_repeated_leaf_aggregate",
        witness_calculation={
            "operation": str(operation),
            "leaf_label": str(leaf_label),
            "leaf_values": [int(value) for value in leaf_values],
            "parent_labels": [str(label) for label in parent_labels],
        },
    )


@register_task
class ChartsCompositionTreemapRepeatedLeafAggregateValueTask:
    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "repeated_leaf_aggregate_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUM_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_treemap_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_build_repeated_leaf_plan,
        )


__all__ = ["ChartsCompositionTreemapRepeatedLeafAggregateValueTask"]
