"""Sum all printed child values inside one treemap parent category."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.composition.values import int_sum
from trace_tasks.tasks.registry import register_task

from ._lifecycle import TreemapTaskPlan, run_treemap_task_from_public_class
from .shared.prompts import (
    ANNOTATION_HINT_GROUP_TOTAL,
    JSON_EXAMPLE_ANSWER_ONLY_GROUP_TOTAL,
    JSON_EXAMPLE_GROUP_TOTAL,
    render_prompt_artifacts,
)
from .shared.sampling import build_treemap_dataset
from .shared.state import DOMAIN


TASK_ID = "task_charts__treemap__group_total_value"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "treemap_group_total_value"
PROGRAM_CODE = "sum(value(child) for child in parent); output=integer_value; annotation=bbox_set(parent_child_rectangles); scene=treemap; scope=group_total_value"


def _parent_selection_probabilities(labels: list[str], selected: str) -> dict[str, float]:
    if not labels:
        return {}
    probability = 1.0 / float(len(labels))
    return {str(label): float(probability) for label in labels if str(label) != str(selected)} | {str(selected): float(probability)}


def _parent_leaf_values(dataset, leaf_ids: tuple[str, ...]) -> list[int]:
    leaves_by_id = {str(leaf.leaf_id): leaf for leaf in dataset.leaves}
    return [int(leaves_by_id[str(leaf_id)].value) for leaf_id in leaf_ids]


def _parent_total_relations(*, dataset, parent, parent_index: int, leaf_ids: tuple[str, ...], leaf_values: list[int]) -> dict[str, Any]:
    return {
        "program_code": PROGRAM_CODE,
        "semantic_operation": "parent_child_sum",
        "operation": "sum",
        "parent_id": str(parent.parent_id),
        "parent_label": str(parent.label),
        "parent_index": int(parent_index),
        "parent_selection_probabilities": _parent_selection_probabilities(
            [str(item.label) for item in dataset.parents],
            str(parent.label),
        ),
        "leaf_ids": [str(leaf_id) for leaf_id in leaf_ids],
        "leaf_values": [int(value) for value in leaf_values],
    }


def _build_group_total_plan(instance_seed: int, params: Mapping[str, Any], selected_branch: str) -> TreemapTaskPlan:
    """Bind one parent rectangle to its child values and total.

    The public task owns the selected parent, supporting leaf ids, integer
    answer, prompt slots, and symbolic witness; rendering remains scene-local.
    """

    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported treemap group-total query_id: {selected_branch}")
    dataset = build_treemap_dataset(params, instance_seed=int(instance_seed))
    parent = uniform_choice(
        spawn_rng(int(instance_seed), f"{TASK_ID}.target_parent"),
        tuple(dataset.parents),
    )
    parent_index = int(tuple(dataset.parents).index(parent))
    annotation_leaf_ids = tuple(str(leaf_id) for leaf_id in parent.leaf_ids)
    leaf_values = _parent_leaf_values(dataset, annotation_leaf_ids)
    answer = int_sum(leaf_values)
    if answer != int(parent.value):
        raise ValueError("treemap parent total mismatch")
    prompt_artifacts = render_prompt_artifacts(
        prompt_key=PROMPT_KEY,
        annotation_hint=ANNOTATION_HINT_GROUP_TOTAL,
        json_example=JSON_EXAMPLE_GROUP_TOTAL,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY_GROUP_TOTAL,
        dynamic_slot_values={
            "parent_label": str(parent.label),
            "leaf_label": "",
        },
        instance_seed=int(instance_seed),
    )
    return TreemapTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(answer)),
        answer_value=int(answer),
        annotation_leaf_ids=tuple(annotation_leaf_ids),
        prompt_artifacts=prompt_artifacts,
        relations=_parent_total_relations(
            dataset=dataset,
            parent=parent,
            parent_index=int(parent_index),
            leaf_ids=annotation_leaf_ids,
            leaf_values=leaf_values,
        ),
        witness_type="treemap_parent_child_sum",
        witness_calculation={
            "operation": "sum",
            "parent_label": str(parent.label),
            "leaf_values": [int(value) for value in leaf_values],
        },
    )


@register_task
class ChartsCompositionTreemapGroupTotalValueTask:
    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "group_total_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_treemap_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_build_group_total_plan,
        )


__all__ = ["ChartsCompositionTreemapGroupTotalValueTask"]
