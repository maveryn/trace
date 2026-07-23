"""Select the treemap parent category with the largest or smallest total."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.composition.values import select_unique_extremum
from trace_tasks.tasks.registry import register_task

from ._lifecycle import TreemapTaskPlan, run_treemap_task_from_public_class
from .shared.prompts import (
    ANSWER_HINT_PARENT_LABEL,
    ANNOTATION_HINT_PARENT_EXTREMUM,
    JSON_EXAMPLE_ANSWER_ONLY_PARENT_EXTREMUM,
    JSON_EXAMPLE_PARENT_EXTREMUM,
    render_prompt_artifacts,
)
from .shared.sampling import build_treemap_dataset
from .shared.state import DOMAIN


TASK_ID = "task_charts__treemap__parent_total_extremum_label"
LARGEST_QUERY_ID = "largest_parent_total"
SMALLEST_QUERY_ID = "smallest_parent_total"
SUPPORTED_QUERY_IDS = (LARGEST_QUERY_ID, SMALLEST_QUERY_ID)
PROMPT_KEY = "treemap_parent_total_extremum_label"
PROGRAM_CODE = "select(parent where sum(value(child) for child in parent) is extremum(direction)); direction={largest,smallest}; output=string_label; annotation=bbox_set(parent_child_rectangles); scene=treemap; scope=parent_total_extremum_label"


def _build_parent_total_extremum_plan(
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
) -> TreemapTaskPlan:
    """Bind one parent-total extremum and all of its child rectangles."""

    if str(selected_branch) not in SUPPORTED_QUERY_IDS:
        raise ValueError(f"unsupported treemap parent-total extremum query_id: {selected_branch}")
    dataset = build_treemap_dataset(params, instance_seed=int(instance_seed))
    select_largest = str(selected_branch) == LARGEST_QUERY_ID
    selected = select_unique_extremum(
        tuple((parent, int(parent.value)) for parent in dataset.parents),
        select_largest=bool(select_largest),
        min_margin=1,
        error_label="treemap parent-total extremum",
        item_label="parents",
    )
    answer_parent = selected.item
    extremum_word = "largest" if select_largest else "smallest"
    prompt_artifacts = render_prompt_artifacts(
        prompt_key=PROMPT_KEY,
        annotation_hint=ANNOTATION_HINT_PARENT_EXTREMUM,
        json_example=JSON_EXAMPLE_PARENT_EXTREMUM,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY_PARENT_EXTREMUM,
        dynamic_slot_values={
            "answer_hint": ANSWER_HINT_PARENT_LABEL,
            "parent_label": "",
            "leaf_label": "",
            "extremum_word": str(extremum_word),
        },
        instance_seed=int(instance_seed),
    )
    parent_totals = [
        {
            "parent_id": str(parent.parent_id),
            "parent_label": str(parent.label),
            "parent_total": int(parent.value),
        }
        for parent in dataset.parents
    ]
    annotation_leaf_ids = tuple(str(leaf_id) for leaf_id in answer_parent.leaf_ids)
    relations = {
        "program_code": PROGRAM_CODE,
        "semantic_operation": "parent_total_extremum_selection",
        "operation": "extremum",
        "extremum_direction": str(selected_branch),
        "extremum_word": str(extremum_word),
        "answer_parent_id": str(answer_parent.parent_id),
        "answer_parent_label": str(answer_parent.label),
        "answer_parent_total": int(answer_parent.value),
        "nearest_parent_total_margin": int(selected.margin),
        "parent_totals": parent_totals,
        "leaf_ids": [str(leaf_id) for leaf_id in annotation_leaf_ids],
    }
    return TreemapTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="string", value=str(answer_parent.label)),
        answer_value=str(answer_parent.label),
        annotation_leaf_ids=annotation_leaf_ids,
        prompt_artifacts=prompt_artifacts,
        relations=relations,
        witness_type="treemap_parent_total_extremum",
        witness_calculation={
            "operation": "parent_total_extremum",
            "direction": str(selected_branch),
            "parent_totals": parent_totals,
            "answer_parent_label": str(answer_parent.label),
            "answer_parent_total": int(answer_parent.value),
        },
    )


@register_task
class ChartsCompositionTreemapParentTotalExtremumLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "parent_total_extremum_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = LARGEST_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_treemap_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_build_parent_total_extremum_plan,
        )


__all__ = [
    "ChartsCompositionTreemapParentTotalExtremumLabelTask",
    "LARGEST_QUERY_ID",
    "SMALLEST_QUERY_ID",
]
