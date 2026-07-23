"""Category-grid lookup task for an ordinal item label."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import slot_item_annotation, target_payload_for_slot
from .shared.defaults import DOMAIN, SCENE_VARIANTS
from .shared.sampling import build_slot_item_case


TASK_ID = "task_pages__category_grid__category_slot_item_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "category_slot_item_label"


def _bind_slot_item(selected_branch, branch_probabilities, case, rendered):
    if case.target_item is None or case.target_slot_index is None:
        raise ValueError("category slot item task requires a selected target item")
    target_payload = target_payload_for_slot(case)
    prompt_binding = _lifecycle.CategoryGridPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "category_label": str(case.target_category.label),
            "subcategory_label": str(case.target_subcategory.label),
            "slot_ordinal": str(target_payload["slot_ordinal"]),
        },
    )
    answer_binding = _lifecycle.string_binding(
        annotation_kind="bbox_map",
        annotation_value=slot_item_annotation(case, rendered.rendered_grid),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=str(case.target_item.label),
        target_payload=target_payload,
        question_format="category_grid_category_slot_item_label",
    )
    return prompt_binding, answer_binding


@register_task
class PagesCategoryGridCategorySlotItemLabelTask:
    """Read an ordinal item label from a category/subcategory grid."""

    task_id = TASK_ID
    reasoning_operations = ('direct_retrieval',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        return _lifecycle.render_bound_category_grid(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_factory=build_slot_item_case,
            binding_factory=_bind_slot_item,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SCENE_VARIANTS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesCategoryGridCategorySlotItemLabelTask",
]
