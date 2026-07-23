"""Category-grid lookup task for counting item rows."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import item_count_annotation, target_payload_for_count
from .shared.defaults import DOMAIN, SCENE_VARIANTS
from .shared.sampling import build_item_count_case


TASK_ID = "task_pages__category_grid__category_item_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "category_item_count"


def _bind_item_count(selected_branch, branch_probabilities, case, rendered):
    target_payload = target_payload_for_count(case)
    prompt_binding = _lifecycle.CategoryGridPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "category_label": str(case.target_category.label),
            "subcategory_label": str(case.target_subcategory.label),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=item_count_annotation(case, rendered.rendered_grid),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(len(case.target_subcategory.items)),
        target_payload=target_payload,
        question_format="category_grid_category_item_count",
    )
    return prompt_binding, answer_binding


@register_task
class PagesCategoryGridCategoryItemCountTask:
    """Count item rows inside one category/subcategory block."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
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
            case_factory=build_item_count_case,
            binding_factory=_bind_item_count,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SCENE_VARIANTS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesCategoryGridCategoryItemCountTask",
]
