"""Sectioned-infographic task for counting item rows in one named section."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle


TASK_ID = "task_pages__sectioned_infographic__section_item_count"
PROMPT_QUERY_KEY = "section_item_count"
TASK_NAMESPACE = "pages.sectioned_infographic.item_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
SCENE_VARIANTS = _lifecycle.SCENE_VARIANTS


@register_task
class PagesSectionedInfographicSectionItemCountTask:
    """Count visible list items in one named infographic section."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Mapping[str, Any],
        max_attempts: int,
    ):
        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
        )
        return _lifecycle.build_sectioned_infographic_response(
            instance_seed=int(instance_seed),
            params=task_params,
            task_namespace=TASK_NAMESPACE,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            prompt_query_key=PROMPT_QUERY_KEY,
            question_format="sectioned_infographic_section_item_count",
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SCENE_VARIANTS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_NAMESPACE",
    "PagesSectionedInfographicSectionItemCountTask",
]
