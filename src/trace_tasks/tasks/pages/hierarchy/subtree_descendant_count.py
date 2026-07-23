"""Hierarchy task for counting descendants under one referenced node."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__hierarchy__subtree_descendant_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "subtree_descendant_count"
QUESTION_FORMAT = "hierarchy_org_chart"
TARGET_DESCENDANT_COUNT_MIN = 3
TARGET_DESCENDANT_COUNT_MAX = 7


def _build_descendant_objective() -> _lifecycle.HierarchyObjectiveBinding:
    """Bind descendant-count answer and annotation semantics to this public task."""

    return _lifecycle.HierarchyObjectiveBinding(
        semantic_branch_key=PROMPT_QUERY_KEY,
        prompt_branch_key=PROMPT_QUERY_KEY,
        answer_type="integer",
        annotation_type="bbox_set",
        question_format=QUESTION_FORMAT,
    )


@register_task
class PagesHierarchySubtreeDescendantCountTask:
    """Count all people working under one named manager in an org chart."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one descendant-count hierarchy instance."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        task_params = {
            **task_params,
            "subtree_descendant_count_min": TARGET_DESCENDANT_COUNT_MIN,
            "subtree_descendant_count_max": TARGET_DESCENDANT_COUNT_MAX,
        }
        objective = _build_descendant_objective()
        return _lifecycle.render_bound_hierarchy(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            objective=objective,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "QUESTION_FORMAT",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesHierarchySubtreeDescendantCountTask",
]
