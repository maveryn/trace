"""Hierarchy task for selecting the manager with the largest total reporting chain."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__hierarchy__manager_most_total_reports_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "manager_most_total_reports_label"
QUESTION_FORMAT = "hierarchy_org_chart"
_RETRYABLE_CONSTRUCTION_ERRORS = frozenset(
    {
        "constructed tree node count fell outside configured bounds",
        "constructed tree depth fell outside configured bounds",
        "constructed org chart winner does not match requested winner",
        "manager extremum construction did not produce a unique winner",
    }
)


def _build_total_reports_objective() -> _lifecycle.HierarchyObjectiveBinding:
    """Bind total-report extremum answer and annotation semantics."""

    return _lifecycle.HierarchyObjectiveBinding(
        semantic_branch_key=PROMPT_QUERY_KEY,
        prompt_branch_key=PROMPT_QUERY_KEY,
        answer_type="string",
        annotation_type="bbox",
        question_format=QUESTION_FORMAT,
    )


@register_task
class PagesHierarchyManagerMostTotalReportsLabelTask:
    """Select the non-CEO manager with the most total people below them."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'ranking', 'topology')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one total-report manager extremum hierarchy instance."""

        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        objective = _build_total_reports_objective()
        attempts = max(1, int(max_attempts))
        last_error: ValueError | None = None
        for attempt_index in range(attempts):
            attempt_seed = (
                int(instance_seed)
                if attempt_index == 0
                else int(
                    hash64(
                        int(instance_seed),
                        f"{TASK_ID}.hierarchy_retry",
                        int(attempt_index),
                    )
                )
            )
            try:
                output = _lifecycle.render_bound_hierarchy(
                    instance_seed=int(attempt_seed),
                    params=task_params,
                    selected_branch=str(selected_branch),
                    branch_probabilities=branch_probabilities,
                    objective=objective,
                )
            except ValueError as exc:
                if str(exc) not in _RETRYABLE_CONSTRUCTION_ERRORS:
                    raise
                last_error = exc
                continue

            attempt_meta = {
                "generation_attempt_index": int(attempt_index),
                "generation_attempt_seed": int(attempt_seed),
            }
            output.trace_payload["query_spec"]["params"].update(attempt_meta)
            output.trace_payload["execution_trace"].update(attempt_meta)
            return output

        raise RuntimeError(
            f"{TASK_ID} failed to construct a valid hierarchy after {attempts} attempts"
        ) from last_error


__all__ = [
    "PROMPT_QUERY_KEY",
    "QUESTION_FORMAT",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesHierarchyManagerMostTotalReportsLabelTask",
]
