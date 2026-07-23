"""Concept-map task for counting marker-filtered child items."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import count_annotation, target_payload
from .shared.defaults import DOMAIN, LAYOUT_VARIANTS as SCENE_VARIANTS, MARKED_TOTAL_KIND


TASK_ID = "task_pages__concept_map__marked_child_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "marked_child_count"


def _bind_marked_child_count(selected_branch, branch_probabilities, case, rendered):
    target = target_payload(case)
    prompt_binding = _lifecycle.ConceptMapPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "branch_label": str(case.selection["branch_label"]),
            "marker_label": str(case.selection["marker_label"]),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=count_annotation(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(case.selection["answer"]),
        target_payload=target,
        question_format="concept_map_marked_child_count",
    )
    return prompt_binding, answer_binding


@register_task
class PagesConceptMapMarkedChildCountTask:
    """Count marked child-item nodes directly under a named concept-map branch."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
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
        return _lifecycle.render_bound_concept_map(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_kind=MARKED_TOTAL_KIND,
            case_defaults={},
            binding_factory=_bind_marked_child_count,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SCENE_VARIANTS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesConceptMapMarkedChildCountTask",
]
