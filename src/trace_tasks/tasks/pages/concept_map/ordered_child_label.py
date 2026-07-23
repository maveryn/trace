"""Concept-map task for reading an ordered child-item label."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import selected_child_annotation, target_payload
from .shared.defaults import DOMAIN, RANKED_CHILD_KIND


TASK_ID = "task_pages__concept_map__ordered_child_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "nth_child_label"
SCENE_VARIANTS = ("left_right_map", "clustered_map")
CASE_DEFAULTS = {
    "branch_count_min": 5,
    "branch_count_max": 6,
    "child_count_min": 5,
    "child_count_max": 6,
}


def _bind_ordered_child_label(selected_branch, branch_probabilities, case, rendered):
    target = target_payload(case)
    prompt_binding = _lifecycle.ConceptMapPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "branch_label": str(case.selection["branch_label"]),
            "rank_ordinal": str(case.selection["rank_ordinal"]),
            "reading_order": str(case.selection["reading_order"]),
        },
    )
    answer_binding = _lifecycle.string_binding(
        annotation_kind="bbox",
        annotation_value=selected_child_annotation(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=str(case.selection["answer"]),
        target_payload=target,
        question_format="concept_map_ordered_child_label",
    )
    return prompt_binding, answer_binding


@register_task
class PagesConceptMapOrderedChildLabelTask:
    """Read the ranked child-item label under a named concept-map branch."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology')
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
        task_params = dict(task_params)
        requested_layout = str(task_params.get("layout_variant", "")).strip()
        if requested_layout not in SCENE_VARIANTS:
            task_params["layout_variant"] = SCENE_VARIANTS[abs(int(instance_seed)) % len(SCENE_VARIANTS)]
        return _lifecycle.render_bound_concept_map(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_kind=RANKED_CHILD_KIND,
            case_defaults=CASE_DEFAULTS,
            binding_factory=_bind_ordered_child_label,
        )


__all__ = [
    "CASE_DEFAULTS",
    "PROMPT_QUERY_KEY",
    "SCENE_VARIANTS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesConceptMapOrderedChildLabelTask",
]
