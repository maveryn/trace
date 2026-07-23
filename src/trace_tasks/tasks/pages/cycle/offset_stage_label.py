"""Cycle task for returning a stage label at a before/after offset along arrows."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import answer_stage_bbox
from .shared.defaults import DOMAIN, SCENE_VARIANTS
from .shared.rendering import render_cycle_case
from .shared.sampling import build_cycle_case


TASK_ID = "task_pages__cycle__offset_stage_label"
AFTER_QUERY_ID = "after_stage_offset_label"
BEFORE_QUERY_ID = "before_stage_offset_label"
SUPPORTED_QUERY_IDS = (AFTER_QUERY_ID, BEFORE_QUERY_ID)
DEFAULT_QUERY_ID = AFTER_QUERY_ID
PROMPT_QUERY_KEY = "offset_stage_label"

_QUERY_RELATIONSHIP_BY_QUERY_ID = {
    AFTER_QUERY_ID: "after",
    BEFORE_QUERY_ID: "before",
}


def _bind_offset_stage(selected_branch, branch_probabilities, case, rendered):
    prompt_binding = _lifecycle.CyclePromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots=dict(case.prompt_slots),
    )
    answer_binding = _lifecycle.string_binding(
        annotation_kind="bbox",
        annotation_value=answer_stage_bbox(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=str(case.answer_stage_label),
        target_payload={
            "query_stage_id": str(case.query_stage_id),
            "query_stage_label": str(case.query_stage_label),
            "answer_stage_id": str(case.answer_stage_id),
            "answer_stage_label": str(case.answer_stage_label),
            "query_relationship": str(case.query_relationship),
            "cycle_direction": str(case.cycle_direction),
            "step_count": int(case.step_count),
        },
        question_format="cycle_offset_stage_label",
    )
    return prompt_binding, answer_binding


@register_task
class PagesCycleOffsetStageLabelTask:
    """Return the exact visible stage reached by walking along a directed cycle."""

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
            default=DEFAULT_QUERY_ID,
            public_task=TASK_ID,
        )
        query_relationship = str(_QUERY_RELATIONSHIP_BY_QUERY_ID[str(selected_branch)])
        relationship_probabilities = {
            str(_QUERY_RELATIONSHIP_BY_QUERY_ID[str(query_id)]): float(probability)
            for query_id, probability in branch_probabilities.items()
        }
        case = build_cycle_case(
            int(instance_seed),
            params=task_params,
            query_relationship=query_relationship,
            query_relationship_probabilities=relationship_probabilities,
        )
        rendered = render_cycle_case(instance_seed=int(instance_seed), params=task_params, case=case)
        prompt_binding, answer_binding = _bind_offset_stage(str(selected_branch), branch_probabilities, case, rendered)
        return _lifecycle.build_cycle_response(
            instance_seed=int(instance_seed),
            case=case,
            rendered=rendered,
            prompt_binding=prompt_binding,
            answer_binding=answer_binding,
        )


__all__ = [
    "AFTER_QUERY_ID",
    "BEFORE_QUERY_ID",
    "DEFAULT_QUERY_ID",
    "PROMPT_QUERY_KEY",
    "SCENE_VARIANTS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesCycleOffsetStageLabelTask",
]
