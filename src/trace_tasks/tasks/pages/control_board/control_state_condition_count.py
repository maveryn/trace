"""Control-board task for counting controls matching one state condition."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import counted_control_bboxes
from .shared.defaults import DISABLED_MODE, DOMAIN, SCENE_VARIANTS, SELECTED_ENABLED_MODE
from .shared.rendering import render_control_board_case
from .shared.sampling import build_control_board_case


TASK_ID = "task_pages__control_board__control_state_condition_count"
DISABLED_QUERY_ID = "disabled_controls_in_group_count"
SELECTED_ENABLED_QUERY_ID = "selected_enabled_controls_in_group_count"
SUPPORTED_QUERY_IDS = (DISABLED_QUERY_ID, SELECTED_ENABLED_QUERY_ID)
DEFAULT_QUERY_ID = DISABLED_QUERY_ID
ANSWER_SUPPORT = (2, 3, 4, 5, 6, 7)

_COUNT_MODE_BY_QUERY_ID = {
    DISABLED_QUERY_ID: DISABLED_MODE,
    SELECTED_ENABLED_QUERY_ID: SELECTED_ENABLED_MODE,
}
_QUESTION_FORMAT_BY_QUERY_ID = {
    DISABLED_QUERY_ID: "control_board_disabled_controls_in_group_count",
    SELECTED_ENABLED_QUERY_ID: "control_board_selected_enabled_controls_in_group_count",
}


def _bind_control_state_count(selected_branch, branch_probabilities, case, rendered):
    prompt_binding = _lifecycle.ControlBoardPromptBinding(
        prompt_branch_key=str(selected_branch),
        dynamic_slots={"group_name": str(case.target_group_name)},
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=counted_control_bboxes(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(case.answer_value),
        target_payload={"group_name": str(case.target_group_name), "count_mode": str(case.count_mode)},
        question_format=str(_QUESTION_FORMAT_BY_QUERY_ID[str(selected_branch)]),
    )
    return prompt_binding, answer_binding


@register_task
class PagesControlBoardControlStateConditionCountTask:
    """Count controls in one visible group matching a requested state condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
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
        count_mode = _COUNT_MODE_BY_QUERY_ID[str(selected_branch)]
        case = build_control_board_case(
            instance_seed=int(instance_seed),
            params=task_params,
            count_mode=str(count_mode),
            default_answer_support=ANSWER_SUPPORT,
        )
        rendered = render_control_board_case(instance_seed=int(instance_seed), params=task_params, case=case)
        prompt_binding, answer_binding = _bind_control_state_count(
            str(selected_branch),
            branch_probabilities,
            case,
            rendered,
        )
        return _lifecycle.build_control_board_response(
            instance_seed=int(instance_seed),
            public_task_id=TASK_ID,
            case=case,
            rendered=rendered,
            prompt_binding=prompt_binding,
            answer_binding=answer_binding,
        )


__all__ = [
    "ANSWER_SUPPORT",
    "DEFAULT_QUERY_ID",
    "DISABLED_QUERY_ID",
    "SCENE_VARIANTS",
    "SELECTED_ENABLED_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesControlBoardControlStateConditionCountTask",
]
