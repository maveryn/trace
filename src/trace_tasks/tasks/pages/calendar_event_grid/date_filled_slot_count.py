"""Calendar-event-grid task for counting filled event slots on one date."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import event_chip_box_set
from .shared.defaults import DOMAIN
from .shared.sampling import build_date_filled_slot_count_case


TASK_ID = "task_pages__calendar_event_grid__date_filled_slot_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "date_filled_slot_count"


def _bind_date_filled_slot_count(selected_branch, branch_probabilities, case, rendered):
    if case.target_date is None:
        raise ValueError("date-filled-slot-count case requires a target date")
    matching_keys = tuple(str(key) for key in case.matching_chip_keys)
    prompt_binding = _lifecycle.EventGridPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={"date_number": int(case.target_date)},
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=event_chip_box_set(
            rendered=rendered,
            chip_keys=matching_keys,
        ),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(len(matching_keys)),
        extra_params={
            "date_number": int(case.target_date),
            "date_filled_slot_count": int(len(matching_keys)),
            "date_filled_slot_count_probabilities": dict(case.target_count_probabilities),
            "matching_chip_keys": [str(key) for key in matching_keys],
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarEventGridDateFilledSlotCountTask:
    """Count how many event slots are filled on one requested date."""

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
        return _lifecycle.render_bound_event_grid(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_factory=build_date_filled_slot_count_case,
            binding_factory=_bind_date_filled_slot_count,
        )


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "PagesCalendarEventGridDateFilledSlotCountTask"]
