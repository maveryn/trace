"""Calendar-event-grid task for counting dates with a category in one slot."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import event_chip_box_set
from .shared.defaults import DOMAIN
from .shared.sampling import build_category_slot_day_count_case


TASK_ID = "task_pages__calendar_event_grid__category_slot_day_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "category_slot_day_count"


def _bind_category_slot_day_count(selected_branch, branch_probabilities, case, rendered):
    prompt_binding = _lifecycle.EventGridPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "category_label": str(case.category_label),
            "slot_label": str(case.slot_label),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=event_chip_box_set(
            rendered=rendered,
            chip_keys=tuple(case.matching_chip_keys),
        ),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(len(case.matching_chip_keys)),
        extra_params={
            "slot_id": str(case.slot_id),
            "slot_label": str(case.slot_label),
            "category_label": str(case.category_label),
            "matching_chip_keys": [str(key) for key in case.matching_chip_keys],
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarEventGridCategorySlotDayCountTask:
    """Count dates whose named slot contains one requested category."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
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
            case_factory=build_category_slot_day_count_case,
            binding_factory=_bind_category_slot_day_count,
        )


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "PagesCalendarEventGridCategorySlotDayCountTask"]
