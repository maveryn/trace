"""Calendar-event-grid task for finding the busiest date."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import event_chip_box_set
from .shared.defaults import DOMAIN
from .shared.sampling import build_busiest_date_label_case


TASK_ID = "task_pages__calendar_event_grid__busiest_date_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "busiest_date_label"


def _bind_busiest_date_label(selected_branch, branch_probabilities, case, rendered):
    if case.target_date is None:
        raise ValueError("busiest-date-label case requires a target date")
    matching_keys = tuple(str(key) for key in case.matching_chip_keys)
    date_chip_counts = Counter(int(chip.day) for chip in case.event_chips)
    target_count = int(date_chip_counts[int(case.target_date)])
    if target_count != len(matching_keys):
        raise ValueError("busiest-date matching keys must cover all target-date chips")
    if any(int(day) != int(case.target_date) and int(count) >= target_count for day, count in date_chip_counts.items()):
        raise ValueError("busiest-date case must have a unique busiest date")
    prompt_binding = _lifecycle.EventGridPromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={},
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=event_chip_box_set(
            rendered=rendered,
            chip_keys=matching_keys,
        ),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(case.target_date),
        extra_params={
            "busiest_date": int(case.target_date),
            "busiest_date_event_count": int(target_count),
            "date_event_counts": {str(int(day)): int(count) for day, count in sorted(date_chip_counts.items())},
            "matching_chip_keys": [str(key) for key in matching_keys],
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarEventGridBusiestDateLabelTask:
    """Return the date number with the most event chips."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'ranking')
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
            case_factory=build_busiest_date_label_case,
            binding_factory=_bind_busiest_date_label,
        )


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "PagesCalendarEventGridBusiestDateLabelTask"]
