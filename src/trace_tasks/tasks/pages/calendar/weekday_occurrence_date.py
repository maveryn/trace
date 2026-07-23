from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.time_format import ordinal_label, weekday_name

from . import _lifecycle
from .shared.defaults import (
    DOMAIN,
    WEEKDAY_OCCURRENCE_TASK_KEY,
)
from .shared.sampling import build_weekday_occurrence_case


TASK_ID = "task_pages__calendar__weekday_occurrence_date"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "weekday_occurrence_date"


def _bind_weekday_occurrence(
    selected_branch,
    branch_probabilities,
    case,
    rendered,
):
    if case.weekday_index is None or case.occurrence is None:
        raise ValueError("weekday occurrence case requires weekday index and occurrence")
    prompt_binding = _lifecycle.CalendarPromptBinding(
        task_key=WEEKDAY_OCCURRENCE_TASK_KEY,
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "ordinal": str(ordinal_label(int(case.occurrence))),
            "weekday_name": str(weekday_name(int(case.weekday_index))),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox",
        annotation_value=_lifecycle.date_box(rendered, int(case.answer_value)),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(case.answer_value),
        extra_params={
            "weekday_index": int(case.weekday_index),
            "weekday_name": str(weekday_name(int(case.weekday_index))),
            "occurrence": int(case.occurrence),
            "occurrence_label": str(ordinal_label(int(case.occurrence))),
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarWeekdayOccurrenceDateTask:
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
        return _lifecycle.render_bound_calendar(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_factory=build_weekday_occurrence_case,
            binding_factory=_bind_weekday_occurrence,
        )
