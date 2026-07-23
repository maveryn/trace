from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.time_format import weekday_abbreviation

from . import _lifecycle
from .shared.defaults import DATE_WEEKDAY_LABEL_TASK_KEY, DOMAIN
from .shared.sampling import build_date_weekday_label_case


TASK_ID = "task_pages__calendar__date_weekday_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "date_weekday_label"


def _bind_date_weekday_label(
    selected_branch,
    branch_probabilities,
    case,
    rendered,
):
    if case.weekday_index is None or case.target_date is None:
        raise ValueError("date weekday label case requires target date and weekday index")
    answer_label = str(weekday_abbreviation(int(case.weekday_index)))
    prompt_binding = _lifecycle.CalendarPromptBinding(
        task_key=DATE_WEEKDAY_LABEL_TASK_KEY,
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "target_date": int(case.target_date),
        },
    )
    answer_binding = _lifecycle.CalendarAnswerBinding(
        answer_gt=TypedValue(type="string", value=answer_label),
        annotation_gt=TypedValue(type="bbox", value=_lifecycle.date_box(rendered, int(case.target_date))),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        extra_params={
            "target_date": int(case.target_date),
            "answer_value": answer_label,
            "answer_weekday_index": int(case.weekday_index),
            "answer_weekday_header_label": answer_label,
            "answer_format": "visible_weekday_header_abbreviation",
            "valid_answer_labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarDateWeekdayLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('direct_retrieval',)
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
            case_factory=build_date_weekday_label_case,
            binding_factory=_bind_date_weekday_label,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesCalendarDateWeekdayLabelTask",
]
