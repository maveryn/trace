from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import DATE_RANGE_DAY_CLASS_COUNT_TASK_KEY, DOMAIN
from .shared.sampling import build_date_range_day_class_count_case, marked_day_class_phrase


TASK_ID = "task_pages__calendar__date_range_day_class_count"
WEEKDAY_RANGE_QUERY_ID = "weekday_range_count"
WEEKEND_RANGE_QUERY_ID = "weekend_range_count"
SUPPORTED_QUERY_IDS = (WEEKDAY_RANGE_QUERY_ID, WEEKEND_RANGE_QUERY_ID)


def _bind_date_range_day_class_count(
    selected_branch,
    branch_probabilities,
    case,
    rendered,
):
    if case.marked_day_class is None or case.reference_date is None or case.target_date is None:
        raise ValueError("date-range day-class count case requires day class and two range boundaries")
    prompt_binding = _lifecycle.CalendarPromptBinding(
        task_key=DATE_RANGE_DAY_CLASS_COUNT_TASK_KEY,
        prompt_branch_key=str(selected_branch),
        dynamic_slots={
            "marked_day_class": str(case.marked_day_class),
            "marked_day_class_phrase": str(marked_day_class_phrase(str(case.marked_day_class))),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_set",
        annotation_value=_lifecycle.date_boxes(rendered, tuple(int(day) for day in case.annotation_dates)),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(case.answer_value),
        extra_params={
            "range_day_class": str(case.marked_day_class),
            "range_day_class_phrase": str(marked_day_class_phrase(str(case.marked_day_class))),
            "range_start_date": int(case.reference_date),
            "range_end_date": int(case.target_date),
            "range_counted_dates": [int(day) for day in case.annotation_dates],
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarDateRangeDayClassCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=WEEKDAY_RANGE_QUERY_ID,
            public_task=TASK_ID,
        )
        day_class = "weekend" if str(selected_branch) == WEEKEND_RANGE_QUERY_ID else "weekday"

        def _build_case(seed, *, params):
            return build_date_range_day_class_count_case(
                int(seed),
                params=params,
                day_class=str(day_class),
            )

        return _lifecycle.render_bound_calendar(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_factory=_build_case,
            binding_factory=_bind_date_range_day_class_count,
        )


__all__ = [
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "WEEKDAY_RANGE_QUERY_ID",
    "WEEKEND_RANGE_QUERY_ID",
    "PagesCalendarDateRangeDayClassCountTask",
]
