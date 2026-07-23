from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import DOMAIN, MARKED_DAY_CLASS_COUNT_TASK_KEY
from .shared.sampling import build_marked_day_class_count_case, marked_day_class_phrase


TASK_ID = "task_pages__calendar__marked_day_class_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "marked_day_class_count"


def _bind_marked_day_class_count(
    selected_branch,
    branch_probabilities,
    case,
    rendered,
):
    if case.marked_day_class is None:
        raise ValueError("marked day-class count case requires marked_day_class")
    prompt_binding = _lifecycle.CalendarPromptBinding(
        task_key=MARKED_DAY_CLASS_COUNT_TASK_KEY,
        prompt_branch_key=PROMPT_QUERY_KEY,
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
            "marked_day_class": str(case.marked_day_class),
            "marked_day_class_phrase": str(marked_day_class_phrase(str(case.marked_day_class))),
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarMarkedDayClassCountTask:
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
        return _lifecycle.render_bound_calendar(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_factory=build_marked_day_class_count_case,
            binding_factory=_bind_marked_day_class_count,
        )
