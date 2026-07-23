from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import DOMAIN, WORKDAY_OFFSET_TASK_KEY
from .shared.sampling import build_workday_offset_case


TASK_ID = "task_pages__calendar__workday_offset_date"
WORKDAY_AFTER_QUERY_ID = "workday_after_offset_date"
WORKDAY_BEFORE_QUERY_ID = "workday_before_offset_date"
SUPPORTED_QUERY_IDS = (WORKDAY_AFTER_QUERY_ID, WORKDAY_BEFORE_QUERY_ID)


def _bind_workday_offset(
    selected_branch,
    branch_probabilities,
    case,
    rendered,
):
    if case.reference_date is None or case.target_date is None or case.workday_offset is None:
        raise ValueError("workday offset case requires reference date, target date, and offset")
    annotation = {
        "reference_date": _lifecycle.date_box(rendered, int(case.reference_date)),
        "target_date": _lifecycle.date_box(rendered, int(case.target_date)),
    }
    prompt_binding = _lifecycle.CalendarPromptBinding(
        task_key=WORKDAY_OFFSET_TASK_KEY,
        prompt_branch_key=str(selected_branch),
        dynamic_slots={
            "workday_offset": int(case.workday_offset),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_kind="bbox_map",
        annotation_value=dict(annotation),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=int(case.answer_value),
        extra_params={
            "workday_direction": str(case.workday_direction),
            "workday_offset": int(case.workday_offset),
            "reference_date": int(case.reference_date),
            "target_date": int(case.target_date),
        },
    )
    return prompt_binding, answer_binding


@register_task
class PagesCalendarWorkdayOffsetDateTask:
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=WORKDAY_AFTER_QUERY_ID,
            public_task=TASK_ID,
        )
        direction = "after" if str(selected_branch) == WORKDAY_AFTER_QUERY_ID else "before"
        def _build_case(seed, *, params):
            call_params = params
            return build_workday_offset_case(int(seed), params=call_params, direction=str(direction))

        return _lifecycle.render_bound_calendar(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            case_factory=_build_case,
            binding_factory=_bind_workday_offset,
        )
