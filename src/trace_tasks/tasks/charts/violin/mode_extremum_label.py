from trace_tasks.tasks.charts.violin._lifecycle import ViolinTaskPlan, generate_violin_task as run, sample_violin_support
from trace_tasks.tasks.charts.violin.shared.defaults import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__violin__mode_extremum_label"
OBJECTIVE_CONTRACT = "mode_extremum_label"
HIGHEST_QUERY_ID = "highest_mode"
LOWEST_QUERY_ID = "lowest_mode"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (HIGHEST_QUERY_ID, LOWEST_QUERY_ID)
DEFAULT_QUERY_ID = HIGHEST_QUERY_ID
DIRECTION_BY_QUERY = {HIGHEST_QUERY_ID: "highest", LOWEST_QUERY_ID: "lowest"}


def _answer_from_support(selected_branch, support_by_label):
    mode_by_label = {str(label): int(values["mode_values"][0]) for label, values in support_by_label.items()}
    if str(selected_branch) == HIGHEST_QUERY_ID:
        target_value = max(mode_by_label.values())
    elif str(selected_branch) == LOWEST_QUERY_ID:
        target_value = min(mode_by_label.values())
    else:
        raise ValueError(f"unsupported violin mode query: {selected_branch}")
    winners = [label for label, value in mode_by_label.items() if int(value) == int(target_value)]
    if len(winners) != 1:
        raise ValueError("violin mode extremum answer is not unique")
    return str(winners[0]), [int(target_value)], str(DIRECTION_BY_QUERY[str(selected_branch)])


def _build_plan(params, instance_seed, selected_branch, probabilities, mark_style):
    violins, support_by_label, trace_extras = sample_violin_support(
        dataset_variant=str(selected_branch),
        params=params,
        instance_seed=int(instance_seed),
        mark_style=mark_style,
    )
    answer_label, annotation_values, direction = _answer_from_support(str(selected_branch), support_by_label)
    return ViolinTaskPlan(
        violins=violins,
        support_by_label=support_by_label,
        trace_extras=trace_extras,
        answer_label=str(answer_label),
        annotation_values=tuple(int(value) for value in annotation_values),
        prompt_query_key=str(selected_branch),
        extra_relations={"mode_direction": str(direction)},
    )


@register_task
class ChartsDistributionViolinModeExtremumLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    default_query_id = DEFAULT_QUERY_ID
    build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run(self, instance_seed, params, max_attempts)
