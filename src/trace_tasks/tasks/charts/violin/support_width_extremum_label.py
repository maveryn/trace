from trace_tasks.tasks.charts.violin._lifecycle import ViolinTaskPlan, generate_violin_task as run, sample_violin_support
from trace_tasks.tasks.charts.violin.shared.defaults import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__violin__support_width_extremum_label"
OBJECTIVE_CONTRACT = "support_width_extremum_label"
WIDEST_QUERY_ID = "widest_support"
NARROWEST_QUERY_ID = "narrowest_support"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (WIDEST_QUERY_ID, NARROWEST_QUERY_ID)
DEFAULT_QUERY_ID = WIDEST_QUERY_ID
DIRECTION_BY_QUERY = {WIDEST_QUERY_ID: "widest", NARROWEST_QUERY_ID: "narrowest"}


def _answer_from_support(selected_branch, support_by_label):
    span_by_label = {str(label): int(values["support_span"]) for label, values in support_by_label.items()}
    if str(selected_branch) == WIDEST_QUERY_ID:
        target_span = max(span_by_label.values())
    elif str(selected_branch) == NARROWEST_QUERY_ID:
        target_span = min(span_by_label.values())
    else:
        raise ValueError(f"unsupported violin support query: {selected_branch}")
    winners = [label for label, span in span_by_label.items() if int(span) == int(target_span)]
    if len(winners) != 1:
        raise ValueError("violin support-width extremum answer is not unique")
    values = support_by_label[str(winners[0])]
    return (
        str(winners[0]),
        [int(values["support_min"]), int(values["support_max"])],
        str(DIRECTION_BY_QUERY[str(selected_branch)]),
    )


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
        extra_relations={"support_direction": str(direction)},
    )


@register_task
class ChartsDistributionViolinSupportWidthExtremumLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    default_query_id = DEFAULT_QUERY_ID
    build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run(self, instance_seed, params, max_attempts)
