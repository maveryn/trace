from trace_tasks.tasks.charts.violin._lifecycle import ViolinTaskPlan, generate_violin_task as run, sample_violin_support
from trace_tasks.tasks.charts.violin.shared.defaults import DOMAIN
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID


TASK_ID = "task_charts__violin__modality_label"
OBJECTIVE_CONTRACT = "modality_label"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (DEFAULT_QUERY_ID,)
PROMPT_QUERY_KEY = "bimodal_label"
DATASET_VARIANT = "bimodal_label"


def _answer_from_support(support_by_label):
    winners = [str(label) for label, values in support_by_label.items() if bool(values.get("bimodal", False))]
    if len(winners) != 1:
        raise ValueError("violin bimodal answer is not unique")
    mode_values = [int(value) for value in support_by_label[winners[0]]["mode_values"]]
    return str(winners[0]), sorted(mode_values)


def _build_plan(params, instance_seed, selected_branch, probabilities, mark_style):
    if str(selected_branch) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    violins, support_by_label, trace_extras = sample_violin_support(
        dataset_variant=DATASET_VARIANT,
        params=params,
        instance_seed=int(instance_seed),
        mark_style=mark_style,
    )
    answer_label, annotation_values = _answer_from_support(support_by_label)
    return ViolinTaskPlan(
        violins=violins,
        support_by_label=support_by_label,
        trace_extras=trace_extras,
        answer_label=str(answer_label),
        annotation_values=tuple(int(value) for value in annotation_values),
        prompt_query_key=PROMPT_QUERY_KEY,
        extra_relations={"prompt_query_key": PROMPT_QUERY_KEY},
    )


@register_task
class ChartsDistributionViolinModalityLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True
    default_query_id = DEFAULT_QUERY_ID
    build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run(self, instance_seed, params, max_attempts)
