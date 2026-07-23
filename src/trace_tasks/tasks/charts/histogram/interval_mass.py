"""Public task for `task_charts__histogram__interval_mass`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.histogram._lifecycle import (
    make_histogram_task_plan,
    run_histogram_lifecycle,
)
from trace_tasks.tasks.charts.histogram.shared.defaults import DOMAIN
from trace_tasks.tasks.charts.histogram.shared.rendering import resolve_mark_style
from trace_tasks.tasks.charts.histogram.shared.sampling import build_histogram_dataset
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__histogram__interval_mass"
OBJECTIVE_CONTRACT = "interval_mass"
PROMPT_QUERY_KEY = "interval_mass"
DATASET_VARIANT = "interval_mass"
INTERVAL_RELATION = "inside"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID


def _build_interval_mass_plan(params, instance_seed, selected_query_id, query_probabilities):
    """Bind the interval-mass objective before neutral rendering."""

    if str(selected_query_id) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query_id}")
    mark_style = resolve_mark_style(params, instance_seed=int(instance_seed), mark_count=1)
    bins, answer_value, annotation_labels, trace_extras = build_histogram_dataset(
        dataset_variant=DATASET_VARIANT,
        params=dict(params),
        instance_seed=int(instance_seed),
        mark_style=mark_style,
    )
    counts_by_label = {str(bin_spec.label): int(bin_spec.count) for bin_spec in bins}
    if int(answer_value) != sum(int(counts_by_label[str(label)]) for label in annotation_labels):
        raise RuntimeError("histogram interval-mass objective lost answer/annotation alignment")
    return make_histogram_task_plan(
        bins=bins,
        params=dict(params),
        mark_style=mark_style,
        answer_value=int(answer_value),
        answer_type="integer",
        question_format="numeric_open",
        annotation_type="bbox_set",
        annotation_labels=tuple(str(label) for label in annotation_labels),
        prompt_query_key=PROMPT_QUERY_KEY,
        dataset_variant=DATASET_VARIANT,
        trace_extras={**dict(trace_extras), "interval_relation": INTERVAL_RELATION},
        query_probabilities=query_probabilities,
        objective_contract=OBJECTIVE_CONTRACT,
        dynamic_slots={
            "query_interval_label": str(trace_extras.get("query_interval_label", "")),
            "query_interval_start_value": str(trace_extras.get("query_interval_start_value", "")),
            "query_interval_end_value": str(trace_extras.get("query_interval_end_value", "")),
            "interval_relation_phrase": INTERVAL_RELATION,
        },
        instance_seed=int(instance_seed),
    )


class ChartsDistributionHistogramIntervalMassTask:
    task_id = TASK_ID
    reasoning_operations = ('aggregation',)
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_histogram_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_interval_mass_plan,
        )
register_task(ChartsDistributionHistogramIntervalMassTask)
