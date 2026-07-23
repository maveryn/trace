"""Public task for `task_charts__part_whole__contiguous_chart_order_sum`."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import finish_part_whole_plan, run_part_whole_task, sample_part_whole_base
from .shared.defaults import DOMAIN, SAMPLING_NAMESPACE
from .shared.sampling import sample_chart_order_span


SUPPORTED_QUERY_IDS = ("clockwise_span", "counterclockwise_span")


def _build_plan(params, instance_seed: int, selected: str, _probabilities):
    """Bind one circular span and sum its visible category shares."""

    direction = "counterclockwise" if str(selected).startswith("counter") else "clockwise"
    base = sample_part_whole_base(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.contiguous.category_count",
        min_key="contiguous_category_count_min",
        max_key="contiguous_category_count_max",
        fallback_min=4,
        fallback_max=6,
    )
    selected_categories, span_extras = sample_chart_order_span(
        base.categories,
        direction=str(direction),
        params=params,
        count_params=base.count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.contiguous.{direction}",
        min_key="contiguous_span_count_min",
        max_key="contiguous_span_count_max",
        fallback_min=2,
        fallback_max=3,
    )
    answer_value = int(span_extras["selected_share_value"])
    extras = {
        **dict(base.base_extras),
        **dict(span_extras),
        "category_list": [str(category.label) for category in selected_categories],
        "calculation": "sum_contiguous_circular_chart_order_segment_shares",
    }
    return finish_part_whole_plan(
        base=base,
        selected=str(selected),
        instance_seed=int(instance_seed),
        answer_value=int(answer_value),
        annotation_labels=tuple(str(category.label) for category in selected_categories),
        trace_extras=dict(extras),
    )


@register_task
class ChartsCompositionChartContiguousOrderSumTask:
    """Return the sum of a contiguous circular chart-order span."""

    task_id = "task_charts__part_whole__contiguous_chart_order_sum"
    reasoning_operations = ('aggregation', 'topology')
    domain = DOMAIN
    objective_contract = "contiguous_chart_order_sum"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Select direction branch and return a generated part-whole instance."""

        return run_part_whole_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsCompositionChartContiguousOrderSumTask"]
