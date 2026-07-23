"""Public task for `task_charts__part_whole__sector_share_to_angle`."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import finish_part_whole_plan, run_part_whole_task, sample_part_whole_base
from .shared.defaults import DOMAIN, SAMPLING_NAMESPACE
from .shared.sampling import sample_chart_order_span


SUPPORTED_QUERY_IDS = ("clockwise_sector_angle", "counterclockwise_sector_angle")


def _build_plan(params, instance_seed: int, selected: str, _probabilities):
    """Bind a circular span whose share converts to an integer angle."""

    direction = "counterclockwise" if str(selected).startswith("counter") else "clockwise"
    base = sample_part_whole_base(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.sector_angle.category_count",
        min_key="part_whole_category_count_min",
        max_key="part_whole_category_count_max",
        fallback_min=3,
        fallback_max=5,
    )
    selected_categories, span_extras = sample_chart_order_span(
        base.categories,
        direction=str(direction),
        params=params,
        count_params=base.count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.sector_angle.{direction}",
        min_key="part_whole_angle_span_count_min",
        max_key="part_whole_angle_span_count_max",
        fallback_min=2,
        fallback_max=2,
        require_share_multiple_of_five=True,
    )
    selected_share = int(span_extras["selected_share_value"])
    answer_value = int(selected_share * 360 // 100)
    extras = {
        **dict(base.base_extras),
        **dict(span_extras),
        "category_list": [str(category.label) for category in selected_categories],
        "central_angle_degrees": int(answer_value),
        "calculation": "convert_circular_chart_order_share_to_degrees",
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
class ChartsCompositionChartSectorShareToAngleTask:
    """Return the central angle for a selected circular chart span."""

    task_id = "task_charts__part_whole__sector_share_to_angle"
    reasoning_operations = ('aggregation', 'topology', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "sector_share_to_angle"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Select direction branch and return a generated part-whole instance."""

        return run_part_whole_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsCompositionChartSectorShareToAngleTask"]
