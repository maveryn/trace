"""Public task for `task_charts__multiseries__category_total_extremum_label`."""
from __future__ import annotations

from typing import Any, Mapping

from ._lifecycle import (
    MultiseriesTaskPlan,
    build_ranked_extremum_label_plan,
    run_configured_multiseries_task,
)
from .shared.defaults import DOMAIN
from ...registry import register_task


TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "category_total_category_count_min": 6,
    "category_total_category_count_max": 10,
    "category_total_series_count_min": 3,
    "category_total_series_count_max": 5,
    "category_total_rank_min": 1,
    "category_total_rank_max": 1,
    "value_min": 1,
    "value_max": 99,
}
LARGEST_CATEGORY_TOTAL_QUERY_ID = "largest_category_total_label"
SMALLEST_CATEGORY_TOTAL_QUERY_ID = "smallest_category_total_label"
CATEGORY_TOTAL_QUERY_IDS = (LARGEST_CATEGORY_TOTAL_QUERY_ID, SMALLEST_CATEGORY_TOTAL_QUERY_ID)
EXTREMUM_DIRECTION_BY_QUERY_ID = {
    LARGEST_CATEGORY_TOTAL_QUERY_ID: "largest",
    SMALLEST_CATEGORY_TOTAL_QUERY_ID: "smallest",
}


def _build_plan(instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> MultiseriesTaskPlan:
    """Bind the category-total rank objective before neutral rendering."""

    extremum_direction = EXTREMUM_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
    task_params = {
        **dict(params),
        "extremum_direction": str(extremum_direction),
        "category_total_rank_min": 1,
        "category_total_rank_max": 1,
    }
    return build_ranked_extremum_label_plan(
        dataset_kind="category_total",
        prompt_query_key="category_total_extremum_label",
        namespace="category_total_extremum_label",
        variant_key="category_total_extremum",
        variant_family="category_total",
        trace_keys=(
            "answer_rank",
            "answer_score",
            "derived_metric",
            "calculation_scope",
            "rank_order",
            "ranked_category_labels",
            "category_totals_by_category",
            "derived_values_by_category",
        ),
        annotation_category_source="answer_label",
        include_target_category_slot=False,
        instance_seed=int(instance_seed),
        params=task_params,
    )


@register_task
class ChartsMultiseriesCategoryTotalExtremumLabelTask:
    """Return the category label with a ranked total across series."""

    task_id = "task_charts__multiseries__category_total_extremum_label"
    reasoning_operations = ('ranking', 'aggregation')
    domain = DOMAIN
    objective_contract = "category_total_extremum_label"
    supported_query_ids = CATEGORY_TOTAL_QUERY_IDS
    default_dataset_enabled = True

    default_query_id = LARGEST_CATEGORY_TOTAL_QUERY_ID
    task_param_defaults = TASK_PARAM_DEFAULTS
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_multiseries_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "CATEGORY_TOTAL_QUERY_IDS",
    "ChartsMultiseriesCategoryTotalExtremumLabelTask",
    "LARGEST_CATEGORY_TOTAL_QUERY_ID",
    "SMALLEST_CATEGORY_TOTAL_QUERY_ID",
]
