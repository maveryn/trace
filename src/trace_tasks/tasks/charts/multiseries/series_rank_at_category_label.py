"""Public task for `task_charts__multiseries__series_rank_at_category_label`."""
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
    "series_rank_category_count_min": 6,
    "series_rank_category_count_max": 12,
    "series_rank_series_count_min": 3,
    "series_rank_series_count_max": 5,
    "series_rank_rank_min": 1,
    "series_rank_rank_max": 1,
    "value_min": 1,
    "value_max": 99,
}
LARGEST_SERIES_AT_CATEGORY_QUERY_ID = "largest_series_at_category_label"
SMALLEST_SERIES_AT_CATEGORY_QUERY_ID = "smallest_series_at_category_label"
SERIES_RANK_AT_CATEGORY_QUERY_IDS = (
    LARGEST_SERIES_AT_CATEGORY_QUERY_ID,
    SMALLEST_SERIES_AT_CATEGORY_QUERY_ID,
)
EXTREMUM_DIRECTION_BY_QUERY_ID = {
    LARGEST_SERIES_AT_CATEGORY_QUERY_ID: "largest",
    SMALLEST_SERIES_AT_CATEGORY_QUERY_ID: "smallest",
}


def _build_plan(instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> MultiseriesTaskPlan:
    """Bind the within-category series-rank objective before rendering."""

    extremum_direction = EXTREMUM_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
    task_params = {
        **dict(params),
        "extremum_direction": str(extremum_direction),
        "series_rank_rank_min": 1,
        "series_rank_rank_max": 1,
    }
    return build_ranked_extremum_label_plan(
        dataset_kind="series_rank",
        prompt_query_key="series_rank_at_category_label",
        namespace="series_rank_at_category_label",
        variant_key="series_rank_at_category",
        variant_family="series_rank",
        trace_keys=(
            "target_category_label",
            "target_category_index",
            "answer_series_label",
            "answer_rank",
            "answer_score",
            "derived_metric",
            "calculation_scope",
            "rank_order",
            "ranked_series_labels",
            "ranked_category_labels",
            "values_by_series_at_target_category",
            "derived_values_by_category",
        ),
        annotation_category_source="target_category_label",
        include_target_category_slot=True,
        instance_seed=int(instance_seed),
        params=task_params,
    )


@register_task
class ChartsMultiseriesSeriesRankAtCategoryLabelTask:
    """Return the ranked series label within one selected category."""

    task_id = "task_charts__multiseries__series_rank_at_category_label"
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "series_rank_at_category_label"
    supported_query_ids = SERIES_RANK_AT_CATEGORY_QUERY_IDS
    default_dataset_enabled = True

    default_query_id = LARGEST_SERIES_AT_CATEGORY_QUERY_ID
    task_param_defaults = TASK_PARAM_DEFAULTS
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_multiseries_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsMultiseriesSeriesRankAtCategoryLabelTask",
    "LARGEST_SERIES_AT_CATEGORY_QUERY_ID",
    "SERIES_RANK_AT_CATEGORY_QUERY_IDS",
    "SMALLEST_SERIES_AT_CATEGORY_QUERY_ID",
]
