"""Public task for `task_charts__multiseries__ranked_series_share_extremum_label`."""
from __future__ import annotations

from typing import Any, Mapping

from ._lifecycle import (
    MultiseriesTaskPlan,
    build_ranked_ratio_label_plan,
    run_configured_multiseries_task,
)
from .shared.defaults import DOMAIN
from ...registry import register_task


TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "ratio_category_count_min": 5,
    "ratio_category_count_max": 10,
    "ratio_series_count_min": 3,
    "ratio_series_count_max": 4,
    "ratio_value_min": 1,
    "ratio_value_max": 80,
    "rank_min": 1,
    "rank_max": 1,
    "share_percent_min": 10,
    "share_percent_max": 75,
    "category_total_min": 35,
    "category_total_max": 160,
    "ratio_score_spread_extra_min": 0,
    "ratio_score_spread_extra_max": 6,
}
LARGEST_SERIES_SHARE_QUERY_ID = "largest_series_share_label"
SMALLEST_SERIES_SHARE_QUERY_ID = "smallest_series_share_label"
SERIES_SHARE_QUERY_IDS = (LARGEST_SERIES_SHARE_QUERY_ID, SMALLEST_SERIES_SHARE_QUERY_ID)
EXTREMUM_DIRECTION_BY_QUERY_ID = {
    LARGEST_SERIES_SHARE_QUERY_ID: "largest",
    SMALLEST_SERIES_SHARE_QUERY_ID: "smallest",
}


def _build_plan(instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> MultiseriesTaskPlan:
    """Bind the ranked series-share objective before neutral rendering."""

    extremum_direction = EXTREMUM_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
    task_params = {
        **dict(params),
        "extremum_direction": str(extremum_direction),
        "rank_min": 1,
        "rank_max": 1,
    }
    return build_ranked_ratio_label_plan(
        ratio_measure="series_share",
        prompt_query_key="ranked_series_share",
        namespace="ranked_series_share_extremum_label",
        annotation_series_mode="all_series",
        trace_keys=(
            "target_series_label",
            "answer_rank",
            "answer_score",
            "derived_metric",
            "rank_order",
            "ranked_category_labels",
            "derived_values_by_category",
            "category_totals_by_category",
            "ratio_percent_by_category",
        ),
        instance_seed=int(instance_seed),
        params=task_params,
    )


@register_task
class ChartsMultiseriesRankedSeriesShareExtremumTask:
    """Return the category label ranked by one series' share of category total."""

    task_id = "task_charts__multiseries__ranked_series_share_extremum_label"
    reasoning_operations = ('ranking', 'aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "ranked_series_share_extremum_label"
    supported_query_ids = SERIES_SHARE_QUERY_IDS
    default_dataset_enabled = True

    default_query_id = LARGEST_SERIES_SHARE_QUERY_ID
    task_param_defaults = TASK_PARAM_DEFAULTS
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_multiseries_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsMultiseriesRankedSeriesShareExtremumTask",
    "LARGEST_SERIES_SHARE_QUERY_ID",
    "SERIES_SHARE_QUERY_IDS",
    "SMALLEST_SERIES_SHARE_QUERY_ID",
]
