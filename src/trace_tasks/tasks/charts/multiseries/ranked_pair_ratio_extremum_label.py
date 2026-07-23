"""Public task for `task_charts__multiseries__ranked_pair_ratio_extremum_label`."""
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
    "pair_ratio_percent_min": 40,
    "pair_ratio_percent_max": 260,
    "ratio_score_spread_extra_min": 0,
    "ratio_score_spread_extra_max": 6,
}
LARGEST_PAIR_RATIO_QUERY_ID = "largest_pair_ratio_label"
SMALLEST_PAIR_RATIO_QUERY_ID = "smallest_pair_ratio_label"
PAIR_RATIO_QUERY_IDS = (LARGEST_PAIR_RATIO_QUERY_ID, SMALLEST_PAIR_RATIO_QUERY_ID)
EXTREMUM_DIRECTION_BY_QUERY_ID = {
    LARGEST_PAIR_RATIO_QUERY_ID: "largest",
    SMALLEST_PAIR_RATIO_QUERY_ID: "smallest",
}


def _build_plan(instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> MultiseriesTaskPlan:
    """Bind the ranked pair-ratio objective before neutral rendering."""

    extremum_direction = EXTREMUM_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
    task_params = {
        **dict(params),
        "extremum_direction": str(extremum_direction),
        "rank_min": 1,
        "rank_max": 1,
    }
    return build_ranked_ratio_label_plan(
        ratio_measure="pair_ratio",
        prompt_query_key="ranked_pair_ratio",
        namespace="ranked_pair_ratio_extremum_label",
        annotation_series_mode="queried_series",
        trace_keys=(
            "numerator_series_label",
            "denominator_series_label",
            "answer_rank",
            "answer_score",
            "derived_metric",
            "rank_order",
            "ranked_category_labels",
            "derived_values_by_category",
            "ratio_percent_by_category",
            "denominator_values_by_category",
        ),
        instance_seed=int(instance_seed),
        params=task_params,
    )


@register_task
class ChartsMultiseriesRankedPairRatioExtremumTask:
    """Return the category label ranked by numerator-over-denominator ratio."""

    task_id = "task_charts__multiseries__ranked_pair_ratio_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "ranked_pair_ratio_extremum_label"
    supported_query_ids = PAIR_RATIO_QUERY_IDS
    default_dataset_enabled = True

    default_query_id = LARGEST_PAIR_RATIO_QUERY_ID
    task_param_defaults = TASK_PARAM_DEFAULTS
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_multiseries_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsMultiseriesRankedPairRatioExtremumTask",
    "LARGEST_PAIR_RATIO_QUERY_ID",
    "PAIR_RATIO_QUERY_IDS",
    "SMALLEST_PAIR_RATIO_QUERY_ID",
]
