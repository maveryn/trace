"""Public task for `task_charts__multiseries__pair_equality_label`."""
from __future__ import annotations

from typing import Any, Mapping

from ....core.query_ids import SINGLE_QUERY_ID
from ._lifecycle import (
    MultiseriesTaskPlan,
    build_pair_equality_label_plan,
    run_configured_multiseries_task,
)
from .shared.defaults import DOMAIN
from ...registry import register_task


TASK_PARAM_DEFAULTS: dict[str, Any] = {
    "equality_category_count_min": 6,
    "equality_category_count_max": 9,
    "equality_series_count_min": 3,
    "equality_series_count_max": 4,
    "equality_value_min": 1,
    "equality_value_max": 99,
}


def _build_plan(instance_seed: int, params: Mapping[str, Any], selected_query_id: str) -> MultiseriesTaskPlan:
    """Bind the exact pair-equality objective before neutral rendering."""

    return build_pair_equality_label_plan(
        namespace="pair_equality_label",
        prompt_query_key="pair_equality_label",
        instance_seed=int(instance_seed),
        params=params,
    )


@register_task
class ChartsMultiseriesPairEqualityLabelTask:
    """Return the unique category where two queried series match exactly."""

    task_id = "task_charts__multiseries__pair_equality_label"
    reasoning_operations = ('comparison', 'matching')
    domain = DOMAIN
    objective_contract = "pair_equality_label"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    default_query_id = SINGLE_QUERY_ID
    task_param_defaults = TASK_PARAM_DEFAULTS
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_configured_multiseries_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsMultiseriesPairEqualityLabelTask"]
