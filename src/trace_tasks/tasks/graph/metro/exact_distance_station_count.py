"""Count stations at an exact metro-route distance from a named station."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import force_query_id_params, select_task_query_id
from ._lifecycle import (
    FALLBACK_DEFAULTS,
    finish_metro_result,
    load_metro_defaults,
    prepare_metro_assets,
    resolve_route_metric_axes,
    select_support_value,
)
from .shared.output import MetroRouteResolvedAxes
from .shared.algorithms import feasible_exact_distance_counts
from .shared.sampling import sample_exact_distance_station_network


TASK_ID = "task_graph__metro__exact_distance_station_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "metro_exact_distance_count"
PROMPT_ANNOTATION_KEY = "annotation_hint_exact_distance_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)


def _resolve_axes(instance_seed: int, params: Mapping[str, Any]) -> MetroRouteResolvedAxes:
    """Exact-distance invariant: target support is filtered by a sampled station distance."""

    gen_defaults, *_ = load_metro_defaults(TASK_ID)
    distance_low = int(params.get("query_distance_min", gen_defaults.get("query_distance_min", FALLBACK_DEFAULTS.query_distance_min)))
    distance_high = int(params.get("query_distance_max", gen_defaults.get("query_distance_max", FALLBACK_DEFAULTS.query_distance_max)))
    query_distance, _distance_probs = select_support_value(
        params=params,
        instance_seed=int(instance_seed),
        owner_id=TASK_ID,
        support=tuple(range(distance_low, distance_high + 1)),
        explicit_keys=("query_distance",),
        namespace_suffix="query_distance_v0",
    )
    return resolve_route_metric_axes(
        instance_seed=int(instance_seed),
        params=params,
        owner_id=TASK_ID,
        feasible_values_fn=feasible_exact_distance_counts,
        metric_kwargs={"query_distance": int(query_distance)},
        explicit_target_keys=("target_count",),
        query_distance=int(query_distance),
    )


@register_task
class GraphRelationMetroExactDistanceCountTask:
    """Count stations exactly k route segments away from a queried station."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Sample the query station, bind exact-distance witnesses, and package output."""

        del max_attempts
        branch_name, _branch_probs, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=QUERY_ID, task_id=TASK_ID, namespace=f"{TASK_ID}.query")
        forced_params = force_query_id_params(task_params, query_id=str(branch_name))
        axes = _resolve_axes(int(instance_seed), forced_params)
        sample = sample_exact_distance_station_network(spawn_rng(int(instance_seed), f"{TASK_ID}.metro_network"), target_count=int(axes.target_count), route_count=int(axes.route_count), query_distance=int(axes.query_distance), label_variant=str(axes.label_variant))
        assets = prepare_metro_assets(owner_id=TASK_ID, branch_name=str(branch_name), prompt_query_key=PROMPT_QUERY_KEY, prompt_annotation_key=PROMPT_ANNOTATION_KEY, instance_seed=int(instance_seed), params=forced_params, sample=sample, axes=axes, answer_value=int(sample.target_exact_distance_count), annotation_labels=tuple(sample.target_labels), ordered_annotation=False)
        return finish_metro_result(assets=assets, branch_name=str(branch_name))


__all__ = ["GraphRelationMetroExactDistanceCountTask", "TASK_ID"]
