"""Count metro stations by route-membership predicate."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import force_query_id_params, select_task_query_id
from ...shared.output_metadata import default_task_versions
from ._lifecycle import (
    FALLBACK_DEFAULTS,
    load_metro_defaults,
    prepare_metro_assets,
    resolve_style_axes,
    route_count_support_for_target,
    select_support_value,
    support_from_bounds,
    with_query_id_probabilities,
)
from .shared.output import MetroRouteResolvedAxes
from .shared.state import SCENE_ID
from .shared.algorithms import feasible_transfer_station_counts
from .shared.sampling import sample_transfer_station_network


TASK_ID = "task_graph__metro__station_membership_count"
TRANSFER_QUERY_ID = "metro_transfer_station_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (TRANSFER_QUERY_ID,)

_QUERY_PROMPT_KEYS: Dict[str, Tuple[str, str]] = {
    TRANSFER_QUERY_ID: ("metro_transfer_station_count", "annotation_hint_transfer_station_count"),
}


def _branch_params(params: Mapping[str, Any], *, branch_name: str) -> Dict[str, Any]:
    branch = dict(params)
    if str(branch_name) != TRANSFER_QUERY_ID:
        raise ValueError(f"unsupported metro station-membership query: {branch_name}")
    branch.setdefault("target_count_min", 1)
    branch.setdefault("target_count_max", 6)
    branch.setdefault("route_count_min", 3)
    branch.setdefault("route_count_max", 5)
    return branch


def _resolve_axes(instance_seed: int, params: Mapping[str, Any], *, branch_name: str) -> MetroRouteResolvedAxes:
    """Membership invariant: both branches count stations by route-cardinality."""

    gen_defaults, _render_defaults, _prompt_defaults, _background_defaults, _noise_defaults = load_metro_defaults(TASK_ID)
    route_low = int(params.get("route_count_min", gen_defaults.get("route_count_min", FALLBACK_DEFAULTS.route_count_min)))
    route_high = int(params.get("route_count_max", gen_defaults.get("route_count_max", FALLBACK_DEFAULTS.route_count_max)))
    if str(branch_name) != TRANSFER_QUERY_ID:
        raise ValueError(f"unsupported metro station-membership query: {branch_name}")
    feasible = feasible_transfer_station_counts(route_count_min=route_low, route_count_max=route_high)
    feasible_for_route = lambda route_count: feasible_transfer_station_counts(
        route_count_min=int(route_count),
        route_count_max=int(route_count),
    )
    target_support = support_from_bounds(
        params=params,
        gen_defaults=gen_defaults,
        low_key="target_count_min",
        high_key="target_count_max",
        default_low=FALLBACK_DEFAULTS.target_count_min,
        default_high=FALLBACK_DEFAULTS.target_count_max,
        feasible=feasible,
    )
    target_count, target_probs = select_support_value(params=params, instance_seed=int(instance_seed), owner_id=TASK_ID, support=target_support, explicit_keys=("target_count", "target_transfer_count"), namespace_suffix=f"{branch_name}_target_support_v0")
    route_support = route_count_support_for_target(params=params, gen_defaults=gen_defaults, target_value=int(target_count), feasible_for_route_count=feasible_for_route)
    route_count, route_probs = select_support_value(params=params, instance_seed=int(instance_seed), owner_id=TASK_ID, support=route_support, explicit_keys=("route_count",), namespace_suffix=f"{branch_name}_route_count_v0")
    label_variant, label_probs, node_color_name, color_probs = resolve_style_axes(params=params, gen_defaults=gen_defaults, instance_seed=int(instance_seed), owner_id=TASK_ID)
    return MetroRouteResolvedAxes(int(target_count), int(route_count), str(label_variant), str(node_color_name), 0, dict(target_probs), dict(route_probs), dict(label_probs), dict(color_probs))


@register_task
class GraphCountingMetroStationMembershipCountTask:
    """Count stations by transfer or single-route membership."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the membership predicate, sample a matching map, and bind matching stations."""

        del max_attempts
        branch_name, branch_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=TRANSFER_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        branch_params = _branch_params(task_params, branch_name=str(branch_name))
        forced_params = force_query_id_params(branch_params, query_id=str(branch_name))
        axes = _resolve_axes(int(instance_seed), forced_params, branch_name=str(branch_name))
        sample_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.metro_network")
        sample = sample_transfer_station_network(
            sample_rng,
            target_count=int(axes.target_count),
            route_count=int(axes.route_count),
            label_variant=str(axes.label_variant),
        )
        answer_value = int(sample.target_transfer_count)
        prompt_query_key, prompt_annotation_key = _QUERY_PROMPT_KEYS[str(branch_name)]
        assets = prepare_metro_assets(
            owner_id=TASK_ID,
            branch_name=str(branch_name),
            prompt_query_key=str(prompt_query_key),
            prompt_annotation_key=str(prompt_annotation_key),
            instance_seed=int(instance_seed),
            params=forced_params,
            sample=sample,
            axes=axes,
            answer_value=int(answer_value),
            annotation_labels=tuple(sample.target_labels),
            ordered_annotation=False,
        )
        return TaskOutput(
            prompt=str(assets.prompt),
            answer_gt=TypedValue(type="integer", value=int(assets.answer_annotation.answer_value)),
            annotation_gt=TypedValue(type=str(assets.answer_annotation.annotation_type), value=list(assets.answer_annotation.annotation_value)),
            image=assets.image,
            image_id="img0",
            trace_payload=with_query_id_probabilities(assets.trace_payload, branch_probs),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(branch_name),
            prompt_variants=dict(assets.prompt_variants),
        )


GraphCountingTransferStationCountTask = GraphCountingMetroStationMembershipCountTask

__all__ = ["GraphCountingMetroStationMembershipCountTask", "GraphCountingTransferStationCountTask", "TASK_ID"]
