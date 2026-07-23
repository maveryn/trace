"""Count stations on a named metro route satisfying a route-membership condition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import force_query_id_params, select_task_query_id
from ._lifecycle import (
    FALLBACK_DEFAULTS,
    finish_metro_result,
    load_metro_defaults,
    prepare_metro_assets,
    resolve_target_route_style_axes,
    with_query_id_probabilities,
)
from .shared.algorithms import (
    feasible_route_single_route_station_counts,
    feasible_route_transfer_station_counts,
)
from .shared.sampling import (
    sample_route_single_route_station_network,
    sample_route_transfer_station_network,
)


TASK_ID = "task_graph__metro__route_condition_station_count"
ROUTE_TRANSFER_QUERY_ID = "metro_route_transfer_station_count"
ROUTE_SINGLE_ROUTE_QUERY_ID = "metro_route_single_route_station_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (ROUTE_TRANSFER_QUERY_ID, ROUTE_SINGLE_ROUTE_QUERY_ID)


@dataclass(frozen=True)
class _RouteConditionProgram:
    query_id: str
    prompt_query_key: str
    prompt_annotation_key: str
    route_min: int
    route_max: int
    target_min: int
    target_max: int
    feasible_counts: Callable[..., Tuple[int, ...]]
    sampler: Callable[..., Any]
    answer_attr: str

    def params_with_defaults(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        branch_params = dict(params)
        branch_params.setdefault("route_count_min", int(self.route_min))
        branch_params.setdefault("route_count_max", int(self.route_max))
        branch_params.setdefault("target_count_min", int(self.target_min))
        branch_params.setdefault("target_count_max", int(self.target_max))
        return branch_params

    def read_answer(self, sample: Any) -> int:
        return int(getattr(sample, self.answer_attr))


_PROGRAMS: Dict[str, _RouteConditionProgram] = {
    ROUTE_TRANSFER_QUERY_ID: _RouteConditionProgram(
        query_id=ROUTE_TRANSFER_QUERY_ID,
        prompt_query_key="metro_route_transfer_station_count",
        prompt_annotation_key="annotation_hint_route_transfer_station_count",
        route_min=2,
        route_max=5,
        target_min=0,
        target_max=3,
        feasible_counts=feasible_route_transfer_station_counts,
        sampler=sample_route_transfer_station_network,
        answer_attr="target_transfer_count",
    ),
    ROUTE_SINGLE_ROUTE_QUERY_ID: _RouteConditionProgram(
        query_id=ROUTE_SINGLE_ROUTE_QUERY_ID,
        prompt_query_key="metro_route_single_route_station_count",
        prompt_annotation_key="annotation_hint_route_single_route_station_count",
        route_min=2,
        route_max=5,
        target_min=2,
        target_max=5,
        feasible_counts=feasible_route_single_route_station_counts,
        sampler=sample_route_single_route_station_network,
        answer_attr="target_single_route_count",
    ),
}


def _program_for(query_id: str) -> _RouteConditionProgram:
    try:
        return _PROGRAMS[str(query_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported route-condition query_id: {query_id}") from exc


def _resolve_axes(instance_seed: int, params: Mapping[str, Any], *, program: _RouteConditionProgram):
    gen_defaults, *_ = load_metro_defaults(TASK_ID)
    route_low = int(params.get("route_count_min", gen_defaults.get("route_count_min", FALLBACK_DEFAULTS.route_count_min)))
    route_high = int(params.get("route_count_max", gen_defaults.get("route_count_max", FALLBACK_DEFAULTS.route_count_max)))
    return resolve_target_route_style_axes(
        instance_seed=int(instance_seed),
        params=params,
        owner_id=TASK_ID,
        gen_defaults=gen_defaults,
        feasible_values=program.feasible_counts(route_count_min=route_low, route_count_max=route_high),
        feasible_for_route_count=lambda route_count: program.feasible_counts(
            route_count_min=int(route_count),
            route_count_max=int(route_count),
        ),
        explicit_target_keys=("target_count",),
        target_namespace=f"{program.query_id}_target_support_v0",
        route_namespace=f"{program.query_id}_route_count_v0",
    )


@register_task
class GraphMetroRouteConditionStationCountTask:
    """Count stations on a named route that match a route-membership condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select a route condition, sample a matching route map, and bind counted stations."""

        del max_attempts
        branch_name, branch_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=ROUTE_TRANSFER_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        program = _program_for(str(branch_name))
        branch_params = program.params_with_defaults(task_params)
        forced_params = force_query_id_params(branch_params, query_id=str(branch_name))
        axes = _resolve_axes(int(instance_seed), forced_params, program=program)
        sample_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.metro_network")
        sample = program.sampler(
            sample_rng,
            target_count=int(axes.target_count),
            route_count=int(axes.route_count),
            label_variant=str(axes.label_variant),
        )
        assets = prepare_metro_assets(
            owner_id=TASK_ID,
            branch_name=str(branch_name),
            prompt_query_key=str(program.prompt_query_key),
            prompt_annotation_key=str(program.prompt_annotation_key),
            instance_seed=int(instance_seed),
            params=forced_params,
            sample=sample,
            axes=axes,
            answer_value=program.read_answer(sample),
            annotation_labels=tuple(sample.target_labels),
            ordered_annotation=False,
        )
        result = finish_metro_result(assets=assets, branch_name=str(branch_name))
        return TaskOutput(
            prompt=result.prompt,
            answer_gt=result.answer_gt,
            annotation_gt=result.annotation_gt,
            image=result.image,
            image_id=result.image_id,
            trace_payload=with_query_id_probabilities(result.trace_payload, branch_probs),
            task_versions=result.task_versions,
            scene_id=result.scene_id,
            query_id=result.query_id,
            prompt_variants=result.prompt_variants,
        )


__all__ = [
    "GraphMetroRouteConditionStationCountTask",
    "ROUTE_SINGLE_ROUTE_QUERY_ID",
    "ROUTE_TRANSFER_QUERY_ID",
    "TASK_ID",
]
