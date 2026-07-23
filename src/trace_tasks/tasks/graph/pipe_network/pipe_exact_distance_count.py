from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ._lifecycle import (
    FALLBACK_DEFAULTS,
    bind_pipe_point_set,
    pipe_query_distance_support,
    resolve_pipe_target_axes,
    run_pipe_objective,
    sample_pipe_target_network,
)
from .shared.sampling import sample_pipe_exact_distance_network


TASK_ID = "task_graph__pipe_network__pipe_exact_distance_count"
PROMPT_QUERY_KEY = "pipe_exact_distance_count"
PROMPT_ANNOTATION_KEY = "annotation_hint_exact_distance_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _resolve_exact_distance_axes(*, instance_seed, params, gen_defaults):
    distance_support = pipe_query_distance_support(params=params, gen_defaults=gen_defaults)
    return resolve_pipe_target_axes(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        owner_id=TASK_ID,
        target_low_key="target_exact_distance_count_min",
        target_high_key="target_exact_distance_count_max",
        default_target_low=FALLBACK_DEFAULTS.target_exact_distance_count_min,
        default_target_high=FALLBACK_DEFAULTS.target_exact_distance_count_max,
        explicit_target_keys=("target_exact_distance_count", "target_count"),
        target_namespace="target_exact_distance_count",
        minimum_nodes_for_target=lambda target, distance: max(5, target + distance + 1),
        query_distance_support=tuple(distance_support),
    )


def _bind_exact_distance_result(sample, rendered):
    return bind_pipe_point_set(
        sample,
        rendered,
        labels=sample.target_labels,
        answer_value=int(sample.target_exact_distance_count),
        answer_key="target_exact_distance_count",
        relation_key="exact_distance_junction_labels",
        witness_extra={"source_label": str(sample.query_label), "distance": int(sample.query_distance)},
        prompt_slots={"query_distance": int(sample.query_distance)},
        trace_extra={"query_distance": int(sample.query_distance)},
    )


def _sample_exact_distance_network(instance_seed, _params, max_attempts, axes):
    return sample_pipe_target_network(
        owner_id=TASK_ID,
        instance_seed=instance_seed,
        max_attempts=max_attempts,
        axes=axes,
        sampler=sample_pipe_exact_distance_network,
        target_keyword="target_exact_distance_count",
        query_distance_keyword="query_distance",
        attempt_floor=400,
        attempt_multiplier=2,
    )


@register_task
class GraphRelationPipeExactDistanceCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_pipe_objective(
            owner_id=TASK_ID,
            prompt_query_key=PROMPT_QUERY_KEY,
            prompt_annotation_key=PROMPT_ANNOTATION_KEY,
            supported_branch_names=SUPPORTED_QUERY_IDS,
            default_branch_name=SINGLE_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            resolve_axes=_resolve_exact_distance_axes,
            sample_network=_sample_exact_distance_network,
            bind_result=_bind_exact_distance_result,
        )


__all__ = ["GraphRelationPipeExactDistanceCountTask", "TASK_ID"]
