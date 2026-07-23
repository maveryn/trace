from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ._lifecycle import (
    FALLBACK_DEFAULTS,
    bind_pipe_point_sequence,
    resolve_pipe_target_axes,
    run_pipe_objective,
)
from .shared.sampling import sample_pipe_shortest_path_network


TASK_ID = "task_graph__pipe_network__shortest_path_length"
PROMPT_QUERY_KEY = "pipe_shortest_path_length"
PROMPT_ANNOTATION_KEY = "annotation_hint_shortest_path_length"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _resolve_path_axes(*, instance_seed, params, gen_defaults):
    return resolve_pipe_target_axes(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        owner_id=TASK_ID,
        target_low_key="target_shortest_path_length_min",
        target_high_key="target_shortest_path_length_max",
        default_target_low=FALLBACK_DEFAULTS.target_shortest_path_length_min,
        default_target_high=FALLBACK_DEFAULTS.target_shortest_path_length_max,
        explicit_target_keys=("target_shortest_path_length", "target_count"),
        target_namespace="target_shortest_path_length",
        minimum_nodes_for_target=lambda target, _distance: int(target) + 2,
    )


def _bind_shortest_path_result(sample, rendered):
    return bind_pipe_point_sequence(
        sample,
        rendered,
        labels=sample.target_labels,
        answer_value=int(sample.target_shortest_path_length),
        answer_key="target_shortest_path_length",
        relation_key="shortest_path_labels",
    )


def _sample_path_network(instance_seed, _params, max_attempts, axes):
    return sample_pipe_shortest_path_network(
        spawn_rng(int(instance_seed), f"{TASK_ID}.pipe_network"),
        node_count=int(axes.node_count),
        target_shortest_path_length=int(axes.target_value),
        grid_shape_variant=str(axes.grid_shape_variant),
        label_variant=str(axes.label_variant),
        max_attempts=max(200, int(max_attempts)),
    )


@register_task
class GraphPathPipeShortestPathLengthTask:
    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
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
            resolve_axes=_resolve_path_axes,
            sample_network=_sample_path_network,
            bind_result=_bind_shortest_path_result,
        )


__all__ = ["GraphPathPipeShortestPathLengthTask", "TASK_ID"]
