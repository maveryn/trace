from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task
from ._lifecycle import (
    bind_pipe_segment_set,
    resolve_pipe_bridge_axes,
    run_pipe_objective,
)
from .shared.sampling import sample_pipe_bridge_network


TASK_ID = "task_graph__pipe_network__bridge_count"
PROMPT_QUERY_KEY = "pipe_bridge_count"
PROMPT_ANNOTATION_KEY = "annotation_hint_bridge_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _resolve_bridge_axes(*, instance_seed, params, gen_defaults):
    return resolve_pipe_bridge_axes(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        owner_id=TASK_ID,
    )


def _bind_bridge_result(sample, rendered):
    return bind_pipe_segment_set(
        sample,
        rendered,
        edges=sample.target_edges,
        answer_value=int(sample.target_bridge_count),
        answer_key="target_bridge_count",
        relation_key="bridge_pipe_labels",
    )


def _sample_bridge_network(instance_seed, _params, max_attempts, axes):
    return sample_pipe_bridge_network(
        spawn_rng(int(instance_seed), f"{TASK_ID}.pipe_network"),
        node_count=int(axes.node_count),
        target_bridge_count=int(axes.target_value),
        grid_shape_variant=str(axes.grid_shape_variant),
        label_variant=str(axes.label_variant),
        max_attempts=max(400, int(max_attempts) * 2),
    )


@register_task
class GraphCountingPipeBridgeCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
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
            resolve_axes=_resolve_bridge_axes,
            sample_network=_sample_bridge_network,
            bind_result=_bind_bridge_result,
        )


__all__ = ["GraphCountingPipeBridgeCountTask", "TASK_ID"]
