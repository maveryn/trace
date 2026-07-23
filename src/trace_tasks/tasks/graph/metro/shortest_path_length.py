"""Measure the unique shortest path length in a metro-route graph."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ....core.seed import spawn_rng
from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import force_query_id_params, select_task_query_id
from ._lifecycle import (
    finish_metro_result,
    prepare_metro_assets,
    resolve_route_metric_axes,
)
from .shared.algorithms import feasible_shortest_path_lengths
from .shared.sampling import sample_shortest_path_network


TASK_ID = "task_graph__metro__shortest_path_length"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "metro_shortest_path_length"
PROMPT_ANNOTATION_KEY = "annotation_hint_shortest_path_length"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)


@register_task
class GraphPathMetroShortestPathLengthTask:
    """Count route segments in a unique shortest station path."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Sample a unique source-goal shortest path and bind its station sequence."""

        del max_attempts
        branch_name, _branch_probs, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=QUERY_ID, task_id=TASK_ID, namespace=f"{TASK_ID}.query")
        forced_params = force_query_id_params(task_params, query_id=str(branch_name))
        axes = resolve_route_metric_axes(
            instance_seed=int(instance_seed),
            params=forced_params,
            owner_id=TASK_ID,
            feasible_values_fn=feasible_shortest_path_lengths,
            low_key="target_shortest_path_length_min",
            high_key="target_shortest_path_length_max",
            explicit_target_keys=("target_shortest_path_length", "target_count"),
        )
        sample = sample_shortest_path_network(spawn_rng(int(instance_seed), f"{TASK_ID}.metro_network"), target_length=int(axes.target_count), route_count=int(axes.route_count), label_variant=str(axes.label_variant))
        path_labels = tuple(str(label) for label in sample.target_labels)
        annotation_labels = path_labels[1:]
        assets = prepare_metro_assets(owner_id=TASK_ID, branch_name=str(branch_name), prompt_query_key=PROMPT_QUERY_KEY, prompt_annotation_key=PROMPT_ANNOTATION_KEY, instance_seed=int(instance_seed), params=forced_params, sample=sample, axes=axes, answer_value=int(sample.target_shortest_path_length), annotation_labels=annotation_labels, ordered_annotation=True, witness_extra={"full_path_labels": list(path_labels)}, json_example_key="json_example_shortest_path_length")
        return finish_metro_result(assets=assets, branch_name=str(branch_name))


__all__ = ["GraphPathMetroShortestPathLengthTask", "TASK_ID"]
