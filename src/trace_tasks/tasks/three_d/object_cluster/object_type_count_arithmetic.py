"""Compute totals or absolute differences over two object-type groups."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.three_d.shared.task_support import resolve_axis_variant_for_namespace

from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import run_object_cluster_lifecycle
from .shared.output import build_count_request
from .shared.state import ClusterRequest
from ..shared.object_scene import ObjectSceneRenderParams


TASK_ID = "task_three_d__object_cluster__object_type_count_arithmetic"
TOTAL_QUERY_ID = "total_count"
DIFFERENCE_QUERY_ID = "difference_count"
SUPPORTED_QUERY_IDS = (TOTAL_QUERY_ID, DIFFERENCE_QUERY_ID)
PROMPT_QUERY_KEY_BY_BRANCH = {
    TOTAL_QUERY_ID: "two_type_total_count",
    DIFFERENCE_QUERY_ID: "two_type_difference_count",
}
MODE_BY_BRANCH = {
    TOTAL_QUERY_ID: "arithmetic_type_total",
    DIFFERENCE_QUERY_ID: "arithmetic_type_difference",
}


@register_task
class ThreeDObjectClusterObjectTypeCountArithmeticTask:
    """Compute arithmetic over two object-type operand counts."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_object_cluster_lifecycle(
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            task_identifier=TASK_ID,
            build_request=self._build_arithmetic_request,
        )

    def _build_arithmetic_request(
        self,
        instance_seed: int,
        params: Mapping[str, Any],
        gen_defaults: Mapping[str, Any],
        _prompt_defaults: Mapping[str, Any],
        render_params: ObjectSceneRenderParams,
    ) -> ClusterRequest:
        branch, probs = resolve_axis_variant_for_namespace(
            params,
            namespace=f"{TASK_ID}.branch",
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            supported_variants=SUPPORTED_QUERY_IDS,
            explicit_key="query_id",
            weights_key="query_id_weights",
            balance_flag_key="balanced_query_id_sampling",
        )
        return build_count_request(
            mode=MODE_BY_BRANCH[str(branch)],
            external_query=str(branch),
            prompt_key=PROMPT_QUERY_KEY_BY_BRANCH[str(branch)],
            branch_probabilities=probs,
            namespace=TASK_ID,
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            render_params=render_params,
        )


__all__ = [
    "DIFFERENCE_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TOTAL_QUERY_ID",
    "ThreeDObjectClusterObjectTypeCountArithmeticTask",
]
