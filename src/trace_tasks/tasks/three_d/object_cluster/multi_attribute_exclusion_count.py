"""Count clustered objects matching one attribute while excluding another."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.three_d.shared.task_support import resolve_axis_variant_for_namespace

from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import run_object_cluster_lifecycle
from .shared.output import build_count_request
from .shared.state import ClusterRequest
from ..shared.object_scene import ObjectSceneRenderParams


TASK_ID = "task_three_d__object_cluster__multi_attribute_exclusion_count"
TYPE_WITHOUT_COLOR_QUERY_ID = "type_and_not_color_count"
COLOR_WITHOUT_TYPE_QUERY_ID = "color_and_not_type_count"
SUPPORTED_QUERY_IDS = (TYPE_WITHOUT_COLOR_QUERY_ID, COLOR_WITHOUT_TYPE_QUERY_ID)


@register_task
class ThreeDObjectClusterMultiAttributeExclusionCountTask:
    """Count clustered objects matching one attribute while excluding another."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_object_cluster_lifecycle(int(instance_seed), params=params, max_attempts=int(max_attempts), task_identifier=TASK_ID, build_request=self._build_exclusion_request)

    def _build_exclusion_request(self, instance_seed: int, params: Mapping[str, Any], gen_defaults: Mapping[str, Any], _prompt_defaults: Mapping[str, Any], render_params: ObjectSceneRenderParams) -> ClusterRequest:
        """Bind either public exclusion branch to its internal predicate."""

        branch, probs = resolve_axis_variant_for_namespace(params, namespace=f"{TASK_ID}.branch", gen_defaults=gen_defaults, instance_seed=int(instance_seed), supported_variants=SUPPORTED_QUERY_IDS, explicit_key="query_id", weights_key="query_id_weights", balance_flag_key="balanced_query_id_sampling")
        mode = "type_without_color" if str(branch) == TYPE_WITHOUT_COLOR_QUERY_ID else "color_without_type"
        return build_count_request(mode=mode, external_query=str(branch), prompt_key=str(branch), branch_probabilities=probs, namespace=TASK_ID, instance_seed=int(instance_seed), params=params, gen_defaults=gen_defaults, render_params=render_params)


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDObjectClusterMultiAttributeExclusionCountTask"]
