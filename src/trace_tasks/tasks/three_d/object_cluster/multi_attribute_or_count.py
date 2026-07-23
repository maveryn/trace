"""Count clustered objects matching a type/color inclusive OR."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import run_object_cluster_lifecycle
from .shared.output import build_count_request
from .shared.state import ClusterRequest
from ..shared.object_scene import ObjectSceneRenderParams


TASK_ID = "task_three_d__object_cluster__multi_attribute_or_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "type_or_color_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


@register_task
class ThreeDObjectClusterMultiAttributeOrCountTask:
    """Count clustered objects matching a type/color inclusive OR."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_object_cluster_lifecycle(int(instance_seed), params=params, max_attempts=int(max_attempts), task_identifier=TASK_ID, build_request=self._build_or_request)

    def _build_or_request(self, instance_seed: int, params: Mapping[str, Any], gen_defaults: Mapping[str, Any], _prompt_defaults: Mapping[str, Any], render_params: ObjectSceneRenderParams) -> ClusterRequest:
        """Bind the public no-branch inclusive-OR contract."""

        if str(params.get("query_id", QUERY_ID)) != QUERY_ID:
            raise ValueError("unsupported query_id")
        return build_count_request(mode="type_or_color", external_query=QUERY_ID, prompt_key=PROMPT_QUERY_KEY, branch_probabilities={QUERY_ID: 1.0}, namespace=TASK_ID, instance_seed=int(instance_seed), params=params, gen_defaults=gen_defaults, render_params=render_params)


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDObjectClusterMultiAttributeOrCountTask"]
