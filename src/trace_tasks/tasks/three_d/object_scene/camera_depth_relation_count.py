"""Public task module for camera-depth relation counting."""

from __future__ import annotations

from typing import Any, Dict

from ...base import TaskOutput
from ...registry import register_task
from ..shared.object_scene_view_relation_count import (
    CAMERA_DEPTH_QUERY_IDS,
    CAMERA_DEPTH_RELATION_COUNT_TASK_ID,
    _ThreeDSpatialViewRelationCountBase,
)


TASK_ID = CAMERA_DEPTH_RELATION_COUNT_TASK_ID
SUPPORTED_QUERY_IDS = CAMERA_DEPTH_QUERY_IDS


@register_task
class ThreeDObjectSceneCameraDepthRelationCountTask(_ThreeDSpatialViewRelationCountBase):
    """Count objects closer or farther than a named reference in camera depth."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate this public object-scene count task through its explicit objective wrapper."""
        seed = int(instance_seed)
        bound_params = dict(params)
        attempts = max(1, int(max_attempts))
        return super().generate(seed, params=bound_params, max_attempts=attempts)


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDObjectSceneCameraDepthRelationCountTask"]
