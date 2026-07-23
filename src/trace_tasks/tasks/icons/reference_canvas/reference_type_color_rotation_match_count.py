"""Public task for exact reference type/color/rotation-match counting."""

from __future__ import annotations

from typing import Any, Dict

from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import SINGLE_QUERY_ID
from ._lifecycle import (
    IconsReferenceCanvasReferenceAttributeMatchCountTaskBase,
)


TASK_ID = "task_icons__reference_canvas__reference_type_color_rotation_match_count"


@register_task
class IconsReferenceCanvasReferenceTypeColorRotationMatchCountTask(
    IconsReferenceCanvasReferenceAttributeMatchCountTaskBase
):
    """Count scene icons matching the reference icon's type, color, and rotation."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'transformation', 'matching')
    domain = "icons"
    supported_query_ids = (SINGLE_QUERY_ID,)
    supported_variants = ("match_type_color_rotation",)
    variant_aliases = {}
    scene_kind = "icons_reference_type_color_rotation_match_count"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one exact reference type/color/rotation-match count task instance."""

        task_params = dict(params)
        output = super().generate(
            int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
        )
        return output


__all__ = ["IconsReferenceCanvasReferenceTypeColorRotationMatchCountTask", "TASK_ID"]
