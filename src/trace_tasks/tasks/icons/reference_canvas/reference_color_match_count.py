"""Public task for reference color-match counting."""

from __future__ import annotations

from typing import Any, Dict

from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import SINGLE_QUERY_ID
from ._lifecycle import (
    IconsReferenceCanvasReferenceAttributeMatchCountTaskBase,
)


TASK_ID = "task_icons__reference_canvas__reference_color_match_count"


@register_task
class IconsReferenceCanvasReferenceColorMatchCountTask(IconsReferenceCanvasReferenceAttributeMatchCountTaskBase):
    """Count scene icons with the same color as the reference icon."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'matching')
    domain = "icons"
    supported_query_ids = (SINGLE_QUERY_ID,)
    supported_variants = ("match_color",)
    variant_aliases = {}
    scene_kind = "icons_reference_color_match_count"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one reference color-match count task instance."""

        task_params = dict(params)
        output = super().generate(
            int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
        )
        return output


__all__ = ["IconsReferenceCanvasReferenceColorMatchCountTask", "TASK_ID"]
