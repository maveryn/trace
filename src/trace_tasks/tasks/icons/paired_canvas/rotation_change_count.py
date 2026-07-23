"""Public task for paired-canvas rotation-change counting."""

from __future__ import annotations

from typing import Any, Dict

from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import SINGLE_QUERY_ID
from ._lifecycle import IconsPanelAttributeChangeCountTaskBase, run_panel_attribute_change_task


TASK_ID = "task_icons__paired_canvas__rotation_change_count"


def _resolve_rotation_change_query_ids() -> tuple[str, ...]:
    """Return the task-owned internal comparison branch."""

    return ("rotation_changed_count",)


@register_task
class IconsPairedCanvasRotationChangeCountTask(IconsPanelAttributeChangeCountTaskBase):
    """Count Right-panel icons whose rotation changed from the Left panel."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'transformation')
    domain = "icons"
    supported_query_ids = (SINGLE_QUERY_ID,)
    query_ids = _resolve_rotation_change_query_ids()
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one rotation-change paired-canvas count instance."""

        return run_panel_attribute_change_task(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["IconsPairedCanvasRotationChangeCountTask", "TASK_ID"]
