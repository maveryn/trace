"""Instruction-panel task for the step containing a control pair."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__instruction_panel__step_for_control_pair_label"
PROMPT_QUERY_KEY = "step_for_control_pair_label"
TASK_NAMESPACE = "pages.instruction_panel.control_pair"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


@register_task
class PagesInstructionPanelStepForControlPairLabelTask:
    """Return the step number containing the two referenced control labels."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        return _lifecycle.build_instruction_panel_response(
            instance_seed=int(instance_seed),
            params=params,
            task_namespace=TASK_NAMESPACE,
            prompt_query_key=PROMPT_QUERY_KEY,
            question_format="instruction_panel_step_for_control_pair_label",
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_NAMESPACE",
    "PagesInstructionPanelStepForControlPairLabelTask",
]
