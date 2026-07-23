"""Instruction-panel task for a control shared by referenced steps."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__instruction_panel__shared_control_for_step_set_label"
PROMPT_QUERY_KEY = "shared_control_for_step_set_label"
TASK_NAMESPACE = "pages.instruction_panel.shared_control"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


@register_task
class PagesInstructionPanelSharedControlForStepSetLabelTask:
    """Return the visible control label common to a referenced step set."""

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
            question_format="instruction_panel_shared_control_for_step_set_label",
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_NAMESPACE",
    "PagesInstructionPanelSharedControlForStepSetLabelTask",
]
