"""Schedule task for counting events longer than a reference event."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle


TASK_ID = "task_pages__schedule__longer_than_reference_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "longer_than_reference_count"


@register_task
class PagesScheduleLongerThanReferenceCountTask:
  """Count scheduled events longer than the highlighted reference event."""

  task_id = TASK_ID
  reasoning_operations = ('filtering', 'counting', 'comparison')
  domain = _lifecycle.DOMAIN
  supported_query_ids = SUPPORTED_QUERY_IDS
  default_dataset_enabled = True

  def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
    del max_attempts
    selected_branch, branch_probabilities, task_params = select_task_query_id(
      instance_seed=int(instance_seed),
      params=params,
      supported_query_ids=SUPPORTED_QUERY_IDS,
      default_query_id=SINGLE_QUERY_ID,
      task_id=TASK_ID,
    )
    return _lifecycle.build_schedule_response(
      instance_seed=int(instance_seed),
      params=task_params,
      selected_branch=str(selected_branch),
      branch_probabilities=branch_probabilities,
      program_mode=_lifecycle.LONGER_THAN_REFERENCE_MODE,
      prompt_query_key=PROMPT_QUERY_KEY,
      source_query_name=PROMPT_QUERY_KEY,
    )


__all__ = [
  "PROMPT_QUERY_KEY",
  "SUPPORTED_QUERY_IDS",
  "TASK_ID",
  "PagesScheduleLongerThanReferenceCountTask",
]
