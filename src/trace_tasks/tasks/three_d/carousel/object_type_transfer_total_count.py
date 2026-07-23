"""Compute destination count after transferring one object type between belts."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID

from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import run_conveyor_transfer_count_lifecycle
from .shared.sampling import PREDICATE_OBJECT_TYPE_TRANSFER


TASK_ID = "task_three_d__carousel__object_type_transfer_total_count"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY_BY_BRANCH = {QUERY_ID: "object_transfer_total_count"}
PREDICATE_KIND_BY_BRANCH = {QUERY_ID: PREDICATE_OBJECT_TYPE_TRANSFER}


@register_task
class ThreeDCarouselObjectTypeTransferTotalCountTask:
    """Compute destination total after moving matching source-belt object types."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update', 'formula_evaluation')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        output = run_conveyor_transfer_count_lifecycle(
            public_name=TASK_ID,
            domain_name=self.domain,
            prompt_query_key_by_branch=PROMPT_QUERY_KEY_BY_BRANCH,
            predicate_kind_by_branch=PREDICATE_KIND_BY_BRANCH,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=QUERY_ID,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
        )
        return output


__all__ = ["QUERY_ID", "SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDCarouselObjectTypeTransferTotalCountTask"]
