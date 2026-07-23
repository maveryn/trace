"""Count ordered adjacent object-type pairs on one straight conveyor lane."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID

from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import run_conveyor_ordered_pair_count_lifecycle
from .shared.sampling import PREDICATE_ORDERED_OBJECT_PAIR


TASK_ID = "task_three_d__conveyor__object_type_ordered_adjacent_pair_count"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY_BY_BRANCH = {QUERY_ID: "object_ordered_pair_count"}
PREDICATE_KIND_BY_BRANCH = {QUERY_ID: PREDICATE_ORDERED_OBJECT_PAIR}


@register_task
class ThreeDConveyorObjectTypeOrderedAdjacentPairCountTask:
    """Count adjacent ordered pairs matching two object types on one lane."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        output = run_conveyor_ordered_pair_count_lifecycle(
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


__all__ = ["QUERY_ID", "SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDConveyorObjectTypeOrderedAdjacentPairCountTask"]
