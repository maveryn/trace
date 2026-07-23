"""Count objects matching both color and type on one straight conveyor belt."""

from __future__ import annotations

from typing import Any, Dict

from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import run_conveyor_lifecycle
from .shared.sampling import PREDICATE_COLOR_TYPE


TASK_ID = "task_three_d__conveyor__scoped_color_type_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "color_type_belt_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


@register_task
class ThreeDConveyorScopedColorTypeCountTask:
    """Count objects matching a target color and object type on one straight conveyor belt."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        output = run_conveyor_lifecycle(
            public_name=TASK_ID,
            domain_name=self.domain,
            prompt_query_key_by_branch={QUERY_ID: PROMPT_QUERY_KEY},
            predicate_kind_by_branch={QUERY_ID: PREDICATE_COLOR_TYPE},
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=QUERY_ID,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
        )
        return output


__all__ = ["SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDConveyorScopedColorTypeCountTask"]
