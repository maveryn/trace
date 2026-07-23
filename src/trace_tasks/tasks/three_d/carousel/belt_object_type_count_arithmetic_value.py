"""Compute arithmetic over object-type counts on carousel belts."""

from __future__ import annotations

from typing import Any, Dict

from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import run_conveyor_count_arithmetic_lifecycle
from .shared.sampling import ARITHMETIC_DIFFERENCE, ARITHMETIC_SUM, PREDICATE_OBJECT_TYPE_ARITHMETIC


TASK_ID = "task_three_d__carousel__belt_object_type_count_arithmetic_value"
TOTAL_QUERY_ID = "total_count"
DIFFERENCE_QUERY_ID = "difference_count"
SUPPORTED_QUERY_IDS = (TOTAL_QUERY_ID, DIFFERENCE_QUERY_ID)
PROMPT_QUERY_KEY_BY_BRANCH = {
    TOTAL_QUERY_ID: "object_count_sum",
    DIFFERENCE_QUERY_ID: "object_count_difference",
}
PREDICATE_KIND_BY_BRANCH = {
    TOTAL_QUERY_ID: PREDICATE_OBJECT_TYPE_ARITHMETIC,
    DIFFERENCE_QUERY_ID: PREDICATE_OBJECT_TYPE_ARITHMETIC,
}
OPERATION_BY_BRANCH = {
    TOTAL_QUERY_ID: ARITHMETIC_SUM,
    DIFFERENCE_QUERY_ID: ARITHMETIC_DIFFERENCE,
}


@register_task
class ThreeDCarouselBeltObjectTypeCountArithmeticValueTask:
    """Compute sum or absolute difference of inner/outer object-type counts."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        output = run_conveyor_count_arithmetic_lifecycle(
            public_name=TASK_ID,
            domain_name=self.domain,
            prompt_query_key_by_branch=PROMPT_QUERY_KEY_BY_BRANCH,
            predicate_kind_by_branch=PREDICATE_KIND_BY_BRANCH,
            operation_by_branch=OPERATION_BY_BRANCH,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=TOTAL_QUERY_ID,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
        )
        return output


__all__ = [
    "DIFFERENCE_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TOTAL_QUERY_ID",
    "ThreeDCarouselBeltObjectTypeCountArithmeticValueTask",
]
