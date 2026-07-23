"""Single-spinner probability for a color-or-shape disjunction."""

from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ...base import TaskOutput
from ...registry import register_task

from ._lifecycle import prepare_single_color_shape_probability_parts, spinner_task_output_fields


TASK_ID = "task_symbolic__spinner__multi_attribute_or_probability"
DOMAIN = "symbolic"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "single_color_or_shape_probability"
OPERATOR = "or"
REASONING_LOAD = 0.45


@register_task
class SymbolicSpinnerMultiAttributeOrProbabilityTask:
    """Compute a probability for a visible color-or-shape sector predicate."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'formula_evaluation')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Bind the disjunction operator to the shared single-spinner lifecycle."""

        operator = OPERATOR
        prompt_query_key = PROMPT_QUERY_KEY
        prepared = prepare_single_color_shape_probability_parts(
            public_task_id=TASK_ID,
            operator=operator,
            prompt_query_key=prompt_query_key,
            reasoning_load_base=REASONING_LOAD,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
        parts = prepared.output_parts
        output = TaskOutput(
            **spinner_task_output_fields(
                output_parts=parts,
                answer_gt=prepared.answer_gt,
                annotation_gt=prepared.annotation_gt,
                query_id=str(prepared.query_id),
            )
        )
        return output


__all__ = [
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "SymbolicSpinnerMultiAttributeOrProbabilityTask",
]
