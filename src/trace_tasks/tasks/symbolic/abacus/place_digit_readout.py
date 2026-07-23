"""Read the digit shown in one queried abacus place-value column."""

from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ...base import TaskOutput
from ...registry import register_task
from ._lifecycle import (
    AbacusColumnReadoutBinding,
    load_abacus_readout_defaults,
    run_abacus_column_readout_instance,
)


TASK_ID = "task_symbolic__abacus__place_digit_readout"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
INTERNAL_QUERY_KEY = "place_digit_readout"
QUESTION_FORMAT = "place_digit_readout"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_abacus_readout_defaults(TASK_ID)


def _build_column_readout_binding() -> AbacusColumnReadoutBinding:
    """Bind this public objective to the shared abacus board lifecycle."""

    return AbacusColumnReadoutBinding(
        public_task_id=TASK_ID,
        internal_query_key=INTERNAL_QUERY_KEY,
        question_format=QUESTION_FORMAT,
        object_description_prefix="object_description_place_digit_readout",
        question_text_key="question_text_place_digit_readout",
        annotation_hint_key="annotation_hint_place_digit_readout",
        answer_hint_key="answer_hint_place_digit_readout",
        json_example_key="json_example_place_digit_readout",
        json_example_answer_only_key="json_example_answer_only_place_digit_readout",
        failure_message=f"failed to generate abacus column-readout instance for {TASK_ID}",
    )


@register_task
class SymbolicAbacusPlaceDigitReadoutTask:
    """Read the digit in one queried place-value column."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "symbolic"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one queried-column digit readout instance."""

        binding = _build_column_readout_binding()
        return run_abacus_column_readout_instance(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            binding=binding,
        )


__all__ = [
    "SymbolicAbacusPlaceDigitReadoutTask",
    "TASK_ID",
]
