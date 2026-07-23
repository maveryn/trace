"""Count a queried symbol after simulating a displayed Turing tape."""

from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task

from ._lifecycle import PreparedTuringScene, TuringObjectiveResult, run_turing_lifecycle
from .shared.annotations import annotation_trace_payload, keyed_bboxes


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__turing_tape__turing_written_symbol_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "written_symbol_count"
TASK_PROMPT_KEY = "turing_written_symbol_count_query"
ANNOTATION_ROLE_ITEM_IDS = {
    "machine_panel": "machine_panel",
    "transition_table": "transition_table",
}


@register_task
class SymbolicTuringTapeWrittenSymbolCountTask:
    """Count the queried tape symbol after fixed-step Turing-style transitions."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one Turing tape simulation/count instance."""

        return run_turing_lifecycle(
            task_identifier=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            task_prompt_key=TASK_PROMPT_KEY,
            prompt_query_key=PROMPT_QUERY_KEY,
            params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_build_written_symbol_count_objective,
        )


def _build_written_symbol_count_objective(prepared: PreparedTuringScene) -> TuringObjectiveResult:
    dataset = prepared.dataset
    annotation_value = keyed_bboxes(prepared.rendered.item_bboxes, ANNOTATION_ROLE_ITEM_IDS)
    witness_symbolic, projected_annotation = annotation_trace_payload(annotation_value)
    return TuringObjectiveResult(
        answer_gt=TypedValue(type="integer", value=int(dataset.answer_count)),
        annotation_gt=TypedValue(type="bbox_map", value=dict(annotation_value)),
        answer_value=int(dataset.answer_count),
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        execution_fields={
            "answer_count": int(dataset.answer_count),
            "supporting_item_ids_by_role": dict(ANNOTATION_ROLE_ITEM_IDS),
            "supporting_item_ids": list(ANNOTATION_ROLE_ITEM_IDS.values()),
        },
    )


__all__ = [
    "ANNOTATION_ROLE_ITEM_IDS",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_PROMPT_KEY",
    "SymbolicTuringTapeWrittenSymbolCountTask",
]
