"""Paired-form task for summed absolute quantity differences."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__paired_forms__sum_absolute_quantity_differences_value"
SUPPORTED_QUERY_IDS = _lifecycle.SUPPORTED_QUERY_IDS
PROMPT_QUERY_KEY = "sum_absolute_quantity_differences"
QUESTION_FORMAT = "paired_forms_sum_absolute_quantity_differences_value"
REASONING_LOAD_BASE = 0.00
EXAMPLE_ANSWER = 42


def _answer_value(item_specs: Sequence[Mapping[str, Any]]) -> int:
    """Return the sum of absolute ordered-versus-received quantity differences."""

    return sum(
        int(spec["absolute_quantity_difference"])
        for spec in item_specs
        if int(spec["absolute_quantity_difference"]) > 0
    )


@register_task
class PagesPairedFormsSumAbsoluteQuantityDifferencesValueTask:
    """Compute the sum of absolute quantity differences across paired-form rows."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation', 'matching')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, object], max_attempts: int):
        del max_attempts
        prompt_query_key = PROMPT_QUERY_KEY
        question_format = QUESTION_FORMAT
        answer_fn = _answer_value
        include_unit_value_support = False
        reasoning_load_base = REASONING_LOAD_BASE
        example_answer = EXAMPLE_ANSWER
        return _lifecycle.build_paired_forms_response(
            instance_seed=int(instance_seed),
            params=params,
            task_id=TASK_ID,
            prompt_query_key=prompt_query_key,
            question_format=question_format,
            answer_fn=answer_fn,
            include_unit_value_support=include_unit_value_support,
            reasoning_load_base=reasoning_load_base,
            example_answer=example_answer,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "QUESTION_FORMAT",
    "REASONING_LOAD_BASE",
    "EXAMPLE_ANSWER",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesPairedFormsSumAbsoluteQuantityDifferencesValueTask",
]
