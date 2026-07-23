"""Form-section task for two-operand currency arithmetic in one section."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_form_section_public_entry
from .shared.sampling import ExpressionPlan, SCENE_VARIANTS


TASK_ID = "task_pages__form_section__two_amount_arithmetic_value"
SUM_QUERY_ID = "sum_two_amounts_in_section_value"
DIFFERENCE_QUERY_ID = "difference_two_amounts_in_section_value"
SUPPORTED_QUERY_IDS = (SUM_QUERY_ID, DIFFERENCE_QUERY_ID)
PROMPT_QUERY_KEYS = {
    SUM_QUERY_ID: SUM_QUERY_ID,
    DIFFERENCE_QUERY_ID: DIFFERENCE_QUERY_ID,
}
QUESTION_FORMATS = {
    SUM_QUERY_ID: "form_section_two_amount_arithmetic_value",
    DIFFERENCE_QUERY_ID: "form_section_two_amount_arithmetic_value",
}
REASONING_LOAD_BASES = {
    SUM_QUERY_ID: 0.34,
    DIFFERENCE_QUERY_ID: 0.38,
}


def _build_expression_plans() -> dict[str, ExpressionPlan]:
    """Bind query branches to their two-operand arithmetic operators."""

    return {
        SUM_QUERY_ID: ExpressionPlan(
            operation_name="sum_two_amounts_in_section",
            operand_count=2,
            operators=("+",),
        ),
        DIFFERENCE_QUERY_ID: ExpressionPlan(
            operation_name="difference_two_amounts_in_section",
            operand_count=2,
            operators=("-",),
            sort_operands_descending=True,
        ),
    }


@register_task
class PagesFormSectionTwoAmountArithmeticValueTask:
    """Compute a two-amount arithmetic result inside one named section."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_form_section_public_entry(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            public_task=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            expression=_build_expression_plans(),
            prompt_query_key=PROMPT_QUERY_KEYS,
            question_format=QUESTION_FORMATS,
            reasoning_load_base=REASONING_LOAD_BASES,
        )


__all__ = [
    "DIFFERENCE_QUERY_ID",
    "PROMPT_QUERY_KEYS",
    "QUESTION_FORMATS",
    "REASONING_LOAD_BASES",
    "SCENE_VARIANTS",
    "SUM_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesFormSectionTwoAmountArithmeticValueTask",
]
