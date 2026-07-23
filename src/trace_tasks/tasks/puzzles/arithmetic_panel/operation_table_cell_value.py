"""Solve one hidden cell in an arithmetic operation table."""

from __future__ import annotations
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from ._lifecycle import ArithmeticObjectivePlan, run_arithmetic_lifecycle
from .shared.rules import build_operation_table_case
from .shared.sampling import resolve_integer_target
from .shared.state import SCENE_ID

TASK_ID = "task_puzzles__arithmetic_panel__operation_table_cell_value"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "operation_table_cell_value"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "puzzles", SCENE_ID, task_id=TASK_ID
    )
)


def _build_operation_table_objective(instance_seed, params) -> ArithmeticObjectivePlan:
    target = resolve_integer_target(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="answer",
        fallback_min=1,
        fallback_max=24,
        namespace="puzzles.arithmetic.operation_table",
    )
    return ArithmeticObjectivePlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        attempt_namespace="puzzles.arithmetic.operation_table",
        query_params={
            "target_answer": int(target.answer_value),
            "target_answer_support": list(target.answer_support),
            "target_answer_probabilities": dict(target.answer_probabilities),
        },
        construct_case=lambda rng: build_operation_table_case(
            rng,
            answer_value=int(target.answer_value),
            answer_support=target.answer_support,
        ),
    )


@register_task
class PuzzlesArithmeticOperationTableCellValueTask:
    """Solve one hidden cell in an arithmetic operation table."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "puzzles"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        selected_branch, branch_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
            namespace="puzzles.arithmetic.operation_table.branch",
        )
        objective = _build_operation_table_objective(int(instance_seed), task_params)
        return run_arithmetic_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            selected_query_id=selected_branch,
            query_probabilities=branch_probs,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            params=task_params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=objective,
        )


__all__ = ["PuzzlesArithmeticOperationTableCellValueTask"]
