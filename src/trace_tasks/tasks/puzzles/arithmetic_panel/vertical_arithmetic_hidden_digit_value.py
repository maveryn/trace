"""Solve the hidden digit in a vertical arithmetic problem."""

from __future__ import annotations
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from ._lifecycle import ArithmeticObjectivePlan, run_arithmetic_lifecycle
from .shared.rules import build_vertical_arithmetic_case
from .shared.sampling import resolve_integer_target
from .shared.state import SCENE_ID

TASK_ID = "task_puzzles__arithmetic_panel__vertical_arithmetic_hidden_digit_value"
ADDITION_QUERY = "hidden_addition_digit_value"
SUBTRACTION_QUERY = "hidden_subtraction_digit_value"
SUPPORTED_QUERY_IDS = (ADDITION_QUERY, SUBTRACTION_QUERY)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "puzzles", SCENE_ID, task_id=TASK_ID
    )
)


def _hidden_digit_target(instance_seed, params):
    """Choose the digit value that will be hidden in the vertical problem."""

    return resolve_integer_target(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="digit_answer",
        fallback_min=0,
        fallback_max=9,
        namespace="puzzles.arithmetic.vertical_digit",
    )


def _vertical_digit_case_builder(operation, target):
    def construct_case(rng):
        return build_vertical_arithmetic_case(
            rng,
            answer_value=int(target.answer_value),
            answer_support=target.answer_support,
            operation=operation,
        )

    return construct_case


def _addition_digit_objective(instance_seed, params) -> ArithmeticObjectivePlan:
    target = _hidden_digit_target(instance_seed, params)
    return ArithmeticObjectivePlan(
        prompt_query_key=ADDITION_QUERY,
        attempt_namespace="puzzles.arithmetic.vertical_digit.addition",
        query_params={
            "operation": "addition",
            "target_answer": int(target.answer_value),
            "target_answer_support": list(target.answer_support),
            "target_answer_probabilities": dict(target.answer_probabilities),
        },
        construct_case=_vertical_digit_case_builder("addition", target),
    )


def _subtraction_digit_objective(instance_seed, params) -> ArithmeticObjectivePlan:
    target = _hidden_digit_target(instance_seed, params)
    return ArithmeticObjectivePlan(
        prompt_query_key=SUBTRACTION_QUERY,
        attempt_namespace="puzzles.arithmetic.vertical_digit.subtraction",
        query_params={
            "operation": "subtraction",
            "target_answer": int(target.answer_value),
            "target_answer_support": list(target.answer_support),
            "target_answer_probabilities": dict(target.answer_probabilities),
        },
        construct_case=_vertical_digit_case_builder("subtraction", target),
    )


def _build_vertical_digit_objective(
    instance_seed, params, selected_branch
) -> ArithmeticObjectivePlan:
    if str(selected_branch) == ADDITION_QUERY:
        return _addition_digit_objective(instance_seed, params)
    if str(selected_branch) == SUBTRACTION_QUERY:
        return _subtraction_digit_objective(instance_seed, params)
    raise ValueError(f"unknown vertical arithmetic query: {selected_branch!r}")


@register_task
class PuzzlesArithmeticVerticalArithmeticHiddenDigitValueTask:
    """Solve the hidden digit in a vertical arithmetic problem."""

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
            default_query_id=ADDITION_QUERY,
            task_id=TASK_ID,
            namespace="puzzles.arithmetic.vertical_digit.branch",
        )
        objective = _build_vertical_digit_objective(
            int(instance_seed), task_params, selected_branch
        )
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


__all__ = ["PuzzlesArithmeticVerticalArithmeticHiddenDigitValueTask"]
