"""Solve one hidden brick in an addition number wall."""

from __future__ import annotations
from dataclasses import dataclass

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from ._lifecycle import ArithmeticObjectivePlan, run_arithmetic_lifecycle
from .shared.rules import build_number_wall_case
from .shared.sampling import resolve_integer_target
from .shared.state import SCENE_ID

TASK_ID = "task_puzzles__arithmetic_panel__number_wall_value"
ADDITION_WALL_QUERY = "addition_wall_missing_value"
SUPPORTED_QUERY_IDS = (ADDITION_WALL_QUERY,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "puzzles", SCENE_ID, task_id=TASK_ID
    )
)


@dataclass(frozen=True)
class NumberWallBranchSpec:
    """Task-local query contract for one number-wall construction family."""

    query_id: str
    wall_kind: str
    support_key: str
    fallback_min: int
    fallback_max: int

    @property
    def namespace(self) -> str:
        return f"puzzles.arithmetic.number_wall.{self.wall_kind}"


_BRANCH_SPECS = {
    ADDITION_WALL_QUERY: NumberWallBranchSpec(
        query_id=ADDITION_WALL_QUERY,
        wall_kind="addition",
        support_key="wall_answer",
        fallback_min=2,
        fallback_max=24,
    ),
}


def _build_number_wall_objective(
    instance_seed, params, selected_branch
) -> ArithmeticObjectivePlan:
    branch_spec = _BRANCH_SPECS[str(selected_branch)]
    target = resolve_integer_target(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key=branch_spec.support_key,
        fallback_min=branch_spec.fallback_min,
        fallback_max=branch_spec.fallback_max,
        namespace=branch_spec.namespace,
    )
    return ArithmeticObjectivePlan(
        prompt_query_key=branch_spec.query_id,
        attempt_namespace=branch_spec.namespace,
        query_params={
            "wall_kind": branch_spec.wall_kind,
            "target_answer": int(target.answer_value),
            "target_answer_support": list(target.answer_support),
            "target_answer_probabilities": dict(target.answer_probabilities),
        },
        construct_case=lambda rng: build_number_wall_case(
            rng,
            answer_value=int(target.answer_value),
            answer_support=target.answer_support,
            wall_kind=branch_spec.wall_kind,
        ),
    )


@register_task
class PuzzlesArithmeticNumberWallValueTask:
    """Solve one hidden brick in an addition number wall."""

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
            default_query_id=ADDITION_WALL_QUERY,
            task_id=TASK_ID,
            namespace="puzzles.arithmetic.number_wall.branch",
        )
        objective = _build_number_wall_objective(
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


__all__ = ["PuzzlesArithmeticNumberWallValueTask"]
