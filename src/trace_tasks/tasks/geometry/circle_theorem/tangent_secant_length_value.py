"""Tangent-secant length circle-theorem task."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_integer_circle_theorem_task
from .shared.construction import _build_tangent_secant_scene
from .shared.sampling import _feasible_tangent_secant_target_kinds
from .shared.state import TANGENT_SECANT_ANSWER_SUPPORT, CircleTheoremProblem

TASK_ID = "task_geometry__circle_theorem__tangent_secant_length_value"
SUPPORTED_QUERY_IDS = ("tangent_secant_length",)
ANSWER_SUPPORT = TANGENT_SECANT_ANSWER_SUPPORT


def _select_tangent_target_kind(instance_seed, params, target_answer):
    feasible = _feasible_tangent_secant_target_kinds(int(target_answer))
    if not feasible:
        raise ValueError(f"unsupported tangent-secant target answer: {target_answer}")
    explicit = params.get("tangent_secant_target_kind")
    selected = str(explicit) if explicit is not None else str(
        spawn_rng(int(instance_seed), f"{TASK_ID}.target_kind").choice(feasible)
    )
    if selected not in set(feasible):
        raise ValueError(f"unsupported tangent-secant target kind: {selected!r}")
    probability = 1.0 / float(len(feasible))
    return selected, {str(kind): float(probability) for kind in feasible}


def _bind_problem(instance_seed, params, _selected_query, target_answer, target_probs):
    """Bind the hidden segment axis for the tangent-secant power theorem."""

    target_kind, kind_probs = _select_tangent_target_kind(
        instance_seed=int(instance_seed),
        params=params,
        target_answer=int(target_answer),
    )
    problem = CircleTheoremProblem(
        target_answer=int(target_answer),
        target_answer_probabilities=dict(target_probs),
        tangent_secant_target_kind=str(target_kind),
        tangent_secant_target_kind_probabilities=dict(kind_probs),
    )
    return (
        problem,
        {
            "tangent_secant_target_kind": str(target_kind),
            "tangent_secant_target_kind_probabilities": dict(kind_probs),
        },
        lambda rng, bound: _build_tangent_secant_scene(rng, problem=bound),
    )


@register_task
class GeometryCircleTangentSecantLengthValueTask:
    """Solve a missing length from one tangent and one secant."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_integer_circle_theorem_task(
            task_id=TASK_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            answer_support=ANSWER_SUPPORT,
            bind_problem=_bind_problem,
        )
