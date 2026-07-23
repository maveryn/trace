"""Solve algebraic angle relations on parallel-line transversal diagrams."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import build_integer_angle_relation_trace, render_angle_relation_runtime, select_angle_relation_case
from .shared.construction import make_parallel_algebraic_case
from .shared.state import DOMAIN, SCENE_ID, AngleRelationCase


TASK_ID = "task_geometry__angle_relations__parallel_algebraic_angle_value"
PUBLIC_QUERY_ID = "single"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (PUBLIC_QUERY_ID,)
TASK_PROMPT_KEY = "parallel_algebraic_angle_value"
PROMPT_QUERY_KEY = "parallel_target_angle_value"

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _parallel_algebraic_case_support() -> tuple[AngleRelationCase, ...]:
    """Build deterministic support cases for two-expression parallel-line algebra."""

    cases: list[AngleRelationCase] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    coefficient_pairs = ((1, 2), (1, 3), (2, 3), (2, 5), (3, 4), (4, 5))
    for x_value in range(5, 31):
        for first_angle in range(45, 79):
            second_angle = 180 - int(first_angle)
            for first_coeff, second_coeff in coefficient_pairs:
                first_const = int(first_angle) - (int(first_coeff) * int(x_value))
                second_const = int(second_angle) - (int(second_coeff) * int(x_value))
                if not -80 <= first_const <= 120:
                    continue
                if not -80 <= second_const <= 120:
                    continue
                key = (
                    int(x_value),
                    int(first_coeff),
                    int(first_const),
                    int(second_coeff),
                    int(second_const),
                    int(second_angle),
                )
                if key in seen:
                    continue
                seen.add(key)
                cases.append(
                    make_parallel_algebraic_case(
                        x_value=int(x_value),
                        first_coeff=int(first_coeff),
                        first_const=int(first_const),
                        second_coeff=int(second_coeff),
                        second_const=int(second_const),
                    )
                )
    return tuple(cases)


PARALLEL_ALGEBRAIC_CASES = _parallel_algebraic_case_support()


def _answer_probability_map(values: tuple[int, ...]) -> dict[str, float]:
    """Return a compact uniform probability map for trace metadata."""

    support = tuple(sorted(set(int(value) for value in values)))
    if not support:
        return {}
    weight = 1.0 / float(len(support))
    return {str(value): weight for value in support}


def _target_angle_answer_support() -> dict[str, float]:
    """Return uniform answer support for the target angle output."""

    values = tuple(int(case.answer) for case in PARALLEL_ALGEBRAIC_CASES)
    return _answer_probability_map(values)


@register_task
class GeometryAngleRelationsParallelAlgebraicAngleValueTask:
    """Solve a parallel-line algebra relation and return the requested integer."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select a semantic output query, sample a case, and bind answer plus annotation."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=PUBLIC_QUERY_ID,
            task_id=TASK_ID,
        )
        case, case_index = select_angle_relation_case(
            cases=PARALLEL_ALGEBRAIC_CASES,
            params=task_params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.case",
        )
        runtime = render_angle_relation_runtime(
            case=case,
            case_index=int(case_index),
            prompt_query_key=PROMPT_QUERY_KEY,
            prompt_task_key=TASK_PROMPT_KEY,
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
        )
        witness = runtime.rendered_context.rendered_scene.witness
        answer_value = int(witness["target_angle_measure"])
        trace_payload = build_integer_angle_relation_trace(
            runtime=runtime,
            branch_name=str(selected_query),
            branch_probabilities=query_probabilities,
            answer_value=int(answer_value),
            query_params={
                "answer_support_probabilities": _target_angle_answer_support(),
                "target_angle_value": int(witness["target_angle_measure"]),
                "variable_x_value": int(witness["x"]),
                "answer_role": "target_angle_value",
                "target_angle_name": str(witness["target_angle_name"]),
                "expression_angle_names": list(witness["expression_angle_names"]),
            },
            scene_relation_fields={
                "answer_role": "target_angle_value",
                "relation_id": str(witness["relation_id"]),
                "target_angle_name": str(witness["target_angle_name"]),
                "expression_angle_names": list(witness["expression_angle_names"]),
            },
            execution_fields_extra={
                "answer_role": "target_angle_value",
                "target_angle_value": int(witness["target_angle_measure"]),
                "variable_x_value": int(witness["x"]),
                "target_angle_name": str(witness["target_angle_name"]),
                "expression_angle_names": list(witness["expression_angle_names"]),
                "expression_angle_values": list(witness["expression_angle_values"]),
                "first_expression": str(witness["first_expression"]),
                "second_expression": str(witness["second_expression"]),
            },
            witness_fields_extra={
                "answer_role": "target_angle_value",
                "target_angle_value": int(witness["target_angle_measure"]),
                "variable_x_value": int(witness["x"]),
                "target_angle_name": str(witness["target_angle_name"]),
                "expression_angle_names": list(witness["expression_angle_names"]),
                "expression_angle_values": list(witness["expression_angle_values"]),
                "first_expression": str(witness["first_expression"]),
                "second_expression": str(witness["second_expression"]),
            },
        )
        return TaskOutput(
            prompt=str(runtime.prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(answer_value)),
            annotation_gt=TypedValue(
                type=str(runtime.annotation_artifacts.annotation_type),
                value=runtime.annotation_artifacts.value,
            ),
            image=runtime.rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(runtime.prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryAngleRelationsParallelAlgebraicAngleValueTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
