"""Solve an algebraic angle relation for angle ABC."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import build_integer_angle_relation_trace, render_angle_relation_runtime
from .shared.construction import make_algebraic_double_extension_case, make_algebraic_single_extension_case
from .shared.sampling import algebraic_case_parameters_for_answer, select_indexed_case
from .shared.state import DOMAIN, SCENE_ID, AngleRelationCase


TASK_ID = "task_geometry__angle_relations__algebraic_angle_value"
TARGET_ANGLE_VALUE_QUERY_ID = "target_angle_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (TARGET_ANGLE_VALUE_QUERY_ID,)
TASK_PROMPT_KEY = "algebraic_angle_value"

SINGLE_EXTENSION_CASE = "single_extension"
DOUBLE_EXTENSION_CASE = "double_extension"
EXTENSION_CASE_SUPPORT: Tuple[str, ...] = (SINGLE_EXTENSION_CASE,)

ALGEBRAIC_SINGLE_EXTENSION_CASE_SUPPORT: Tuple[Tuple[int, int, int, int, int], ...] = tuple(
    algebraic_case_parameters_for_answer(answer_value, variant_index=3) for answer_value in range(35, 104)
)
ALGEBRAIC_DOUBLE_EXTENSION_CASE_SUPPORT: Tuple[Tuple[int, int, int, int, int], ...] = tuple(
    algebraic_case_parameters_for_answer(answer_value, variant_index=19) for answer_value in range(35, 104)
)

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _cases_for_extension_case(extension_case: str) -> tuple[AngleRelationCase, ...]:
    """Return the construction family selected by task-owned sampling metadata."""

    if str(extension_case) == SINGLE_EXTENSION_CASE:
        return tuple(make_algebraic_single_extension_case(*values) for values in ALGEBRAIC_SINGLE_EXTENSION_CASE_SUPPORT)
    if str(extension_case) == DOUBLE_EXTENSION_CASE:
        return tuple(make_algebraic_double_extension_case(*values) for values in ALGEBRAIC_DOUBLE_EXTENSION_CASE_SUPPORT)
    raise ValueError(f"unsupported extension_case for {TASK_ID}: {extension_case}")


def _select_extension_case(*, params: Mapping[str, Any], instance_seed: int) -> tuple[str, int]:
    """Select single-vs-double construction as trace metadata, not query identity."""

    explicit_extension_case = params.get("extension_case")
    if explicit_extension_case is not None:
        extension_case = str(explicit_extension_case)
        if extension_case not in EXTENSION_CASE_SUPPORT:
            raise ValueError(f"unsupported extension_case for {TASK_ID}: {extension_case}")
        return extension_case, EXTENSION_CASE_SUPPORT.index(extension_case)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.extension_case")
    extension_case_index = int(uniform_choice(rng, tuple(range(len(EXTENSION_CASE_SUPPORT)))))
    return EXTENSION_CASE_SUPPORT[extension_case_index], int(extension_case_index)


def _answer_for_query(*, selected_query: str, witness: Mapping[str, Any]) -> tuple[int, str]:
    """Bind the requested answer from the rendered algebraic execution trace."""

    if str(selected_query) == TARGET_ANGLE_VALUE_QUERY_ID:
        return int(witness["answer_angle_ABC"]), "target_angle_value"
    raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")


@register_task
class GeometryAngleRelationsAlgebraicAngleValueTask:
    """Solve the visible algebraic angle relation and return the requested integer."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select a semantic query, sample a construction case, and bind answer plus annotation."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=TARGET_ANGLE_VALUE_QUERY_ID,
            task_id=TASK_ID,
        )
        extension_case, extension_case_index = _select_extension_case(
            params=task_params,
            instance_seed=int(instance_seed),
        )
        case, case_index = select_indexed_case(
            cases=_cases_for_extension_case(str(extension_case)),
            params=task_params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.case.{extension_case}",
        )
        runtime = render_angle_relation_runtime(
            case=case,
            case_index=int(case_index),
            prompt_query_key=str(selected_query),
            prompt_task_key=TASK_PROMPT_KEY,
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
        )
        witness = runtime.rendered_context.rendered_scene.witness
        answer_value, answer_role = _answer_for_query(selected_query=str(selected_query), witness=witness)
        annotation_gt = TypedValue(
            type=str(runtime.annotation_artifacts.annotation_type),
            value=runtime.annotation_artifacts.value,
        )
        trace_payload = build_integer_angle_relation_trace(
            runtime=runtime,
            branch_name=str(selected_query),
            branch_probabilities=query_probabilities,
            answer_value=int(answer_value),
            query_params={
                "extension_case": str(extension_case),
                "extension_case_index": int(extension_case_index),
                "target_angle_value": int(witness["answer_angle_ABC"]),
                "variable_x_value": int(witness["x"]),
                "answer_role": str(answer_role),
            },
            scene_relation_fields={
                "extension_case": str(extension_case),
                "answer_role": str(answer_role),
            },
            execution_fields_extra={
                "extension_case": str(extension_case),
                "extension_case_index": int(extension_case_index),
                "target_angle_value": int(witness["answer_angle_ABC"]),
                "variable_x_value": int(witness["x"]),
                "answer_role": str(answer_role),
            },
            witness_fields_extra={
                "extension_case": str(extension_case),
                "target_angle_value": int(witness["answer_angle_ABC"]),
                "variable_x_value": int(witness["x"]),
                "answer_role": str(answer_role),
            },
        )
        return TaskOutput(
            prompt=str(runtime.prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(answer_value)),
            annotation_gt=annotation_gt,
            image=runtime.rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(runtime.prompt_artifacts.prompt_variants),
        )


__all__ = ["GeometryAngleRelationsAlgebraicAngleValueTask"]
