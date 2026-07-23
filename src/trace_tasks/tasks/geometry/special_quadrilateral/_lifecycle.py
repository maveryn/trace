"""Neutral render plumbing for special-quadrilateral tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import special_quadrilateral_point_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, load_special_quadrilateral_defaults
from .shared.rendering import create_render_context, draw_special_quadrilateral_scene
from .shared.sampling import select_case_from_answer_support
from .shared.state import (
    SCENE_ID,
    QuadrilateralCase,
    RenderedSpecialQuadrilateralScene,
    SpecialQuadrilateralProblem,
)


@dataclass(frozen=True)
class SpecialQuadrilateralRenderParts:
    """Rendered scene artifacts that are independent of public task identity."""

    image: Image.Image
    rendered: RenderedSpecialQuadrilateralScene
    annotation_artifacts: PixelAnnotationArtifacts
    noise_meta: dict[str, Any]
    prompt_defaults: Mapping[str, Any]
    task_versions: dict[str, str]


def expression_payload(case: QuadrilateralCase) -> dict[str, Any]:
    """Serialize visible linear expressions when the resolved case uses them."""

    if case.target_expression is None or case.support_expression is None:
        return {}
    x_value = int(case.x_value or 0)
    return {
        "x_value": int(x_value),
        "target_expression": {
            "coefficient": int(case.target_expression.coefficient),
            "constant": int(case.target_expression.constant),
            "value": int(case.target_expression.evaluate(x_value)),
            "display": str(case.target_label),
        },
        "support_expression": {
            "coefficient": int(case.support_expression.coefficient),
            "constant": int(case.support_expression.constant),
            "value": int(case.support_expression.evaluate(x_value)),
            "display": str(case.support_label),
        },
    }


def render_special_quadrilateral_problem(
    *,
    problem: SpecialQuadrilateralProblem,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> SpecialQuadrilateralRenderParts:
    """Render one resolved special-quadrilateral case and project annotation."""

    _generation_defaults, render_defaults, prompt_defaults = load_special_quadrilateral_defaults()
    last_error: Exception | None = None
    rendered: RenderedSpecialQuadrilateralScene | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            attempt_seed = int(instance_seed) + int(attempt)
            ctx = create_render_context(
                instance_seed=int(attempt_seed),
                params=params,
                render_defaults=render_defaults,
            )
            rendered = draw_special_quadrilateral_scene(
                problem=replace(problem, layout_seed=int(attempt_seed)),
                ctx=ctx,
            )
            rendered.render_map["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            rendered.render_map["style"] = {
                "technical_diagram": dict(ctx.diagram_style_meta),
                "background": dict(ctx.background_meta),
            }
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise RuntimeError("failed to render special quadrilateral scene") from last_error
    if rendered is None:
        raise RuntimeError("failed to render special quadrilateral scene")

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = special_quadrilateral_point_annotation(rendered)
    return SpecialQuadrilateralRenderParts(
        image=image,
        rendered=rendered,
        annotation_artifacts=annotation_artifacts,
        noise_meta=dict(noise_meta),
        prompt_defaults=dict(prompt_defaults),
        task_versions=default_task_versions(),
    )


def _select_expression_problem(
    *,
    task_id: str,
    supported_queries: Sequence[str],
    cases_by_branch: Mapping[str, tuple[QuadrilateralCase, ...]],
    params: dict[str, Any],
    instance_seed: int,
    unsupported_query_subject: str,
) -> tuple[str, dict[str, float], dict[str, Any], SpecialQuadrilateralProblem, dict[str, float]]:
    """Resolve one public query and answer-balanced expression case."""

    selected_query, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(query) for query in supported_queries),
        default_query_id=str(tuple(supported_queries)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    cases = cases_by_branch.get(str(selected_query))
    if not cases:
        raise ValueError(f"unsupported special quadrilateral {unsupported_query_subject} query: {selected_query}")
    case, case_index, answer_probabilities = select_case_from_answer_support(
        cases=cases,
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.{selected_query}.case",
    )
    problem = SpecialQuadrilateralProblem(
        case=case,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )
    return (
        str(selected_query),
        dict(branch_probabilities),
        dict(task_params),
        problem,
        dict(answer_probabilities),
    )


def _expression_query_params(
    *,
    branch_probabilities: Mapping[str, float],
    answer_probabilities: Mapping[str, float],
    problem: SpecialQuadrilateralProblem,
) -> dict[str, Any]:
    """Build query metadata shared by expression-based objectives."""

    return {
        "scene_id": SCENE_ID,
        "query_id_probabilities": dict(branch_probabilities),
        "answer_support_probabilities": dict(answer_probabilities),
        "case_index": int(problem.case_index),
        "shape_kind": str(problem.case.shape_kind),
        "theorem": str(problem.case.theorem),
        **expression_payload(problem.case),
    }


def _run_expression_relation(
    *,
    task_id: str,
    supported_queries: Sequence[str],
    cases_by_branch: Mapping[str, tuple[QuadrilateralCase, ...]],
    task_prompt_key: str,
    witness_type: str,
    unsupported_query_subject: str,
    instance_seed: int,
    params: dict[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the common expression-relation lifecycle for one public objective."""

    from .shared.output import common_trace_sections, prompt_artifacts_for_bound_case

    selected_query, branch_probabilities, task_params, problem, answer_probabilities = _select_expression_problem(
        task_id=str(task_id),
        supported_queries=tuple(str(query) for query in supported_queries),
        cases_by_branch=cases_by_branch,
        params=dict(params),
        instance_seed=int(instance_seed),
        unsupported_query_subject=str(unsupported_query_subject),
    )
    parts = render_special_quadrilateral_problem(
        problem=problem,
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=max(1, int(max_attempts)),
    )
    prompt_artifacts = prompt_artifacts_for_bound_case(
        prompt_defaults=parts.prompt_defaults,
        task_prompt_key=str(task_prompt_key),
        branch_prompt_key=str(selected_query),
        target_name=str(problem.case.target_name),
        annotation_roles=tuple(parts.annotation_artifacts.value.keys()),
        answer_value=int(problem.case.answer),
        instance_seed=int(instance_seed),
    )
    trace_payload = common_trace_sections(
        branch_probabilities=dict(branch_probabilities),
        answer_probabilities=dict(answer_probabilities),
        prompt_artifacts=prompt_artifacts,
        problem=problem,
        parts=parts,
        extra_case_values=expression_payload(problem.case),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=_expression_query_params(
            branch_probabilities=branch_probabilities,
            answer_probabilities=answer_probabilities,
            problem=problem,
        ),
    )
    query_spec["scene_id"] = SCENE_ID
    trace_payload["query_spec"] = query_spec
    trace_payload["scene_ir"].update({"task_id": str(task_id), "query_id": str(selected_query)})
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query)
    trace_payload["render_spec"].update({"task_id": str(task_id), "query_id": str(selected_query)})
    trace_payload["execution_trace"]["query_id"] = str(selected_query)
    trace_payload["witness_symbolic"] = {
        "type": str(witness_type),
        "task_id": str(task_id),
        **dict(trace_payload["execution_trace"]),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(problem.case.answer)),
        annotation_gt=TypedValue(
            type=parts.annotation_artifacts.annotation_type,
            value=parts.annotation_artifacts.value,
        ),
        image=parts.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=parts.task_versions,
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "SpecialQuadrilateralRenderParts",
    "expression_payload",
    "render_special_quadrilateral_problem",
    "_run_expression_relation",
]
