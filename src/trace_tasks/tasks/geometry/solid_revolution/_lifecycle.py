"""Neutral rendering, prompt, and trace plumbing for solid-revolution tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import annotation_roles_metadata, bbox_map_annotation
from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, SCENE_KIND, load_solid_revolution_defaults
from .shared.measurements import answer_support_probability_map, round_volume, volume_cylinder
from .shared.prompts import solid_revolution_prompt_artifacts
from .shared.rendering import create_render_context, render_cylinder_revolution
from .shared.sampling import cylinder_diagonal_case_pool, select_case_from_pool, support_from_cases
from .shared.state import RenderContext, RenderedSolidRevolutionScene, SolidRevolutionProblem

RenderBuilder = Callable[[RenderContext, SolidRevolutionProblem], RenderedSolidRevolutionScene]


@dataclass(frozen=True)
class SolidRevolutionObjectivePlan:
    """Task-owned objective binding prepared by one public task file."""

    prompt_key: str
    problem: SolidRevolutionProblem
    render_scene: RenderBuilder
    annotation_keys: tuple[str, ...]
    answer_value: float
    query_params: Mapping[str, Any]
    trace_values: Mapping[str, Any]


@dataclass(frozen=True)
class SolidRevolutionTaskParts:
    """Prepared non-verifier output fields for one public task."""

    prompt: str
    prompt_variants: dict[str, str]
    image: Image.Image
    annotation_value: dict[str, list[float]]
    trace_payload: dict[str, Any]
    task_versions: dict[str, str]
    scene_id: str


def build_solid_revolution_plan(
    *,
    prompt_key: str,
    problem: SolidRevolutionProblem,
    render_scene: RenderBuilder,
    annotation_keys: Sequence[str],
    branch_probabilities: Mapping[str, float],
    support_probabilities: Mapping[str, float],
) -> SolidRevolutionObjectivePlan:
    """Assemble neutral plan fields after a public task binds formula values."""

    problem_values = _problem_trace_values(problem)
    return SolidRevolutionObjectivePlan(
        prompt_key=str(prompt_key),
        problem=problem,
        render_scene=render_scene,
        annotation_keys=tuple(str(value) for value in annotation_keys),
        answer_value=float(problem.answer),
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "target_support_probabilities": dict(support_probabilities),
            **problem_values,
        },
        trace_values=problem_values,
    )


def _problem_trace_values(problem: SolidRevolutionProblem) -> dict[str, Any]:
    """Serialize task-bound revolution measurements into trace metadata."""

    return {
        "solid_kind": str(problem.solid_kind),
        "generating_shape": str(problem.generating_shape),
        "formula_family": str(problem.formula_family),
        "formula": str(problem.formula),
        "radius": None if problem.radius is None else float(problem.radius),
        "diameter": None if problem.diameter is None else float(problem.diameter),
        "radial_input_kind": problem.radial_input_kind,
        "height": None if problem.height is None else float(problem.height),
        "slant_height": None if problem.slant_height is None else float(problem.slant_height),
        "diagonal": None if problem.diagonal is None else float(problem.diagonal),
        "half_height": None if problem.half_height is None else float(problem.half_height),
        "top_radius": None if problem.top_radius is None else float(problem.top_radius),
        "bottom_radius": None if problem.bottom_radius is None else float(problem.bottom_radius),
        "total_height": None if problem.total_height is None else float(problem.total_height),
        "rotation_degrees": 360,
        "answer_support_size": len(problem.answer_support_probabilities or {}),
        "construction_case_count_for_answer": int(problem.construction_case_count_for_answer),
    }


def _trace_payload(
    *,
    task_id: str,
    selected_query: str,
    branch_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    image_size: tuple[int, int],
    plan: SolidRevolutionObjectivePlan,
    rendered: RenderedSolidRevolutionScene,
    projected_annotation: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    style_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build trace sections after the public task binds objective values."""

    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "query_id_probabilities": dict(branch_probabilities),
        **dict(plan.query_params),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID
    trace_values = {
        "answer_type": "number",
        "answer_value": float(plan.answer_value),
        "answer_rounding": "one_decimal",
        "annotation_roles": annotation_roles_metadata(projected_annotation.get("bbox_map", {})),
        **dict(plan.trace_values),
    }
    return {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "type": str(plan.problem.formula_family),
                **dict(trace_values),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "style": dict(style_meta),
            "post_image_noise": dict(noise_meta),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "query_id_probabilities": dict(branch_probabilities),
            **dict(trace_values),
        },
        "witness_symbolic": {
            "type": "solid_revolution_formula",
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "answer_value": float(plan.answer_value),
            "source_witness_type": "bbox_map",
            "original_annotation_value": dict(projected_annotation.get("bbox_map", {})),
            **dict(plan.trace_values),
        },
        "projected_annotation": dict(projected_annotation),
    }


def prepare_solid_revolution_task_parts(
    *,
    task_id: str,
    selected_query: str,
    branch_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    plan: SolidRevolutionObjectivePlan,
    instance_seed: int,
    max_attempts: int,
) -> SolidRevolutionTaskParts:
    """Render shared visual artifacts after the public task binds the answer."""

    _generation_defaults, render_defaults, prompt_defaults = load_solid_revolution_defaults(str(task_id))
    last_error: Exception | None = None
    rendered: RenderedSolidRevolutionScene | None = None
    ctx: RenderContext | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            attempt_seed = int(instance_seed) + int(attempt_index)
            ctx = create_render_context(
                instance_seed=attempt_seed,
                params={**dict(params), "_render_attempt": int(attempt_index)},
                rendering_defaults=render_defaults,
                random_namespace=f"{task_id}.render",
            )
            rendered = plan.render_scene(ctx, plan.problem)
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None or ctx is None:
        raise RuntimeError(f"failed to generate {task_id}") from last_error

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_gt, projected_annotation = bbox_map_annotation(rendered, plan.annotation_keys)
    _prompt_defaults, prompt_artifacts = solid_revolution_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_key=str(plan.prompt_key),
        annotation_keys=tuple(plan.annotation_keys),
        answer=float(plan.answer_value),
        instance_seed=int(instance_seed),
    )
    trace_payload = _trace_payload(
        task_id=str(task_id),
        selected_query=str(selected_query),
        branch_probabilities=branch_probabilities,
        prompt_artifacts=prompt_artifacts,
        image_size=(int(image.size[0]), int(image.size[1])),
        plan=plan,
        rendered=rendered,
        projected_annotation=projected_annotation,
        noise_meta=noise_meta,
        style_meta={
            "background": dict(ctx.background_meta),
            "diagram_style": dict(ctx.diagram_style_meta),
            "font": dict(ctx.font_meta),
            "palette": dict(ctx.palette_meta),
            "line_width": int(ctx.line_width),
        },
    )
    return SolidRevolutionTaskParts(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        image=image,
        annotation_value=dict(annotation_gt.value),
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
    )


def _run_cylinder_diagonal(
    *,
    task_id: str,
    query_id: str,
    supported_query_ids: Sequence[str],
    annotation_keys: Sequence[str],
    instance_seed: int,
    params: dict[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the diagonal-derived cylinder objective without duplicating a public body."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    cases = cylinder_diagonal_case_pool()
    case = select_case_from_pool(
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=f"{task_id}.{selected_query}.case",
        cases=cases,
    )
    support_probabilities = answer_support_probability_map(support_from_cases(cases), case.answer)
    problem = SolidRevolutionProblem(
        solid_kind="cylinder",
        generating_shape="rectangle",
        answer=round_volume(volume_cylinder(diameter=case.diameter, height=case.height)),
        formula_family="cylinder_volume_from_diagonal_rectangle",
        formula="d^2 = diag^2 - h^2, then V = pi (d/2)^2 h",
        radius=float(case.diameter) / 2.0,
        diameter=float(case.diameter),
        radial_input_kind="diagonal",
        height=float(case.height),
        diagonal=float(case.diagonal or 0),
        answer_support_probabilities=support_probabilities,
        construction_case_count_for_answer=1,
    )
    plan = build_solid_revolution_plan(
        prompt_key=str(query_id),
        problem=problem,
        render_scene=render_cylinder_revolution,
        annotation_keys=tuple(str(key) for key in annotation_keys),
        branch_probabilities=query_probabilities,
        support_probabilities=support_probabilities,
    )
    parts = prepare_solid_revolution_task_parts(
        task_id=str(task_id),
        selected_query=str(selected_query),
        branch_probabilities=query_probabilities,
        params=task_params,
        plan=plan,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
    )
    return TaskOutput(
        prompt=parts.prompt,
        answer_gt=TypedValue(type="number", value=float(plan.answer_value)),
        annotation_gt=TypedValue(type="bbox_map", value=dict(parts.annotation_value)),
        image=parts.image,
        image_id="img0",
        trace_payload=parts.trace_payload,
        task_versions=parts.task_versions,
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(parts.prompt_variants),
    )


__all__ = [
    "SolidRevolutionObjectivePlan",
    "SolidRevolutionTaskParts",
    "build_solid_revolution_plan",
    "prepare_solid_revolution_task_parts",
    "_run_cylinder_diagonal",
]
