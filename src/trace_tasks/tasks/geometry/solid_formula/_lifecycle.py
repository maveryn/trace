"""Neutral rendering, prompt, and trace plumbing for solid-formula tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import annotation_roles_metadata, solid_bbox_annotation
from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, SCENE_KIND, load_solid_formula_defaults
from .shared.prompts import solid_formula_prompt_artifacts
from .shared.rendering import create_render_context
from .shared.state import RenderContext, RenderedSolidFormulaScene, SolidFormulaProblem

RenderBuilder = Callable[[RenderContext, SolidFormulaProblem], RenderedSolidFormulaScene]


@dataclass(frozen=True)
class SolidFormulaObjectivePlan:
    """Task-owned objective binding prepared by one public task file."""

    prompt_key: str
    problem: SolidFormulaProblem
    render_scene: RenderBuilder
    answer_value: float
    query_params: Mapping[str, Any]
    trace_values: Mapping[str, Any]


@dataclass(frozen=True)
class SolidFormulaTaskParts:
    """Prepared non-verifier output fields for one public task."""

    prompt: str
    prompt_variants: dict[str, str]
    image: Image.Image
    annotation_value: list[float]
    trace_payload: dict[str, Any]
    task_versions: dict[str, str]
    scene_id: str


def build_solid_formula_plan(
    *,
    prompt_key: str,
    problem: SolidFormulaProblem,
    render_scene: RenderBuilder,
    branch_probabilities: Mapping[str, float],
    support_probabilities: Mapping[str, float],
) -> SolidFormulaObjectivePlan:
    """Assemble neutral plan fields after a public task binds formula values."""

    return SolidFormulaObjectivePlan(
        prompt_key=str(prompt_key),
        problem=problem,
        render_scene=render_scene,
        answer_value=float(problem.answer),
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "target_support_probabilities": dict(support_probabilities),
            **_problem_trace_values(problem),
        },
        trace_values=_problem_trace_values(problem),
    )


def _problem_trace_values(problem: SolidFormulaProblem) -> dict[str, Any]:
    """Serialize task-bound formula measurements into trace metadata."""

    return {
        "solid_kind": str(problem.solid_kind),
        "unknown_dimension": str(problem.unknown_dimension),
        "radius": None if problem.radius is None else float(problem.radius),
        "total_height": None if problem.total_height is None else float(problem.total_height),
        "cylinder_height": None if problem.cylinder_height is None else float(problem.cylinder_height),
        "cone_height": None if problem.cone_height is None else float(problem.cone_height),
        "volume": None if problem.volume is None else float(problem.volume),
        "volume_pi_multiple": None
        if problem.volume_pi_multiple is None
        else float(problem.volume_pi_multiple),
        "side_a": None if problem.side_a is None else float(problem.side_a),
        "side_b": None if problem.side_b is None else float(problem.side_b),
        "prism_height": None if problem.prism_height is None else float(problem.prism_height),
        "pyramid_height": None if problem.pyramid_height is None else float(problem.pyramid_height),
        "triangle_base": None if problem.triangle_base is None else float(problem.triangle_base),
        "prism_length": None if problem.prism_length is None else float(problem.prism_length),
        "wall_height": None if problem.wall_height is None else float(problem.wall_height),
        "roof_height": None if problem.roof_height is None else float(problem.roof_height),
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
    plan: SolidFormulaObjectivePlan,
    rendered: RenderedSolidFormulaScene,
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
        "formula_family": str(plan.problem.formula_family),
        "formula": str(plan.problem.formula),
        "answer_type": "number",
        "answer_value": float(plan.answer_value),
        "answer_rounding": "one_decimal",
        "annotation_roles": annotation_roles_metadata(projected_annotation),
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
            "type": str(plan.problem.formula_family),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "answer_value": float(plan.answer_value),
            "source_witness_type": str(projected_annotation.get("type", "bbox")),
            "original_annotation_value": list(projected_annotation.get("bbox", [])),
            **dict(plan.trace_values),
        },
        "projected_annotation": dict(projected_annotation),
    }


def prepare_solid_formula_task_parts(
    *,
    task_id: str,
    selected_query: str,
    branch_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    plan: SolidFormulaObjectivePlan,
    instance_seed: int,
    max_attempts: int,
) -> SolidFormulaTaskParts:
    """Render shared visual artifacts after the public task binds the answer."""

    _generation_defaults, render_defaults, prompt_defaults = load_solid_formula_defaults(str(task_id))
    last_error: Exception | None = None
    rendered: RenderedSolidFormulaScene | None = None
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
    annotation_gt, projected_annotation = solid_bbox_annotation(rendered)
    _prompt_defaults, prompt_artifacts = solid_formula_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_key=str(plan.prompt_key),
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
            "label_stroke_width": int(ctx.label_stroke_width),
        },
    )
    return SolidFormulaTaskParts(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        image=image,
        annotation_value=list(annotation_gt.value),
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
    )


__all__ = [
    "SolidFormulaObjectivePlan",
    "SolidFormulaTaskParts",
    "build_solid_formula_plan",
    "prepare_solid_formula_task_parts",
]
