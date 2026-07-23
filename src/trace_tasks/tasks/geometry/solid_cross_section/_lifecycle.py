"""Neutral rendering, prompt, and trace plumbing for solid cross-section tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import annotation_roles_metadata, cross_section_bbox_annotation
from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, SCENE_KIND, load_solid_cross_section_defaults
from .shared.prompts import solid_cross_section_prompt_artifacts
from .shared.rendering import create_render_context
from .shared.state import RenderContext, RenderedSolidCrossSectionScene, SolidCrossSectionProblem

RenderBuilder = Callable[[RenderContext, SolidCrossSectionProblem], RenderedSolidCrossSectionScene]


@dataclass(frozen=True)
class SolidCrossSectionObjectivePlan:
    """Task-owned objective binding prepared by one public task file."""

    prompt_key: str
    object_description: str
    problem: SolidCrossSectionProblem
    render_scene: RenderBuilder
    answer_value: float
    query_params: Mapping[str, Any]
    trace_values: Mapping[str, Any]


@dataclass(frozen=True)
class SolidCrossSectionTaskParts:
    """Prepared non-verifier output fields for one public task."""

    prompt: str
    prompt_variants: dict[str, str]
    image: Image.Image
    annotation_value: list[float]
    trace_payload: dict[str, Any]
    task_versions: dict[str, str]
    scene_id: str


def _trace_payload(
    *,
    task_id: str,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    image_size: tuple[int, int],
    plan: SolidCrossSectionObjectivePlan,
    rendered: RenderedSolidCrossSectionScene,
    projected_annotation: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    style_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build trace sections after the public task binds objective values."""

    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "query_id_probabilities": dict(query_probabilities),
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
            "query_id_probabilities": dict(query_probabilities),
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


def prepare_solid_cross_section_task_parts(
    *,
    task_id: str,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    plan: SolidCrossSectionObjectivePlan,
    instance_seed: int,
    max_attempts: int,
) -> SolidCrossSectionTaskParts:
    """Render shared visual artifacts after the public task binds the answer."""

    _generation_defaults, render_defaults, prompt_defaults = load_solid_cross_section_defaults(str(task_id))
    last_error: Exception | None = None
    rendered: RenderedSolidCrossSectionScene | None = None
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
    annotation_gt, projected_annotation = cross_section_bbox_annotation(rendered)
    _prompt_defaults, prompt_artifacts = solid_cross_section_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_key=str(plan.prompt_key),
        object_description=str(plan.object_description),
        answer=float(plan.answer_value),
        instance_seed=int(instance_seed),
    )
    trace_payload = _trace_payload(
        task_id=str(task_id),
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
        prompt_artifacts=prompt_artifacts,
        image_size=image.size,
        plan=plan,
        rendered=rendered,
        projected_annotation=projected_annotation,
        noise_meta=noise_meta,
        style_meta={
            "technical_diagram": dict(ctx.diagram_style_meta),
            "background": dict(ctx.background_meta),
            "font": dict(ctx.font_meta),
            "palette": dict(ctx.palette_meta),
        },
    )
    return SolidCrossSectionTaskParts(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        image=image,
        annotation_value=list(annotation_gt.value),
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
    )


__all__ = [
    "prepare_solid_cross_section_task_parts",
    "SolidCrossSectionObjectivePlan",
    "SolidCrossSectionTaskParts",
]
