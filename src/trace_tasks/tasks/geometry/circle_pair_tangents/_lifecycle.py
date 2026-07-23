"""Neutral render and trace plumbing for circle-pair tangent public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

from .shared.annotations import keyed_point_annotation
from .shared.prompts import pair_tangent_prompt_artifacts
from .shared.rendering import create_pair_tangent_render_context, render_pair_tangent_scene
from .shared.state import (
    PairTangentDiagramSpec,
    PairTangentRenderContext,
    RenderedPairTangentScene,
    SCENE_ID,
)
from .shared.construction import TangentLayout, tangent_answer_support


@dataclass(frozen=True)
class PairTangentValueProblem:
    """Task-bound tangent value facts supplied by a public objective file."""

    diagram_spec: PairTangentDiagramSpec
    answer: int
    radius_o1: int
    radius_o2: int
    center_distance: int
    tangent_length: int
    radius_difference: int
    larger_circle_side: str
    tangent_side: str
    formula_family: str
    formula: str
    unknown_role: str
    tangent_case_key: str
    tangent_case_probabilities: dict[str, float]
    larger_side_probabilities: dict[str, float]
    tangent_side_probabilities: dict[str, float]
    answer_support_probabilities: dict[str, float]


@dataclass(frozen=True)
class PairTangentPreparedScene:
    """Rendered image, prompt, and projected annotation primitives."""

    render_context: PairTangentRenderContext
    rendered: RenderedPairTangentScene
    image: Any
    noise_meta: dict[str, Any]
    prompt_artifacts: Any
    annotation_value: dict[str, list[float]]


def prepare_pair_tangent_scene(
    *,
    diagram_spec: PairTangentDiagramSpec,
    prompt_defaults: Mapping[str, Any],
    prompt_key: str,
    answer_value: int,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
    style_namespace: str,
) -> PairTangentPreparedScene:
    """Render, noise, prompt, and annotation primitives for one tangent diagram."""

    rendered: RenderedPairTangentScene | None = None
    render_context: PairTangentRenderContext | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_params = dict(params)
        attempt_params["_render_attempt"] = int(attempt)
        try:
            render_context = create_pair_tangent_render_context(
                instance_seed=int(instance_seed) + int(attempt),
                params=attempt_params,
                rendering_defaults=render_defaults,
            )
            rendered = render_pair_tangent_scene(
                render_context,
                diagram_spec,
                instance_seed=int(instance_seed) + int(attempt),
                render_namespace=str(style_namespace),
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None or render_context is None:
        raise RuntimeError("failed to render circle-pair tangent scene") from last_error

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    _prompt_defaults, prompt_artifacts = pair_tangent_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(prompt_key),
        answer_value=int(answer_value),
        annotation_keys=rendered.annotation_roles,
        instance_seed=int(instance_seed),
    )
    return PairTangentPreparedScene(
        render_context=render_context,
        rendered=rendered,
        image=image,
        noise_meta=dict(noise_meta),
        prompt_artifacts=prompt_artifacts,
        annotation_value=keyed_point_annotation(rendered),
    )


def pair_tangent_trace_base(
    *,
    prepared: PairTangentPreparedScene,
    branch_name: str,
    branch_params: Mapping[str, Any],
    unknown_role: str,
    center_distance: int,
    radius_difference: int,
    tangent_length: int,
) -> dict[str, Any]:
    """Build objective-neutral trace sections for one prepared tangent scene."""

    query_spec = build_prompt_query_spec(
        prompt_artifacts=prepared.prompt_artifacts,
        query_id=str(branch_name),
        params=dict(branch_params),
    )
    query_spec["scene_id"] = SCENE_ID
    return {
        "scene_ir": {
            "domain": "geometry",
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "entities": [dict(entity) for entity in prepared.rendered.scene_entities],
            "relations": {
                "type": "external_common_tangent_right_triangle",
                "center_distance": int(center_distance),
                "radius_difference": int(radius_difference),
                "tangent_length": int(tangent_length),
                "unknown_role": str(unknown_role),
                "annotation_roles": list(prepared.rendered.annotation_roles),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "canvas": {"width": int(prepared.image.size[0]), "height": int(prepared.image.size[1])},
            "single_object_scene_rotation": prepared.render_context.scene_transform.metadata(),
            "style": {
                "technical_diagram": dict(prepared.render_context.diagram_style_meta),
                "background": dict(prepared.render_context.background_meta),
                "font_bold": False,
                "label_stroke_width": int(prepared.render_context.label_stroke_width),
                "post_image_noise": dict(prepared.noise_meta),
            },
            "prompt": {
                "prompt_variant": dict(prepared.prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prepared.prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prepared.prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": dict(prepared.rendered.render_map),
        "projected_annotation": {
            "type": "point_map",
            "point_map": dict(prepared.annotation_value),
            "pixel_point_map": dict(prepared.annotation_value),
        },
    }


def pair_tangent_trace_payload(
    *,
    prepared: PairTangentPreparedScene,
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    problem: PairTangentValueProblem,
) -> dict[str, Any]:
    """Build trace metadata from one public task-bound tangent value problem."""

    branch_params = {
        "query_id": str(branch_name),
        "query_id_probabilities": dict(branch_probabilities),
        "tangent_case_key": str(problem.tangent_case_key),
        "tangent_case_probabilities": dict(problem.tangent_case_probabilities),
        "larger_circle_side": str(problem.larger_circle_side),
        "larger_circle_side_probabilities": dict(problem.larger_side_probabilities),
        "tangent_side": str(problem.tangent_side),
        "tangent_side_probabilities": dict(problem.tangent_side_probabilities),
        "answer_support_probabilities": dict(problem.answer_support_probabilities),
        "answer_value": int(problem.answer),
    }
    payload = pair_tangent_trace_base(
        prepared=prepared,
        branch_name=str(branch_name),
        branch_params=branch_params,
        unknown_role=str(problem.unknown_role),
        center_distance=int(problem.center_distance),
        radius_difference=int(problem.radius_difference),
        tangent_length=int(problem.tangent_length),
    )
    payload["execution_trace"] = {
        "scene_id": SCENE_ID,
        "query_id": str(branch_name),
        "radius_o1": int(problem.radius_o1),
        "radius_o2": int(problem.radius_o2),
        "center_distance": int(problem.center_distance),
        "radius_difference": int(problem.radius_difference),
        "tangent_length": int(problem.tangent_length),
        "answer": int(problem.answer),
        "tangent_side": str(problem.tangent_side),
        "larger_circle_side": str(problem.larger_circle_side),
        "formula_family": str(problem.formula_family),
        "formula": str(problem.formula),
        "unknown_role": str(problem.unknown_role),
        "annotation_roles": list(prepared.rendered.annotation_roles),
    }
    payload["witness_symbolic"] = {
        "scene_id": SCENE_ID,
        "query_id": str(branch_name),
        "formula_family": str(problem.formula_family),
        "unknown_role": str(problem.unknown_role),
        "radius_o1": int(problem.radius_o1),
        "radius_o2": int(problem.radius_o2),
        "center_distance": int(problem.center_distance),
        "radius_difference": int(problem.radius_difference),
        "tangent_length": int(problem.tangent_length),
        "answer_value": int(problem.answer),
    }
    return payload


def build_pair_tangent_value_problem(
    *,
    layout: TangentLayout,
    answer: int,
    center_segment_label: str,
    tangent_segment_label: str,
    formula_family: str,
    formula: str,
    unknown_role: str,
    answer_metric: str,
) -> PairTangentValueProblem:
    """Build the shared problem dataclass from public task-bound objective facts."""

    return PairTangentValueProblem(
        diagram_spec=PairTangentDiagramSpec(
            radius_o1=int(layout.radius_o1),
            radius_o2=int(layout.radius_o2),
            center_distance=int(layout.case.center_distance),
            tangent_length=int(layout.case.tangent_length),
            larger_circle_side=str(layout.larger_circle_side),
            tangent_side=str(layout.tangent_side),
            center_segment_label=str(center_segment_label),
            tangent_segment_label=str(tangent_segment_label),
        ),
        answer=int(answer),
        radius_o1=int(layout.radius_o1),
        radius_o2=int(layout.radius_o2),
        center_distance=int(layout.case.center_distance),
        tangent_length=int(layout.case.tangent_length),
        radius_difference=abs(int(layout.radius_o2) - int(layout.radius_o1)),
        larger_circle_side=str(layout.larger_circle_side),
        tangent_side=str(layout.tangent_side),
        formula_family=str(formula_family),
        formula=str(formula),
        unknown_role=str(unknown_role),
        tangent_case_key=str(layout.case.key),
        tangent_case_probabilities=dict(layout.tangent_case_probabilities),
        larger_side_probabilities=dict(layout.larger_side_probabilities),
        tangent_side_probabilities=dict(layout.tangent_side_probabilities),
        answer_support_probabilities=tangent_answer_support(selected=int(answer), metric=str(answer_metric)),
    )


__all__ = [
    "PairTangentValueProblem",
    "PairTangentPreparedScene",
    "build_pair_tangent_value_problem",
    "pair_tangent_trace_base",
    "pair_tangent_trace_payload",
    "prepare_pair_tangent_scene",
]
