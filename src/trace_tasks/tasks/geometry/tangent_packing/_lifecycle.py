"""Neutral lifecycle plumbing for tangent-packing public task files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import tangent_packing_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, load_tangent_packing_defaults
from .shared.prompts import build_tangent_packing_prompt_artifacts, tangent_packing_object_description
from .shared.rendering import create_render_context
from .shared.state import SCENE_ID, SCENE_KIND, RenderContext, RenderedTangentPackingScene, TangentPackingProblem

RenderBuilder = Callable[[RenderContext, TangentPackingProblem], RenderedTangentPackingScene]


def _public_answer_value(answer_value: float | int, answer_type: str) -> float | int:
    if str(answer_type) == "integer":
        return int(round(float(answer_value)))
    return float(answer_value)


@dataclass(frozen=True)
class TangentPackingRenderedAttempt:
    """Rendered image and annotation artifacts returned by one task."""

    image: Image.Image
    rendered: RenderedTangentPackingScene
    render_meta: Mapping[str, Any]
    noise_meta: Mapping[str, Any]
    annotation_artifacts: PixelAnnotationArtifacts


def render_tangent_packing_attempts(
    *,
    problem: TangentPackingProblem,
    render_scene: RenderBuilder,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> TangentPackingRenderedAttempt:
    """Retry neutral rendering after a public task has bound its objective."""

    last_error: Exception | None = None
    for _attempt in range(max(1, int(max_attempts))):
        try:
            context, render_meta_attempt = create_render_context(
                instance_seed=int(instance_seed),
                params=params,
                render_defaults=render_defaults,
                namespace=str(namespace),
            )
            rendered = render_scene(context, problem)
            render_meta = dict(render_meta_attempt)
            render_meta["single_object_scene_rotation"] = context.scene_transform.metadata()
            image, noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(instance_seed),
                params=params,
                default_config=POST_IMAGE_NOISE_DEFAULTS,
            )
            return TangentPackingRenderedAttempt(
                image=image,
                rendered=rendered,
                render_meta=render_meta,
                noise_meta=dict(noise_meta),
                annotation_artifacts=tangent_packing_annotation(rendered),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render tangent-packing scene") from last_error


def build_tangent_packing_trace_payload(
    *,
    task_identity: str,
    selected_query: str,
    branch_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    attempt: TangentPackingRenderedAttempt,
    answer_value: float | int,
    answer_type: str,
    answer_rounding: str,
    reasoning_steps: int,
    query_params_extra: Mapping[str, Any],
    trace_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Build trace sections from task-owned objective metadata."""

    query_params = {
        "scene_id": SCENE_ID,
        "query_id_probabilities": dict(branch_probabilities),
        **dict(query_params_extra),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["task_id"] = str(task_identity)
    query_spec["scene_id"] = SCENE_ID
    rendered = attempt.rendered
    public_answer = _public_answer_value(answer_value, answer_type)
    return {
        "scene_ir": {
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "task_id": str(task_identity),
            "query_id": str(selected_query),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(selected_query),
                "answer_value": public_answer,
                "annotation_roles": list(rendered.annotation_roles),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "task_id": str(task_identity),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "canvas": {"width": int(attempt.image.size[0]), "height": int(attempt.image.size[1])},
            "style": {
                **dict(attempt.render_meta),
                "post_image_noise": dict(attempt.noise_meta),
            },
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_trace": {
            "task_id": str(task_identity),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            **dict(trace_values),
            "answer_type": str(answer_type),
            "answer_value": public_answer,
            "answer_rounding": str(answer_rounding),
            "annotation_roles": list(rendered.annotation_roles),
            "reasoning_steps": int(reasoning_steps),
        },
        "witness_symbolic": {
            "task_id": str(task_identity),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "type": "circle_square_tangent_packing_formula",
            "source_witness_type": "bbox",
            **dict(trace_values),
            "answer_value": public_answer,
        },
        "projected_annotation": dict(attempt.annotation_artifacts.projected_annotation),
    }


def tangent_packing_output_metadata(*, prompt_artifacts: Any, selected_query: str) -> dict[str, Any]:
    """Return neutral TaskOutput metadata that is not objective-specific."""

    return {
        "task_versions": default_task_versions(),
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "prompt_variants": dict(prompt_artifacts.prompt_variants),
    }


def run_tangent_packing_public_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run neutral lifecycle plumbing around task-owned formula hooks."""

    selected_query, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
        namespace=f"{task.task_id}.query",
    )
    _generation_defaults, render_defaults, prompt_defaults = load_tangent_packing_defaults()
    problem, answer_value, trace_values = task.prepare_objective(
        instance_seed=int(instance_seed),
        params=task_params,
        selected_query=str(selected_query),
        branch_probabilities=branch_probabilities,
    )
    attempt = render_tangent_packing_attempts(
        problem=problem,
        render_scene=task.render_scene,
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        render_defaults=render_defaults,
        namespace=str(task.task_id),
    )
    prompt_artifacts = build_tangent_packing_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        task_prompt_key=str(task.task_prompt_key),
        prompt_query_key=str(selected_query),
        annotation_roles=attempt.rendered.annotation_roles,
        answer_value=_public_answer_value(answer_value, str(problem.answer_type)),
        answer_type=str(problem.answer_type),
        object_description=tangent_packing_object_description(problem.construction_kind),
        instance_seed=int(instance_seed),
    )
    public_answer = _public_answer_value(answer_value, str(problem.answer_type))
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(problem.answer_type), value=public_answer),
        annotation_gt=TypedValue(
            type=str(attempt.annotation_artifacts.annotation_type),
            value=attempt.annotation_artifacts.value,
        ),
        image=attempt.image,
        image_id="img0",
        trace_payload=build_tangent_packing_trace_payload(
            task_identity=str(task.task_id),
            selected_query=str(selected_query),
            branch_probabilities=branch_probabilities,
            prompt_artifacts=prompt_artifacts,
            attempt=attempt,
            answer_value=public_answer,
            answer_type=str(problem.answer_type),
            answer_rounding=str(problem.answer_rounding),
            reasoning_steps=int(problem.reasoning_steps),
            query_params_extra=trace_values,
            trace_values=trace_values,
        ),
        **tangent_packing_output_metadata(
            prompt_artifacts=prompt_artifacts,
            selected_query=str(selected_query),
        ),
    )


__all__ = [
    "TangentPackingRenderedAttempt",
    "build_tangent_packing_trace_payload",
    "render_tangent_packing_attempts",
    "run_tangent_packing_public_entry",
    "tangent_packing_output_metadata",
]
