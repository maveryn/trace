"""Neutral lifecycle plumbing for triangle-relations public task files."""

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
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import triangle_relations_annotation
from .shared.construction import case_trace_values
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, load_triangle_relations_defaults
from .shared.prompts import build_triangle_relations_prompt_artifacts
from .shared.rendering import create_render_context
from .shared.state import (
    DOMAIN,
    SCENE_ID,
    SCENE_KIND,
    RenderContext,
    RenderedTriangleRelationsScene,
    TriangleRelationsProblem,
)

RenderBuilder = Callable[[RenderContext, TriangleRelationsProblem], RenderedTriangleRelationsScene]


@dataclass(frozen=True)
class TriangleRelationsRenderedAttempt:
    """Rendered image plus annotation artifacts for one attempt."""

    image: Image.Image
    rendered: RenderedTriangleRelationsScene
    render_meta: Mapping[str, Any]
    noise_meta: Mapping[str, Any]
    annotation_artifacts: PixelAnnotationArtifacts


def render_triangle_relations_attempts(
    *,
    problem: TriangleRelationsProblem,
    render_scene: RenderBuilder,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_defaults: Mapping[str, Any],
) -> TriangleRelationsRenderedAttempt:
    """Retry neutral rendering after a public task has resolved its case."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            context = create_render_context(
                instance_seed=attempt_seed,
                params={**dict(params), "_render_attempt": attempt_index},
                render_defaults=dict(render_defaults),
            )
            rendered = render_scene(context, problem)
            image, noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=attempt_seed,
                params=params,
                default_config=POST_IMAGE_NOISE_DEFAULTS,
            )
            return TriangleRelationsRenderedAttempt(
                image=image,
                rendered=rendered,
                render_meta={
                    "technical_diagram": dict(context.diagram_style_meta),
                    "background": dict(context.background_meta),
                    "single_object_scene_rotation": context.scene_transform.metadata(),
                },
                noise_meta=dict(noise_meta),
                annotation_artifacts=triangle_relations_annotation(rendered),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render triangle-relations scene") from last_error


def build_triangle_relations_trace_payload(
    *,
    task_identity: str,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    prompt_artifacts: PromptTraceArtifacts,
    attempt: TriangleRelationsRenderedAttempt,
    problem: TriangleRelationsProblem,
    answer_value: int | float,
    answer_type: str,
    answer_rounding: str,
    query_params_extra: Mapping[str, Any],
    trace_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Build trace sections from task-owned formula and witness metadata."""

    rendered = attempt.rendered
    case = problem.case
    query_params = {
        "scene_id": SCENE_ID,
        "query_id_probabilities": dict(branch_probabilities),
        **dict(query_params_extra),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=query_params,
    )
    query_spec["task_id"] = str(task_identity)
    query_spec["scene_id"] = SCENE_ID
    execution_values = {
        "task_id": str(task_identity),
        "scene_id": SCENE_ID,
        "query_id": str(selected_branch),
        "answer": answer_value,
        "answer_type": str(answer_type),
        "answer_rounding": str(answer_rounding),
        "annotation_type": str(attempt.annotation_artifacts.annotation_type),
        "annotation_roles": list(rendered.annotation_roles),
        **case_trace_values(case),
        **dict(trace_values),
    }
    return {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "task_id": str(task_identity),
            "query_id": str(selected_branch),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "formula_family": str(case.formula_family),
                "answer": answer_value,
                "annotation_type": str(attempt.annotation_artifacts.annotation_type),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "task_id": str(task_identity),
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
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
        "execution_trace": execution_values,
        "witness_symbolic": {
            "task_id": str(task_identity),
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            "type": "triangle_relations_formula",
            "source_witness_type": str(attempt.annotation_artifacts.annotation_type),
            **execution_values,
        },
        "projected_annotation": dict(attempt.annotation_artifacts.projected_annotation),
    }


def triangle_relations_output_metadata(*, prompt_artifacts: PromptTraceArtifacts, selected_branch: str) -> dict[str, Any]:
    """Return neutral TaskOutput metadata that is not objective-specific."""

    return {
        "task_versions": default_task_versions(),
        "scene_id": SCENE_ID,
        "query_id": str(selected_branch),
        "prompt_variants": dict(prompt_artifacts.prompt_variants),
    }


def run_triangle_relations_public_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run neutral lifecycle plumbing around task-owned formula hooks."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
        namespace=f"{task.task_id}.query",
    )
    _generation_defaults, render_defaults, prompt_defaults = load_triangle_relations_defaults()
    problem, answer_value, trace_values = task.prepare_objective(
        instance_seed=int(instance_seed),
        params=task_params,
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
    )
    answer_type = str(problem.case.answer_type)
    answer_rounding = str(problem.case.answer_rounding)
    attempt = render_triangle_relations_attempts(
        problem=problem,
        render_scene=task.render_scene,
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        render_defaults=render_defaults,
    )
    prompt_artifacts = build_triangle_relations_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        task_prompt_key=str(task.task_prompt_key),
        prompt_branch_key=str(selected_branch),
        annotation_mode=str(problem.annotation_mode),
        annotation_roles=tuple(attempt.rendered.annotation_roles),
        answer_value=answer_value,
        target_name=str(problem.prompt_target),
        instance_seed=int(instance_seed),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type=answer_type, value=answer_value),
        annotation_gt=TypedValue(
            type=str(attempt.annotation_artifacts.annotation_type),
            value=attempt.annotation_artifacts.value,
        ),
        image=attempt.image,
        image_id="img0",
        trace_payload=build_triangle_relations_trace_payload(
            task_identity=str(task.task_id),
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            prompt_artifacts=prompt_artifacts,
            attempt=attempt,
            problem=problem,
            answer_value=answer_value,
            answer_type=answer_type,
            answer_rounding=answer_rounding,
            query_params_extra=trace_values,
            trace_values=trace_values,
        ),
        **triangle_relations_output_metadata(
            prompt_artifacts=prompt_artifacts,
            selected_branch=str(selected_branch),
        ),
    )


__all__ = [
    "TriangleRelationsRenderedAttempt",
    "build_triangle_relations_trace_payload",
    "render_triangle_relations_attempts",
    "run_triangle_relations_public_entry",
    "triangle_relations_output_metadata",
]
