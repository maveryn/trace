"""Neutral output assembly for composite-shape scene-package tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .shared.annotations import composite_shape_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .shared.prompts import composite_shape_prompt_artifacts
from .shared.rendering import create_composite_render_context, render_composite_shape
from .shared.state import CompositeShapeProblem, RenderedCompositeShape


def _json_safe(value: Any) -> Any:
    """Convert scene-internal tuples to JSON-safe values before trace export."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class PreparedCompositeShapeScene:
    """Rendered image plus prompt and annotation artifacts."""

    rendered: RenderedCompositeShape
    image: Any
    noise_meta: Dict[str, Any]
    render_meta: Dict[str, Any]
    prompt_artifacts: Any
    annotation_artifacts: Any


def prepare_composite_shape_scene(
    *,
    problem: CompositeShapeProblem,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_namespace: str,
) -> PreparedCompositeShapeScene:
    """Render and bind prompt/annotation artifacts for an objective-owned problem."""

    rendered: RenderedCompositeShape | None = None
    render_meta: Dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_params = dict(params)
        attempt_params["_render_attempt"] = int(attempt)
        try:
            ctx, attempt_meta = create_composite_render_context(
                instance_seed=int(instance_seed) + int(attempt),
                params=attempt_params,
                render_defaults=render_defaults,
                render_namespace=str(render_namespace),
            )
            rendered = render_composite_shape(ctx, problem)
            render_meta = dict(attempt_meta)
            if ctx.scene_transform is not None and ctx.scene_transform.resolved:
                render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None or render_meta is None:
        raise RuntimeError("failed to render composite shape scene") from last_error

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = composite_shape_annotation(rendered)
    _prompt_defaults, prompt_artifacts = composite_shape_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(problem.prompt_key),
        annotation_type=str(annotation_artifacts.annotation_type),
        annotation_keys=tuple(rendered.annotation_roles),
        answer_value=problem.answer_value,
        prompt_slots=problem.prompt_slots,
        instance_seed=int(instance_seed),
    )
    return PreparedCompositeShapeScene(
        rendered=rendered,
        image=image,
        noise_meta=dict(noise_meta),
        render_meta=dict(render_meta),
        prompt_artifacts=prompt_artifacts,
        annotation_artifacts=annotation_artifacts,
    )


def composite_shape_trace_payload(
    *,
    task_id: str,
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    problem: CompositeShapeProblem,
    prepared: PreparedCompositeShapeScene,
) -> Dict[str, Any]:
    """Assemble trace metadata from task-owned answer and annotation facts."""

    rendered = prepared.rendered
    prompt_metadata_fields: Dict[str, Any] = {
        "scene_id": SCENE_ID,
        "query_id": str(branch_name),
        "query_id_probabilities": _json_safe(dict(branch_probabilities)),
        "answer_value": problem.answer_value,
        **_json_safe(dict(problem.metadata_fields)),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prepared.prompt_artifacts,
        query_id=str(branch_name),
        params=prompt_metadata_fields,
    )
    prompt_query_spec["scene_id"] = SCENE_ID
    prompt_query_spec["task_id"] = str(task_id)
    return {
        "scene_ir": {
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(branch_name),
            "scene_kind": str(problem.scene_kind),
            "entities": [_json_safe(dict(entity)) for entity in rendered.scene_entities],
            "relations": {
                "shape_family": str(problem.shape_family),
                "metric_kind": str(problem.metric_kind),
                "answer_value": problem.answer_value,
                "annotation_roles": list(rendered.annotation_roles),
            },
        },
        "query_spec": prompt_query_spec,
        "render_spec": {
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(branch_name),
            "canvas_size": [int(prepared.image.size[0]), int(prepared.image.size[1])],
            "coord_space": "pixel",
            "post_image_noise": _json_safe(dict(prepared.noise_meta)),
            **_json_safe(dict(prepared.render_meta)),
            "prompt": {
                "prompt_variant": _json_safe(dict(prepared.prompt_artifacts.prompt_variant)),
                "prompt_variant_active_key": str(prepared.prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": _json_safe(dict(prepared.prompt_artifacts.prompt_variants_for_trace)),
            },
        },
        "render_map": _json_safe(dict(rendered.render_map)),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(branch_name),
            "query_id_probabilities": _json_safe(dict(branch_probabilities)),
            "answer_type": str(problem.answer_type),
            "answer_value": problem.answer_value,
            "reasoning_kind": str(problem.reasoning_kind),
            "reasoning_steps": int(problem.reasoning_steps),
            "annotation_roles": list(rendered.annotation_roles),
            **_json_safe(dict(problem.execution_fields)),
            **_json_safe(dict(rendered.witness)),
        },
        "witness_symbolic": {
            "type": str(problem.witness_type),
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(branch_name),
            "answer_value": problem.answer_value,
            "source_witness_type": str(prepared.annotation_artifacts.annotation_type),
            "original_annotation_value": list(rendered.annotation_roles),
            **_json_safe(dict(rendered.witness)),
        },
        "projected_annotation": _json_safe(dict(prepared.annotation_artifacts.projected_annotation)),
    }


def complete_composite_shape_task(
    *,
    task_id: str,
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    problem: CompositeShapeProblem,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_namespace: str,
):
    """Complete rendering, prompt construction, trace binding, and TaskOutput."""

    scene_defaults = get_scene_defaults("geometry", SCENE_ID)
    _gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults,
        task_id=str(task_id),
    )
    prepared = prepare_composite_shape_scene(
        problem=problem,
        prompt_defaults=prompt_defaults,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        render_namespace=str(render_namespace),
    )
    trace_payload = composite_shape_trace_payload(
        task_id=str(task_id),
        branch_name=str(branch_name),
        branch_probabilities=branch_probabilities,
        problem=problem,
        prepared=prepared,
    )
    answer_value = problem.answer_value
    if str(problem.answer_type) == "integer":
        answer_gt = TypedValue(type="integer", value=int(answer_value))
    else:
        answer_gt = TypedValue(type="number", value=float(answer_value))
    return TaskOutput(
        prompt=str(prepared.prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=TypedValue(
            type=str(prepared.annotation_artifacts.annotation_type),
            value=prepared.annotation_artifacts.value,
        ),
        image=prepared.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(branch_name),
        prompt_variants=dict(prepared.prompt_artifacts.prompt_variants),
    )


def run_composite_shape_public_entry(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    resolve_problem: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
):
    """Run neutral query/render plumbing around a public file's objective hook."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(query_id) for query_id in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
    )
    problem = resolve_problem(
        selected_query=str(selected_query),
        instance_seed=int(instance_seed),
        params=task_params,
    )
    return complete_composite_shape_task(
        task_id=str(task_id),
        branch_name=str(selected_query),
        branch_probabilities=query_probabilities,
        problem=problem,
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        render_namespace=f"{task_id}.{selected_query}",
    )
