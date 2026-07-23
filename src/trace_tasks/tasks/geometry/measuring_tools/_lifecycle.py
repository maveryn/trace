"""Scene-private trace helpers for measuring-tool tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import (
    measuring_tool_point_annotation,
    measuring_tool_segment_annotation,
)
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_DEFAULTS
from .shared.defaults import SCENE_ID, SCENE_KIND, SCENE_VARIANT
from .shared.prompts import measuring_tool_prompt_artifacts
from .shared.rendering import make_context
from .shared.state import RenderContext, RenderedToolScene


def measuring_tools_trace_payload(
    *,
    task_id: str,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    rendered: RenderedToolScene,
    annotation_artifacts: PixelAnnotationArtifacts,
    prompt_artifacts: PromptTraceArtifacts,
    answer_value: int,
    answer_probabilities: Mapping[str, float],
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    reasoning_steps: int,
) -> dict[str, Any]:
    """Build trace sections after a public task has bound answer and annotation."""

    if isinstance(annotation_artifacts.value, Mapping):
        annotation_roles = [str(role) for role in annotation_artifacts.value.keys()]
    elif str(annotation_artifacts.annotation_type) == "segment":
        annotation_roles = ["measured_segment"]
    else:
        annotation_roles = [str(annotation_artifacts.annotation_type)]
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "query_id_probabilities": dict(query_probabilities),
        "target_support_probabilities": dict(answer_probabilities),
        **dict(rendered.witness),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID
    return {
        "scene_ir": {
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(selected_query),
                "scene_variant": SCENE_VARIANT,
                "answer_value": int(answer_value),
                "annotation_roles": list(annotation_roles),
                **dict(rendered.witness),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(noise_meta),
            **dict(render_meta),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "answer_type": "integer",
            "answer_value": int(answer_value),
            "annotation_roles": list(annotation_roles),
            "reasoning_steps": int(reasoning_steps),
            **dict(rendered.witness),
        },
        "witness_symbolic": {
            "type": "measuring_tool_on_shape_readout",
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "answer_value": int(answer_value),
            "source_witness_type": annotation_artifacts.annotation_type,
            "original_annotation_value": annotation_artifacts.value,
            **dict(rendered.witness),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }


def run_measuring_tool_task(
    *,
    task_id: str,
    supported_queries: tuple[str, ...],
    build_plan: Callable[[int, Mapping[str, Any], Mapping[str, Any]], Any],
    render_measurement: Callable[
        [RenderContext, Any, int, Mapping[str, Any], Mapping[str, Any]],
        RenderedToolScene,
    ],
    prompt_task_key: str,
    object_description: str,
    annotation_type: str,
    annotation_keys: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run neutral scene plumbing after a public task binds measurement semantics."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_queries),
        default_query_id="single",
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    (
        gen_defaults,
        render_defaults,
        prompt_defaults,
    ) = split_scene_generation_rendering_prompt_defaults(
        SCENE_DEFAULTS,
        task_id=str(task_id),
    )
    plan = build_plan(int(instance_seed), task_params, gen_defaults)
    rendered: RenderedToolScene | None = None
    render_meta: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            ctx, render_meta_attempt = make_context(
                instance_seed=int(instance_seed) + int(attempt),
                params=task_params,
                render_defaults=render_defaults,
            )
            rendered = render_measurement(
                ctx,
                plan,
                int(instance_seed) + int(attempt),
                task_params,
                render_defaults,
            )
            render_meta = dict(render_meta_attempt)
            break
        except Exception as exc:
            last_error = exc
    if rendered is None or render_meta is None:
        raise RuntimeError(f"failed to render {task_id}") from last_error

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    if str(annotation_type) == "segment":
        if len(annotation_keys) != 2:
            raise ValueError("segment annotation requires exactly two annotation keys")
        annotation_artifacts = measuring_tool_segment_annotation(
            rendered.annotation_points,
            start_role=str(annotation_keys[0]),
            end_role=str(annotation_keys[1]),
        )
    elif str(annotation_type) == "point_map":
        annotation_artifacts = measuring_tool_point_annotation(
            rendered.annotation_points,
            roles=annotation_keys,
        )
    else:
        raise ValueError(f"unsupported measuring-tools annotation type: {annotation_type}")
    _prompt_defaults, prompt_artifacts = measuring_tool_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        object_description=str(object_description),
        annotation_type=str(annotation_type),
        annotation_keys=tuple(str(key) for key in annotation_keys),
        answer=int(rendered.answer),
        instance_seed=int(instance_seed),
    )
    trace_payload = measuring_tools_trace_payload(
        task_id=str(task_id),
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
        rendered=rendered,
        annotation_artifacts=annotation_artifacts,
        prompt_artifacts=prompt_artifacts,
        answer_value=int(rendered.answer),
        answer_probabilities=plan.answer_probabilities,
        render_meta=render_meta,
        noise_meta=noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        reasoning_steps=1,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(rendered.answer)),
        annotation_gt=TypedValue(
            type=annotation_artifacts.annotation_type,
            value=annotation_artifacts.value,
        ),
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def run_measuring_public_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the common lifecycle from task-owned class attributes and hooks."""

    return run_measuring_tool_task(
        task_id=str(task.task_id),
        supported_queries=tuple(str(value) for value in task.supported_query_ids),
        build_plan=task.build_plan,
        render_measurement=task.render_measurement,
        prompt_task_key=str(task.prompt_task_key),
        object_description=str(task.object_description),
        annotation_type=str(task.annotation_type),
        annotation_keys=tuple(str(key) for key in task.annotation_keys),
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
    )


__all__ = ["measuring_tools_trace_payload", "run_measuring_public_entry", "run_measuring_tool_task"]
