"""Neutral lifecycle plumbing for polygon equation diagram tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.annotation_values import keyed_point_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_DEFAULTS, SCENE_ID, SCENE_KIND, SCENE_VARIANT
from .shared.output import render_map, trace_common
from .shared.prompts import polygon_equation_prompt_artifacts
from .shared.rendering import make_render_context, render_polygon_equation_case
from .shared.state import PolygonEquationCase, polygon_kind

BuildCase = Callable[..., PolygonEquationCase]


def run_polygon_equation_task(
    *,
    task_id: str,
    build_case: BuildCase,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run shared scene plumbing around a public task-selected equation case."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=("single",),
        default_query_id="single",
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        SCENE_DEFAULTS,
        task_id=str(task_id),
    )

    case: PolygonEquationCase | None = None
    rendered: Any | None = None
    ctx: Any | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt)
        attempt_params = {**dict(task_params), "_render_attempt": int(attempt)}
        try:
            case = build_case(
                instance_seed=int(attempt_seed),
                params=attempt_params,
                generation_defaults=generation_defaults,
            )
            ctx = make_render_context(
                int(attempt_seed),
                attempt_params,
                rendering_defaults,
            )
            rendered = render_polygon_equation_case(
                case=case,
                ctx=ctx,
                instance_seed=int(attempt_seed),
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    if case is None or rendered is None or ctx is None:
        raise RuntimeError(f"failed to generate {task_id}") from last_error

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = keyed_point_annotation_artifacts(
        rendered.annotation_keyed_points,
        roles=rendered.annotation_roles,
    )
    _prompt_defaults, prompt_artifacts = polygon_equation_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_defaults["task_key"]),
        annotation_keys=tuple(annotation_artifacts.value.keys()),
        target_name=str(case.target_name),
        variable_name=str(case.variable_name),
        shape_name=polygon_kind(int(case.side_count)),
        answer=int(case.answer),
        instance_seed=int(instance_seed),
    )
    common = trace_common(
        case=case,
        rendered=rendered,
        annotation_artifacts=annotation_artifacts,
    )
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "query_id_probabilities": dict(query_probabilities),
        **dict(common["query_params"]),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID
    trace_payload: dict[str, Any] = {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "task_id": str(task_id),
            "query_id": str(selected_query),
            "entities": common["entities"],
            "relations": common["relations"],
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "canvas": {"width": int(ctx.width), "height": int(ctx.height)},
            "style": {
                "technical_diagram": dict(ctx.diagram_style_meta),
                "background": dict(ctx.background_meta),
                "post_image_noise": dict(noise_meta),
            },
            "single_object_scene_rotation": ctx.scene_transform.metadata(),
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": render_map(rendered),
        "execution_trace": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            **dict(common["execution"]),
        },
        "witness_symbolic": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            **dict(common["witness"]),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(case.answer)),
        annotation_gt=TypedValue(
            type=annotation_artifacts.annotation_type,
            value=dict(annotation_artifacts.value),
        ),
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = ["run_polygon_equation_task"]
