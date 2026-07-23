"""Neutral render retry plumbing for bearing-route public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.core.visual.noise import apply_post_image_noise

from .shared.annotations import annotation_bbox_list, annotation_point_list, keyed_bboxes, keyed_points
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.output import common_trace_sections, projected_keyed_point_annotation
from .shared.prompts import build_bearing_route_prompt_artifacts
from .shared.rendering import make_render_context
from .shared.state import RenderedBearingScene, RouteCase, SCENE_ID


@dataclass(frozen=True)
class BearingRouteRuntime:
    """Rendered scene and render metadata for one resolved route case."""

    rendered: RenderedBearingScene
    render_meta: dict[str, Any]


@dataclass(frozen=True)
class PreparedBearingRoute:
    """Rendered image, prompt, and projected annotation primitives."""

    runtime: BearingRouteRuntime
    image: Any
    noise_meta: dict[str, Any]
    prompt_artifacts: Any
    annotation_bboxes: list[list[float]]
    annotation_points: list[list[float]]
    annotation_keyed_bboxes: dict[str, list[float]]
    annotation_keyed_points: dict[str, list[float]]


def render_bearing_route_runtime(
    *,
    route_case: RouteCase,
    scene_renderer: Callable[[Any, RouteCase], RenderedBearingScene],
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
    style_namespace: str,
) -> BearingRouteRuntime:
    """Render one resolved route case after final layout/style sampling."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            ctx, render_meta = make_render_context(
                instance_seed=int(instance_seed) + int(attempt),
                params=params,
                render_defaults=render_defaults,
                style_namespace=str(style_namespace),
            )
            return BearingRouteRuntime(
                rendered=scene_renderer(ctx, route_case),
                render_meta=dict(render_meta),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render bearing-route scene") from last_error


def prepare_bearing_route_rendering(
    *,
    route_case: RouteCase,
    scene_renderer: Callable[[Any, RouteCase], RenderedBearingScene],
    domain: str,
    prompt_defaults: Mapping[str, Any],
    prompt_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
    style_namespace: str,
) -> PreparedBearingRoute:
    """Render, noise, prompt, and annotation primitives for one route case."""

    runtime = render_bearing_route_runtime(
        route_case=route_case,
        scene_renderer=scene_renderer,
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        max_attempts=int(max_attempts),
        style_namespace=str(style_namespace),
    )
    image, noise_meta = apply_post_image_noise(
        runtime.rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    _prompt_defaults, prompt_artifacts = build_bearing_route_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(prompt_key),
        instance_seed=int(instance_seed),
    )
    annotation_bboxes = annotation_bbox_list(runtime.rendered)
    annotation_points = annotation_point_list(runtime.rendered, annotation_bboxes)
    return PreparedBearingRoute(
        runtime=runtime,
        image=image,
        noise_meta=dict(noise_meta),
        prompt_artifacts=prompt_artifacts,
        annotation_bboxes=annotation_bboxes,
        annotation_points=annotation_points,
        annotation_keyed_bboxes=keyed_bboxes(runtime.rendered),
        annotation_keyed_points=keyed_points(runtime.rendered, annotation_points),
    )


def build_bearing_route_trace_payload(
    *,
    runtime: BearingRouteRuntime,
    prompt_artifacts: Any,
    noise_meta: Mapping[str, Any],
    branch_name: str,
    branch_params: Mapping[str, Any],
    scene_variant: str,
    answer_type: str,
    answer_value: Any,
    witness_kind: str,
    annotation_bboxes: Sequence[Sequence[float]],
    annotation_points: Sequence[Sequence[float]],
    annotation_keyed_bboxes: Mapping[str, Sequence[float]],
    annotation_keyed_points: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Build trace sections for one public task-bound bearing-route result."""

    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(branch_name),
        params=dict(branch_params),
    )
    query_spec["scene_id"] = SCENE_ID
    payload = common_trace_sections(
        rendered=runtime.rendered,
        image_size=runtime.rendered.image.size,
        render_meta=runtime.render_meta,
        noise_meta=noise_meta,
        scene_variant=str(scene_variant),
        answer_type=str(answer_type),
        answer_value=answer_value,
        reasoning_steps=2,
    )
    payload["query_spec"] = query_spec
    payload["scene_ir"]["relations"]["query_id"] = str(branch_name)
    payload["execution_trace"]["query_id"] = str(branch_name)
    payload["witness_symbolic"] = {
        "type": str(witness_kind),
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "query_id": str(branch_name),
        "source_witness_type": "point_map",
        "original_annotation_value": dict(annotation_keyed_points),
        "answer_value": answer_value,
        **dict(runtime.rendered.witness),
    }
    payload["projected_annotation"] = projected_keyed_point_annotation(
        annotation_bboxes=annotation_bboxes,
        annotation_points=annotation_points,
        annotation_keyed_bboxes=annotation_keyed_bboxes,
        annotation_keyed_points=annotation_keyed_points,
    )
    return payload


__all__ = [
    "BearingRouteRuntime",
    "PreparedBearingRoute",
    "build_bearing_route_trace_payload",
    "prepare_bearing_route_rendering",
    "render_bearing_route_runtime",
]
