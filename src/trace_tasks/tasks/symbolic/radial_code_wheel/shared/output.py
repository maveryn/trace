"""Output projection helpers for radial code-wheel scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from .....core.visual.noise import apply_post_image_noise
from ....shared.output_metadata import default_task_versions
from ...shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style

from .annotations import round_bbox_map, round_point_map, witness_payload
from .defaults import POST_IMAGE_NOISE_DEFAULTS
from .state import RadialRenderParams, RenderedRadialScene
from .styles import resolve_render_params


@dataclass(frozen=True)
class RadialCanvasRuntime:
    """Base canvas, style, and background metadata for one rendered scene."""

    image: Image.Image
    style: Any
    scene_style_metadata: dict[str, Any]
    background_metadata: dict[str, Any]


@dataclass(frozen=True)
class RadialRenderedRuntime:
    """Rounded projection maps and stable render payloads for one radial render."""

    image: Image.Image
    item_bboxes: dict[str, list[float]]
    item_points: dict[str, list[float]]
    entity_records: list[dict[str, Any]]
    render_spec: dict[str, Any]
    render_map: dict[str, Any]
    task_versions: dict[str, str]


@dataclass(frozen=True)
class RadialOutputArtifacts:
    """Final image plus rounded projection data for one radial render."""

    image: Image.Image
    projection: RadialRenderedRuntime


def make_radial_canvas(*, instance_seed: int, namespace: str, canvas_width: int, canvas_height: int) -> RadialCanvasRuntime:
    """Prepare the reusable styled canvas before objective-specific drawing."""

    scene_style, scene_style_metadata = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    image, background_metadata = make_symbolic_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=scene_style,
    )
    return RadialCanvasRuntime(
        image=image,
        style=scene_style,
        scene_style_metadata=dict(scene_style_metadata),
        background_metadata=dict(background_metadata),
    )


def apply_radial_post_noise(image: Image.Image, *, instance_seed: int, params: Mapping[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Apply the scene's configured post-image noise profile."""

    return apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )


def project_radial_render(
    rendered_scene: RenderedRadialScene,
    *,
    render_params: RadialRenderParams,
    canvas: RadialCanvasRuntime,
    post_noise_metadata: Mapping[str, Any],
    annotation_source: str,
) -> RadialRenderedRuntime:
    """Serialize scene-level radial render maps after final placement."""

    item_bboxes = round_bbox_map(rendered_scene.item_bboxes)
    item_points = round_point_map(rendered_scene.item_points)
    render_spec = {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_style": dict(canvas.scene_style_metadata),
        "radial_code_wheel_style": dict(rendered_scene.style_metadata),
        "background_style": dict(canvas.background_metadata),
        "post_image_noise": dict(post_noise_metadata),
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "render_params": {
            "wheel_center_x_px": int(render_params.wheel_center_x_px),
            "wheel_center_y_px": int(render_params.wheel_center_y_px),
            "ring_width_px": int(render_params.ring_width_px),
            "terminal_label_radius_px": int(render_params.terminal_label_radius_px),
        },
    }
    render_map = {
        "image_id": "img0",
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "item_bboxes_px": dict(item_bboxes),
        "item_points_px": dict(item_points),
        "annotation_source": str(annotation_source),
    }
    return RadialRenderedRuntime(
        image=rendered_scene.image,
        item_bboxes=dict(item_bboxes),
        item_points=dict(item_points),
        entity_records=[dict(entity) for entity in rendered_scene.entities],
        render_spec=render_spec,
        render_map=render_map,
        task_versions=default_task_versions(),
    )


def radial_entity_trace_records(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact radial records for task-owned execution traces."""

    records: list[dict[str, Any]] = []
    for entity in entities:
        record = {
            "item_id": str(entity["item_id"]),
            "entity_type": str(entity.get("entity_type", "")),
            "role": str(entity.get("role", "")),
        }
        for key in ("title", "value", "label", "code", "output_label", "terminal_index"):
            if key in entity:
                record[str(key)] = entity[key]
        records.append(record)
    return records


def draw_radial_scene_artifacts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
    render_scene: Callable[..., RenderedRadialScene],
    render_kwargs: Mapping[str, Any],
    annotation_source: str,
) -> RadialOutputArtifacts:
    """Run the shared radial render pipeline around a caller-chosen layout."""

    render_params = resolve_render_params(params, render_defaults)
    canvas = make_radial_canvas(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
    )
    rendered_scene = render_scene(canvas.image, params=render_params, style=canvas.style, **dict(render_kwargs))
    image, post_noise_metadata = apply_radial_post_noise(rendered_scene.image, instance_seed=int(instance_seed), params=params)
    projection = project_radial_render(
        rendered_scene,
        render_params=render_params,
        canvas=canvas,
        post_noise_metadata=post_noise_metadata,
        annotation_source=str(annotation_source),
    )
    return RadialOutputArtifacts(image=image, projection=projection)


def compose_radial_trace_payload(
    *,
    scene_id: str,
    scene_variant: str,
    projection: RadialRenderedRuntime,
    prompt_query_spec: Mapping[str, Any],
    prompt_bundle_id: str,
    relations: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    annotation_artifacts: Any,
    answer_gt: Any,
) -> dict[str, Any]:
    """Compose reusable radial trace sections from task-owned values."""

    return {
        "scene_ir": {
            "scene_kind": str(scene_id),
            "entities": list(projection.entity_records),
            "relations": dict(relations),
        },
        "query_spec": {
            **dict(prompt_query_spec),
            "template_id": str(prompt_bundle_id),
        },
        "render_spec": {
            "scene_id": str(scene_id),
            "scene_variant": str(scene_variant),
            **dict(projection.render_spec),
        },
        "render_map": dict(projection.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": witness_payload(annotation_artifacts),
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_artifacts.annotation_gt.to_dict(),
    }


__all__ = [
    "RadialCanvasRuntime",
    "RadialOutputArtifacts",
    "RadialRenderedRuntime",
    "apply_radial_post_noise",
    "compose_radial_trace_payload",
    "draw_radial_scene_artifacts",
    "make_radial_canvas",
    "project_radial_render",
    "radial_entity_trace_records",
]
