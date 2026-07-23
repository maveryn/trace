"""Shared helpers for mixed-object illustration tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...shared.config_defaults import group_default
from .mixed_object_scene import (
    choose_background_id,
    render_mixed_object_scene,
    resolve_content_bbox,
    sample_background_layout,
    sample_placements,
    scene_entities,
)
from .object_library import STYLE_IDS, display_name_for_object_type, object_types
from .object_rendering import serialize_rendered_illustration_object
from .style_registry import resolve_art_style_weights


MIXED_QUERY_OBJECT_TYPES: Tuple[str, ...] = tuple(
    object_type
    for object_type in object_types()
    if object_type not in {"cloud", "sun", "quadruped"}
)


def positive_style_weights(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> Dict[str, float]:
    return resolve_art_style_weights(params, render_defaults, style_ids=STYLE_IDS)


def positive_background_weights(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> Dict[str, float]:
    raw = params.get(
        "background_weights",
        group_default(
            render_defaults,
            "background_weights",
            {"studio": 1.0, "meadow": 1.0, "sky_ground": 1.0, "tabletop": 1.0, "paper": 1.0, "shelf": 1.0},
        ),
    )
    if not isinstance(raw, Mapping):
        raise ValueError("background_weights must be a mapping")
    return {str(key): float(value) for key, value in raw.items()}


def mixed_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "canvas_width": int(params.get("canvas_width", group_default(render_defaults, "canvas_width", int(fallback["canvas_width"])))),
        "canvas_height": int(params.get("canvas_height", group_default(render_defaults, "canvas_height", int(fallback["canvas_height"])))),
        "outer_margin_px": int(params.get("outer_margin_px", group_default(render_defaults, "outer_margin_px", int(fallback["outer_margin_px"])))),
        "object_size_min_px": int(
            params.get("object_size_min_px", group_default(render_defaults, "object_size_min_px", int(fallback["object_size_min_px"])))
        ),
        "object_size_max_px": int(
            params.get("object_size_max_px", group_default(render_defaults, "object_size_max_px", int(fallback["object_size_max_px"])))
        ),
        "object_min_gap_px": int(params.get("object_min_gap_px", group_default(render_defaults, "object_min_gap_px", int(fallback["object_min_gap_px"])))),
        "max_overlap_fraction": float(
            params.get("max_overlap_fraction", group_default(render_defaults, "max_overlap_fraction", float(fallback["max_overlap_fraction"])))
        ),
        "placement_max_attempts": int(
            params.get("placement_max_attempts", group_default(render_defaults, "placement_max_attempts", int(fallback["placement_max_attempts"])))
        ),
        "render_scale": int(params.get("render_scale", group_default(render_defaults, "render_scale", int(fallback["render_scale"])))),
    }


def render_mixed_scene_from_types(
    *,
    task_id: str,
    instance_seed: int,
    attempt_index: int,
    object_types_for_scene: Sequence[str],
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback: Mapping[str, Any],
):
    render_params = mixed_render_params(params, render_defaults, fallback=fallback)
    content_bbox = resolve_content_bbox(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        margin_px=int(render_params["outer_margin_px"]),
    )
    scene_rng = spawn_rng(int(instance_seed), f"{task_id}:scene", int(attempt_index))
    background_weights = positive_background_weights(params, render_defaults)
    background_id = choose_background_id(scene_rng, background_weights, object_types=object_types_for_scene)
    background_layout = sample_background_layout(
        scene_rng,
        background_id=str(background_id),
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
    )
    placements = sample_placements(
        object_types=tuple(str(value) for value in object_types_for_scene),
        rng=scene_rng,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        content_bbox=content_bbox,
        object_size_min_px=int(render_params["object_size_min_px"]),
        object_size_max_px=int(render_params["object_size_max_px"]),
        min_gap_px=int(render_params["object_min_gap_px"]),
        max_overlap_fraction=float(render_params["max_overlap_fraction"]),
        placement_max_attempts=int(render_params["placement_max_attempts"]),
        style_weights=positive_style_weights(params, render_defaults),
        background_id=str(background_id),
        background_layout=background_layout,
    )
    return render_mixed_object_scene(
        placements=placements,
        rng=scene_rng,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        background_weights=background_weights,
        render_scale=int(render_params["render_scale"]),
        content_bbox=content_bbox,
        background_id=str(background_id),
        background_layout=background_layout,
    )


def serialize_mixed_scene(scene) -> Tuple[list[dict[str, Any]], Dict[str, list[float]], Dict[str, list[float]]]:
    serialized_objects = [serialize_rendered_illustration_object(obj) for obj in scene.objects]
    object_bboxes = {str(obj["object_id"]): list(obj["bbox"]) for obj in serialized_objects}
    part_bboxes = {
        str(part["part_id"]): list(part["bbox"])
        for obj in serialized_objects
        for part in obj["parts"]
    }
    return serialized_objects, object_bboxes, part_bboxes


def sort_object_bboxes(object_bboxes: Mapping[str, Sequence[float]], object_ids: Sequence[str]) -> list[list[float]]:
    boxes = [(str(object_id), [round(float(v), 3) for v in object_bboxes[str(object_id)]]) for object_id in object_ids]
    ordered = sorted(boxes, key=lambda item: (float(item[1][1]), float(item[1][0]), str(item[0])))
    return [box for _object_id, box in ordered]


def object_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3])))


def relation_side(candidate_bbox: Sequence[float], reference_bbox: Sequence[float]) -> Dict[str, bool]:
    cx, cy = object_center(candidate_bbox)
    rx, ry = object_center(reference_bbox)
    return {
        "left": bool(cx < rx),
        "right": bool(cx > rx),
        "above": bool(cy < ry),
        "below": bool(cy > ry),
    }


__all__ = [
    "MIXED_QUERY_OBJECT_TYPES",
    "display_name_for_object_type",
    "mixed_render_params",
    "object_center",
    "positive_background_weights",
    "positive_style_weights",
    "relation_side",
    "render_mixed_scene_from_types",
    "sample_background_layout",
    "scene_entities",
    "serialize_mixed_scene",
    "sort_object_bboxes",
]
