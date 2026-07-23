"""Identity-free trace serialization helpers for bearing-route scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .state import RenderedBearingScene, SCENE_ID


def common_trace_sections(
    *,
    rendered: RenderedBearingScene,
    image_size: tuple[int, int],
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    scene_variant: str,
    answer_type: str,
    answer_value: Any,
    reasoning_steps: int,
) -> dict[str, Any]:
    """Return scene/render/execution payload sections without public routing.

    Public task files add branch metadata, query specs, and objective-specific
    symbolic witnesses after answer and annotation have been bound locally.
    """

    return {
        "scene_ir": {
            "scene_kind": "geometry_bearing_route",
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "answer_value": answer_value,
                "annotation_roles": list(rendered.annotation_roles),
            },
        },
        "render_spec": {
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(noise_meta),
            **dict(render_meta),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "answer_type": str(answer_type),
            "answer_value": answer_value,
            "annotation_roles": list(rendered.annotation_roles),
            "reasoning_steps": int(reasoning_steps),
            **dict(rendered.witness),
        },
    }


def projected_keyed_point_annotation(
    *,
    annotation_bboxes: Sequence[Sequence[float]],
    annotation_points: Sequence[Sequence[float]],
    annotation_keyed_bboxes: Mapping[str, Sequence[float]],
    annotation_keyed_points: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Return normalized keyed-point annotation payloads for review/reward."""

    return {
        "type": "point_map",
        "bbox_map": dict(annotation_keyed_bboxes),
        "pixel_bbox_map": dict(annotation_keyed_bboxes),
        "point_map": dict(annotation_keyed_points),
        "pixel_point_map": dict(annotation_keyed_points),
        "bbox_set": list(annotation_bboxes),
        "pixel_bbox_set": list(annotation_bboxes),
        "point_set": list(annotation_points),
        "pixel_point_set": list(annotation_points),
    }


__all__ = ["common_trace_sections", "projected_keyed_point_annotation"]
