"""Trace fragment helpers for isometric harbor public tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .spatial_primitives import rounded_bbox
from .state import IsoHarborScene


def bbox_set_projection(bboxes: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return the unordered bbox-set projection payload."""

    values = [rounded_bbox(bbox) for bbox in bboxes]
    return {"type": "bbox_set", "bbox_set": values, "pixel_bbox_set": values}


def bbox_projection(bbox: Sequence[float]) -> dict[str, Any]:
    """Return the scalar bbox projection payload."""

    value = rounded_bbox(bbox)
    return {"type": "bbox", "bbox": value, "pixel_bbox": value, "value": value}


def isometric_harbor_scene_ir(
    *,
    domain: str,
    scene_id: str,
    scene: IsoHarborScene,
    relations: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the common scene-IR fragment for one isometric harbor scene."""

    return {
        "domain": str(domain),
        "scene_id": str(scene_id),
        "tiles": [tile.as_dict() for tile in scene.tiles],
        "entities": [entity.as_dict() for entity in scene.entities],
        "relations": dict(relations),
    }


def isometric_harbor_render_spec(scene: IsoHarborScene, *, scene_id: str) -> dict[str, Any]:
    """Return the render-spec fragment for one harbor scene."""

    projection = dict(scene.trace.get("projection", {})) if isinstance(scene.trace, Mapping) else {}
    return {
        "canvas_size": [int(scene.image.size[0]), int(scene.image.size[1])],
        "coord_space": "pixel",
        "scene_id": str(scene_id),
        "style": {
            "renderer_id": str(scene.trace.get("renderer_id", "")),
            "style_id": str(scene.trace.get("renderer_style", "")),
            "theme_id": str(scene.trace.get("theme_id", "")),
            "canvas_profile": str(scene.trace.get("canvas_profile", "")),
            "canvas_profile_probabilities": dict(scene.trace.get("canvas_profile_probabilities", {})),
            "projection": projection,
            "tile_count": int(scene.trace.get("tile_count", 0)),
            "background_rgb": list(scene.trace.get("background_rgb", [])),
            "background_tone_id": str(scene.trace.get("background_tone_id", "")),
            "background_tone_rgb": list(scene.trace.get("background_tone_rgb", [])),
            "background_tone_family": str(scene.trace.get("background_tone_family", "")),
            "terrain_tile_counts": dict(scene.trace.get("terrain_tile_counts", {})),
            "entity_count": int(scene.trace.get("entity_count", 0)),
        },
    }


def isometric_harbor_boat_count_render_map(
    *,
    scene: IsoHarborScene,
    target_side: str,
    counted_entity_ids: Sequence[str],
) -> dict[str, Any]:
    """Return task render-map fields for boat-side counting."""

    entities_by_id = {str(entity.entity_id): entity for entity in scene.entities}
    counted_ids = [str(entity_id) for entity_id in counted_entity_ids]
    counted_bboxes = [rounded_bbox(entities_by_id[entity_id].bbox_xyxy) for entity_id in counted_ids if entity_id in entities_by_id]
    boat_sides = {
        str(entity.entity_id): str(entity.metadata.get("dock_side", ""))
        for entity in scene.entities
        if str(entity.object_type) == "boat"
    }
    return {
        "image_id": "img0",
        "target_side": str(target_side),
        "counted_entity_ids": counted_ids,
        "counted_entity_bboxes_px": counted_bboxes,
        "boat_sides_by_id": boat_sides,
        "boat_bboxes_px_by_id": {
            str(entity.entity_id): rounded_bbox(entity.bbox_xyxy)
            for entity in scene.entities
            if str(entity.object_type) == "boat"
        },
        "boat_counts_by_side": dict(scene.trace.get("boat_counts_by_side", {})),
        "answer_count": len(counted_ids),
    }


def isometric_harbor_mooring_status_count_render_map(
    *,
    scene: IsoHarborScene,
    target_status: str,
    counted_entity_ids: Sequence[str],
) -> dict[str, Any]:
    """Return task render-map fields for boat mooring-status counting."""

    entities_by_id = {str(entity.entity_id): entity for entity in scene.entities}
    counted_ids = [str(entity_id) for entity_id in counted_entity_ids]
    counted_bboxes = [rounded_bbox(entities_by_id[entity_id].bbox_xyxy) for entity_id in counted_ids if entity_id in entities_by_id]
    boat_statuses = {
        str(entity.entity_id): str(entity.metadata.get("mooring_status", ""))
        for entity in scene.entities
        if str(entity.object_type) == "boat"
    }
    return {
        "image_id": "img0",
        "target_mooring_status": str(target_status),
        "counted_entity_ids": counted_ids,
        "counted_entity_bboxes_px": counted_bboxes,
        "boat_mooring_status_by_id": boat_statuses,
        "boat_bboxes_px_by_id": {
            str(entity.entity_id): rounded_bbox(entity.bbox_xyxy)
            for entity in scene.entities
            if str(entity.object_type) == "boat"
        },
        "boat_counts_by_mooring_status": dict(scene.trace.get("boat_counts_by_mooring_status", {})),
        "answer_count": len(counted_ids),
    }


def isometric_harbor_heading_status_count_render_map(
    *,
    scene: IsoHarborScene,
    target_status: str,
    counted_entity_ids: Sequence[str],
) -> dict[str, Any]:
    """Return task render-map fields for boat heading-status counting."""

    entities_by_id = {str(entity.entity_id): entity for entity in scene.entities}
    counted_ids = [str(entity_id) for entity_id in counted_entity_ids]
    counted_bboxes = [rounded_bbox(entities_by_id[entity_id].bbox_xyxy) for entity_id in counted_ids if entity_id in entities_by_id]
    boat_statuses = {
        str(entity.entity_id): str(entity.metadata.get("heading_status", ""))
        for entity in scene.entities
        if str(entity.object_type) == "boat"
    }
    return {
        "image_id": "img0",
        "target_heading_status": str(target_status),
        "counted_entity_ids": counted_ids,
        "counted_entity_bboxes_px": counted_bboxes,
        "boat_heading_status_by_id": boat_statuses,
        "boat_bboxes_px_by_id": {
            str(entity.entity_id): rounded_bbox(entity.bbox_xyxy)
            for entity in scene.entities
            if str(entity.object_type) == "boat"
        },
        "boat_counts_by_heading_status": dict(scene.trace.get("boat_counts_by_heading_status", {})),
        "answer_count": len(counted_ids),
    }


def isometric_harbor_shoreline_nearest_boat_render_map(
    *,
    scene: IsoHarborScene,
    candidate_boat_ids_by_label: Mapping[str, str],
    selected_label: str,
) -> dict[str, Any]:
    """Return task render-map fields for shoreline-nearest boat selection."""

    entities_by_id = {str(entity.entity_id): entity for entity in scene.entities}
    selected_boat_id = str(candidate_boat_ids_by_label[str(selected_label)])
    selected_boat = entities_by_id[selected_boat_id]
    boat_bboxes = {
        str(label): rounded_bbox(entities_by_id[str(entity_id)].bbox_xyxy)
        for label, entity_id in candidate_boat_ids_by_label.items()
        if str(entity_id) in entities_by_id
    }
    label_bboxes = {
        str(label): list(scene.trace.get("candidate_label_bboxes_px_by_label", {}).get(str(label), []))
        for label in candidate_boat_ids_by_label
    }
    return {
        "image_id": "img0",
        "candidate_boat_ids_by_label": dict(candidate_boat_ids_by_label),
        "candidate_boat_bboxes_px_by_label": boat_bboxes,
        "candidate_label_bboxes_px_by_label": label_bboxes,
        "shoreline_distance_tiles_by_label": dict(scene.trace.get("shoreline_distance_tiles_by_label", {})),
        "shoreline_bow_points_px_by_label": dict(scene.trace.get("shoreline_bow_points_px_by_label", {})),
        "selected_label": str(selected_label),
        "selected_boat_id": selected_boat_id,
        "selected_boat_bbox_px": rounded_bbox(selected_boat.bbox_xyxy),
        "selected_shoreline_distance_tiles": int(selected_boat.metadata.get("shoreline_distance_tiles", 0)),
        "shoreline_reference": str(scene.trace.get("shoreline_reference", "")),
    }


__all__ = [
    "bbox_projection",
    "bbox_set_projection",
    "isometric_harbor_boat_count_render_map",
    "isometric_harbor_heading_status_count_render_map",
    "isometric_harbor_mooring_status_count_render_map",
    "isometric_harbor_shoreline_nearest_boat_render_map",
    "isometric_harbor_render_spec",
    "isometric_harbor_scene_ir",
]
