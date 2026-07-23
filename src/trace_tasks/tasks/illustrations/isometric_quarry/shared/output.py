"""Trace fragment helpers for isometric quarry public tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .sampling import LabelTaskSampleSpec
from .state import IsoQuarryScene, IsoQuarryTile


def rounded_bbox(bbox: Sequence[float]) -> list[float]:
    """Return one rounded bbox in final pixel coordinates."""

    return [round(float(value), 3) for value in bbox[:4]]


def bbox_projection(bbox: Sequence[float]) -> dict[str, Any]:
    """Return the scalar bbox projection payload."""

    value = rounded_bbox(bbox)
    return {"type": "bbox", "bbox": value, "pixel_bbox": value, "value": value}


def bbox_set_projection(bboxes: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return the unordered bbox-set projection payload."""

    values = [rounded_bbox(bbox) for bbox in bboxes]
    return {"type": "bbox_set", "bbox_set": values, "pixel_bbox_set": values}


def isometric_quarry_scene_ir(
    *,
    domain: str,
    scene_id: str,
    scene: IsoQuarryScene,
    relations: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the common scene-IR fragment for one isometric quarry scene."""

    return {
        "domain": str(domain),
        "scene_id": str(scene_id),
        "tiles": [tile.as_dict() for tile in scene.tiles],
        "entities": [entity.as_dict() for entity in scene.entities],
        "transitions": [transition.as_dict() for transition in scene.transitions],
        "relations": dict(relations),
    }


def isometric_quarry_render_spec(scene: IsoQuarryScene, *, scene_id: str) -> dict[str, Any]:
    """Return the render-spec fragment for one quarry scene."""

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
            "levels": list(scene.trace.get("levels", [])),
            "tile_count": int(scene.trace.get("tile_count", 0)),
            "entity_count": int(scene.trace.get("entity_count", 0)),
            "background_rgb": list(scene.trace.get("background_rgb", [])),
            "background_tone_id": str(scene.trace.get("background_tone_id", "")),
            "background_tone_rgb": list(scene.trace.get("background_tone_rgb", [])),
            "background_tone_family": str(scene.trace.get("background_tone_family", "")),
        },
    }


def isometric_quarry_elevation_render_map(
    *,
    scene: IsoQuarryScene,
    candidate_tile_ids_by_label: Mapping[str, str],
    selected_label: str,
) -> dict[str, Any]:
    """Return task render-map fields for terrain-elevation option selection."""

    tiles_by_id = {str(tile.tile_id): tile for tile in scene.tiles}
    label_bboxes = {
        str(label): rounded_bbox(scene.label_bboxes_by_tile_id[str(tile_id)])
        for label, tile_id in candidate_tile_ids_by_label.items()
        if str(tile_id) in scene.label_bboxes_by_tile_id
    }
    tile_bboxes = {
        str(label): rounded_bbox(tiles_by_id[str(tile_id)].bbox_xyxy)
        for label, tile_id in candidate_tile_ids_by_label.items()
    }
    tile_levels = {
        str(label): int(tiles_by_id[str(tile_id)].level)
        for label, tile_id in candidate_tile_ids_by_label.items()
    }
    selected_tile_id = str(candidate_tile_ids_by_label[str(selected_label)])
    selected_tile = tiles_by_id[selected_tile_id]
    return {
        "image_id": "img0",
        "candidate_tile_ids_by_label": dict(candidate_tile_ids_by_label),
        "candidate_tile_bboxes_px_by_label": tile_bboxes,
        "candidate_label_bboxes_px_by_label": label_bboxes,
        "candidate_levels_by_label": tile_levels,
        "selected_label": str(selected_label),
        "selected_tile_id": selected_tile_id,
        "selected_tile_level": int(selected_tile.level),
        "selected_tile_bbox_px": rounded_bbox(selected_tile.bbox_xyxy),
    }


def isometric_quarry_object_count_render_map(
    *,
    scene: IsoQuarryScene,
    target_object_type: str,
    target_level: int,
    counted_entity_ids: Sequence[str],
) -> dict[str, Any]:
    """Return task render-map fields for terrain-level object counting."""

    entities_by_id = {str(entity.entity_id): entity for entity in scene.entities}
    level_object_counts: dict[str, dict[str, int]] = {}
    for entity in scene.entities:
        level_key = str(int(entity.level))
        object_key = (
            str(entity.metadata.get("quarry_object_type", ""))
            if str(entity.object_type) == "quarry_object"
            else str(entity.object_type)
        )
        level_object_counts.setdefault(level_key, {})
        level_object_counts[level_key][object_key] = int(level_object_counts[level_key].get(object_key, 0)) + 1
    counted_ids = [str(entity_id) for entity_id in counted_entity_ids]
    counted_bboxes = [rounded_bbox(entities_by_id[entity_id].bbox_xyxy) for entity_id in counted_ids if entity_id in entities_by_id]
    return {
        "image_id": "img0",
        "target_object_type": str(target_object_type),
        "target_level": int(target_level),
        "counted_entity_ids": counted_ids,
        "counted_entity_bboxes_px": counted_bboxes,
        "entity_levels_by_id": {str(entity.entity_id): int(entity.level) for entity in scene.entities},
        "entity_object_types_by_id": {
            str(entity.entity_id): (
                str(entity.metadata.get("quarry_object_type", ""))
                if str(entity.object_type) == "quarry_object"
                else str(entity.object_type)
            )
            for entity in scene.entities
        },
        "entity_base_object_types_by_id": {str(entity.entity_id): str(entity.object_type) for entity in scene.entities},
        "level_object_counts": level_object_counts,
        "answer_count": len(counted_ids),
    }


def isometric_quarry_render_spec_with_label_font(
    scene: IsoQuarryScene,
    *,
    scene_id: str,
    label_font_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Return render-spec fields plus the sampled candidate-label font."""

    render_spec = isometric_quarry_render_spec(scene, scene_id=scene_id)
    return {
        **render_spec,
        "style": {
            **dict(render_spec["style"]),
            "label_font": dict(label_font_trace),
        },
    }


def make_tile_label_selection_params(
    *,
    sample: LabelTaskSampleSpec,
    candidate_labels: Sequence[str],
    candidate_tile_ids_by_label: Mapping[str, str],
    render_map: Mapping[str, Any],
    selected_label: str,
    selected_tile: IsoQuarryTile,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return common selection params for bbox-annotated terrain-tile label tasks."""

    selected_tile_id = str(candidate_tile_ids_by_label[str(selected_label)])
    query_key_name = "query" + "_id"
    params = {
        query_key_name: str(sample.selected_key),
        "prompt_query_key": str(sample.prompt_key),
        query_key_name + "_probabilities": dict(sample.selection_probabilities),
        "candidate_count": int(sample.candidate_count),
        "candidate_count_probabilities": dict(sample.candidate_count_probabilities),
        "candidate_labels": [str(label) for label in candidate_labels],
        "candidate_tile_ids_by_label": dict(candidate_tile_ids_by_label),
        "candidate_levels_by_label": dict(render_map["candidate_levels_by_label"]),
        "selected_label": str(selected_label),
        "selected_tile_id": selected_tile_id,
        "selected_tile_level": int(selected_tile.level),
        "canvas_profile": str(sample.canvas_profile),
        "canvas_profile_probabilities": dict(sample.canvas_profile_probabilities),
    }
    params.update(dict(extra_fields or {}))
    return params


def make_tile_label_trace_payload(
    *,
    domain: str,
    scene_id: str,
    task_identity: str,
    scene: IsoQuarryScene,
    sample: LabelTaskSampleSpec,
    prompt_artifacts: Any,
    label_font_trace: Mapping[str, Any],
    render_map: Mapping[str, Any],
    candidate_tile_ids_by_label: Mapping[str, str],
    candidate_labels: Sequence[str],
    selected_label: str,
    selected_tile: IsoQuarryTile,
    annotation_value: Sequence[float],
    relations: Mapping[str, Any],
    selection_params_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
    witness_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common trace payload for terrain-tile option-label tasks."""

    annotation = rounded_bbox(annotation_value)
    selection_params = make_tile_label_selection_params(
        sample=sample,
        candidate_labels=candidate_labels,
        candidate_tile_ids_by_label=candidate_tile_ids_by_label,
        render_map=render_map,
        selected_label=selected_label,
        selected_tile=selected_tile,
        extra_fields=selection_params_extra,
    )
    selected_tile_id = str(candidate_tile_ids_by_label[str(selected_label)])
    query_key_name = "query" + "_id"
    task_key_name = "task" + "_id"
    execution = {
        query_key_name: str(sample.selected_key),
        "prompt_query_key": str(sample.prompt_key),
        "scene_id": str(scene_id),
        "answer": str(selected_label),
        "selected_label": str(selected_label),
        "selected_tile_id": selected_tile_id,
        "selected_tile_level": int(selected_tile.level),
        "candidate_tile_ids_by_label": dict(candidate_tile_ids_by_label),
        "candidate_levels_by_label": dict(render_map["candidate_levels_by_label"]),
        "renderer": dict(scene.trace),
    }
    execution.update(dict(execution_extra or {}))
    witness = {
        "answer_label": str(selected_label),
        "selected_tile_id": selected_tile_id,
        "selected_tile_level": int(selected_tile.level),
        "selected_tile_bbox": list(annotation),
    }
    witness.update(dict(witness_extra or {}))
    return {
        "scene_ir": isometric_quarry_scene_ir(
            domain=str(domain),
            scene_id=str(scene_id),
            scene=scene,
            relations=relations,
        ),
        "query_spec": {
            task_key_name: str(task_identity),
            query_key_name: str(sample.selected_key),
            "prompt_query_key": str(sample.prompt_key),
            "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": selection_params,
        },
        "render_spec": isometric_quarry_render_spec_with_label_font(
            scene,
            scene_id=str(scene_id),
            label_font_trace=label_font_trace,
        ),
        "render_map": dict(render_map),
        "execution_trace": execution,
        "witness_symbolic": witness,
        "projected_annotation": bbox_projection(annotation),
    }


__all__ = [
    "bbox_set_projection",
    "bbox_projection",
    "isometric_quarry_elevation_render_map",
    "isometric_quarry_object_count_render_map",
    "isometric_quarry_render_spec",
    "isometric_quarry_render_spec_with_label_font",
    "isometric_quarry_scene_ir",
    "make_tile_label_selection_params",
    "make_tile_label_trace_payload",
    "rounded_bbox",
]
