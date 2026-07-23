"""Objective-neutral trace assembly for simplified darts games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedDartsTaskContext
from .state import DartsSampledScene, DartsSceneAxes


def _dart_specs_for_trace(sample: DartsSampledScene) -> list[Dict[str, Any]]:
    """Return JSON-friendly dart specs for trace payloads."""

    return [
        {
            "dart_id": str(dart.dart_id),
            "label": None if dart.label is None else str(dart.label),
            "area_kind": str(dart.area_kind),
            "sector_value": None if dart.sector_value is None else int(dart.sector_value),
            "score": int(dart.score),
            "is_marked": bool(dart.is_marked),
        }
        for dart in sample.darts
    ]


def build_darts_common_trace_params(
    *,
    axes: DartsSceneAxes,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared darts prompt params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_darts_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    axes: DartsSceneAxes,
    sample: DartsSampledScene,
    rendered_context: RenderedDartsTaskContext,
    prompt_defaults: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    answer_value: Any,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble darts trace sections after task-specific answer binding."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": f"games_darts_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
                "target_score": sample.target_score,
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "dart_fill_color": list(rendered_scene.render_map.get("dart_fill_color") or []),
            "dart_fill_min_lab_distance": rendered_scene.render_map.get("dart_fill_min_lab_distance"),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "answer": answer_value,
            "dart_specs": _dart_specs_for_trace(sample),
            "dart_count": int(len(sample.darts)),
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "target_score": sample.target_score,
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = ["build_darts_common_trace_params", "build_darts_trace_payload"]
