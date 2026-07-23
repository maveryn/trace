"""Objective-neutral trace fragments for arithmetic-constraint puzzles."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .rendering import RenderedArithmeticContext
from .sampling import ResolvedSceneAxes
from .state import ArithmeticCase, SCENE_ID


def _json_value(value: Any) -> Any:
    """Convert arithmetic trace values to JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_json_value(inner) for inner in value]
    if isinstance(value, list):
        return [_json_value(inner) for inner in value]
    return value


def arithmetic_trace_params(
    *,
    axes: ResolvedSceneAxes,
    case: ArithmeticCase,
    prompt_query_key: str,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return scene/task params for prompt-backed query-spec metadata."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "case_kind": str(case.kind),
        "layout_style": str(case.layout_style),
        "answer_range": [int(case.answer_range[0]), int(case.answer_range[1])],
    }
    if extra_params:
        params.update(_json_value(dict(extra_params)))
    return params


def build_arithmetic_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_item_id: str,
    axes: ResolvedSceneAxes,
    case: ArithmeticCase,
    rendered_context: RenderedArithmeticContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble common trace payload sections after task-owned binding."""

    rendered_scene = rendered_context.rendered_scene
    execution = {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "case_kind": str(case.kind),
        "layout_style": str(case.layout_style),
        "answer_value": int(case.answer_value),
        "answer_support": [int(value) for value in case.answer_support],
        "constraint_data": _json_value(case.data),
        "target_item_id": str(annotation_item_id),
    }
    if execution_extra:
        execution.update(_json_value(dict(execution_extra)))
    return {
        "scene_ir": {
            "scene_kind": f"puzzles_arithmetic_panel_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "case_kind": str(case.kind),
                "layout_style": str(case.layout_style),
                "target_item_id": str(annotation_item_id),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            **dict(rendered_context.render_meta),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": {"font": dict(rendered_context.font_meta)},
            "background_style": dict(rendered_context.background_meta),
        },
        "render_map": {
            "annotation_source": "item_bboxes_px",
            "item_bboxes_px": {
                str(item_id): [float(value) for value in bbox]
                for item_id, bbox in rendered_scene.item_bbox_map.items()
            },
            "scene_bbox_px": [float(value) for value in rendered_scene.scene_bbox_px],
        },
        "execution_trace": execution,
        "witness_symbolic": {
            "type": "single_object",
            "id": str(annotation_item_id),
        },
        "projected_annotation": annotation_artifacts.projected_annotation,
        "prompt_spec": {
            "defaults": dict(prompt_defaults),
            "active": dict(prompt_artifacts.prompt_variant),
        },
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = [
    "arithmetic_trace_params",
    "build_arithmetic_trace_payload",
]
