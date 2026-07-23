"""Trace payload helpers for Star Battle puzzle tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter

from .state import RenderedStarBattleScene, SCENE_ID, StarBattleRenderParams


def json_ready(value: Any) -> Any:
    """Convert nested tuples and mappings into JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [json_ready(inner) for inner in value]
    return value


def build_render_spec(
    *,
    rendered_scene: RenderedStarBattleScene,
    render_params: StarBattleRenderParams,
    scene_variant: str,
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build common render metadata for Star Battle tasks."""

    return {
        "scene_id": SCENE_ID,
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(scene_variant),
        "background_style": dict(background_meta),
        "scene_style": dict(scene_style_meta),
        "post_image_noise": dict(post_noise_meta),
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "unit_size_jitter": dict(render_params.unit_size_jitter),
    }


def build_render_map(
    *,
    rendered_scene: RenderedStarBattleScene,
    render_params: StarBattleRenderParams,
    annotation_source: str,
) -> Dict[str, Any]:
    """Build pixel render map sections used by annotation projection."""

    return with_puzzle_unit_size_jitter(
        {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "cell_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.cell_bbox_map.items()
            },
            "row_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.row_bbox_map.items()
            },
            "col_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.col_bbox_map.items()
            },
            "region_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.region_bbox_map.items()
            },
            "item_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.item_bbox_map.items()
            },
            "annotation_source": str(annotation_source),
        },
        render_params.unit_size_jitter,
    )


def build_trace_payload(
    *,
    scene_ir: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    render_map: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    answer_gt: Mapping[str, Any],
    annotation_gt: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
) -> Dict[str, Any]:
    """Assemble one JSON-stable Star Battle trace payload."""

    return {
        "scene_ir": json_ready(dict(scene_ir)),
        "query_spec": json_ready(dict(query_spec)),
        "render_spec": json_ready(dict(render_spec)),
        "render_map": json_ready(dict(render_map)),
        "execution_trace": json_ready(dict(execution_trace)),
        "witness_symbolic": json_ready(dict(witness_symbolic)),
        "projected_annotation": json_ready(dict(projected_annotation)),
        "answer_gt": json_ready(dict(answer_gt)),
        "annotation_gt": json_ready(dict(annotation_gt)),
        "prompt_spec": {
            "defaults": json_ready(dict(prompt_defaults)),
            "active": json_ready(dict(prompt_artifacts.prompt_variant)),
        },
    }


__all__ = [
    "build_render_map",
    "build_render_spec",
    "build_trace_payload",
    "json_ready",
]
