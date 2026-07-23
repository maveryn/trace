"""Objective-neutral trace-section helpers for sliding-block tasks."""

from __future__ import annotations

from typing import Any, Mapping

from .state import SCENE_ID, RenderedSlidingBlockScene, SlidingBlockRenderParams


def _annotation_value_from_payload(annotation_payload: Mapping[str, Any]) -> Any:
    """Return the public annotation value from a projected annotation payload."""

    annotation_type = str(annotation_payload.get("type", "bbox_set"))
    if annotation_type == "bbox_map":
        return dict(annotation_payload.get("bbox_map", {}))
    if annotation_type == "bbox_set_map":
        return dict(annotation_payload.get("bbox_set_map", {}))
    if annotation_type == "bbox":
        return list(annotation_payload.get("bbox", []))
    return list(annotation_payload.get("bbox_set", []))


def common_query_params(
    *,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    exit_side: str,
    exit_side_probabilities: Mapping[str, float],
    dataset: Mapping[str, Any],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return shared query params plus task-owned params for trace metadata."""

    params: dict[str, Any] = {
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "exit_side": str(exit_side),
        "exit_side_probabilities": dict(exit_side_probabilities),
        "rows": int(dataset["rows"]),
        "cols": int(dataset["cols"]),
        "blocker_count": int(dataset.get("blocker_count", 0)),
        "movable_count": int(dataset.get("movable_count", 0)),
        "non_target_block_count": int(dataset["non_target_block_count"]),
        "move_count": int(len(dataset.get("move_sequence", []))),
        "option_count": int(len(dataset.get("option_boards", []))),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_common_trace_sections(
    *,
    rendered_scene: RenderedSlidingBlockScene,
    render_params: SlidingBlockRenderParams,
    render_map: Mapping[str, Any],
    scene_variant: str,
    exit_side: str,
    dataset: Mapping[str, Any],
    answer_value: int | str,
    answer_block_ids: list[str],
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    annotation_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble common trace sections after task-owned answer and annotation binding."""

    return {
        "scene_ir": {
            "scene_kind": "game_sliding_block",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(scene_variant),
                "exit_side": str(exit_side),
                "target_block_id": str(dataset["target_block_id"]),
                "blocking_block_ids": [str(item) for item in dataset["blocking_block_ids"]],
                "movable_block_ids": [str(item) for item in dataset.get("movable_block_ids", [])],
                "answer_value": answer_value,
                "view_family": SCENE_ID,
            },
        },
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "exit_side": str(exit_side),
            "panel_scene_style": dict(scene_style_meta),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "board_bbox_px": list(rendered_scene.board_bbox_px),
            "path_bbox_px": list(rendered_scene.path_bbox_px),
            "exit_arrow_bbox_px": list(rendered_scene.exit_arrow_bbox_px),
            "option_panel_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.option_panel_bbox_map.items()
            },
            "text_style": {
                "label_font_size_px": int(render_params.label_font_size_px),
                "target_label_font_size_px": int(render_params.target_label_font_size_px),
            },
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        "render_map": dict(render_map),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "exit_side": str(exit_side),
            "rows": int(dataset["rows"]),
            "cols": int(dataset["cols"]),
            "blocks": [dict(block) for block in dataset["blocks"]],
            "target_block_id": str(dataset["target_block_id"]),
            "blocking_block_ids": [str(item) for item in dataset["blocking_block_ids"]],
            "movable_block_ids": [str(item) for item in dataset.get("movable_block_ids", [])],
            "movable_count": int(dataset.get("movable_count", 0)),
            "answer_block_ids": [str(item) for item in answer_block_ids],
            "target_path_cells": [list(item) for item in dataset["target_path_cells"]],
            "move_sequence": [dict(item) for item in dataset.get("move_sequence", [])],
            "move_sequence_description": str(dataset.get("move_sequence_description", "")),
            "moved_block_ids": [str(item) for item in dataset.get("moved_block_ids", [])],
            "option_boards": [dict(item) for item in dataset.get("option_boards", [])],
            "correct_option_id": str(dataset.get("correct_option_id", "")),
            "answer_value": answer_value,
            "view_family": SCENE_ID,
        },
        "witness_symbolic": {
            "type": str(annotation_payload.get("type", "bbox_set")),
            "value": _annotation_value_from_payload(annotation_payload),
        },
        "projected_annotation": dict(annotation_payload),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = ["build_common_trace_sections", "common_query_params"]
