"""Trace payload helpers for polyomino assembly puzzles."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter

from .state import SCENE_ID


def json_ready(value: Any) -> Any:
    """Convert nested tuples/mappings to JSON-ready containers."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [json_ready(child) for child in value]
    if isinstance(value, list):
        return [json_ready(child) for child in value]
    return value


def build_trace_payload(
    *,
    dataset: Mapping[str, Any],
    rendered_scene: Any,
    render_params: Any,
    prompt_meta: Mapping[str, Any],
    task_fields: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    question_format: str,
    view_family: str,
) -> dict[str, Any]:
    """Serialize scene/render/execution traces for one public task."""

    item_map = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered_scene.item_bbox_map.items()
    }
    return {
        "scene_ir": {
            "scene_kind": f"puzzle_{SCENE_ID}_{dataset.get('scene_variant', '')}",
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.get("scene_variant", "")),
                "answer_option_label": str(dataset["answer_option_label"]),
                "correct_option_choice_id": str(dataset["correct_option_choice_id"]),
                "view_family": str(view_family),
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.get("scene_variant", "")),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": [
                round(float(value), 3) for value in rendered_scene.scene_bbox_px
            ],
            "text_style": {
                "option_label_font_size_px": int(render_params.option_label_font_size_px),
            },
            "shape_color": {
                "name": str(render_params.shape_color_name),
                "rgb": [int(value) for value in render_params.shape_fill_rgb],
            },
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        "render_map": with_puzzle_unit_size_jitter(
            {
                "image_id": "img0",
                "scene_bbox_px": [
                    round(float(value), 3) for value in rendered_scene.scene_bbox_px
                ],
                "item_bboxes_px": dict(item_map),
                "annotation_source": "item_bboxes_px",
            },
            render_params.unit_size_jitter,
        ),
        "execution_trace": {
            **dict(task_fields),
            "scene_variant": str(dataset.get("scene_variant", "")),
            "question_format": str(question_format),
            "view_family": str(view_family),
            "answer_option_label": str(dataset["answer_option_label"]),
            "correct_option_index": int(dataset["correct_option_index"]),
            "correct_option_choice_id": str(dataset["correct_option_choice_id"]),
            "shape_color_name": str(render_params.shape_color_name),
            "shape_color_rgb": [int(value) for value in render_params.shape_fill_rgb],
            "annotation_item_roles": ["selected_option"],
            "option_specs": [dict(spec) for spec in dataset.get("option_specs", [])],
            "solver_trace": dict(dataset.get("solver_trace", {})),
            "variant_payload": {
                str(key): json_ready(value)
                for key, value in dict(dataset).items()
                if str(key) not in {"option_specs", "solver_trace"}
            },
        },
        "witness_symbolic": {
            "type": str(projected_annotation.get("type", "")),
            "value": projected_annotation.get("bbox", projected_annotation.get("value")),
        },
        "projected_annotation": dict(projected_annotation),
        "prompt_spec": {
            "bundle_id": str(prompt_meta["bundle_id"]),
            "active": dict(prompt_meta["prompt_variant"]),
        },
    }


__all__ = ["build_trace_payload", "json_ready"]
