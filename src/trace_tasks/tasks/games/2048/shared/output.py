"""Objective-neutral output assembly for the 2048 games scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .rendering import Rendered2048TaskContext
from .sampling import Resolved2048Axes
from .state import EMPTY, SCENE_ID, Move2048Result, Sample2048


def move_result_trace(result: Move2048Result) -> Dict[str, Any]:
    """Serialize one 2048 move result into JSON-compatible trace data."""

    return {
        "direction": str(result.direction),
        "after": [[int(value) for value in row] for row in result.after],
        "merge_pairs": [
            [[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]]
            for a, b in result.merge_pairs
        ],
        "score": int(result.score),
        "moved": bool(result.moved),
        "result_sources": [
            {
                "dest": [int(dest[0]), int(dest[1])],
                "sources": [[int(source[0]), int(source[1])] for source in sources],
            }
            for dest, sources in sorted(result.result_sources.items())
        ],
    }


def filled_2048_cell_count(sample: Sample2048) -> int:
    """Count non-empty cells in one generated 2048 board."""

    return sum(1 for row in sample.board for value in row if int(value) != EMPTY)


def common_2048_trace_params(
    axes: Resolved2048Axes,
    sample: Sample2048,
    *,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared 2048 trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "move_direction": str(sample.move_direction),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "move_direction_probabilities": dict(axes.move_direction_probabilities),
        "filled_count": int(filled_2048_cell_count(sample)),
        "result_option_count": int(len(sample.result_option_boards)),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_2048_common_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_entity_ids: tuple[str, ...],
    annotation_cell_id_pairs: list[list[str]],
    axes: Resolved2048Axes,
    sample: Sample2048,
    rendered_context: Rendered2048TaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble objective-neutral trace sections after task-specific binding."""

    rendered_scene = rendered_context.rendered_scene
    if str(annotation_artifacts.annotation_type) == "segment_set":
        witness_symbolic = {
            "type": "object_pair_set",
            "pairs": [list(pair) for pair in annotation_cell_id_pairs],
            "ids": [str(entity_id) for pair in annotation_cell_id_pairs for entity_id in pair],
        }
    else:
        witness_symbolic = {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        }
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"games_2048_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "move_direction": str(sample.move_direction),
                "result_option_labels": [str(label) for label in sample.result_option_boards.keys()],
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
                "annotation_entity_id_pairs": [list(pair) for pair in annotation_cell_id_pairs],
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
            "effective_cell_size_px": rendered_scene.render_map.get("effective_cell_size_px"),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "board_before": [[int(value) for value in row] for row in sample.board],
            "move_direction": str(sample.move_direction),
            "move_result": move_result_trace(sample.move_result),
            "all_move_results": {
                str(direction): move_result_trace(result)
                for direction, result in sample.all_move_results.items()
            },
            "result_option_boards": {
                str(label): [[int(value) for value in row] for row in option_board]
                for label, option_board in sample.result_option_boards.items()
            },
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "annotation_entity_id_pairs": [list(pair) for pair in annotation_cell_id_pairs],
            "construction_mode": str(sample.construction_mode),
            **dict(execution_extra),
        },
        "witness_symbolic": witness_symbolic,
        "projected_annotation": annotation_artifacts.projected_annotation,
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }
    return trace_payload


__all__ = [
    "build_2048_common_trace_payload",
    "common_2048_trace_params",
    "filled_2048_cell_count",
    "move_result_trace",
]
