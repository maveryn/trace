"""Identity-free trace serialization helpers for Connect Four scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import board_dimensions, player_name
from .rendering import RenderedConnectFourTaskContext
from .state import ConnectFourColumnProfileSample, ConnectFourCountSample, ConnectFourLabelSample, ConnectFourSceneAxes


def common_trace_params(
    *,
    axes: ConnectFourSceneAxes,
    sample: ConnectFourCountSample | ConnectFourLabelSample | ConnectFourColumnProfileSample,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return shared Connect Four query params plus task-owned fields."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "board_size_variant": str(axes.board_size_variant),
        "board_row_count": int(axes.board_rows),
        "board_column_count": int(axes.board_columns),
        "style_variant": str(axes.style_variant),
        "current_player": player_name(int(sample.current_player)).lower(),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "board_size_variant_probabilities": dict(axes.board_size_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "occupied_count": int(sample.occupied_count),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def common_trace_sections(
    *,
    axes: ConnectFourSceneAxes,
    sample: ConnectFourCountSample | ConnectFourLabelSample | ConnectFourColumnProfileSample,
    rendered_context: RenderedConnectFourTaskContext,
    annotation_artifacts: AnnotationArtifacts,
    query_spec: dict[str, Any],
    execution_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return trace sections shared by Connect Four objective-owned tasks."""

    rendered_scene = rendered_context.rendered_scene
    rows, columns = board_dimensions(sample.board)
    annotation_entity_ids = [str(entity_id) for entity_id in sample.evaluation.annotation_entity_ids]
    return {
        "scene_ir": {
            "scene_kind": f"games_connect_four_{str(sample.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(sample.scene_variant),
                "board_size_variant": str(sample.board_size_variant),
                "style_variant": str(sample.style_variant),
                "current_player": player_name(int(sample.current_player)).lower(),
                "annotation_entity_ids": list(annotation_entity_ids),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(sample.scene_variant),
            "board_size_variant": str(sample.board_size_variant),
            "style_variant": str(sample.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
            "effective_cell_size_px": rendered_scene.render_map.get("effective_cell_size_px"),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "board_size_variant": str(sample.board_size_variant),
            "style_variant": str(sample.style_variant),
            "board": [[int(cell) for cell in row] for row in sample.board],
            "board_row_count": int(rows),
            "board_column_count": int(columns),
            "current_player": player_name(int(sample.current_player)).lower(),
            "winning_move_coords": [[int(row), int(col)] for row, col in sample.evaluation.winning_move_coords],
            "safe_move_coords": [[int(row), int(col)] for row, col in sample.evaluation.safe_move_coords],
            "annotation_coords": [[int(row), int(col)] for row, col in sample.evaluation.annotation_coords],
            "annotation_entity_ids": list(annotation_entity_ids),
            "occupied_count": int(sample.occupied_count),
            "construction_mode": str(sample.construction_mode),
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "cell_set",
            "ids": list(annotation_entity_ids),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = ["common_trace_params", "common_trace_sections"]
