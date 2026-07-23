"""Objective-neutral trace fragments for match-3 game tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .state import Match3Sample, Match3SceneAxes, RenderedMatch3Scene


def build_match3_common_trace_params(
    *,
    axes: Match3SceneAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared scene params plus task-owned prompt/replay params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_match3_option_trace(sample: Match3Sample) -> list[dict[str, Any]]:
    """Serialize displayed swap options and immediate-clear outcomes."""

    return [
        {
            "label": str(option.label),
            "entity_id": str(option.entity_id),
            "from_cell": [int(option.outcome.move.a[0] + 1), int(option.outcome.move.a[1] + 1)],
            "to_cell": [int(option.outcome.move.b[0] + 1), int(option.outcome.move.b[1] + 1)],
            "clear_count": int(option.outcome.clear_count),
            "run_count": int(option.outcome.run_count),
            "cleared_cells": [[int(row + 1), int(col + 1)] for row, col in option.outcome.cleared_cells],
            "runs": [[[int(row + 1), int(col + 1)] for row, col in run] for run in option.outcome.runs],
            "is_answer": bool(option.is_answer),
        }
        for option in sample.option_specs
    ]


def build_match3_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: Match3SceneAxes,
    sample: Match3Sample,
    rendered: RenderedMatch3Scene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: int | str,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble match-3 trace sections after task-specific answer binding."""

    return {
        "scene_ir": {
            "scene_kind": f"games_match3_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "rows": int(len(sample.board)),
                "cols": int(len(sample.board[0]) if sample.board else 0),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered.style_meta),
            "match3_style": dict(rendered.render_map.get("match3_style", {})),
            "text_style": dict(rendered.render_map.get("text_style", {})),
            "effective_cell_size_px": rendered.render_map.get("effective_cell_size_px"),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "board_before": [list(row) for row in sample.board],
            "swap_options": build_match3_option_trace(sample),
            "answer": answer_value,
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_match3_common_trace_params",
    "build_match3_option_trace",
    "build_match3_trace_payload",
]
