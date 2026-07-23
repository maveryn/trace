"""Objective-neutral trace payload helpers for Snakes and Ladders tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import board_last_square
from .state import SCENE_ID, SnakesLaddersAxes, SnakesLaddersMove, SnakesLaddersSample


def move_trace(move: SnakesLaddersMove) -> dict[str, Any]:
    """Serialize one move trace."""

    return move.to_trace()


def snakes_ladders_trace_params(
    *,
    axes: SnakesLaddersAxes,
    sample: SnakesLaddersSample,
    prompt_query_key: str,
    answer_support: list[int] | None,
    query_id_probabilities: Mapping[str, float],
    trace_extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return common query params plus task-owned extras."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "prompt_query_key": str(prompt_query_key),
        "style_variant": str(axes.style_variant),
        "board_side": int(axes.board_side),
        "last_square": int(board_last_square(int(axes.board_side))),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "query_id_probabilities": dict(query_id_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "board_side_probabilities": dict(axes.board_side_probabilities),
        "answer_value": int(sample.answer),
        "answer_support": answer_support,
    }
    if trace_extra_params:
        params.update(dict(trace_extra_params))
    return params


def snakes_ladders_execution_trace(
    *,
    axes: SnakesLaddersAxes,
    sample: SnakesLaddersSample,
    prompt_query_key: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return symbolic execution trace from the same sample as answer/annotation."""

    trace = {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "prompt_query_key": str(prompt_query_key),
        "style_variant": str(axes.style_variant),
        "board_side": int(axes.board_side),
        "last_square": int(board_last_square(int(axes.board_side))),
        "start_square": int(sample.start_square),
        "jumps": [jump.to_trace() for jump in sample.jumps],
        "shown_move": None if sample.move is None else move_trace(sample.move),
        "horizon_roll_count": None if sample.horizon_roll_count is None else int(sample.horizon_roll_count),
        "optimal_route": [move_trace(move) for move in sample.optimal_route],
        "answer": int(sample.answer),
        "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        "construction_mode": str(sample.construction_mode),
    }
    if execution_extra:
        trace.update(dict(execution_extra))
    return trace


def build_snakes_ladders_trace_payload(
    *,
    axes: SnakesLaddersAxes,
    sample: SnakesLaddersSample,
    rendered_entities: list[dict[str, Any]],
    render_map: Mapping[str, Any],
    image_size: tuple[int, int],
    prompt_query_key: str,
    query_id_probabilities: Mapping[str, float],
    background_meta: Mapping[str, Any],
    panel_style_meta: Mapping[str, Any],
    text_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    annotation_artifacts: AnnotationArtifacts,
    witness_symbolic: Mapping[str, Any],
    answer_support: list[int] | None,
    params_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble common trace sections before lifecycle injects prompt metadata."""

    trace_params = snakes_ladders_trace_params(
        axes=axes,
        sample=sample,
        prompt_query_key=str(prompt_query_key),
        answer_support=answer_support,
        query_id_probabilities=query_id_probabilities,
        trace_extra_params=params_extra,
    )
    return {
        "scene_ir": {
            "scene_kind": f"games_snakes_ladders_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "prompt_query_key": str(prompt_query_key),
                "style_variant": str(axes.style_variant),
                "board_side": int(axes.board_side),
                "last_square": int(board_last_square(int(axes.board_side))),
                "start_square": int(sample.start_square),
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "scene_variant": str(axes.scene_variant),
            "prompt_query_key": str(prompt_query_key),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "board_side": int(axes.board_side),
            "last_square": int(board_last_square(int(axes.board_side))),
            "layout_jitter": dict(render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(text_style_meta),
        },
        "render_map": dict(render_map),
        "execution_trace": snakes_ladders_execution_trace(
            axes=axes,
            sample=sample,
            prompt_query_key=str(prompt_query_key),
            execution_extra=execution_extra,
        ),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "params_for_prompt": trace_params,
    }


__all__ = [
    "build_snakes_ladders_trace_payload",
    "move_trace",
    "snakes_ladders_execution_trace",
    "snakes_ladders_trace_params",
]
