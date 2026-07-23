"""Objective-neutral trace helpers for Snake tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import coord_to_cell_id, simulate_snake_moves, visible_snake_trace
from .state import SCENE_ID, SnakeSample, SnakeSceneAxes


def snake_trace_params(
    *,
    axes: SnakeSceneAxes,
    sample: SnakeSample,
    answer_value: int | str,
    answer_support: list[int] | list[str] | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return common prompt-query params plus task-owned extras."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "board_size": int(sample.state.board_size),
        "body_length": len(sample.state.body),
        "obstacle_count": len(tuple(sample.state.obstacles)),
        "obstacle_cell_ids": [coord_to_cell_id(coord) for coord in sample.state.obstacles],
        "planned_move_count": len(sample.planned_moves) if sample.planned_moves else None,
        "planned_moves": [str(move) for move in sample.planned_moves],
        "safe_directions": [str(direction) for direction in sample.safe_directions],
        "snake_length": int(len(sample.state.body) + 1),
        "head_cell_id": coord_to_cell_id(sample.state.head),
        "body_cell_ids": [coord_to_cell_id(coord) for coord in sample.state.body],
        "answer_value": answer_value,
        "answer_support": answer_support,
        "result_options": [dict(option) for option in sample.result_options],
        "target_outcome": sample.target_outcome,
        "observed_event_step": sample.observed_event_step,
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "board_size_probabilities": dict(axes.board_size_probabilities),
        "body_length_probabilities": dict(axes.body_length_probabilities),
        "planned_move_count_probabilities": dict(axes.planned_move_count_probabilities),
        "obstacle_count_probabilities": dict(axes.obstacle_count_probabilities),
    }
    if extra:
        params.update(dict(extra))
    return params


def snake_execution_trace(
    *,
    axes: SnakeSceneAxes,
    sample: SnakeSample,
    answer_value: int | str,
    prompt_key: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return symbolic execution trace derived from the same sample as the answer."""

    simulation_trace = None
    if sample.planned_moves:
        simulation = simulate_snake_moves(sample.state, sample.planned_moves)
        simulation_trace = {
            "outcome": str(simulation.outcome),
            "event_step": int(simulation.event_step),
            "traversed_coords": [[int(row), int(col)] for row, col in simulation.traversed_coords],
            "collision_coord": None if simulation.collision_coord is None else [int(simulation.collision_coord[0]), int(simulation.collision_coord[1])],
            "final_head": [int(simulation.final_head[0]), int(simulation.final_head[1])],
        }
    trace = {
        "scene_id": SCENE_ID,
        "objective_key": str(prompt_key),
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "state": dict(visible_snake_trace(sample.state)),
        "planned_moves": [str(move) for move in sample.planned_moves],
        "planned_move_count": len(sample.planned_moves) if sample.planned_moves else None,
        "safe_directions": [str(direction) for direction in sample.safe_directions],
        "snake_length": int(len(sample.state.body) + 1),
        "head_cell_id": coord_to_cell_id(sample.state.head),
        "body_cell_ids": [coord_to_cell_id(coord) for coord in sample.state.body],
        "answer_value": answer_value,
        "result_options": [dict(option) for option in sample.result_options],
        "target_outcome": sample.target_outcome,
        "observed_event_step": sample.observed_event_step,
        "simulation": simulation_trace,
        "annotation_cell_ids": [str(cell_id) for cell_id in sample.annotation_cell_ids],
        "construction_mode": str(sample.construction_mode),
    }
    if extra:
        trace.update(dict(extra))
    return trace


def build_snake_trace_payload(
    *,
    axes: SnakeSceneAxes,
    sample: SnakeSample,
    rendered_entities: list[dict[str, Any]],
    render_map: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: int | str,
    prompt_key: str,
    background_meta: Mapping[str, Any],
    panel_style_meta: Mapping[str, Any],
    text_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    annotation_artifacts: AnnotationArtifacts,
    answer_support: list[int] | list[str] | None,
    params_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble common trace sections before lifecycle injects prompt metadata."""

    trace_params = snake_trace_params(
        axes=axes,
        sample=sample,
        answer_value=answer_value,
        answer_support=answer_support,
        extra=params_extra,
    )
    return {
        "scene_ir": {
            "scene_kind": f"games_snake_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "objective_key": str(prompt_key),
                "style_variant": str(axes.style_variant),
                "board_size": int(sample.state.board_size),
                "obstacle_count": len(tuple(sample.state.obstacles)),
                "obstacle_cell_ids": [coord_to_cell_id(coord) for coord in sample.state.obstacles],
                "annotation_cell_ids": [str(cell_id) for cell_id in sample.annotation_cell_ids],
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(text_style_meta),
        },
        "render_map": dict(render_map),
        "execution_trace": snake_execution_trace(
            axes=axes,
            sample=sample,
            answer_value=answer_value,
            prompt_key=str(prompt_key),
            extra=execution_extra,
        ),
        "witness_symbolic": {
            "type": "cell_set",
            "ids": [str(cell_id) for cell_id in sample.annotation_cell_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "params_for_prompt": trace_params,
    }


__all__ = ["build_snake_trace_payload", "snake_execution_trace", "snake_trace_params"]
