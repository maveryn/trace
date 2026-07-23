"""Objective-neutral trace fragments for Minecraft-like block-world tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .sampling import MinecraftAxes
from .state import MinecraftSceneSample, RenderedMinecraftScene


def build_minecraft_common_trace_params(
    *,
    axes: MinecraftAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared scene params plus task-owned replay params."""

    params: Dict[str, Any] = {
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "grid_width": int(axes.grid_width),
        "grid_depth": int(axes.grid_depth),
        "grid_width_probabilities": dict(axes.grid_width_probabilities),
        "grid_depth_probabilities": dict(axes.grid_depth_probabilities),
        "target_answer": int(axes.target_answer),
        "answer_probabilities": dict(axes.answer_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_minecraft_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: MinecraftAxes,
    sample: MinecraftSceneSample,
    rendered: RenderedMinecraftScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    panel_style_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: int,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble trace sections after a public task binds answer and annotation."""

    block_trace = [
        {
            "block_id": str(block.block_id),
            "x": int(block.x),
            "y": int(block.y),
            "z": int(block.z),
            "kind": str(block.kind),
        }
        for block in sample.blocks
    ]
    terrain_trace = [
        {
            "cell_id": str(cell.cell_id),
            "x": int(cell.x),
            "y": int(cell.y),
            "kind": str(cell.kind),
        }
        for cell in sample.terrain_cells
    ]
    return {
        "scene_ir": {
            "scene_kind": "games_minecraft_block_world",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "style_variant": str(sample.style_variant),
                "grid_width": int(sample.grid_width),
                "grid_depth": int(sample.grid_depth),
                "river_width": int(sample.river_width),
                "scaffold_cost": int(sample.scaffold_cost),
                "target_stack_height": int(sample.target_stack_height),
                "stack_height_condition": str(sample.stack_height_condition),
                "track_cells": [list(cell) for cell in sample.track_cells],
                "track_raised_block_count": int(sample.answer) if sample.track_cells else 0,
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "style_variant": str(sample.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(rendered.render_map.get("text_style", {})),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "style_variant": str(sample.style_variant),
            "answer": int(answer_value),
            "grid_width": int(sample.grid_width),
            "grid_depth": int(sample.grid_depth),
            "river_width": int(sample.river_width),
            "scaffold_cost": int(sample.scaffold_cost),
            "ladder_present": bool(sample.ladder_present),
            "ladder_columns": [list(column) for column in sample.ladder_columns],
            "target_stack_height": int(sample.target_stack_height),
            "stack_height_condition": str(sample.stack_height_condition),
            "track_cells": [list(cell) for cell in sample.track_cells],
            "track_raised_block_count": int(sample.answer) if sample.track_cells else 0,
            "target_resource_kind": str(sample.target_resource_kind),
            "counted_resource_kind": str(sample.counted_resource_kind),
            "player_cell": list(sample.player_cell) if sample.player_cell is not None else None,
            "target_cell": list(sample.target_cell) if sample.target_cell is not None else None,
            "terrain_cells": terrain_trace,
            "blocks": block_trace,
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(sample.construction_mode),
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "panel_scene_style": dict(panel_style_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_minecraft_common_trace_params",
    "build_minecraft_trace_payload",
]
