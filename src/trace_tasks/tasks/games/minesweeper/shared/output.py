"""Objective-neutral trace fragments for Minesweeper games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .annotations import cell_ids_for_coords
from .sampling import MinesweeperAxes
from .state import MinesweeperSample
from .rendering import RenderedMinesweeperScene


def _coord_rows(coords: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Return coordinate tuples as JSON-friendly integer rows."""

    return [[int(row), int(col)] for row, col in coords]


def _option_coord_rows(options: Sequence[tuple[str, tuple[int, int]]]) -> list[dict[str, Any]]:
    """Return in-board option-label coordinates as JSON-friendly rows."""

    return [
        {"label": str(label), "coord": [int(coord[0]), int(coord[1])]}
        for label, coord in options
    ]


def build_minesweeper_common_trace_params(
    *,
    axes: MinesweeperAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared scene params plus task-owned replay params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "board_size": int(axes.board_size),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "board_size_probabilities": dict(axes.board_size_probabilities),
        "target_answer": int(axes.target_answer) if axes.target_answer is not None else None,
        "target_answer_support": [int(value) for value in axes.target_answer_support],
        "target_answer_probabilities": dict(axes.target_answer_probabilities),
        "hidden_count": 0,
        "flagged_count": 0,
    }
    if axes.option_count_support:
        params.update(
            {
                "option_count": int(axes.option_count),
                "option_count_support": [int(value) for value in axes.option_count_support],
                "option_count_probabilities": dict(axes.option_count_probabilities or {}),
            }
        )
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_minesweeper_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    keyed_annotation_entity_ids: Mapping[str, Sequence[str]],
    axes: MinesweeperAxes,
    sample: MinesweeperSample,
    rendered: RenderedMinesweeperScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    panel_style_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: Any,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble trace sections after a public task binds answer and annotation."""

    relation_map = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "board_size": int(sample.size),
        "target_answer": sample.target_answer,
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
        "keyed_annotation_entity_ids": {
            str(key): [str(entity_id) for entity_id in value]
            for key, value in keyed_annotation_entity_ids.items()
        },
        "candidate_option_cell_ids": {
            str(label): cell_ids_for_coords((coord,))[0]
            for label, coord in sample.candidate_option_coords
        },
    }
    execution = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "board_size": int(sample.size),
        "target_answer": sample.target_answer,
        "target_answer_support": [int(value) for value in axes.target_answer_support],
        "mine_coords": _coord_rows(sample.mine_coords),
        "revealed_coords": _coord_rows(sample.revealed_coords),
        "flagged_coords": _coord_rows(sample.flagged_coords),
        "hidden_coords": _coord_rows(sample.hidden_coords),
        "forced_mine_coords": _coord_rows(sample.forced_mine_coords),
        "forced_safe_coords": _coord_rows(sample.forced_safe_coords),
        "forcing_clue_coords": _coord_rows(sample.forcing_clue_coords),
        "annotation_coords": _coord_rows(sample.annotation_coords),
        "candidate_option_coords": _option_coord_rows(sample.candidate_option_coords),
        "candidate_option_cell_ids": {
            str(label): cell_ids_for_coords((coord,))[0]
            for label, coord in sample.candidate_option_coords
        },
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
        "keyed_annotation_entity_ids": {
            str(key): [str(entity_id) for entity_id in value]
            for key, value in keyed_annotation_entity_ids.items()
        },
        "answer": answer_value,
        "distractor_hidden_count": int(sample.distractor_hidden_count),
        "construction_mode": str(sample.construction_mode),
    }
    execution.update(dict(execution_extra or {}))
    return {
        "scene_ir": {
            "scene_kind": f"games_minesweeper_grid_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": relation_map,
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(rendered.render_map.get("text_style", {})),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": execution,
        "witness_symbolic": {
            "type": "keyed_object_set_map" if keyed_annotation_entity_ids else "object_set",
            **(
                {
                    "ids_by_role": {
                        str(key): [str(entity_id) for entity_id in value]
                        for key, value in keyed_annotation_entity_ids.items()
                    }
                }
                if keyed_annotation_entity_ids
                else {"ids": [str(entity_id) for entity_id in annotation_entity_ids]}
            ),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "panel_scene_style": dict(panel_style_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


def ids_for_sample_annotation(sample: MinesweeperSample) -> tuple[str, ...]:
    """Return cell ids for the sample's homogeneous annotation coordinates."""

    return cell_ids_for_coords(sample.annotation_coords)


__all__ = [
    "build_minesweeper_common_trace_params",
    "build_minesweeper_trace_payload",
    "ids_for_sample_annotation",
]
