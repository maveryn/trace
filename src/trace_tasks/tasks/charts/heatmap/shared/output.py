"""Rendering and trace helpers for heatmap chart scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.heatmap.shared.annotations import annotation_refs
from trace_tasks.tasks.charts.heatmap.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_ID,
    _condition_support,
)
from trace_tasks.tasks.charts.heatmap.shared.rendering import _render_heatmap, _resolve_render_params, _value_palette
from trace_tasks.tasks.shared.font_assets import font_asset_version


def render_dataset(
    dataset: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Render one already-constructed heatmap dataset and record style metadata."""

    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    render_params = _resolve_render_params(render_style_params)
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        render_params=render_params,
        protected_colors=_value_palette(str(dataset["scene_variant"])),
    )
    rendered_scene = _render_heatmap(
        background,
        scene_title=str(dataset["scene_title"]),
        scene_variant=str(dataset["scene_variant"]),
        row_labels=list(dataset["row_labels"]),
        column_labels=list(dataset["column_labels"]),
        cells=list(dataset["cells"]),
        render_params=render_params,
        colorbar_ticks=tuple(int(value) for value in dataset.get("colorbar_ticks", ())),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered_scene = replace(rendered_scene, image=image)
    render_meta = {
        "scene_variant": str(dataset["scene_variant"]),
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "row_count": int(dataset["row_count"]),
        "column_count": int(dataset["column_count"]),
        "heat_bin_count": int(dataset["heat_bin_count"]),
        "colorbar_value_min": dataset.get("colorbar_value_min"),
        "colorbar_value_max": dataset.get("colorbar_value_max"),
        "colorbar_ticks": list(dataset.get("colorbar_ticks", [])),
        "cell_gap_px": int(render_params.cell_gap_px),
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "font_assets": {
            "asset_version": font_asset_version(),
            "chart_font_family": str(render_params.font_family),
        },
        "background_style": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    return rendered_scene, dict(render_meta), {**dict(background_meta), "information_scene_style": dict(information_style_meta)}, dict(post_noise_meta)


def build_trace_scaffold(
    *,
    dataset: Mapping[str, Any],
    rendered: Any,
    render_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_value: Any,
    answer_value: int | str,
) -> Dict[str, Any]:
    """Assemble scene-neutral trace sections before task relation params are added."""

    annotation_cell_ids = [str(cell_id) for cell_id in dataset["annotation_cell_ids"]]
    prompt_key = str(dataset["prompt_key"])
    question_params = dict(dataset["question_params"])
    return {
        "scene_ir": {
            "scene_kind": "chart_heatmap",
            "entities": [dict(item) for item in rendered.entities],
            "relations": {
                "prompt_key": str(prompt_key),
                "scene_variant": str(dataset["scene_variant"]),
                "query_axis": str(dataset["query_axis"]),
                "answer_value": answer_value,
                "answer_row_index": int(dataset["answer_row_index"]),
                "answer_column_index": int(dataset["answer_column_index"]),
                "annotation_cell_ids": list(annotation_cell_ids),
                "answerability": "unanswerable" if bool(dataset["is_unanswerable"]) else "answerable",
                **({"absence_proof": dict(dataset["absence_proof"])} if bool(dataset["is_unanswerable"]) else {}),
            },
        },
        "render_spec": dict(render_meta),
        "render_map": {
            "panel_bbox_px": list(rendered.panel_bbox_px),
            "title_bbox_px": list(rendered.title_bbox_px),
            "grid_bbox_px": list(rendered.grid_bbox_px),
            "legend_bbox_px": list(rendered.legend_bbox_px),
            "cell_bboxes_px": dict(rendered.cell_bbox_map),
            "row_label_bboxes_px": dict(rendered.row_label_bbox_map),
            "column_label_bboxes_px": dict(rendered.column_label_bbox_map),
        },
        "execution_trace": {
            "prompt_key": str(prompt_key),
            "scene_variant": str(dataset["scene_variant"]),
            "question_format": "heatmap_query",
            "scene_title": str(dataset["scene_title"]),
            "row_count": int(dataset["row_count"]),
            "column_count": int(dataset["column_count"]),
            "row_labels": list(dataset["row_labels"]),
            "column_labels": list(dataset["column_labels"]),
            "heat_bin_count": int(dataset["heat_bin_count"]),
            "colorbar_value_min": dataset.get("colorbar_value_min"),
            "colorbar_value_max": dataset.get("colorbar_value_max"),
            "colorbar_ticks": list(dataset.get("colorbar_ticks", [])),
            "values": [[int(value) for value in row] for row in dataset["values"]],
            "cells": [dict(cell) for cell in dataset["cells"]],
            "cells_by_id": {str(key): dict(value) for key, value in dict(dataset["cells_by_id"]).items()},
            "answer_value": answer_value,
            "answer_type": str(dataset["answer_type"]),
            "answer_row_index": int(dataset["answer_row_index"]),
            "answer_column_index": int(dataset["answer_column_index"]),
            "annotation_cell_ids": list(annotation_cell_ids),
            "query_axis": str(dataset["query_axis"]),
            "condition_kind": str(dataset["condition_kind"]),
            "extremum_direction": str(dataset["extremum_direction"]),
            **dict(question_params),
            "annotation_semantics": str(prompt_key),
            "answerability": "unanswerable" if bool(dataset["is_unanswerable"]) else "answerable",
            **({"absence_proof": dict(dataset["absence_proof"])} if bool(dataset["is_unanswerable"]) else {}),
        },
        "witness_symbolic": {
            "type": "heatmap_witness",
            "candidate_cell_ids": list(annotation_cell_ids),
            "answer_value": answer_value,
            "answer_row_index": int(dataset["answer_row_index"]),
            "answer_column_index": int(dataset["answer_column_index"]),
            "answerability": "unanswerable" if bool(dataset["is_unanswerable"]) else "answerable",
            **({"absence_proof": dict(dataset["absence_proof"])} if bool(dataset["is_unanswerable"]) else {}),
        },
        "projected_annotation": dict(projected_annotation),
        "annotation_refs": annotation_refs(
            annotation_cell_ids=list(annotation_cell_ids),
            annotation_value=annotation_value,
            annotation_type=str(projected_annotation["type"]),
        ),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


def discrete_relation_params(
    *,
    dataset: Mapping[str, Any],
    scene_variant_probabilities: Mapping[str, float],
    objective_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build scene-level relation metadata shared by discrete heatmap objectives."""

    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset["scene_variant"]),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "row_count": int(dataset["row_count"]),
        "column_count": int(dataset["column_count"]),
        "row_count_probabilities": dict(dataset["row_count_probabilities"]),
        "column_count_probabilities": dict(dataset["column_count_probabilities"]),
        "heat_bin_count": int(dataset["heat_bin_count"]),
        "query_axis": str(dataset["query_axis"]),
        **dict(objective_params),
        **dict(dataset["question_params"]),
    }


def continuous_colorbar_relation_params(
    *,
    dataset: Mapping[str, Any],
    objective_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build scene-level relation metadata shared by continuous-colorbar objectives."""

    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset["scene_variant"]),
        "scene_variant_probabilities": {"continuous_colorbar_heatmap": 1.0},
        "row_count": int(dataset["row_count"]),
        "column_count": int(dataset["column_count"]),
        "row_count_probabilities": dict(dataset["row_count_probabilities"]),
        "column_count_probabilities": dict(dataset["column_count_probabilities"]),
        "heat_bin_count": int(dataset["heat_bin_count"]),
        "colorbar_value_min": dataset.get("colorbar_value_min"),
        "colorbar_value_max": dataset.get("colorbar_value_max"),
        "colorbar_ticks": list(dataset.get("colorbar_ticks", [])),
        **dict(objective_params or {}),
        **dict(dataset["question_params"]),
    }


def condition_relation_params(
    *,
    dataset: Mapping[str, Any],
    scene_variant_probabilities: Mapping[str, float],
    condition_kind_probabilities: Mapping[str, float],
    query_axis_probabilities: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Build relation metadata for heatmap objectives based on cell conditions."""

    return discrete_relation_params(
        dataset=dataset,
        scene_variant_probabilities=scene_variant_probabilities,
        objective_params={
            **({"query_axis_probabilities": dict(query_axis_probabilities)} if query_axis_probabilities else {}),
            "condition_kind_probabilities": dict(condition_kind_probabilities),
            "condition_support": list(_condition_support(str(dataset["scene_variant"]))),
        },
    )


def axis_cell_relation_params(
    *,
    dataset: Mapping[str, Any],
    scene_variant_probabilities: Mapping[str, float],
    query_axis: str,
    extremum_direction: str,
) -> Dict[str, Any]:
    """Build relation metadata for a named-axis cell-extremum lookup."""

    return discrete_relation_params(
        dataset=dataset,
        scene_variant_probabilities=scene_variant_probabilities,
        objective_params={
            "query_axis_probabilities": {str(query_axis): 1.0},
            "extremum_direction_probabilities": {str(extremum_direction): 1.0},
            "condition_support": list(_condition_support(str(dataset["scene_variant"]))),
        },
    )


__all__ = [
    "build_trace_scaffold",
    "axis_cell_relation_params",
    "condition_relation_params",
    "continuous_colorbar_relation_params",
    "discrete_relation_params",
    "render_dataset",
]
