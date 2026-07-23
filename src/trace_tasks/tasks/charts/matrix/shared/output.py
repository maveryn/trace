"""Neutral render and trace scaffolding for matrix charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .rendering import MatrixRenderResult, font_assets_payload


def render_spec_payload(
    *,
    dataset: Mapping[str, Any],
    rendered: MatrixRenderResult,
    scene_variant: str,
    palette_variant: str,
    header_layout: str,
    grid_style: str,
) -> dict[str, Any]:
    """Return render metadata independent of public task identity."""

    render_params = rendered.render_params
    return {
        "scene_variant": str(scene_variant),
        "palette_variant": str(palette_variant),
        "header_layout": str(header_layout),
        "grid_style": str(grid_style),
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "row_count": int(dataset["row_count"]),
        "column_count": int(dataset["column_count"]),
        "value_min": int(dataset["value_min"]),
        "value_max": int(dataset["value_max"]),
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "font_assets": font_assets_payload(render_params),
        "background_style": dict(rendered.background_meta),
        "information_scene_style": dict(rendered.background_meta["information_scene_style"]),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def render_map_payload(*, rendered: MatrixRenderResult) -> dict[str, Any]:
    """Return pixel lookup maps for rendered matrix elements."""

    return {
        "panel_bbox_px": list(rendered.rendered_scene.panel_bbox_px),
        "title_bbox_px": list(rendered.rendered_scene.title_bbox_px),
        "matrix_bbox_px": list(rendered.rendered_scene.matrix_bbox_px),
        "legend_bbox_px": list(rendered.rendered_scene.legend_bbox_px),
        "cell_bboxes_px": dict(rendered.rendered_scene.cell_bbox_map),
        "row_label_bboxes_px": dict(rendered.rendered_scene.row_label_bbox_map),
        "column_label_bboxes_px": dict(rendered.rendered_scene.column_label_bbox_map),
    }


def base_prompt_params(
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    palette_variant: str,
    palette_variant_probabilities: Mapping[str, float],
    header_layout: str,
    header_layout_probabilities: Mapping[str, float],
    grid_style: str,
    grid_style_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Return common prompt/replay params before task-local branch fields."""

    return {
        "scene_variant": str(scene_variant),
        "palette_variant": str(palette_variant),
        "header_layout": str(header_layout),
        "grid_style": str(grid_style),
        "row_count": int(dataset["row_count"]),
        "column_count": int(dataset["column_count"]),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "palette_variant_probabilities": dict(palette_variant_probabilities),
        "header_layout_probabilities": dict(header_layout_probabilities),
        "grid_style_probabilities": dict(grid_style_probabilities),
        **dict(dataset["question_params"]),
    }


def build_trace_scaffold(
    *,
    dataset: Mapping[str, Any],
    rendered: MatrixRenderResult,
    scene_variant: str,
    palette_variant: str,
    header_layout: str,
    grid_style: str,
    prompt_spec: Mapping[str, Any],
    relations: Mapping[str, Any],
    question_format: str,
    answer_value: int | str,
    answer_type: str,
    annotation_type: str,
    annotation_cell_ids: Sequence[str],
    support_header_keys: Sequence[str],
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    annotation_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build common trace sections around task-owned answer and annotation."""

    answerability = "unanswerable" if bool(dataset["is_unanswerable"]) else "answerable"
    absence = {"absence_proof": dict(dataset["absence_proof"])} if bool(dataset["is_unanswerable"]) else {}
    normalized_cell_ids = [str(cell_id) for cell_id in annotation_cell_ids]
    normalized_header_keys = [str(key) for key in support_header_keys]
    return {
        "scene_ir": {
            "scene_kind": "chart_annotated_matrix",
            "entities": [dict(item) for item in rendered.rendered_scene.entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "answer_value": answer_value,
                "answer_row_index": int(dataset["answer_row_index"]),
                "answer_column_index": int(dataset["answer_column_index"]),
                "annotation_cell_ids": list(normalized_cell_ids),
                "support_header_keys": list(normalized_header_keys),
                "answerability": answerability,
                **absence,
                **dict(relations),
            },
        },
        "query_spec": dict(prompt_spec),
        "render_spec": render_spec_payload(
            dataset=dataset,
            rendered=rendered,
            scene_variant=str(scene_variant),
            palette_variant=str(palette_variant),
            header_layout=str(header_layout),
            grid_style=str(grid_style),
        ),
        "render_map": render_map_payload(rendered=rendered),
        "execution_trace": {
            "scene_id": "matrix",
            "scene_variant": str(scene_variant),
            "question_format": str(question_format),
            "scene_title": str(dataset["scene_title"]),
            "row_count": int(dataset["row_count"]),
            "column_count": int(dataset["column_count"]),
            "row_labels": list(dataset["row_labels"]),
            "column_labels": list(dataset["column_labels"]),
            "values": [[None if value is None else int(value) for value in row] for row in dataset["values"]],
            "cells": [dict(cell) for cell in dataset["cells"]],
            "cells_by_id": {str(key): dict(value) for key, value in dict(dataset["cells_by_id"]).items()},
            "answer_value": answer_value,
            "answer_type": str(answer_type),
            "answer_row_index": int(dataset["answer_row_index"]),
            "answer_column_index": int(dataset["answer_column_index"]),
            "annotation_type": str(annotation_type),
            "annotation_cell_ids": list(normalized_cell_ids),
            "support_header_keys": list(normalized_header_keys),
            "query_axis": str(dataset.get("query_axis", "")),
            "extremum_direction": str(dataset.get("extremum_direction", "")),
            "comparison": str(dataset.get("comparison", "")),
            "extremum_rank": int(dataset.get("extremum_rank", 0)),
            "scene_meta": dict(dataset["scene_meta"]),
            "answerability": answerability,
            **absence,
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
        "annotation_refs": [dict(ref) for ref in annotation_refs],
    }


__all__ = [
    "base_prompt_params",
    "build_trace_scaffold",
    "render_map_payload",
    "render_spec_payload",
]
