"""Trace-section helpers for the 3D bar-grid scene."""

from __future__ import annotations

from typing import Any, Mapping

from .state import (
    BarGridRenderArtifacts,
    _Dataset,
)


def values_by_category(dataset: _Dataset) -> dict[str, dict[str, int]]:
    return {
        str(x_label): {
            str(series_label): int(
                next(
                    bar.value
                    for bar in dataset.bars
                    if str(bar.x_label) == str(x_label) and str(bar.series_label) == str(series_label)
                )
            )
            for series_label in dataset.series_labels
        }
        for x_label in dataset.x_labels
    }


def bar_grid_relation_fields(
    *,
    dataset: _Dataset,
    ranges: Mapping[str, Any],
    selection_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Return neutral relation fields shared by 3D bar-grid objectives."""

    fields: dict[str, Any] = {
        "category_count": int(len(dataset.x_labels)),
        "series_count": int(len(dataset.series_labels)),
        "category_labels": [str(label) for label in dataset.x_labels],
        "series_labels": [str(label) for label in dataset.series_labels],
        "annotation_kind": str(dataset.selection.annotation_kind),
        "annotation_bar_ids": [str(bar_id) for bar_id in dataset.selection.annotation_bar_ids],
        "answer_value": int(dataset.selection.answer),
        **dict(ranges),
        **dict(selection_trace),
    }
    if dataset.selection.annotation_bar_id_pairs:
        fields["annotation_bar_id_pairs"] = [
            [str(first), str(second)]
            for first, second in dataset.selection.annotation_bar_id_pairs
        ]
    if dataset.selection.annotation_bar_id_groups:
        fields["annotation_bar_id_groups"] = {
            str(key): [str(bar_id) for bar_id in bar_ids]
            for key, bar_ids in dataset.selection.annotation_bar_id_groups.items()
        }
    return fields


def render_trace_sections(artifacts: BarGridRenderArtifacts) -> tuple[dict[str, Any], dict[str, Any]]:
    rendered = artifacts.rendered
    render_spec = {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "coord_space": "pixel",
        "scene_variant": "three_d_bar_grid",
        "background_style": dict(artifacts.background_style),
        "information_scene_style": dict(artifacts.background_style["information_scene_style"]),
        "font_assets": dict(artifacts.font_assets),
        "post_image_noise": dict(artifacts.post_image_noise),
        "plot_bbox_px": list(rendered.plot_bbox_px),
        "y_axis_max": int(rendered.y_axis_max),
        "y_ticks": [int(tick) for tick in rendered.y_ticks],
        "layout_jitter": dict(rendered.layout_jitter_meta),
        "bar_style": dict(rendered.bar_style_meta),
    }
    render_map = {
        "image_id": "img0",
        "plot_bbox_px": list(rendered.plot_bbox_px),
        "bar_traces": [dict(trace) for trace in rendered.bar_traces],
        "legend_traces": [dict(trace) for trace in rendered.legend_traces],
    }
    return render_spec, render_map


def build_trace_scaffold(
    *,
    dataset: _Dataset,
    artifacts: BarGridRenderArtifacts,
    relations: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    question_format: str,
) -> dict[str, Any]:
    render_spec, render_map = render_trace_sections(artifacts)
    return {
        "scene_ir": {
            "scene_kind": "chart_three_d_bar_grid",
            "entities": [dict(entity) for entity in artifacts.rendered.entities],
            "relations": dict(relations),
        },
        "render_spec": render_spec,
        "render_map": render_map,
        "execution_trace": {
            "answer_value": int(dataset.selection.answer),
            "question_format": str(question_format),
            "values_by_category": values_by_category(dataset),
            "annotation_bar_ids": [str(bar_id) for bar_id in dataset.selection.annotation_bar_ids],
            **dict(relations),
        },
        "witness_symbolic": {
            "type": "three_d_bar_grid_values",
            **dict(witness_symbolic),
        },
        "projected_annotation": dict(projected_annotation),
    }
