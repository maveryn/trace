"""Neutral rendering and trace assembly helpers for error-bar series scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.errorbar_series.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
)
from trace_tasks.tasks.charts.errorbar_series.shared.rendering import render_errorbar_series_chart, resolve_errorbar_render_params
from trace_tasks.tasks.charts.errorbar_series.shared.state import SCENE_ID, ErrorbarDataset, ErrorbarRendered
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


def render_dataset(
    dataset: ErrorbarDataset,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[ErrorbarRendered, Dict[str, Any], Dict[str, Any]]:
    """Render one dataset and return image metadata used by review artifacts."""

    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace="charts_errorbar_series.chart_font",
        params=params,
    )
    render_params = resolve_errorbar_render_params(
        params,
        instance_seed=int(instance_seed),
        chart_font_family=str(chart_font_family),
    )
    protected_colors = tuple(tuple(int(channel) for channel in series.color_rgb) for series in dataset.series)
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id=SCENE_ID,
        render_params=render_params,
        protected_colors=protected_colors,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = render_errorbar_series_chart(
            background,
            dataset=dataset,
            params=params,
            instance_seed=int(instance_seed),
            chart_font_family=str(chart_font_family),
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered = replace(rendered, image=image)
    render_meta = {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "coord_space": "pixel",
        "scene_variant": str(dataset.scene_variant),
        "plot_bbox_px": list(rendered.plot_bbox_px),
        "font_assets": chart_font_asset_metadata(str(chart_font_family)),
        "background_style": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
        **dict(rendered.render_meta),
    }
    sidecar_meta = {
        "background": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    return rendered, dict(render_meta), dict(sidecar_meta)


def errorbar_series_records(dataset: ErrorbarDataset) -> list[dict[str, Any]]:
    """Return symbolic series rows for trace/audit payloads."""

    rows: list[dict[str, Any]] = []
    for series in dataset.series:
        rows.append(
            {
                "series_id": str(series.series_id),
                "label": str(series.label),
                "color_rgb": list(series.color_rgb),
                "lower_values": list(series.lower_values),
                "mid_values": list(series.mid_values),
                "upper_values": list(series.upper_values),
            }
        )
    return rows


def build_trace_scaffold(
    *,
    dataset: ErrorbarDataset,
    rendered: ErrorbarRendered,
    render_meta: Mapping[str, Any],
    sidecar_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_refs: list[dict[str, Any]],
    answer_value: int | str,
) -> Dict[str, Any]:
    """Build the scene trace around task-bound answer and annotation records."""

    execution_trace = {
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.scene_variant),
        "question_format": "errorbar_series",
        "answer_value": answer_value,
        "answer_type": str(dataset.query.answer_type),
        "x_count": int(len(dataset.x_labels)),
        "series_count": int(len(dataset.series)),
        "x_labels": list(dataset.x_labels),
        "x_label_meta": dict(dataset.x_label_meta),
        "series_label_meta": dict(dataset.series_label_meta),
        "series": errorbar_series_records(dataset),
        "target_series_id": str(dataset.target_series_id),
        "target_x_index": dataset.target_x_index,
        "threshold_value": dataset.threshold_value,
        "query_params": dict(dataset.query.params),
        "annotation_item_keys": list(dataset.query.annotation_item_keys),
        "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
    }
    return {
        "scene_ir": {
            "scene_kind": "chart_errorbar_series",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(dataset.scene_variant),
                "target_series_id": str(dataset.target_series_id),
                "target_x_index": dataset.target_x_index,
            },
        },
        "render_spec": dict(render_meta),
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "errorbar_bboxes_px": dict(rendered.errorbar_bboxes_px),
            "point_map_px": dict(rendered.point_map_px),
            "threshold_bbox_px": rendered.threshold_bbox_px,
        },
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": "errorbar_series_witness",
            "annotation_kind": str(dataset.query.annotation_kind),
            "annotation_item_keys": list(dataset.query.annotation_item_keys),
            "answer": answer_value,
        },
        "projected_annotation": dict(projected_annotation),
        "sidecar": dict(sidecar_meta),
        "annotation_refs": [dict(ref) for ref in annotation_refs],
    }


__all__ = ["build_trace_scaffold", "errorbar_series_records", "render_dataset"]
