"""Output helpers for density-curve chart scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.density_curve.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_ID,
    SCENE_NAMESPACE,
    SCENE_VARIANT,
    resolve_density_curve_render_params,
)
from trace_tasks.tasks.charts.density_curve.shared.rendering import render_density_curve_scene
from trace_tasks.tasks.charts.density_curve.shared.state import DensityCurveDataset, DensityCurveRendered
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.tasks.shared.font_assets import font_asset_version, sample_font_family
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


def render_dataset(
    dataset: DensityCurveDataset,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[DensityCurveRendered, Dict[str, Any]]:
    """Render a dataset with chart background/font/noise context."""

    render_params = resolve_density_curve_render_params(params, instance_seed=int(instance_seed))
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        render_params=render_params,
        protected_colors=(),
    )
    chart_font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
        exclude_tags=("display",),
        explicit_key="chart_font_family",
        weights_key="chart_font_family_weights",
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = render_density_curve_scene(
            background.copy(),
            dataset=dataset,
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered = replace(rendered, image=image)
    render_meta = {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": SCENE_VARIANT,
        "background_style": dict(background_meta),
        "information_scene_style": dict(information_style_meta),
        "post_image_noise": dict(post_noise_meta),
        "layout_jitter": dict(render_params.layout_jitter_meta or {}),
        "font_assets": {
            "asset_version": str(font_asset_version()),
            "chart_font_family": str(chart_font_family),
        },
        "chart_font_family": str(chart_font_family),
        "font_asset_version": str(font_asset_version()),
        "text_style": {
            "label_font_size_px": int(render_params.label_font_size_px),
            "tick_font_size_px": int(render_params.tick_font_size_px),
            "label_stroke_width_px": int(render_params.label_stroke_width_px),
        },
        "axis_style": {
            "axis_line_width_px": int(render_params.axis_line_width_px),
            "grid_line_width_px": int(render_params.grid_line_width_px),
            "tick_length_px": int(render_params.tick_length_px),
        },
        "plot_bbox_px": list(rendered.plot_bbox_px),
        "legend_bbox_px": list(rendered.legend_bbox_px),
        **dict(rendered.render_meta),
    }
    return rendered, dict(render_meta)


def curve_records(dataset: DensityCurveDataset) -> list[dict[str, Any]]:
    """Return trace-friendly curve records."""

    return [
        {
            "label": str(curve.label),
            "family": str(curve.family),
            "component_count": int(curve.component_count),
            "color_rgb": [int(channel) for channel in curve.color_rgb],
            "line_style": str(curve.line_style),
            "mean_x": round(float(curve.mean_x), 4),
            "mode_x": round(float(curve.mode_x), 4),
            "mode_y": round(float(curve.mode_y), 8),
            "interval_mass": round(float(curve.interval_mass), 6),
            "density_at_x": round(float(curve.density_at_x), 8),
        }
        for curve in dataset.curves
    ]


def build_trace_scaffold(
    *,
    dataset: DensityCurveDataset,
    rendered: DensityCurveRendered,
    render_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    answer_label: str,
) -> Dict[str, Any]:
    """Build common trace sections for a density-curve task."""

    return {
        "scene_ir": {
            "scene_kind": "chart_density_curve",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": SCENE_VARIANT,
                "answer_label": str(answer_label),
                "annotation_key": str(dataset.query.annotation_key),
                "visible_role": str(dataset.query.visible_role),
            },
        },
        "render_spec": dict(render_meta),
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.plot_bbox_px),
            "legend_item_bboxes_px": dict(rendered.legend_item_bboxes_px),
            "curve_bboxes_px": dict(rendered.curve_bboxes_px),
            "mean_marker_bboxes_px": dict(rendered.mean_marker_bboxes_px),
            "mode_marker_bboxes_px": dict(rendered.mode_marker_bboxes_px),
            "interval_mass_bboxes_px": dict(rendered.interval_mass_bboxes_px),
            "interval_mass_points_px": dict(rendered.interval_mass_points_px),
            "density_at_x_points_px": dict(rendered.density_at_x_points_px),
        },
        "execution_trace": {
            **dict(dataset.query.trace),
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "answer": str(answer_label),
            "curve_records": curve_records(dataset),
            "question_format": "density_curve_label_selection",
        },
        "witness_symbolic": {
            "type": "object_set",
            "value": [str(answer_label)],
        },
        "projected_annotation": dict(projected_annotation),
    }
