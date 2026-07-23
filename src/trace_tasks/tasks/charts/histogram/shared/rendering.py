"""Rendering and trace helpers for histogram chart scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.histogram.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_FALLBACKS,
    RENDERING_DEFAULTS,
    SCENE_ID,
    SCENE_NAMESPACE,
    SCENE_VARIANT,
)
from trace_tasks.tasks.charts.histogram.shared.state import HistogramRenderArtifacts
from trace_tasks.tasks.charts.shared.chart_scene_histogram import render_histogram_scene
from trace_tasks.tasks.charts.shared.chart_scene_primitives import value_axis_render_metadata
from trace_tasks.tasks.charts.shared.chart_scene_types import HistogramBinSpec, RenderedChartScene
from trace_tasks.tasks.charts.shared.labeled_chart_marks import resolve_chart_mark_colors
from trace_tasks.tasks.charts.shared.labeled_chart_render_params import resolve_chart_render_params_for_task
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    sample_chart_font_family,
)
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


def resolve_mark_style(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    mark_count: int = 1,
) -> dict[str, Any]:
    """Resolve histogram mark colors without public task identity."""

    return dict(
        resolve_chart_mark_colors(
            params,
            render_defaults=RENDERING_DEFAULTS,
            defaults=RENDER_FALLBACKS,
            instance_seed=int(instance_seed),
            scene_variant=SCENE_VARIANT,
            mark_count=int(mark_count),
        )
    )


def _render_params(params: Mapping[str, Any], mark_style: Mapping[str, Any], *, instance_seed: int) -> Any:
    return resolve_chart_render_params_for_task(
        {**dict(params), **dict(mark_style)},
        render_defaults=RENDERING_DEFAULTS,
        defaults=RENDER_FALLBACKS,
        instance_seed=int(instance_seed),
    )


def render_histogram_dataset(
    *,
    bins: Sequence[HistogramBinSpec],
    params: Mapping[str, Any],
    mark_style: Mapping[str, Any],
    instance_seed: int,
) -> HistogramRenderArtifacts:
    """Render one histogram dataset and retain projection metadata."""

    render_params = _render_params(params, mark_style, instance_seed=int(instance_seed))
    styled_render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id=SCENE_ID,
        render_params=render_params,
        protected_colors=(
            tuple(int(value) for value in mark_style["mark_fill_rgb"]),
            tuple(int(value) for value in mark_style["mark_outline_rgb"]),
        ),
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_histogram_scene(
            background,
            bins=bins,
            render_params=styled_render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return HistogramRenderArtifacts(
        rendered_scene=replace(rendered_scene, image=image),
        render_params=styled_render_params,
        background_style=dict(background_meta),
        information_scene_style=dict(information_style_meta),
        font_assets=chart_font_asset_metadata(str(chart_font_family)),
        mark_style=dict(mark_style),
        post_image_noise=dict(post_noise_meta),
    )


def label_centers(rendered_scene: RenderedChartScene) -> dict[str, list[float]]:
    """Return x-axis label centers keyed by visible label."""

    return {
        str(mark["label"]): list(mark["label_center_px"])
        for mark in rendered_scene.mark_traces
    }


def counts_by_label(rendered_scene: RenderedChartScene) -> dict[str, int]:
    """Return visible bar counts keyed by x-axis label."""

    return {
        str(mark["label"]): int(mark["value"])
        for mark in rendered_scene.mark_traces
    }


def render_trace_sections(artifacts: HistogramRenderArtifacts) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build render trace sections for one rendered histogram."""

    rendered_scene = artifacts.rendered_scene
    render_params = artifacts.render_params
    render_spec = {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": SCENE_VARIANT,
        "background_style": dict(artifacts.background_style),
        "information_scene_style": dict(artifacts.information_scene_style),
        "post_image_noise": dict(artifacts.post_image_noise),
        "layout_jitter": dict(render_params.layout_jitter_meta or {}),
        "text_style": {
            "label_font_size_px": int(render_params.label_font_size_px),
            "tick_font_size_px": int(render_params.tick_font_size_px),
            "label_stroke_width_px": int(render_params.label_stroke_width_px),
        },
        "font_assets": dict(artifacts.font_assets),
        "axis_style": {
            "axis_line_width_px": int(render_params.axis_line_width_px),
            "grid_line_width_px": int(render_params.grid_line_width_px),
            "tick_length_px": int(render_params.tick_length_px),
        },
        "mark_style": dict(artifacts.mark_style),
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "y_axis_max": int(rendered_scene.y_axis_max),
        "y_ticks": [int(value) for value in rendered_scene.y_ticks],
        **value_axis_render_metadata(rendered_scene),
    }
    render_map = {
        "image_id": "img0",
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "label_centers_px": label_centers(rendered_scene),
    }
    return render_spec, render_map


__all__ = [
    "HistogramRenderArtifacts",
    "counts_by_label",
    "label_centers",
    "render_histogram_dataset",
    "render_trace_sections",
    "resolve_mark_style",
]
