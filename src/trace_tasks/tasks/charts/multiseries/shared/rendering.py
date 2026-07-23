"""Rendering helpers for multiseries chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from .....core.visual.noise import apply_post_image_noise
from ....shared.font_assets import font_asset_version
from ....shared.text_rendering import temporary_default_font_family
from ...shared.chart_scene_multiseries import render_multiseries_chart_scene
from ...shared.chart_scene_primitives import value_axis_render_metadata
from ...shared.chart_scene_types import MultiSeriesChartMarkSpec
from ...shared.labeled_chart_render_params import resolve_chart_render_params_for_task
from .defaults import (
    DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    sample_chart_font_family,
)
from .state import MultiseriesRenderResult
from .styles import resolve_multiseries_chart_colors


def build_multiseries_mark_specs(
    *,
    category_labels: Sequence[str],
    series_labels: Sequence[str],
    values_by_category: Mapping[str, Mapping[str, int]],
    mark_style: Mapping[str, Any],
) -> List[MultiSeriesChartMarkSpec]:
    """Build chart-mark specs for one multiseries chart."""

    fill_palette = [tuple(int(channel) for channel in value) for value in mark_style["series_fill_palette_rgb"]]
    outline_palette = [tuple(int(channel) for channel in value) for value in mark_style["series_outline_palette_rgb"]]
    if len(fill_palette) != len(series_labels) or len(outline_palette) != len(series_labels):
        raise ValueError("multiseries charts require one color per series")

    specs: List[MultiSeriesChartMarkSpec] = []
    for category_rank, category_label in enumerate(category_labels):
        values_for_category = values_by_category[str(category_label)]
        for series_rank, series_label in enumerate(series_labels):
            specs.append(
                MultiSeriesChartMarkSpec(
                    category_label=str(category_label),
                    series_label=str(series_label),
                    category_rank=int(category_rank),
                    series_rank=int(series_rank),
                    value=int(values_for_category[str(series_label)]),
                    fill_rgb=fill_palette[int(series_rank)],
                    outline_rgb=outline_palette[int(series_rank)],
                )
            )
    return specs



def render_multiseries_dataset(
    *,
    values_by_category: Mapping[str, Mapping[str, int]],
    trace_extras: Mapping[str, Any],
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> MultiseriesRenderResult:
    """Render one neutral multiseries dataset with shared chart styling.

    Task files provide the already-bound category/series values and annotation
    targets. This renderer only samples visual presentation, draws the selected
    multiseries variant, and records projection metadata for later annotation
    binding.
    """

    category_labels = [str(label) for label in trace_extras["category_labels"]]
    series_labels = [str(label) for label in trace_extras["series_labels"]]
    mark_style = resolve_multiseries_chart_colors(
        params,
        render_defaults=RENDER_DEFAULTS,
        defaults=DEFAULTS,
        instance_seed=int(instance_seed),
        series_count=len(series_labels),
    )
    marks = build_multiseries_mark_specs(
        category_labels=category_labels,
        series_labels=series_labels,
        values_by_category=values_by_category,
        mark_style=mark_style,
    )
    render_params = resolve_chart_render_params_for_task(
        {**dict(params), **mark_style},
        render_defaults=RENDER_DEFAULTS,
        defaults=DEFAULTS,
        instance_seed=int(instance_seed),
    )
    protected_colors = tuple(
        tuple(int(channel) for channel in color)
        for key in ("series_fill_palette_rgb", "series_outline_palette_rgb")
        for color in mark_style.get(key, ())
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="multiseries",
        render_params=render_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(int(instance_seed), params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_multiseries_chart_scene(
            background,
            scene_variant=str(scene_variant),
            marks=marks,
            render_params=render_params,
            instance_seed=int(instance_seed),
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return MultiseriesRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        mark_style=dict(mark_style),
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
        category_labels=list(category_labels),
        series_labels=list(series_labels),
    )


def category_label_centers(rendered_scene: Any) -> Dict[str, list[float]]:
    return {
        str(mark["category_label"]): list(mark["category_label_center_px"])
        for mark in rendered_scene.mark_traces
    }


def render_spec_payload(result: MultiseriesRenderResult, *, scene_variant: str) -> Dict[str, Any]:
    rendered_scene = result.rendered_scene
    render_params = result.render_params
    return {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(scene_variant),
        "background_style": dict(result.background_meta),
        "information_scene_style": dict(result.background_meta["information_scene_style"]),
        "post_image_noise": dict(result.post_noise_meta),
        "font_assets": {
            "asset_version": font_asset_version(),
            "chart_font_family": str(result.chart_font_family),
        },
        "layout_jitter": dict(render_params.layout_jitter_meta or {}),
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
        "mark_style": {
            "sampling_policy": str(result.mark_style["sampling_policy"]),
            **{str(key): value for key, value in result.mark_style.items()},
        },
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "y_axis_max": int(rendered_scene.y_axis_max),
        "y_ticks": [int(value) for value in rendered_scene.y_ticks],
        **value_axis_render_metadata(rendered_scene),
    }


def render_map_payload(result: MultiseriesRenderResult) -> Dict[str, Any]:
    rendered_scene = result.rendered_scene
    return {
        "image_id": "img0",
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "legend_bbox_px": list(rendered_scene.legend_bbox_px),
        "legend_item_bboxes_px": {
            str(label): list(bbox)
            for label, bbox in rendered_scene.legend_item_bboxes_px.items()
        },
        "context_protected_bboxes_px": {
            "plot": list(rendered_scene.plot_bbox_px),
            **({"legend": list(rendered_scene.legend_bbox_px)} if rendered_scene.legend_bbox_px else {}),
        },
        "category_label_centers_px": category_label_centers(rendered_scene),
    }


__all__ = [
    "MultiseriesRenderResult",
    "category_label_centers",
    "render_map_payload",
    "render_multiseries_dataset",
    "render_spec_payload",
]
