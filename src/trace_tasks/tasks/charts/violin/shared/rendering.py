"""Rendering helpers for violin chart tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.chart_scene_violin import render_violin_scene
from trace_tasks.tasks.charts.shared.labeled_chart_marks import resolve_chart_mark_colors
from trace_tasks.tasks.charts.shared.labeled_chart_render_params import resolve_chart_render_params_for_task
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    sample_chart_font_family,
)
from trace_tasks.tasks.charts.violin.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    RENDER_FALLBACKS,
    SCENE_ID,
    SCENE_VARIANT,
)
from trace_tasks.tasks.charts.violin.shared.state import ViolinRenderArtifacts
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


def resolve_mark_style(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> dict[str, Any]:
    """Resolve a neutral violin mark style."""

    return dict(
        resolve_chart_mark_colors(
            params,
            render_defaults=RENDERING_DEFAULTS,
            defaults=RENDER_FALLBACKS,
            instance_seed=int(instance_seed),
            scene_variant=SCENE_VARIANT,
            mark_count=1,
        )
    )


def render_violin_dataset(
    *,
    violins: Sequence[Any],
    params: Mapping[str, Any],
    instance_seed: int,
    mark_style: Mapping[str, Any],
) -> ViolinRenderArtifacts:
    """Render a sampled violin dataset and retain projection metadata."""

    render_params = resolve_chart_render_params_for_task(
        {**dict(params), **dict(mark_style)},
        render_defaults=RENDERING_DEFAULTS,
        defaults=RENDER_FALLBACKS,
        instance_seed=int(instance_seed),
    )
    protected_colors = tuple(
        tuple(int(channel) for channel in mark_style[key])
        for key in ("mark_fill_rgb", "mark_outline_rgb")
        if key in mark_style
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="violin",
        render_params=render_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace="charts.violin.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_violin_scene(
            background,
            violins=violins,
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return ViolinRenderArtifacts(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_style={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_image_noise=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
        mark_style=dict(mark_style),
    )


def render_spec_from_artifacts(artifacts: ViolinRenderArtifacts) -> dict[str, Any]:
    """Build scene-level render metadata common to all violin objectives."""

    render_params = artifacts.render_params
    rendered_scene = artifacts.rendered_scene
    mark_style = artifacts.mark_style
    return {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": SCENE_VARIANT,
        "background_style": dict(artifacts.background_style),
        "information_scene_style": dict(artifacts.background_style["information_scene_style"]),
        "post_image_noise": dict(artifacts.post_image_noise),
        "font_assets": chart_font_asset_metadata(str(artifacts.chart_font_family)),
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
        "violin_style": {
            "mode_line_style": str(render_params.violin_mode_line_style),
            "fill_style": str(render_params.violin_fill_style),
            "width_scale": round(float(render_params.violin_width_scale), 4),
            "smoothing_scale": round(float(render_params.violin_smoothing_scale), 4),
            "palette_mode": str(render_params.violin_palette_mode),
            "palette_offset": int(render_params.violin_palette_offset),
        },
        "mark_style": {
            "sampling_policy": str(mark_style["sampling_policy"]),
            "mark_fill_rgb": list(mark_style["mark_fill_rgb"]),
            "mark_outline_rgb": list(mark_style["mark_outline_rgb"]),
            **{
                str(key): value
                for key, value in mark_style.items()
                if key not in {"sampling_policy", "mark_fill_rgb", "mark_outline_rgb"}
            },
        },
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "y_axis_max": int(rendered_scene.y_axis_max),
        "y_ticks": [int(value) for value in rendered_scene.y_ticks],
    }


__all__ = [
    "render_spec_from_artifacts",
    "render_violin_dataset",
    "resolve_mark_style",
]
