"""Rendering primitives for single-series chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.chart_scene_labeled import render_labeled_chart_scene
from trace_tasks.tasks.charts.shared.chart_scene_primitives import value_axis_render_metadata
from trace_tasks.tasks.charts.shared.chart_scene_types import ChartMarkSpec
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.tasks.charts.shared.labeled_chart_marks import (
    build_chart_mark_specs,
    resolve_chart_mark_colors,
)
from trace_tasks.tasks.charts.shared.labeled_chart_render_params import resolve_chart_render_params_for_task
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family

from .defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, RENDER_DEFAULTS
from .state import SCENE_ID, SCENE_NAMESPACE, SingleSeriesDataset, SingleSeriesRenderResult


def render_dataset(
    *,
    dataset: SingleSeriesDataset,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    hidden_labels: Sequence[str] = (),
) -> SingleSeriesRenderResult:
    """Render one dataset and project chart marks without task-specific branching."""

    mark_style = resolve_chart_mark_colors(
        params,
        render_defaults=RENDER_DEFAULTS,
        defaults=DEFAULTS,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        mark_count=len(dataset.labels),
    )
    marks = _mark_specs(
        labels=dataset.labels,
        values=dataset.values,
        scene_variant=str(scene_variant),
        mark_style=mark_style,
        hidden_labels=tuple(str(label) for label in hidden_labels),
    )
    render_params = resolve_chart_render_params_for_task(
        {**dict(params), **dict(mark_style)},
        render_defaults=RENDER_DEFAULTS,
        defaults=DEFAULTS,
        instance_seed=int(instance_seed),
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        render_params=render_params,
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_labeled_chart_scene(
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
    return SingleSeriesRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        chart_font_family=str(chart_font_family),
        mark_style=dict(mark_style),
        background_meta=dict(background_meta),
        information_style_meta=dict(information_style_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def _mark_specs(
    *,
    labels: Sequence[str],
    values: Sequence[int],
    scene_variant: str,
    mark_style: Mapping[str, Any],
    hidden_labels: Sequence[str],
) -> tuple[ChartMarkSpec, ...]:
    base_marks = build_chart_mark_specs(
        labels=tuple(str(label) for label in labels),
        values=tuple(int(value) for value in values),
        scene_variant=str(scene_variant),
        mark_style=mark_style,
    )
    hidden = {str(label) for label in hidden_labels}
    if not hidden:
        return tuple(base_marks)
    marks: list[ChartMarkSpec] = []
    for mark in base_marks:
        if str(mark.label) not in hidden:
            marks.append(mark)
            continue
        marks.append(
            ChartMarkSpec(
                label=str(mark.label),
                value=int(mark.value),
                fill_rgb=mark.fill_rgb,
                outline_rgb=mark.outline_rgb,
                visible=False,
            )
        )
    return tuple(marks)


def font_assets_payload(*, chart_font_family: str) -> dict[str, Any]:
    return chart_font_asset_metadata(str(chart_font_family))


def axis_render_metadata(rendered: SingleSeriesRenderResult) -> dict[str, Any]:
    return value_axis_render_metadata(rendered.rendered_scene)


__all__ = [
    "axis_render_metadata",
    "font_assets_payload",
    "render_dataset",
]
