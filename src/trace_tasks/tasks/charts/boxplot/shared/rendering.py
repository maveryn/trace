"""Rendering and trace helpers for the boxplot chart scene."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.boxplot.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_FALLBACKS,
    RENDERING_DEFAULTS,
    SCENE_NAMESPACE,
    SCENE_VARIANT,
)
from trace_tasks.tasks.charts.shared.chart_scene_boxplot import render_boxplot_scene, render_paired_boxplot_scene
from trace_tasks.tasks.charts.shared.chart_scene_primitives import value_axis_render_metadata
from trace_tasks.tasks.charts.shared.chart_scene_types import BoxPlotSpec, RenderedChartScene
from trace_tasks.tasks.charts.shared.cartesian.annotations import projected_mark_annotation
from trace_tasks.tasks.charts.shared.labeled_chart_marks import resolve_chart_mark_colors
from trace_tasks.tasks.charts.shared.labeled_chart_render_params import resolve_chart_render_params_for_task
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    sample_chart_font_family,
)
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


@dataclass(frozen=True)
class BoxplotRenderArtifacts:
    rendered_scene: RenderedChartScene
    background_style: dict[str, Any]
    font_assets: dict[str, Any]
    mark_style: dict[str, Any]
    post_image_noise: dict[str, Any]


def resolve_mark_style(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    mark_count: int = 1,
) -> dict[str, Any]:
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


def _protected_mark_colors(mark_style: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    colors: list[tuple[int, int, int]] = []
    for key in ("mark_fill_rgb", "mark_outline_rgb"):
        value = mark_style.get(str(key))
        if value is not None:
            colors.append(tuple(int(channel) for channel in value))
    return tuple(colors)


def render_single_boxplot_scene(
    *,
    boxplots: Sequence[BoxPlotSpec],
    params: Mapping[str, Any],
    mark_style: Mapping[str, Any],
    instance_seed: int,
) -> BoxplotRenderArtifacts:
    """Render one boxplot canvas while leaving objective binding to public task files."""

    render_params = _render_params(params, mark_style, instance_seed=int(instance_seed))
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="boxplot",
        render_params=render_params,
        protected_colors=_protected_mark_colors(mark_style),
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_boxplot_scene(
            background,
            boxplots=boxplots,
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return BoxplotRenderArtifacts(
        rendered_scene=replace(rendered_scene, image=image),
        background_style={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        font_assets=chart_font_asset_metadata(str(chart_font_family)),
        mark_style=dict(mark_style),
        post_image_noise=dict(post_noise_meta),
    )


def render_paired_boxplot_panels(
    *,
    before_boxplots: Sequence[BoxPlotSpec],
    after_boxplots: Sequence[BoxPlotSpec],
    params: Mapping[str, Any],
    mark_style: Mapping[str, Any],
    before_title: str,
    after_title: str,
    instance_seed: int,
) -> BoxplotRenderArtifacts:
    """Render before/after boxplot panels with a shared value axis and labels."""

    render_params = _render_params(params, mark_style, instance_seed=int(instance_seed))
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="boxplot",
        render_params=render_params,
        protected_colors=_protected_mark_colors(mark_style),
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_paired_boxplot_scene(
            background,
            before_boxplots=before_boxplots,
            after_boxplots=after_boxplots,
            render_params=render_params,
            before_title=str(before_title),
            after_title=str(after_title),
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return BoxplotRenderArtifacts(
        rendered_scene=replace(rendered_scene, image=image),
        background_style={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        font_assets=chart_font_asset_metadata(str(chart_font_family)),
        mark_style=dict(mark_style),
        post_image_noise=dict(post_noise_meta),
    )


def point_map_for_labels(
    rendered_scene: RenderedChartScene,
    labels: Sequence[str],
) -> dict[str, list[float]]:
    projection = projected_mark_annotation(rendered_scene, [str(label) for label in labels])
    points = [list(point) for point in projection["pixel_point_set"]]
    return {
        str(label): [round(float(point[0]), 3), round(float(point[1]), 3)]
        for label, point in zip(labels, points)
    }


def box_bbox_map_for_labels(
    rendered_scene: RenderedChartScene,
    labels: Sequence[str],
) -> dict[str, list[float]]:
    """Return rendered IQR box bboxes for the requested labels."""

    requested = {str(label) for label in labels}
    return {
        str(mark["label"]): [round(float(value), 3) for value in mark["box_bbox_px"]]
        for mark in rendered_scene.mark_traces
        if str(mark.get("label")) in requested and "box_bbox_px" in mark
    }


def label_centers(rendered_scene: RenderedChartScene) -> dict[str, list[float]]:
    return {
        str(mark["label"]): list(mark["label_center_px"])
        for mark in rendered_scene.mark_traces
    }


def render_trace_sections(artifacts: BoxplotRenderArtifacts) -> tuple[dict[str, Any], dict[str, Any]]:
    rendered_scene = artifacts.rendered_scene
    render_spec = {
        "canvas_width": int(rendered_scene.image.size[0]),
        "canvas_height": int(rendered_scene.image.size[1]),
        "coord_space": "pixel",
        "scene_variant": SCENE_VARIANT,
        "background_style": dict(artifacts.background_style),
        "information_scene_style": dict(artifacts.background_style["information_scene_style"]),
        "font_assets": dict(artifacts.font_assets),
        "post_image_noise": dict(artifacts.post_image_noise),
        "text_style": {
            "label_font_size_px": int(RENDERING_DEFAULTS.get("label_font_size_px", 22)),
            "tick_font_size_px": int(RENDERING_DEFAULTS.get("tick_font_size_px", 18)),
            "label_stroke_width_px": int(RENDERING_DEFAULTS.get("label_stroke_width_px", 2)),
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
        "box_bboxes_px": box_bbox_map_for_labels(
            rendered_scene,
            [str(mark["label"]) for mark in rendered_scene.mark_traces],
        ),
    }
    return render_spec, render_map


def build_trace_scaffold(
    *,
    artifacts: BoxplotRenderArtifacts,
    relations: Mapping[str, Any],
    question_format: str,
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    render_spec, render_map = render_trace_sections(artifacts)
    rendered_scene = artifacts.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": "chart_boxplot_distribution",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": dict(relations),
        },
        "render_spec": render_spec,
        "render_map": render_map,
        "execution_trace": {
            "question_format": str(question_format),
            "labels": [str(mark["label"]) for mark in rendered_scene.mark_traces],
            **dict(relations),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


__all__ = [
    "BoxplotRenderArtifacts",
    "box_bbox_map_for_labels",
    "build_trace_scaffold",
    "point_map_for_labels",
    "render_paired_boxplot_panels",
    "render_single_boxplot_scene",
    "resolve_mark_style",
]
