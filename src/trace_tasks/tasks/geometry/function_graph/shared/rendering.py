"""Rendering helpers for function-graph scenes."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from PIL import Image

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame
from trace_tasks.tasks.geometry.shared.labeled_point_annotation import (
    empty_graph_point_set_annotation_artifacts,
    graph_point_set_annotation_artifacts,
)
from trace_tasks.tasks.geometry.shared.shape_style import extract_background_anchor_colors, sample_geometry_shape_style
from trace_tasks.tasks.geometry.shared.single_object_scene import (
    GraphSceneContext,
    finalize_graph_scene_image,
    make_graph_scene_canvas,
    resolve_graph_scene_context,
)

from .defaults import DEFAULTS, POST_IMAGE_BACKGROUND_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, RENDER_DEFAULTS
from .projection import draw_function_polyline, graph_units_to_pixel_float
from .state import GraphPoint, GraphPolylinePoint, RenderedFunctionGraph, SampledFunctionGraph


def graph_context_and_canvas(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[GraphSceneContext, Image.Image, Any, Mapping[str, Any]]:
    """Create a graph-paper canvas using this scene's visual defaults."""

    context = resolve_graph_scene_context(
        rng,
        instance_seed=int(instance_seed),
        scene_id="function_graph",
        params=params,
        render_defaults=RENDER_DEFAULTS,
        background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
        fallback_canvas_min=DEFAULTS.canvas_size_min,
        fallback_canvas_max=DEFAULTS.canvas_size_max,
        fallback_cells_min=DEFAULTS.graph_cells_min,
        fallback_cells_max=DEFAULTS.graph_cells_max,
        require_graph_paper_background=True,
        graph_style_overrides={
            "origin_fraction_x": 0.5,
            "origin_fraction_y": 0.5,
            "axis_scale_label_max_abs": 8,
            "origin_label_enabled": True,
        },
    )
    image, draw, background_meta = make_graph_scene_canvas(
        instance_seed=int(instance_seed),
        context=context,
        background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
        require_graph_paper=True,
    )
    return context, image, draw, background_meta


def style_and_widths(rng, *, params: Mapping[str, Any], context: GraphSceneContext, background_meta: Mapping[str, Any]):
    """Resolve line widths and contrast-aware shape style."""

    line_width = int(params.get("line_width", group_default(RENDER_DEFAULTS, "line_width", DEFAULTS.line_width)))
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=RENDER_DEFAULTS,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    return (
        int(line_width) * int(context.scene_scale),
        shape_style,
    )


def pixel_point(point: GraphPoint | GraphPolylinePoint, *, context: GraphSceneContext) -> Tuple[float, float]:
    """Project one graph coordinate into canonical pixel coordinates."""

    return graph_units_to_pixel_float(
        point,
        graph_origin=context.graph_origin,
        graph_spacing=int(context.graph_spacing),
    )


def render_count_graph(
    draw,
    *,
    context: GraphSceneContext,
    sampled_scene: SampledFunctionGraph,
    shape_style,
    line_width: int,
) -> RenderedFunctionGraph:
    """Draw a sampled function graph and bind its point-set annotation."""

    render_map = dict(sampled_scene.render_map)
    render_polyline = draw_function_polyline(
        draw,
        polyline_graph=sampled_scene.polyline_graph,
        graph_origin=context.graph_origin,
        graph_spacing=int(context.graph_spacing),
        scene_scale=int(context.scene_scale),
        line_width=int(line_width),
        line_color=shape_style.line_color,
    )
    render_map["function_polyline_pixel"] = [
        [round(float(pixel_point(point, context=context)[0]), 3), round(float(pixel_point(point, context=context)[1]), 3)]
        for point in sampled_scene.polyline_graph
    ]
    render_map["function_polyline_render"] = [
        [round(float(point[0]), 3), round(float(point[1]), 3)]
        for point in render_polyline
    ]

    points_by_label = {
        f"point_{index + 1}": pixel_point(point, context=context)
        for index, point in enumerate(sampled_scene.annotation_graph_points)
    }
    annotation = (
        graph_point_set_annotation_artifacts(
            points_by_label=points_by_label,
            graph_origin=context.graph_origin,
            graph_spacing=int(context.graph_spacing),
            witness_type="function_graph_feature_points",
            ordered_labels=tuple(points_by_label.keys()),
        )
        if points_by_label
        else empty_graph_point_set_annotation_artifacts(witness_type="function_graph_feature_points")
    )
    return RenderedFunctionGraph(
        answer_value=int(len(sampled_scene.annotation_graph_points)),
        annotation_type=str(annotation["annotation_type"]),
        annotation_value=[list(point) for point in annotation["annotation_value"]],
        projected_annotation=dict(annotation["projected_annotation"]),
        witness_symbolic=dict(annotation["witness_symbolic"]),
        required_annotation_labels=list(annotation["required_labels"]),
        scene_entities=list(sampled_scene.scene_entities),
        render_map=dict(render_map),
        execution_trace=dict(sampled_scene.execution_trace),
        object_count=int(sampled_scene.object_count),
    )


def finalize_graph_image(
    image: Image.Image,
    *,
    instance_seed: int,
    context: GraphSceneContext,
    background_meta: Mapping[str, Any],
) -> Tuple[Image.Image, Mapping[str, Any], Mapping[str, Any]]:
    """Apply final graph-scene postprocessing and noise."""

    return finalize_graph_scene_image(
        image,
        instance_seed=int(instance_seed),
        context=context,
        background_meta=background_meta,
        noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
    )


def graph_render_spec(
    *,
    context: GraphSceneContext,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    shape_style,
    family: str,
) -> dict[str, Any]:
    """Build render metadata common to function-graph tasks."""

    return {
        "canvas_size": int(context.canvas_size),
        "coord_space": "pixel",
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "shape_style": dict(shape_style.to_trace_dict()),
        "graph_coordinate_frame": dict(context.graph_frame),
        "graph_paper_grid": graph_paper_grid_from_frame(context.graph_frame),
        **dict(context.graph_layout_metadata),
        "scene_variant": str(family),
    }
