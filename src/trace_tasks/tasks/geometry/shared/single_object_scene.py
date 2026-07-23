"""Reusable scene setup helpers for single-object geometry tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...shared.geometry_primitives import Point
from .diagram_style import (
    GEOMETRY_STYLE_PROFILE_COORDINATE_GRID,
    GeometryDiagramStyle,
    geometry_graph_style_from_diagram_style,
    make_geometry_diagram_background,
    resolve_geometry_diagram_style,
)
from .graph_panel_layout import (
    GraphPanelLayout,
    graph_coordinate_frame_from_panel_layout,
    resolve_graph_panel_layout,
)
from .graph_paper import resolve_graph_cells_per_side, resolve_square_canvas_size
from .graph_rendering import (
    FALLBACK_GRAPH_STYLE,
    enforce_graph_paper_background,
    resolve_graph_style_from_params,
    scaled_graph_style_for_scene,
)


@dataclass(frozen=True)
class GraphSceneContext:
    """Resolved render/grid context for one single-object geometry scene."""

    canvas_size: int
    graph_cells: int
    graph_spacing: int
    graph_frame: Dict[str, Any]
    graph_origin: Point
    graph_style: Dict[str, Any]
    graph_panel_layout: GraphPanelLayout
    graph_layout_metadata: Dict[str, Any]
    diagram_style: GeometryDiagramStyle
    diagram_style_meta: Dict[str, Any]
    scene_scale: int
    render_params: Dict[str, Any]
    style_scene_id: str


_GRAPH_STYLE_CONTROL_KEYS: Tuple[str, ...] = (
    "origin_fraction_x",
    "origin_fraction_y",
    "axis_enabled",
    "axis_arrows_enabled",
    "center_point_enabled",
    "axis_scale_labels_enabled",
    "axis_scale_label_max_abs",
    "origin_label_enabled",
    "origin_label_text",
    "scene_supersample_scale",
    "supersample_scale",
)


def resolve_graph_scene_context(
    rng,
    *,
    instance_seed: int | None = None,
    scene_id: str = "graph_paper_panel",
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    fallback_canvas_min: int,
    fallback_canvas_max: int,
    fallback_cells_min: int,
    fallback_cells_max: int,
    require_graph_paper_background: bool = True,
    graph_style_overrides: Mapping[str, Any] | None = None,
) -> GraphSceneContext:
    """Resolve deterministic graph-space scene parameters for one instance."""
    canvas_size = resolve_square_canvas_size(
        rng,
        params=params,
        render_defaults=render_defaults,
        fallback_min=int(fallback_canvas_min),
        fallback_max=int(fallback_canvas_max),
    )
    graph_cells = resolve_graph_cells_per_side(
        rng,
        params=params,
        render_defaults=render_defaults,
        canvas_size=int(canvas_size),
        fallback_min=int(fallback_cells_min),
        fallback_max=int(fallback_cells_max),
        min_spacing_px=4,
    )
    base_graph_style = resolve_graph_style_from_params(
        params,
        default_background_config=background_defaults,
        fallback_style=FALLBACK_GRAPH_STYLE,
    )
    scene_scale = int(max(1, int(base_graph_style.get("scene_supersample_scale", 1))))
    graph_style = dict(base_graph_style)
    if isinstance(graph_style_overrides, Mapping):
        graph_style.update(dict(graph_style_overrides))
    diagram_style, diagram_style_meta = resolve_geometry_diagram_style(
        instance_seed=int(instance_seed or 0),
        params=params,
        scene_id=str(scene_id),
        require_grid=True,
        style_profile=GEOMETRY_STYLE_PROFILE_COORDINATE_GRID,
    )
    graph_panel_layout = resolve_graph_panel_layout(
        canvas_width=int(canvas_size),
        canvas_height=int(canvas_size),
        graph_cells=int(graph_cells),
        params=params,
        render_defaults=render_defaults,
        min_spacing_px=4,
        origin_fraction_x=float(graph_style.get("origin_fraction_x", 0.5)),
        origin_fraction_y=float(graph_style.get("origin_fraction_y", 0.5)),
    )
    graph_spacing = int(graph_panel_layout.graph_spacing)
    graph_controls = {
        str(key): graph_style[str(key)]
        for key in _GRAPH_STYLE_CONTROL_KEYS
        if str(key) in graph_style
    }
    graph_style = geometry_graph_style_from_diagram_style(
        diagram_style,
        spacing_px=int(graph_spacing),
        outer_margin_px=0,
        axis_enabled=bool(graph_controls.get("axis_enabled", True)),
    )
    graph_style.update(dict(graph_controls))
    graph_style["spacing"] = int(graph_spacing)
    graph_style["outer_margin_px"] = 0
    graph_style["origin_pixel"] = [
        int(graph_panel_layout.local_graph_origin_px[0]),
        int(graph_panel_layout.local_graph_origin_px[1]),
    ]
    graph_frame = graph_coordinate_frame_from_panel_layout(
        graph_panel_layout,
        origin_fraction_x=float(graph_style.get("origin_fraction_x", 0.5)),
        origin_fraction_y=float(graph_style.get("origin_fraction_y", 0.5)),
    )
    graph_origin = (
        float(graph_frame["origin_pixel"][0]),
        float(graph_frame["origin_pixel"][1]),
    )
    render_graph_style = scaled_graph_style_for_scene(graph_style, scene_scale=int(scene_scale))
    if bool(require_graph_paper_background):
        render_params = enforce_graph_paper_background(params, graph_style=render_graph_style)
    else:
        render_params = dict(params)
    return GraphSceneContext(
        canvas_size=int(canvas_size),
        graph_cells=int(graph_cells),
        graph_spacing=int(graph_spacing),
        graph_frame=dict(graph_frame),
        graph_origin=(float(graph_origin[0]), float(graph_origin[1])),
        graph_style=dict(graph_style),
        graph_panel_layout=graph_panel_layout,
        graph_layout_metadata={
            **dict(graph_panel_layout.to_metadata()),
            "technical_diagram_style": dict(diagram_style_meta),
        },
        diagram_style=diagram_style,
        diagram_style_meta=dict(diagram_style_meta),
        scene_scale=int(scene_scale),
        render_params=dict(render_params),
        style_scene_id=str(scene_id),
    )


def make_graph_scene_canvas(
    *,
    instance_seed: int,
    context: GraphSceneContext,
    background_defaults: Mapping[str, Any],
    require_graph_paper: bool = True,
) -> Tuple[Image.Image, ImageDraw.ImageDraw, Dict[str, Any]]:
    """Create one background image + draw handle for one resolved scene context."""
    render_canvas_size = int(context.canvas_size) * int(context.scene_scale)
    scale = int(context.scene_scale)
    layout = context.graph_panel_layout
    outer_image, outer_background_meta = make_geometry_diagram_background(
        canvas_width=int(context.canvas_size),
        canvas_height=int(context.canvas_size),
        style=context.diagram_style,
        instance_seed=int(instance_seed),
        namespace=f"geometry.{context.style_scene_id}.technical_diagram_background",
    )
    image = (
        outer_image.resize((int(render_canvas_size), int(render_canvas_size)), resample=Image.Resampling.BICUBIC)
        if int(scale) > 1
        else outer_image
    )
    draw = ImageDraw.Draw(image)

    panel_left, panel_top, panel_right, panel_bottom = layout.panel_bbox_px
    content_left, content_top, content_right, content_bottom = layout.content_bbox_px
    panel_bbox_render = [
        int(panel_left) * int(scale),
        int(panel_top) * int(scale),
        (int(panel_right) * int(scale)) - 1,
        (int(panel_bottom) * int(scale)) - 1,
    ]
    content_width = max(1, int(content_right) - int(content_left))
    content_height = max(1, int(content_bottom) - int(content_top))
    panel_fill = tuple(int(value) for value in context.graph_style.get("base_color", [255, 255, 255])[:3])
    frame_color = tuple(int(value) for value in context.graph_style.get("axis_color", [118, 128, 146])[:3])
    draw.rectangle(panel_bbox_render, fill=panel_fill)
    panel_render_params = enforce_graph_paper_background(
        context.render_params,
        graph_style=scaled_graph_style_for_scene(context.graph_style, scene_scale=int(scale)),
    )
    panel_image, panel_background_meta = make_background_canvas(
        canvas_width=int(content_width) * int(scale),
        canvas_height=int(content_height) * int(scale),
        instance_seed=int(instance_seed),
        params=dict(panel_render_params),
        default_config=background_defaults,
        fallback_color=(248, 248, 248),
    )
    if str(panel_background_meta.get("selected_style", "")) != "graph_paper":
        raise RuntimeError("bounded geometry graph panel must render a graph_paper content background")
    image.paste(panel_image, (int(content_left) * int(scale), int(content_top) * int(scale)))
    draw.rectangle(
        panel_bbox_render,
        outline=frame_color,
        width=max(1, int(layout.panel_frame_width_px) * int(scale)),
    )
    panel_style_spec = dict(panel_background_meta.get("style_spec", {}))
    composite_style_spec = {
        **panel_style_spec,
        "kind": "bounded_graph_paper",
        "canvas_color": list(context.diagram_style.canvas_rgb),
        "panel_fill_color": list(panel_fill),
        "panel_frame_color": list(frame_color),
        "panel_frame_width_px": int(layout.panel_frame_width_px),
        "panel_padding_px": int(layout.panel_padding_px),
        "graph_panel_bbox_px": [int(value) for value in layout.panel_bbox_px],
        "graph_content_bbox_px": [int(value) for value in layout.content_bbox_px],
        "graph_origin_px": [int(layout.graph_origin_px[0]), int(layout.graph_origin_px[1])],
        "graph_spacing_px": int(layout.graph_spacing),
        "technical_diagram_style_pack": str(context.diagram_style.style_pack),
        "geometry_style_profile": context.diagram_style_meta.get("geometry_style_profile"),
        "technical_profile": context.diagram_style_meta.get("technical_profile"),
        "available_theme_ids": list(context.diagram_style_meta.get("available_theme_ids", [])),
    }
    background_meta = {
        "enabled": True,
        "selected_style": "graph_paper_panel",
        "available_styles": ["graph_paper_panel"],
        "geometry_style_profile": context.diagram_style_meta.get("geometry_style_profile"),
        "technical_profile": context.diagram_style_meta.get("technical_profile"),
        "available_theme_ids": list(context.diagram_style_meta.get("available_theme_ids", [])),
        "style_spec": composite_style_spec,
        "outer_background_style": dict(outer_background_meta),
        "panel_background_style": dict(panel_background_meta),
        "technical_diagram_style_resolution": dict(context.diagram_style_meta),
    }
    if bool(require_graph_paper) and str(background_meta.get("selected_style", "")) not in {"graph_paper", "graph_paper_panel"}:
        raise RuntimeError("geometry measurement tasks must render on graph_paper backgrounds")
    return image, ImageDraw.Draw(image), dict(background_meta)


def finalize_graph_scene_image(
    image: Image.Image,
    *,
    instance_seed: int,
    context: GraphSceneContext,
    background_meta: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
    """Downscale supersampled render, attach metadata, and apply post-noise."""
    out_image = image
    if int(context.scene_scale) > 1:
        out_image = out_image.resize(
            (int(context.canvas_size), int(context.canvas_size)),
            resample=Image.Resampling.LANCZOS,
        )
    background = dict(background_meta)
    render_style_spec = background.get("style_spec", {})
    selected_style = str(background.get("selected_style", ""))
    if str(selected_style) in {"graph_paper", "graph_paper_panel"}:
        resolved_style_spec = dict(context.graph_style)
        if isinstance(render_style_spec, Mapping):
            for key in (
                "kind",
                "canvas_color",
                "panel_fill_color",
                "panel_frame_color",
                "panel_frame_width_px",
                "panel_padding_px",
                "graph_panel_bbox_px",
                "graph_content_bbox_px",
                "graph_origin_px",
                "graph_spacing_px",
                "technical_diagram_style_pack",
                "base_color",
                "line_color",
                "major_line_color",
                "axis_color",
                "center_point_color",
                "origin_label_color",
                "color_variation_enabled",
                "color_variation_applied",
                "color_variation_sampled",
                "base_color_jitter",
                "line_color_jitter",
                "major_line_darken_range",
                "axis_darken_range",
                "center_point_darken_extra_range",
                "origin_label_darken_extra_range",
                "style_variant",
                "style_variant_probabilities",
            ):
                if key in render_style_spec:
                    resolved_style_spec[key] = render_style_spec[key]
    else:
        resolved_style_spec = dict(render_style_spec) if isinstance(render_style_spec, Mapping) else {}
    background["style_spec"] = resolved_style_spec
    background["render_scale"] = int(context.scene_scale)
    out_image, noise_meta = apply_post_image_noise(
        out_image,
        instance_seed=int(instance_seed),
        params=dict(context.render_params),
        default_config=noise_defaults,
    )
    return out_image, background, dict(noise_meta)
