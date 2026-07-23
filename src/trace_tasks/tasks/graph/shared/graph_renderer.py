"""Orchestration for rendering node-link graph scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.font_assets import font_asset_version, get_font_family_record
from ...shared.text_legibility import draw_centered_readable_text, resolve_readable_text_style
from ...shared.visual_style.information_scene import (
    information_scene_style_from_metadata,
    make_information_scene_background,
)
from .graph_sample_types import GraphTopologySample
from .graph_render_context import (
    _apply_content_layout_jitter,
    _draw_context_text_blocks,
    _draw_context_text_chips,
)
from .graph_render_edges import (
    _draw_edge,
    _draw_edge_boxed_label,
    _draw_edge_weight_label,
    _resolve_edge_boxed_label_box,
    _resolve_edge_route_controls,
    _resolve_edge_weight_label_box,
)
from .graph_render_geometry import _count_edge_crossings
from .graph_render_layout import _resolve_positions
from .graph_render_nodes import _draw_node_shape, _resolve_effective_node_radius_px, _resolve_node_label_font
from .graph_render_panel import _draw_panel_chrome, _resolve_panel_geometry
from .graph_render_types import (
    BBox,
    GraphRenderParams,
    Point,
    RenderedGraphEdge,
    RenderedGraphNode,
    RenderedGraphScene,
    SUPPORTED_EDGE_ROUTING_VARIANTS,
)


def render_graph_scene(
    *,
    graph_sample: GraphTopologySample,
    layout_variant: str,
    layout_transform_variant: str,
    render_params: GraphRenderParams,
    layout_seed: int,
    scene_title: str = "Graph",
    directed: bool = False,
    base_image: Image.Image | None = None,
    edge_weights_by_label: Mapping[Tuple[str, str], int] | None = None,
    edge_weight_label_font_size_px: int | None = None,
    edge_weight_label_offset_px: int = 12,
    edge_weight_label_padding_px: int = 4,
    edge_text_labels_by_label: Mapping[Tuple[str, str], str] | None = None,
    edge_text_label_font_size_px: int | None = None,
    edge_text_label_offset_px: int = 12,
    edge_text_label_padding_px: int = 5,
    edge_text_label_strict_placement: bool = False,
    node_style_by_label: Mapping[str, Mapping[str, Any]] | None = None,
    edge_style_by_label: Mapping[Tuple[str, str], Mapping[str, Any]] | None = None,
    layout_fallback_variants: Sequence[str] | None = None,
) -> RenderedGraphScene:
    """Render one labeled single-panel node-link graph scene."""

    panel_geometry = _resolve_panel_geometry(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        outer_margin_px=int(render_params.outer_margin_px),
        panel_padding_px=int(render_params.panel_padding_px),
        title_font_size_px=int(render_params.panel_title_font_size_px),
    )
    if isinstance(render_params.information_scene_style, Mapping):
        panel_geometry["information_scene_style"] = dict(render_params.information_scene_style)
    if isinstance(render_params.text_legibility, Mapping):
        panel_geometry["text_legibility"] = dict(render_params.text_legibility)
    panel_geometry["font_family"] = str(render_params.font_family or "")
    panel_geometry["font_asset"] = (
        dict(render_params.font_asset)
        if isinstance(render_params.font_asset, Mapping)
        else dict(get_font_family_record(str(render_params.font_family)).to_trace())
        if str(render_params.font_family or "").strip()
        else {}
    )
    panel_geometry["font_asset_version"] = str(render_params.font_asset_version or font_asset_version())
    panel_geometry["font_exclusion_reason"] = str(render_params.font_exclusion_reason)
    information_style_meta = render_params.information_scene_style if isinstance(render_params.information_scene_style, Mapping) else None
    information_background_meta: Dict[str, Any] | None = None
    if information_style_meta is not None:
        try:
            information_style = information_scene_style_from_metadata(information_style_meta)
            image, information_background_meta = make_information_scene_background(
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
                style=information_style,
                instance_seed=int(layout_seed),
                namespace="graph.node_link.information_scene_background",
            )
            fill_background = False
        except Exception:
            if base_image is None:
                image = Image.new("RGB", (int(render_params.canvas_width), int(render_params.canvas_height)))
                fill_background = True
            else:
                image = base_image.convert("RGB").copy()
                fill_background = False
    elif base_image is None:
        image = Image.new("RGB", (int(render_params.canvas_width), int(render_params.canvas_height)))
        fill_background = True
    else:
        image = base_image.convert("RGB").copy()
        fill_background = False
    if information_background_meta is not None:
        panel_geometry["information_scene_background_meta"] = dict(information_background_meta)
    _draw_panel_chrome(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        scene_title=str(scene_title),
        fill_background=bool(fill_background),
        layout_seed=int(layout_seed),
    )
    block_context_elements: List[Dict[str, Any]] = list(
        _draw_context_text_blocks(
            image,
            panel_geometry=panel_geometry,
            render_params=render_params,
            layout_seed=int(layout_seed),
        )
    )
    chip_context_elements = _draw_context_text_chips(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )
    panel_context_elements = list(panel_geometry.get("context_text_elements", []))
    panel_context_elements.extend([dict(element) for element in block_context_elements])
    panel_context_elements.extend([dict(element) for element in chip_context_elements])
    if panel_context_elements:
        panel_geometry["context_text_elements"] = [dict(element) for element in panel_context_elements]
    draw = ImageDraw.Draw(image)
    _apply_content_layout_jitter(
        panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )
    content_bbox = tuple(int(value) for value in panel_geometry["scene_content_xyxy"])
    effective_node_radius_px = _resolve_effective_node_radius_px(
        node_labels=tuple(str(label) for label in graph_sample.node_labels),
        render_params=render_params,
    )
    effective_render_params = replace(render_params, node_radius_px=int(effective_node_radius_px))
    panel_geometry["effective_node_radius_px"] = int(effective_node_radius_px)
    positions, actual_layout_variant, actual_layout_transform_variant = _resolve_positions(
        graph_sample,
        layout_variant=str(layout_variant),
        layout_transform_variant=str(layout_transform_variant),
        content_bbox=content_bbox,
        node_radius_px=int(effective_render_params.node_radius_px),
        layout_seed=int(layout_seed),
        layout_fallback_variants=layout_fallback_variants,
    )

    label_to_node = {str(label): int(node) for node, label in zip(graph_sample.graph.nodes(), graph_sample.node_labels)}
    edge_weight_lookup = {
        (str(left), str(right)): int(weight)
        for (left, right), weight in (edge_weights_by_label or {}).items()
    }
    edge_text_label_lookup = {
        (str(left), str(right)): str(text)
        for (left, right), text in (edge_text_labels_by_label or {}).items()
    }
    actual_edge_routing_variant = (
        str(render_params.edge_routing_variant)
        if str(render_params.edge_routing_variant) in SUPPORTED_EDGE_ROUTING_VARIANTS
        else "straight"
    )
    node_bbox_lookup = {
        str(label): (
            int(positions[int(node)][0] - int(effective_render_params.node_radius_px)),
            int(positions[int(node)][1] - int(effective_render_params.node_radius_px)),
            int(positions[int(node)][0] + int(effective_render_params.node_radius_px)),
            int(positions[int(node)][1] + int(effective_render_params.node_radius_px)),
        )
        for node, label in zip(graph_sample.graph.nodes(), graph_sample.node_labels)
    }
    edge_route_controls = _resolve_edge_route_controls(
        edge_labels=tuple((str(left), str(right)) for left, right in graph_sample.edge_labels),
        label_to_node=label_to_node,
        positions=positions,
        content_bbox=content_bbox,
        edge_routing_variant=str(actual_edge_routing_variant),
        node_radius_px=int(effective_render_params.node_radius_px),
    )
    edge_styles = {
        (str(left), str(right)): dict(style)
        for (left, right), style in (edge_style_by_label or {}).items()
    }
    edge_segments: List[Tuple[Tuple[str, str], Tuple[Point, Point]]] = []
    rendered_edges: List[RenderedGraphEdge] = []
    for left_label, right_label in graph_sample.edge_labels:
        left_node = int(label_to_node[str(left_label)])
        right_node = int(label_to_node[str(right_label)])
        start = tuple(int(value) for value in positions[left_node])
        end = tuple(int(value) for value in positions[right_node])
        control = edge_route_controls.get((str(left_label), str(right_label)))
        edge_style = edge_styles.get((str(left_label), str(right_label)), {})
        edge_color_rgb = tuple(int(v) for v in edge_style.get("edge_color_rgb", render_params.edge_color_rgb))
        edge_width_px = int(edge_style.get("edge_width_px", render_params.edge_width_px))
        segment = _draw_edge(
            draw,
            start=start,
            end=end,
            control=control,
            node_radius_px=int(effective_render_params.node_radius_px),
            edge_width_px=max(1, int(edge_width_px)),
            edge_color_rgb=edge_color_rgb,
            directed=bool(directed),
            arrow_length_px=int(render_params.arrow_length_px),
            arrow_width_px=int(render_params.arrow_width_px),
        )
        edge_segments.append(((str(left_label), str(right_label)), segment))
        rendered_edges.append(
            RenderedGraphEdge(
                edge_id=f"edge_{str(left_label)}_{str(right_label)}",
                node_u_label=str(left_label),
                node_v_label=str(right_label),
                directed=bool(directed),
                segment_px=segment,
                route_variant="arc" if control is not None else "straight",
                control_px=tuple(int(value) for value in control) if control is not None else None,
                color_name=str(edge_style["color_name"]) if edge_style.get("color_name") is not None else None,
                edge_color_rgb=tuple(int(value) for value in edge_color_rgb),
            )
        )

    reserved_edge_label_boxes: List[BBox] = []
    labeled_edges: List[RenderedGraphEdge] = []
    all_segments = [tuple(segment) for _, segment in edge_segments]
    for edge in rendered_edges:
        edge_text_label = edge_text_label_lookup.get((str(edge.node_u_label), str(edge.node_v_label)))
        edge_text_label_bbox = None
        if edge_text_label is not None:
            side_seed = sum(ord(char) for char in f"{edge.node_u_label}-{edge.node_v_label}-label")
            edge_text_label_bbox = _resolve_edge_boxed_label_box(
                draw,
                segment=tuple(edge.segment_px),
                text=str(edge_text_label),
                font_size_px=int(
                    edge_text_label_font_size_px
                    if edge_text_label_font_size_px is not None
                    else max(13, int(render_params.label_font_size_px) - 4)
                ),
                font_family=str(render_params.font_family or ""),
                offset_px=int(edge_text_label_offset_px),
                padding_px=int(edge_text_label_padding_px),
                content_bbox=content_bbox,
                other_segments=tuple(
                    segment for segment in all_segments if tuple(segment) != tuple(edge.segment_px)
                ),
                reserved_boxes=tuple(reserved_edge_label_boxes),
                node_bboxes=tuple(node_bbox_lookup.values()),
                side_seed=int(side_seed),
                require_strict=bool(edge_text_label_strict_placement),
            )
            reserved_edge_label_boxes.append(tuple(int(value) for value in edge_text_label_bbox))
            _draw_edge_boxed_label(
                draw,
                box=tuple(int(value) for value in edge_text_label_bbox),
                text=str(edge_text_label),
                font_size_px=int(
                    edge_text_label_font_size_px
                    if edge_text_label_font_size_px is not None
                    else max(13, int(render_params.label_font_size_px) - 4)
                ),
                font_family=str(render_params.font_family or ""),
                box_fill_rgb=tuple(int(v) for v in render_params.panel_fill_rgb),
                box_border_rgb=tuple(int(v) for v in render_params.panel_border_rgb),
                text_rgb=tuple(int(v) for v in render_params.title_color_rgb),
                layout_seed=int(layout_seed),
            )

        edge_weight = edge_weight_lookup.get((str(edge.node_u_label), str(edge.node_v_label)))
        weight_label_bbox = None
        if edge_weight is not None:
            side_seed = sum(ord(char) for char in f"{edge.node_u_label}-{edge.node_v_label}")
            weight_label_bbox = _resolve_edge_weight_label_box(
                draw,
                segment=tuple(edge.segment_px),
                weight=int(edge_weight),
                font_size_px=int(
                    edge_weight_label_font_size_px
                    if edge_weight_label_font_size_px is not None
                    else max(12, int(render_params.label_font_size_px) - 4)
                ),
                font_family=str(render_params.font_family or ""),
                offset_px=int(edge_weight_label_offset_px),
                padding_px=int(edge_weight_label_padding_px),
                content_bbox=content_bbox,
                other_segments=tuple(
                    segment for segment in all_segments if tuple(segment) != tuple(edge.segment_px)
                ),
                reserved_boxes=tuple(reserved_edge_label_boxes),
                node_bboxes=tuple(node_bbox_lookup.values()),
                side_seed=int(side_seed),
            )
            reserved_edge_label_boxes.append(tuple(int(value) for value in weight_label_bbox))
            _draw_edge_weight_label(
                draw,
                box=tuple(int(value) for value in weight_label_bbox),
                weight=int(edge_weight),
                font_size_px=int(
                    edge_weight_label_font_size_px
                    if edge_weight_label_font_size_px is not None
                    else max(12, int(render_params.label_font_size_px) - 4)
                ),
                font_family=str(render_params.font_family or ""),
                box_fill_rgb=tuple(int(v) for v in render_params.panel_fill_rgb),
                box_border_rgb=tuple(int(v) for v in render_params.panel_border_rgb),
                text_rgb=tuple(int(v) for v in render_params.title_color_rgb),
                layout_seed=int(layout_seed),
            )
        labeled_edges.append(
            RenderedGraphEdge(
                edge_id=str(edge.edge_id),
                node_u_label=str(edge.node_u_label),
                node_v_label=str(edge.node_v_label),
                directed=bool(edge.directed),
                segment_px=tuple(edge.segment_px),
                route_variant=str(edge.route_variant),
                control_px=tuple(int(value) for value in edge.control_px) if edge.control_px is not None else None,
                weight=int(edge_weight) if edge_weight is not None else None,
                weight_label_bbox_xyxy=tuple(int(value) for value in weight_label_bbox) if weight_label_bbox is not None else None,
                edge_label=str(edge_text_label) if edge_text_label is not None else None,
                edge_label_bbox_xyxy=tuple(int(value) for value in edge_text_label_bbox) if edge_text_label_bbox is not None else None,
                color_name=str(edge.color_name) if edge.color_name is not None else None,
                edge_color_rgb=tuple(int(value) for value in edge.edge_color_rgb) if edge.edge_color_rgb is not None else None,
            )
        )
    rendered_edges = labeled_edges

    label_font, label_stroke_width = _resolve_node_label_font(
        draw,
        node_labels=tuple(str(label) for label in graph_sample.node_labels),
        render_params=effective_render_params,
    )
    rendered_nodes: List[RenderedGraphNode] = []
    radius = int(effective_render_params.node_radius_px)
    node_styles = {str(key): dict(value) for key, value in (node_style_by_label or {}).items()}
    for node, label in zip(graph_sample.graph.nodes(), graph_sample.node_labels):
        center = tuple(int(value) for value in positions[int(node)])
        node_style = node_styles.get(str(label), {})
        fill_rgb = tuple(int(v) for v in node_style.get("fill_rgb", render_params.node_fill_rgb))
        border_rgb = tuple(int(v) for v in node_style.get("border_rgb", render_params.node_border_rgb))
        label_text_rgb = tuple(int(v) for v in node_style.get("label_text_rgb", render_params.label_text_rgb))
        label_stroke_rgb = tuple(int(v) for v in node_style.get("label_stroke_rgb", render_params.label_stroke_rgb))
        label_style = resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.node_link.node_label_text.{str(label)}",
            role="graph_node_label_text",
            surface_rgbs=(fill_rgb,),
            preferred_rgbs=(label_text_rgb, label_stroke_rgb, (255, 255, 255), (10, 14, 22)),
            min_contrast_ratio=4.0,
            min_lab_distance=24.0,
        )
        label_text_rgb = tuple(int(value) for value in label_style.fill_rgb)
        label_stroke_rgb = tuple(int(value) for value in label_style.stroke_rgb)
        if node_style.get("halo_rgb") is not None:
            halo_pad = int(node_style.get("halo_pad_px", 6))
            halo_width = int(node_style.get("halo_width_px", 3))
            halo_bbox = (
                int(center[0] - radius - halo_pad),
                int(center[1] - radius - halo_pad),
                int(center[0] + radius + halo_pad),
                int(center[1] + radius + halo_pad),
            )
            draw.ellipse(
                halo_bbox,
                outline=tuple(int(v) for v in node_style.get("halo_rgb", render_params.title_color_rgb)),
                width=max(1, int(halo_width)),
            )
        bbox = _draw_node_shape(
            draw,
            center=center,
            radius=int(radius),
            node_shape_variant=str(render_params.node_shape_variant),
            fill_rgb=fill_rgb,
            outline_rgb=border_rgb,
            outline_width=int(render_params.node_border_width_px),
        )
        draw_centered_readable_text(
            draw,
            text=str(label),
            center=(float(center[0]), float(center[1])),
            font=label_font,
            style=label_style,
            stroke_width=int(label_stroke_width),
        )
        rendered_nodes.append(
            RenderedGraphNode(
                label=str(label),
                degree=int(graph_sample.degrees_by_label[str(label)]),
                center_xy=center,
                bbox_xyxy=tuple(int(value) for value in bbox),
                neighbors=tuple(str(value) for value in graph_sample.adjacency_by_label[str(label)]),
                successors=tuple(str(value) for value in graph_sample.successors_by_label[str(label)]),
                predecessors=tuple(str(value) for value in graph_sample.predecessors_by_label[str(label)]),
                color_name=str(node_style["color_name"]) if node_style.get("color_name") is not None else None,
                fill_rgb=tuple(int(value) for value in fill_rgb),
                border_rgb=tuple(int(value) for value in border_rgb),
                label_text_rgb=tuple(int(value) for value in label_text_rgb),
                label_stroke_rgb=tuple(int(value) for value in label_stroke_rgb),
            )
        )

    return RenderedGraphScene(
        image=image,
        panel_geometry={str(key): value for key, value in panel_geometry.items()},
        nodes=tuple(rendered_nodes),
        edges=tuple(rendered_edges),
        layout_variant=str(actual_layout_variant),
        layout_transform_variant=str(actual_layout_transform_variant),
        edge_routing_variant=str(actual_edge_routing_variant),
        crossing_count=_count_edge_crossings(edge_segments),
        resolved_label_font_size_px=int(getattr(label_font, "size", int(render_params.label_font_size_px))),
        resolved_label_stroke_width_px=int(label_stroke_width),
    )


__all__ = [
    "render_graph_scene",
]
