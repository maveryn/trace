"""Rendering and layout for graph binary-tree scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.font_assets import font_asset_version, get_font_family_record
from ....shared.text_legibility import draw_centered_readable_text, resolve_readable_text_style
from ....shared.text_rendering import fit_font_to_box, load_font
from ...shared.graph_scene import (
    GraphRenderParams,
    SUPPORTED_NODE_SHAPE_VARIANTS,
    apply_graph_content_layout_jitter,
    draw_graph_context_text_blocks,
    draw_graph_context_text_chips,
)
from .state import BinaryTreeSample, RenderedBinaryTreeEdge, RenderedBinaryTreeNode, RenderedBinaryTreeScene


def _resolve_panel_geometry(render_params: GraphRenderParams) -> Dict[str, Any]:
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    panel = (margin, margin, width - margin, height - margin)
    title_band_height = max(42, int(round(float(render_params.panel_title_font_size_px) * 1.85)))
    title_band = (panel[0], panel[1], panel[2], panel[1] + title_band_height)
    content = (
        panel[0] + int(render_params.panel_padding_px),
        title_band[3] + max(14, int(render_params.panel_padding_px // 2)),
        panel[2] - int(render_params.panel_padding_px),
        panel[3] - int(render_params.panel_padding_px),
    )
    return {
        "canvas_size": [width, height],
        "panel_xyxy": [int(value) for value in panel],
        "scene_panel_xyxy": [int(value) for value in panel],
        "title_band_xyxy": [int(value) for value in title_band],
        "scene_content_xyxy": [int(value) for value in content],
    }


def _draw_node_shape(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[int, int],
    radius: int,
    shape_variant: str,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    outline_width: int,
) -> Tuple[int, int, int, int]:
    """Draw one node marker and return its pixel bbox for projection."""

    bbox = (
        int(center[0] - radius),
        int(center[1] - radius),
        int(center[0] + radius),
        int(center[1] + radius),
    )
    fill = tuple(int(value) for value in fill_rgb)
    outline = tuple(int(value) for value in outline_rgb)
    if str(shape_variant) == "rounded_square":
        draw.rounded_rectangle(
            bbox,
            radius=max(5, int(round(float(radius) * 0.32))),
            fill=fill,
            outline=outline,
            width=max(1, int(outline_width)),
        )
    elif str(shape_variant) == "hexagon":
        points = []
        for index in range(6):
            angle = math.radians(30.0 + (60.0 * float(index)))
            points.append(
                (
                    int(round(float(center[0]) + (float(radius) * math.cos(angle)))),
                    int(round(float(center[1]) + (float(radius) * math.sin(angle)))),
                )
            )
        draw.polygon(points, fill=fill, outline=outline)
        if int(outline_width) > 1:
            draw.line(points + [points[0]], fill=outline, width=max(1, int(outline_width)))
    else:
        draw.ellipse(bbox, fill=fill, outline=outline, width=max(1, int(outline_width)))
    return bbox


def _trim_segment(
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    trim_px: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return (tuple(start), tuple(end))
    ux = dx / length
    uy = dy / length
    return (
        (int(round(float(start[0]) + (ux * float(trim_px)))), int(round(float(start[1]) + (uy * float(trim_px))))),
        (int(round(float(end[0]) - (ux * float(trim_px)))), int(round(float(end[1]) - (uy * float(trim_px))))),
    )


def _connector_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    *,
    connector_style_variant: str,
    trim_px: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return a rendered parent-child connector path outside node interiors."""

    if str(connector_style_variant) != "elbow_edges":
        segment = _trim_segment(start, end, trim_px=max(1, int(trim_px)))
        return (tuple(segment[0]), tuple(segment[1]))

    direction = 1 if int(end[1]) >= int(start[1]) else -1
    start_trim = (int(start[0]), int(start[1] + (direction * max(1, int(trim_px)))))
    end_trim = (int(end[0]), int(end[1] - (direction * max(1, int(trim_px)))))
    mid_y = int(round(0.5 * float(start_trim[1] + end_trim[1])))
    return (
        tuple(start_trim),
        (int(start_trim[0]), int(mid_y)),
        (int(end_trim[0]), int(mid_y)),
        tuple(end_trim),
    )


def _connector_style_for_scene(scene_variant: str) -> str:
    """Use worksheet-style elbow connectors on paper trees, diagonal elsewhere."""

    return "elbow_edges" if str(scene_variant) == "paper_tree" else "diagonal_edges"


def _binary_tree_positions(
    sample: BinaryTreeSample,
    *,
    content_bbox: Sequence[int],
    node_radius_px: int,
) -> Dict[str, Tuple[int, int]]:
    node_by_id = {str(node.node_id): node for node in sample.nodes}

    def inorder_ids(node_id: str) -> List[str]:
        node = node_by_id[str(node_id)]
        result: List[str] = []
        if node.left_id is not None:
            result.extend(inorder_ids(str(node.left_id)))
        result.append(str(node_id))
        if node.right_id is not None:
            result.extend(inorder_ids(str(node.right_id)))
        return result

    ordered = inorder_ids(sample.root_id)
    x0, y0, x1, y1 = (int(value) for value in content_bbox)
    height = max(1, int(y1 - y0))
    usable_x0 = x0 + max(int(node_radius_px) + 10, 18)
    usable_x1 = x1 - max(int(node_radius_px) + 10, 18)
    if len(ordered) <= 1:
        x_by_id = {ordered[0]: int(round(0.5 * float(usable_x0 + usable_x1)))}
    else:
        step = float(max(1, usable_x1 - usable_x0)) / float(len(ordered) - 1)
        x_by_id = {node_id: int(round(float(usable_x0) + (float(index) * step))) for index, node_id in enumerate(ordered)}
    level_count = max(1, int(sample.max_depth) + 1)
    if level_count <= 1:
        y_by_depth = {0: int(round(0.5 * float(y0 + y1)))}
    else:
        step_y = float(max(1, height - (2 * int(node_radius_px)) - 16)) / float(level_count - 1)
        top_y = y0 + int(node_radius_px) + 10
        y_by_depth = {depth: int(round(float(top_y) + (float(depth) * step_y))) for depth in range(level_count)}
    return {
        str(node.node_id): (int(x_by_id[str(node.node_id)]), int(y_by_depth[int(node.depth)]))
        for node in sample.nodes
    }


def render_binary_tree_scene(
    *,
    sample: BinaryTreeSample,
    render_params: GraphRenderParams,
    scene_variant: str,
    scene_title: str,
    layout_seed: int = 0,
    base_image: Image.Image | None = None,
) -> RenderedBinaryTreeScene:
    """Render one top-down labeled binary-tree scene."""

    image = (
        base_image.convert("RGB").copy()
        if base_image is not None
        else Image.new(
            "RGB",
            (int(render_params.canvas_width), int(render_params.canvas_height)),
            tuple(render_params.background_color_rgb),
        )
    )
    draw = ImageDraw.Draw(image)
    panel_geometry = _resolve_panel_geometry(render_params)
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
    panel = tuple(int(value) for value in panel_geometry["panel_xyxy"])
    title_band = tuple(int(value) for value in panel_geometry["title_band_xyxy"])
    panel_fill = tuple(int(value) for value in render_params.panel_fill_rgb)
    panel_border = tuple(int(value) for value in render_params.panel_border_rgb)
    resolved_layout_seed = int(layout_seed) + int(sum(ord(ch) for ch in str(scene_title)))
    draw.rounded_rectangle(
        panel,
        radius=max(6, int(render_params.panel_corner_radius_px)),
        fill=panel_fill,
        outline=panel_border,
        width=2,
    )
    title_font = load_font(
        int(render_params.panel_title_font_size_px),
        bold=True,
        font_family=str(render_params.font_family or ""),
    )
    title_style = resolve_readable_text_style(
        instance_seed=int(resolved_layout_seed + int(render_params.canvas_width) + int(render_params.canvas_height)),
        namespace="graph.binary_tree.panel_title_text",
        role="graph_panel_title_text",
        surface_rgbs=(panel_fill,),
        preferred_rgbs=(tuple(int(value) for value in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    draw_centered_readable_text(
        draw,
        text=str(scene_title),
        center=(0.5 * float(title_band[0] + title_band[2]), 0.5 * float(title_band[1] + title_band[3])),
        font=title_font,
        style=title_style,
        stroke_width=1,
    )

    block_context_elements = list(
        draw_graph_context_text_blocks(
            image,
            panel_geometry=panel_geometry,
            render_params=render_params,
            layout_seed=int(resolved_layout_seed),
        )
    )
    chip_context_elements = draw_graph_context_text_chips(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        layout_seed=int(resolved_layout_seed),
    )
    panel_context_elements = list(panel_geometry.get("context_text_elements", []))
    panel_context_elements.extend([dict(element) for element in block_context_elements])
    panel_context_elements.extend([dict(element) for element in chip_context_elements])
    if panel_context_elements:
        panel_geometry["context_text_elements"] = [dict(element) for element in panel_context_elements]

    apply_graph_content_layout_jitter(
        panel_geometry,
        render_params=render_params,
        layout_seed=int(resolved_layout_seed),
    )
    content = tuple(int(value) for value in panel_geometry["scene_content_xyxy"])
    if str(scene_variant) == "paper_tree":
        line_color = tuple(max(0, min(255, int((int(v) + 178) / 2))) for v in panel_border)
        for y in range(content[1] + 10, content[3], 34):
            draw.line((content[0], y, content[2], y), fill=line_color, width=1)
    elif str(scene_variant) == "boxed_tree":
        draw.rounded_rectangle(
            (content[0] - 8, content[1] - 8, content[2] + 8, content[3] + 8),
            radius=max(8, int(render_params.panel_corner_radius_px // 2)),
            outline=tuple(int(value) for value in render_params.panel_border_rgb),
            width=1,
        )

    positions = _binary_tree_positions(
        sample,
        content_bbox=content,
        node_radius_px=int(render_params.node_radius_px),
    )
    node_by_id = {str(node.node_id): node for node in sample.nodes}
    rendered_edges: List[RenderedBinaryTreeEdge] = []
    connector_style = _connector_style_for_scene(str(scene_variant))
    for node in sample.nodes:
        for side, child_id in (("left", node.left_id), ("right", node.right_id)):
            if child_id is None:
                continue
            start = tuple(int(value) for value in positions[str(node.node_id)])
            end = tuple(int(value) for value in positions[str(child_id)])
            path = _connector_path(
                start,
                end,
                connector_style_variant=str(connector_style),
                trim_px=max(1, int(render_params.node_radius_px) - 1),
            )
            draw.line(
                list(path),
                fill=tuple(int(value) for value in render_params.edge_color_rgb),
                width=max(1, int(render_params.edge_width_px)),
            )
            rendered_edges.append(
                RenderedBinaryTreeEdge(
                    edge_id=f"edge_{node.label}_{node_by_id[str(child_id)].label}",
                    parent_label=str(node.label),
                    child_label=str(node_by_id[str(child_id)].label),
                    child_side=str(side),
                    segment_px=(tuple(path[0]), tuple(path[-1])),
                    connector_path_px=tuple(tuple(point) for point in path),
                    connector_style_variant=str(connector_style),
                )
            )

    rendered_nodes: List[RenderedBinaryTreeNode] = []
    resolved_font_size = int(render_params.label_font_size_px)
    stroke_width = max(1, int(round(float(render_params.label_font_size_px) * 0.08)))
    shape_variant = str(render_params.node_shape_variant)
    if str(scene_variant) == "boxed_tree" and shape_variant not in SUPPORTED_NODE_SHAPE_VARIANTS:
        shape_variant = "rounded_square"
    for node in sample.nodes:
        center = tuple(int(value) for value in positions[str(node.node_id)])
        bbox = _draw_node_shape(
            draw,
            center=center,
            radius=int(render_params.node_radius_px),
            shape_variant=str(shape_variant),
            fill_rgb=tuple(int(value) for value in render_params.node_fill_rgb),
            outline_rgb=tuple(int(value) for value in render_params.node_border_rgb),
            outline_width=int(render_params.node_border_width_px),
        )
        font = fit_font_to_box(
            draw,
            text=str(node.label),
            max_width=max(8, int(bbox[2] - bbox[0]) - 6),
            max_height=max(8, int(bbox[3] - bbox[1]) - 6),
            bold=True,
            font_family=str(render_params.font_family or ""),
            min_size_px=8,
            max_size_px=int(render_params.label_font_size_px),
            fill_ratio=0.86,
        )
        resolved_font_size = min(int(resolved_font_size), int(getattr(font, "size", resolved_font_size)))
        label_style = resolve_readable_text_style(
            instance_seed=int(sum(ord(ch) for ch in str(node.label)) + int(center[0]) + (997 * int(center[1]))),
            namespace=f"graph.binary_tree.node_label_text.{str(node.label)}",
            role="graph_node_label_text",
            surface_rgbs=(tuple(int(value) for value in render_params.node_fill_rgb),),
            preferred_rgbs=(
                tuple(int(value) for value in render_params.label_text_rgb),
                tuple(int(value) for value in render_params.label_stroke_rgb),
                (255, 255, 255),
                (10, 14, 22),
            ),
            min_contrast_ratio=4.0,
            min_lab_distance=24.0,
        )
        draw_centered_readable_text(
            draw,
            text=str(node.label),
            center=(float(center[0]), float(center[1])),
            font=font,
            style=label_style,
            stroke_width=stroke_width,
        )
        parent_label = str(node_by_id[str(node.parent_id)].label) if node.parent_id is not None else None
        left_label = str(node_by_id[str(node.left_id)].label) if node.left_id is not None else None
        right_label = str(node_by_id[str(node.right_id)].label) if node.right_id is not None else None
        rendered_nodes.append(
            RenderedBinaryTreeNode(
                node_id=str(node.node_id),
                label=str(node.label),
                parent_label=parent_label,
                left_label=left_label,
                right_label=right_label,
                depth=int(node.depth),
                center_xy=tuple(center),
                bbox_xyxy=tuple(int(value) for value in bbox),
            )
        )

    return RenderedBinaryTreeScene(
        image=image,
        panel_geometry={str(key): value for key, value in panel_geometry.items()},
        nodes=tuple(rendered_nodes),
        edges=tuple(rendered_edges),
        scene_variant=str(scene_variant),
        connector_style_variant=str(connector_style),
        resolved_label_font_size_px=int(resolved_font_size),
        resolved_label_stroke_width_px=int(stroke_width),
    )


__all__ = ["render_binary_tree_scene"]
