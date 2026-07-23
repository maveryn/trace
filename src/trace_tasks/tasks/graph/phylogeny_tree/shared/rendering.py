"""Rendering helpers for phylogeny-tree graph scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import hash64
from ....shared.bbox_projection import round_bbox
from ....shared.text_rendering import load_font
from ....shared.text_legibility import draw_text_traced
from ...shared.graph_scene import GraphRenderParams
from .algorithms import descendant_leaf_labels
from .state import (
    SUPPORTED_PHYLOGENY_SCENE_VARIANTS,
    PhylogenySample,
    RenderedPhylogenyEdge,
    RenderedPhylogenyNode,
    RenderedPhylogenyScene,
    _leaf_node_id_by_label,
    _node_map,
)


def _ordered_children(sample: PhylogenySample, node_id: str, *, layout_seed: int) -> Tuple[str, ...]:
    node = _node_map(sample)[str(node_id)]
    child_ids = list(node.child_ids)
    if len(child_ids) <= 1:
        return tuple(child_ids)
    key_values = [
        (hash64(int(layout_seed), f"phylogeny_child_order:{node_id}:{child_id}", 0), str(child_id))
        for child_id in child_ids
    ]
    return tuple(child_id for _key, child_id in sorted(key_values))


def _leaf_order(sample: PhylogenySample, *, layout_seed: int) -> Tuple[str, ...]:
    labels: List[str] = []

    def visit(node_id: str) -> None:
        node = _node_map(sample)[str(node_id)]
        if node.leaf_label is not None:
            labels.append(str(node.leaf_label))
            return
        for child_id in _ordered_children(sample, str(node_id), layout_seed=int(layout_seed)):
            visit(str(child_id))

    visit(str(sample.root_id))
    return tuple(labels)


def _point_bbox(center: Sequence[float], radius: float = 8.0) -> Tuple[int, int, int, int]:
    x, y = float(center[0]), float(center[1])
    r = float(radius)
    return (
        int(round(x - r)),
        int(round(y - r)),
        int(round(x + r)),
        int(round(y + r)),
    )


def _draw_text_at(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    xy: Sequence[float],
    font,
    fill: Sequence[int],
    stroke_fill: Sequence[int],
    stroke_width: int,
) -> Tuple[int, int, int, int]:
    x, y = float(xy[0]), float(xy[1])
    bbox = draw.textbbox((x, y), str(text), font=font, stroke_width=int(stroke_width))
    draw_text_traced(
        draw,
        (x, y),
        str(text),
        font=font,
        fill=tuple(int(value) for value in fill),
        stroke_fill=tuple(int(value) for value in stroke_fill),
        stroke_width=int(stroke_width),
        role="graph_phylogeny_label_text",
        required=False,
    )
    return tuple(int(round(float(value))) for value in bbox)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Sequence[float],
    font,
    fill: Sequence[int],
    stroke_fill: Sequence[int],
    stroke_width: int,
) -> Tuple[int, int, int, int]:
    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    x = float(center[0]) - (0.5 * width) - float(bbox[0])
    y = float(center[1]) - (0.5 * height) - float(bbox[1])
    return _draw_text_at(
        draw,
        text=str(text),
        xy=(x, y),
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=int(stroke_width),
    )


def _draw_paper_lines(draw: ImageDraw.ImageDraw, bbox: Sequence[float], *, color: Sequence[int]) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    y = top + 22.0
    while y < bottom - 8.0:
        draw.line((left + 8.0, y, right - 8.0, y), fill=tuple(int(value) for value in color), width=1)
        y += 24.0


def _panel_content_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    title: str | None,
    render_params: GraphRenderParams,
    title_font,
) -> List[float]:
    left, top, right, bottom = [float(value) for value in panel_bbox]
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(int(value) for value in render_params.panel_fill_rgb),
        outline=tuple(int(value) for value in render_params.panel_border_rgb),
        width=max(1, int(render_params.node_border_width_px)),
    )
    pad = float(render_params.panel_padding_px)
    title_band = 0.0
    if title:
        title_band = max(44.0, pad + 24.0)
        _draw_centered_text(
            draw,
            text=str(title),
            center=(0.5 * (left + right), top + 0.46 * title_band),
            font=title_font,
            fill=render_params.title_color_rgb,
            stroke_fill=(255, 255, 255),
            stroke_width=1,
        )
    return round_bbox((left + pad, top + title_band + 8.0, right - pad, bottom - pad))


def _layout_positions(
    sample: PhylogenySample,
    *,
    content_bbox: Sequence[float],
    layout_seed: int,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Place leaves on a fixed right rail and internal nodes by depth plus child midpoint."""

    nodes = _node_map(sample)
    left, top, right, bottom = [float(value) for value in content_bbox]
    label_space = max(42.0, min(86.0, 0.18 * (right - left)))
    tree_left = left + 18.0
    tree_right = right - label_space
    leaf_y_top = top + 18.0
    leaf_y_bottom = bottom - 18.0
    leaf_order = _leaf_order(sample, layout_seed=int(layout_seed))
    if len(leaf_order) == 1:
        leaf_y = {str(leaf_order[0]): 0.5 * (leaf_y_top + leaf_y_bottom)}
    else:
        leaf_y = {
            str(label): leaf_y_top + ((leaf_y_bottom - leaf_y_top) * float(index) / float(len(leaf_order) - 1))
            for index, label in enumerate(leaf_order)
        }
    max_depth = max(1, int(sample.max_depth))
    depth_to_x = {
        depth: tree_left + ((tree_right - tree_left) * float(depth) / float(max_depth))
        for depth in range(max_depth + 1)
    }
    centers: Dict[str, Tuple[float, float]] = {}

    def place(node_id: str) -> Tuple[float, float]:
        node = nodes[str(node_id)]
        if node.leaf_label is not None:
            center = (tree_right, float(leaf_y[str(node.leaf_label)]))
            centers[str(node_id)] = center
            return center
        child_centers = [place(str(child_id)) for child_id in _ordered_children(sample, str(node_id), layout_seed=int(layout_seed))]
        y = sum(float(center[1]) for center in child_centers) / float(len(child_centers))
        x = float(depth_to_x.get(int(node.depth), tree_left))
        centers[str(node_id)] = (x, y)
        return (x, y)

    place(str(sample.root_id))
    label_positions = {
        str(label): (tree_right + 12.0, float(leaf_y[str(label)]))
        for label in leaf_order
    }
    return centers, label_positions


def _draw_cladogram_into_panel(
    draw: ImageDraw.ImageDraw,
    *,
    sample: PhylogenySample,
    panel_bbox: Sequence[float],
    render_params: GraphRenderParams,
    scene_variant: str,
    layout_seed: int,
    title: str | None,
    marked_node_id: str | None = None,
    option_label: str | None = None,
) -> Tuple[Tuple[RenderedPhylogenyNode, ...], Tuple[RenderedPhylogenyEdge, ...], Dict[str, Any]]:
    """Draw one panel and return projections tied to the final displayed branch geometry."""

    title_font = load_font(int(render_params.panel_title_font_size_px), bold=True, font_family=str(render_params.font_family or ""))
    label_font_size = max(14, min(int(render_params.label_font_size_px), 24))
    label_font = load_font(label_font_size, bold=True, font_family=str(render_params.font_family or ""))
    content_bbox = _panel_content_bbox(
        draw,
        panel_bbox=panel_bbox,
        title=title,
        render_params=render_params,
        title_font=title_font,
    )
    if str(scene_variant) == "paper_cladogram":
        _draw_paper_lines(draw, content_bbox, color=(222, 228, 238))
    if option_label is not None:
        label_font_option = load_font(max(18, int(render_params.panel_title_font_size_px)), bold=True, font_family=str(render_params.font_family or ""))
        left, top = float(panel_bbox[0]), float(panel_bbox[1])
        _draw_centered_text(
            draw,
            text=str(option_label),
            center=(left + 30.0, top + 30.0),
            font=label_font_option,
            fill=render_params.title_color_rgb,
            stroke_fill=(255, 255, 255),
            stroke_width=1,
        )

    nodes = _node_map(sample)
    centers, label_positions = _layout_positions(sample, content_bbox=content_bbox, layout_seed=int(layout_seed))
    stroke_width = 1
    edge_width = max(2, int(render_params.edge_width_px))
    edge_rgb = tuple(int(value) for value in render_params.edge_color_rgb)
    highlight_rgb = (221, 100, 36)
    rendered_edges: List[RenderedPhylogenyEdge] = []

    def edge_path(parent_center: Tuple[float, float], child_center: Tuple[float, float]) -> Tuple[Tuple[int, int], ...]:
        px, py = parent_center
        cx, cy = child_center
        if str(scene_variant) == "diagonal_cladogram":
            return ((int(round(px)), int(round(py))), (int(round(cx)), int(round(cy))))
        return (
            (int(round(px)), int(round(py))),
            (int(round(px)), int(round(cy))),
            (int(round(cx)), int(round(cy))),
        )

    marked_edge_path: Tuple[Tuple[int, int], ...] | None = None
    for node in sample.nodes:
        for child_id in node.child_ids:
            path = edge_path(centers[str(node.node_id)], centers[str(child_id)])
            rendered_edges.append(
                RenderedPhylogenyEdge(
                    edge_id=f"{node.node_id}->{child_id}",
                    parent_id=str(node.node_id),
                    child_id=str(child_id),
                    path_px=tuple(path),
                )
            )
            draw.line(path, fill=edge_rgb, width=edge_width, joint="curve")
            if marked_node_id is not None and str(child_id) == str(marked_node_id):
                marked_edge_path = tuple(path)

    if marked_edge_path is not None:
        draw.line(marked_edge_path, fill=highlight_rgb, width=edge_width + 4, joint="curve")

    rendered_nodes: List[RenderedPhylogenyNode] = []
    leaf_label_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    for node in sample.nodes:
        cx, cy = centers[str(node.node_id)]
        is_leaf = node.leaf_label is not None
        label_bbox = None
        if is_leaf:
            label = str(node.leaf_label)
            lx, ly = label_positions[label]
            text_bbox = draw.textbbox((0, 0), label, font=label_font, stroke_width=stroke_width)
            text_height = float(text_bbox[3] - text_bbox[1])
            label_bbox = _draw_text_at(
                draw,
                text=label,
                xy=(lx, ly - 0.5 * text_height),
                font=label_font,
                fill=render_params.title_color_rgb,
                stroke_fill=(255, 255, 255),
                stroke_width=stroke_width,
            )
            leaf_label_bboxes[label] = label_bbox
            radius = max(3, int(edge_width))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=edge_rgb)
        else:
            radius = max(4, int(edge_width + 1))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=edge_rgb)
        rendered_nodes.append(
            RenderedPhylogenyNode(
                node_id=str(node.node_id),
                leaf_label=(None if node.leaf_label is None else str(node.leaf_label)),
                descendant_leaf_labels=tuple(descendant_leaf_labels(sample, str(node.node_id))),
                center_xy=(int(round(cx)), int(round(cy))),
                bbox_xyxy=_point_bbox((cx, cy), radius=max(8.0, float(edge_width + 4))),
                label_bbox_xyxy=label_bbox,
                is_leaf=bool(is_leaf),
            )
        )

    if marked_node_id is not None:
        descendant_labels = descendant_leaf_labels(sample, str(marked_node_id))
        leaf_ys = [float(centers[_leaf_node_id_by_label(sample)[str(label)]][1]) for label in descendant_labels]
        if leaf_ys:
            bracket_x = max(float(content_bbox[0]) + 18.0, min(float(content_bbox[2]) - 48.0, max(float(centers[_leaf_node_id_by_label(sample)[str(label)]][0]) for label in descendant_labels) - 22.0))
            top_y = min(leaf_ys) - 7.0
            bottom_y = max(leaf_ys) + 7.0
            draw.line((bracket_x, top_y, bracket_x, bottom_y), fill=highlight_rgb, width=edge_width + 2)
            draw.line((bracket_x, top_y, bracket_x + 12.0, top_y), fill=highlight_rgb, width=edge_width + 2)
            draw.line((bracket_x, bottom_y, bracket_x + 12.0, bottom_y), fill=highlight_rgb, width=edge_width + 2)

    metadata = {
        "content_bbox": list(content_bbox),
        "leaf_label_bboxes": {label: list(bbox) for label, bbox in leaf_label_bboxes.items()},
    }
    return tuple(rendered_nodes), tuple(rendered_edges), metadata


def render_phylogeny_tree_scene(
    *,
    sample: PhylogenySample,
    render_params: GraphRenderParams,
    scene_variant: str,
    scene_title: str,
    layout_seed: int,
    base_image: Image.Image,
    marked_node_id: str | None = None,
) -> RenderedPhylogenyScene:
    """Render one phylogeny tree into a full-size panel."""

    if str(scene_variant) not in set(SUPPORTED_PHYLOGENY_SCENE_VARIANTS):
        raise ValueError(f"unsupported phylogeny scene_variant: {scene_variant}")
    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    margin = float(render_params.outer_margin_px)
    panel_bbox = round_bbox((margin, margin, float(render_params.canvas_width) - margin, float(render_params.canvas_height) - margin))
    nodes, edges, metadata = _draw_cladogram_into_panel(
        draw,
        sample=sample,
        panel_bbox=panel_bbox,
        render_params=render_params,
        scene_variant=str(scene_variant),
        layout_seed=int(layout_seed),
        title=str(scene_title),
        marked_node_id=marked_node_id,
    )
    panel_geometry = {
        "canvas_size": [int(render_params.canvas_width), int(render_params.canvas_height)],
        "panel_bbox": list(panel_bbox),
        "content_bbox": list(metadata["content_bbox"]),
    }
    return RenderedPhylogenyScene(
        image=image,
        panel_geometry=panel_geometry,
        nodes=tuple(nodes),
        edges=tuple(edges),
        scene_variant=str(scene_variant),
        resolved_label_font_size_px=max(14, min(int(render_params.label_font_size_px), 24)),
        resolved_label_stroke_width_px=1,
        option_panel_bboxes={},
    )


def render_phylogeny_option_scene(
    *,
    option_specs: Sequence[Mapping[str, Any]],
    render_params: GraphRenderParams,
    scene_variant: str,
    layout_seed: int,
    base_image: Image.Image,
) -> RenderedPhylogenyScene:
    """Render four phylogeny options into fixed A-D option panels."""

    if len(option_specs) != 4:
        raise ValueError("phylogeny option scene requires exactly four options")
    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    margin_x = float(render_params.outer_margin_px)
    margin_y = float(render_params.outer_margin_px)
    gap_x = 22.0
    gap_y = 22.0
    width = float(render_params.canvas_width)
    height = float(render_params.canvas_height)
    panel_w = (width - (2.0 * margin_x) - gap_x) / 2.0
    panel_h = (height - (2.0 * margin_y) - gap_y) / 2.0
    option_panel_bboxes: Dict[str, List[float]] = {}
    all_nodes: List[RenderedPhylogenyNode] = []
    all_edges: List[RenderedPhylogenyEdge] = []
    for index, spec in enumerate(option_specs):
        row = int(index // 2)
        col = int(index % 2)
        left = margin_x + (float(col) * (panel_w + gap_x))
        top = margin_y + (float(row) * (panel_h + gap_y))
        panel_bbox = round_bbox((left, top, left + panel_w, top + panel_h))
        option_label = str(spec["option_label"])
        option_panel_bboxes[option_label] = list(panel_bbox)
        nodes, edges, _metadata = _draw_cladogram_into_panel(
            draw,
            sample=spec["sample"],
            panel_bbox=panel_bbox,
            render_params=render_params,
            scene_variant=str(scene_variant),
            layout_seed=int(spec.get("layout_seed", hash64(int(layout_seed), f"option:{option_label}", int(index)))),
            title=None,
            option_label=str(option_label),
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)
    panel_geometry = {
        "canvas_size": [int(render_params.canvas_width), int(render_params.canvas_height)],
        "option_panel_bboxes": {key: list(value) for key, value in sorted(option_panel_bboxes.items())},
    }
    return RenderedPhylogenyScene(
        image=image,
        panel_geometry=panel_geometry,
        nodes=tuple(all_nodes),
        edges=tuple(all_edges),
        scene_variant=str(scene_variant),
        resolved_label_font_size_px=max(14, min(int(render_params.label_font_size_px), 24)),
        resolved_label_stroke_width_px=1,
        option_panel_bboxes={key: list(value) for key, value in sorted(option_panel_bboxes.items())},
    )


def phylogeny_scene_entities(
    sample: PhylogenySample,
    rendered_scene: RenderedPhylogenyScene,
) -> Tuple[Dict[str, Any], ...]:
    """Build trace scene entities for a rendered phylogeny."""

    node_by_id = _node_map(sample)
    entities: List[Dict[str, Any]] = []
    for rendered in rendered_scene.nodes:
        sample_node = node_by_id[str(rendered.node_id)]
        entities.append(
            {
                "entity_id": f"phylo_node_{rendered.node_id}",
                "entity_kind": "phylogeny_leaf" if rendered.is_leaf else "phylogeny_internal_node",
                "node_id": str(rendered.node_id),
                "leaf_label": rendered.leaf_label,
                "parent_id": sample_node.parent_id,
                "child_ids": list(sample_node.child_ids),
                "descendant_leaf_labels": list(rendered.descendant_leaf_labels),
                "depth": int(sample_node.depth),
                "center_px": list(rendered.center_xy),
                "bbox_xyxy": list(rendered.bbox_xyxy),
                "label_bbox_xyxy": (None if rendered.label_bbox_xyxy is None else list(rendered.label_bbox_xyxy)),
            }
        )
    for edge in rendered_scene.edges:
        entities.append(
            {
                "entity_id": f"phylo_edge_{edge.edge_id}",
                "entity_kind": "phylogeny_branch",
                "parent_id": str(edge.parent_id),
                "child_id": str(edge.child_id),
                "path_px": [list(point) for point in edge.path_px],
            }
        )
    return tuple(entities)


__all__ = [
    "phylogeny_scene_entities",
    "render_phylogeny_option_scene",
    "render_phylogeny_tree_scene",
]
