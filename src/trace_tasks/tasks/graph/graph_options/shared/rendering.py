"""Rendering primitives for graph option-panel scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.sampling import uniform_choice
from .....core.seed import hash64, spawn_rng
from .....core.visual.background import make_background_canvas
from .....core.visual.noise import apply_post_image_noise
from ....shared.bbox_projection import round_bbox as _round_bbox
from ....shared.config_defaults import group_default
from ....shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ....shared.render_variation import resolve_render_int, resolve_render_rgb
from ....shared.text_legibility import contrast_ratio, draw_centered_readable_text, resolve_readable_text_style
from ....shared.text_rendering import fit_font_to_box, load_font
from ...shared.style import apply_graph_panel_style, build_graph_named_color_theme
from .sampling import edge_set, is_directed
from .state import GraphOptionsDataset, GraphOptionsRenderedScene, GraphOptionsRenderParams


_RENDER_FALLBACKS = GraphOptionsRenderParams()


def _node_fill_with_readable_label(fill_rgb: Sequence[int]) -> Tuple[int, int, int]:
    """Adjust saturated node fills only when standard label colors cannot pass."""

    base = tuple(max(0, min(255, int(value))) for value in fill_rgb[:3])
    if max(float(contrast_ratio((255, 255, 255), base)), float(contrast_ratio((10, 14, 22), base))) >= 7.0:
        return base
    for factor in (0.90, 0.82, 0.74, 0.66, 0.58, 0.50, 0.44, 0.38, 0.32):
        candidate = tuple(max(0, min(255, int(round(float(channel) * float(factor))))) for channel in base)
        if float(contrast_ratio((255, 255, 255), candidate)) >= 7.0:
            return candidate
    return tuple(max(0, min(255, int(round(float(channel) * 0.28)))) for channel in base)


def resolve_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> GraphOptionsRenderParams:
    """Resolve base render constants from scene/task defaults."""

    render_namespace = f"{namespace}.render"
    return GraphOptionsRenderParams(
        canvas_width=resolve_render_int(params, render_defaults, "canvas_width", _RENDER_FALLBACKS.canvas_width, instance_seed=instance_seed, namespace=render_namespace),
        canvas_height=resolve_render_int(params, render_defaults, "canvas_height", _RENDER_FALLBACKS.canvas_height, instance_seed=instance_seed, namespace=render_namespace),
        margin_x_px=resolve_render_int(params, render_defaults, "margin_x_px", _RENDER_FALLBACKS.margin_x_px, instance_seed=instance_seed, namespace=render_namespace),
        margin_top_px=resolve_render_int(params, render_defaults, "margin_top_px", _RENDER_FALLBACKS.margin_top_px, instance_seed=instance_seed, namespace=render_namespace),
        reference_panel_height_px=resolve_render_int(params, render_defaults, "reference_panel_height_px", _RENDER_FALLBACKS.reference_panel_height_px, instance_seed=instance_seed, namespace=render_namespace),
        reference_to_options_gap_px=resolve_render_int(params, render_defaults, "reference_to_options_gap_px", _RENDER_FALLBACKS.reference_to_options_gap_px, instance_seed=instance_seed, namespace=render_namespace),
        option_gap_px=resolve_render_int(params, render_defaults, "option_gap_px", _RENDER_FALLBACKS.option_gap_px, instance_seed=instance_seed, namespace=render_namespace),
        option_row_gap_px=resolve_render_int(params, render_defaults, "option_row_gap_px", _RENDER_FALLBACKS.option_row_gap_px, instance_seed=instance_seed, namespace=render_namespace),
        panel_padding_px=resolve_render_int(params, render_defaults, "panel_padding_px", _RENDER_FALLBACKS.panel_padding_px, instance_seed=instance_seed, namespace=render_namespace),
        panel_corner_radius_px=resolve_render_int(params, render_defaults, "panel_corner_radius_px", _RENDER_FALLBACKS.panel_corner_radius_px, instance_seed=instance_seed, namespace=render_namespace),
        border_width_px=resolve_render_int(params, render_defaults, "border_width_px", _RENDER_FALLBACKS.border_width_px, instance_seed=instance_seed, namespace=render_namespace),
        title_font_size_px=resolve_render_int(params, render_defaults, "title_font_size_px", _RENDER_FALLBACKS.title_font_size_px, instance_seed=instance_seed, namespace=render_namespace),
        option_label_font_size_px=resolve_render_int(params, render_defaults, "option_label_font_size_px", _RENDER_FALLBACKS.option_label_font_size_px, instance_seed=instance_seed, namespace=render_namespace),
        node_radius_px=resolve_render_int(params, render_defaults, "node_radius_px", _RENDER_FALLBACKS.node_radius_px, instance_seed=instance_seed, namespace=render_namespace),
        edge_width_px=resolve_render_int(params, render_defaults, "edge_width_px", _RENDER_FALLBACKS.edge_width_px, instance_seed=instance_seed, namespace=render_namespace),
        panel_fill_rgb=resolve_render_rgb(params, render_defaults, "panel_fill_rgb", _RENDER_FALLBACKS.panel_fill_rgb, instance_seed=instance_seed, namespace=render_namespace),
        option_fill_rgb=resolve_render_rgb(params, render_defaults, "option_fill_rgb", _RENDER_FALLBACKS.option_fill_rgb, instance_seed=instance_seed, namespace=render_namespace),
        border_rgb=resolve_render_rgb(params, render_defaults, "border_rgb", _RENDER_FALLBACKS.border_rgb, instance_seed=instance_seed, namespace=render_namespace),
        edge_rgb=resolve_render_rgb(params, render_defaults, "edge_rgb", _RENDER_FALLBACKS.edge_rgb, instance_seed=instance_seed, namespace=render_namespace),
        node_fill_rgb=resolve_render_rgb(params, render_defaults, "node_fill_rgb", _RENDER_FALLBACKS.node_fill_rgb, instance_seed=instance_seed, namespace=render_namespace),
        node_outline_rgb=resolve_render_rgb(params, render_defaults, "node_outline_rgb", _RENDER_FALLBACKS.node_outline_rgb, instance_seed=instance_seed, namespace=render_namespace),
        text_rgb=resolve_render_rgb(params, render_defaults, "text_rgb", _RENDER_FALLBACKS.text_rgb, instance_seed=instance_seed, namespace=render_namespace),
        text_stroke_rgb=resolve_render_rgb(params, render_defaults, "text_stroke_rgb", _RENDER_FALLBACKS.text_stroke_rgb, instance_seed=instance_seed, namespace=render_namespace),
        notebook_line_rgb=resolve_render_rgb(params, render_defaults, "notebook_line_rgb", _RENDER_FALLBACKS.notebook_line_rgb, instance_seed=instance_seed, namespace=render_namespace),
    )


def _draw_readable_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: Any,
    style: Any,
    stroke_width: int = 1,
) -> List[float]:
    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    x = float(center[0]) - (0.5 * width) - float(bbox[0])
    y = float(center[1]) - (0.5 * height) - float(bbox[1])
    draw_centered_readable_text(
        draw,
        center=center,
        text=str(text),
        font=font,
        style=style,
        stroke_width=int(stroke_width),
    )
    return _round_bbox((x + bbox[0], y + bbox[1], x + bbox[2], y + bbox[3]))


def _draw_notebook_lines(
    draw: ImageDraw.ImageDraw,
    *,
    content_bbox: Sequence[float],
    color: Sequence[int],
) -> None:
    left, top, right, bottom = [float(value) for value in content_bbox]
    step = 24.0
    y = top + step
    while y < bottom - 4:
        draw.line((left + 8, y, right - 8, y), fill=tuple(int(value) for value in color), width=1)
        y += step
    x = left + step
    while x < right - 4:
        draw.line((x, top + 8, x, bottom - 8), fill=tuple(int(value) for value in color), width=1)
        x += step


def _panel_content_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    title: str,
    render_params: GraphOptionsRenderParams,
    title_font: Any,
    text_style: Any,
    panel_fill_rgb: Sequence[int],
    border_rgb: Sequence[int],
) -> List[float]:
    left, top, right, bottom = [float(value) for value in panel_bbox]
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(int(value) for value in panel_fill_rgb),
        outline=tuple(int(value) for value in border_rgb),
        width=int(render_params.border_width_px),
    )
    title_band = max(50.0, float(render_params.panel_padding_px) + 24.0)
    title_center_y = top + 0.46 * float(title_band)
    _draw_readable_centered_text(
        draw,
        text=str(title),
        center=(0.5 * (left + right), title_center_y),
        font=title_font,
        style=text_style,
        stroke_width=1,
    )
    pad = float(render_params.panel_padding_px)
    return _round_bbox((left + pad, top + title_band, right - pad, bottom - pad))


def _layout_structure_positions(
    *,
    labels: Sequence[str],
    content_bbox: Sequence[float],
    layout_variant: str,
    rng: Any,
) -> Dict[str, Tuple[float, float]]:
    """Place nodes inside one option panel with deterministic jitter."""

    label_list = [str(label) for label in labels]
    left, top, right, bottom = [float(value) for value in content_bbox]
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    cx = left + 0.5 * width
    cy = top + 0.5 * height
    rx = max(28.0, 0.40 * width)
    ry = max(28.0, 0.36 * height)
    order = list(label_list)
    rng.shuffle(order)
    positions: Dict[str, Tuple[float, float]] = {}
    if str(layout_variant) == "chain":
        if len(order) == 1:
            positions[order[0]] = (cx, cy)
        else:
            for index, label in enumerate(order):
                t = float(index) / float(len(order) - 1)
                x = left + 0.14 * width + (0.72 * width * t)
                y = cy + (0.22 * height * math.sin((t * math.pi * 2.0) + rng.uniform(-0.3, 0.3)))
                positions[str(label)] = (float(x), float(y))
    elif str(layout_variant) == "branch":
        if order:
            positions[order[0]] = (cx, cy)
        for index, label in enumerate(order[1:], start=1):
            angle = (2.0 * math.pi * float(index - 1) / max(1.0, float(len(order) - 1))) + rng.uniform(-0.22, 0.22)
            radius_scale = 0.58 + (0.18 * (index % 2))
            positions[str(label)] = (
                float(cx + (radius_scale * rx * math.cos(angle))),
                float(cy + (radius_scale * ry * math.sin(angle))),
            )
    else:
        offset = rng.uniform(-0.25, 0.25)
        for index, label in enumerate(order):
            angle = (2.0 * math.pi * float(index) / max(1.0, float(len(order)))) + offset
            positions[str(label)] = (float(cx + rx * math.cos(angle)), float(cy + ry * math.sin(angle)))
    jitter_x = 0.035 * width
    jitter_y = 0.035 * height
    resolved = {
        str(label): (
            max(left + 24.0, min(right - 24.0, float(x + rng.uniform(-jitter_x, jitter_x)))),
            max(top + 24.0, min(bottom - 24.0, float(y + rng.uniform(-jitter_y, jitter_y)))),
        )
        for label, (x, y) in positions.items()
    }
    min_sep = min(54.0, max(40.0, 0.30 * min(width, height)))
    for _ in range(48):
        moved = False
        for i, left_label in enumerate(label_list):
            for right_label in label_list[i + 1 :]:
                lx, ly = resolved[str(left_label)]
                rx2, ry2 = resolved[str(right_label)]
                dx = float(rx2 - lx)
                dy = float(ry2 - ly)
                dist = max(1e-6, math.hypot(dx, dy))
                if dist >= min_sep:
                    continue
                push = 0.5 * (min_sep - dist)
                ux = dx / dist
                uy = dy / dist
                resolved[str(left_label)] = (
                    max(left + 24.0, min(right - 24.0, lx - (push * ux))),
                    max(top + 24.0, min(bottom - 24.0, ly - (push * uy))),
                )
                resolved[str(right_label)] = (
                    max(left + 24.0, min(right - 24.0, rx2 + (push * ux))),
                    max(top + 24.0, min(bottom - 24.0, ry2 + (push * uy))),
                )
                moved = True
        if not moved:
            break
    return resolved


def _draw_structure(
    draw: ImageDraw.ImageDraw,
    *,
    spec: Mapping[str, Any],
    content_bbox: Sequence[float],
    panel_id: str,
    scene_variant: str,
    render_params: GraphOptionsRenderParams,
    palette_colors: Sequence[Sequence[int]],
    style_seed: int,
    rng: Any,
    namespace: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Draw one graph structure into a panel and return node/edge entities."""

    layout_variant = str(rng.choice(["circle", "chain", "branch"]))
    labels = [str(label) for label in spec["labels"]]
    directed = is_directed(spec)
    positions = _layout_structure_positions(labels=labels, content_bbox=content_bbox, layout_variant=layout_variant, rng=rng)
    atom_palette = [tuple(int(channel) for channel in color) for color in palette_colors] or [(84, 118, 194)]
    if str(scene_variant) == "notebook_graph_options":
        _draw_notebook_lines(draw, content_bbox=content_bbox, color=render_params.notebook_line_rgb)

    edge_rgb = tuple(int(value) for value in render_params.edge_rgb)
    for u_label, v_label in spec["edges"]:
        ux, uy = positions[str(u_label)]
        vx, vy = positions[str(v_label)]
        if directed:
            dx = float(vx - ux)
            dy = float(vy - uy)
            distance = max(1e-6, math.hypot(dx, dy))
            shrink = float(render_params.node_radius_px) + 3.0
            start = (ux + (dx / distance) * shrink, uy + (dy / distance) * shrink)
            end = (vx - (dx / distance) * shrink, vy - (dy / distance) * shrink)
        else:
            start = (ux, uy)
            end = (vx, vy)
        draw.line(
            (start[0], start[1], end[0], end[1]),
            fill=edge_rgb,
            width=int(render_params.edge_width_px),
            joint="curve",
        )
        if directed:
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            arrow_len = max(12.0, float(render_params.edge_width_px) * 3.1)
            arrow_w = max(8.0, float(render_params.edge_width_px) * 2.0)
            tip = end
            left = (
                tip[0] - arrow_len * math.cos(angle) + arrow_w * math.cos(angle + math.pi / 2.0),
                tip[1] - arrow_len * math.sin(angle) + arrow_w * math.sin(angle + math.pi / 2.0),
            )
            right = (
                tip[0] - arrow_len * math.cos(angle) + arrow_w * math.cos(angle - math.pi / 2.0),
                tip[1] - arrow_len * math.sin(angle) + arrow_w * math.sin(angle - math.pi / 2.0),
            )
            draw.polygon((tip, left, right), fill=edge_rgb)

    entities: List[Dict[str, Any]] = []
    bbox_map: Dict[str, List[float]] = {}
    radius = float(render_params.node_radius_px)
    for index, label in enumerate(labels):
        cx, cy = positions[str(label)]
        if str(scene_variant) == "colored_node_graph_options":
            node_fill = _node_fill_with_readable_label(atom_palette[index % len(atom_palette)])
        else:
            node_fill = _node_fill_with_readable_label(render_params.node_fill_rgb)
        outline = tuple(int(value) for value in render_params.node_outline_rgb)
        node_bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(tuple(node_bbox), fill=node_fill, outline=outline, width=max(2, int(render_params.border_width_px)))
        label_font = fit_font_to_box(
            draw,
            text=str(label),
            max_width=1.35 * radius,
            max_height=1.15 * radius,
            bold=True,
            font_family=str(render_params.font_family or ""),
            min_size_px=12,
            max_size_px=24,
            fill_ratio=0.95,
        )
        label_style = resolve_readable_text_style(
            instance_seed=int(hash64(int(style_seed), f"{panel_id}.node_label_text.{label}", int(index))),
            namespace=f"{namespace}.render.node_label_text.{panel_id}.{label}",
            role="graph_structure_node_label_text",
            surface_rgbs=(tuple(int(value) for value in node_fill),),
            preferred_rgbs=(
                tuple(int(value) for value in render_params.text_rgb),
                (255, 255, 255),
                (10, 14, 22),
            ),
        )
        _draw_readable_centered_text(
            draw,
            text=str(label),
            center=(cx, cy),
            font=label_font,
            style=label_style,
            stroke_width=1,
        )
        entity_id = f"{panel_id}_node_{label}"
        rounded_bbox = _round_bbox(node_bbox)
        bbox_map[entity_id] = rounded_bbox
        entities.append(
            {
                "entity_id": entity_id,
                "entity_kind": "structure_node",
                "panel_id": str(panel_id),
                "label": str(label),
                "center_px": [round(float(cx), 3), round(float(cy), 3)],
                "bbox_xyxy": list(rounded_bbox),
            }
        )
    for edge_index, (u_label, v_label) in enumerate(sorted(edge_set(spec))):
        ux, uy = positions[str(u_label)]
        vx, vy = positions[str(v_label)]
        entities.append(
            {
                "entity_id": f"{panel_id}_edge_{edge_index}",
                "entity_kind": "structure_edge",
                "panel_id": str(panel_id),
                "node_u_label": str(u_label),
                "node_v_label": str(v_label),
                "segment_px": [[round(float(ux), 3), round(float(uy), 3)], [round(float(vx), 3), round(float(vy), 3)]],
            }
        )
    entities.append(
        {
            "entity_id": f"{panel_id}_structure",
            "entity_kind": "structure_diagram",
            "panel_id": str(panel_id),
            "labels": list(labels),
            "edges": [list(edge) for edge in sorted(edge_set(spec))],
            "directed": bool(directed),
            "layout_variant": str(layout_variant),
        }
    )
    return entities, bbox_map


def _render_scene(
    *,
    dataset: GraphOptionsDataset,
    scene_variant: str,
    render_params: GraphOptionsRenderParams,
    instance_seed: int,
    background: Image.Image,
    palette_colors: Sequence[Sequence[int]],
    namespace: str,
) -> Tuple[Image.Image, List[Dict[str, Any]], List[float], Dict[str, List[float]], Dict[str, List[float]]]:
    """Render the top source graph and option grid before image noise."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    title_font = load_font(
        int(render_params.title_font_size_px),
        bold=True,
        font_family=str(render_params.font_family or ""),
    )
    option_font = load_font(
        int(render_params.option_label_font_size_px),
        bold=True,
        font_family=str(render_params.font_family or ""),
    )
    panel_text_style = resolve_readable_text_style(
        instance_seed=int(hash64(int(instance_seed), f"{namespace}.render.panel_text")),
        namespace=f"{namespace}.render.panel_text",
        role="graph_structure_panel_title_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params.panel_fill_rgb),
            tuple(int(value) for value in render_params.option_fill_rgb),
        ),
        preferred_rgbs=(
            tuple(int(value) for value in render_params.text_rgb),
            (10, 14, 22),
            (36, 45, 64),
            (250, 252, 255),
        ),
    )
    entities: List[Dict[str, Any]] = []
    bbox_map: Dict[str, List[float]] = {}
    option_panel_bbox_map: Dict[str, List[float]] = {}

    ref_left = float(render_params.margin_x_px)
    ref_top = float(render_params.margin_top_px)
    ref_right = float(render_params.canvas_width - render_params.margin_x_px)
    ref_bottom = ref_top + float(render_params.reference_panel_height_px)
    reference_bbox = _round_bbox((ref_left, ref_top, ref_right, ref_bottom))
    reference_content = _panel_content_bbox(
        draw,
        panel_bbox=reference_bbox,
        title=str(dataset.panel_title),
        render_params=render_params,
        title_font=title_font,
        text_style=panel_text_style,
        panel_fill_rgb=render_params.panel_fill_rgb,
        border_rgb=render_params.border_rgb,
    )
    ref_entities, ref_bboxes = _draw_structure(
        draw,
        spec=dataset.source_structure_spec,
        content_bbox=reference_content,
        panel_id="query_panel",
        scene_variant=str(scene_variant),
        render_params=render_params,
        palette_colors=palette_colors,
        style_seed=int(hash64(int(instance_seed), f"{namespace}.render.query_structure_style")),
        rng=spawn_rng(int(instance_seed), f"{namespace}.render.query_structure"),
        namespace=str(namespace),
    )
    entities.extend(ref_entities)
    bbox_map.update(ref_bboxes)
    bbox_map["query_panel"] = list(reference_bbox)
    entities.append(
        {
            "entity_id": "query_panel",
            "entity_kind": "query_panel",
            "title": str(dataset.panel_title),
            "bbox_xyxy": list(reference_bbox),
            "content_bbox_xyxy": list(reference_content),
        }
    )

    option_count = int(dataset.option_count)
    cols = 2 if int(option_count) <= 4 else 3
    rows = int(math.ceil(float(option_count) / float(cols)))
    options_top = ref_bottom + float(render_params.reference_to_options_gap_px)
    usable_width = float(render_params.canvas_width - (2 * render_params.margin_x_px))
    option_w = (usable_width - (float(cols - 1) * float(render_params.option_gap_px))) / float(cols)
    remaining_height = float(render_params.canvas_height) - options_top - float(render_params.margin_top_px)
    option_h = (remaining_height - (float(rows - 1) * float(render_params.option_row_gap_px))) / float(rows)

    for option_index, option_spec in enumerate(dataset.option_specs):
        row = int(option_index // cols)
        col = int(option_index % cols)
        left = ref_left + (float(col) * (option_w + float(render_params.option_gap_px)))
        top = options_top + (float(row) * (option_h + float(render_params.option_row_gap_px)))
        panel_bbox = _round_bbox((left, top, left + option_w, top + option_h))
        option_panel_id = str(option_spec["option_panel_id"])
        option_panel_bbox_map[option_panel_id] = list(panel_bbox)
        bbox_map[option_panel_id] = list(panel_bbox)
        content_bbox = _panel_content_bbox(
            draw,
            panel_bbox=panel_bbox,
            title=str(option_spec["option_label"]),
            render_params=render_params,
            title_font=option_font,
            text_style=panel_text_style,
            panel_fill_rgb=render_params.option_fill_rgb,
            border_rgb=render_params.border_rgb,
        )
        option_entities, option_bboxes = _draw_structure(
            draw,
            spec=option_spec["structure_spec"],
            content_bbox=content_bbox,
            panel_id=option_panel_id,
            scene_variant=str(scene_variant),
            render_params=render_params,
            palette_colors=palette_colors,
            style_seed=int(hash64(int(instance_seed), f"{namespace}.render.{option_panel_id}_style", int(option_index))),
            rng=spawn_rng(int(instance_seed), f"{namespace}.render.{option_panel_id}", int(option_index)),
            namespace=str(namespace),
        )
        entities.extend(option_entities)
        bbox_map.update(option_bboxes)
        entities.append(
            {
                "entity_id": option_panel_id,
                "entity_kind": "option_panel",
                "option_label": str(option_spec["option_label"]),
                "is_correct": bool(option_spec["is_correct"]),
                "bbox_xyxy": list(panel_bbox),
                "content_bbox_xyxy": list(content_bbox),
            }
        )

    scene_bbox = _round_bbox((ref_left, ref_top, ref_right, float(render_params.canvas_height - render_params.margin_top_px)))
    return image, entities, scene_bbox, bbox_map, option_panel_bbox_map


def render_graph_options_scene(
    *,
    dataset: GraphOptionsDataset,
    scene_variant: str,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> GraphOptionsRenderedScene:
    """Resolve style, draw panels/options, and apply coordinate-preserving noise."""

    base_render_params = resolve_render_params(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    color_names = ("blue", "green", "orange", "purple", "cyan", "magenta", "maroon")
    node_color_name = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.node_color_name"),
            color_names,
        )
    )
    panel_style_variants = ("default", "cool", "warm", "mint", "paper")
    panel_style = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.panel_style_variant"),
            panel_style_variants,
        )
    )
    graph_theme = apply_graph_panel_style(
        build_graph_named_color_theme(str(node_color_name)),
        panel_style_variant=str(panel_style),
    )
    palette_colors = (
        graph_theme.node_fill_rgb,
        (217, 119, 6),
        (22, 163, 74),
        (147, 51, 234),
        (8, 145, 178),
        (220, 38, 38),
    )
    scene_style_meta = {
        "node_color_name": str(node_color_name),
        "panel_style_variant": str(panel_style),
    }
    font_params: Dict[str, Any] = dict(render_defaults)
    font_params.update(dict(params))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.render.font_family",
        params=font_params,
    )
    font_record = get_font_family_record(str(font_family))
    render_params = GraphOptionsRenderParams(
        **{
            **base_render_params.__dict__,
            "panel_fill_rgb": tuple(int(value) for value in graph_theme.panel_fill_rgb),
            "option_fill_rgb": (255, 255, 255),
            "border_rgb": tuple(int(value) for value in graph_theme.panel_border_rgb),
            "edge_rgb": tuple(int(value) for value in graph_theme.edge_color_rgb),
            "node_fill_rgb": tuple(int(value) for value in graph_theme.node_fill_rgb),
            "node_outline_rgb": tuple(int(value) for value in graph_theme.node_border_rgb),
            "text_rgb": tuple(int(value) for value in graph_theme.title_color_rgb),
            "text_stroke_rgb": (255, 255, 255),
            "notebook_line_rgb": tuple(int(value) for value in graph_theme.panel_border_rgb),
            "font_family": str(font_family),
            "font_asset": dict(font_record.to_trace()),
            "font_asset_version": str(font_asset_version()),
            "font_exclusion_reason": "readout font pool; no scene-local exclusion",
        }
    )
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=background_defaults,
        fallback_color=tuple(int(value) for value in graph_theme.background_color_rgb),
    )
    base_image, entities, scene_bbox, bbox_map, option_panel_bbox_map = _render_scene(
        dataset=dataset,
        scene_variant=str(scene_variant),
        render_params=render_params,
        instance_seed=int(instance_seed),
        background=background,
        palette_colors=palette_colors,
        namespace=str(namespace),
    )
    image, post_noise_meta = apply_post_image_noise(
        base_image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return GraphOptionsRenderedScene(
        image=image,
        entities=entities,
        scene_bbox_px=scene_bbox,
        bbox_map=bbox_map,
        option_panel_bbox_map=option_panel_bbox_map,
        render_params=render_params,
        scene_variant=str(scene_variant),
        edge_mode=str(dataset.edge_mode),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        scene_style_meta=dict(scene_style_meta),
    )


__all__ = [
    "render_graph_options_scene",
    "resolve_render_params",
]
