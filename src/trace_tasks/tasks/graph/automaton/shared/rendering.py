"""Rendering primitives for graph automaton diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.visual.background import make_background_canvas
from .....core.visual.noise import apply_post_image_noise
from ....shared.config_defaults import group_default
from ....shared.text_legibility import (
    draw_centered_readable_text,
    draw_readable_text,
    resolve_readable_text_style,
)
from ....shared.text_rendering import load_font
from ...shared.graph_scene import GraphRenderParams, RenderedGraphScene, render_graph_scene
from .state import AcceptanceAxes, AcceptanceRender, AcceptanceSample, OPTION_LABELS


def _draw_arrowhead(draw: ImageDraw.ImageDraw, *, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    """Draw a compact triangular arrowhead at ``end`` pointing from ``start``."""

    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    size = 11.0
    back_x = float(end[0]) - (ux * size)
    back_y = float(end[1]) - (uy * size)
    points = [
        (int(round(end[0])), int(round(end[1]))),
        (int(round(back_x + px * size * 0.48)), int(round(back_y + py * size * 0.48))),
        (int(round(back_x - px * size * 0.48)), int(round(back_y - py * size * 0.48))),
    ]
    draw.polygon(points, fill=tuple(int(value) for value in color))


def decorate_automaton_scene(
    rendered_scene: RenderedGraphScene,
    *,
    start_label: str,
    accepting_labels: Sequence[str],
    render_params: GraphRenderParams,
    layout_seed: int,
) -> None:
    """Draw start and accepting-state glyphs without changing task semantics."""

    draw = ImageDraw.Draw(rendered_scene.image)
    node_by_label = {str(node.label): node for node in rendered_scene.nodes}
    ink = tuple(int(value) for value in render_params.title_color_rgb)
    radius = int(render_params.node_radius_px)
    for label in accepting_labels:
        node = node_by_label.get(str(label))
        if node is None:
            continue
        x0, y0, x1, y1 = (int(value) for value in node.bbox_xyxy)
        inset = max(4, int(round(float(radius) * 0.22)))
        draw.ellipse((x0 + inset, y0 + inset, x1 - inset, y1 - inset), outline=ink, width=2)

    start_node = node_by_label.get(str(start_label))
    if start_node is None:
        return
    cx, cy = (int(value) for value in start_node.center_xy)
    content = tuple(int(value) for value in rendered_scene.panel_geometry.get("scene_content_xyxy", (0, 0, rendered_scene.image.width, rendered_scene.image.height)))
    if cx - content[0] >= 92:
        start = (int(max(content[0] + 12, cx - 82)), int(cy))
        end = (int(cx - radius - 6), int(cy))
        text_xy = (int(start[0]), int(cy - 26))
    else:
        start = (int(cx), int(max(content[1] + 12, cy - 82)))
        end = (int(cx), int(cy - radius - 6))
        text_xy = (int(cx + 12), int(start[1]))
    draw.line((start, end), fill=ink, width=3)
    _draw_arrowhead(draw, start=start, end=end, color=ink)
    font = load_font(14, bold=True, font_family=str(render_params.font_family or ""))
    start_label_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.automaton.start_label_text",
        role="automaton_start_label_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params.panel_fill_rgb),
            tuple(int(value) for value in render_params.background_color_rgb),
        ),
        preferred_rgbs=(ink,),
        min_contrast_ratio=7.0,
        min_lab_distance=38.0,
    )
    draw_readable_text(
        draw,
        xy=text_xy,
        text="start",
        font=font,
        style=start_label_style,
        stroke_width=2,
    )


def draw_candidate_options_panel(
    *,
    rendered_scene: RenderedGraphScene,
    sample: AcceptanceSample,
    render_params: GraphRenderParams,
    option_panel_height_px: int,
    layout_seed: int,
) -> Tuple[Image.Image, Dict[str, list[int]], Dict[str, Any]]:
    """Extend the graph render with a labeled candidate-string option panel."""

    graph_image = rendered_scene.image.convert("RGB")
    width, graph_height = graph_image.size
    panel_height = max(118, int(option_panel_height_px))
    image = Image.new(
        "RGB",
        (int(width), int(graph_height + panel_height)),
        tuple(int(value) for value in render_params.background_color_rgb),
    )
    image.paste(graph_image, (0, 0))
    draw = ImageDraw.Draw(image)

    margin = int(render_params.outer_margin_px)
    panel = (
        int(margin),
        int(graph_height + 14),
        int(width - margin),
        int(graph_height + panel_height - 22),
    )
    draw.rounded_rectangle(
        panel,
        radius=16,
        fill=tuple(int(value) for value in render_params.panel_fill_rgb),
        outline=tuple(int(value) for value in render_params.panel_border_rgb),
        width=2,
    )
    title_font = load_font(17, bold=True, font_family=str(render_params.font_family or ""))
    option_font = load_font(18, bold=True, font_family=str(render_params.font_family or ""))
    text_fill = tuple(int(value) for value in render_params.title_color_rgb)
    title_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.automaton.candidate_panel_title_text",
        role="automaton_candidate_panel_title_text",
        surface_rgbs=(tuple(int(value) for value in render_params.panel_fill_rgb),),
        preferred_rgbs=(text_fill,),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    option_chip_fill = tuple(int(value) for value in render_params.panel_fill_rgb)
    option_chip_outline = tuple(int(value) for value in render_params.panel_border_rgb)
    option_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.automaton.candidate_option_text",
        role="automaton_candidate_option_text",
        surface_rgbs=(option_chip_fill,),
        preferred_rgbs=(text_fill,),
        min_contrast_ratio=7.0,
        min_lab_distance=38.0,
    )
    draw_readable_text(
        draw,
        xy=(panel[0] + 18, panel[1] + 10),
        text="Candidate strings",
        font=title_font,
        style=title_style,
        stroke_width=2,
    )

    option_bboxes: Dict[str, list[int]] = {}
    candidate_count = int(len(sample.candidate_strings_by_option))
    columns = 2 if int(candidate_count) <= 4 else 3
    rows = int(math.ceil(float(candidate_count) / float(columns)))
    gap_x = 14
    gap_y = 10
    chip_top = int(panel[1] + 40)
    chip_left = int(panel[0] + 18)
    chip_right = int(panel[2] - 18)
    chip_bottom = int(panel[3] - 12)
    cell_w = int((chip_right - chip_left - (columns - 1) * gap_x) / columns)
    cell_h = int((chip_bottom - chip_top - (rows - 1) * gap_y) / rows)
    for index, option_label in enumerate(OPTION_LABELS[: len(sample.candidate_strings_by_option)]):
        row = int(index // columns)
        col = int(index % columns)
        x0 = int(chip_left + col * (cell_w + gap_x))
        y0 = int(chip_top + row * (cell_h + gap_y))
        x1 = int(x0 + cell_w)
        y1 = int(y0 + cell_h)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=10,
            fill=option_chip_fill,
            outline=option_chip_outline,
            width=1,
        )
        text = f"{str(option_label)}: {sample.candidate_strings_by_option[str(option_label)]}"
        draw_centered_readable_text(
            draw,
            center=(0.5 * float(x0 + x1), 0.5 * float(y0 + y1)),
            text=text,
            font=option_font,
            style=option_style,
            stroke_width=2,
        )
        option_bboxes[str(option_label)] = [int(x0), int(y0), int(x1), int(y1)]

    return image, option_bboxes, {
        "candidate_options_panel_xyxy": [int(value) for value in panel],
        "candidate_option_bboxes_xyxy": {str(key): list(value) for key, value in option_bboxes.items()},
        "candidate_option_grid": {
            "candidate_count": int(candidate_count),
            "columns": int(columns),
            "rows": int(rows),
        },
    }


def render_acceptance_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    post_image_background_defaults: Mapping[str, Any],
    post_image_noise_defaults: Mapping[str, Any],
    axes: AcceptanceAxes,
    sample: AcceptanceSample,
    render_params: GraphRenderParams,
    option_panel_height_fallback: int,
    layout_seed: int,
) -> AcceptanceRender:
    """Render one acceptance scene from already sampled task-owned semantics."""

    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=post_image_background_defaults,
    )
    node_style_by_label = {
        str(sample.start_label): {
            "halo_rgb": tuple(int(value) for value in render_params.title_color_rgb),
            "halo_width_px": 3,
            "halo_pad_px": 6,
        }
    }
    rendered_scene = render_graph_scene(
        graph_sample=sample.graph_sample,
        layout_variant=str(axes.layout_variant),
        layout_transform_variant=str(axes.layout_transform_variant),
        render_params=render_params,
        layout_seed=int(layout_seed),
        scene_title="State Transition Diagram",
        directed=True,
        base_image=background,
        edge_text_labels_by_label=sample.transition_labels_by_edge,
        edge_text_label_font_size_px=max(15, int(render_params.label_font_size_px) - 3),
        node_style_by_label=node_style_by_label,
        layout_fallback_variants=("circular", "shell", "layered", "spring"),
    )
    decorate_automaton_scene(
        rendered_scene,
        start_label=str(sample.start_label),
        accepting_labels=tuple(str(label) for label in sample.accepting_labels),
        render_params=render_params,
        layout_seed=int(layout_seed),
    )
    image_with_options, option_bboxes, option_panel_meta = draw_candidate_options_panel(
        rendered_scene=rendered_scene,
        sample=sample,
        render_params=render_params,
        option_panel_height_px=int(
            params.get(
                "option_panel_height_px",
                group_default(render_defaults, "option_panel_height_px", int(option_panel_height_fallback)),
            )
        ),
        layout_seed=int(layout_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        image_with_options,
        instance_seed=int(instance_seed),
        params=params,
        default_config=post_image_noise_defaults,
    )
    return AcceptanceRender(
        render_params=render_params,
        rendered_scene=rendered_scene,
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        option_bboxes={str(key): list(value) for key, value in option_bboxes.items()},
        option_panel_meta=dict(option_panel_meta),
    )


__all__ = [
    "decorate_automaton_scene",
    "draw_candidate_options_panel",
    "render_acceptance_scene",
]
