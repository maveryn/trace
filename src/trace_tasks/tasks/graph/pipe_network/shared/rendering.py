"""Rendering primitives for the pipe-network graph scene."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.font_assets import font_asset_version, get_font_family_record
from ....shared.text_legibility import (
    draw_centered_readable_text,
    resolve_readable_text_style,
    text_legibility_summary_from_records,
)
from ....shared.text_rendering import fit_font_to_box, load_font
from ...shared.graph_sample_types import graph_label_sort_key
from ...shared.graph_scene import (
    BBox,
    GraphRenderParams,
    Point,
    apply_graph_content_layout_jitter,
    draw_graph_context_text_blocks,
    draw_graph_context_text_chips,
)
from .algorithms import canonical_node_edge, label_edge, parse_pipe_grid_shape
from .state import PIPE_VISUAL_STYLE_IDS, PipeJunctionNetworkSample, RenderedPipeJunctionEdge, RenderedPipeJunctionNode, RenderedPipeJunctionScene


BLOCKED_PIPE_X_RGB: Tuple[int, int, int] = (220, 38, 38)


def _resolve_pipe_panel_geometry(render_params: GraphRenderParams) -> Dict[str, Any]:
    """Resolve the single-panel pipe scene geometry."""

    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    panel = (margin, margin, width - margin, height - margin)
    title_band_height = max(40, int(round(float(render_params.panel_title_font_size_px) * 1.8)))
    title_band = (panel[0], panel[1], panel[2], panel[1] + title_band_height)
    content = (
        panel[0] + int(render_params.panel_padding_px),
        title_band[3] + max(12, int(render_params.panel_padding_px // 2)),
        panel[2] - int(render_params.panel_padding_px),
        panel[3] - int(render_params.panel_padding_px),
    )
    return {
        "canvas_size": [int(width), int(height)],
        "scene_panel_xyxy": [int(value) for value in panel],
        "title_band_xyxy": [int(value) for value in title_band],
        "scene_content_xyxy": [int(value) for value in content],
    }


def _draw_panel(
    image: Image.Image,
    *,
    panel_geometry: Mapping[str, Any],
    render_params: GraphRenderParams,
    scene_title: str,
    layout_seed: int,
) -> None:
    """Draw panel chrome for the pipe graph."""

    draw = ImageDraw.Draw(image)
    panel = tuple(int(value) for value in panel_geometry["scene_panel_xyxy"])
    draw.rounded_rectangle(
        panel,
        radius=max(0, int(render_params.panel_corner_radius_px)),
        fill=tuple(int(value) for value in render_params.panel_fill_rgb),
        outline=tuple(int(value) for value in render_params.panel_border_rgb),
        width=2,
    )
    title_band = tuple(int(value) for value in panel_geometry["title_band_xyxy"])
    title_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.pipe_network.panel_title_text",
        role="graph_panel_title_text",
        surface_rgbs=(tuple(int(value) for value in render_params.panel_fill_rgb),),
        preferred_rgbs=(tuple(int(value) for value in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    draw_centered_readable_text(
        draw,
        text=str(scene_title),
        center=(0.5 * float(title_band[0] + title_band[2]), 0.5 * float(title_band[1] + title_band[3])),
        font=load_font(
            int(render_params.panel_title_font_size_px),
            bold=True,
            font_family=str(render_params.font_family or ""),
        ),
        style=title_style,
        stroke_width=2,
    )


def _clamp_channel(value: float) -> int:
    """Clamp one color channel to an integer RGB value."""

    return max(0, min(255, int(round(float(value)))))


def _mix_rgb(left: Sequence[int], right: Sequence[int], amount: float) -> Tuple[int, int, int]:
    """Linearly mix two RGB colors."""

    t = max(0.0, min(1.0, float(amount)))
    return tuple(
        _clamp_channel((float(left[index]) * (1.0 - t)) + (float(right[index]) * t))
        for index in range(3)
    )


def _resolve_pipe_visual_style(*, render_params: GraphRenderParams, layout_seed: int) -> Dict[str, Any]:
    """Resolve a deterministic physical-pipe visual style."""

    rng = random.Random((int(layout_seed) ^ 0x5A17C0DE) & 0xFFFFFFFF)
    style_id = PIPE_VISUAL_STYLE_IDS[int(rng.randrange(len(PIPE_VISUAL_STYLE_IDS)))]
    style_map: Dict[str, Dict[str, Tuple[int, int, int]]] = {
        "industrial_steel": {
            "board_fill_rgb": (239, 242, 245),
            "board_grid_rgb": (215, 222, 229),
            "tube_shadow_rgb": (98, 107, 116),
            "tube_outline_rgb": (54, 64, 74),
            "tube_fill_rgb": (151, 165, 177),
            "tube_highlight_rgb": (232, 237, 242),
            "blocked_outline_rgb": (100, 104, 111),
            "blocked_fill_rgb": (185, 188, 193),
            "blocked_highlight_rgb": (243, 244, 246),
            "blocked_marker_rgb": (182, 65, 55),
            "fitting_shadow_rgb": (117, 124, 132),
            "fitting_outer_rgb": (58, 68, 78),
            "fitting_ring_rgb": (122, 137, 150),
            "fitting_inner_rgb": (238, 242, 246),
            "bolt_rgb": (45, 53, 61),
        },
        "copper_plumbing": {
            "board_fill_rgb": (249, 244, 235),
            "board_grid_rgb": (224, 210, 192),
            "tube_shadow_rgb": (118, 76, 51),
            "tube_outline_rgb": (99, 57, 35),
            "tube_fill_rgb": (190, 111, 65),
            "tube_highlight_rgb": (249, 190, 124),
            "blocked_outline_rgb": (111, 87, 75),
            "blocked_fill_rgb": (187, 178, 170),
            "blocked_highlight_rgb": (248, 241, 233),
            "blocked_marker_rgb": (154, 54, 48),
            "fitting_shadow_rgb": (127, 85, 59),
            "fitting_outer_rgb": (110, 62, 38),
            "fitting_ring_rgb": (187, 111, 67),
            "fitting_inner_rgb": (255, 241, 222),
            "bolt_rgb": (86, 48, 30),
        },
        "teal_plant": {
            "board_fill_rgb": (237, 246, 244),
            "board_grid_rgb": (204, 224, 220),
            "tube_shadow_rgb": (60, 102, 101),
            "tube_outline_rgb": (30, 74, 76),
            "tube_fill_rgb": (62, 148, 143),
            "tube_highlight_rgb": (197, 239, 232),
            "blocked_outline_rgb": (90, 104, 107),
            "blocked_fill_rgb": (177, 191, 191),
            "blocked_highlight_rgb": (242, 250, 248),
            "blocked_marker_rgb": (174, 66, 58),
            "fitting_shadow_rgb": (70, 116, 113),
            "fitting_outer_rgb": (31, 78, 79),
            "fitting_ring_rgb": (66, 145, 139),
            "fitting_inner_rgb": (230, 250, 246),
            "bolt_rgb": (25, 58, 60),
        },
        "blueprint_tubes": {
            "board_fill_rgb": (226, 238, 250),
            "board_grid_rgb": (181, 203, 226),
            "tube_shadow_rgb": (51, 83, 121),
            "tube_outline_rgb": (28, 58, 96),
            "tube_fill_rgb": (74, 128, 183),
            "tube_highlight_rgb": (213, 235, 255),
            "blocked_outline_rgb": (77, 92, 113),
            "blocked_fill_rgb": (166, 181, 201),
            "blocked_highlight_rgb": (236, 244, 252),
            "blocked_marker_rgb": (178, 66, 62),
            "fitting_shadow_rgb": (55, 87, 124),
            "fitting_outer_rgb": (33, 67, 106),
            "fitting_ring_rgb": (82, 139, 194),
            "fitting_inner_rgb": (236, 247, 255),
            "bolt_rgb": (23, 50, 82),
        },
    }
    resolved = dict(style_map[str(style_id)])
    resolved["fitting_ring_rgb"] = _mix_rgb(
        resolved["fitting_ring_rgb"],
        tuple(int(value) for value in render_params.node_fill_rgb),
        0.22,
    )
    treatment_rng = random.Random((int(layout_seed) ^ 0x3D71B0A9) & 0xFFFFFFFF)
    if str(style_id) == "blueprint_tubes":
        treatment_support = ("plain_panel", "plate_seams", "blueprint_grid")
    else:
        treatment_support = ("plain_panel", "plate_seams", "perforated_panel")
    board_treatment = str(treatment_support[int(treatment_rng.randrange(len(treatment_support)))])
    return {
        "style_id": str(style_id),
        "board_treatment": str(board_treatment),
        **{key: tuple(int(channel) for channel in value) for key, value in resolved.items()},
    }


def _pipe_style_metadata(style: Mapping[str, Any]) -> Dict[str, Any]:
    """Return JSON-friendly pipe style metadata."""

    out: Dict[str, Any] = {
        "style_id": str(style.get("style_id", "")),
        "board_treatment": str(style.get("board_treatment", "")),
    }
    for key, value in style.items():
        if key in {"style_id", "board_treatment"}:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            out[str(key)] = [int(channel) for channel in value[:3]]
    return out


def _draw_pipe_board_background(
    draw: ImageDraw.ImageDraw,
    *,
    content_bbox: Sequence[int],
    style: Mapping[str, Any],
) -> None:
    """Draw the physical board that the pipe network sits on."""

    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    fill = tuple(int(value) for value in style["board_fill_rgb"])
    grid = tuple(int(value) for value in style["board_grid_rgb"])
    draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=fill, outline=_mix_rgb(grid, (40, 48, 58), 0.18), width=2)
    treatment = str(style.get("board_treatment", "plain_panel"))
    if treatment == "blueprint_grid":
        step = 56
        for x in range(x0 + step, x1, step):
            draw.line((x, y0 + 8, x, y1 - 8), fill=grid, width=1)
        for y in range(y0 + step, y1, step):
            draw.line((x0 + 8, y, x1 - 8, y), fill=grid, width=1)
    elif treatment == "plate_seams":
        seam = _mix_rgb(grid, (40, 48, 58), 0.22)
        for fraction in (0.34, 0.67):
            x = int(round(float(x0) + (float(x1 - x0) * fraction)))
            draw.line((x, y0 + 18, x, y1 - 18), fill=seam, width=2)
        y = int(round(float(y0) + (float(y1 - y0) * 0.52)))
        draw.line((x0 + 18, y, x1 - 18, y), fill=seam, width=2)
    elif treatment == "perforated_panel":
        dot_fill = _mix_rgb(grid, (45, 52, 60), 0.35)
        spacing = 62
        radius = 2
        for y in range(y0 + 38, y1 - 18, spacing):
            for x in range(x0 + 38, x1 - 18, spacing):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=dot_fill)
    screw_radius = 4
    for sx, sy in (
        (x0 + 15, y0 + 15),
        (x1 - 15, y0 + 15),
        (x0 + 15, y1 - 15),
        (x1 - 15, y1 - 15),
    ):
        draw.ellipse(
            (sx - screw_radius, sy - screw_radius, sx + screw_radius, sy + screw_radius),
            fill=_mix_rgb(grid, (45, 52, 60), 0.40),
            outline=_mix_rgb(grid, (0, 0, 0), 0.25),
            width=1,
        )


def _grid_cell_centers(
    *,
    grid_shape_variant: str,
    content_bbox: Sequence[int],
    node_radius_px: int,
) -> Dict[Tuple[int, int], Point]:
    """Return pixel centers for every grid cell."""

    rows, cols = parse_pipe_grid_shape(str(grid_shape_variant))
    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    inset = max(8, int(node_radius_px) + 10)
    usable_x0 = x0 + inset
    usable_y0 = y0 + inset
    usable_x1 = x1 - inset
    usable_y1 = y1 - inset
    centers: Dict[Tuple[int, int], Point] = {}
    for row in range(int(rows)):
        for col in range(int(cols)):
            x = usable_x0 + ((usable_x1 - usable_x0) * (float(col) / float(max(1, int(cols) - 1))))
            y = usable_y0 + ((usable_y1 - usable_y0) * (float(row) / float(max(1, int(rows) - 1))))
            centers[(int(row), int(col))] = (int(round(x)), int(round(y)))
    return centers


def _trim_segment_to_node_radius(start: Point, end: Point, *, radius: int) -> Tuple[Point, Point]:
    """Trim one pipe segment so it meets node boundaries."""

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = float(math.hypot(dx, dy))
    if norm <= 1e-6:
        return (start, end)
    ux = float(dx / norm)
    uy = float(dy / norm)
    trim = float(max(1, int(radius) - 1))
    return (
        (int(round(x0 + (ux * trim))), int(round(y0 + (uy * trim)))),
        (int(round(x1 - (ux * trim))), int(round(y1 - (uy * trim)))),
    )


def _draw_blocked_pipe_marker(
    draw: ImageDraw.ImageDraw,
    *,
    segment: Tuple[Point, Point],
    color_rgb: Sequence[int],
    width_px: int,
) -> None:
    """Draw a clear X-shaped blocked-pipe marker at segment midpoint."""

    start, end = segment
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    dx = x1 - x0
    dy = y1 - y0
    norm = max(1.0, math.hypot(dx, dy))
    ux = dx / norm
    uy = dy / norm
    px = -dy / norm
    py = dx / norm
    half = max(16.0, min(max(22.0, float(width_px) * 3.2), norm * 0.34))
    stroke_width = max(5, int(round(float(width_px) * 0.85)))
    halo_width = stroke_width + 5
    line_pairs = (
        (
            (mx - ux * half - px * half, my - uy * half - py * half),
            (mx + ux * half + px * half, my + uy * half + py * half),
        ),
        (
            (mx - ux * half + px * half, my - uy * half + py * half),
            (mx + ux * half - px * half, my + uy * half - py * half),
        ),
    )
    for left, right in line_pairs:
        draw.line(
            (left[0], left[1], right[0], right[1]),
            fill=(248, 250, 252),
            width=halo_width,
        )
    for left, right in line_pairs:
        draw.line(
            (left[0], left[1], right[0], right[1]),
            fill=tuple(int(value) for value in color_rgb),
            width=stroke_width,
        )


def _offset_segment(segment: Tuple[Point, Point], *, offset_px: float) -> Tuple[Point, Point]:
    """Offset one pipe segment perpendicular to its direction."""

    start, end = segment
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    norm = max(1.0, math.hypot(dx, dy))
    px = -dy / norm
    py = dx / norm
    return (
        (int(round(float(start[0]) + (px * float(offset_px)))), int(round(float(start[1]) + (py * float(offset_px))))),
        (int(round(float(end[0]) + (px * float(offset_px)))), int(round(float(end[1]) + (py * float(offset_px))))),
    )


def _draw_pipe_tube(
    draw: ImageDraw.ImageDraw,
    *,
    segment: Tuple[Point, Point],
    width_px: int,
    outline_rgb: Sequence[int],
    fill_rgb: Sequence[int],
    highlight_rgb: Sequence[int],
    shadow_rgb: Sequence[int],
) -> None:
    """Draw one thick cylindrical-looking pipe tube."""

    start, end = segment
    tube_width = max(10, int(width_px))
    outer_width = int(tube_width + max(6, round(tube_width * 0.34)))
    shadow_segment = _offset_segment((start, end), offset_px=max(2.0, float(tube_width) * 0.15))
    draw.line(
        (shadow_segment[0][0], shadow_segment[0][1], shadow_segment[1][0], shadow_segment[1][1]),
        fill=tuple(int(value) for value in shadow_rgb),
        width=outer_width + 4,
    )
    draw.line((start[0], start[1], end[0], end[1]), fill=tuple(int(value) for value in outline_rgb), width=outer_width)
    draw.line((start[0], start[1], end[0], end[1]), fill=tuple(int(value) for value in fill_rgb), width=tube_width)
    highlight_segment = _offset_segment((start, end), offset_px=-max(2.0, float(tube_width) * 0.18))
    draw.line(
        (highlight_segment[0][0], highlight_segment[0][1], highlight_segment[1][0], highlight_segment[1][1]),
        fill=tuple(int(value) for value in highlight_rgb),
        width=max(3, int(round(float(tube_width) * 0.18))),
    )


def _draw_blocked_valve_marker(
    draw: ImageDraw.ImageDraw,
    *,
    segment: Tuple[Point, Point],
    style: Mapping[str, Any],
    width_px: int,
) -> None:
    """Draw a valve/plate marker for one blocked pipe."""

    start, end = segment
    mx = int(round(0.5 * float(start[0] + end[0])))
    my = int(round(0.5 * float(start[1] + end[1])))
    dx = abs(int(end[0] - start[0]))
    dy = abs(int(end[1] - start[1]))
    half_long = max(18, int(round(float(width_px) * 1.05)))
    half_short = max(8, int(round(float(width_px) * 0.42)))
    if dx >= dy:
        valve_box = (mx - half_short, my - half_long, mx + half_short, my + half_long)
    else:
        valve_box = (mx - half_long, my - half_short, mx + half_long, my + half_short)
    marker_fill = tuple(int(value) for value in style["blocked_highlight_rgb"])
    marker_outline = _mix_rgb(tuple(int(value) for value in style["blocked_outline_rgb"]), (20, 20, 20), 0.30)
    draw.rounded_rectangle(valve_box, radius=6, fill=marker_fill, outline=marker_outline, width=3)
    _draw_blocked_pipe_marker(
        draw,
        segment=segment,
        color_rgb=BLOCKED_PIPE_X_RGB,
        width_px=max(4, int(round(float(width_px) * 0.42))),
    )


def _draw_pipe_fitting(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    radius: int,
    style: Mapping[str, Any],
) -> BBox:
    """Draw one flanged pipe junction fitting."""

    cx, cy = int(center[0]), int(center[1])
    outer_radius = int(radius)
    shadow_offset = max(2, int(round(float(outer_radius) * 0.10)))
    shadow = tuple(int(value) for value in style["fitting_shadow_rgb"])
    outer = tuple(int(value) for value in style["fitting_outer_rgb"])
    ring = tuple(int(value) for value in style["fitting_ring_rgb"])
    inner = tuple(int(value) for value in style["fitting_inner_rgb"])
    bolt = tuple(int(value) for value in style["bolt_rgb"])
    shadow_box = (
        cx - outer_radius + shadow_offset,
        cy - outer_radius + shadow_offset,
        cx + outer_radius + shadow_offset,
        cy + outer_radius + shadow_offset,
    )
    bbox = (cx - outer_radius, cy - outer_radius, cx + outer_radius, cy + outer_radius)
    draw.ellipse(shadow_box, fill=shadow)
    draw.ellipse(bbox, fill=outer, outline=_mix_rgb(outer, (0, 0, 0), 0.35), width=2)
    ring_inset = max(4, int(round(float(outer_radius) * 0.18)))
    ring_box = (
        cx - outer_radius + ring_inset,
        cy - outer_radius + ring_inset,
        cx + outer_radius - ring_inset,
        cy + outer_radius - ring_inset,
    )
    draw.ellipse(ring_box, fill=ring, outline=_mix_rgb(ring, (0, 0, 0), 0.22), width=2)
    inner_radius = max(13, int(round(float(outer_radius) * 0.58)))
    inner_box = (cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius)
    draw.ellipse(inner_box, fill=inner, outline=_mix_rgb(ring, (0, 0, 0), 0.35), width=2)
    bolt_radius = max(2, int(round(float(outer_radius) * 0.08)))
    bolt_ring_radius = max(inner_radius + 4, int(round(float(outer_radius) * 0.78)))
    for angle_index in range(8):
        theta = (math.tau * float(angle_index)) / 8.0
        bx = int(round(float(cx) + (math.cos(theta) * float(bolt_ring_radius))))
        by = int(round(float(cy) + (math.sin(theta) * float(bolt_ring_radius))))
        draw.ellipse(
            (bx - bolt_radius, by - bolt_radius, bx + bolt_radius, by + bolt_radius),
            fill=bolt,
            outline=_mix_rgb(bolt, (255, 255, 255), 0.25),
            width=1,
        )
    return bbox


def _pipe_text_legibility_records(
    render_params: GraphRenderParams,
    *,
    layout_seed: int,
    style: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Return pipe-network required text styles for panel metadata."""

    existing_records: list[Dict[str, Any]] = []
    if isinstance(render_params.text_legibility, Mapping):
        raw_records = render_params.text_legibility.get("records")
        if isinstance(raw_records, list):
            skip_roles = {"graph_panel_title_text", "graph_node_label_text"}
            existing_records = [
                dict(record)
                for record in raw_records
                if isinstance(record, Mapping) and str(record.get("role")) not in skip_roles
            ]
    title_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.pipe_network.panel_title_text",
        role="graph_panel_title_text",
        surface_rgbs=(tuple(int(value) for value in render_params.panel_fill_rgb),),
        preferred_rgbs=(tuple(int(value) for value in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    junction_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.pipe_network.junction_label_text",
        role="graph_node_label_text",
        surface_rgbs=(tuple(int(value) for value in style["fitting_inner_rgb"]),),
        preferred_rgbs=(
            tuple(int(value) for value in render_params.label_text_rgb),
            tuple(int(value) for value in render_params.label_stroke_rgb),
            (255, 255, 255),
            (10, 14, 22),
        ),
        min_contrast_ratio=4.0,
        min_lab_distance=24.0,
    )
    return tuple([*existing_records, title_style.metadata(), junction_style.metadata()])


def render_pipe_network_scene(
    *,
    pipe_sample: PipeJunctionNetworkSample,
    render_params: GraphRenderParams,
    base_image: Image.Image,
    scene_title: str = "Pipe Junction Board",
    layout_seed: int = 0,
) -> RenderedPipeJunctionScene:
    """Render one sampled pipe-junction graph scene."""

    image = base_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    pipe_style = _resolve_pipe_visual_style(render_params=render_params, layout_seed=int(layout_seed))
    panel_geometry = _resolve_pipe_panel_geometry(render_params)
    if isinstance(render_params.information_scene_style, Mapping):
        panel_geometry["information_scene_style"] = dict(render_params.information_scene_style)
    panel_geometry["text_legibility"] = text_legibility_summary_from_records(
        _pipe_text_legibility_records(render_params, layout_seed=int(layout_seed), style=pipe_style)
    )
    panel_geometry["pipe_visual_style"] = _pipe_style_metadata(pipe_style)
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
    _draw_panel(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        scene_title=str(scene_title),
        layout_seed=int(layout_seed),
    )
    block_context_elements = list(
        draw_graph_context_text_blocks(
            image,
            panel_geometry=panel_geometry,
            render_params=render_params,
            layout_seed=int(layout_seed),
        )
    )
    chip_context_elements = draw_graph_context_text_chips(
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
    apply_graph_content_layout_jitter(
        panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )

    content_bbox = tuple(int(value) for value in panel_geometry["scene_content_xyxy"])
    _draw_pipe_board_background(draw, content_bbox=content_bbox, style=pipe_style)
    grid_centers = _grid_cell_centers(
        grid_shape_variant=str(pipe_sample.grid_shape_variant),
        content_bbox=content_bbox,
        node_radius_px=int(render_params.node_radius_px),
    )
    node_centers = {
        int(node): grid_centers[tuple(int(value) for value in cell)]
        for node, cell in pipe_sample.node_grid_cells.items()
    }

    open_width = max(18, int(render_params.edge_width_px) * 4 + 4)
    blocked_width = max(16, int(render_params.edge_width_px) * 4)

    rendered_edges: list[RenderedPipeJunctionEdge] = []
    all_edges = [(edge, "blocked") for edge in pipe_sample.blocked_edges] + [(edge, "open") for edge in pipe_sample.open_edges]
    for edge, pipe_state in all_edges:
        left, right = int(edge[0]), int(edge[1])
        start, end = _trim_segment_to_node_radius(
            node_centers[int(left)],
            node_centers[int(right)],
            radius=int(render_params.node_radius_px),
        )
        if str(pipe_state) == "blocked":
            _draw_pipe_tube(
                draw,
                segment=(start, end),
                width_px=int(blocked_width),
                outline_rgb=pipe_style["blocked_outline_rgb"],
                fill_rgb=pipe_style["blocked_fill_rgb"],
                highlight_rgb=pipe_style["blocked_highlight_rgb"],
                shadow_rgb=pipe_style["fitting_shadow_rgb"],
            )
            _draw_blocked_valve_marker(
                draw,
                segment=(start, end),
                style=pipe_style,
                width_px=int(blocked_width),
            )
        else:
            _draw_pipe_tube(
                draw,
                segment=(start, end),
                width_px=int(open_width),
                outline_rgb=pipe_style["tube_outline_rgb"],
                fill_rgb=pipe_style["tube_fill_rgb"],
                highlight_rgb=pipe_style["tube_highlight_rgb"],
                shadow_rgb=pipe_style["tube_shadow_rgb"],
            )
        labeled_edge = label_edge(pipe_sample.label_by_node, canonical_node_edge(left, right))
        rendered_edges.append(
            RenderedPipeJunctionEdge(
                edge_id=f"pipe_{labeled_edge[0]}_{labeled_edge[1]}_{pipe_state}",
                node_u_label=str(labeled_edge[0]),
                node_v_label=str(labeled_edge[1]),
                pipe_state=str(pipe_state),
                segment_px=(tuple(start), tuple(end)),
            )
        )

    rendered_nodes: list[RenderedPipeJunctionNode] = []
    fitting_radius = max(
        int(render_params.node_radius_px) + 7,
        int(round(float(render_params.node_radius_px) * 1.35)),
    )
    label_inner_radius = max(14, int(round(float(fitting_radius) * 0.58)))
    label_font = fit_font_to_box(
        draw,
        text=max(pipe_sample.node_labels, key=len),
        max_width=max(10, int(label_inner_radius * 1.65)),
        max_height=max(10, int(label_inner_radius * 1.20)),
        max_size_px=int(render_params.label_font_size_px),
        min_size_px=11,
        bold=True,
        font_family=str(render_params.font_family or ""),
    )
    resolved_font_size = int(getattr(label_font, "size", render_params.label_font_size_px))
    stroke_width = max(1, int(round(float(resolved_font_size) * 0.10)))
    label_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.pipe_network.junction_label_text",
        role="graph_node_label_text",
        surface_rgbs=(tuple(int(value) for value in pipe_style["fitting_inner_rgb"]),),
        preferred_rgbs=(
            tuple(int(value) for value in render_params.label_text_rgb),
            tuple(int(value) for value in render_params.label_stroke_rgb),
            (255, 255, 255),
            (10, 14, 22),
        ),
        min_contrast_ratio=4.0,
        min_lab_distance=24.0,
    )
    for node in sorted(pipe_sample.graph.nodes()):
        label = str(pipe_sample.label_by_node[int(node)])
        center = tuple(int(value) for value in node_centers[int(node)])
        bbox = _draw_pipe_fitting(draw, center=center, radius=int(fitting_radius), style=pipe_style)
        draw_centered_readable_text(
            draw,
            text=label,
            center=(float(center[0]), float(center[1])),
            font=label_font,
            style=label_style,
            stroke_width=int(stroke_width),
        )
        open_neighbors = tuple(
            sorted((str(pipe_sample.label_by_node[int(neighbor)]) for neighbor in pipe_sample.graph.neighbors(int(node))), key=graph_label_sort_key)
        )
        rendered_nodes.append(
            RenderedPipeJunctionNode(
                label=label,
                open_degree=int(pipe_sample.graph.degree(int(node))),
                grid_cell=tuple(int(value) for value in pipe_sample.node_grid_cells[int(node)]),
                center_xy=center,
                bbox_xyxy=tuple(int(value) for value in bbox),
                open_neighbors=open_neighbors,
            )
        )

    return RenderedPipeJunctionScene(
        image=image,
        panel_geometry=dict(panel_geometry),
        nodes=tuple(rendered_nodes),
        edges=tuple(rendered_edges),
        grid_shape_variant=str(pipe_sample.grid_shape_variant),
        resolved_label_font_size_px=int(resolved_font_size),
        resolved_label_stroke_width_px=int(stroke_width),
        open_pipe_width_px=int(open_width),
        blocked_pipe_width_px=int(blocked_width),
    )


__all__ = ["BLOCKED_PIPE_X_RGB", "render_pipe_network_scene"]
