"""Rendering and projection helpers for electrostatic-field diagrams."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import (
    PHYSICS_ELECTROSTATICS_SEMANTIC_COLORS,
    PhysicsDiagramStyle,
    make_physics_diagram_background,
    physics_electrostatics_theme_from_diagram_style,
    resolve_physics_diagram_style,
)
from trace_tasks.tasks.physics.shared.vector_arrows import (
    arrow_bbox,
    centered_arrow_endpoints,
    draw_arrow_with_bbox,
)
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many as _bbox_union
from trace_tasks.tasks.shared.drawing import (
    draw_arrow,
    draw_centered_text,
    draw_dashed_line,
    draw_rounded_rect,
)
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .state import (
    Charge,
    DIRECTION_VECTORS,
    OPTION_LETTERS,
    RenderDefaults,
    RenderedElectrostaticCore,
    RenderedElectrostaticScene,
    SCENE_ID,
    SCENE_MODE_DIRECTION,
    SCENE_MODE_ZERO_FIELD,
    SceneSpec,
    PotentialCharge,
)


_DEFAULTS = RenderDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
RENDER_INT_KEYS = (
    "canvas_width",
    "canvas_height",
    "board_left_px",
    "board_top_px",
    "board_width_px",
    "board_height_px",
    "coord_extent",
    "grid_line_width_px",
    "dense_grid_line_width_px",
    "axis_width_px",
    "charge_radius_px",
    "point_radius_px",
    "candidate_point_radius_px",
    "label_font_size_px",
    "option_font_size_px",
    "note_font_size_px",
    "charge_font_size_px",
    "option_panel_left_px",
    "option_panel_top_px",
    "option_cell_width_px",
    "option_cell_height_px",
    "option_cell_gap_x_px",
    "option_cell_gap_y_px",
    "option_arrow_length_px",
    "option_arrow_width_px",
    "option_arrow_head_length_px",
    "option_arrow_head_width_px",
)

def _annotation_entity_key_map(scene_spec: SceneSpec) -> Dict[str, str]:
    """Return neutral visible annotation keys by rendered entity id."""

    key_by_entity_id: Dict[str, str] = {}
    charge_index = 1
    for entity_id in scene_spec.annotation_entity_ids:
        if str(entity_id) == "query_point":
            key_by_entity_id[str(entity_id)] = "P"
        elif str(entity_id).startswith("candidate_"):
            key_by_entity_id[str(entity_id)] = "zero_point"
        else:
            key_by_entity_id[str(entity_id)] = f"Q{int(charge_index)}"
            charge_index += 1
    return key_by_entity_id


def _board_bbox(render_defaults: Mapping[str, Any]) -> List[float]:
    """Return the main coordinate board bbox."""

    return [
        float(render_defaults["board_left_px"]),
        float(render_defaults["board_top_px"]),
        float(render_defaults["board_left_px"]) + float(render_defaults["board_width_px"]),
        float(render_defaults["board_top_px"]) + float(render_defaults["board_height_px"]),
    ]


def _electrostatics_content_bbox(render_defaults: Mapping[str, Any], *, scene_mode: str) -> List[float]:
    """Return a conservative bbox for the whole rendered electrostatics diagram."""

    board = _board_bbox(render_defaults)
    left = float(board[0]) - 46.0
    top = float(board[1]) - 50.0
    right = float(board[2]) + 58.0
    bottom = float(board[3]) + 50.0
    if str(scene_mode) == SCENE_MODE_DIRECTION:
        option_left = float(render_defaults["option_panel_left_px"])
        option_top = float(render_defaults["option_panel_top_px"])
        option_right = option_left + (2.0 * float(render_defaults["option_cell_width_px"])) + float(render_defaults["option_cell_gap_x_px"])
        option_bottom = option_top + (4.0 * float(render_defaults["option_cell_height_px"])) + (3.0 * float(render_defaults["option_cell_gap_y_px"]))
        left = min(left, option_left)
        top = min(top, option_top)
        right = max(right, option_right)
        bottom = max(bottom, option_bottom)
    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def _resolve_electrostatics_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    jitter_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    scene_mode: str,
    namespace: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve a whole-diagram offset before rendering and annotation projection."""

    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    content_bbox = _electrostatics_content_bbox(render_defaults, scene_mode=str(scene_mode))
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        jitter_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    min_margin = int(jitter.get("min_margin_px", 24))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if int(min_dx) > int(max_dx):
        min_dx = 0
        max_dx = 0
    if int(min_dy) > int(max_dy):
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))

    adjusted = dict(render_defaults)
    for key in ("board_left_px", "option_panel_left_px"):
        adjusted[key] = int(adjusted[key]) + int(dx)
    for key in ("board_top_px", "option_panel_top_px"):
        adjusted[key] = int(adjusted[key]) + int(dy)

    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_electrostatics_diagram_offset",
            "content_bbox_px": list(content_bbox),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "default_origin_px": [round(float(content_left), 3), round(float(content_top), 3)],
            "final_origin_px": [round(float(content_left) + float(dx), 3), round(float(content_top) + float(dy), 3)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement


def _coord_to_px(*, bbox: Sequence[float], x: float, y: float, coord_extent: int) -> Tuple[float, float]:
    """Map board coordinates to pixel coordinates."""

    left, top, right, bottom = [float(value) for value in bbox[:4]]
    extent = max(1, int(coord_extent))
    px = float(left + ((float(x) + float(extent)) / float(2 * extent)) * (right - left))
    py = float(bottom - ((float(y) + float(extent)) / float(2 * extent)) * (bottom - top))
    return px, py


def _point_bbox(center: Tuple[float, float], *, radius_px: float, padding_px: float = 0.0) -> List[float]:
    """Return a bbox around one rendered circular marker."""

    cx, cy = float(center[0]), float(center[1])
    radius = float(radius_px) + float(padding_px)
    return [
        round(float(cx - radius), 3),
        round(float(cy - radius), 3),
        round(float(cx + radius), 3),
        round(float(cy + radius), 3),
    ]


def _expand_bbox(bbox: Sequence[float], padding_px: float) -> List[float]:
    """Return one bbox expanded by a constant pixel margin."""

    return [
        round(float(bbox[0]) - float(padding_px), 3),
        round(float(bbox[1]) - float(padding_px), 3),
        round(float(bbox[2]) + float(padding_px), 3),
        round(float(bbox[3]) + float(padding_px), 3),
    ]


def _bbox_overlaps(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether two axis-aligned bboxes overlap."""

    return not (
        float(left[2]) <= float(right[0])
        or float(left[0]) >= float(right[2])
        or float(left[3]) <= float(right[1])
        or float(left[1]) >= float(right[3])
    )


def _bbox_intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the positive intersection area between two bboxes."""

    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _bbox_inside(inner: Sequence[float], outer: Sequence[float], padding_px: float = 0.0) -> bool:
    """Return whether one bbox stays inside another with optional margin."""

    return (
        float(inner[0]) >= float(outer[0]) + float(padding_px)
        and float(inner[1]) >= float(outer[1]) + float(padding_px)
        and float(inner[2]) <= float(outer[2]) - float(padding_px)
        and float(inner[3]) <= float(outer[3]) - float(padding_px)
    )


def _text_tag_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    stroke_width_px: int,
) -> List[float]:
    """Return the outer bbox for one rounded text tag before drawing it."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width_px)))
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    pad_x = 11.0
    pad_y = 7.0
    center_x, center_y = float(center[0]), float(center[1])
    return [
        round(float(center_x - (0.5 * text_width) - pad_x), 3),
        round(float(center_y - (0.5 * text_height) - pad_y), 3),
        round(float(center_x + (0.5 * text_width) + pad_x), 3),
        round(float(center_y + (0.5 * text_height) + pad_y), 3),
    ]


def _draw_text_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    stroke_width_px: int,
) -> List[float]:
    """Draw one rounded text tag and return its outer bbox."""

    center_x, center_y = float(center[0]), float(center[1])
    tag_bbox = _text_tag_bbox(draw, text=str(text), center=(center_x, center_y), font=font, stroke_width_px=int(stroke_width_px))
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in tag_bbox),
        radius=9,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(1, int(stroke_width_px)),
    )
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(float(center_x), float(center_y)),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=1,
    )
    return _bbox_union(tag_bbox, text_draw_bbox)


def _draw_board(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    theme,
    diagram_style: PhysicsDiagramStyle,
    label_font,
) -> Dict[str, Any]:
    """Draw the coordinate board and return bbox metadata."""

    left, top, right, bottom = [float(value) for value in bbox[:4]]
    extent = int(render_defaults["coord_extent"])
    grid_width = int(render_defaults["dense_grid_line_width_px"]) if str(scene_variant) == "dense_grid" else int(render_defaults["grid_line_width_px"])
    board_fill = tuple(int(value) for value in theme.board_alt_fill_rgb) if str(scene_variant) == "paper_grid" else tuple(int(value) for value in theme.board_fill_rgb)
    draw.rectangle(tuple(float(value) for value in bbox), fill=board_fill)
    frame_mode = str(diagram_style.frame_mode)
    if frame_mode == "plain_outline":
        draw.rectangle(
            tuple(float(value) for value in bbox),
            outline=tuple(int(value) for value in theme.board_outline_rgb),
            width=max(1, int(diagram_style.panel_border_width_px)),
        )
    elif frame_mode == "matching_outline":
        draw.rectangle(
            tuple(float(value) for value in bbox),
            outline=tuple(int(value) for value in theme.board_outline_rgb),
            width=max(2, int(diagram_style.panel_border_width_px)),
        )
        inset = 7.0
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            outline=tuple(int(value) for value in diagram_style.canvas_accent_rgb),
            width=1,
        )
    tick_bboxes: List[List[float]] = []
    for coord in range(-extent, extent + 1):
        x, _ = _coord_to_px(bbox=bbox, x=coord, y=0, coord_extent=extent)
        _, y = _coord_to_px(bbox=bbox, x=0, y=coord, coord_extent=extent)
        draw.line([(float(x), float(top)), (float(x), float(bottom))], fill=tuple(int(value) for value in theme.grid_rgb), width=grid_width)
        draw.line([(float(left), float(y)), (float(right), float(y))], fill=tuple(int(value) for value in theme.grid_rgb), width=grid_width)
        if coord in {-4, -2, 0, 2, 4}:
            xb = draw_centered_text(
                draw,
                text=str(coord),
                center=(float(x), float(bottom + 22.0)),
                font=label_font,
                fill=tuple(int(value) for value in theme.axis_text_rgb),
                stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
                stroke_width=1,
            )
            yb = draw_centered_text(
                draw,
                text=str(coord),
                center=(float(left - 24.0), float(y)),
                font=label_font,
                fill=tuple(int(value) for value in theme.axis_text_rgb),
                stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
                stroke_width=1,
            )
            tick_bboxes.extend([list(xb), list(yb)])
    origin = _coord_to_px(bbox=bbox, x=0, y=0, coord_extent=extent)
    x_end = _coord_to_px(bbox=bbox, x=extent, y=0, coord_extent=extent)
    y_end = _coord_to_px(bbox=bbox, x=0, y=extent, coord_extent=extent)
    draw_arrow(
        draw,
        start=(float(left), float(origin[1])),
        end=(float(x_end[0] + 30.0), float(origin[1])),
        fill=tuple(int(value) for value in theme.axis_rgb),
        width=int(render_defaults["axis_width_px"]),
        head_length_px=22.0,
        head_width_px=18.0,
    )
    draw_arrow(
        draw,
        start=(float(origin[0]), float(bottom)),
        end=(float(origin[0]), float(y_end[1] - 30.0)),
        fill=tuple(int(value) for value in theme.axis_rgb),
        width=int(render_defaults["axis_width_px"]),
        head_length_px=22.0,
        head_width_px=18.0,
    )
    x_label = draw_centered_text(
        draw,
        text="+x",
        center=(float(right + 22.0), float(origin[1] + 26.0)),
        font=label_font,
        fill=tuple(int(value) for value in theme.axis_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
        stroke_width=1,
    )
    y_label = draw_centered_text(
        draw,
        text="+y",
        center=(float(origin[0] - 28.0), float(top - 20.0)),
        font=label_font,
        fill=tuple(int(value) for value in theme.axis_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
        stroke_width=1,
    )
    return {
        "board_bbox_px": [round(float(value), 3) for value in bbox],
        "axis_bbox_px": _bbox_union(bbox, x_label, y_label, *tick_bboxes),
        "tick_label_bboxes_px": [list(bbox_value) for bbox_value in tick_bboxes],
    }


def _charge_key_value_text(display_label: str, charge_value: int) -> str:
    """Return one compact visible label binding charge key to charge value."""

    return f"{str(display_label)}={int(charge_value):+d}"


def _draw_center_marker_sign(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    sign: str,
    radius_px: float,
    fill_rgb: Sequence[int],
) -> List[float]:
    """Draw a centered plus/minus sign as marker strokes and return its bbox."""

    sign_text = str(sign).strip()
    if sign_text not in {"+", "-"}:
        return [float(center[0]), float(center[1]), float(center[0]), float(center[1])]
    half_length = max(8.0, float(radius_px) * 0.48)
    stroke_width = max(4, int(round(float(radius_px) * 0.16)))
    x, y = float(center[0]), float(center[1])
    color = tuple(int(value) for value in fill_rgb)
    draw.line((x - half_length, y, x + half_length, y), fill=color, width=int(stroke_width))
    if sign_text == "+":
        draw.line((x, y - half_length, x, y + half_length), fill=color, width=int(stroke_width))
    pad = float(stroke_width) / 2.0
    return [
        round(float(x - half_length - pad), 3),
        round(float(y - half_length - pad), 3),
        round(float(x + half_length + pad), 3),
        round(float(y + half_length + pad), 3),
    ]


def _draw_charge(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    coord_extent: int,
    charge: Charge | PotentialCharge,
    display_label: str,
    render_defaults: Mapping[str, Any],
    theme,
    charge_font,
    label_font,
) -> Tuple[Dict[str, Any], List[float]]:
    """Draw one point charge and return entity/render bbox."""

    center = _coord_to_px(bbox=bbox, x=int(charge.x), y=int(charge.y), coord_extent=int(coord_extent))
    radius = float(render_defaults["charge_radius_px"])
    is_positive = int(charge.charge_value) > 0
    fill_rgb = theme.positive_fill_rgb if is_positive else theme.negative_fill_rgb
    outline_rgb = theme.positive_outline_rgb if is_positive else theme.negative_outline_rgb
    text_rgb = theme.positive_text_rgb if is_positive else theme.negative_text_rgb
    circle_bbox = _point_bbox(center, radius_px=radius)
    draw.ellipse(
        tuple(float(value) for value in circle_bbox),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=4,
    )
    sign_bbox = _draw_center_marker_sign(
        draw,
        center=(float(center[0]), float(center[1])),
        sign="+" if is_positive else "-",
        radius_px=radius,
        fill_rgb=tuple(int(value) for value in text_rgb),
    )
    label_y = float(center[1] + radius + 24.0)
    if float(label_y) > float(bbox[3] - 16.0):
        label_y = float(center[1] - radius - 22.0)
    charge_label_text = _charge_key_value_text(str(display_label), int(charge.charge_value))
    label_bbox = _draw_text_tag(
        draw,
        text=str(charge_label_text),
        center=(float(center[0]), label_y),
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=2,
    )
    marker_bbox = _bbox_union(circle_bbox, sign_bbox)
    entity_bbox = _bbox_union(marker_bbox, label_bbox)
    entity = {
        "entity_id": str(charge.charge_id),
        "entity_type": "point_charge",
        "bbox_px": list(entity_bbox),
        "meta": {
            "charge_value": int(charge.charge_value),
            "display_label": str(display_label),
            "x": int(charge.x),
            "y": int(charge.y),
            "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
            "charge_marker_bbox_px": list(marker_bbox),
            "charge_sign_bbox_px": list(sign_bbox),
            "charge_id_label_bbox_px": list(label_bbox),
            "charge_label_bbox_px": list(label_bbox),
            "charge_label_text": str(charge_label_text),
        },
    }
    if isinstance(charge, PotentialCharge):
        entity["meta"]["distance_units"] = int(charge.distance_units)
        entity["meta"]["potential_contribution"] = int(charge.contribution)
    return entity, list(entity_bbox)


def _choose_point_label_center(
    draw: ImageDraw.ImageDraw,
    *,
    marker_center: Tuple[float, float],
    label: str,
    font,
    board_bbox: Sequence[float],
    avoid_bboxes: Sequence[Sequence[float]],
    stroke_width_px: int,
    compact: bool = False,
) -> Tuple[Tuple[float, float], List[float]]:
    """Choose a point-label location that avoids nearby charge/candidate labels."""

    compact_offsets = (
        (17.0, -15.0),
        (-17.0, -15.0),
        (17.0, 16.0),
        (-17.0, 16.0),
        (0.0, -26.0),
        (0.0, 27.0),
        (27.0, 0.0),
        (-27.0, 0.0),
        (34.0, -18.0),
        (-34.0, -18.0),
        (34.0, 18.0),
        (-34.0, 18.0),
        (0.0, -42.0),
        (0.0, 43.0),
        (48.0, 0.0),
        (-48.0, 0.0),
    )
    regular_offsets = (
        (24.0, -22.0),
        (-24.0, -22.0),
        (24.0, 24.0),
        (-24.0, 24.0),
        (0.0, -38.0),
        (0.0, 38.0),
        (40.0, 0.0),
        (-40.0, 0.0),
        (0.0, -64.0),
        (0.0, 64.0),
        (62.0, -42.0),
        (-62.0, -42.0),
        (62.0, 42.0),
        (-62.0, 42.0),
        (88.0, 0.0),
        (-88.0, 0.0),
        (120.0, 0.0),
        (-120.0, 0.0),
        (120.0, -42.0),
        (-120.0, -42.0),
        (120.0, 42.0),
        (-120.0, 42.0),
    )
    offsets = compact_offsets if bool(compact) else regular_offsets
    expanded_avoid = [_expand_bbox(bbox, 8.0) for bbox in avoid_bboxes]
    fallback_center = (float(marker_center[0] + offsets[0][0]), float(marker_center[1] + offsets[0][1]))
    fallback_bbox = _text_tag_bbox(draw, text=str(label), center=fallback_center, font=font, stroke_width_px=int(stroke_width_px))
    best_score = float("inf")
    for dx, dy in offsets:
        center = (float(marker_center[0] + float(dx)), float(marker_center[1] + float(dy)))
        label_bbox = _text_tag_bbox(draw, text=str(label), center=center, font=font, stroke_width_px=int(stroke_width_px))
        if not _bbox_inside(label_bbox, board_bbox, padding_px=4.0):
            continue
        expanded_label = _expand_bbox(label_bbox, 2.0)
        overlap_score = sum(_bbox_intersection_area(expanded_label, avoid) for avoid in expanded_avoid)
        if overlap_score < best_score:
            best_score = float(overlap_score)
            fallback_center = center
            fallback_bbox = list(label_bbox)
        if any(_bbox_overlaps(expanded_label, avoid) for avoid in expanded_avoid):
            continue
        return center, label_bbox
    return fallback_center, fallback_bbox


def _draw_point_marker_with_label_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    coord_extent: int,
    x: int,
    y: int,
    label: str,
    render_defaults: Mapping[str, Any],
    theme,
    point_font,
    target: bool,
    inner_sign: str | None = None,
    compact_label: bool = False,
    avoid_bboxes: Sequence[Sequence[float]] = (),
) -> Tuple[List[float], List[float]]:
    """Draw a labeled point marker and return full and label bboxes."""

    center = _coord_to_px(bbox=bbox, x=int(x), y=int(y), coord_extent=int(coord_extent))
    base_radius = (
        float(render_defaults["point_radius_px"])
        if bool(target)
        else float(render_defaults.get("candidate_point_radius_px", render_defaults["point_radius_px"]))
    )
    radius = float(base_radius) + (4.0 if bool(target) else 0.0)
    fill_rgb = theme.target_fill_rgb if bool(target) else theme.point_fill_rgb
    outline_rgb = theme.target_outline_rgb if bool(target) else theme.point_outline_rgb
    circle_bbox = _point_bbox(center, radius_px=radius)
    draw.ellipse(
        tuple(float(value) for value in circle_bbox),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=3,
    )
    marker_bbox = list(circle_bbox)
    if inner_sign is not None:
        sign_bbox = _draw_center_marker_sign(
            draw,
            center=(float(center[0]), float(center[1])),
            sign=str(inner_sign),
            radius_px=float(radius) * 0.78,
            fill_rgb=tuple(int(value) for value in theme.point_text_rgb),
        )
        marker_bbox = list(_bbox_union(marker_bbox, sign_bbox))
    label_center, _ = _choose_point_label_center(
        draw,
        marker_center=center,
        label=str(label),
        font=point_font,
        board_bbox=bbox,
        avoid_bboxes=avoid_bboxes,
        stroke_width_px=2,
        compact=bool(compact_label),
    )
    label_bbox = _draw_text_tag(
        draw,
        text=str(label),
        center=label_center,
        font=point_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in outline_rgb),
        text_rgb=tuple(int(value) for value in theme.point_text_rgb),
        stroke_width_px=2,
    )
    return list(_bbox_union(marker_bbox, label_bbox)), list(label_bbox)


def _draw_point_marker(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    coord_extent: int,
    x: int,
    y: int,
    label: str,
    render_defaults: Mapping[str, Any],
    theme,
    point_font,
    target: bool,
    inner_sign: str | None = None,
    compact_label: bool = False,
) -> List[float]:
    """Draw a labeled point marker and return its bbox."""

    point_bbox, _ = _draw_point_marker_with_label_bbox(
        draw,
        bbox=bbox,
        coord_extent=int(coord_extent),
        x=int(x),
        y=int(y),
        label=str(label),
        render_defaults=render_defaults,
        theme=theme,
        point_font=point_font,
        target=bool(target),
        inner_sign=inner_sign,
        compact_label=compact_label,
    )
    return list(point_bbox)


def _draw_direction_options(
    draw: ImageDraw.ImageDraw,
    *,
    render_defaults: Mapping[str, Any],
    scenario: _DirectionScenario,
    theme,
    option_font,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Draw the visible option panel while preserving one bbox per arrow choice."""

    entities: List[Dict[str, Any]] = []
    option_bboxes: Dict[str, List[float]] = {}
    option_directions: Dict[str, str] = {}
    for index, letter in enumerate(OPTION_LETTERS):
        col = int(index % 2)
        row = int(index // 2)
        cell_left = float(render_defaults["option_panel_left_px"]) + float(col) * (
            float(render_defaults["option_cell_width_px"]) + float(render_defaults["option_cell_gap_x_px"])
        )
        cell_top = float(render_defaults["option_panel_top_px"]) + float(row) * (
            float(render_defaults["option_cell_height_px"]) + float(render_defaults["option_cell_gap_y_px"])
        )
        cell_bbox = [
            float(cell_left),
            float(cell_top),
            float(cell_left + float(render_defaults["option_cell_width_px"])),
            float(cell_top + float(render_defaults["option_cell_height_px"])),
        ]
        draw_rounded_rect(
            draw,
            tuple(float(value) for value in cell_bbox),
            radius=8,
            fill=tuple(int(value) for value in theme.label_fill_rgb),
            outline=tuple(int(value) for value in theme.option_outline_rgb),
            width=2,
        )
        direction = str(scenario.option_directions[str(letter)])
        length = float(render_defaults["option_arrow_length_px"])
        cx = float((cell_bbox[0] + cell_bbox[2]) / 2.0 + 14.0)
        cy = float((cell_bbox[1] + cell_bbox[3]) / 2.0 + 8.0)
        start, end = centered_arrow_endpoints(
            (cx, cy),
            direction=direction,
            length_px=length,
            direction_vectors=DIRECTION_VECTORS,
            half_fraction=0.42,
        )
        arrow_bbox_value = draw_arrow_with_bbox(
            draw,
            start=start,
            end=end,
            fill=tuple(int(value) for value in theme.option_arrow_rgb),
            width=int(render_defaults["option_arrow_width_px"]),
            head_length_px=float(render_defaults["option_arrow_head_length_px"]),
            head_width_px=float(render_defaults["option_arrow_head_width_px"]),
            padding_px=22.0,
        )
        label_bbox = draw_centered_text(
            draw,
            text=str(letter),
            center=(float(cell_bbox[0] + 24.0), float(cell_bbox[1] + 22.0)),
            font=option_font,
            fill=tuple(int(value) for value in theme.axis_text_rgb),
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.axis_text_rgb)),
            stroke_width=1,
        )
        option_bbox = _bbox_union(cell_bbox, label_bbox, arrow_bbox_value)
        option_bboxes[str(letter)] = list(option_bbox)
        option_directions[str(letter)] = str(direction)
        entities.append(
            {
                "entity_id": f"option_{str(letter)}",
                "entity_type": "candidate_direction_arrow",
                "bbox_px": list(option_bbox),
                "meta": {
                    "option_letter": str(letter),
                    "direction": str(direction),
                    "is_correct": str(direction) == str(scenario.requested_direction),
                },
            }
        )
    return entities, {
        "option_bboxes_px": {key: list(value) for key, value in option_bboxes.items()},
        "option_directions": dict(option_directions),
    }


def _draw_distance_label(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    text: str,
    theme,
    label_font,
) -> List[float]:
    """Draw one dashed distance guide and label."""

    draw_dashed_line(
        draw,
        start=start,
        end=end,
        fill=tuple(int(value) for value in theme.guide_rgb),
        width=3,
        dash_px=10.0,
        gap_px=6.0,
    )
    guide_bbox = arrow_bbox(start, end, padding_px=6.0)
    mid = (float((start[0] + end[0]) / 2.0), float((start[1] + end[1]) / 2.0))
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if abs(dx) < abs(dy):
        label_center = (float(mid[0] + 58.0), float(mid[1]))
    else:
        label_center = (float(mid[0]), float(mid[1] - 28.0))
    label_bbox = _draw_text_tag(
        draw,
        text=str(text),
        center=label_center,
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=2,
    )
    return list(_bbox_union(guide_bbox, label_bbox))


def _render_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    diagram_style: PhysicsDiagramStyle,
    scene_spec: SceneSpec,
    font_family: str,
) -> RenderedElectrostaticCore:
    """Render one symbolic electrostatic scene into pixels and keyed witnesses."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    theme = physics_electrostatics_theme_from_diagram_style(
        diagram_style,
        accent_color_name=str(accent_color_name),
    )
    resolved_font_family = str(font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=False, font_family=resolved_font_family)
    option_font = load_font(int(render_defaults["option_font_size_px"]), bold=True, font_family=resolved_font_family)
    note_font = load_font(int(render_defaults["note_font_size_px"]), bold=True, font_family=resolved_font_family)
    charge_font = load_font(int(render_defaults["charge_font_size_px"]), bold=True, font_family=resolved_font_family)
    board = _board_bbox(render_defaults)
    board_meta = _draw_board(
        draw,
        bbox=board,
        render_defaults=render_defaults,
        scene_variant=str(scene_spec.scene_variant),
        theme=theme,
        diagram_style=diagram_style,
        label_font=label_font,
    )
    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "coordinate_board",
            "entity_type": "electrostatics_coordinate_board",
            "bbox_px": list(board_meta["axis_bbox_px"]),
            "meta": {"coord_extent": int(render_defaults["coord_extent"])},
        }
    ]
    render_map: Dict[str, Any] = {
        "accent_color_name": str(accent_color_name),
        "scene_variant": str(scene_spec.scene_variant),
        "scene_mode": str(scene_spec.scene_mode),
        "technical_diagram_frame_mode": str(diagram_style.frame_mode),
    }
    coord_extent = int(render_defaults["coord_extent"])
    annotation_key_by_entity_id = _annotation_entity_key_map(scene_spec)

    if str(scene_spec.scene_mode) == SCENE_MODE_DIRECTION:
        if scene_spec.direction_scenario is None:
            raise ValueError("direction render requires a direction scenario")
        scenario = scene_spec.direction_scenario
        charge_bboxes: List[List[float]] = []
        for charge in scenario.charges:
            entity, charge_bbox = _draw_charge(
                draw,
                bbox=board,
                coord_extent=coord_extent,
                charge=charge,
                display_label=str(annotation_key_by_entity_id[str(charge.charge_id)]),
                render_defaults=render_defaults,
                theme=theme,
                charge_font=charge_font,
                label_font=label_font,
            )
            scene_entities.append(entity)
            charge_bboxes.append(list(charge_bbox))
        point_center = _coord_to_px(bbox=board, x=int(scenario.point_x), y=int(scenario.point_y), coord_extent=coord_extent)
        point_bbox = _draw_point_marker(
            draw,
            bbox=board,
            coord_extent=coord_extent,
            x=int(scenario.point_x),
            y=int(scenario.point_y),
            label="P",
            render_defaults=render_defaults,
            theme=theme,
            point_font=option_font,
            target=True,
            inner_sign=scenario.test_charge_sign,
        )
        scene_entities.append(
            {
                "entity_id": "query_point",
                "entity_type": "query_point",
                "bbox_px": list(point_bbox),
                "meta": {
                    "x": int(scenario.point_x),
                    "y": int(scenario.point_y),
                    "label": "P",
                    "test_charge_sign": scenario.test_charge_sign,
                    "center_px": [round(float(point_center[0]), 3), round(float(point_center[1]), 3)],
                },
            }
        )
        option_entities, option_map = _draw_direction_options(
            draw,
            render_defaults=render_defaults,
            scenario=scenario,
            theme=theme,
            option_font=option_font,
        )
        scene_entities.extend(option_entities)
        render_map.update(board_meta)
        render_map.update(option_map)
        render_map["query_point_bbox_px"] = list(point_bbox)
        render_map["query_point_test_charge_sign"] = scenario.test_charge_sign
        render_map["charge_bboxes_px"] = [list(bbox) for bbox in charge_bboxes]
    elif str(scene_spec.scene_mode) == SCENE_MODE_ZERO_FIELD:
        if scene_spec.zero_field_scenario is None:
            raise ValueError("zero-field render requires a zero-field scenario")
        scenario = scene_spec.zero_field_scenario
        charge_bboxes = []
        avoid_bboxes: List[List[float]] = []
        charge_label_bboxes: List[List[float]] = []
        for charge in scenario.charges:
            entity, charge_bbox = _draw_charge(
                draw,
                bbox=board,
                coord_extent=coord_extent,
                charge=charge,
                display_label=str(annotation_key_by_entity_id[str(charge.charge_id)]),
                render_defaults=render_defaults,
                theme=theme,
                charge_font=charge_font,
                label_font=label_font,
            )
            scene_entities.append(entity)
            charge_bboxes.append(list(charge_bbox))
            avoid_bboxes.append(list(charge_bbox))
            label_bbox = entity.get("meta", {}).get("charge_label_bbox_px", [])
            if isinstance(label_bbox, Sequence) and len(label_bbox) == 4:
                charge_label_bboxes.append([float(value) for value in label_bbox])
        candidate_bboxes: Dict[str, List[float]] = {}
        candidate_marker_bboxes: Dict[str, List[float]] = {}
        candidate_label_bboxes: Dict[str, List[float]] = {}
        for point in scenario.candidate_points:
            point_center = _coord_to_px(bbox=board, x=int(point.x), y=int(point.y), coord_extent=coord_extent)
            marker_bbox = _point_bbox(
                point_center,
                radius_px=float(render_defaults.get("candidate_point_radius_px", render_defaults["point_radius_px"])),
            )
            point_bbox, point_label_bbox = _draw_point_marker_with_label_bbox(
                draw,
                bbox=board,
                coord_extent=coord_extent,
                x=int(point.x),
                y=int(point.y),
                label=str(point.letter),
                render_defaults=render_defaults,
                theme=theme,
                point_font=option_font,
                target=False,
                compact_label=True,
                avoid_bboxes=avoid_bboxes,
            )
            candidate_bboxes[str(point.letter)] = list(point_bbox)
            candidate_marker_bboxes[str(point.letter)] = list(marker_bbox)
            candidate_label_bboxes[str(point.letter)] = list(point_label_bbox)
            avoid_bboxes.append(list(point_bbox))
            scene_entities.append(
                {
                    "entity_id": f"candidate_{str(point.letter)}",
                    "entity_type": "candidate_zero_field_point",
                    "bbox_px": list(point_bbox),
                    "meta": {
                        "option_letter": str(point.letter),
                        "x": int(point.x),
                        "y": int(point.y),
                        "is_correct": bool(point.is_correct),
                        "center_px": [round(float(point_center[0]), 3), round(float(point_center[1]), 3)],
                        "candidate_marker_bbox_px": list(marker_bbox),
                    },
                }
            )
        render_map.update(board_meta)
        render_map["charge_bboxes_px"] = [list(bbox) for bbox in charge_bboxes]
        render_map["charge_label_bboxes_px"] = [list(bbox) for bbox in charge_label_bboxes]
        render_map["candidate_point_bboxes_px"] = {key: list(value) for key, value in candidate_bboxes.items()}
        render_map["candidate_marker_bboxes_px"] = {key: list(value) for key, value in candidate_marker_bboxes.items()}
        render_map["candidate_label_bboxes_px"] = {key: list(value) for key, value in candidate_label_bboxes.items()}
    else:
        if scene_spec.potential_scenario is None:
            raise ValueError("potential render requires a potential scenario")
        scenario = scene_spec.potential_scenario
        note_bbox = _draw_text_tag(
            draw,
            text="Use k=1 and V=sum(q/r)",
            center=(float(board[2] - 150.0), float(board[1] - 34.0)),
            font=note_font,
            fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
            outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
            text_rgb=tuple(int(value) for value in theme.label_text_rgb),
            stroke_width_px=2,
        )
        scene_entities.append(
            {
                "entity_id": "potential_formula_label",
                "entity_type": "formula_label",
                "bbox_px": list(note_bbox),
                "meta": {"text": "Use k=1 and V=sum(q/r)"},
            }
        )
        point_center = _coord_to_px(bbox=board, x=int(scenario.point_x), y=int(scenario.point_y), coord_extent=coord_extent)
        point_bbox = _draw_point_marker(
            draw,
            bbox=board,
            coord_extent=coord_extent,
            x=int(scenario.point_x),
            y=int(scenario.point_y),
            label="P",
            render_defaults=render_defaults,
            theme=theme,
            point_font=option_font,
            target=True,
        )
        scene_entities.append(
            {
                "entity_id": "query_point",
                "entity_type": "query_point",
                "bbox_px": list(point_bbox),
                "meta": {
                    "x": int(scenario.point_x),
                    "y": int(scenario.point_y),
                    "label": "P",
                    "center_px": [round(float(point_center[0]), 3), round(float(point_center[1]), 3)],
                },
            }
        )
        charge_bboxes = []
        distance_bboxes = []
        for charge in scenario.charges:
            entity, charge_bbox = _draw_charge(
                draw,
                bbox=board,
                coord_extent=coord_extent,
                charge=charge,
                display_label=str(annotation_key_by_entity_id[str(charge.charge_id)]),
                render_defaults=render_defaults,
                theme=theme,
                charge_font=charge_font,
                label_font=label_font,
            )
            scene_entities.append(entity)
            charge_bboxes.append(list(charge_bbox))
            charge_center = _coord_to_px(bbox=board, x=int(charge.x), y=int(charge.y), coord_extent=coord_extent)
            distance_bbox = _draw_distance_label(
                draw,
                start=charge_center,
                end=point_center,
                text=f"r={int(charge.distance_units)}",
                theme=theme,
                label_font=label_font,
            )
            distance_bboxes.append(list(distance_bbox))
            scene_entities.append(
                {
                    "entity_id": f"distance_guide_{str(charge.charge_id)}",
                    "entity_type": "charge_to_point_distance_guide",
                    "bbox_px": list(distance_bbox),
                    "meta": {
                        "charge_id": str(charge.charge_id),
                        "point_id": "query_point",
                        "distance_units": int(charge.distance_units),
                    },
                }
            )
        witness_bbox = _bbox_union(point_bbox, note_bbox, *charge_bboxes, *distance_bboxes)
        scene_entities.append(
            {
                "entity_id": "potential_witness_region",
                "entity_type": "potential_witness_region",
                "bbox_px": list(witness_bbox),
                "meta": {
                    "members": ["query_point"] + [str(charge.charge_id) for charge in scenario.charges],
                    "resolved_value": int(scenario.potential_value),
                },
            }
        )
        render_map.update(board_meta)
        render_map["query_point_bbox_px"] = list(point_bbox)
        render_map["charge_bboxes_px"] = [list(bbox) for bbox in charge_bboxes]
        render_map["distance_guide_bboxes_px"] = [list(bbox) for bbox in distance_bboxes]
        render_map["potential_witness_region_bbox_px"] = list(witness_bbox)

    entity_bbox_map = {
        str(entity["entity_id"]): list(entity["bbox_px"])
        for entity in scene_entities
        if entity.get("bbox_px") is not None
    }
    entity_point_map = {
        str(entity["entity_id"]): list(entity.get("meta", {}).get("center_px", []))
        for entity in scene_entities
        if "center_px" in entity.get("meta", {})
    }
    annotation_bboxes: List[List[float]] = []
    annotation_points: List[List[float]] = []
    annotation_point_map: Dict[str, List[float]] = {}
    for entity_id in scene_spec.annotation_entity_ids:
        entity_key = str(entity_id)
        if entity_key not in entity_bbox_map or entity_key not in entity_point_map:
            raise ValueError(f"missing electrostatics annotation projection for entity {entity_key!r}")
        annotation_key = str(annotation_key_by_entity_id[entity_key])
        annotation_bboxes.append(list(entity_bbox_map[entity_key]))
        annotation_points.append(list(entity_point_map[entity_key]))
        annotation_point_map[annotation_key] = list(entity_point_map[entity_key])
    render_map["annotation_entity_ids"] = list(scene_spec.annotation_entity_ids)
    render_map["annotation_key_by_entity_id"] = dict(annotation_key_by_entity_id)
    render_map["annotation_points_px"] = [list(point) for point in annotation_points]
    render_map["annotation_keyed_points_px"] = {str(key): list(point) for key, point in annotation_point_map.items()}
    return RenderedElectrostaticCore(
        image=image,
        annotation_bboxes=[list(bbox) for bbox in annotation_bboxes],
        annotation_points=[list(point) for point in annotation_points],
        annotation_point_map={str(key): list(point) for key, point in annotation_point_map.items()},
        annotation_entity_ids=list(scene_spec.annotation_entity_ids),
        annotation_key_by_entity_id={str(key): str(value) for key, value in annotation_key_by_entity_id.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )



def resolve_render_defaults(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> Dict[str, Any]:
    """Resolve integer rendering knobs from scene config and params."""

    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(_DEFAULTS, key)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in RENDER_INT_KEYS
    }


def render_electrostatic_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_spec: SceneSpec,
    rendering_defaults: Mapping[str, Any],
    accent_color_name: str,
    namespace: str,
) -> RenderedElectrostaticScene:
    """Render one electrostatic-field scene after final layout placement."""

    render_defaults = resolve_render_defaults(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
        namespace=str(namespace),
    )
    style_params = dict(rendering_defaults)
    style_params.update(dict(params))
    diagram_style, diagram_style_meta = resolve_physics_diagram_style(
        instance_seed=int(instance_seed),
        params=style_params,
        scene_id=SCENE_ID,
        protected_colors=PHYSICS_ELECTROSTATICS_SEMANTIC_COLORS,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    styled = dict(render_defaults)
    styled["grid_line_width_px"] = max(1, int(diagram_style.grid_minor_width_px))
    styled["dense_grid_line_width_px"] = max(
        int(styled["grid_line_width_px"]),
        int(diagram_style.grid_major_width_px),
    )
    styled["axis_width_px"] = max(3, int(diagram_style.axis_stroke_width_px))
    styled, placement_meta = _resolve_electrostatics_layout_placement(
        render_defaults=styled,
        jitter_defaults=rendering_defaults,
        params=params,
        instance_seed=int(instance_seed),
        scene_mode=str(scene_spec.scene_mode),
        namespace=str(namespace),
    )
    background, background_meta = make_physics_diagram_background(
        canvas_width=int(styled["canvas_width"]),
        canvas_height=int(styled["canvas_height"]),
        style=diagram_style,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    core = _render_scene(
        background=background,
        render_defaults=styled,
        accent_color_name=str(accent_color_name),
        diagram_style=diagram_style,
        scene_spec=scene_spec,
        font_family=str(font_family),
    )
    image, post_noise_meta = apply_post_image_noise(
        core.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map = dict(core.render_map)
    render_map["technical_diagram_style"] = dict(diagram_style_meta)
    render_map["background_style"] = dict(background_meta)
    render_map["layout_placement"] = dict(placement_meta)
    render_map["post_image_noise"] = dict(post_noise_meta)
    render_map["font_family"] = str(font_family)
    return RenderedElectrostaticScene(
        image=image,
        annotation_bboxes=[list(value) for value in core.annotation_bboxes],
        annotation_points=[list(value) for value in core.annotation_points],
        annotation_point_map={str(key): list(value) for key, value in core.annotation_point_map.items()},
        annotation_entity_ids=list(core.annotation_entity_ids),
        annotation_key_by_entity_id=dict(core.annotation_key_by_entity_id),
        scene_entities=[dict(entity) for entity in core.scene_entities],
        render_map=render_map,
        font_family=str(font_family),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        layout_placement_meta=dict(placement_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def render_with_attempts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    rendering_defaults: Mapping[str, Any],
    accent_color_name: str,
    namespace: str,
    make_scene_spec: Callable[[Any], SceneSpec],
) -> tuple[SceneSpec, RenderedElectrostaticScene]:
    """Retry symbolic construction until one final rendered scene is valid."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(int(instance_seed), f"{namespace}.attempt.{attempt_index}")
        try:
            scene_spec = make_scene_spec(attempt_rng)
            rendered = render_electrostatic_scene(
                instance_seed=int(instance_seed),
                params=params,
                scene_spec=scene_spec,
                rendering_defaults=rendering_defaults,
                accent_color_name=str(accent_color_name),
                namespace=str(namespace),
            )
            return scene_spec, rendered
        except ValueError as exc:
            last_error = exc
    raise RuntimeError("failed to render a valid electrostatic-field scene") from last_error
