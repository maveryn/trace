"""Rendering helpers for cyclic-order puzzle scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.puzzles.shared.symbol_rendering import draw_puzzle_shape_icon
from trace_tasks.tasks.shared.bbox_projection import round_bbox as _round_bbox
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import CyclicOrderRenderParams, SCENE_VARIANTS


@dataclass(frozen=True)
class RenderedCyclicOrderScene:
    """Rendered cyclic-order scene plus traced geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    reference_loop_bbox_px: List[float]
    option_choice_bbox_map: Dict[str, List[float]]


def _scene_panel_style(
    scene_variant: str,
    *,
    render_params: CyclicOrderRenderParams,
) -> Tuple[Tuple[int, int, int] | None, Tuple[int, int, int] | None]:
    """Resolve outer reference-panel styling for one scene variant."""

    if str(scene_variant) == "token_ring_outline":
        return None, render_params.border_color_rgb
    if str(scene_variant) in {"necklace_board", "route_loop_diagram"}:
        return render_params.instruction_fill_rgb, None
    return render_params.panel_fill_rgb, render_params.border_color_rgb


def _loop_bbox_within_image(
    image_bbox: Sequence[float],
    *,
    loop_shape_variant: str,
) -> Tuple[float, float, float, float]:
    """Return one loop bbox centered inside an image slot."""

    left, top, right, bottom = [float(value) for value in image_bbox]
    width = float(right - left)
    height = float(bottom - top)
    if str(loop_shape_variant) == "wide":
        width_ratio, height_ratio = 0.76, 0.56
    elif str(loop_shape_variant) == "tall":
        width_ratio, height_ratio = 0.58, 0.78
    else:
        width_ratio, height_ratio = 0.68, 0.68
    loop_width = float(width * width_ratio)
    loop_height = float(height * height_ratio)
    x1 = float(left + 0.5 * (width - loop_width))
    y1 = float(top + 0.5 * (height - loop_height))
    return (
        float(x1),
        float(y1),
        float(x1 + loop_width),
        float(y1 + loop_height),
    )


def _token_center_points(
    loop_bbox: Sequence[float],
    *,
    token_count: int,
    start_angle_deg: int,
    loop_path_style: str,
) -> List[Tuple[float, float]]:
    """Return token centers in clockwise order around the loop."""

    path_style = str(loop_path_style)
    left, top, right, bottom = [float(value) for value in loop_bbox]
    cx = float(0.5 * (left + right))
    cy = float(0.5 * (top + bottom))
    rx = float(0.5 * (right - left))
    ry = float(0.5 * (bottom - top))
    points: List[Tuple[float, float]] = []
    for index in range(int(token_count)):
        angle = math.radians(float(start_angle_deg) - (360.0 * float(index) / float(token_count)))
        if path_style == "rounded_rect":
            cos_v = math.cos(angle)
            sin_v = math.sin(angle)
            power = 0.5
            x = cx + rx * (1.0 if cos_v >= 0.0 else -1.0) * (abs(cos_v) ** power)
            y = cy + ry * (1.0 if sin_v >= 0.0 else -1.0) * (abs(sin_v) ** power)
        elif path_style == "polygon_loop":
            sides = 8
            vertex_angle = (2.0 * math.pi / float(sides)) * round(float(sides) * angle / (2.0 * math.pi))
            blend = 0.72
            x = cx + rx * ((blend * math.cos(angle)) + ((1.0 - blend) * math.cos(vertex_angle)))
            y = cy + ry * ((blend * math.sin(angle)) + ((1.0 - blend) * math.sin(vertex_angle)))
        elif path_style == "wavy_loop":
            wave = 1.0 + (0.08 * math.sin((3.0 * angle) + 0.6))
            x = cx + rx * wave * math.cos(angle)
            y = cy + ry * wave * math.sin(angle)
        else:
            x = cx + rx * math.cos(angle)
            y = cy + ry * math.sin(angle)
        points.append((float(x), float(y)))
    return points


def _sample_loop_path_points(
    loop_bbox: Sequence[float],
    *,
    loop_path_style: str,
    point_count: int = 96,
) -> List[Tuple[float, float]]:
    """Return closed loop path points for non-ellipse styles."""

    left, top, right, bottom = [float(value) for value in loop_bbox]
    cx = float(0.5 * (left + right))
    cy = float(0.5 * (top + bottom))
    rx = float(0.5 * (right - left))
    ry = float(0.5 * (bottom - top))
    style = str(loop_path_style)
    points: List[Tuple[float, float]] = []
    count = max(8, int(point_count))
    if style == "polygon_loop":
        count = 8
    for index in range(count):
        angle = (2.0 * math.pi * float(index)) / float(count)
        if style == "rounded_rect":
            cos_v = math.cos(angle)
            sin_v = math.sin(angle)
            power = 0.5
            x = cx + rx * (1.0 if cos_v >= 0.0 else -1.0) * (abs(cos_v) ** power)
            y = cy + ry * (1.0 if sin_v >= 0.0 else -1.0) * (abs(sin_v) ** power)
        elif style == "wavy_loop":
            wave = 1.0 + (0.08 * math.sin((3.0 * angle) + 0.6))
            x = cx + rx * wave * math.cos(angle)
            y = cy + ry * wave * math.sin(angle)
        else:
            x = cx + rx * math.cos(angle)
            y = cy + ry * math.sin(angle)
        points.append((float(x), float(y)))
    return points


def _draw_loop_path(
    draw: ImageDraw.ImageDraw,
    *,
    loop_bbox: Sequence[float],
    loop_path_style: str,
    render_params: CyclicOrderRenderParams,
) -> None:
    """Draw the closed path that carries the ordered tokens."""

    style = str(loop_path_style)
    line_width = max(2, int(render_params.loop_stroke_width_px))
    if style in {"ellipse", "beaded_string"}:
        draw.ellipse(
            tuple(float(value) for value in loop_bbox),
            outline=render_params.loop_color_rgb,
            width=line_width,
        )
        if style == "beaded_string":
            inner_bbox = (
                float(loop_bbox[0]) + 4.0,
                float(loop_bbox[1]) + 4.0,
                float(loop_bbox[2]) - 4.0,
                float(loop_bbox[3]) - 4.0,
            )
            draw.ellipse(
                inner_bbox,
                outline=render_params.border_color_rgb,
                width=max(1, line_width // 2),
            )
        return

    points = _sample_loop_path_points(loop_bbox, loop_path_style=style)
    if len(points) >= 3:
        draw.line(points + [points[0]], fill=render_params.loop_color_rgb, width=line_width, joint="curve")


def _draw_token(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    token_size_px: float,
    token_spec: Mapping[str, Any],
    render_params: CyclicOrderRenderParams,
) -> List[float]:
    """Draw one loop token and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    half = float(0.5 * token_size_px)
    token_bbox = (
        float(cx - half),
        float(cy - half),
        float(cx + half),
        float(cy + half),
    )
    render_mode = str(token_spec["render_mode"])
    fill_rgb = tuple(int(value) for value in token_spec.get("fill_rgb") or render_params.shape_fill_rgb)
    outline_rgb = tuple(int(value) for value in render_params.border_color_rgb)
    if render_mode == "color":
        draw.ellipse(
            token_bbox,
            fill=fill_rgb,
            outline=outline_rgb,
            width=max(2, int(render_params.border_width_px)),
        )
    elif render_mode == "symbol_badge":
        draw.ellipse(
            token_bbox,
            fill=tuple(render_params.panel_fill_rgb),
            outline=outline_rgb,
            width=max(1, int(render_params.border_width_px)),
        )
        draw_puzzle_shape_icon(
            draw,
            bbox=token_bbox,
            object_type=str(token_spec["object_type"]),
            fill_rgb=fill_rgb,
            outline_rgb=outline_rgb,
            width=max(1, int(render_params.border_width_px) - 1),
            inset_px=float(max(4.0, render_params.shape_bead_inset_px + 5.0)),
        )
    else:
        icon_fill = tuple(render_params.panel_fill_rgb) if render_mode == "outline_shape" else fill_rgb
        draw_puzzle_shape_icon(
            draw,
            bbox=token_bbox,
            object_type=str(token_spec["object_type"]),
            fill_rgb=icon_fill,
            outline_rgb=outline_rgb,
            width=max(2, int(render_params.border_width_px)),
            inset_px=float(max(0.0, render_params.shape_bead_inset_px)),
        )
    return _round_bbox(token_bbox)


def _draw_loop_image(
    draw: ImageDraw.ImageDraw,
    *,
    image_bbox: Sequence[float],
    loop_id: str,
    loop_shape_variant: str,
    loop_path_style: str,
    start_angle_deg: int,
    token_specs: Sequence[Mapping[str, Any]],
    render_params: CyclicOrderRenderParams,
    loop_entity_type: str,
    token_entity_type: str,
    position_labels: Sequence[str] | None = None,
    gap_labels: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Draw one loop and return traced loop/token entities."""

    loop_bbox = _loop_bbox_within_image(image_bbox, loop_shape_variant=str(loop_shape_variant))
    _draw_loop_path(
        draw,
        loop_bbox=loop_bbox,
        loop_path_style=str(loop_path_style),
        render_params=render_params,
    )
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": str(loop_id),
            "entity_type": str(loop_entity_type),
            "bbox_px": _round_bbox(loop_bbox),
            "attrs": {
                "loop_shape_variant": str(loop_shape_variant),
                "loop_path_style": str(loop_path_style),
                "start_angle_deg": int(start_angle_deg),
                "token_count": int(len(token_specs)),
            },
        }
    ]
    centers = _token_center_points(
        loop_bbox,
        token_count=int(len(token_specs)),
        start_angle_deg=int(start_angle_deg),
        loop_path_style=str(loop_path_style),
    )
    label_font = load_font(max(18, int(render_params.option_label_font_size_px * 0.58)), bold=True)
    gap_font = load_font(max(26, int(render_params.option_label_font_size_px * 0.82)), bold=True)
    loop_center = (
        float(0.5 * (loop_bbox[0] + loop_bbox[2])),
        float(0.5 * (loop_bbox[1] + loop_bbox[3])),
    )
    for token_index, (center, token_spec) in enumerate(zip(centers, token_specs), start=1):
        token_bbox = _draw_token(
            draw,
            center=center,
            token_size_px=float(render_params.bead_size_px),
            token_spec=token_spec,
            render_params=render_params,
        )
        entities.append(
            {
                "entity_id": f"{str(loop_id)}_token_{int(token_index)}",
                "entity_type": str(token_entity_type),
                "bbox_px": list(token_bbox),
                "attrs": {
                    "token_label": str(token_spec["token_label"]),
                    "object_type": str(token_spec["object_type"]),
                    "render_mode": str(token_spec["render_mode"]),
                },
            }
        )
        if position_labels is not None:
            direction_x = float(center[0] - loop_center[0])
            direction_y = float(center[1] - loop_center[1])
            distance = max(1.0, math.hypot(direction_x, direction_y))
            badge_center = (
                float(center[0] + (direction_x / distance) * 1.02 * render_params.bead_size_px),
                float(center[1] + (direction_y / distance) * 1.02 * render_params.bead_size_px),
            )
            badge_radius = max(16.0, float(render_params.bead_size_px) * 0.36)
            badge_bbox = (
                float(badge_center[0] - badge_radius),
                float(badge_center[1] - badge_radius),
                float(badge_center[0] + badge_radius),
                float(badge_center[1] + badge_radius),
            )
            draw.ellipse(
                badge_bbox,
                fill=tuple(render_params.panel_fill_rgb),
                outline=tuple(render_params.border_color_rgb),
                width=2,
            )
            draw_centered_text(
                draw,
                text=str(position_labels[int(token_index) - 1]),
                center=badge_center,
                font=label_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=1,
            )
    if gap_labels is not None:
        for gap_index, label in enumerate(gap_labels, start=1):
            current_center = centers[int(gap_index) - 1]
            next_center = centers[int(gap_index) % len(centers)]
            midpoint = (
                float(0.5 * (current_center[0] + next_center[0])),
                float(0.5 * (current_center[1] + next_center[1])),
            )
            direction_x = float(midpoint[0] - loop_center[0])
            direction_y = float(midpoint[1] - loop_center[1])
            distance = max(1.0, math.hypot(direction_x, direction_y))
            badge_center = (
                float(midpoint[0] + (direction_x / distance) * 0.50 * render_params.bead_size_px),
                float(midpoint[1] + (direction_y / distance) * 0.50 * render_params.bead_size_px),
            )
            badge_radius = max(19.0, float(render_params.bead_size_px) * 0.44)
            badge_bbox = (
                float(badge_center[0] - badge_radius),
                float(badge_center[1] - badge_radius),
                float(badge_center[0] + badge_radius),
                float(badge_center[1] + badge_radius),
            )
            draw.ellipse(
                badge_bbox,
                fill=tuple(render_params.instruction_fill_rgb),
                outline=tuple(render_params.border_color_rgb),
                width=2,
            )
            label_bbox = draw_centered_text(
                draw,
                text=str(label),
                center=badge_center,
                font=gap_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=1,
            )
            entities.append(
                {
                    "entity_id": f"{str(loop_id)}_gap_{int(gap_index)}",
                    "entity_type": "puzzle_cyclic_order_gap_label",
                    "bbox_px": _round_bbox(badge_bbox),
                    "attrs": {
                        "gap_label": str(label),
                        "label_bbox_px": list(label_bbox),
                    },
                }
            )
    return entities


def _draw_titled_loop_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    title: str,
    loop_id: str,
    loop_shape_variant: str,
    loop_path_style: str,
    start_angle_deg: int,
    token_specs: Sequence[Mapping[str, Any]],
    render_params: CyclicOrderRenderParams,
    position_labels: Sequence[str] | None = None,
    gap_labels: Sequence[str] | None = None,
) -> tuple[List[Dict[str, Any]], List[float]]:
    """Draw one titled loop panel for repair-style cyclic puzzles."""

    panel = tuple(float(value) for value in panel_bbox)
    draw_rounded_rect(
        draw,
        panel,
        radius=int(render_params.panel_corner_radius_px),
        fill=render_params.panel_fill_rgb,
        outline=render_params.border_color_rgb,
        width=max(1, int(render_params.border_width_px)),
    )
    title_font = load_font(int(render_params.reference_label_font_size_px), bold=True)
    title_bbox = draw_centered_text(
        draw,
        text=str(title),
        center=(float(0.5 * (panel[0] + panel[2])), float(panel[1] + 31.0)),
        font=title_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=1,
    )
    loop_image_bbox = (
        float(panel[0] + 28.0),
        float(panel[1] + 60.0),
        float(panel[2] - 28.0),
        float(panel[3] - 24.0),
    )
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": f"{loop_id}_panel",
            "entity_type": "puzzle_cyclic_order_loop_panel",
            "bbox_px": _round_bbox(panel),
            "attrs": {"title": str(title)},
        },
        {
            "entity_id": f"{loop_id}_title",
            "entity_type": "puzzle_cyclic_order_loop_panel_title",
            "bbox_px": list(title_bbox),
            "attrs": {"text": str(title)},
        },
    ]
    loop_entities = _draw_loop_image(
        draw,
        image_bbox=loop_image_bbox,
        loop_id=str(loop_id),
        loop_shape_variant=str(loop_shape_variant),
        loop_path_style=str(loop_path_style),
        start_angle_deg=int(start_angle_deg),
        token_specs=token_specs,
        render_params=render_params,
        loop_entity_type="puzzle_cyclic_order_repair_loop",
        token_entity_type="puzzle_cyclic_order_repair_token",
        position_labels=position_labels,
        gap_labels=gap_labels,
    )
    entities.extend(loop_entities)
    return entities, list(loop_entities[0]["bbox_px"])


def render_cyclic_order_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    reference_token_specs: Sequence[Mapping[str, Any]],
    reference_loop_shape_variant: str,
    reference_loop_path_style: str,
    reference_start_angle_deg: int,
    option_specs: Sequence[Mapping[str, Any]],
    render_params: CyclicOrderRenderParams,
) -> RenderedCyclicOrderScene:
    """Render one cyclic-order reference loop with option loops."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(SCENE_VARIANTS):
        raise ValueError(f"unsupported cyclic-order scene_variant: {scene_variant}")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    reference_font = load_font(int(render_params.reference_label_font_size_px), bold=True)
    option_font = load_font(int(render_params.option_label_font_size_px), bold=True)
    label_bbox_sample = draw.textbbox((0.0, 0.0), "A", font=option_font, stroke_width=1)
    option_label_height = float(label_bbox_sample[3] - label_bbox_sample[1])

    scene_left = float(render_params.scene_margin_left_px)
    scene_top = float(render_params.scene_margin_top_px)
    scene_right = float(render_params.canvas_width - render_params.scene_margin_right_px)
    scene_bottom = float(render_params.canvas_height - render_params.scene_margin_bottom_px)

    reference_panel_bbox = (
        float(scene_left),
        float(scene_top),
        float(scene_right),
        float(scene_top + render_params.reference_panel_height_px),
    )
    reference_fill, reference_outline = _scene_panel_style(selected_variant, render_params=render_params)
    if reference_fill is not None or reference_outline is not None:
        draw_rounded_rect(
            draw,
            reference_panel_bbox,
            radius=int(render_params.panel_corner_radius_px),
            fill=reference_fill if reference_fill is not None else (255, 255, 255),
            outline=reference_outline if reference_outline is not None else (255, 255, 255),
            width=max(1, int(render_params.border_width_px if reference_outline is not None else 1)),
        )

    reference_label_bbox = draw_centered_text(
        draw,
        text="Reference",
        center=(float(0.5 * (scene_left + scene_right)), float(scene_top + 30.0)),
        font=reference_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=1,
    )
    reference_image_bbox = (
        float(scene_left + 0.5 * ((scene_right - scene_left) - render_params.reference_loop_width_px)),
        float(scene_top + render_params.reference_panel_padding_px + 30.0),
        float(
            scene_left
            + 0.5 * ((scene_right - scene_left) - render_params.reference_loop_width_px)
            + render_params.reference_loop_width_px
        ),
        float(scene_top + render_params.reference_panel_padding_px + 30.0 + render_params.reference_loop_height_px),
    )

    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "reference_panel",
            "entity_type": "puzzle_cyclic_order_reference_panel",
            "bbox_px": _round_bbox(reference_panel_bbox),
            "attrs": {"scene_variant": selected_variant},
        },
        {
            "entity_id": "reference_label",
            "entity_type": "puzzle_cyclic_order_reference_label",
            "bbox_px": list(reference_label_bbox),
            "attrs": {"text": "Reference"},
        },
    ]
    reference_loop_entities = _draw_loop_image(
        draw,
        image_bbox=reference_image_bbox,
        loop_id="reference_loop",
        loop_shape_variant=str(reference_loop_shape_variant),
        loop_path_style=str(reference_loop_path_style),
        start_angle_deg=int(reference_start_angle_deg),
        token_specs=reference_token_specs,
        render_params=render_params,
        loop_entity_type="puzzle_cyclic_order_reference_loop",
        token_entity_type="puzzle_cyclic_order_reference_token",
    )
    entities.extend(reference_loop_entities)
    reference_loop_bbox = list(reference_loop_entities[0]["bbox_px"])

    option_count = int(len(option_specs))
    option_columns = 2 if option_count == 4 else (3 if option_count <= 6 else 4)
    option_rows = int(math.ceil(float(option_count) / float(option_columns)))
    option_block_height = float(render_params.option_image_height_px + render_params.option_label_gap_px + option_label_height)
    options_total_width = float(
        (option_columns * render_params.option_image_width_px)
        + (max(0, option_columns - 1) * render_params.option_gap_px)
    )
    options_total_height = float(
        (option_rows * option_block_height)
        + (max(0, option_rows - 1) * render_params.option_row_gap_px)
    )
    options_left = float(scene_left + 0.5 * ((scene_right - scene_left) - options_total_width))
    options_top = float(reference_panel_bbox[3] + render_params.reference_to_options_gap_px)
    option_choice_bbox_map: Dict[str, List[float]] = {}

    for option_index, option_spec in enumerate(option_specs):
        row_index = int(option_index // option_columns)
        col_index = int(option_index % option_columns)
        image_left = float(options_left + col_index * (render_params.option_image_width_px + render_params.option_gap_px))
        image_top = float(options_top + row_index * (option_block_height + render_params.option_row_gap_px))
        image_bbox = (
            float(image_left),
            float(image_top),
            float(image_left + render_params.option_image_width_px),
            float(image_top + render_params.option_image_height_px),
        )
        option_choice_id = str(option_spec["option_choice_id"])
        option_choice_bbox_map[option_choice_id] = _round_bbox(image_bbox)
        entities.append(
            {
                "entity_id": str(option_choice_id),
                "entity_type": "puzzle_cyclic_order_option_choice",
                "bbox_px": _round_bbox(image_bbox),
                "attrs": {
                    "option_label": str(option_spec["option_label"]),
                    "is_valid": bool(option_spec["is_valid"]),
                },
            }
        )
        entities.extend(
            _draw_loop_image(
                draw,
                image_bbox=image_bbox,
                loop_id=f"{option_choice_id}_loop",
                loop_shape_variant=str(option_spec["loop_shape_variant"]),
                loop_path_style=str(option_spec.get("loop_path_style", reference_loop_path_style)),
                start_angle_deg=int(option_spec["start_angle_deg"]),
                token_specs=list(option_spec["bead_specs"]),
                render_params=render_params,
                loop_entity_type="puzzle_cyclic_order_option_loop",
                token_entity_type="puzzle_cyclic_order_option_token",
            )
        )
        label_center = (
            float(0.5 * (image_bbox[0] + image_bbox[2])),
            float(image_bbox[3] + render_params.option_label_gap_px + 0.5 * option_label_height),
        )
        label_bbox = draw_centered_text(
            draw,
            text=str(option_spec["option_label"]),
            center=label_center,
            font=option_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        entities.append(
            {
                "entity_id": f"{option_choice_id}_label",
                "entity_type": "puzzle_cyclic_order_option_label",
                "bbox_px": list(label_bbox),
                "attrs": {"option_label": str(option_spec["option_label"])},
            }
        )

    scene_bbox = [
        round(float(scene_left), 3),
        round(float(scene_top), 3),
        round(float(scene_right), 3),
        round(float(max(scene_bottom, options_top + options_total_height)), 3),
    ]

    return RenderedCyclicOrderScene(
        image=image,
        entities=entities,
        scene_bbox_px=list(scene_bbox),
        reference_loop_bbox_px=list(reference_loop_bbox),
        option_choice_bbox_map=option_choice_bbox_map,
    )


def render_swap_repair_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    reference_token_specs: Sequence[Mapping[str, Any]],
    reference_loop_shape_variant: str,
    reference_loop_path_style: str,
    reference_start_angle_deg: int,
    broken_token_specs: Sequence[Mapping[str, Any]],
    broken_loop_shape_variant: str,
    broken_loop_path_style: str,
    broken_start_angle_deg: int,
    option_specs: Sequence[Mapping[str, Any]],
    render_params: CyclicOrderRenderParams,
) -> RenderedCyclicOrderScene:
    """Render a reference loop, a numbered broken loop, and swap-option cards."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(SCENE_VARIANTS):
        raise ValueError(f"unsupported cyclic-order scene_variant: {scene_variant}")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    scene_left = float(render_params.scene_margin_left_px)
    scene_top = float(render_params.scene_margin_top_px)
    scene_right = float(render_params.canvas_width - render_params.scene_margin_right_px)
    scene_bottom = float(render_params.canvas_height - render_params.scene_margin_bottom_px)
    panel_gap = 42.0
    panel_width = float((scene_right - scene_left - panel_gap) / 2.0)
    top_panel_height = 305.0
    reference_panel = (
        scene_left,
        scene_top,
        scene_left + panel_width,
        scene_top + top_panel_height,
    )
    broken_panel = (
        scene_left + panel_width + panel_gap,
        scene_top,
        scene_right,
        scene_top + top_panel_height,
    )

    entities: List[Dict[str, Any]] = []
    reference_entities, reference_loop_bbox = _draw_titled_loop_panel(
        draw,
        panel_bbox=reference_panel,
        title="Reference",
        loop_id="reference_loop",
        loop_shape_variant=str(reference_loop_shape_variant),
        loop_path_style=str(reference_loop_path_style),
        start_angle_deg=int(reference_start_angle_deg),
        token_specs=reference_token_specs,
        render_params=render_params,
    )
    entities.extend(reference_entities)
    broken_entities, _broken_loop_bbox = _draw_titled_loop_panel(
        draw,
        panel_bbox=broken_panel,
        title="Broken loop",
        loop_id="broken_loop",
        loop_shape_variant=str(broken_loop_shape_variant),
        loop_path_style=str(broken_loop_path_style),
        start_angle_deg=int(broken_start_angle_deg),
        token_specs=broken_token_specs,
        render_params=render_params,
        position_labels=[str(index) for index in range(1, len(broken_token_specs) + 1)],
    )
    entities.extend(broken_entities)

    option_count = int(len(option_specs))
    option_columns = 2 if option_count == 4 else (3 if option_count <= 6 else 4)
    option_rows = int(math.ceil(float(option_count) / float(option_columns)))
    option_gap = float(render_params.option_gap_px)
    option_row_gap = 28.0
    options_top = float(reference_panel[3] + 64.0)
    card_width = 260.0
    card_height = 104.0
    options_total_width = float(
        (option_columns * card_width)
        + (max(0, option_columns - 1) * option_gap)
    )
    options_left = float(scene_left + 0.5 * ((scene_right - scene_left) - options_total_width))
    option_choice_bbox_map: Dict[str, List[float]] = {}
    option_font = load_font(int(render_params.option_label_font_size_px), bold=True)
    swap_font = load_font(30, bold=True)

    for option_index, option_spec in enumerate(option_specs):
        row_index = int(option_index // option_columns)
        col_index = int(option_index % option_columns)
        card_left = float(options_left + col_index * (card_width + option_gap))
        card_top = float(options_top + row_index * (card_height + option_row_gap))
        card_bbox = (
            card_left,
            card_top,
            card_left + card_width,
            card_top + card_height,
        )
        option_choice_id = str(option_spec["option_choice_id"])
        option_choice_bbox_map[option_choice_id] = _round_bbox(card_bbox)
        draw_rounded_rect(
            draw,
            card_bbox,
            radius=18,
            fill=render_params.instruction_fill_rgb,
            outline=render_params.border_color_rgb,
            width=max(1, int(render_params.border_width_px)),
        )
        draw_centered_text(
            draw,
            text=str(option_spec["option_label"]),
            center=(float(card_left + 28.0), float(card_top + 28.0)),
            font=option_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        swap_text = f'{int(option_spec["first_position"])} <-> {int(option_spec["second_position"])}'
        draw_centered_text(
            draw,
            text=swap_text,
            center=(float(0.5 * (card_bbox[0] + card_bbox[2])), float(card_top + 57.0)),
            font=swap_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        entities.append(
            {
                "entity_id": str(option_choice_id),
                "entity_type": "puzzle_cyclic_order_swap_option",
                "bbox_px": _round_bbox(card_bbox),
                "attrs": {
                    "option_label": str(option_spec["option_label"]),
                    "first_position": int(option_spec["first_position"]),
                    "second_position": int(option_spec["second_position"]),
                    "is_valid": bool(option_spec["is_valid"]),
                },
            }
        )

    scene_bbox = [
        round(float(scene_left), 3),
        round(float(scene_top), 3),
        round(float(scene_right), 3),
        round(
            float(
                max(
                    scene_bottom,
                    options_top + (option_rows * card_height) + ((option_rows - 1) * option_row_gap),
                )
            ),
            3,
        ),
    ]
    return RenderedCyclicOrderScene(
        image=image,
        entities=entities,
        scene_bbox_px=list(scene_bbox),
        reference_loop_bbox_px=list(reference_loop_bbox),
        option_choice_bbox_map=option_choice_bbox_map,
    )


def render_insertion_position_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    reference_token_specs: Sequence[Mapping[str, Any]],
    reference_loop_shape_variant: str,
    reference_loop_path_style: str,
    reference_start_angle_deg: int,
    partial_token_specs: Sequence[Mapping[str, Any]],
    partial_loop_shape_variant: str,
    partial_loop_path_style: str,
    partial_start_angle_deg: int,
    partial_gap_labels: Sequence[str],
    option_specs: Sequence[Mapping[str, Any]],
    render_params: CyclicOrderRenderParams,
) -> RenderedCyclicOrderScene:
    """Render a reference loop and partial loop with labeled insertion gaps."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(SCENE_VARIANTS):
        raise ValueError(f"unsupported cyclic-order scene_variant: {scene_variant}")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    scene_left = float(render_params.scene_margin_left_px)
    scene_top = float(render_params.scene_margin_top_px)
    scene_right = float(render_params.canvas_width - render_params.scene_margin_right_px)
    scene_bottom = float(render_params.canvas_height - render_params.scene_margin_bottom_px)
    panel_gap = 42.0
    loop_panel_width = float((scene_right - scene_left - panel_gap) / 2.0)
    top_panel_height = 335.0
    reference_panel = (
        scene_left,
        scene_top,
        scene_left + loop_panel_width,
        scene_top + top_panel_height,
    )
    partial_panel = (
        reference_panel[2] + panel_gap,
        scene_top,
        reference_panel[2] + panel_gap + loop_panel_width,
        scene_top + top_panel_height,
    )

    entities: List[Dict[str, Any]] = []
    reference_entities, reference_loop_bbox = _draw_titled_loop_panel(
        draw,
        panel_bbox=reference_panel,
        title="Reference",
        loop_id="reference_loop",
        loop_shape_variant=str(reference_loop_shape_variant),
        loop_path_style=str(reference_loop_path_style),
        start_angle_deg=int(reference_start_angle_deg),
        token_specs=reference_token_specs,
        render_params=render_params,
    )
    entities.extend(reference_entities)
    partial_entities, _partial_loop_bbox = _draw_titled_loop_panel(
        draw,
        panel_bbox=partial_panel,
        title="Candidate gaps",
        loop_id="partial_loop",
        loop_shape_variant=str(partial_loop_shape_variant),
        loop_path_style=str(partial_loop_path_style),
        start_angle_deg=int(partial_start_angle_deg),
        token_specs=partial_token_specs,
        render_params=render_params,
        gap_labels=[str(value) for value in partial_gap_labels],
    )
    entities.extend(partial_entities)

    gap_bbox_by_label = {
        str(entity.get("attrs", {}).get("gap_label")): list(entity["bbox_px"])
        for entity in partial_entities
        if str(entity.get("entity_type")) == "puzzle_cyclic_order_gap_label"
    }
    option_choice_bbox_map: Dict[str, List[float]] = {}
    for option_spec in option_specs:
        option_choice_id = str(option_spec["option_choice_id"])
        option_label = str(option_spec["option_label"])
        if option_label not in gap_bbox_by_label:
            raise ValueError(f"missing insertion gap bbox for option {option_label}")
        option_choice_bbox_map[option_choice_id] = _round_bbox(gap_bbox_by_label[option_label])

    scene_bbox = [
        round(float(scene_left), 3),
        round(float(scene_top), 3),
        round(float(scene_right), 3),
        round(float(max(scene_bottom, reference_panel[3])), 3),
    ]
    return RenderedCyclicOrderScene(
        image=image,
        entities=entities,
        scene_bbox_px=list(scene_bbox),
        reference_loop_bbox_px=list(reference_loop_bbox),
        option_choice_bbox_map=option_choice_bbox_map,
    )


__all__ = [
    "RenderedCyclicOrderScene",
    "render_cyclic_order_scene",
    "render_insertion_position_scene",
    "render_swap_repair_scene",
]
