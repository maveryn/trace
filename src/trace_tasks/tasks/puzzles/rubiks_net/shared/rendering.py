"""Rendering helpers for Rubik cube-net puzzle scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.bbox_projection import round_bbox
from trace_tasks.tasks.shared.text_rendering import load_font

from .rules import sticker_id
from .state import (
    FACE_LAYOUT,
    FACE_ORDER,
    RGB,
    RenderedRubiksScene,
    RubiksRenderParams,
    SUPPORTED_SCENE_VARIANTS,
    StickerKey,
)


def _state_color(
    state: Mapping[StickerKey, str],
    color_map: Mapping[str, Mapping[str, Any]],
    *,
    face: str,
    row: int,
    col: int,
) -> RGB:
    color_name = str(state[(str(face), int(row), int(col))])
    rgb = color_map[str(color_name)]["color_rgb"]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    fill: Sequence[int],
    outline: Sequence[int],
    radius: int,
    width: int,
) -> None:
    draw_rounded_rect(
        draw,
        (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        radius=int(radius),
        fill=tuple(int(value) for value in fill),
        outline=tuple(int(value) for value in outline),
        width=int(width),
    )


def _draw_cube_net(
    draw: ImageDraw.ImageDraw,
    *,
    state: Mapping[StickerKey, str],
    color_map: Mapping[str, Mapping[str, Any]],
    origin: tuple[float, float],
    cell_size_px: float,
    sticker_gap_px: float,
    outline_rgb: Sequence[int],
    face_label_font,
    text_rgb: Sequence[int],
    text_stroke_rgb: Sequence[int],
    include_face_labels: bool,
) -> tuple[dict[str, list[float]], list[float]]:
    """Draw a 2D cube net and return sticker boxes plus the net box."""

    sticker_bbox_map: dict[str, list[float]] = {}
    ox, oy = float(origin[0]), float(origin[1])
    cell = float(cell_size_px)
    gap = max(0.0, float(sticker_gap_px))
    for face in FACE_ORDER:
        face_grid_x, face_grid_y = FACE_LAYOUT[str(face)]
        face_left = float(ox + (int(face_grid_x) * 3.0 * cell))
        face_top = float(oy + (int(face_grid_y) * 3.0 * cell))
        for row in range(3):
            for col in range(3):
                x0 = float(face_left + (int(col) * cell) + gap)
                y0 = float(face_top + ((2 - int(row)) * cell) + gap)
                x1 = float(face_left + ((int(col) + 1) * cell) - gap)
                y1 = float(face_top + ((3 - int(row)) * cell) - gap)
                bbox = (x0, y0, x1, y1)
                draw.rectangle(
                    bbox,
                    fill=_state_color(
                        state,
                        color_map,
                        face=str(face),
                        row=int(row),
                        col=int(col),
                    ),
                    outline=tuple(int(value) for value in outline_rgb),
                    width=max(1, int(round(cell * 0.045))),
                )
                sticker_bbox_map[sticker_id(str(face), int(row), int(col))] = (
                    round_bbox(bbox)
                )
        if bool(include_face_labels):
            label_x = float(face_left + (1.5 * cell))
            label_y = float(face_top - (0.34 * cell))
            if str(face) == "D":
                label_y = float(face_top + (3.34 * cell))
            elif str(face) == "L":
                label_x = float(face_left - (0.35 * cell))
                label_y = float(face_top + (1.5 * cell))
            elif str(face) in {"F", "R", "B"}:
                label_y = float(face_top - (0.28 * cell))
            draw_centered_text(
                draw,
                text=str(face),
                center=(label_x, label_y),
                font=face_label_font,
                fill=text_rgb,
                stroke_fill=text_stroke_rgb,
                stroke_width=2,
            )
    net_bbox = [ox, oy, float(ox + (12.0 * cell)), float(oy + (9.0 * cell))]
    return sticker_bbox_map, round_bbox(net_bbox)


def _draw_coordinate_reference(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    params: RubiksRenderParams,
) -> list[float]:
    """Draw the small coordinate inset used by sticker-location prompts."""

    _draw_panel(
        draw,
        bbox,
        fill=params.coordinate_fill_rgb,
        outline=params.coordinate_grid_rgb,
        radius=10,
        width=2,
    )
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cell_w = float((x1 - x0) / 3.0)
    cell_h = float((y1 - y0) / 3.0)
    font = load_font(max(10, min(14, int(params.small_label_font_size_px))), bold=True)
    for index in range(1, 3):
        draw.line(
            [(float(x0 + (index * cell_w)), y0), (float(x0 + (index * cell_w)), y1)],
            fill=tuple(int(v) for v in params.coordinate_grid_rgb),
            width=1,
        )
        draw.line(
            [(x0, float(y0 + (index * cell_h))), (x1, float(y0 + (index * cell_h)))],
            fill=tuple(int(v) for v in params.coordinate_grid_rgb),
            width=1,
        )
    for row in range(3):
        for col in range(3):
            cx = float(x0 + ((col + 0.5) * cell_w))
            cy = float(y0 + (((2 - row) + 0.5) * cell_h))
            draw_centered_text(
                draw,
                text=f"({col},{row})",
                center=(cx, cy),
                font=font,
                fill=params.text_color_rgb,
                stroke_fill=params.text_stroke_rgb,
                stroke_width=1,
            )
    return round_bbox(bbox)


def _draw_color_or_number_options(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    params: RubiksRenderParams,
    top_left: tuple[float, float],
) -> dict[str, list[float]]:
    """Draw scalar option panels, either swatches or numeric counts."""

    label_font = load_font(int(params.option_label_font_size_px), bold=True)
    number_font = load_font(int(params.number_font_size_px), bold=True)
    option_map: dict[str, list[float]] = {}
    start_x, start_y = float(top_left[0]), float(top_left[1])
    for index, option in enumerate(dataset["option_specs"]):
        row = int(index) // 4
        col = int(index) % 4
        x0 = float(
            start_x + (col * (params.option_panel_width_px + params.option_gap_px))
        )
        y0 = float(
            start_y + (row * (params.option_panel_height_px + params.option_row_gap_px))
        )
        bbox = [
            x0,
            y0,
            float(x0 + params.option_panel_width_px),
            float(y0 + params.option_panel_height_px),
        ]
        _draw_panel(
            draw,
            bbox,
            fill=params.option_panel_fill_rgb,
            outline=params.border_color_rgb,
            radius=params.panel_corner_radius_px,
            width=params.border_width_px,
        )
        label = str(option["option_label"])
        draw_centered_text(
            draw,
            text=label,
            center=(
                float(x0 + (params.option_panel_width_px / 2.0)),
                float(y0 + 24),
            ),
            font=label_font,
            fill=params.text_color_rgb,
            stroke_fill=params.text_stroke_rgb,
            stroke_width=1,
        )
        content_cx = float(x0 + (params.option_panel_width_px / 2.0))
        content_cy = float(y0 + 91)
        if "color_rgb" in option:
            half = float(params.swatch_size_px / 2.0)
            swatch_bbox = [
                content_cx - half,
                content_cy - half,
                content_cx + half,
                content_cy + half,
            ]
            _draw_panel(
                draw,
                swatch_bbox,
                fill=tuple(int(v) for v in option["color_rgb"]),
                outline=params.sticker_outline_rgb,
                radius=14,
                width=3,
            )
        else:
            draw_centered_text(
                draw,
                text=str(option["count_value"]),
                center=(content_cx, content_cy),
                font=number_font,
                fill=params.text_color_rgb,
                stroke_fill=params.text_stroke_rgb,
                stroke_width=1,
            )
        option_map[str(option["option_id"])] = round_bbox(bbox)
    return option_map


def _draw_target_swatch(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    params: RubiksRenderParams,
    bbox: Sequence[float],
) -> list[float]:
    """Draw the target color swatch for face-count prompts."""

    _draw_panel(
        draw,
        bbox,
        fill=params.target_swatch_panel_fill_rgb,
        outline=params.border_color_rgb,
        radius=params.panel_corner_radius_px,
        width=params.border_width_px,
    )
    label_font = load_font(int(params.small_label_font_size_px) + 6, bold=True)
    label_rgb = (26, 31, 38)
    draw_centered_text(
        draw,
        text="TARGET",
        center=(float((bbox[0] + bbox[2]) / 2.0), float(bbox[1] + 28.0)),
        font=label_font,
        fill=label_rgb,
        stroke_fill=label_rgb,
        stroke_width=0,
    )
    half = float(params.swatch_size_px / 2.0)
    cx = float((bbox[0] + bbox[2]) / 2.0)
    cy = float(bbox[1] + 100.0)
    swatch_bbox = [cx - half, cy - half, cx + half, cy + half]
    _draw_panel(
        draw,
        swatch_bbox,
        fill=tuple(int(v) for v in dataset["target_color_rgb"]),
        outline=params.sticker_outline_rgb,
        radius=14,
        width=3,
    )
    return round_bbox(swatch_bbox)


def _draw_result_options(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    params: RubiksRenderParams,
    color_map: Mapping[str, Mapping[str, Any]],
    top_left: tuple[float, float],
    cell_size_px: float,
    sticker_gap_px: float,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Draw candidate cube-net option panels for move-result prompts."""

    label_font = load_font(int(params.option_label_font_size_px), bold=True)
    face_font = load_font(max(9, int(params.small_label_font_size_px) - 3), bold=True)
    option_panel_map: dict[str, list[float]] = {}
    candidate_net_map: dict[str, list[float]] = {}
    start_x, start_y = float(top_left[0]), float(top_left[1])
    cell = float(cell_size_px)
    option_columns = 2 if len(dataset["option_specs"]) == 4 else 3
    net_width = float(12.0 * cell)
    net_height = float(9.0 * cell)
    panel_width = float(net_width + 36.0)
    panel_height = float(net_height + 64.0)
    for index, option in enumerate(dataset["option_specs"]):
        row = int(index) // int(option_columns)
        col = int(index) % int(option_columns)
        x0 = float(
            start_x + (col * (panel_width + float(params.result_option_gap_px)))
        )
        y0 = float(
            start_y + (row * (panel_height + float(params.result_option_row_gap_px)))
        )
        panel_bbox = [
            x0,
            y0,
            float(x0 + panel_width),
            float(y0 + panel_height),
        ]
        _draw_panel(
            draw,
            panel_bbox,
            fill=params.option_panel_fill_rgb,
            outline=params.border_color_rgb,
            radius=params.panel_corner_radius_px,
            width=params.border_width_px,
        )
        label = str(option["option_label"])
        draw_centered_text(
            draw,
            text=label,
            center=(
                float(x0 + (panel_width / 2.0)),
                float(y0 + 24),
            ),
            font=label_font,
            fill=params.text_color_rgb,
            stroke_fill=params.text_stroke_rgb,
            stroke_width=1,
        )
        net_origin = (
            float(x0 + (0.5 * (panel_width - net_width))),
            float(y0 + 50.0),
        )
        _stickers, net_bbox = _draw_cube_net(
            draw,
            state=option["state"],
            color_map=color_map,
            origin=net_origin,
            cell_size_px=float(cell),
            sticker_gap_px=float(sticker_gap_px),
            outline_rgb=params.sticker_outline_rgb,
            face_label_font=face_font,
            text_rgb=params.text_color_rgb,
            text_stroke_rgb=params.text_stroke_rgb,
            include_face_labels=False,
        )
        option_panel_map[str(option["option_id"])] = round_bbox(panel_bbox)
        candidate_net_map[str(option["option_id"])] = list(net_bbox)
    return option_panel_map, candidate_net_map


def _variant_sticker_gaps(
    scene_variant: str,
    *,
    main_gap_px: float,
) -> tuple[float, float, int]:
    """Return scene-variant line treatment for main and candidate cube nets."""

    if str(scene_variant) == "paper_net":
        return float(main_gap_px + 1.0), 1.2, -1
    if str(scene_variant) == "cool_net":
        return float(max(0.0, main_gap_px - 0.5)), 0.4, 1
    return float(main_gap_px), 0.8, 0


def render_rubiks_scene(
    image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    render_params: RubiksRenderParams,
) -> RenderedRubiksScene:
    """Render the Rubik net and task-neutral option panels."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(SUPPORTED_SCENE_VARIANTS):
        raise ValueError(f"unsupported Rubik scene_variant: {scene_variant}")
    params = render_params
    draw = ImageDraw.Draw(image)
    entities: list[dict[str, Any]] = []
    scene_bbox = [0.0, 0.0, float(params.canvas_width), float(params.canvas_height)]
    main_sticker_gap, candidate_sticker_gap, border_delta = _variant_sticker_gaps(
        selected_variant,
        main_gap_px=float(params.sticker_gap_px),
    )
    result_cell_size = min(float(params.main_cell_size_px), 28.0)
    panel_border_width = max(1, int(params.border_width_px + int(border_delta)))

    label_font = load_font(int(params.face_label_font_size_px), bold=True)
    small_font = load_font(int(params.small_label_font_size_px), bold=True)
    color_map = dataset["color_map"]

    target_swatch_bbox: list[float] | None = None
    candidate_net_bbox_map: dict[str, list[float]] = {}
    option_panel_bbox_map: dict[str, list[float]] = {}

    if str(dataset["render_mode"]) == "candidate_nets":
        net_width = float(12.0 * result_cell_size)
        net_height = float(9.0 * result_cell_size)
        net_left = float(params.scene_margin_left_px + 42)
        net_top = float(params.scene_margin_top_px + 64)
        source_panel_width = float(max(580.0, net_width + 92.0))
        source_panel_height = float(max(430.0, net_height + 156.0))
        net_panel_bbox = [
            float(params.scene_margin_left_px),
            float(params.scene_margin_top_px),
            float(params.scene_margin_left_px + source_panel_width),
            float(params.scene_margin_top_px + source_panel_height),
        ]
        _draw_panel(
            draw,
            net_panel_bbox,
            fill=params.net_panel_fill_rgb,
            outline=params.border_color_rgb,
            radius=params.panel_corner_radius_px,
            width=panel_border_width,
        )
        draw_centered_text(
            draw,
            text="Start",
            center=(float(net_panel_bbox[0] + 54), float(net_panel_bbox[1] + 30)),
            font=small_font,
            fill=params.text_color_rgb,
            stroke_fill=params.text_stroke_rgb,
            stroke_width=1,
        )
        sticker_bbox_map, net_bbox = _draw_cube_net(
            draw,
            state=dataset["start_state"],
            color_map=color_map,
            origin=(net_left, net_top),
            cell_size_px=float(result_cell_size),
            sticker_gap_px=float(main_sticker_gap),
            outline_rgb=params.sticker_outline_rgb,
            face_label_font=label_font,
            text_rgb=params.text_color_rgb,
            text_stroke_rgb=params.text_stroke_rgb,
            include_face_labels=True,
        )
        option_panel_bbox_map, candidate_net_bbox_map = _draw_result_options(
            draw,
            dataset=dataset,
            params=params,
            color_map=color_map,
            cell_size_px=float(result_cell_size),
            sticker_gap_px=float(candidate_sticker_gap),
            top_left=(
                float(net_panel_bbox[2] + 30.0),
                float(params.scene_margin_top_px),
            ),
        )
    else:
        net_panel_bbox = [
            float(params.scene_margin_left_px),
            float(params.scene_margin_top_px),
            float(params.scene_margin_left_px + 700),
            float(params.scene_margin_top_px + 520),
        ]
        _draw_panel(
            draw,
            net_panel_bbox,
            fill=params.net_panel_fill_rgb,
            outline=params.border_color_rgb,
            radius=params.panel_corner_radius_px,
            width=panel_border_width,
        )
        sticker_bbox_map, net_bbox = _draw_cube_net(
            draw,
            state=dataset["start_state"],
            color_map=color_map,
            origin=(
                float(params.scene_margin_left_px + 46),
                float(params.scene_margin_top_px + 58),
            ),
            cell_size_px=float(params.main_cell_size_px),
            sticker_gap_px=float(main_sticker_gap),
            outline_rgb=params.sticker_outline_rgb,
            face_label_font=label_font,
            text_rgb=params.text_color_rgb,
            text_stroke_rgb=params.text_stroke_rgb,
            include_face_labels=True,
        )
        if str(dataset["render_mode"]) == "count_options":
            target_swatch_bbox = _draw_target_swatch(
                draw,
                dataset=dataset,
                params=params,
                bbox=[
                    float(params.scene_margin_left_px + 742),
                    float(params.scene_margin_top_px + 44),
                    float(params.scene_margin_left_px + 898),
                    float(params.scene_margin_top_px + 178),
                ],
            )
            options_top_left = (
                float(params.scene_margin_left_px + 742),
                float(params.scene_margin_top_px + 216),
            )
        else:
            coord_bbox = [
                float(net_panel_bbox[2] - 224),
                float(net_panel_bbox[1] + 330),
                float(net_panel_bbox[2] - 38),
                float(net_panel_bbox[1] + 476),
            ]
            _draw_coordinate_reference(draw, bbox=coord_bbox, params=params)
            options_top_left = (
                float(params.scene_margin_left_px + 742),
                float(params.scene_margin_top_px + 74),
            )
        option_panel_bbox_map = _draw_color_or_number_options(
            draw,
            dataset=dataset,
            params=params,
            top_left=options_top_left,
        )

    for visible_sticker_id, bbox in sticker_bbox_map.items():
        entities.append(
            {
                "entity_id": str(visible_sticker_id),
                "type": "rubiks_sticker",
                "bbox_px": list(bbox),
            }
        )
    for option_id, bbox in option_panel_bbox_map.items():
        entities.append(
            {
                "entity_id": str(option_id),
                "type": "rubiks_option_panel",
                "bbox_px": list(bbox),
            }
        )
    entities.append(
        {
            "entity_id": "rubiks_net",
            "type": "rubiks_cube_net",
            "bbox_px": list(net_bbox),
        }
    )

    return RenderedRubiksScene(
        image=image,
        entities=entities,
        scene_bbox_px=round_bbox(scene_bbox),
        net_panel_bbox_px=round_bbox(net_panel_bbox),
        net_bbox_px=list(net_bbox),
        target_swatch_bbox_px=target_swatch_bbox,
        sticker_bbox_map=dict(sticker_bbox_map),
        option_panel_bbox_map=dict(option_panel_bbox_map),
        candidate_net_bbox_map=dict(candidate_net_bbox_map),
    )


__all__ = ["render_rubiks_scene"]
