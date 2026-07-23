"""Rendering primitives for the 3D Tic-Tac-Toe board scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import (
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_centered_game_text
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import DEFAULTS, GEN_DEFAULTS, RENDER_DEFAULTS
from .rules import board_get, coord_id, layer_id
from .state import (
    BOARD_SIZE,
    BBox,
    BoardVisualStyle,
    LAYERS,
    OPTION_LABELS,
    RenderedTicTacToe3DScene,
    SCENE_NAMESPACE,
    STYLE_VARIANTS,
    TicTacToe3DAxes,
    TicTacToe3DSample,
)


def resolve_board_visual_style(style_variant: str) -> tuple[BoardVisualStyle, dict[str, Any]]:
    """Resolve a scene-local board palette that preserves X/O contrast."""

    styles = {
        "classic_grid": BoardVisualStyle(
            layer_fill_rgb=(235, 241, 250),
            cell_fill_rgb=(249, 252, 255),
            grid_rgb=(75, 94, 126),
            border_rgb=(35, 58, 92),
            x_rgb=(7, 39, 104),
            o_rgb=(126, 19, 36),
            option_fill_rgb=(255, 224, 79),
            option_outline_rgb=(53, 62, 83),
            option_text_rgb=(27, 31, 43),
            label_rgb=(30, 48, 74),
        ),
        "paper_board": BoardVisualStyle(
            layer_fill_rgb=(241, 234, 213),
            cell_fill_rgb=(252, 248, 232),
            grid_rgb=(129, 105, 76),
            border_rgb=(92, 70, 51),
            x_rgb=(18, 52, 96),
            o_rgb=(125, 39, 27),
            option_fill_rgb=(242, 211, 98),
            option_outline_rgb=(84, 67, 45),
            option_text_rgb=(35, 30, 24),
            label_rgb=(64, 48, 34),
        ),
        "arcade_blue": BoardVisualStyle(
            layer_fill_rgb=(23, 43, 76),
            cell_fill_rgb=(32, 61, 104),
            grid_rgb=(98, 210, 234),
            border_rgb=(73, 232, 237),
            x_rgb=(226, 252, 255),
            o_rgb=(255, 219, 93),
            option_fill_rgb=(255, 231, 67),
            option_outline_rgb=(255, 255, 255),
            option_text_rgb=(28, 31, 39),
            label_rgb=(225, 247, 255),
        ),
        "mint_table": BoardVisualStyle(
            layer_fill_rgb=(212, 235, 225),
            cell_fill_rgb=(238, 249, 241),
            grid_rgb=(72, 135, 113),
            border_rgb=(32, 93, 76),
            x_rgb=(5, 70, 132),
            o_rgb=(128, 24, 51),
            option_fill_rgb=(255, 227, 111),
            option_outline_rgb=(36, 92, 67),
            option_text_rgb=(28, 44, 35),
            label_rgb=(27, 74, 61),
        ),
        "charcoal_lines": BoardVisualStyle(
            layer_fill_rgb=(54, 58, 65),
            cell_fill_rgb=(72, 77, 86),
            grid_rgb=(171, 183, 197),
            border_rgb=(226, 232, 240),
            x_rgb=(235, 250, 255),
            o_rgb=(255, 216, 118),
            option_fill_rgb=(255, 218, 78),
            option_outline_rgb=(250, 250, 245),
            option_text_rgb=(24, 27, 32),
            label_rgb=(239, 244, 250),
        ),
    }
    resolved = str(style_variant) if str(style_variant) in styles else "classic_grid"
    return styles[resolved], {
        "style_variant": str(resolved),
        "available_styles": list(STYLE_VARIANTS),
        "board_style_policy": "scene_local_tic_tac_toe_3d_palette",
    }


def _int_default(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer rendering default from params or config."""

    if str(key) in params:
        return int(params[str(key)])
    return int(group_default(RENDER_DEFAULTS, str(key), int(fallback)))


def _center_of_bbox(bbox: Sequence[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _polygon_bbox(points: Sequence[Sequence[float]]) -> BBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _polygon_center(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    if not points:
        raise ValueError("points must be non-empty")
    return (
        sum(float(point[0]) for point in points) / float(len(points)),
        sum(float(point[1]) for point in points) / float(len(points)),
    )


def _grid_point(
    *,
    origin_x: float,
    origin_y: float,
    row: int,
    col: int,
    col_step: float,
    row_step: float,
    skew_x: float,
    skew_y: float,
) -> tuple[float, float]:
    return (
        float(origin_x + (int(col) * float(col_step)) + (int(row) * float(skew_x))),
        float(origin_y + (int(row) * float(row_step)) + (int(col) * float(skew_y))),
    )


def _cell_polygon(
    *,
    origin_x: float,
    origin_y: float,
    row: int,
    col: int,
    col_step: float,
    row_step: float,
    skew_x: float,
    skew_y: float,
) -> tuple[tuple[float, float], ...]:
    return (
        _grid_point(origin_x=origin_x, origin_y=origin_y, row=row, col=col, col_step=col_step, row_step=row_step, skew_x=skew_x, skew_y=skew_y),
        _grid_point(origin_x=origin_x, origin_y=origin_y, row=row, col=col + 1, col_step=col_step, row_step=row_step, skew_x=skew_x, skew_y=skew_y),
        _grid_point(origin_x=origin_x, origin_y=origin_y, row=row + 1, col=col + 1, col_step=col_step, row_step=row_step, skew_x=skew_x, skew_y=skew_y),
        _grid_point(origin_x=origin_x, origin_y=origin_y, row=row + 1, col=col, col_step=col_step, row_step=row_step, skew_x=skew_x, skew_y=skew_y),
    )


def _contrast_halo_rgb(mark_rgb: Sequence[int]) -> tuple[int, int, int]:
    """Return an opposite-luminance halo for hand-drawn X/O board marks."""

    return (10, 14, 22) if sum(int(channel) for channel in mark_rgb[:3]) >= 470 else (250, 252, 255)


def _draw_tic_tac_toe_mark(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    mark: str,
    mark_rgb: Sequence[int],
    mark_size_px: int,
) -> None:
    """Draw a high-contrast X/O mark without depending on font glyph shape."""

    cx, cy = float(center[0]), float(center[1])
    size = max(24.0, float(mark_size_px))
    half_x = float(size) * 0.34
    half_y = float(size) * 0.23
    stroke_width = max(4, int(round(float(size) * 0.11)))
    halo_width = int(stroke_width + max(3, int(round(float(size) * 0.055))))
    fill = tuple(int(channel) for channel in mark_rgb[:3])
    halo = _contrast_halo_rgb(fill)
    if str(mark) == "X":
        segments = (
            ((cx - half_x, cy - half_y), (cx + half_x, cy + half_y)),
            ((cx - half_x, cy + half_y), (cx + half_x, cy - half_y)),
        )
        for p0, p1 in segments:
            draw.line((p0, p1), fill=halo, width=halo_width)
        for p0, p1 in segments:
            draw.line((p0, p1), fill=fill, width=stroke_width)
        return
    bbox = (cx - half_x, cy - half_y, cx + half_x, cy + half_y)
    draw.ellipse(bbox, outline=halo, width=halo_width)
    draw.ellipse(bbox, outline=fill, width=stroke_width)


def render_tic_tac_toe_3d_scene(
    *,
    sample: TicTacToe3DSample,
    axes: TicTacToe3DAxes,
    instance_seed: int,
    params: Mapping[str, Any],
) -> RenderedTicTacToe3DScene:
    """Render stacked boards and record every cell projection for annotation."""

    base_canvas_width = _int_default(params, "canvas_width", DEFAULTS.canvas_width)
    base_canvas_height = _int_default(params, "canvas_height", DEFAULTS.canvas_height)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_style",
        treatment_weights=group_default(GEN_DEFAULTS, "panel_scene_treatment_weights", None),
        palette_weights=group_default(GEN_DEFAULTS, "panel_scene_palette_weights", None),
    )
    layout_jitter = resolve_games_layout_jitter(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    unit_scale, unit_meta = resolve_games_unit_size_scale(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.unit_size",
    )
    cell = scale_games_px(_int_default(params, "cell_size_px", DEFAULTS.cell_size_px), unit_scale, min_px=30)
    cell_gap = scale_games_px(_int_default(params, "cell_gap_px", DEFAULTS.cell_gap_px), unit_scale, min_px=0)
    layer_gap = scale_games_px(_int_default(params, "layer_gap_px", DEFAULTS.layer_gap_px), unit_scale, min_px=18)
    skew_x = scale_games_px(_int_default(params, "skew_x_px", DEFAULTS.skew_x_px), unit_scale, min_px=14)
    skew_y = scale_games_px(_int_default(params, "skew_y_px", DEFAULTS.skew_y_px), unit_scale, min_px=7)
    inner = scale_games_px(_int_default(params, "panel_inner_padding_px", DEFAULTS.panel_inner_padding_px), unit_scale, min_px=26)
    col_step = float(cell)
    row_step = float(cell) * 0.48
    layer_width = int(round((BOARD_SIZE * col_step) + (BOARD_SIZE * skew_x)))
    layer_height = int(round((BOARD_SIZE * row_step) + (BOARD_SIZE * skew_y)))
    raw_panel_width = int(layer_width + 2 * inner)
    raw_panel_height = int((BOARD_SIZE * layer_height) + ((BOARD_SIZE - 1) * layer_gap) + 2 * inner)
    dynamic_canvas_enabled = bool(params.get("dynamic_canvas_size_enabled", group_default(RENDER_DEFAULTS, "dynamic_canvas_size_enabled", True)))
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        side_padding = _int_default(params, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px)
        canvas_width = min(
            int(base_canvas_width),
            max(_int_default(params, "canvas_min_width_px", DEFAULTS.canvas_min_width_px), int(raw_panel_width + side_padding)),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        side_padding = _int_default(params, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px)
        canvas_height = min(
            int(base_canvas_height),
            max(_int_default(params, "canvas_min_height_px", DEFAULTS.canvas_min_height_px), int(raw_panel_height + side_padding)),
        )
    margin = _int_default(params, "panel_margin_px", DEFAULTS.panel_margin_px)
    panel_width = min(int(raw_panel_width), int(canvas_width - 2 * margin))
    panel_height = min(int(raw_panel_height), int(canvas_height - 2 * margin))
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=panel_style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    base_panel = (
        float((canvas_width - panel_width) / 2.0),
        float((canvas_height - panel_height) / 2.0),
        float((canvas_width + panel_width) / 2.0),
        float((canvas_height + panel_height) / 2.0),
    )
    panel_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_panel,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    draw_panel_scene_chrome(
        draw,
        bbox=tuple(int(round(value)) for value in panel_bbox),
        style=panel_style,
        radius=18,
        border_width=3,
    )
    board_style, board_style_meta = resolve_board_visual_style(str(axes.style_variant))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font_family",
        params=params,
    )
    mark_size = scale_games_px(_int_default(params, "mark_size_px", DEFAULTS.mark_size_px), unit_scale, min_px=30)
    option_font = load_font(
        scale_games_px(_int_default(params, "option_font_size_px", DEFAULTS.option_font_size_px), unit_scale, min_px=13),
        bold=True,
        font_family=str(font_family),
    )
    content_left = float(panel_bbox[0] + inner)
    content_top = float(panel_bbox[1] + inner)
    layer_origins = {
        int(z): (float(content_left), float(content_top + float(z * (layer_height + layer_gap))))
        for z in range(BOARD_SIZE)
    }
    option_label_by_coord = {coord: OPTION_LABELS[index] for index, coord in enumerate(sample.option_cells)}
    entities: list[dict[str, Any]] = []
    entity_bboxes: dict[str, list[float]] = {}
    entity_points: dict[str, list[float]] = {}
    layer_bboxes: dict[str, list[float]] = {}
    cell_bboxes: dict[str, list[float]] = {}
    cell_centers: dict[str, list[float]] = {}

    for z in range(BOARD_SIZE):
        origin_x, origin_y = layer_origins[int(z)]
        layer_poly = (
            _grid_point(origin_x=origin_x, origin_y=origin_y, row=0, col=0, col_step=col_step, row_step=row_step, skew_x=float(skew_x), skew_y=float(skew_y)),
            _grid_point(origin_x=origin_x, origin_y=origin_y, row=0, col=BOARD_SIZE, col_step=col_step, row_step=row_step, skew_x=float(skew_x), skew_y=float(skew_y)),
            _grid_point(origin_x=origin_x, origin_y=origin_y, row=BOARD_SIZE, col=BOARD_SIZE, col_step=col_step, row_step=row_step, skew_x=float(skew_x), skew_y=float(skew_y)),
            _grid_point(origin_x=origin_x, origin_y=origin_y, row=BOARD_SIZE, col=0, col_step=col_step, row_step=row_step, skew_x=float(skew_x), skew_y=float(skew_y)),
        )
        layer_bbox = _polygon_bbox(layer_poly)
        layer_key = layer_id(int(z))
        rounded_layer_bbox = [round(float(value), 3) for value in layer_bbox]
        layer_bboxes[layer_key] = list(rounded_layer_bbox)
        entity_bboxes[layer_key] = list(rounded_layer_bbox)
        entities.append(
            {
                "entity_id": str(layer_key),
                "entity_type": "tic_tac_toe_3d_layer",
                "layer_name": str(LAYERS[int(z)][0]),
                "bbox_px": list(rounded_layer_bbox),
            }
        )
        draw.polygon(layer_poly, fill=tuple(board_style.layer_fill_rgb), outline=tuple(board_style.border_rgb))
        draw.line((*layer_poly, layer_poly[0]), fill=tuple(board_style.border_rgb), width=3)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                coord = (int(z), int(row), int(col))
                cell_poly = _cell_polygon(
                    origin_x=origin_x,
                    origin_y=origin_y,
                    row=int(row),
                    col=int(col),
                    col_step=col_step,
                    row_step=row_step,
                    skew_x=float(skew_x),
                    skew_y=float(skew_y),
                )
                bbox = _polygon_bbox(cell_poly)
                draw.polygon(cell_poly, fill=tuple(board_style.cell_fill_rgb), outline=tuple(board_style.grid_rgb))
                draw.line((*cell_poly, cell_poly[0]), fill=tuple(board_style.grid_rgb), width=2)
                cell_key = coord_id(coord)
                bbox_list = [round(float(value), 3) for value in bbox]
                center = _polygon_center(cell_poly)
                center_list = [round(float(center[0]), 3), round(float(center[1]), 3)]
                entity_bboxes[cell_key] = list(bbox_list)
                cell_bboxes[cell_key] = list(bbox_list)
                entity_points[cell_key] = list(center_list)
                cell_centers[cell_key] = list(center_list)
                value = board_get(sample.board, coord)
                entities.append(
                    {
                        "entity_id": str(cell_key),
                        "entity_type": "tic_tac_toe_3d_cell",
                        "layer_name": str(LAYERS[int(z)][0]),
                        "layer_index": int(z),
                        "row": int(row + 1),
                        "col": int(col + 1),
                        "mark": str(value),
                        "option_label": str(option_label_by_coord.get(coord, "")),
                        "bbox_px": list(bbox_list),
                        "center_px": list(center_list),
                    }
                )
                if value:
                    mark_fill = board_style.x_rgb if str(value) == "X" else board_style.o_rgb
                    _draw_tic_tac_toe_mark(
                        draw,
                        center=center,
                        mark=str(value),
                        mark_rgb=mark_fill,
                        mark_size_px=int(mark_size),
                    )
                if coord in option_label_by_coord:
                    label = option_label_by_coord[coord]
                    radius = max(8.0, float(cell) * 0.18)
                    label_bbox = (
                        float(center[0] - radius),
                        float(center[1] - radius),
                        float(center[0] + radius),
                        float(center[1] + radius),
                    )
                    draw.ellipse(label_bbox, fill=tuple(board_style.option_fill_rgb), outline=tuple(board_style.option_outline_rgb), width=2)
                    draw_centered_game_text(
                        draw,
                        text=str(label),
                        center=_center_of_bbox(label_bbox),
                        font=option_font,
                        fill=board_style.option_text_rgb,
                        stroke_fill=board_style.option_fill_rgb,
                        stroke_width=0,
                        role="board_mark",
                        required=True,
                        surface_rgbs=[board_style.option_fill_rgb],
                        preferred_rgbs=[board_style.option_text_rgb],
                        candidate_rgbs=[board_style.option_text_rgb],
                        instance_seed=int(instance_seed),
                        namespace=f"{SCENE_NAMESPACE}.option.{label}",
                    )

    render_map = {
        "entity_bboxes_px": dict(entity_bboxes),
        "entity_points_px": dict(entity_points),
        "layer_bboxes_px": dict(layer_bboxes),
        "cell_bboxes_px": dict(cell_bboxes),
        "cell_centers_px": dict(cell_centers),
        "panel_bbox_px": [round(float(value), 3) for value in panel_bbox],
        "layout_variant": str(axes.layout_variant),
        "panel_scene_style": dict(panel_style_meta),
        "tic_tac_toe_3d_board_style": dict(board_style_meta),
        "font_family": str(font_family),
        "text_style": {"font_family": str(font_family), "board_marks": "drawn_x_o_strokes_v1"},
        "layout_jitter": attach_games_unit_size_jitter(resolved_jitter, unit_meta),
        "effective_cell_size_px": int(cell),
        "effective_cell_gap_px": int(cell_gap),
        "effective_mark_size_px": int(mark_size),
        "effective_projection": {
            "col_step_px": round(float(col_step), 3),
            "row_step_px": round(float(row_step), 3),
            "skew_x_px": int(skew_x),
            "skew_y_px": int(skew_y),
            "layer_width_px": int(layer_width),
            "layer_height_px": int(layer_height),
        },
        "dynamic_canvas": {
            "enabled": bool(dynamic_canvas_enabled),
            "base_canvas_width": int(base_canvas_width),
            "base_canvas_height": int(base_canvas_height),
            "raw_panel_width_px": int(raw_panel_width),
            "raw_panel_height_px": int(raw_panel_height),
            "resolved_canvas_width": int(canvas_width),
            "resolved_canvas_height": int(canvas_height),
        },
    }
    return RenderedTicTacToe3DScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=dict(render_map),
        style_meta={
            "panel_scene_style": dict(panel_style_meta),
            "tic_tac_toe_3d_board_style": dict(board_style_meta),
            "text_style": {
                "font_family": str(font_family),
                "font_asset": get_font_family_record(str(font_family)).to_trace(),
            },
        },
        background_meta=dict(background_meta),
    )


__all__ = ["render_tic_tac_toe_3d_scene", "resolve_board_visual_style"]
