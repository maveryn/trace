"""Rendering helpers for circular-chess games tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image, ImageDraw

from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.piece_board_renderer import draw_chess_piece_symbol
from trace_tasks.tasks.games.shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.style import build_games_chess_theme
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family

from .defaults import FALLBACK_RENDERING_DEFAULTS, RING_COUNT, SCENE_ID, SECTOR_COUNT
from .rules import circular_coord_to_cell_id, circular_piece_to_entity_id, occupied_coords
from .state import Board, CircularChessRenderParams, Coord


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


@dataclass(frozen=True)
class CircularChessTheme:
    """Visual palette for one circular-chess board."""

    frame_rgb: tuple[int, int, int]
    inner_hole_rgb: tuple[int, int, int]
    light_cell_rgb: tuple[int, int, int]
    dark_cell_rgb: tuple[int, int, int]
    outline_rgb: tuple[int, int, int]
    marked_rgb: tuple[int, int, int]
    target_rgb: tuple[int, int, int]
    text_rgb: tuple[int, int, int]
    chess_style: str


@dataclass(frozen=True)
class RenderedCircularChessScene:
    """Rendered image plus trace-friendly scene metadata."""

    image: Image.Image
    scene_entities: tuple[dict[str, Any], ...]
    render_map: dict[str, Any]


def circular_chess_theme(style_variant: str) -> CircularChessTheme:
    """Return one resolved circular-chess theme."""

    variant = str(style_variant)
    if variant == "slate_ring":
        return CircularChessTheme((42, 48, 58), (230, 235, 238), (214, 221, 229), (105, 124, 145), (38, 45, 54), (218, 54, 62), (218, 54, 62), (24, 29, 36), "charcoal")
    if variant == "parchment_ring":
        return CircularChessTheme((119, 86, 52), (248, 240, 219), (242, 218, 170), (172, 124, 78), (96, 68, 42), (204, 45, 48), (204, 45, 48), (46, 32, 20), "classic")
    if variant == "emerald_ring":
        return CircularChessTheme((24, 91, 74), (225, 241, 233), (205, 231, 218), (62, 150, 121), (21, 75, 62), (216, 47, 57), (216, 47, 57), (20, 48, 42), "soft")
    if variant == "monochrome_ring":
        return CircularChessTheme((34, 34, 38), (246, 246, 244), (230, 230, 226), (132, 134, 138), (48, 50, 54), (210, 44, 54), (210, 44, 54), (24, 24, 26), "monochrome_glyph")
    return CircularChessTheme((56, 73, 122), (239, 244, 250), (222, 232, 244), (118, 150, 194), (52, 68, 104), (220, 56, 62), (220, 56, 62), (26, 34, 50), "classic")


def _sector_polygon(
    *,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    sector: int,
    segments: int = 8,
) -> tuple[tuple[float, float], ...]:
    """Return a polygon approximating one annular board sector."""

    start = -90.0 + (360.0 * float(sector) / float(SECTOR_COUNT))
    end = -90.0 + (360.0 * float(sector + 1) / float(SECTOR_COUNT))
    outer_points: list[tuple[float, float]] = []
    inner_points: list[tuple[float, float]] = []
    for index in range(int(segments) + 1):
        angle = math.radians(start + ((end - start) * float(index) / float(segments)))
        outer_points.append((center[0] + (outer_radius * math.cos(angle)), center[1] + (outer_radius * math.sin(angle))))
    for index in range(int(segments), -1, -1):
        angle = math.radians(start + ((end - start) * float(index) / float(segments)))
        inner_points.append((center[0] + (inner_radius * math.cos(angle)), center[1] + (inner_radius * math.sin(angle))))
    return tuple((round(float(x), 3), round(float(y), 3)) for x, y in [*outer_points, *inner_points])


def _coord_center(
    *,
    center: tuple[float, float],
    inner_radius: float,
    ring_width: float,
    coord: Coord,
) -> tuple[float, float]:
    """Return the pixel center of one circular board cell."""

    ring, sector = int(coord[0]), int(coord[1])
    radius = float(inner_radius + ((ring + 0.5) * ring_width))
    angle = math.radians(-90.0 + (360.0 * (float(sector) + 0.5) / float(SECTOR_COUNT)))
    return (round(float(center[0] + (radius * math.cos(angle))), 3), round(float(center[1] + (radius * math.sin(angle))), 3))


def _piece_bbox(center: tuple[float, float], *, size_px: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(size_px)
    return (
        round(float(center[0] - half), 3),
        round(float(center[1] - half), 3),
        round(float(center[0] + half), 3),
        round(float(center[1] + half), 3),
    )


def resolve_circular_chess_render_params(params: Mapping[str, Any], *, instance_seed: int) -> CircularChessRenderParams:
    """Resolve rendering controls for one circular-chess board."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.circular_chess.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.circular_chess.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.circular_chess.layout",
        ),
        unit_scale_meta,
    )
    max_board_size_px = scale_games_px(
        params.get("max_board_size_px", group_default(_RENDER_DEFAULTS, "max_board_size_px", FALLBACK_RENDERING_DEFAULTS["max_board_size_px"])),
        unit_scale,
        min_px=360,
    )
    dynamic_canvas_enabled = bool(params.get("dynamic_canvas_size_enabled", group_default(_RENDER_DEFAULTS, "dynamic_canvas_size_enabled", FALLBACK_RENDERING_DEFAULTS["dynamic_canvas_size_enabled"])))
    base_canvas_width = int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", FALLBACK_RENDERING_DEFAULTS["canvas_width"])))
    base_canvas_height = int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", FALLBACK_RENDERING_DEFAULTS["canvas_height"])))
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(_RENDER_DEFAULTS, "canvas_min_width_px", FALLBACK_RENDERING_DEFAULTS["canvas_min_width_px"]))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_side_padding_px", group_default(_RENDER_DEFAULTS, "canvas_side_padding_px", FALLBACK_RENDERING_DEFAULTS["canvas_side_padding_px"])))))),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(_RENDER_DEFAULTS, "canvas_min_height_px", FALLBACK_RENDERING_DEFAULTS["canvas_min_height_px"]))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_vertical_padding_px", group_default(_RENDER_DEFAULTS, "canvas_vertical_padding_px", FALLBACK_RENDERING_DEFAULTS["canvas_vertical_padding_px"])))))),
            ),
        )
    return CircularChessRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", FALLBACK_RENDERING_DEFAULTS["panel_margin_px"]))),
        max_board_size_px=int(max_board_size_px),
        board_frame_width_px=scale_games_px(params.get("board_frame_width_px", group_default(_RENDER_DEFAULTS, "board_frame_width_px", FALLBACK_RENDERING_DEFAULTS["board_frame_width_px"])), unit_scale, min_px=4),
        cell_outline_width_px=scale_games_px(params.get("cell_outline_width_px", group_default(_RENDER_DEFAULTS, "cell_outline_width_px", FALLBACK_RENDERING_DEFAULTS["cell_outline_width_px"])), unit_scale, min_px=1),
        piece_font_size_px=scale_games_px(params.get("piece_font_size_px", group_default(_RENDER_DEFAULTS, "piece_font_size_px", FALLBACK_RENDERING_DEFAULTS["piece_font_size_px"])), unit_scale, min_px=34),
        piece_bbox_fraction=float(params.get("piece_bbox_fraction", group_default(_RENDER_DEFAULTS, "piece_bbox_fraction", FALLBACK_RENDERING_DEFAULTS["piece_bbox_fraction"]))),
        marker_width_px=scale_games_px(params.get("marker_width_px", group_default(_RENDER_DEFAULTS, "marker_width_px", FALLBACK_RENDERING_DEFAULTS["marker_width_px"])), unit_scale, min_px=3),
        layout_jitter_meta=dict(layout_jitter),
        font_family=str(font_family),
    )


def resolve_scene_background(
    *,
    params: Mapping[str, Any],
    render_params: CircularChessRenderParams,
    instance_seed: int,
) -> tuple[Image.Image, dict[str, Any], GamePanelSceneStyle | None, dict[str, Any]]:
    """Resolve panel-style background for one scene."""

    allowed_raw = params.get("panel_scene_treatments", group_default(_RENDER_DEFAULTS, "panel_scene_treatments", None))
    if isinstance(allowed_raw, str):
        allowed = (str(allowed_raw),)
    elif allowed_raw is None:
        allowed = None
    else:
        allowed = tuple(str(item) for item in allowed_raw)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.circular_chess.panel_scene_style",
        treatments=allowed,
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(_RENDER_DEFAULTS, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(_RENDER_DEFAULTS, "panel_scene_palette_weights", None)),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    return background, dict(background_meta), panel_style, dict(panel_style_meta)


def text_style_metadata(font_family: str) -> dict[str, Any]:
    """Return trace metadata for rendered text."""

    return {
        "font_family": str(font_family),
        "font_asset": get_font_family_record(str(font_family)).to_trace(),
    }


def render_circular_chess_scene(
    *,
    board: Board,
    background: Image.Image,
    style_variant: str,
    params: CircularChessRenderParams,
    marked_coord: Coord | None,
    target_coord: Coord | None,
    panel_style: GamePanelSceneStyle | None,
) -> RenderedCircularChessScene:
    """Render the circular board and record every cell/piece projection."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    theme = circular_chess_theme(str(style_variant))
    board_size = int(params.max_board_size_px)
    board_left = int(0.5 * (int(params.canvas_width) - board_size))
    board_top = int(0.5 * (int(params.canvas_height) - board_size))
    board_bbox = (float(board_left), float(board_top), float(board_left + board_size), float(board_top + board_size))
    _group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=dict(params.layout_jitter_meta),
    )
    board_bbox = (
        round(float(board_bbox[0] + dx), 3),
        round(float(board_bbox[1] + dy), 3),
        round(float(board_bbox[2] + dx), 3),
        round(float(board_bbox[3] + dy), 3),
    )
    if panel_style is not None:
        pad = max(18, int(round(float(params.panel_margin_px) * 0.42)))
        panel_bbox = (
            max(4, int(round(board_bbox[0])) - pad),
            max(4, int(round(board_bbox[1])) - pad),
            min(int(params.canvas_width) - 4, int(round(board_bbox[2])) + pad),
            min(int(params.canvas_height) - 4, int(round(board_bbox[3])) + pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=30,
            border_width=max(2, int(round(float(params.board_frame_width_px) * 0.45))),
        )
    else:
        panel_bbox = None

    cx = float(0.5 * (board_bbox[0] + board_bbox[2]))
    cy = float(0.5 * (board_bbox[1] + board_bbox[3]))
    outer_radius = float(0.5 * (board_bbox[2] - board_bbox[0]))
    inner_radius = float(0.18 * outer_radius)
    ring_width = float((outer_radius - inner_radius) / float(RING_COUNT))
    center = (float(cx), float(cy))
    draw.ellipse(
        (
            cx - outer_radius - params.board_frame_width_px,
            cy - outer_radius - params.board_frame_width_px,
            cx + outer_radius + params.board_frame_width_px,
            cy + outer_radius + params.board_frame_width_px,
        ),
        fill=tuple(int(value) for value in theme.frame_rgb),
    )

    cell_centers: dict[str, tuple[float, float]] = {}
    cell_polygons: dict[str, list[list[float]]] = {}
    scene_entities: list[dict[str, Any]] = []
    for ring in range(RING_COUNT):
        inner = float(inner_radius + (ring * ring_width))
        outer = float(inner + ring_width)
        for sector in range(SECTOR_COUNT):
            coord = (int(ring), int(sector))
            cell_id = circular_coord_to_cell_id(coord)
            polygon = _sector_polygon(center=center, inner_radius=inner, outer_radius=outer, sector=int(sector))
            fill_rgb = theme.light_cell_rgb if (int(ring) + int(sector)) % 2 == 0 else theme.dark_cell_rgb
            draw.polygon(polygon, fill=tuple(int(value) for value in fill_rgb))
            draw.line([*polygon, polygon[0]], fill=tuple(int(value) for value in theme.outline_rgb), width=int(params.cell_outline_width_px))
            cell_center = _coord_center(center=center, inner_radius=inner_radius, ring_width=ring_width, coord=coord)
            cell_centers[str(cell_id)] = tuple(cell_center)
            cell_polygons[str(cell_id)] = [[float(x), float(y)] for x, y in polygon]
            occupant = board[int(ring)][int(sector)]
            scene_entities.append(
                {
                    "entity_id": str(cell_id),
                    "entity_type": "circular_chess_cell",
                    "ring": int(ring),
                    "sector": int(sector),
                    "occupant": None if occupant is None else f"{occupant.color}_{occupant.kind}",
                    "center_px": [float(cell_center[0]), float(cell_center[1])],
                    "polygon_px": [[float(x), float(y)] for x, y in polygon],
                }
            )

    draw.ellipse(
        (cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius),
        fill=tuple(int(value) for value in theme.inner_hole_rgb),
        outline=tuple(int(value) for value in theme.outline_rgb),
        width=int(params.cell_outline_width_px),
    )

    piece_centers: dict[str, tuple[float, float]] = {}
    piece_bboxes: dict[str, tuple[float, float, float, float]] = {}
    chess_theme = build_games_chess_theme(style_variant=str(theme.chess_style))
    piece_size = max(20.0, float(params.piece_bbox_fraction) * float(ring_width))
    for coord in occupied_coords(board):
        piece = board[int(coord[0])][int(coord[1])]
        if piece is None:
            continue
        piece_id = circular_piece_to_entity_id(coord, piece)
        center_px = cell_centers[circular_coord_to_cell_id(coord)]
        bbox = _piece_bbox(center_px, size_px=piece_size)
        draw_chess_piece_symbol(draw, bbox_px=bbox, piece=piece, theme=chess_theme, font_size_px=int(params.piece_font_size_px))
        piece_centers[str(piece_id)] = tuple(center_px)
        piece_bboxes[str(piece_id)] = tuple(bbox)
        scene_entities.append(
            {
                "entity_id": str(piece_id),
                "entity_type": "circular_chess_piece",
                "ring": int(coord[0]),
                "sector": int(coord[1]),
                "color": str(piece.color),
                "kind": str(piece.kind),
                "center_px": [float(center_px[0]), float(center_px[1])],
                "bbox_px": [float(value) for value in bbox],
            }
        )

    marker_meta: dict[str, Any] = {}
    if target_coord is not None:
        target_center = cell_centers[circular_coord_to_cell_id(target_coord)]
        radius = max(10.0, 0.33 * float(ring_width))
        bbox = (
            float(target_center[0] - radius),
            float(target_center[1] - radius),
            float(target_center[0] + radius),
            float(target_center[1] + radius),
        )
        draw.ellipse(bbox, outline=tuple(int(value) for value in theme.target_rgb), width=int(params.marker_width_px))
        marker_meta["target_cell_marker_bbox_px"] = [float(value) for value in bbox]
    if marked_coord is not None:
        marked_center = cell_centers[circular_coord_to_cell_id(marked_coord)]
        radius = max(11.0, 0.36 * float(ring_width))
        bbox = (
            float(marked_center[0] - radius),
            float(marked_center[1] - radius),
            float(marked_center[0] + radius),
            float(marked_center[1] + radius),
        )
        draw.ellipse(bbox, outline=tuple(int(value) for value in theme.marked_rgb), width=int(params.marker_width_px))
        marker_meta["marked_piece_marker_bbox_px"] = [float(value) for value in bbox]

    return RenderedCircularChessScene(
        image=image,
        scene_entities=tuple(scene_entities),
        render_map={
            "board_bbox_px": [float(value) for value in board_bbox],
            "scene_panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
            "cell_centers_px": {str(key): [float(value[0]), float(value[1])] for key, value in cell_centers.items()},
            "cell_polygons_px": dict(cell_polygons),
            "piece_centers_px": {str(key): [float(value[0]), float(value[1])] for key, value in piece_centers.items()},
            "piece_bboxes_px": {str(key): [float(v) for v in value] for key, value in piece_bboxes.items()},
            "layout_jitter": dict(layout_jitter),
            "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
            "style_variant": str(style_variant),
            "effective_ring_width_px": float(ring_width),
            "font_family": str(params.font_family),
            **marker_meta,
        },
    )


__all__ = [
    "RenderedCircularChessScene",
    "circular_chess_theme",
    "render_circular_chess_scene",
    "resolve_circular_chess_render_params",
    "resolve_scene_background",
    "text_style_metadata",
]
