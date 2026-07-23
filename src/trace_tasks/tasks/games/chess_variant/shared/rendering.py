"""Rendering helpers for chess-variant games tasks."""

from __future__ import annotations

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
from trace_tasks.tasks.games.shared.piece_board_rules import BOARD_SIZE, Board, Coord, coord_to_cell_id, piece_to_entity_id
from trace_tasks.tasks.games.shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.style import build_games_chess_theme
from trace_tasks.tasks.games.shared.text import draw_game_text_traced
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, resolve_text_stroke_fill

from .defaults import FALLBACK_RENDERING_DEFAULTS, SCENE_ID
from .state import ChessVariantRenderParams, ChessVariantSceneAxes


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def _rendering_default(key: str) -> Any:
    return group_default(_RENDER_DEFAULTS, str(key), FALLBACK_RENDERING_DEFAULTS[str(key)])


def resolve_chess_variant_render_params(params: Mapping[str, Any], *, instance_seed: int) -> ChessVariantRenderParams:
    """Resolve rendering parameters for one chess-variant scene."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.chess_variant.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.chess_variant.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.chess_variant.layout",
        ),
        unit_scale_meta,
    )
    max_board_size_px = scale_games_px(
        params.get("max_board_size_px", _rendering_default("max_board_size_px")),
        unit_scale,
        min_px=390,
    )
    rule_badge_height_px = scale_games_px(
        params.get("rule_badge_height_px", _rendering_default("rule_badge_height_px")),
        unit_scale,
        min_px=38,
    )
    rule_badge_width_px = scale_games_px(
        params.get("rule_badge_width_px", _rendering_default("rule_badge_width_px")),
        unit_scale,
        min_px=260,
    )
    header_gap_px = scale_games_px(
        params.get("header_gap_px", _rendering_default("header_gap_px")),
        unit_scale,
        min_px=10,
    )
    dynamic_canvas_enabled = bool(params.get("dynamic_canvas_size_enabled", _rendering_default("dynamic_canvas_size_enabled")))
    base_canvas_width = int(params.get("canvas_width", _rendering_default("canvas_width")))
    base_canvas_height = int(params.get("canvas_height", _rendering_default("canvas_height")))
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", _rendering_default("canvas_min_width_px"))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_side_padding_px", _rendering_default("canvas_side_padding_px")))))),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", _rendering_default("canvas_min_height_px"))),
                int(
                    round(
                        float(max_board_size_px)
                        + float(rule_badge_height_px)
                        + float(header_gap_px)
                        + (2.0 * float(params.get("canvas_vertical_padding_px", _rendering_default("canvas_vertical_padding_px"))))
                    )
                ),
            ),
        )
    return ChessVariantRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", _rendering_default("panel_margin_px"))),
        rule_badge_height_px=int(rule_badge_height_px),
        rule_badge_width_px=int(rule_badge_width_px),
        header_gap_px=int(header_gap_px),
        max_board_size_px=int(max_board_size_px),
        board_corner_radius_px=scale_games_px(
            params.get("board_corner_radius_px", _rendering_default("board_corner_radius_px")),
            unit_scale,
            min_px=10,
        ),
        board_frame_width_px=scale_games_px(
            params.get("board_frame_width_px", _rendering_default("board_frame_width_px")),
            unit_scale,
            min_px=5,
        ),
        piece_inset_fraction=float(params.get("piece_inset_fraction", _rendering_default("piece_inset_fraction"))),
        marked_square_outline_width_px=scale_games_px(
            params.get("marked_square_outline_width_px", _rendering_default("marked_square_outline_width_px")),
            unit_scale,
            min_px=4,
        ),
        rule_badge_font_size_px=scale_games_px(
            params.get("rule_badge_font_size_px", _rendering_default("rule_badge_font_size_px")),
            unit_scale,
            min_px=16,
        ),
        piece_font_size_px=scale_games_px(
            params.get("piece_font_size_px", _rendering_default("piece_font_size_px")),
            unit_scale,
            min_px=32,
        ),
        layout_jitter_meta=layout_jitter,
        font_family=str(font_family),
    )


def piece_bbox(cell_bbox: tuple[float, float, float, float], *, inset_fraction: float) -> tuple[float, float, float, float]:
    """Return the piece bbox inside one board cell."""

    left, top, right, bottom = cell_bbox
    inset = float(max(5.0, float(inset_fraction) * min(right - left, bottom - top)))
    return (round(left + inset, 3), round(top + inset, 3), round(right - inset, 3), round(bottom - inset, 3))


def rule_badge_text(rule_family: str, range_k: int, prompt_defaults: Mapping[str, Any]) -> str:
    """Return the visible rule-card text."""

    key = f"rule_badge_{str(rule_family)}"
    return str(prompt_defaults[key]).format(range_k=int(range_k))


def render_chess_variant_scene(
    *,
    board: Board,
    axes: ChessVariantSceneAxes,
    background: Image.Image,
    params: ChessVariantRenderParams,
    badge_text: str,
    panel_style: GamePanelSceneStyle | None = None,
) -> tuple[Image.Image, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Render a chess-variant board and return image, render map, and entities."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    theme = build_games_chess_theme(style_variant=str(axes.style_variant))
    cell_size = min(
        int(params.max_board_size_px) // BOARD_SIZE,
        (int(params.canvas_width) - (2 * int(params.panel_margin_px))) // BOARD_SIZE,
        (
            int(params.canvas_height)
            - (2 * int(params.panel_margin_px))
            - int(params.rule_badge_height_px)
            - int(params.header_gap_px)
        )
        // BOARD_SIZE,
    )
    board_size_px = int(cell_size) * BOARD_SIZE
    board_left = int(0.5 * (int(params.canvas_width) - int(board_size_px)))
    available_height = (
        int(params.canvas_height)
        - (2 * int(params.panel_margin_px))
        - int(params.rule_badge_height_px)
        - int(params.header_gap_px)
    )
    board_top = int(
        params.panel_margin_px
        + params.rule_badge_height_px
        + params.header_gap_px
        + max(0, 0.5 * (available_height - board_size_px))
    )
    board_bbox = (float(board_left), float(board_top), float(board_left + board_size_px), float(board_top + board_size_px))
    badge_width = int(params.rule_badge_width_px)
    badge_left = int(0.5 * (int(params.canvas_width) - int(badge_width)))
    badge_top = int(params.panel_margin_px)
    badge_bbox = (float(badge_left), float(badge_top), float(badge_left + badge_width), float(badge_top + int(params.rule_badge_height_px)))
    group_bbox = (
        min(board_bbox[0], badge_bbox[0]),
        min(board_bbox[1], badge_bbox[1]),
        max(board_bbox[2], badge_bbox[2]),
        max(board_bbox[3], badge_bbox[3]),
    )
    _group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=group_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left = float(board_left + dx)
    board_top = float(board_top + dy)
    badge_bbox = (badge_bbox[0] + dx, badge_bbox[1] + dy, badge_bbox[2] + dx, badge_bbox[3] + dy)
    board_bbox = (board_bbox[0] + dx, board_bbox[1] + dy, board_bbox[2] + dx, board_bbox[3] + dy)

    scene_panel_bbox: tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(18, int(round(float(params.panel_margin_px) * 0.42)))
        scene_panel_bbox = (
            max(4, int(round(min(board_bbox[0], badge_bbox[0]))) - panel_pad),
            max(4, int(round(min(board_bbox[1], badge_bbox[1]))) - panel_pad),
            min(int(params.canvas_width) - 4, int(round(max(board_bbox[2], badge_bbox[2]))) + panel_pad),
            min(int(params.canvas_height) - 4, int(round(max(board_bbox[3], badge_bbox[3]))) + panel_pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=scene_panel_bbox,
            style=panel_style,
            radius=26,
            border_width=max(2, int(round(float(params.board_frame_width_px) * 0.55))),
        )

    draw.rounded_rectangle(board_bbox, radius=int(params.board_corner_radius_px), fill=tuple(theme.board_frame_rgb))
    draw.rounded_rectangle(
        badge_bbox,
        radius=int(0.5 * int(params.rule_badge_height_px)),
        fill=tuple(theme.badge_fill_rgb),
        outline=tuple(theme.badge_outline_rgb),
        width=2,
    )
    badge_font = fit_font_to_box(
        draw,
        text=str(badge_text),
        max_width=float((badge_bbox[2] - badge_bbox[0]) - 36),
        max_height=float((badge_bbox[3] - badge_bbox[1]) - 12),
        bold=True,
        font_family=str(params.font_family) or None,
        min_size_px=12,
        max_size_px=int(params.rule_badge_font_size_px),
        fill_ratio=0.98,
    )
    text_bbox = draw.textbbox((0, 0), str(badge_text), font=badge_font, stroke_width=1)
    text_rgb = tuple(theme.badge_text_rgb)
    draw_game_text_traced(
        draw,
        (
            float(badge_bbox[0] + ((badge_bbox[2] - badge_bbox[0]) - (text_bbox[2] - text_bbox[0])) / 2.0),
            float(badge_bbox[1] + ((badge_bbox[3] - badge_bbox[1]) - (text_bbox[3] - text_bbox[1])) / 2.0 - text_bbox[1]),
        ),
        str(badge_text),
        font=badge_font,
        fill=text_rgb,
        stroke_width=1,
        stroke_fill=tuple(resolve_text_stroke_fill(text_rgb)),
        role="readout",
        required=False,
    )

    inner_inset = float(params.board_frame_width_px)
    cell_bboxes: dict[str, tuple[float, float, float, float]] = {}
    piece_bboxes: dict[str, tuple[float, float, float, float]] = {}
    entities: list[dict[str, Any]] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            left = float(board_left + (col * cell_size) + inner_inset)
            top = float(board_top + (row * cell_size) + inner_inset)
            right = float(board_left + ((col + 1) * cell_size) - inner_inset)
            bottom = float(board_top + ((row + 1) * cell_size) - inner_inset)
            cell_bbox = (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))
            is_light = (row + col) % 2 == 0
            fill_rgb = tuple(theme.light_square_rgb if is_light else theme.dark_square_rgb)
            draw.rectangle(cell_bbox, fill=fill_rgb, outline=tuple(theme.grid_line_rgb), width=int(theme.grid_line_width_px))
            if str(theme.square_rendering) == "inset":
                inset = max(2.0, 0.045 * min(cell_bbox[2] - cell_bbox[0], cell_bbox[3] - cell_bbox[1]))
                draw.rectangle(
                    [cell_bbox[0] + inset, cell_bbox[1] + inset, cell_bbox[2] - inset, cell_bbox[3] - inset],
                    outline=tuple(theme.grid_line_rgb),
                    width=1,
                )
            cell_id = coord_to_cell_id((row, col))
            cell_bboxes[cell_id] = cell_bbox
            occupant = board[row][col]
            occupant_text = "empty" if occupant is None else f"{occupant.color}_{occupant.kind}"
            entities.append(
                {
                    "id": cell_id,
                    "type": "chess_variant_cell",
                    "row": row,
                    "col": col,
                    "occupant": occupant_text,
                    "bbox_px": list(cell_bbox),
                }
            )
            if occupant is None:
                continue
            occupant_bbox = piece_bbox(cell_bbox, inset_fraction=float(params.piece_inset_fraction))
            draw_chess_piece_symbol(
                draw,
                bbox_px=occupant_bbox,
                piece=occupant,
                theme=theme,
                font_size_px=int(params.piece_font_size_px),
            )
            piece_id = piece_to_entity_id((row, col), occupant)
            piece_bboxes[piece_id] = occupant_bbox
            entities.append(
                {
                    "id": piece_id,
                    "type": "chess_variant_piece",
                    "color": occupant.color,
                    "kind": occupant.kind,
                    "row": row,
                    "col": col,
                    "bbox_px": list(occupant_bbox),
                }
            )
    return image, {
        "board_bbox_px": [round(float(v), 3) for v in board_bbox],
        "scene_panel_bbox_px": None if scene_panel_bbox is None else [int(v) for v in scene_panel_bbox],
        "badge_bbox_px": [round(float(v), 3) for v in badge_bbox],
        "cell_bboxes_px": {str(k): list(v) for k, v in cell_bboxes.items()},
        "piece_bboxes_px": {str(k): list(v) for k, v in piece_bboxes.items()},
        "effective_cell_size_px": float(cell_size),
        "layout_jitter": dict(layout_jitter),
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "font_family": str(params.font_family),
        "board_size": int(BOARD_SIZE),
        "piece_rendering": "filled_chess_glyph",
    }, tuple(entities)


def draw_marked_outline(
    image: Image.Image,
    render_map: Mapping[str, Any],
    marked_coord: Coord,
    params: ChessVariantRenderParams,
    *,
    outline_rgb: tuple[int, int, int],
) -> None:
    """Draw the visible marked-cell outline."""

    draw = ImageDraw.Draw(image)
    cell_id = coord_to_cell_id(marked_coord)
    bbox = render_map["cell_bboxes_px"][cell_id]
    inset = max(3.0, 0.06 * min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])))
    draw.rectangle(
        [float(bbox[0]) + inset, float(bbox[1]) + inset, float(bbox[2]) - inset, float(bbox[3]) - inset],
        fill=None,
    )
    draw.rectangle(
        [float(bbox[0]) + inset, float(bbox[1]) + inset, float(bbox[2]) - inset, float(bbox[3]) - inset],
        outline=tuple(int(v) for v in outline_rgb),
        width=int(params.marked_square_outline_width_px),
    )


def text_style_metadata(font_family: str) -> dict[str, Any]:
    """Return font metadata for trace payloads."""

    return {
        "font_family": str(font_family),
        "font_asset": get_font_family_record(str(font_family)).to_trace(),
    }


def resolve_scene_background(
    *,
    params: Mapping[str, Any],
    render_params: ChessVariantRenderParams,
    instance_seed: int,
) -> tuple[Image.Image, dict[str, Any], GamePanelSceneStyle, dict[str, Any]]:
    """Resolve the panel treatment and canvas background for the scene."""

    allowed_panel_treatments_raw = params.get(
        "panel_scene_treatments",
        group_default(_RENDER_DEFAULTS, "panel_scene_treatments", None),
    )
    if isinstance(allowed_panel_treatments_raw, str):
        allowed_panel_treatments = (str(allowed_panel_treatments_raw),)
    elif allowed_panel_treatments_raw is None:
        allowed_panel_treatments = None
    else:
        allowed_panel_treatments = tuple(str(item) for item in allowed_panel_treatments_raw)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.chess_variant.panel_scene_style",
        treatments=allowed_panel_treatments,
        treatment_weights=params.get(
            "panel_scene_treatment_weights",
            group_default(_RENDER_DEFAULTS, "panel_scene_treatment_weights", None),
        ),
        palette_weights=params.get(
            "panel_scene_palette_weights",
            group_default(_RENDER_DEFAULTS, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    return background, background_meta, panel_style, panel_style_meta


__all__ = [
    "draw_marked_outline",
    "render_chess_variant_scene",
    "resolve_chess_variant_render_params",
    "resolve_scene_background",
    "rule_badge_text",
    "text_style_metadata",
]
