"""Shared Minesweeper-grid renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.games.shared.layout import (
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)

from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_game_text_traced as draw_text_traced
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.marking import draw_semantic_bbox_marker, resolve_semantic_marker_style
from .defaults import DEFAULTS
from .rules import clue_number
from .state import Coord, all_coords, coord_to_cell_id
from ...shared.style import MinesweeperTheme, build_games_minesweeper_theme


@dataclass(frozen=True)
class MinesweeperRenderParams:
    """Resolved render controls for one Minesweeper scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_size_px: int
    board_border_width_px: int
    grid_line_width_px: int
    cell_padding_px: int
    number_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None
    instance_seed: int = 0


@dataclass(frozen=True)
class MinesweeperCellSpec:
    """One rendered Minesweeper cell."""

    cell_id: str
    row: int
    col: int
    state: str
    has_mine: bool
    adjacent_mine_count: int
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedMinesweeperScene:
    """Rendered Minesweeper image plus trace-friendly cell geometry."""

    image: Image.Image
    cell_specs: Tuple[MinesweeperCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def _cell_bbox(
    *,
    board_left: float,
    board_top: float,
    cell_size: float,
    row: int,
    col: int,
    padding_px: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Return the bbox for one Minesweeper cell, with optional inset padding."""

    left = float(board_left + (int(col) * float(cell_size)) + float(padding_px))
    top = float(board_top + (int(row) * float(cell_size)) + float(padding_px))
    right = float(board_left + ((int(col) + 1) * float(cell_size)) - float(padding_px))
    bottom = float(board_top + ((int(row) + 1) * float(cell_size)) - float(padding_px))
    return (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))


def _mix_rgb(rgb: Tuple[int, int, int], target_rgb: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    """Blend one RGB color toward a target color by a fixed amount."""

    weight = max(0.0, min(1.0, float(amount)))
    return tuple(
        int(round((float(channel) * (1.0 - weight)) + (float(target) * weight)))
        for channel, target in zip(rgb, target_rgb)
    )


def _draw_hidden_tile(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: MinesweeperTheme,
    cell_size: float,
) -> None:
    """Draw one covered Minesweeper cell as a raised tile."""

    left, top, right, bottom = [float(value) for value in bbox_px]
    fill_rgb = tuple(int(value) for value in theme.hidden_cell_fill_rgb)
    light_rgb = _mix_rgb(fill_rgb, (255, 255, 255), 0.42)
    dark_rgb = _mix_rgb(fill_rgb, (0, 0, 0), 0.34)
    border_rgb = tuple(int(value) for value in theme.hidden_cell_border_rgb)
    bevel_width = max(2, int(round(0.055 * float(cell_size))))
    inset = max(1.0, 0.055 * float(cell_size))
    inner = (left + inset, top + inset, right - inset, bottom - inset)
    draw.rectangle((left, top, right, bottom), fill=fill_rgb)
    draw.line([(inner[0], inner[3]), (inner[2], inner[3]), (inner[2], inner[1])], fill=dark_rgb, width=bevel_width)
    draw.line([(inner[0], inner[3]), (inner[0], inner[1]), (inner[2], inner[1])], fill=light_rgb, width=bevel_width)
    draw.rectangle(
        (
            inner[0] + (0.5 * bevel_width),
            inner[1] + (0.5 * bevel_width),
            inner[2] - (0.5 * bevel_width),
            inner[3] - (0.5 * bevel_width),
        ),
        outline=border_rgb,
        width=max(1, int(round(0.022 * float(cell_size)))),
    )


def _draw_number(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    value: int,
    theme: MinesweeperTheme,
    font_size_px: int,
    font_family: str = "",
) -> None:
    """Draw one centered Minesweeper clue number."""

    number = int(value)
    if number <= 0:
        return
    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    text = str(number)
    font = fit_font_to_box(
        draw,
        text=text,
        max_width=float(width),
        max_height=float(height),
        bold=True,
        min_size_px=14,
        max_size_px=int(font_size_px),
        fill_ratio=0.72,
        font_family=str(font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    text_x = float(left + (0.5 * (width - text_w)) - float(text_bbox[0]))
    text_y = float(top + (0.5 * (height - text_h)) - float(text_bbox[1]))
    colors = tuple(theme.number_rgb_by_value)
    fill = colors[min(max(int(number), 0), len(colors) - 1)]
    draw_text_traced(draw,(text_x, text_y), text, fill=tuple(int(v) for v in fill), font=font, role="readout", required=False)


def _draw_flag(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: MinesweeperTheme,
) -> None:
    """Draw a compact flag marker inside one hidden cell."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    pole_x = float(left + 0.45 * width)
    pole_top = float(top + 0.24 * height)
    pole_bottom = float(top + 0.74 * height)
    draw.line(
        [(pole_x, pole_top), (pole_x, pole_bottom)],
        fill=tuple(int(v) for v in theme.flag_pole_rgb),
        width=max(2, int(round(0.05 * width))),
    )
    flag = [
        (pole_x, pole_top),
        (float(left + 0.74 * width), float(top + 0.34 * height)),
        (pole_x, float(top + 0.46 * height)),
    ]
    draw.polygon(flag, fill=tuple(int(v) for v in theme.flag_rgb))
    base_y = float(top + 0.77 * height)
    draw.line(
        [(float(left + 0.30 * width), base_y), (float(left + 0.62 * width), base_y)],
        fill=tuple(int(v) for v in theme.flag_pole_rgb),
        width=max(2, int(round(0.05 * width))),
    )


def _draw_option_label(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    label: str,
    cell_size: float,
    font_family: str = "",
) -> list[float]:
    """Draw a readable in-cell option badge and return its center point."""

    left, top, right, bottom = [float(value) for value in bbox_px]
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    radius = 0.30 * min(float(right - left), float(bottom - top), float(cell_size))
    badge_bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(badge_bbox, fill=(255, 249, 226, 242), outline=(172, 45, 45, 255), width=max(2, int(round(0.055 * cell_size))))
    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=float(1.25 * radius),
        max_height=float(1.20 * radius),
        bold=True,
        min_size_px=12,
        max_size_px=max(14, int(round(0.52 * float(cell_size)))),
        fill_ratio=0.95,
        font_family=str(font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), str(label), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    text_x = float(cx - (0.5 * text_w) - float(text_bbox[0]))
    text_y = float(cy - (0.5 * text_h) - float(text_bbox[1]))
    draw_text_traced(draw, (text_x, text_y), str(label), fill=(45, 31, 30), font=font, role="option_label", required=True)
    return [round(float(cx), 3), round(float(cy), 3)]


def resolve_minesweeper_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> MinesweeperRenderParams:
    """Resolve Minesweeper render dimensions, font, jitter, and unit scaling."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.layout",
        ),
        unit_scale_meta,
    )
    max_board_size_px = scale_games_px(
        params.get("max_board_size_px", group_default(render_defaults, "max_board_size_px", DEFAULTS.max_board_size_px)),
        unit_scale,
        min_px=360,
    )
    default_canvas_width = int(group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))
    default_canvas_height = int(group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))
    canvas_size = int(max(500, min(max(default_canvas_width, default_canvas_height), int(max_board_size_px) + 160)))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font_family",
        params=params,
    )
    return MinesweeperRenderParams(
        canvas_width=int(params.get("canvas_width", canvas_size)),
        canvas_height=int(params.get("canvas_height", canvas_size)),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        max_board_size_px=int(max_board_size_px),
        board_border_width_px=scale_games_px(
            params.get("board_border_width_px", group_default(render_defaults, "board_border_width_px", DEFAULTS.board_border_width_px)),
            unit_scale,
            min_px=2,
        ),
        grid_line_width_px=scale_games_px(
            params.get("grid_line_width_px", group_default(render_defaults, "grid_line_width_px", DEFAULTS.grid_line_width_px)),
            unit_scale,
            min_px=1,
        ),
        cell_padding_px=scale_games_px(
            params.get("cell_padding_px", group_default(render_defaults, "cell_padding_px", DEFAULTS.cell_padding_px)),
            unit_scale,
            min_px=3,
        ),
        number_font_size_px=scale_games_px(
            params.get("number_font_size_px", group_default(render_defaults, "number_font_size_px", DEFAULTS.number_font_size_px)),
            unit_scale,
            min_px=18,
        ),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
        instance_seed=int(instance_seed),
    )


def render_minesweeper_grid_scene(
    *,
    size: int,
    mine_coords: Sequence[Coord],
    revealed_coords: Sequence[Coord],
    flagged_coords: Sequence[Coord],
    hidden_coords: Sequence[Coord],
    background: Image.Image,
    style_variant: str,
    params: MinesweeperRenderParams,
    highlighted_clue_coords: Sequence[Coord] | None = None,
    option_label_coords: Sequence[Tuple[str, Coord]] | None = None,
) -> RenderedMinesweeperScene:
    """Render one Minesweeper grid with hidden, flagged, and revealed cells."""

    board_size = int(size)
    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_minesweeper_theme(style_variant=str(style_variant))
    board_canvas_height = int(params.canvas_height)

    max_board_size = min(
        int(params.max_board_size_px),
        int(params.canvas_width) - (2 * int(params.panel_margin_px)),
        int(board_canvas_height) - (2 * int(params.panel_margin_px)),
    )
    board_left = int(0.5 * (int(params.canvas_width) - int(max_board_size)))
    board_top = int(0.5 * (int(board_canvas_height) - int(max_board_size)))
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + max_board_size), 3),
        round(float(board_top + max_board_size), 3),
    )
    board_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(board_canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left = float(board_bbox[0])
    board_top = float(board_bbox[1])
    cell_size = float((float(board_bbox[2]) - float(board_bbox[0])) / float(board_size))

    draw.rectangle(board_bbox, fill=tuple(int(v) for v in theme.board_fill_rgb))

    mines = {(int(row), int(col)) for row, col in mine_coords}
    revealed = {(int(row), int(col)) for row, col in revealed_coords}
    flagged = {(int(row), int(col)) for row, col in flagged_coords}
    hidden = {(int(row), int(col)) for row, col in hidden_coords}
    highlighted_clues = {(int(row), int(col)) for row, col in (highlighted_clue_coords or ())}
    option_labels = tuple((str(label), (int(coord[0]), int(coord[1]))) for label, coord in (option_label_coords or ()))

    cell_bboxes_px: Dict[str, List[float]] = {}
    full_cell_bboxes_px: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    cell_specs: List[MinesweeperCellSpec] = []
    for row, col in all_coords(size=int(board_size)):
        coord = (int(row), int(col))
        cell_id = coord_to_cell_id(coord)
        full_bbox = _cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=int(row),
            col=int(col),
        )
        bbox_px = _cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=int(row),
            col=int(col),
            padding_px=float(params.cell_padding_px),
        )
        cell_bboxes_px[cell_id] = list(bbox_px)
        full_cell_bboxes_px[cell_id] = list(full_bbox)
        if coord in revealed:
            draw.rectangle(full_bbox, fill=tuple(int(v) for v in theme.revealed_cell_fill_rgb))
        else:
            _draw_hidden_tile(draw, bbox_px=full_bbox, theme=theme, cell_size=float(cell_size))
        if coord in flagged:
            _draw_flag(draw, bbox_px=bbox_px, theme=theme)
        elif coord in revealed:
            _draw_number(
                draw,
                bbox_px=bbox_px,
                value=clue_number(coord, mine_coords=mines, size=int(board_size)),
                theme=theme,
                font_size_px=int(params.number_font_size_px),
                font_family=str(params.font_family),
            )
        if coord in revealed:
            state = "revealed"
        elif coord in flagged:
            state = "flagged"
        else:
            state = "hidden"
        adjacent = clue_number(coord, mine_coords=mines, size=int(board_size))
        scene_entities.append(
            {
                "entity_id": str(cell_id),
                "entity_type": "minesweeper_cell",
                "row": int(row),
                "col": int(col),
                "state": str(state),
                "has_mine": bool(coord in mines),
                "adjacent_mine_count": int(adjacent),
                "is_highlighted_clue": bool(coord in highlighted_clues),
                "bbox_px": list(bbox_px),
            }
        )
        cell_specs.append(
            MinesweeperCellSpec(
                cell_id=str(cell_id),
                row=int(row),
                col=int(col),
                state=str(state),
                has_mine=bool(coord in mines),
                adjacent_mine_count=int(adjacent),
                bbox_px=bbox_px,
            )
        )

    for index in range(board_size + 1):
        x = float(board_left + (index * cell_size))
        y = float(board_top + (index * cell_size))
        draw.line(
            [(x, board_top), (x, float(board_bbox[3]))],
            fill=tuple(int(v) for v in theme.grid_line_rgb),
            width=int(params.grid_line_width_px),
        )
        draw.line(
            [(board_left, y), (float(board_bbox[2]), y)],
            fill=tuple(int(v) for v in theme.grid_line_rgb),
            width=int(params.grid_line_width_px),
        )
    for coord in sorted(highlighted_clues):
        cell_id = coord_to_cell_id(coord)
        if cell_id not in full_cell_bboxes_px:
            continue
        left, top, right, bottom = [float(value) for value in full_cell_bboxes_px[cell_id]]
        inset = max(2.0, 0.055 * float(cell_size))
        highlight_bbox = (left + inset, top + inset, right - inset, bottom - inset)
        width = max(4, int(round(0.075 * float(cell_size))))
        coord_surface_rgb = theme.revealed_cell_fill_rgb
        marker_style = resolve_semantic_marker_style(
            instance_seed=int(params.instance_seed),
            namespace=f"games.minesweeper.highlighted_clue.{cell_id}",
            role="minesweeper_highlighted_clue",
            surface_rgbs=(tuple(int(v) for v in coord_surface_rgb),),
            preferred_rgbs=((255, 212, 74),),
        )
        draw_semantic_bbox_marker(
            draw,
            highlight_bbox,
            style=marker_style,
            width=width,
            marker_kind="minesweeper_highlighted_clue_outline",
            extra_metadata={"cell_id": str(cell_id)},
            )

    option_label_points_px: Dict[str, List[float]] = {}
    option_label_cell_ids: Dict[str, str] = {}
    for label, coord in option_labels:
        cell_id = coord_to_cell_id(coord)
        if cell_id not in full_cell_bboxes_px:
            continue
        point = _draw_option_label(
            draw,
            bbox_px=tuple(float(value) for value in full_cell_bboxes_px[cell_id]),
            label=str(label),
            cell_size=float(cell_size),
            font_family=str(params.font_family),
        )
        option_label_points_px[str(label)] = list(point)
        option_label_cell_ids[str(label)] = str(cell_id)
        scene_entities.append(
            {
                "entity_id": f"option_{str(label)}",
                "entity_type": "minesweeper_option_label",
                "label": str(label),
                "cell_id": str(cell_id),
                "row": int(coord[0]),
                "col": int(coord[1]),
                "point_px": list(point),
                "bbox_px": list(full_cell_bboxes_px[cell_id]),
            }
        )
    draw.rectangle(
        board_bbox,
        outline=tuple(int(v) for v in theme.board_border_rgb),
        width=int(params.board_border_width_px),
    )

    render_map = {
        "board_bbox_px": list(board_bbox),
        "cell_bboxes_px": dict(cell_bboxes_px),
        "revealed_cell_ids": [coord_to_cell_id(coord) for coord in sorted(revealed)],
        "flagged_cell_ids": [coord_to_cell_id(coord) for coord in sorted(flagged)],
        "hidden_cell_ids": [coord_to_cell_id(coord) for coord in sorted(hidden)],
        "highlighted_clue_cell_ids": [coord_to_cell_id(coord) for coord in sorted(highlighted_clues)],
        "option_label_points_px": dict(option_label_points_px),
        "option_label_cell_ids": dict(option_label_cell_ids),
        "style_variant": str(style_variant),
        "text_style": {"font_family": str(params.font_family)},
        "layout_jitter": dict(layout_jitter),
    }
    return RenderedMinesweeperScene(
        image=image,
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "MinesweeperCellSpec",
    "MinesweeperRenderParams",
    "RenderedMinesweeperScene",
    "render_minesweeper_grid_scene",
    "resolve_minesweeper_render_params",
]
