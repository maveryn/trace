"""Rendering helpers for the 2048 games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_rounded_rect
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box
from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.option_layout import option_grid_position, option_grid_size
from trace_tasks.tasks.games.shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_centered_game_text as draw_centered_text
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from .defaults import FALLBACK_RENDERING_DEFAULTS
from .sampling import Resolved2048Axes
from .state import (
    Board,
    EMPTY,
    SCENE_ID,
    SIZE,
    Sample2048,
    coord_to_cell_id,
)


@dataclass(frozen=True)
class TwentyFortyEightRenderParams:
    """Resolved render controls for one 2048 board."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    board_size_px: int
    board_radius_px: int
    cell_gap_px: int
    cell_radius_px: int
    tile_font_size_px: int
    arrow_width_px: int
    label_font_size_px: int
    font_family: str = ""
    instance_seed: int = 0
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class TwentyFortyEightTheme:
    """Resolved palette for one 2048 visual style."""

    board_fill_rgb: Tuple[int, int, int]
    board_outline_rgb: Tuple[int, int, int]
    empty_cell_rgb: Tuple[int, int, int]
    tile_text_rgb_dark: Tuple[int, int, int]
    tile_text_rgb_light: Tuple[int, int, int]
    arrow_rgb: Tuple[int, int, int]
    arrow_label_fill_rgb: Tuple[int, int, int]
    arrow_label_text_rgb: Tuple[int, int, int]
    tile_palette: Mapping[int, Tuple[int, int, int]]


@dataclass(frozen=True)
class Rendered2048Scene:
    """Rendered 2048 image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class Rendered2048TaskContext:
    """Rendered 2048 scene context shared by objective-owned task files."""

    image: Image.Image
    rendered_scene: Rendered2048Scene
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def resolve_2048_render_params(params: Mapping[str, Any], *, instance_seed: int) -> TwentyFortyEightRenderParams:
    """Resolve 2048 rendering parameters from config/defaults."""

    fallback = FALLBACK_RENDERING_DEFAULTS
    font_exclude_raw = params.get(
        "text_font_exclude_tags",
        group_default(_RENDER_DEFAULTS, "text_font_exclude_tags", fallback["text_font_exclude_tags"]),
    )
    font_exclude_tags = (
        (str(font_exclude_raw),)
        if isinstance(font_exclude_raw, str)
        else tuple(str(item) for item in (font_exclude_raw or ()))
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.2048.text_font",
        params=params,
        exclude_tags=font_exclude_tags,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.2048.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.2048.layout",
        ),
        unit_scale_meta,
    )
    board_size_px = scale_games_px(
        params.get("board_size_px", group_default(_RENDER_DEFAULTS, "board_size_px", fallback["board_size_px"])),
        unit_scale,
        min_px=280,
    )
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(_RENDER_DEFAULTS, "dynamic_canvas_size_enabled", fallback["dynamic_canvas_size_enabled"]),
        )
    )
    canvas_min_size_px = int(
        params.get(
            "canvas_min_size_px",
            group_default(_RENDER_DEFAULTS, "canvas_min_size_px", fallback["canvas_min_size_px"]),
        )
    )
    canvas_side_padding_px = int(
        params.get(
            "canvas_side_padding_px",
            group_default(_RENDER_DEFAULTS, "canvas_side_padding_px", fallback["canvas_side_padding_px"]),
        )
    )
    canvas_side_padding_fraction = float(
        params.get(
            "canvas_side_padding_fraction",
            group_default(_RENDER_DEFAULTS, "canvas_side_padding_fraction", fallback["canvas_side_padding_fraction"]),
        )
    )
    dynamic_canvas_size = max(
        int(canvas_min_size_px),
        int(round(float(board_size_px) + (2.0 * max(float(canvas_side_padding_px), float(board_size_px) * float(canvas_side_padding_fraction))))),
    )
    base_canvas_width = int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", fallback["canvas_width"])))
    base_canvas_height = int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", fallback["canvas_height"])))
    canvas_width = (
        int(base_canvas_width)
        if params.get("canvas_width") is not None or not dynamic_canvas_enabled
        else min(int(base_canvas_width), int(dynamic_canvas_size))
    )
    canvas_height = (
        int(base_canvas_height)
        if params.get("canvas_height") is not None or not dynamic_canvas_enabled
        else min(int(base_canvas_height), int(dynamic_canvas_size))
    )
    return TwentyFortyEightRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", fallback["panel_margin_px"]))),
        board_size_px=int(board_size_px),
        board_radius_px=scale_games_px(params.get("board_radius_px", group_default(_RENDER_DEFAULTS, "board_radius_px", fallback["board_radius_px"])), unit_scale, min_px=8),
        cell_gap_px=scale_games_px(params.get("cell_gap_px", group_default(_RENDER_DEFAULTS, "cell_gap_px", fallback["cell_gap_px"])), unit_scale, min_px=6),
        cell_radius_px=scale_games_px(params.get("cell_radius_px", group_default(_RENDER_DEFAULTS, "cell_radius_px", fallback["cell_radius_px"])), unit_scale, min_px=5),
        tile_font_size_px=scale_games_px(params.get("tile_font_size_px", group_default(_RENDER_DEFAULTS, "tile_font_size_px", fallback["tile_font_size_px"])), unit_scale, min_px=22),
        arrow_width_px=scale_games_px(params.get("arrow_width_px", group_default(_RENDER_DEFAULTS, "arrow_width_px", fallback["arrow_width_px"])), unit_scale, min_px=4),
        label_font_size_px=scale_games_px(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", fallback["label_font_size_px"])), unit_scale, min_px=18),
        font_family=str(font_family),
        instance_seed=int(instance_seed),
        layout_jitter_meta=layout_jitter,
    )


def build_2048_theme(*, style_variant: str) -> TwentyFortyEightTheme:
    """Return a 2048 palette for one style variant."""

    palettes: Dict[str, TwentyFortyEightTheme] = {
        "classic": TwentyFortyEightTheme(
            board_fill_rgb=(179, 165, 148),
            board_outline_rgb=(117, 103, 89),
            empty_cell_rgb=(205, 193, 180),
            tile_text_rgb_dark=(116, 102, 88),
            tile_text_rgb_light=(250, 246, 238),
            arrow_rgb=(63, 91, 125),
            arrow_label_fill_rgb=(246, 243, 232),
            arrow_label_text_rgb=(41, 48, 58),
            tile_palette={
                2: (238, 228, 218),
                4: (237, 224, 200),
                8: (242, 177, 121),
                16: (245, 149, 99),
                32: (246, 124, 95),
                64: (246, 94, 59),
                128: (237, 207, 114),
                256: (237, 204, 97),
                512: (237, 200, 80),
            },
        ),
        "dark": TwentyFortyEightTheme(
            board_fill_rgb=(38, 45, 54),
            board_outline_rgb=(18, 22, 28),
            empty_cell_rgb=(61, 70, 82),
            tile_text_rgb_dark=(30, 35, 44),
            tile_text_rgb_light=(247, 250, 252),
            arrow_rgb=(96, 196, 181),
            arrow_label_fill_rgb=(21, 27, 36),
            arrow_label_text_rgb=(238, 246, 248),
            tile_palette={
                2: (184, 205, 214),
                4: (149, 188, 202),
                8: (99, 168, 198),
                16: (72, 143, 194),
                32: (118, 122, 211),
                64: (147, 92, 198),
                128: (207, 93, 167),
                256: (225, 91, 117),
                512: (238, 137, 82),
            },
        ),
        "paper": TwentyFortyEightTheme(
            board_fill_rgb=(214, 204, 183),
            board_outline_rgb=(91, 84, 69),
            empty_cell_rgb=(239, 232, 214),
            tile_text_rgb_dark=(74, 67, 56),
            tile_text_rgb_light=(255, 252, 242),
            arrow_rgb=(106, 86, 63),
            arrow_label_fill_rgb=(255, 248, 228),
            arrow_label_text_rgb=(73, 58, 42),
            tile_palette={
                2: (246, 236, 209),
                4: (232, 218, 178),
                8: (222, 185, 111),
                16: (209, 151, 84),
                32: (196, 114, 81),
                64: (170, 85, 78),
                128: (143, 116, 73),
                256: (111, 119, 91),
                512: (83, 114, 113),
            },
        ),
        "neon": TwentyFortyEightTheme(
            board_fill_rgb=(22, 24, 55),
            board_outline_rgb=(83, 95, 173),
            empty_cell_rgb=(40, 43, 88),
            tile_text_rgb_dark=(14, 20, 35),
            tile_text_rgb_light=(245, 248, 255),
            arrow_rgb=(255, 91, 155),
            arrow_label_fill_rgb=(30, 31, 68),
            arrow_label_text_rgb=(249, 250, 255),
            tile_palette={
                2: (124, 231, 213),
                4: (80, 202, 237),
                8: (87, 148, 245),
                16: (130, 105, 244),
                32: (183, 89, 229),
                64: (237, 82, 177),
                128: (255, 109, 118),
                256: (255, 163, 86),
                512: (246, 222, 91),
            },
        ),
        "pastel": TwentyFortyEightTheme(
            board_fill_rgb=(174, 185, 198),
            board_outline_rgb=(91, 103, 121),
            empty_cell_rgb=(226, 231, 235),
            tile_text_rgb_dark=(61, 73, 88),
            tile_text_rgb_light=(255, 255, 255),
            arrow_rgb=(82, 112, 166),
            arrow_label_fill_rgb=(249, 252, 255),
            arrow_label_text_rgb=(48, 58, 77),
            tile_palette={
                2: (232, 220, 239),
                4: (215, 229, 249),
                8: (198, 233, 226),
                16: (232, 235, 190),
                32: (244, 213, 178),
                64: (241, 188, 179),
                128: (208, 190, 232),
                256: (175, 206, 235),
                512: (154, 203, 191),
            },
        ),
    }
    return palettes.get(str(style_variant), palettes["classic"])


def _cell_bbox(
    *,
    board_left: float,
    board_top: float,
    cell_size: float,
    gap_px: float,
    row: int,
    col: int,
) -> Tuple[float, float, float, float]:
    """Return the bbox for one 2048 cell."""

    left = float(board_left + gap_px + (int(col) * (cell_size + gap_px)))
    top = float(board_top + gap_px + (int(row) * (cell_size + gap_px)))
    return (
        round(left, 3),
        round(top, 3),
        round(float(left + cell_size), 3),
        round(float(top + cell_size), 3),
    )


def _tile_fill(value: int, theme: TwentyFortyEightTheme) -> Tuple[int, int, int]:
    """Return a deterministic tile fill for one value."""

    if int(value) == EMPTY:
        return tuple(int(v) for v in theme.empty_cell_rgb)
    palette = dict(theme.tile_palette)
    if int(value) in palette:
        return tuple(int(v) for v in palette[int(value)])
    return tuple(int(v) for v in palette[max(palette)])


def _draw_tile_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    value: int,
    theme: TwentyFortyEightTheme,
    max_font_size_px: int,
    font_family: str,
) -> None:
    """Draw one centered 2048 tile value."""

    if int(value) == EMPTY:
        return
    left, top, right, bottom = [float(v) for v in bbox]
    text = str(int(value))
    font = fit_font_to_box(
        draw,
        text=text,
        max_width=float(right - left),
        max_height=float(bottom - top),
        bold=True,
        min_size_px=18,
        max_size_px=int(max_font_size_px),
        fill_ratio=0.72,
        font_family=str(font_family) or None,
    )
    fill = theme.tile_text_rgb_dark if int(value) <= 4 else theme.tile_text_rgb_light
    tile_fill = _tile_fill(int(value), theme)
    draw_centered_text(
        draw,
        text=text,
        center=(float((left + right) / 2.0), float((top + bottom) / 2.0)),
        font=font,
        fill=fill,
        stroke_fill=tile_fill,
        stroke_width=0,
    )


def _direction_arrow_points(
    *,
    direction: str,
    board_bbox: Tuple[float, float, float, float],
    canvas_width: int,
    canvas_height: int,
    span_px: float | None = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return start/end points for one prominent move arrow."""

    left, top, right, bottom = [float(v) for v in board_bbox]
    cx = float((left + right) / 2.0)
    cy = float((top + bottom) / 2.0)
    span = float(span_px) if span_px is not None else min(float(canvas_width), float(canvas_height)) * 0.085
    if str(direction) == "up":
        return (cx, float(top - span * 0.15)), (cx, float(top - span * 0.95))
    if str(direction) == "down":
        return (cx, float(bottom + span * 0.15)), (cx, float(bottom + span * 0.95))
    if str(direction) == "left":
        return (float(left - span * 0.15), cy), (float(left - span * 0.95), cy)
    if str(direction) == "right":
        return (float(right + span * 0.15), cy), (float(right + span * 0.95), cy)
    raise ValueError(f"unsupported direction: {direction!r}")


def _draw_board_grid_at(
    draw: ImageDraw.ImageDraw,
    *,
    board: Board,
    board_bbox: Tuple[float, float, float, float],
    params: TwentyFortyEightRenderParams,
    theme: TwentyFortyEightTheme,
    entity_prefix: str,
    entity_type: str,
) -> Tuple[Dict[str, Tuple[float, float, float, float]], Tuple[Dict[str, Any], ...]]:
    """Draw one 2048 board into a fixed bbox and return cell geometry."""

    left, top, right, bottom = [float(v) for v in board_bbox]
    board_size = max(1.0, min(float(right - left), float(bottom - top)))
    nominal_size = max(1.0, float(params.board_size_px))
    scale = float(board_size / nominal_size)
    board_radius = max(6, int(round(float(params.board_radius_px) * scale)))
    cell_radius = max(4, int(round(float(params.cell_radius_px) * scale)))
    gap = max(4.0, float(params.cell_gap_px) * scale)
    tile_font_size = max(13, int(round(float(params.tile_font_size_px) * scale)))

    draw_rounded_rect(
        draw,
        (left, top, left + board_size, top + board_size),
        radius=board_radius,
        fill=theme.board_fill_rgb,
        outline=theme.board_outline_rgb,
        width=max(2, int(round(4.0 * scale))),
    )
    cell_size = float((board_size - ((SIZE + 1) * gap)) / float(SIZE))
    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    for row in range(SIZE):
        for col in range(SIZE):
            value = int(board[row][col])
            cell_id = f"{entity_prefix}_{coord_to_cell_id((row, col))}"
            bbox = _cell_bbox(
                board_left=left,
                board_top=top,
                cell_size=cell_size,
                gap_px=gap,
                row=row,
                col=col,
            )
            entity_bboxes[cell_id] = bbox
            fill = _tile_fill(value, theme)
            draw_rounded_rect(
                draw,
                bbox,
                radius=cell_radius,
                fill=fill,
                outline=theme.board_outline_rgb if value == EMPTY else fill,
                width=max(1, int(round(2.0 * scale))),
            )
            _draw_tile_text(
                draw,
                bbox=bbox,
                value=value,
                theme=theme,
                max_font_size_px=tile_font_size,
                font_family=str(params.font_family),
            )
            scene_entities.append(
                {
                    "id": cell_id,
                    "type": str(entity_type),
                    "row": int(row),
                    "col": int(col),
                    "value": int(value),
                    "bbox": list(bbox),
                }
            )
    return entity_bboxes, tuple(scene_entities)


def _draw_result_option_label(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    center: Tuple[float, float],
    radius: float,
    params: TwentyFortyEightRenderParams,
    theme: TwentyFortyEightTheme,
) -> Tuple[float, float, float, float]:
    """Draw one high-contrast result-board option label."""

    cx, cy = float(center[0]), float(center[1])
    bbox = (
        round(cx - float(radius), 3),
        round(cy - float(radius), 3),
        round(cx + float(radius), 3),
        round(cy + float(radius), 3),
    )
    badge_fill = tuple(int(v) for v in theme.arrow_rgb)
    badge_outline = tuple(int(v) for v in theme.arrow_label_fill_rgb)
    draw.ellipse(bbox, fill=badge_fill, outline=badge_outline, width=4)
    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=float(2.0 * radius),
        max_height=float(2.0 * radius),
        bold=True,
        min_size_px=14,
        max_size_px=max(18, int(params.label_font_size_px)),
        fill_ratio=0.72,
        font_family=str(params.font_family) or None,
    )
    draw_centered_text(
        draw,
        text=str(label),
        center=(cx, cy),
        font=font,
        fill=theme.arrow_label_text_rgb,
        stroke_fill=badge_fill,
        stroke_width=1,
        role="option_label",
        required=True,
        surface_rgbs=(badge_fill,),
        preferred_rgbs=(theme.arrow_label_text_rgb,),
        instance_seed=int(params.instance_seed),
        namespace="games.2048.result_board_option_label",
    )
    return bbox


def render_2048_board_scene(
    *,
    board: Board,
    background: Image.Image,
    style_variant: str,
    params: TwentyFortyEightRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
    move_direction: str | None = None,
) -> Rendered2048Scene:
    """Render one 2048 board with an optional move arrow."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_2048_theme(style_variant=str(style_variant))

    board_size = min(
        int(params.board_size_px),
        int(params.canvas_width) - (2 * int(params.panel_margin_px)),
        int(params.canvas_height) - (2 * int(params.panel_margin_px)),
    )
    board_left = float((int(params.canvas_width) - int(board_size)) / 2.0)
    board_top = float((int(params.canvas_height) - int(board_size)) / 2.0)
    board_bbox = (
        round(board_left, 3),
        round(board_top, 3),
        round(float(board_left + board_size), 3),
        round(float(board_top + board_size), 3),
    )
    board_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left, board_top, board_right, board_bottom = [float(v) for v in board_bbox]

    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(14, int(round(float(params.cell_gap_px) * 1.5)))
        panel_bbox = (
            max(4, int(round(board_left)) - panel_pad),
            max(4, int(round(board_top)) - panel_pad),
            min(int(params.canvas_width) - 4, int(round(board_right)) + panel_pad),
            min(int(params.canvas_height) - 4, int(round(board_bottom)) + panel_pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=max(14, int(params.board_radius_px) + 10),
            border_width=max(2, int(round(float(params.cell_gap_px) * 0.18))),
        )

    draw_rounded_rect(
        draw,
        board_bbox,
        radius=int(params.board_radius_px),
        fill=theme.board_fill_rgb,
        outline=theme.board_outline_rgb,
        width=4,
    )

    gap = float(params.cell_gap_px)
    cell_size = float((board_right - board_left - ((SIZE + 1) * gap)) / float(SIZE))
    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    for row in range(SIZE):
        for col in range(SIZE):
            value = int(board[row][col])
            cell_id = coord_to_cell_id((row, col))
            bbox = _cell_bbox(
                board_left=board_left,
                board_top=board_top,
                cell_size=cell_size,
                gap_px=gap,
                row=row,
                col=col,
            )
            entity_bboxes[cell_id] = bbox
            fill = _tile_fill(value, theme)
            draw_rounded_rect(
                draw,
                bbox,
                radius=int(params.cell_radius_px),
                fill=fill,
                outline=theme.board_outline_rgb if value == EMPTY else fill,
                width=2,
            )
            _draw_tile_text(
                draw,
                bbox=bbox,
                value=value,
                theme=theme,
                max_font_size_px=int(params.tile_font_size_px),
                font_family=str(params.font_family),
            )
            scene_entities.append(
                {
                    "id": cell_id,
                    "type": "2048_cell",
                    "row": int(row),
                    "col": int(col),
                    "value": int(value),
                    "bbox": list(bbox),
                }
            )

    arrow_entities: list[Dict[str, Any]] = []
    if move_direction is not None:
        start, end = _direction_arrow_points(
            direction=str(move_direction),
            board_bbox=board_bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
        )
        draw_arrow(
            draw,
            start=start,
            end=end,
            fill=theme.arrow_rgb,
            width=int(params.arrow_width_px),
            head_length_px=26,
            head_width_px=31,
        )
        arrow_entities.append(
            {
                "id": "shown_move",
                "type": "2048_move_arrow",
                "direction": str(move_direction),
                "bbox": [
                    round(min(start[0], end[0]) - 20.0, 3),
                    round(min(start[1], end[1]) - 20.0, 3),
                    round(max(start[0], end[0]) + 20.0, 3),
                    round(max(start[1], end[1]) + 20.0, 3),
                ],
            }
        )

    scene_entities.extend(arrow_entities)
    render_map: Dict[str, Any] = {
        "entity_bboxes_px": {key: list(value) for key, value in entity_bboxes.items()},
        "board_bbox_px": list(board_bbox),
        "panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
        "layout_jitter": dict(layout_jitter),
        "style_variant": str(style_variant),
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "effective_cell_size_px": round(float(cell_size), 3),
        "font_family": str(params.font_family),
    }
    return Rendered2048Scene(
        image=image,
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


def render_2048_result_options_scene(
    *,
    board: Board,
    option_boards: Mapping[str, Board],
    background: Image.Image,
    style_variant: str,
    params: TwentyFortyEightRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
    move_direction: str,
) -> Rendered2048Scene:
    """Render one source 2048 board plus labeled candidate result boards."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_2048_theme(style_variant=str(style_variant))
    canvas_width = int(params.canvas_width)
    canvas_height = int(params.canvas_height)

    board_size = float(min(210, max(188, int(round(min(canvas_width, canvas_height) * 0.23)))))
    source_size = float(board_size)
    option_size = float(board_size)
    arrow_span = float(min(54.0, max(42.0, float(source_size) * 0.25)))
    arrow_clearance = float(max(16.0, float(params.arrow_width_px) * 1.8))
    option_label_band = float(max(34, int(round(float(option_size) * 0.19))))
    option_panel_height = float(option_size + option_label_band)
    option_gap_x = float(max(20, int(round(float(option_size) * 0.10))))
    option_gap_y = float(max(22, int(round(float(option_size) * 0.11))))
    source_to_options_gap = float(max(86.0, (arrow_span * 0.95) + arrow_clearance + 18.0, float(source_size) * 0.40))
    arrow_pad = float(max(56.0, (arrow_span * 1.05) + arrow_clearance))
    options_width, options_height = option_grid_size(
        len(option_boards),
        item_width=option_size,
        item_height=option_panel_height,
        gap_x=option_gap_x,
        gap_y=option_gap_y,
    )
    content_width = float(max(source_size + (2.0 * arrow_pad), options_width))
    content_height = float(arrow_pad + source_size + source_to_options_gap + options_height)
    group_left = float((canvas_width - content_width) / 2.0)
    group_top = float((canvas_height - content_height) / 2.0)
    group_bbox = (
        round(group_left, 3),
        round(group_top, 3),
        round(group_left + content_width, 3),
        round(group_top + content_height, 3),
    )
    group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=group_bbox,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        jitter=params.layout_jitter_meta,
    )
    group_left, group_top, _group_right, _group_bottom = [float(v) for v in group_bbox]
    source_left = float(group_left + ((content_width - source_size) / 2.0))
    source_top = float(group_top + arrow_pad)
    source_bbox = (
        round(source_left, 3),
        round(source_top, 3),
        round(source_left + source_size, 3),
        round(source_top + source_size, 3),
    )
    options_left = float(group_left + ((content_width - options_width) / 2.0))
    options_top = float(source_top + source_size + source_to_options_gap)

    scene_entities: list[Dict[str, Any]] = []
    source_cell_bboxes, source_entities = _draw_board_grid_at(
        draw,
        board=board,
        board_bbox=source_bbox,
        params=params,
        theme=theme,
        entity_prefix="source",
        entity_type="2048_source_cell",
    )
    scene_entities.extend(source_entities)

    start, end = _direction_arrow_points(
        direction=str(move_direction),
        board_bbox=source_bbox,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        span_px=arrow_span,
    )
    draw_arrow(
        draw,
        start=start,
        end=end,
        fill=theme.arrow_rgb,
        width=int(params.arrow_width_px),
        head_length_px=26,
        head_width_px=31,
    )
    scene_entities.append(
        {
            "id": "shown_move",
            "type": "2048_move_arrow",
            "direction": str(move_direction),
            "bbox": [
                round(min(start[0], end[0]) - 20.0, 3),
                round(min(start[1], end[1]) - 20.0, 3),
                round(max(start[0], end[0]) + 20.0, 3),
                round(max(start[1], end[1]) + 20.0, 3),
            ],
        }
    )

    option_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    option_panel_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    option_cell_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    label_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for option_index, (label, option_board) in enumerate(option_boards.items()):
        _row, _col, panel_left, panel_top = option_grid_position(
            option_index,
            len(option_boards),
            left=options_left,
            top=options_top,
            item_width=option_size,
            item_height=option_panel_height,
            gap_x=option_gap_x,
            gap_y=option_gap_y,
        )
        panel_bbox = (
            round(panel_left - 8.0, 3),
            round(panel_top - 4.0, 3),
            round(panel_left + option_size + 8.0, 3),
            round(panel_top + option_panel_height + 8.0, 3),
        )
        option_panel_bboxes[str(label)] = panel_bbox
        if panel_style is not None:
            draw_rounded_rect(
                draw,
                panel_bbox,
                radius=14,
                fill=tuple(int(v) for v in panel_style.option_fill_rgb),
                outline=tuple(int(v) for v in panel_style.panel_border_rgb),
                width=2,
            )
        board_top = float(panel_top)
        option_bbox = (
            round(panel_left, 3),
            round(board_top, 3),
            round(panel_left + option_size, 3),
            round(board_top + option_size, 3),
        )
        option_bboxes[str(label)] = option_bbox
        label_bboxes[str(label)] = _draw_result_option_label(
            draw,
            label=str(label),
            center=(
                float(panel_left + (option_size / 2.0)),
                float(panel_top + option_size + (option_label_band / 2.0)),
            ),
            radius=max(15.0, float(option_label_band) * 0.45),
            params=params,
            theme=theme,
        )
        cell_bboxes, cell_entities = _draw_board_grid_at(
            draw,
            board=option_board,
            board_bbox=option_bbox,
            params=params,
            theme=theme,
            entity_prefix=f"result_option_{label}",
            entity_type="2048_result_option_cell",
        )
        option_cell_bboxes.update(cell_bboxes)
        scene_entities.append(
            {
                "id": f"result_option_{label}",
                "type": "2048_result_option_board",
                "label": str(label),
                "bbox": list(option_bbox),
            }
        )
        scene_entities.extend(cell_entities)

    render_map: Dict[str, Any] = {
        "entity_bboxes_px": {
            **{str(key): list(value) for key, value in source_cell_bboxes.items()},
            **{str(key): list(value) for key, value in option_cell_bboxes.items()},
            **{f"result_option_{label}": list(bbox) for label, bbox in option_bboxes.items()},
        },
        "source_board_bbox_px": list(source_bbox),
        "result_option_bboxes_px": {str(key): list(value) for key, value in option_bboxes.items()},
        "result_option_panel_bboxes_px": {str(key): list(value) for key, value in option_panel_bboxes.items()},
        "result_option_label_bboxes_px": {str(key): list(value) for key, value in label_bboxes.items()},
        "layout_jitter": dict(layout_jitter),
        "layout_offset_px": [round(float(dx), 3), round(float(dy), 3)],
        "style_variant": str(style_variant),
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "effective_cell_size_px": round(float((option_size - ((SIZE + 1) * max(4.0, float(params.cell_gap_px) * (option_size / max(1.0, float(params.board_size_px)))))) / float(SIZE)), 3),
        "font_family": str(params.font_family),
    }
    return Rendered2048Scene(
        image=image,
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


def render_2048_sample(
    *,
    axes: Resolved2048Axes,
    sample: Sample2048,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Rendered2048TaskContext:
    """Render one sampled 2048 scene without binding a task answer."""

    render_params = resolve_2048_render_params(params, instance_seed=int(instance_seed))
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
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
        namespace="games.2048_board.panel_scene_style",
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
    if sample.result_option_boards:
        rendered_scene = render_2048_result_options_scene(
            board=sample.board,
            option_boards=sample.result_option_boards,
            background=background,
            style_variant=str(axes.style_variant),
            params=render_params,
            panel_style=panel_style,
            move_direction=str(sample.move_direction),
        )
    else:
        rendered_scene = render_2048_board_scene(
            board=sample.board,
            background=background,
            style_variant=str(axes.style_variant),
            params=render_params,
            panel_style=panel_style,
            move_direction=str(sample.move_direction),
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return Rendered2048TaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "TwentyFortyEightRenderParams",
    "Rendered2048Scene",
    "Rendered2048TaskContext",
    "build_2048_theme",
    "render_2048_board_scene",
    "render_2048_sample",
    "render_2048_result_options_scene",
    "resolve_2048_render_params",
]
