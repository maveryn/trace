"""Shared Connect Four board renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from ....shared.text_rendering import fit_font_to_box, load_font, resolve_text_stroke_fill
from ...shared.text import draw_game_text_traced as draw_text_traced
from .rules import RED, YELLOW, Coord, board_dimensions, coord_to_cell_id, player_name
from ...shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    offset_bbox,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from ...shared.style import ConnectFourTheme, build_games_connect_four_theme
from ...shared.visual_defaults import load_games_scene_noise_defaults
from .defaults import FALLBACK_RENDERING_DEFAULTS, SCENE_ID
from .state import ConnectFourColumnProfileSample, ConnectFourCountSample, ConnectFourLabelSample


@dataclass(frozen=True)
class ConnectFourRenderParams:
    """Resolved render controls for one Connect Four scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    player_badge_height_px: int
    player_badge_width_px: int
    header_gap_px: int
    max_board_width_px: int
    board_corner_radius_px: int
    board_frame_width_px: int
    disc_inset_fraction: float
    player_badge_font_size_px: int
    marked_square_outline_width_px: int
    layout_jitter_meta: Dict[str, Any] | None = None
    font_family: str = ""


@dataclass(frozen=True)
class ConnectFourCellSpec:
    """One board cell after layout/render assignment."""

    cell_id: str
    row: int
    col: int
    occupant: str
    bbox_px: Tuple[float, float, float, float]
    disc_bbox_px: Tuple[float, float, float, float] | None


@dataclass(frozen=True)
class RenderedConnectFourScene:
    """Rendered Connect Four scene plus trace-friendly metadata."""

    image: Image.Image
    cell_specs: Tuple[ConnectFourCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedConnectFourTaskContext:
    """Rendered Connect Four scene context shared by objective-owned tasks."""

    image: Image.Image
    rendered_scene: RenderedConnectFourScene
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def resolve_connect_four_render_params(params: Mapping[str, Any], *, instance_seed: int) -> ConnectFourRenderParams:
    """Resolve Connect Four rendering parameters from scene config/defaults."""

    fallback = FALLBACK_RENDERING_DEFAULTS
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.connect_four.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.connect_four.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.connect_four.layout",
        ),
        unit_scale_meta,
    )
    max_board_width_px = scale_games_px(
        params.get(
            "max_board_width_px",
            group_default(_RENDER_DEFAULTS, "max_board_width_px", fallback["max_board_width_px"]),
        ),
        unit_scale,
        min_px=390,
    )
    player_badge_height_px = int(
        params.get(
            "player_badge_height_px",
            group_default(_RENDER_DEFAULTS, "player_badge_height_px", fallback["player_badge_height_px"]),
        )
    )
    player_badge_width_px = int(
        params.get(
            "player_badge_width_px",
            group_default(_RENDER_DEFAULTS, "player_badge_width_px", fallback["player_badge_width_px"]),
        )
    )
    header_gap_px = int(params.get("header_gap_px", group_default(_RENDER_DEFAULTS, "header_gap_px", fallback["header_gap_px"])))
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(_RENDER_DEFAULTS, "dynamic_canvas_size_enabled", fallback["dynamic_canvas_size_enabled"]),
        )
    )
    base_canvas_width = int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", fallback["canvas_width"])))
    base_canvas_height = int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", fallback["canvas_height"])))
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(_RENDER_DEFAULTS, "canvas_min_width_px", fallback["canvas_min_width_px"]))),
                int(
                    round(
                        float(max_board_width_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_side_padding_px",
                                    group_default(_RENDER_DEFAULTS, "canvas_side_padding_px", fallback["canvas_side_padding_px"]),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(_RENDER_DEFAULTS, "canvas_min_height_px", fallback["canvas_min_height_px"]))),
                int(
                    round(
                        float(max_board_width_px)
                        + float(player_badge_height_px)
                        + float(header_gap_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_vertical_padding_px",
                                    group_default(_RENDER_DEFAULTS, "canvas_vertical_padding_px", fallback["canvas_vertical_padding_px"]),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    return ConnectFourRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", fallback["panel_margin_px"]))),
        player_badge_height_px=int(player_badge_height_px),
        player_badge_width_px=int(player_badge_width_px),
        header_gap_px=int(header_gap_px),
        max_board_width_px=int(max_board_width_px),
        board_corner_radius_px=scale_games_px(
            params.get(
                "board_corner_radius_px",
                group_default(_RENDER_DEFAULTS, "board_corner_radius_px", fallback["board_corner_radius_px"]),
            ),
            unit_scale,
            min_px=12,
        ),
        board_frame_width_px=scale_games_px(
            params.get(
                "board_frame_width_px",
                group_default(_RENDER_DEFAULTS, "board_frame_width_px", fallback["board_frame_width_px"]),
            ),
            unit_scale,
            min_px=8,
        ),
        disc_inset_fraction=float(params.get("disc_inset_fraction", group_default(_RENDER_DEFAULTS, "disc_inset_fraction", fallback["disc_inset_fraction"]))),
        player_badge_font_size_px=int(
            params.get(
                "player_badge_font_size_px",
                group_default(_RENDER_DEFAULTS, "player_badge_font_size_px", fallback["player_badge_font_size_px"]),
            )
        ),
        marked_square_outline_width_px=scale_games_px(
            params.get(
                "marked_square_outline_width_px",
                group_default(_RENDER_DEFAULTS, "marked_square_outline_width_px", fallback["marked_square_outline_width_px"]),
            ),
            unit_scale,
            min_px=3,
        ),
        layout_jitter_meta=layout_jitter,
        font_family=str(font_family),
    )


def _disc_bbox(
    cell_bbox: Tuple[float, float, float, float],
    *,
    inset_fraction: float,
) -> Tuple[float, float, float, float]:
    """Return one inscribed disc bbox inside one cell."""

    left, top, right, bottom = cell_bbox
    inset = float(max(5.0, inset_fraction * min(right - left, bottom - top)))
    return (
        round(float(left + inset), 3),
        round(float(top + inset), 3),
        round(float(right - inset), 3),
        round(float(bottom - inset), 3),
    )


def _adjust_rgb(rgb: Tuple[int, int, int], delta: int) -> Tuple[int, int, int]:
    """Return `rgb` shifted by `delta` with channel clamping."""

    return tuple(max(0, min(255, int(value) + int(delta))) for value in rgb)


def _shrink_bbox(bbox_px: Tuple[float, float, float, float], inset_px: float) -> Tuple[float, float, float, float]:
    """Return `bbox_px` inset on all sides."""

    return (
        round(float(bbox_px[0] + inset_px), 3),
        round(float(bbox_px[1] + inset_px), 3),
        round(float(bbox_px[2] - inset_px), 3),
        round(float(bbox_px[3] - inset_px), 3),
    )


def _draw_rgba_rounded_rectangle(
    image: Image.Image,
    *,
    bbox_px: Tuple[float, float, float, float],
    radius_px: int,
    fill_rgba: Tuple[int, int, int, int],
) -> None:
    """Composite one translucent rounded rectangle into `image`."""

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        bbox_px,
        radius=int(radius_px),
        fill=tuple(int(value) for value in fill_rgba),
    )
    image.alpha_composite(overlay)


def _draw_disc(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: ConnectFourTheme,
    player: int,
) -> None:
    """Draw one Connect Four disc with simple highlight chrome."""

    if int(player) == int(RED):
        fill_rgb = theme.red_disc_fill_rgb
        outline_rgb = theme.red_disc_outline_rgb
        shine_rgb = theme.red_disc_shine_rgb
    else:
        fill_rgb = theme.yellow_disc_fill_rgb
        outline_rgb = theme.yellow_disc_outline_rgb
        shine_rgb = theme.yellow_disc_shine_rgb
    draw.ellipse(
        bbox_px,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=int(theme.disc_outline_width_px),
    )
    if str(theme.disc_rendering) == "flat":
        return
    left, top, right, bottom = bbox_px
    if str(theme.disc_rendering) == "token":
        ring_inset = max(3.0, 0.16 * min(right - left, bottom - top))
        draw.ellipse(
            _shrink_bbox(bbox_px, ring_inset),
            outline=tuple(int(value) for value in _adjust_rgb(outline_rgb, 38)),
            width=max(2, int(theme.disc_outline_width_px) - 1),
        )
    shine_w = 0.34 * (right - left)
    shine_h = 0.24 * (bottom - top)
    draw.ellipse(
        [
            left + 0.18 * (right - left),
            top + 0.16 * (bottom - top),
            left + 0.18 * (right - left) + shine_w,
            top + 0.16 * (bottom - top) + shine_h,
        ],
        fill=tuple(int(value) for value in shine_rgb),
    )


def _draw_cell_well(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: ConnectFourTheme,
    cell_size: int,
) -> None:
    """Draw one empty Connect Four well using the active board style."""

    well_inset = max(5.0, 0.10 * float(cell_size))
    well_bbox = (
        float(bbox_px[0] + well_inset),
        float(bbox_px[1] + well_inset),
        float(bbox_px[2] - well_inset),
        float(bbox_px[3] - well_inset),
    )
    rendering = str(theme.cell_well_rendering)
    if rendering == "inset":
        draw.ellipse(
            well_bbox,
            fill=tuple(int(value) for value in _adjust_rgb(theme.cell_well_outline_rgb, -18)),
        )
        draw.ellipse(
            _shrink_bbox(well_bbox, max(2.0, 0.035 * float(cell_size))),
            fill=tuple(int(value) for value in theme.cell_well_rgb),
            outline=tuple(int(value) for value in theme.cell_well_outline_rgb),
            width=int(theme.cell_well_outline_width_px),
        )
        return
    if rendering == "ring":
        draw.ellipse(
            well_bbox,
            fill=tuple(int(value) for value in theme.cell_well_outline_rgb),
        )
        draw.ellipse(
            _shrink_bbox(well_bbox, max(2.0, 0.045 * float(cell_size))),
            fill=tuple(int(value) for value in theme.cell_well_rgb),
            outline=tuple(int(value) for value in _adjust_rgb(theme.cell_well_outline_rgb, -12)),
            width=max(1, int(theme.cell_well_outline_width_px) - 1),
        )
        return
    draw.ellipse(
        well_bbox,
        fill=tuple(int(value) for value in theme.cell_well_rgb),
        outline=tuple(int(value) for value in theme.cell_well_outline_rgb),
        width=int(theme.cell_well_outline_width_px),
    )


def render_connect_four_board_scene(
    *,
    board: Sequence[Sequence[int]],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    current_player: int,
    params: ConnectFourRenderParams,
    marked_square: Coord | None,
    panel_style: GamePanelSceneStyle | None = None,
    column_labels: Sequence[str] | None = None,
) -> RenderedConnectFourScene:
    """Render one visible Connect Four board with an optional marked move square."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    theme = build_games_connect_four_theme(style_variant=str(style_variant))
    rows, columns = board_dimensions(board)
    visible_column_labels = tuple(str(label) for label in column_labels) if column_labels is not None else tuple()
    if visible_column_labels and len(visible_column_labels) != int(columns):
        raise ValueError("column_labels length must match the board column count")
    label_band_height = max(24, int(round(0.34 * int(params.player_badge_height_px)))) if visible_column_labels else 0
    label_gap_px = max(6, int(round(0.10 * int(params.player_badge_height_px)))) if visible_column_labels else 0

    cell_size = min(
        int(params.max_board_width_px) // int(columns),
        (int(params.canvas_width) - (2 * int(params.panel_margin_px))) // int(columns),
        (
            int(params.canvas_height)
            - (2 * int(params.panel_margin_px))
            - int(params.player_badge_height_px)
            - int(params.header_gap_px)
            - int(label_gap_px)
            - int(label_band_height)
        )
        // int(rows),
    )
    board_width = int(cell_size) * int(columns)
    board_height = int(cell_size) * int(rows)
    board_left = int(0.5 * (int(params.canvas_width) - int(board_width)))
    available_height = (
        int(params.canvas_height)
        - (2 * int(params.panel_margin_px))
        - int(params.player_badge_height_px)
        - int(params.header_gap_px)
        - int(label_gap_px)
        - int(label_band_height)
    )
    board_top = int(
        params.panel_margin_px
        + params.player_badge_height_px
        + params.header_gap_px
        + max(0, 0.5 * (available_height - int(board_height)))
    )
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_width), 3),
        round(float(board_top + board_height), 3),
    )
    column_label_band_bbox = (
        round(float(board_left), 3),
        round(float(board_top + board_height + label_gap_px), 3),
        round(float(board_left + board_width), 3),
        round(float(board_top + board_height + label_gap_px + label_band_height), 3),
    )

    badge_font = load_font(
        int(params.player_badge_font_size_px),
        bold=True,
        font_family=str(params.font_family) or None,
    )
    badge_text = f"{player_name(int(current_player))} to move"
    badge_text_bbox = draw.textbbox((0, 0), badge_text, font=badge_font, stroke_width=1)
    badge_width = max(
        int(params.player_badge_width_px),
        int((badge_text_bbox[2] - badge_text_bbox[0]) + params.player_badge_height_px + 34),
    )
    badge_left = int(0.5 * (int(params.canvas_width) - int(badge_width)))
    badge_top = int(params.panel_margin_px)
    badge_bbox = (
        round(float(badge_left), 3),
        round(float(badge_top), 3),
        round(float(badge_left + badge_width), 3),
        round(float(badge_top + params.player_badge_height_px), 3),
    )
    group_bbox = (
        min(float(board_bbox[0]), float(badge_bbox[0])),
        min(float(board_bbox[1]), float(badge_bbox[1])),
        max(float(board_bbox[2]), float(badge_bbox[2])),
        max(float(board_bbox[3]), float(badge_bbox[3]), float(column_label_band_bbox[3]) if visible_column_labels else float(board_bbox[3])),
    )
    _group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=group_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left = float(board_left + dx)
    board_top = float(board_top + dy)
    badge_left = float(badge_left + dx)
    badge_top = float(badge_top + dy)
    board_bbox = offset_bbox(board_bbox, dx=dx, dy=dy)
    badge_bbox = offset_bbox(badge_bbox, dx=dx, dy=dy)
    column_label_band_bbox = offset_bbox(column_label_band_bbox, dx=dx, dy=dy)

    scene_panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(18, int(round(float(params.panel_margin_px) * 0.42)))
        scene_panel_bbox = (
            max(4, int(round(min(board_bbox[0], badge_bbox[0]))) - panel_pad),
            max(4, int(round(min(board_bbox[1], badge_bbox[1]))) - panel_pad),
            min(int(params.canvas_width) - 4, int(round(max(board_bbox[2], badge_bbox[2]))) + panel_pad),
            min(
                int(params.canvas_height) - 4,
                int(
                    round(
                        max(
                            board_bbox[3],
                            badge_bbox[3],
                            column_label_band_bbox[3] if visible_column_labels else board_bbox[3],
                        )
                    )
                )
                + panel_pad,
            ),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=scene_panel_bbox,
            style=panel_style,
            radius=28,
            border_width=max(2, int(round(float(params.board_frame_width_px) * 0.45))),
        )

    if int(theme.board_shadow_alpha) > 0:
        shadow_dx, shadow_dy = theme.board_shadow_offset_px
        shadow_bbox = offset_bbox(board_bbox, dx=float(shadow_dx), dy=float(shadow_dy))
        _draw_rgba_rounded_rectangle(
            image,
            bbox_px=shadow_bbox,
            radius_px=int(params.board_corner_radius_px),
            fill_rgba=(
                int(theme.board_shadow_rgb[0]),
                int(theme.board_shadow_rgb[1]),
                int(theme.board_shadow_rgb[2]),
                int(theme.board_shadow_alpha),
            ),
        )
        draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        board_bbox,
        radius=int(params.board_corner_radius_px),
        fill=tuple(int(value) for value in theme.board_frame_rgb),
    )
    inner_inset = float(params.board_frame_width_px)
    inner_bbox = (
        round(float(board_bbox[0] + inner_inset), 3),
        round(float(board_bbox[1] + inner_inset), 3),
        round(float(board_bbox[2] - inner_inset), 3),
        round(float(board_bbox[3] - inner_inset), 3),
    )
    draw.rounded_rectangle(
        inner_bbox,
        radius=max(8, int(params.board_corner_radius_px) - int(params.board_frame_width_px)),
        fill=tuple(int(value) for value in theme.board_fill_rgb),
    )
    if str(theme.board_rendering) == "inset":
        inner_outline_bbox = _shrink_bbox(inner_bbox, max(2.0, 0.20 * float(params.board_frame_width_px)))
        draw.rounded_rectangle(
            inner_outline_bbox,
            radius=max(8, int(params.board_corner_radius_px) - int(params.board_frame_width_px) - 2),
            outline=tuple(int(value) for value in _adjust_rgb(theme.board_fill_rgb, 34)),
            width=2,
        )
    draw.rounded_rectangle(
        badge_bbox,
        radius=int(0.5 * int(params.player_badge_height_px)),
        fill=tuple(int(value) for value in theme.badge_fill_rgb),
        outline=tuple(int(value) for value in theme.badge_outline_rgb),
        width=2,
    )
    disc_d = int(params.player_badge_height_px) - 16
    disc_left = int(badge_left + 12)
    disc_top = int(badge_top + 8)
    _draw_disc(
        draw,
        bbox_px=(float(disc_left), float(disc_top), float(disc_left + disc_d), float(disc_top + disc_d)),
        theme=theme,
        player=int(current_player),
    )
    badge_text_rgb = tuple(int(value) for value in theme.badge_text_rgb)
    draw_text_traced(draw,
        (
            float(disc_left + disc_d + 12),
            float(badge_top + 0.5 * (int(params.player_badge_height_px) - (badge_text_bbox[3] - badge_text_bbox[1]))),
        ),
        badge_text,
        font=badge_font,
        fill=badge_text_rgb,
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(badge_text_rgb)),
     role="readout", required=False,)

    cell_specs: List[ConnectFourCellSpec] = []
    scene_entities: List[Dict[str, Any]] = []
    cell_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    disc_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    column_label_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    column_label_to_col: Dict[str, int] = {}

    marked_square_bbox_px: Tuple[float, float, float, float] | None = None
    for row in range(int(rows)):
        for col in range(int(columns)):
            cell_id = coord_to_cell_id((int(row), int(col)))
            cell_bbox = (
                round(float(board_left + (col * cell_size)), 3),
                round(float(board_top + (row * cell_size)), 3),
                round(float(board_left + ((col + 1) * cell_size)), 3),
                round(float(board_top + ((row + 1) * cell_size)), 3),
            )
            _draw_cell_well(draw, bbox_px=cell_bbox, theme=theme, cell_size=int(cell_size))
            if marked_square is not None and (int(row), int(col)) == (int(marked_square[0]), int(marked_square[1])):
                marked_square_bbox_px = cell_bbox
                inset = 4.0
                draw.rounded_rectangle(
                    [
                        cell_bbox[0] + inset,
                        cell_bbox[1] + inset,
                        cell_bbox[2] - inset,
                        cell_bbox[3] - inset,
                    ],
                    radius=max(8, int(0.18 * cell_size)),
                    outline=tuple(int(value) for value in theme.marked_square_outline_rgb),
                    width=int(params.marked_square_outline_width_px),
                    fill=tuple(int(value) for value in theme.marked_square_fill_rgba),
                )
            occupant_value = int(board[row][col])
            occupant_name = "empty" if int(occupant_value) == 0 else "red" if int(occupant_value) == int(RED) else "yellow"
            disc_bbox_px = None
            if int(occupant_value) in {int(RED), int(YELLOW)}:
                disc_bbox_px = _disc_bbox(cell_bbox, inset_fraction=float(params.disc_inset_fraction))
                disc_bboxes_px[str(cell_id)] = disc_bbox_px
                _draw_disc(draw, bbox_px=disc_bbox_px, theme=theme, player=int(occupant_value))
            cell_specs.append(
                ConnectFourCellSpec(
                    cell_id=str(cell_id),
                    row=int(row),
                    col=int(col),
                    occupant=str(occupant_name),
                    bbox_px=cell_bbox,
                    disc_bbox_px=disc_bbox_px,
                )
            )
            cell_bboxes_px[str(cell_id)] = cell_bbox
            entity: Dict[str, Any] = {
                "entity_id": str(cell_id),
                "entity_type": "board_cell",
                "row": int(row),
                "col": int(col),
                "occupant": str(occupant_name),
                "bbox": list(cell_bbox),
            }
            if disc_bbox_px is not None:
                entity["disc_bbox"] = list(disc_bbox_px)
            scene_entities.append(entity)

    if visible_column_labels:
        label_rgb = (24, 28, 35)
        label_stroke = (255, 255, 255)
        for col, label in enumerate(visible_column_labels):
            label_bbox = (
                round(float(board_left + (col * cell_size)), 3),
                round(float(column_label_band_bbox[1]), 3),
                round(float(board_left + ((col + 1) * cell_size)), 3),
                round(float(column_label_band_bbox[3]), 3),
            )
            font = fit_font_to_box(
                draw,
                text=str(label),
                max_width=max(1.0, float(label_bbox[2] - label_bbox[0])),
                max_height=max(1.0, float(label_bbox[3] - label_bbox[1])),
                bold=True,
                font_family=str(params.font_family) or None,
                min_size_px=10,
                max_size_px=max(12, int(round(0.46 * float(cell_size)))),
                fill_ratio=0.76,
            )
            text_bbox = draw.textbbox((0, 0), str(label), font=font, stroke_width=2)
            text_w = float(text_bbox[2] - text_bbox[0])
            text_h = float(text_bbox[3] - text_bbox[1])
            x = float(label_bbox[0] + (0.5 * ((label_bbox[2] - label_bbox[0]) - text_w)) - text_bbox[0])
            y = float(label_bbox[1] + (0.5 * ((label_bbox[3] - label_bbox[1]) - text_h)) - text_bbox[1])
            label_surface_xy = (
                int(round(0.5 * (float(label_bbox[0]) + float(label_bbox[2])))),
                int(round(0.5 * (float(label_bbox[1]) + float(label_bbox[3])))),
            )
            label_surface_rgb = tuple(int(value) for value in image.getpixel(label_surface_xy)[:3])
            draw_text_traced(
                draw,
                (float(x), float(y)),
                str(label),
                font=font,
                fill=label_rgb,
                stroke_width=2,
                stroke_fill=label_stroke,
                role="column_label",
                required=True,
                surface_rgbs=(label_surface_rgb,),
                preferred_rgbs=(label_rgb,),
                namespace=f"games.connect_four.column_label.{label}",
            )
            column_label_bboxes_px[str(label)] = label_bbox
            column_label_to_col[str(label)] = int(col)

    return RenderedConnectFourScene(
        image=image,
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map={
            "board_bbox_px": list(board_bbox),
            "scene_panel_bbox_px": None if scene_panel_bbox is None else [int(value) for value in scene_panel_bbox],
            "cell_bboxes_px": {str(key): list(value) for key, value in cell_bboxes_px.items()},
            "disc_bboxes_px": {str(key): list(value) for key, value in disc_bboxes_px.items()},
            "column_label_bboxes_px": {str(key): list(value) for key, value in column_label_bboxes_px.items()},
            "column_label_to_col": {str(key): int(value) for key, value in column_label_to_col.items()},
            "player_badge_bbox_px": list(badge_bbox),
            "marked_square_bbox_px": None if marked_square_bbox_px is None else list(marked_square_bbox_px),
            "rows": int(rows),
            "columns": int(columns),
            "effective_cell_size_px": float(cell_size),
            "scene_variant": str(scene_variant),
            "style_variant": str(style_variant),
            "layout_jitter": dict(layout_jitter),
            "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
            "font_family": str(params.font_family),
        },
    )


def render_connect_four_sample(
    *,
    sample: ConnectFourCountSample | ConnectFourLabelSample | ConnectFourColumnProfileSample,
    params: Mapping[str, Any],
    instance_seed: int,
    marked_square: Coord | None = None,
    column_labels: Sequence[str] | None = None,
) -> RenderedConnectFourTaskContext:
    """Render one Connect Four sample without binding a public answer."""

    render_params = resolve_connect_four_render_params(params, instance_seed=int(instance_seed))
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
        namespace="games.connect_four.panel_scene_style",
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
    rendered_scene = render_connect_four_board_scene(
        board=sample.board,
        background=background,
        scene_variant=str(sample.scene_variant),
        style_variant=str(sample.style_variant),
        current_player=int(sample.current_player),
        params=render_params,
        marked_square=marked_square,
        panel_style=panel_style,
        column_labels=column_labels,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedConnectFourTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "ConnectFourCellSpec",
    "ConnectFourRenderParams",
    "RenderedConnectFourScene",
    "RenderedConnectFourTaskContext",
    "render_connect_four_board_scene",
    "render_connect_four_sample",
    "resolve_connect_four_render_params",
]
