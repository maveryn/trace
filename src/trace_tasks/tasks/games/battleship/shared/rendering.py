"""Rendering helpers for the Battleship games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_game_text_traced as draw_text_traced
from .state import Coord, FLEET_SHAPES, BattleshipSample, BattleshipShapeOption, all_coords, coord_to_cell_id
from .rules import shape_orientations
from .defaults import FALLBACK_RENDERING_DEFAULTS
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.layout import attach_games_unit_size_jitter, resolve_games_layout_jitter, resolve_games_unit_size_scale, scale_games_px
from ...shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.style import BattleshipTheme, build_games_battleship_theme
from ...shared.visual_defaults import load_games_scene_noise_defaults


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    "battleship",
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id="battleship", apply_prob=0.0)


@dataclass(frozen=True)
class BattleshipRenderParams:
    """Resolved render controls for one Battleship scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_size_px: int
    board_border_width_px: int
    grid_line_width_px: int
    cell_padding_px: int
    fleet_panel_width_px: int
    board_panel_gap_px: int
    fleet_icon_cell_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class BattleshipCellSpec:
    """One rendered Battleship tracking-grid cell."""

    cell_id: str
    row: int
    col: int
    state: str
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedBattleshipScene:
    """Rendered Battleship image plus trace-friendly geometry."""

    image: Image.Image
    cell_specs: Tuple[BattleshipCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedBattleshipTaskContext:
    """Rendered Battleship scene plus common render metadata."""

    image: Image.Image
    rendered_scene: RenderedBattleshipScene
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


def resolve_battleship_render_params(params: Mapping[str, Any], *, instance_seed: int) -> BattleshipRenderParams:
    """Resolve Battleship rendering parameters from config/defaults."""

    fallback = FALLBACK_RENDERING_DEFAULTS
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.battleship.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.battleship.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.battleship.layout",
        ),
        unit_scale_meta,
    )
    return BattleshipRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", fallback["canvas_width"]))),
        canvas_height=int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", fallback["canvas_height"]))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", fallback["panel_margin_px"]))),
        max_board_size_px=scale_games_px(params.get("max_board_size_px", group_default(_RENDER_DEFAULTS, "max_board_size_px", fallback["max_board_size_px"])), unit_scale, min_px=320),
        board_border_width_px=scale_games_px(params.get("board_border_width_px", group_default(_RENDER_DEFAULTS, "board_border_width_px", fallback["board_border_width_px"])), unit_scale, min_px=2),
        grid_line_width_px=scale_games_px(params.get("grid_line_width_px", group_default(_RENDER_DEFAULTS, "grid_line_width_px", fallback["grid_line_width_px"])), unit_scale, min_px=1),
        cell_padding_px=scale_games_px(params.get("cell_padding_px", group_default(_RENDER_DEFAULTS, "cell_padding_px", fallback["cell_padding_px"])), unit_scale, min_px=3),
        fleet_panel_width_px=int(params.get("fleet_panel_width_px", group_default(_RENDER_DEFAULTS, "fleet_panel_width_px", fallback["fleet_panel_width_px"]))),
        board_panel_gap_px=int(params.get("board_panel_gap_px", group_default(_RENDER_DEFAULTS, "board_panel_gap_px", fallback["board_panel_gap_px"]))),
        fleet_icon_cell_px=scale_games_px(params.get("fleet_icon_cell_px", group_default(_RENDER_DEFAULTS, "fleet_icon_cell_px", fallback["fleet_icon_cell_px"])), unit_scale, min_px=9),
        label_font_size_px=scale_games_px(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", fallback["label_font_size_px"])), unit_scale, min_px=12),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


def _cell_bbox(
    *,
    board_left: float,
    board_top: float,
    cell_size: float,
    row: int,
    col: int,
    padding_px: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Return the bbox for one Battleship cell, with optional inset padding."""

    left = float(board_left + (int(col) * float(cell_size)) + float(padding_px))
    top = float(board_top + (int(row) * float(cell_size)) + float(padding_px))
    right = float(board_left + ((int(col) + 1) * float(cell_size)) - float(padding_px))
    bottom = float(board_top + ((int(row) + 1) * float(cell_size)) - float(padding_px))
    return (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))


def _draw_centered_label(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    text: str,
    fill: Tuple[int, int, int],
    max_size_px: int,
    bold: bool = False,
    font_family: str | None = None,
    required: bool = False,
) -> None:
    """Draw text centered inside one bounding box."""

    left, top, right, bottom = bbox_px
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left)),
        max_height=max(1.0, float(bottom - top)),
        bold=bool(bold),
        min_size_px=10,
        max_size_px=int(max_size_px),
        fill_ratio=0.82,
        font_family=font_family,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    text_x = float(left + (0.5 * (float(right - left) - text_w)) - float(text_bbox[0]))
    text_y = float(top + (0.5 * (float(bottom - top) - text_h)) - float(text_bbox[1]))
    draw_text_traced(
        draw,
        (text_x, text_y),
        str(text),
        fill=tuple(int(v) for v in fill),
        font=font,
        role="readout",
        required=bool(required),
    )


def _draw_hit_marker(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: BattleshipTheme,
) -> None:
    """Draw one red hit marker."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    cx = float(left + (0.5 * width))
    cy = float(top + (0.5 * height))
    radius = float(0.32 * min(width, height))
    outline_width = max(2, int(round(0.05 * min(width, height))))
    style = str(theme.hit_marker_style)
    fill = tuple(int(v) for v in theme.hit_fill_rgb)
    outline = tuple(int(v) for v in theme.hit_outline_rgb)
    if style == "cross":
        pad = float(0.24 * min(width, height))
        draw.line([(left + pad, top + pad), (right - pad, bottom - pad)], fill=fill, width=outline_width + 2)
        draw.line([(left + pad, bottom - pad), (right - pad, top + pad)], fill=fill, width=outline_width + 2)
        return
    if style == "square":
        pad = float(0.22 * min(width, height))
        draw.rectangle((left + pad, top + pad, right - pad, bottom - pad), fill=fill, outline=outline, width=outline_width)
        return
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=outline, width=outline_width)


def _draw_miss_marker(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: BattleshipTheme,
) -> None:
    """Draw one miss marker."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    cx = float(left + (0.5 * width))
    cy = float(top + (0.5 * height))
    radius = float(0.20 * min(width, height))
    line_width = max(2, int(round(0.04 * min(width, height))))
    fill = tuple(int(v) for v in theme.miss_rgb)
    if str(theme.miss_marker_style) == "cross":
        pad = float(0.31 * min(width, height))
        draw.line([(left + pad, top + pad), (right - pad, bottom - pad)], fill=fill, width=line_width)
        draw.line([(left + pad, bottom - pad), (right - pad, top + pad)], fill=fill, width=line_width)
    elif str(theme.miss_marker_style) == "dot":
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
    else:
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=fill, width=line_width)


def _draw_ship_body_cell(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: BattleshipTheme,
    is_sunk: bool,
) -> None:
    """Draw the visible ship body inside one board cell."""

    left, top, right, bottom = bbox_px
    pad = float(0.12 * min(float(right - left), float(bottom - top)))
    fill_alpha = 118 if bool(is_sunk) else 82
    outline_alpha = 214 if bool(is_sunk) else 168
    draw.rounded_rectangle(
        (left + pad, top + pad, right - pad, bottom - pad),
        radius=max(3, int(round(0.12 * min(float(right - left), float(bottom - top))))),
        fill=tuple(int(v) for v in theme.ship_icon_fill_rgb) + (int(fill_alpha),),
        outline=tuple(int(v) for v in theme.ship_icon_outline_rgb) + (int(outline_alpha),),
        width=max(1, int(round(0.035 * min(float(right - left), float(bottom - top))))),
    )


def _draw_candidate_label(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    label: str,
    params: BattleshipRenderParams,
) -> None:
    """Draw one visible option label badge inside a candidate board cell."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    cx = float(left + (0.5 * width))
    cy = float(top + (0.5 * height))
    radius = float(0.33 * min(width, height))
    badge_bbox = (
        round(cx - radius, 3),
        round(cy - radius, 3),
        round(cx + radius, 3),
        round(cy + radius, 3),
    )
    draw.ellipse(
        badge_bbox,
        fill=(255, 244, 184, 244),
        outline=(24, 38, 62, 255),
        width=max(2, int(round(0.045 * min(width, height)))),
    )
    _draw_centered_label(
        draw,
        bbox_px=badge_bbox,
        text=str(label),
        fill=(20, 29, 45),
        max_size_px=max(12, int(round(0.44 * min(width, height)))),
        bold=True,
        font_family=str(params.font_family) or None,
        required=True,
    )


def _draw_fleet_shape_icon(
    draw: ImageDraw.ImageDraw,
    *,
    origin_xy: Tuple[float, float],
    shape_offsets: Sequence[Coord],
    cell_px: int,
    theme: BattleshipTheme,
) -> Tuple[float, float, float, float]:
    """Draw one fleet-shape icon and return its bbox."""

    oriented = shape_orientations(shape_offsets)[0]
    min_row = min(row for row, _col in oriented)
    min_col = min(col for _row, col in oriented)
    max_row = max(row for row, _col in oriented)
    max_col = max(col for _row, col in oriented)
    left, top = origin_xy
    for row, col in oriented:
        x0 = float(left + (int(col - min_col) * int(cell_px)))
        y0 = float(top + (int(row - min_row) * int(cell_px)))
        x1 = float(x0 + int(cell_px))
        y1 = float(y0 + int(cell_px))
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=max(2, int(round(0.16 * int(cell_px)))),
            fill=tuple(int(v) for v in theme.ship_icon_fill_rgb),
            outline=tuple(int(v) for v in theme.ship_icon_outline_rgb),
            width=max(1, int(round(0.08 * int(cell_px)))),
        )
    return (
        round(float(left), 3),
        round(float(top), 3),
        round(float(left + ((int(max_col - min_col) + 1) * int(cell_px))), 3),
        round(float(top + ((int(max_row - min_row) + 1) * int(cell_px))), 3),
    )


def _draw_fleet_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Tuple[float, float, float, float],
    params: BattleshipRenderParams,
    theme: BattleshipTheme,
) -> Dict[str, List[float]]:
    """Draw the fleet panel and return icon bboxes by shape id."""

    draw.rounded_rectangle(
        panel_bbox,
        radius=12,
        fill=tuple(int(v) for v in theme.panel_fill_rgb),
        outline=tuple(int(v) for v in theme.panel_border_rgb),
        width=3,
    )
    title_bbox = (panel_bbox[0] + 16, panel_bbox[1] + 16, panel_bbox[2] - 16, panel_bbox[1] + 58)
    _draw_centered_label(
        draw,
        bbox_px=title_bbox,
        text="Fleet shapes",
        fill=tuple(int(v) for v in theme.panel_text_rgb),
        max_size_px=int(params.label_font_size_px) + 4,
        bold=True,
        font_family=str(params.font_family) or None,
    )

    icon_bboxes: Dict[str, List[float]] = {}
    row_top = float(panel_bbox[1] + 68)
    row_gap = max(42.0, min(66.0, float((float(panel_bbox[3]) - row_top - 16.0) / float(len(FLEET_SHAPES)))))
    icon_left = float(panel_bbox[0] + 26)
    label_left = float(panel_bbox[0] + 128)
    cell_px = int(params.fleet_icon_cell_px)
    for index, shape in enumerate(FLEET_SHAPES):
        top = float(row_top + (index * row_gap))
        oriented = shape_orientations(shape.offsets)[0]
        icon_height = float((max(row for row, _col in oriented) - min(row for row, _col in oriented) + 1) * cell_px)
        icon_top = float(top + max(2.0, 0.5 * (row_gap - icon_height)))
        icon_bbox = _draw_fleet_shape_icon(
            draw,
            origin_xy=(icon_left, icon_top),
            shape_offsets=shape.offsets,
            cell_px=cell_px,
            theme=theme,
        )
        icon_bboxes[str(shape.shape_id)] = [float(value) for value in icon_bbox]
        label_bbox = (label_left, top, panel_bbox[2] - 14, min(float(panel_bbox[3]) - 10.0, top + row_gap))
        _draw_centered_label(
            draw,
            bbox_px=label_bbox,
            text=shape.display_name,
            fill=tuple(int(v) for v in theme.panel_text_rgb),
            max_size_px=int(params.label_font_size_px),
            bold=False,
            font_family=str(params.font_family) or None,
        )
    return icon_bboxes


def _draw_shape_option_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Tuple[float, float, float, float],
    shape_options: Sequence[BattleshipShapeOption],
    params: BattleshipRenderParams,
    theme: BattleshipTheme,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Draw labeled fleet-shape answer choices and return icon/option bboxes."""

    draw.rounded_rectangle(
        panel_bbox,
        radius=12,
        fill=tuple(int(v) for v in theme.panel_fill_rgb),
        outline=tuple(int(v) for v in theme.panel_border_rgb),
        width=3,
    )
    title_bbox = (panel_bbox[0] + 14, panel_bbox[1] + 14, panel_bbox[2] - 14, panel_bbox[1] + 58)
    _draw_centered_label(
        draw,
        bbox_px=title_bbox,
        text="Answer choices",
        fill=tuple(int(v) for v in theme.panel_text_rgb),
        max_size_px=int(params.label_font_size_px) + 3,
        bold=True,
        font_family=str(params.font_family) or None,
    )

    shapes_by_id = {str(shape.shape_id): shape for shape in FLEET_SHAPES}
    icon_bboxes: Dict[str, List[float]] = {}
    option_bboxes: Dict[str, List[float]] = {}
    row_count = max(1, len(tuple(shape_options)))
    content_top = float(panel_bbox[1] + 68)
    content_bottom = float(panel_bbox[3] - 18)
    row_gap = float((content_bottom - content_top) / float(row_count))
    row_height = max(36.0, min(96.0, float(row_gap - 8.0)))
    option_left = float(panel_bbox[0] + 14)
    option_right = float(panel_bbox[2] - 14)
    badge_size = max(24.0, min(42.0, row_height * 0.46))
    cell_px = max(8, min(int(round(float(params.fleet_icon_cell_px) * 0.9)), int(round(row_height * 0.32))))
    icon_left = float(option_left + badge_size + 18.0)
    text_left = float(option_left + badge_size + 128.0)

    for index, option in enumerate(shape_options):
        row_mid = float(content_top + (index * row_gap) + (0.5 * row_gap))
        option_bbox = (
            round(option_left, 3),
            round(row_mid - (0.5 * row_height), 3),
            round(option_right, 3),
            round(row_mid + (0.5 * row_height), 3),
        )
        option_bboxes[str(option.label)] = [float(value) for value in option_bbox]
        draw.rounded_rectangle(
            option_bbox,
            radius=10,
            fill=tuple(int(v) for v in theme.board_fill_rgb) + (182,),
            outline=tuple(int(v) for v in theme.panel_border_rgb),
            width=2,
        )
        badge_bbox = (
            round(float(option_bbox[0] + 10.0), 3),
            round(float(row_mid - (0.5 * badge_size)), 3),
            round(float(option_bbox[0] + 10.0 + badge_size), 3),
            round(float(row_mid + (0.5 * badge_size)), 3),
        )
        draw.ellipse(
            badge_bbox,
            fill=(255, 244, 184, 250),
            outline=(24, 38, 62, 255),
            width=2,
        )
        _draw_centered_label(
            draw,
            bbox_px=badge_bbox,
            text=str(option.label),
            fill=(20, 29, 45),
            max_size_px=max(14, int(round(badge_size * 0.52))),
            bold=True,
            font_family=str(params.font_family) or None,
            required=True,
        )

        shape = shapes_by_id[str(option.shape_id)]
        oriented = shape_orientations(shape.offsets)[0]
        icon_height = float((max(row for row, _col in oriented) - min(row for row, _col in oriented) + 1) * cell_px)
        icon_top = float(row_mid - (0.5 * icon_height))
        icon_bbox = _draw_fleet_shape_icon(
            draw,
            origin_xy=(icon_left, icon_top),
            shape_offsets=shape.offsets,
            cell_px=cell_px,
            theme=theme,
        )
        icon_bboxes[str(option.shape_id)] = [float(value) for value in icon_bbox]
        label_bbox = (
            text_left,
            float(option_bbox[1] + 8.0),
            float(option_bbox[2] - 10.0),
            float(option_bbox[3] - 8.0),
        )
        _draw_centered_label(
            draw,
            bbox_px=label_bbox,
            text=str(option.display_name),
            fill=tuple(int(v) for v in theme.panel_text_rgb),
            max_size_px=int(params.label_font_size_px),
            bold=False,
            font_family=str(params.font_family) or None,
        )
    return icon_bboxes, option_bboxes


def _bbox_union(bboxes: Sequence[Sequence[float]]) -> List[float]:
    """Return the tight axis-aligned union for non-empty bboxes."""

    if not bboxes:
        raise ValueError("cannot union an empty bbox list")
    xs0 = [float(bbox[0]) for bbox in bboxes]
    ys0 = [float(bbox[1]) for bbox in bboxes]
    xs1 = [float(bbox[2]) for bbox in bboxes]
    ys1 = [float(bbox[3]) for bbox in bboxes]
    return [
        round(float(min(xs0)), 3),
        round(float(min(ys0)), 3),
        round(float(max(xs1)), 3),
        round(float(max(ys1)), 3),
    ]


def render_battleship_grid_scene(
    *,
    board_size: int,
    ship_cells_by_id: Mapping[str, Sequence[Coord]],
    sunk_ship_ids: Sequence[str],
    hit_coords: Sequence[Coord],
    miss_coords: Sequence[Coord],
    background: Image.Image,
    style_variant: str,
    params: BattleshipRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
    show_ship_bodies: bool = True,
    candidate_labels_by_coord: Mapping[Coord, str] | None = None,
    shape_options: Sequence[BattleshipShapeOption] = tuple(),
) -> RenderedBattleshipScene:
    """Render one Battleship grid with visible ships, red hits, misses, and a fleet panel."""

    size = int(board_size)
    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_battleship_theme(style_variant=str(style_variant))

    available_width = int(params.canvas_width) - (2 * int(params.panel_margin_px)) - int(params.fleet_panel_width_px) - int(params.board_panel_gap_px)
    available_height = int(params.canvas_height) - (2 * int(params.panel_margin_px))
    board_px = min(int(params.max_board_size_px), int(available_width), int(available_height))
    content_width = float(board_px + int(params.board_panel_gap_px) + int(params.fleet_panel_width_px))
    content_height = float(board_px)
    content_left = float(0.5 * (int(params.canvas_width) - content_width))
    content_top = float(0.5 * (int(params.canvas_height) - content_height))
    content_bbox = (
        round(content_left, 3),
        round(content_top, 3),
        round(content_left + content_width, 3),
        round(content_top + content_height, 3),
    )
    content_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=content_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left = float(content_bbox[0])
    board_top = float(content_bbox[1])
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_px), 3),
        round(float(board_top + board_px), 3),
    )
    board_left = float(board_bbox[0])
    board_top = float(board_bbox[1])
    board_px = float(board_bbox[2] - board_bbox[0])
    cell_size = float(board_px / float(size))
    panel_left = float(board_bbox[2] + int(params.board_panel_gap_px))
    panel_top = float(board_bbox[1])
    panel_bbox = (
        round(panel_left, 3),
        round(panel_top, 3),
        round(panel_left + int(params.fleet_panel_width_px), 3),
        round(panel_top + board_px, 3),
    )

    scene_panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(18, int(round(float(params.panel_margin_px) * 0.42)))
        scene_panel_bbox = (
            max(4, int(round(board_bbox[0])) - panel_pad),
            max(4, int(round(min(board_bbox[1], panel_bbox[1]))) - panel_pad),
            min(int(params.canvas_width) - 4, int(round(panel_bbox[2])) + panel_pad),
            min(int(params.canvas_height) - 4, int(round(max(board_bbox[3], panel_bbox[3]))) + panel_pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=scene_panel_bbox,
            style=panel_style,
            radius=24,
            border_width=max(2, int(round(float(params.board_border_width_px) * 0.55))),
        )

    draw.rounded_rectangle(
        board_bbox,
        radius=10,
        fill=tuple(int(v) for v in theme.board_fill_rgb),
        outline=tuple(int(v) for v in theme.board_border_rgb),
        width=int(params.board_border_width_px),
    )

    hits = {(int(row), int(col)) for row, col in hit_coords}
    misses = {(int(row), int(col)) for row, col in miss_coords}
    candidate_labels = {
        (int(row), int(col)): str(label)
        for (row, col), label in dict(candidate_labels_by_coord or {}).items()
    }
    ship_cells_by_coord: Dict[Coord, str] = {}
    for ship_id, coords in ship_cells_by_id.items():
        for row, col in coords:
            ship_cells_by_coord[(int(row), int(col))] = str(ship_id)
    sunk_ids = {str(ship_id) for ship_id in sunk_ship_ids}
    cell_bboxes_px: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    cell_specs: List[BattleshipCellSpec] = []

    for row, col in all_coords(size=int(size)):
        coord = (int(row), int(col))
        cell_id = coord_to_cell_id(coord)
        full_bbox = _cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=row,
            col=col,
        )
        bbox_px = _cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=row,
            col=col,
            padding_px=float(params.cell_padding_px),
        )
        fill = theme.cell_alt_fill_rgb if (int(row) + int(col)) % 2 else theme.cell_fill_rgb
        draw.rectangle(full_bbox, fill=tuple(int(v) for v in fill))
        ship_id = ship_cells_by_coord.get(coord)
        if ship_id is not None and bool(show_ship_bodies):
            _draw_ship_body_cell(
                draw,
                bbox_px=bbox_px,
                theme=theme,
                is_sunk=str(ship_id) in sunk_ids,
            )
        if coord in hits:
            _draw_hit_marker(draw, bbox_px=bbox_px, theme=theme)
            state = "ship_hit" if ship_id is not None else "hit"
        elif coord in misses:
            _draw_miss_marker(draw, bbox_px=bbox_px, theme=theme)
            state = "miss"
        elif ship_id is not None:
            state = "ship_unhit"
        else:
            state = "unknown"
        cell_bboxes_px[str(cell_id)] = [float(value) for value in bbox_px]
        scene_entities.append(
            {
                "entity_id": str(cell_id),
                "entity_type": "battleship_cell",
                "row": int(row),
                "col": int(col),
                "state": str(state),
                "ship_id": None if ship_id is None else str(ship_id),
                "is_ship_cell": bool(ship_id is not None),
                "is_sunk_ship_cell": bool(ship_id is not None and str(ship_id) in sunk_ids),
                "bbox_px": list(bbox_px),
            }
        )
        cell_specs.append(
            BattleshipCellSpec(
                cell_id=str(cell_id),
                row=int(row),
                col=int(col),
                state=str(state),
                bbox_px=bbox_px,
            )
        )

    for index in range(size + 1):
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
    draw.rounded_rectangle(
        board_bbox,
        radius=10,
        outline=tuple(int(v) for v in theme.board_border_rgb),
        width=int(params.board_border_width_px),
    )
    candidate_label_cell_ids: Dict[str, str] = {}
    candidate_label_bboxes_px: Dict[str, List[float]] = {}
    candidate_label_points_px: Dict[str, List[float]] = {}
    for coord, label in sorted(candidate_labels.items(), key=lambda item: str(item[1])):
        row, col = coord
        cell_id = coord_to_cell_id(coord)
        bbox_px = tuple(float(value) for value in cell_bboxes_px[str(cell_id)])
        _draw_candidate_label(
            draw,
            bbox_px=bbox_px,
            label=str(label),
            params=params,
        )
        point = [
            round((float(bbox_px[0]) + float(bbox_px[2])) / 2.0, 3),
            round((float(bbox_px[1]) + float(bbox_px[3])) / 2.0, 3),
        ]
        candidate_label_cell_ids[str(label)] = str(cell_id)
        candidate_label_bboxes_px[str(label)] = [float(value) for value in bbox_px]
        candidate_label_points_px[str(label)] = list(point)
        scene_entities.append(
            {
                "entity_id": f"candidate_{str(label)}",
                "entity_type": "battleship_candidate_cell",
                "label": str(label),
                "cell_id": str(cell_id),
                "row": int(row),
                "col": int(col),
                "bbox_px": [float(value) for value in bbox_px],
                "point_px": list(point),
            }
        )
    shape_option_bboxes_px: Dict[str, List[float]] = {}
    if shape_options:
        fleet_icon_bboxes_px, shape_option_bboxes_px = _draw_shape_option_panel(
            draw,
            panel_bbox=panel_bbox,
            shape_options=tuple(shape_options),
            params=params,
            theme=theme,
        )
        for option in shape_options:
            scene_entities.append(
                {
                    "entity_id": f"shape_option_{str(option.label)}",
                    "entity_type": "battleship_shape_option",
                    "label": str(option.label),
                    "shape_id": str(option.shape_id),
                    "display_name": str(option.display_name),
                    "is_answer": bool(option.is_answer),
                    "bbox_px": list(shape_option_bboxes_px[str(option.label)]),
                }
            )
    else:
        fleet_icon_bboxes_px = _draw_fleet_panel(
            draw,
            panel_bbox=panel_bbox,
            params=params,
            theme=theme,
        )
    ship_bboxes_px: Dict[str, List[float]] = {}
    for ship_id, coords in ship_cells_by_id.items():
        cell_ids = [coord_to_cell_id((int(row), int(col))) for row, col in coords]
        ship_bbox = _bbox_union([cell_bboxes_px[str(cell_id)] for cell_id in cell_ids])
        ship_bboxes_px[str(ship_id)] = list(ship_bbox)
        scene_entities.append(
            {
                "entity_id": str(ship_id),
                "entity_type": "battleship_ship",
                "ship_id": str(ship_id),
                "cell_ids": [str(cell_id) for cell_id in cell_ids],
                "bbox_px": list(ship_bbox),
                "is_sunk": bool(str(ship_id) in sunk_ids),
            }
        )

    render_map = {
        "board_bbox_px": list(board_bbox),
        "fleet_panel_bbox_px": list(panel_bbox),
        "scene_panel_bbox_px": None if scene_panel_bbox is None else [int(value) for value in scene_panel_bbox],
        "cell_bboxes_px": dict(cell_bboxes_px),
        "hit_cell_ids": [coord_to_cell_id(coord) for coord in sorted(hits)],
        "miss_cell_ids": [coord_to_cell_id(coord) for coord in sorted(misses)],
        "ship_cell_ids": [coord_to_cell_id(coord) for coord in sorted(ship_cells_by_coord)],
        "sunk_ship_cell_ids": [
            coord_to_cell_id(coord)
            for coord, ship_id in sorted(ship_cells_by_coord.items())
            if str(ship_id) in sunk_ids
        ],
        "ship_cells_by_id": {
            str(ship_id): [coord_to_cell_id(coord) for coord in sorted((int(row), int(col)) for row, col in coords)]
            for ship_id, coords in ship_cells_by_id.items()
        },
        "ship_bboxes_px": dict(ship_bboxes_px),
        "fleet_icon_bboxes_px": dict(fleet_icon_bboxes_px),
        "shape_option_bboxes_px": dict(shape_option_bboxes_px),
        "show_ship_bodies": bool(show_ship_bodies),
        "candidate_label_cell_ids": dict(candidate_label_cell_ids),
        "candidate_label_bboxes_px": dict(candidate_label_bboxes_px),
        "candidate_label_points_px": dict(candidate_label_points_px),
        "layout_jitter": {
            **dict(layout_jitter),
            "jittered_bbox_kind": "board_fleet_group",
            "fleet_panel_dx_px": float(dx),
            "fleet_panel_dy_px": float(dy),
        },
        "style_variant": str(style_variant),
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "font_family": str(params.font_family),
    }
    return RenderedBattleshipScene(
        image=image,
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


def render_battleship_sample(
    *,
    sample: BattleshipSample,
    style_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedBattleshipTaskContext:
    """Render one generated Battleship sample and attach render metadata."""

    render_params = resolve_battleship_render_params(params, instance_seed=int(instance_seed))
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
        namespace="games.battleship_grid.panel_scene_style",
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
    candidate_labels_by_coord = {
        (int(option.coord[0]), int(option.coord[1])): str(option.label)
        for option in sample.candidate_options
    }
    rendered_scene = render_battleship_grid_scene(
        board_size=int(sample.board_size),
        ship_cells_by_id={
            str(ship.ship_id): tuple(ship.coords)
            for ship in sample.ship_placements
        },
        sunk_ship_ids=[
            str(ship.ship_id)
            for ship in sample.ship_placements
            if bool(ship.is_sunk)
        ],
        hit_coords=sample.hit_coords,
        miss_coords=sample.miss_coords,
        background=background,
        style_variant=str(style_variant),
        params=render_params,
        panel_style=panel_style,
        show_ship_bodies=not bool(sample.candidate_options or sample.shape_options),
        candidate_labels_by_coord=candidate_labels_by_coord,
        shape_options=sample.shape_options,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedBattleshipTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
    )


__all__ = [
    "BattleshipCellSpec",
    "BattleshipRenderParams",
    "RenderedBattleshipScene",
    "RenderedBattleshipTaskContext",
    "render_battleship_sample",
    "render_battleship_grid_scene",
    "resolve_battleship_render_params",
]
