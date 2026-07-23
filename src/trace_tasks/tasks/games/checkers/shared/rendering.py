"""Shared Checkers board renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from ....shared.marker_legibility import draw_semantic_bbox_marker, resolve_semantic_marker_style
from ....shared.text_rendering import load_font, resolve_text_stroke_fill
from ...shared.text import draw_game_text_traced as draw_text_traced
from .rules import BLACK, BOARD_SIZE, RED, Coord, coord_to_cell_id, piece_to_entity_id, player_name
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
from ...shared.style import CheckersTheme, build_games_checkers_theme
from ...shared.visual_defaults import load_games_scene_noise_defaults
from .defaults import FALLBACK_RENDERING_DEFAULTS
from .state import SCENE_ID, ResolvedCheckersSceneAxes, SampledCheckersScene


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def _adjust_rgb(rgb: Sequence[int], delta: int) -> Tuple[int, int, int]:
    """Return an RGB color lightened/darkened by a small channel delta."""

    return tuple(max(0, min(255, int(value) + int(delta))) for value in rgb[:3])


def _inset_square_rgb(rgb: Sequence[int]) -> Tuple[int, int, int]:
    """Return a subtle inner-square shade for inset board styles."""

    brightness = sum(int(value) for value in rgb[:3]) / 3.0
    return _adjust_rgb(rgb, 10 if brightness < 158.0 else -7)


@dataclass(frozen=True)
class CheckersRenderParams:
    """Resolved render controls for one Checkers scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    player_badge_height_px: int
    player_badge_width_px: int
    header_gap_px: int
    max_board_size_px: int
    board_corner_radius_px: int
    board_frame_width_px: int
    piece_inset_fraction: float
    player_badge_font_size_px: int
    layout_jitter_meta: Dict[str, Any] | None = None
    font_family: str = ""
    instance_seed: int = 0


@dataclass(frozen=True)
class CheckersCellSpec:
    """One board cell after layout/render assignment."""

    cell_id: str
    row: int
    col: int
    playable: bool
    occupant: str
    bbox_px: Tuple[float, float, float, float]
    piece_bbox_px: Tuple[float, float, float, float] | None


@dataclass(frozen=True)
class RenderedCheckersScene:
    """Rendered Checkers scene plus trace-friendly metadata."""

    image: Image.Image
    cell_specs: Tuple[CheckersCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedCheckersTaskContext:
    """Rendered Checkers task image plus trace-friendly render context."""

    image: Image.Image
    rendered_scene: RenderedCheckersScene
    panel_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


def resolve_checkers_render_params(params: Mapping[str, Any], *, instance_seed: int) -> CheckersRenderParams:
    """Resolve Checkers rendering parameters from scene config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.checkers.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.checkers.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.checkers.layout",
        ),
        unit_scale_meta,
    )
    max_board_size_px = scale_games_px(
        params.get(
            "max_board_size_px",
            group_default(_RENDER_DEFAULTS, "max_board_size_px", FALLBACK_RENDERING_DEFAULTS["max_board_size_px"]),
        ),
        unit_scale,
        min_px=390,
    )
    player_badge_height_px = int(
        params.get(
            "player_badge_height_px",
            group_default(_RENDER_DEFAULTS, "player_badge_height_px", FALLBACK_RENDERING_DEFAULTS["player_badge_height_px"]),
        )
    )
    player_badge_width_px = int(
        params.get(
            "player_badge_width_px",
            group_default(_RENDER_DEFAULTS, "player_badge_width_px", FALLBACK_RENDERING_DEFAULTS["player_badge_width_px"]),
        )
    )
    header_gap_px = int(
        params.get(
            "header_gap_px",
            group_default(_RENDER_DEFAULTS, "header_gap_px", FALLBACK_RENDERING_DEFAULTS["header_gap_px"]),
        )
    )
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(
                _RENDER_DEFAULTS,
                "dynamic_canvas_size_enabled",
                FALLBACK_RENDERING_DEFAULTS["dynamic_canvas_size_enabled"],
            ),
        )
    )
    base_canvas_width = int(
        params.get(
            "canvas_width",
            group_default(_RENDER_DEFAULTS, "canvas_width", FALLBACK_RENDERING_DEFAULTS["canvas_width"]),
        )
    )
    base_canvas_height = int(
        params.get(
            "canvas_height",
            group_default(_RENDER_DEFAULTS, "canvas_height", FALLBACK_RENDERING_DEFAULTS["canvas_height"]),
        )
    )
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(
                    params.get(
                        "canvas_min_width_px",
                        group_default(
                            _RENDER_DEFAULTS,
                            "canvas_min_width_px",
                            FALLBACK_RENDERING_DEFAULTS["canvas_min_width_px"],
                        ),
                    )
                ),
                int(
                    round(
                        float(max_board_size_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_side_padding_px",
                                    group_default(
                                        _RENDER_DEFAULTS,
                                        "canvas_side_padding_px",
                                        FALLBACK_RENDERING_DEFAULTS["canvas_side_padding_px"],
                                    ),
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
                int(
                    params.get(
                        "canvas_min_height_px",
                        group_default(
                            _RENDER_DEFAULTS,
                            "canvas_min_height_px",
                            FALLBACK_RENDERING_DEFAULTS["canvas_min_height_px"],
                        ),
                    )
                ),
                int(
                    round(
                        float(max_board_size_px)
                        + float(player_badge_height_px)
                        + float(header_gap_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_vertical_padding_px",
                                    group_default(
                                        _RENDER_DEFAULTS,
                                        "canvas_vertical_padding_px",
                                        FALLBACK_RENDERING_DEFAULTS["canvas_vertical_padding_px"],
                                    ),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    return CheckersRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(
            params.get(
                "panel_margin_px",
                group_default(_RENDER_DEFAULTS, "panel_margin_px", FALLBACK_RENDERING_DEFAULTS["panel_margin_px"]),
            )
        ),
        player_badge_height_px=int(player_badge_height_px),
        player_badge_width_px=int(player_badge_width_px),
        header_gap_px=int(header_gap_px),
        max_board_size_px=int(max_board_size_px),
        board_corner_radius_px=scale_games_px(
            params.get(
                "board_corner_radius_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "board_corner_radius_px",
                    FALLBACK_RENDERING_DEFAULTS["board_corner_radius_px"],
                ),
            ),
            unit_scale,
            min_px=10,
        ),
        board_frame_width_px=scale_games_px(
            params.get(
                "board_frame_width_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "board_frame_width_px",
                    FALLBACK_RENDERING_DEFAULTS["board_frame_width_px"],
                ),
            ),
            unit_scale,
            min_px=5,
        ),
        piece_inset_fraction=float(
            params.get(
                "piece_inset_fraction",
                group_default(
                    _RENDER_DEFAULTS,
                    "piece_inset_fraction",
                    FALLBACK_RENDERING_DEFAULTS["piece_inset_fraction"],
                ),
            )
        ),
        player_badge_font_size_px=int(
            params.get(
                "player_badge_font_size_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "player_badge_font_size_px",
                    FALLBACK_RENDERING_DEFAULTS["player_badge_font_size_px"],
                ),
            )
        ),
        layout_jitter_meta=layout_jitter,
        font_family=str(font_family),
        instance_seed=int(instance_seed),
    )


def render_checkers_task_scene(
    *,
    axes: ResolvedCheckersSceneAxes,
    sample: SampledCheckersScene,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedCheckersTaskContext:
    """Render one generated Checkers sample with shared scene styling."""

    render_params = resolve_checkers_render_params(params, instance_seed=int(instance_seed))
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
        namespace="games.checkers.panel_scene_style",
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
    rendered_scene = render_checkers_board_scene(
        board=sample.board,
        background=background,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        current_player=int(sample.current_player),
        params=render_params,
        marked_coord=sample.evaluation.marked_coord,
        king_coords=() if sample.evaluation.marked_coord is None else (sample.evaluation.marked_coord,),
        panel_style=panel_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    return RenderedCheckersTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        text_style_meta=dict(text_style_meta),
    )


def _piece_bbox(
    cell_bbox: Tuple[float, float, float, float],
    *,
    inset_fraction: float,
) -> Tuple[float, float, float, float]:
    """Return one inscribed checker-piece bbox inside one board cell."""

    left, top, right, bottom = cell_bbox
    inset = float(max(4.0, inset_fraction * min(right - left, bottom - top)))
    return (
        round(float(left + inset), 3),
        round(float(top + inset), 3),
        round(float(right - inset), 3),
        round(float(bottom - inset), 3),
    )


def _draw_piece(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    theme: CheckersTheme,
    player: int,
    is_king: bool = False,
    font_family: str = "",
) -> None:
    """Draw one ordinary checker piece with simple inner-ring chrome."""

    if int(player) == int(RED):
        fill_rgb = theme.red_piece_fill_rgb
        outline_rgb = theme.red_piece_outline_rgb
        shine_rgb = theme.red_piece_shine_rgb
    else:
        fill_rgb = theme.black_piece_fill_rgb
        outline_rgb = theme.black_piece_outline_rgb
        shine_rgb = theme.black_piece_shine_rgb
    left, top, right, bottom = bbox_px
    if int(theme.piece_shadow_alpha) > 0:
        shadow_offset = max(1, int(round(0.045 * min(right - left, bottom - top))))
        draw.ellipse(
            [left + shadow_offset, top + shadow_offset, right + shadow_offset, bottom + shadow_offset],
            fill=tuple(int(value) for value in theme.piece_shadow_rgb) + (int(theme.piece_shadow_alpha),),
        )
    draw.ellipse(
        bbox_px,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=int(theme.piece_outline_width_px),
    )
    if str(theme.piece_rendering) == "flat":
        shine_radius = max(3.0, 0.13 * (right - left))
        draw.ellipse(
            [
                left + 0.23 * (right - left),
                top + 0.18 * (bottom - top),
                left + 0.23 * (right - left) + shine_radius,
                top + 0.18 * (bottom - top) + shine_radius,
            ],
            fill=tuple(int(value) for value in shine_rgb),
        )
    else:
        inner_inset_x = 0.16 * (right - left)
        inner_inset_y = 0.16 * (bottom - top)
        draw.ellipse(
            [
                left + inner_inset_x,
                top + inner_inset_y,
                right - inner_inset_x,
                bottom - inner_inset_y,
            ],
            outline=tuple(int(value) for value in shine_rgb),
            width=max(2, int(0.08 * (right - left))),
        )
        if str(theme.piece_rendering) == "double_ring":
            second_inset_x = 0.30 * (right - left)
            second_inset_y = 0.30 * (bottom - top)
            draw.ellipse(
                [
                    left + second_inset_x,
                    top + second_inset_y,
                    right - second_inset_x,
                    bottom - second_inset_y,
                ],
                outline=tuple(int(value) for value in shine_rgb),
                width=max(1, int(0.04 * (right - left))),
            )
    if bool(is_king):
        font = load_font(
            max(14, int(0.42 * min(right - left, bottom - top))),
            bold=True,
            font_family=str(font_family) or None,
        )
        text = "K"
        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width = float(text_bbox[2] - text_bbox[0])
        text_height = float(text_bbox[3] - text_bbox[1])
        text_rgb = tuple(int(value) for value in shine_rgb)
        draw_text_traced(draw,
            (
                float(left + (0.5 * ((right - left) - text_width)) - text_bbox[0]),
                float(top + (0.5 * ((bottom - top) - text_height)) - text_bbox[1]),
            ),
            text,
            font=font,
            fill=text_rgb,
            stroke_width=1,
            stroke_fill=tuple(int(value) for value in outline_rgb),
         role="readout", required=False,)


def render_checkers_board_scene(
    *,
    board: Sequence[Sequence[int]],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    current_player: int,
    params: CheckersRenderParams,
    marked_coord: Coord | None = None,
    king_coords: Sequence[Coord] = (),
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedCheckersScene:
    """Render one visible Checkers board state."""

    del scene_variant
    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    theme = build_games_checkers_theme(style_variant=str(style_variant))
    marked_cell = None if marked_coord is None else (int(marked_coord[0]), int(marked_coord[1]))
    king_coord_set = {(int(row), int(col)) for row, col in king_coords}

    cell_size = min(
        int(params.max_board_size_px) // BOARD_SIZE,
        (int(params.canvas_width) - (2 * int(params.panel_margin_px))) // BOARD_SIZE,
        (
            int(params.canvas_height)
            - (2 * int(params.panel_margin_px))
            - int(params.player_badge_height_px)
            - int(params.header_gap_px)
        )
        // BOARD_SIZE,
    )
    board_size_px = int(cell_size) * BOARD_SIZE
    board_left = int(0.5 * (int(params.canvas_width) - int(board_size_px)))
    available_height = (
        int(params.canvas_height)
        - (2 * int(params.panel_margin_px))
        - int(params.player_badge_height_px)
        - int(params.header_gap_px)
    )
    board_top = int(
        params.panel_margin_px
        + params.player_badge_height_px
        + params.header_gap_px
        + max(0, 0.5 * (available_height - int(board_size_px)))
    )
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_size_px), 3),
        round(float(board_top + board_size_px), 3),
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
        max(float(board_bbox[3]), float(badge_bbox[3])),
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

    scene_panel_bbox: Tuple[int, int, int, int] | None = None
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

    draw.rounded_rectangle(
        board_bbox,
        radius=int(params.board_corner_radius_px),
        fill=tuple(int(value) for value in theme.board_frame_rgb),
    )
    draw.rounded_rectangle(
        badge_bbox,
        radius=int(0.5 * int(params.player_badge_height_px)),
        fill=tuple(int(value) for value in theme.badge_fill_rgb),
        outline=tuple(int(value) for value in theme.badge_outline_rgb),
        width=2,
    )
    sample_piece_d = int(params.player_badge_height_px) - 16
    sample_piece_left = int(badge_left + 12)
    sample_piece_top = int(badge_top + 8)
    _draw_piece(
        draw,
        bbox_px=(
            float(sample_piece_left),
            float(sample_piece_top),
            float(sample_piece_left + sample_piece_d),
            float(sample_piece_top + sample_piece_d),
        ),
        theme=theme,
        player=int(current_player),
        font_family=str(params.font_family),
    )
    badge_text_rgb = tuple(int(value) for value in theme.badge_text_rgb)
    draw_text_traced(draw,
        (
            float(sample_piece_left + sample_piece_d + 12),
            float(badge_top + 0.5 * (int(params.player_badge_height_px) - (badge_text_bbox[3] - badge_text_bbox[1]))),
        ),
        badge_text,
        font=badge_font,
        fill=badge_text_rgb,
        stroke_width=1,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(badge_text_rgb)),
     role="readout", required=False,)

    cell_specs: List[CheckersCellSpec] = []
    scene_entities: List[Dict[str, Any]] = []
    cell_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    piece_bboxes_px: Dict[str, Tuple[float, float, float, float]] = {}
    marked_cell_bbox: Tuple[float, float, float, float] | None = None
    marked_surface_rgb: Tuple[int, int, int] | None = None

    inner_inset = float(params.board_frame_width_px)
    playable_square_count = 0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            left = float(board_left + (col * cell_size) + inner_inset)
            top = float(board_top + (row * cell_size) + inner_inset)
            right = float(board_left + ((col + 1) * cell_size) - inner_inset)
            bottom = float(board_top + ((row + 1) * cell_size) - inner_inset)
            cell_bbox = (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))
            playable = (row + col) % 2 == 1
            square_rgb = tuple(int(value) for value in (theme.dark_square_rgb if playable else theme.light_square_rgb))
            draw.rectangle(
                cell_bbox,
                fill=square_rgb,
                outline=tuple(int(value) for value in theme.grid_line_rgb),
                width=int(theme.grid_line_width_px),
            )
            if str(theme.square_rendering) == "inset":
                inset = max(2.0, 0.045 * min(cell_bbox[2] - cell_bbox[0], cell_bbox[3] - cell_bbox[1]))
                draw.rectangle(
                    [
                        cell_bbox[0] + inset,
                        cell_bbox[1] + inset,
                        cell_bbox[2] - inset,
                        cell_bbox[3] - inset,
                    ],
                    fill=_inset_square_rgb(square_rgb),
                )
            cell_id = coord_to_cell_id((int(row), int(col)))
            occupant_value = int(board[row][col])
            occupant = "red" if occupant_value == int(RED) else "black" if occupant_value == int(BLACK) else "empty"
            piece_bbox_px: Tuple[float, float, float, float] | None = None
            if occupant_value != 0:
                piece_bbox_px = _piece_bbox(cell_bbox, inset_fraction=float(params.piece_inset_fraction))
                is_king = (int(row), int(col)) in king_coord_set
                _draw_piece(
                    draw,
                    bbox_px=piece_bbox_px,
                    theme=theme,
                    player=occupant_value,
                    is_king=bool(is_king),
                    font_family=str(params.font_family),
                )
                piece_entity_id = piece_to_entity_id((int(row), int(col)), player=occupant_value)
                piece_bboxes_px[str(piece_entity_id)] = piece_bbox_px
                scene_entities.append(
                    {
                        "id": str(piece_entity_id),
                        "kind": "checker_piece",
                        "player": str(occupant),
                        "cell_id": str(cell_id),
                        "king": bool(is_king),
                        "bbox_px": list(piece_bbox_px),
                    }
                )
            if marked_cell is not None and (int(row), int(col)) == marked_cell:
                marked_cell_bbox = cell_bbox
                marked_surface_rgb = square_rgb
            cell_bboxes_px[str(cell_id)] = cell_bbox
            scene_entities.append(
                {
                    "id": str(cell_id),
                    "kind": "board_cell",
                    "row": int(row),
                    "col": int(col),
                    "playable": bool(playable),
                    "occupant": str(occupant),
                    "marked": bool(marked_cell is not None and (int(row), int(col)) == marked_cell),
                    "bbox_px": list(cell_bbox),
                }
            )
            cell_specs.append(
                CheckersCellSpec(
                    cell_id=str(cell_id),
                    row=int(row),
                    col=int(col),
                    playable=bool(playable),
                    occupant=str(occupant),
                    bbox_px=cell_bbox,
                    piece_bbox_px=piece_bbox_px,
                )
            )
            if playable:
                playable_square_count += 1

    if marked_cell_bbox is not None:
        inset = max(3.0, 0.06 * min(marked_cell_bbox[2] - marked_cell_bbox[0], marked_cell_bbox[3] - marked_cell_bbox[1]))
        marker_bbox = [
            marked_cell_bbox[0] + inset,
            marked_cell_bbox[1] + inset,
            marked_cell_bbox[2] - inset,
            marked_cell_bbox[3] - inset,
        ]
        marker_style = resolve_semantic_marker_style(
            instance_seed=int(params.instance_seed),
            namespace="games.checkers.marked_cell",
            role="marked_cell_outline",
            surface_rgbs=(marked_surface_rgb or theme.dark_square_rgb,),
            preferred_rgbs=((34, 102, 214),),
        )
        draw_semantic_bbox_marker(
            draw,
            marker_bbox,
            radius=max(5, int(0.10 * min(marked_cell_bbox[2] - marked_cell_bbox[0], marked_cell_bbox[3] - marked_cell_bbox[1]))),
            style=marker_style,
            width=7,
            marker_kind="cell_outline",
            extra_metadata={"source": "games_checkers_marked_cell"},
        )

    render_map = {
        "board_bbox_px": list(board_bbox),
        "scene_panel_bbox_px": None if scene_panel_bbox is None else [int(value) for value in scene_panel_bbox],
        "cell_bboxes_px": {str(key): list(value) for key, value in cell_bboxes_px.items()},
        "piece_bboxes_px": {str(key): list(value) for key, value in piece_bboxes_px.items()},
        "marked_cell_id": None if marked_cell is None else coord_to_cell_id(marked_cell),
        "king_piece_ids": [
            piece_to_entity_id((int(row), int(col)), player=int(board[int(row)][int(col)]))
            for row, col in sorted(king_coord_set)
            if int(board[int(row)][int(col)]) != 0
        ],
        "player_badge_bbox_px": list(badge_bbox),
        "playable_square_count": int(playable_square_count),
        "effective_cell_size_px": float(cell_size),
        "layout_jitter": dict(layout_jitter),
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "font_family": str(params.font_family),
    }
    return RenderedCheckersScene(
        image=image,
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "CheckersCellSpec",
    "CheckersRenderParams",
    "RenderedCheckersTaskContext",
    "RenderedCheckersScene",
    "render_checkers_task_scene",
    "render_checkers_board_scene",
    "resolve_checkers_render_params",
]
