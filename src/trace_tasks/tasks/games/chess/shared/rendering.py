"""Rendering helpers for Chess games scene tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Sequence

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.games.shared.layout import attach_games_unit_size_jitter, resolve_games_layout_jitter, resolve_games_unit_size_scale, scale_games_px
from trace_tasks.tasks.games.shared.piece_board_renderer import ChessRenderParams, RenderedChessScene, render_chess_board_scene
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family

from .defaults import FALLBACK_RENDERING_DEFAULTS
from .state import SCENE_ID

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class RenderedChessTaskContext:
    """Rendered Chess image plus trace metadata."""

    image: Image.Image
    rendered_scene: RenderedChessScene
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]


def _render_default(key: str, fallback: Any) -> Any:
    return group_default(_RENDER_DEFAULTS, str(key), FALLBACK_RENDERING_DEFAULTS[str(key)] if str(key) in FALLBACK_RENDERING_DEFAULTS else fallback)


def resolve_chess_render_params(params: Mapping[str, Any], *, instance_seed: int, checkmate: bool = False) -> ChessRenderParams:
    """Resolve Chess render parameters from scene config and params."""

    font_family = sample_font_family(role="readout", instance_seed=int(instance_seed), namespace="games.chess.text_font", params=params)
    unit_scale, unit_meta = resolve_games_unit_size_scale(params, _RENDER_DEFAULTS, instance_seed=int(instance_seed), namespace="games.chess.unit_size")
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(params, _RENDER_DEFAULTS, instance_seed=int(instance_seed), namespace="games.chess.layout"),
        unit_meta,
    )
    max_board_key = "checkmate_max_board_size_px" if bool(checkmate) else "max_board_size_px"
    max_board_size_px = scale_games_px(params.get("max_board_size_px", _render_default(max_board_key, 780)), unit_scale, min_px=390)
    player_badge_height_px = int(params.get("player_badge_height_px", _render_default("player_badge_height_px", 52)))
    player_badge_width_px = int(params.get("player_badge_width_px", _render_default("player_badge_width_px", 270)))
    header_gap_px = int(params.get("header_gap_px", _render_default("header_gap_px", 18)))
    dynamic_canvas = bool(params.get("dynamic_canvas_size_enabled", _render_default("dynamic_canvas_size_enabled", True)))
    base_canvas_width = int(params.get("canvas_width", _render_default("canvas_width", 980)))
    base_canvas_height = int(params.get("canvas_height", _render_default("canvas_height", 920)))
    option_panel_height = int(params.get("option_panel_height_px", _render_default("checkmate_option_panel_height_px" if checkmate else "option_panel_height_px", 154 if checkmate else 0)))
    option_panel_gap = int(params.get("option_panel_gap_px", _render_default("checkmate_option_panel_gap_px" if checkmate else "option_panel_gap_px", 46 if checkmate else 18)))
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", _render_default("canvas_min_width_px", 560))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_side_padding_px", _render_default("canvas_side_padding_px", 132)))) + (120 if checkmate else 0))),
            ),
        )
    if dynamic_canvas and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", _render_default("canvas_min_height_px", 560))),
                int(round(float(max_board_size_px) + float(player_badge_height_px) + float(header_gap_px) + float(option_panel_height) + float(option_panel_gap) + (2.0 * float(params.get("canvas_vertical_padding_px", _render_default("canvas_vertical_padding_px", 92)))))),
            ),
        )
    return ChessRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", _render_default("panel_margin_px", 48))),
        player_badge_height_px=int(player_badge_height_px),
        player_badge_width_px=int(player_badge_width_px),
        header_gap_px=int(header_gap_px),
        max_board_size_px=int(max_board_size_px),
        board_corner_radius_px=scale_games_px(params.get("board_corner_radius_px", _render_default("board_corner_radius_px", 26)), unit_scale, min_px=10),
        board_frame_width_px=scale_games_px(params.get("board_frame_width_px", _render_default("checkmate_board_frame_width_px" if checkmate else "board_frame_width_px", 8 if checkmate else 10)), unit_scale, min_px=5),
        piece_inset_fraction=float(params.get("piece_inset_fraction", _render_default("piece_inset_fraction", 0.12))),
        piece_font_size_px=scale_games_px(params.get("piece_font_size_px", _render_default("piece_font_size_px", 78)), unit_scale, min_px=36),
        marked_square_outline_width_px=scale_games_px(params.get("marked_square_outline_width_px", _render_default("marked_square_outline_width_px", 7)), unit_scale, min_px=3),
        player_badge_font_size_px=int(params.get("player_badge_font_size_px", _render_default("player_badge_font_size_px", 22))),
        coordinate_label_font_size_px=int(params.get("coordinate_label_font_size_px", _render_default("checkmate_coordinate_label_font_size_px" if checkmate else "coordinate_label_font_size_px", 18))),
        option_panel_gap_px=scale_games_px(option_panel_gap, unit_scale, min_px=8),
        option_panel_height_px=scale_games_px(option_panel_height, unit_scale, min_px=0),
        option_panel_font_size_px=int(params.get("option_panel_font_size_px", _render_default("checkmate_option_panel_font_size_px" if checkmate else "option_panel_font_size_px", 18 if checkmate else 20))),
        layout_jitter_meta=layout_jitter,
        font_family=str(font_family),
        instance_seed=int(instance_seed),
    )


def render_chess_task_scene(
    *,
    board,
    scene_variant: str,
    style_variant: str,
    badge_text: str,
    marked_coord,
    target_coord=None,
    params: Mapping[str, Any],
    instance_seed: int,
    show_coordinates: bool = False,
    move_options: Sequence[Mapping[str, Any]] = (),
) -> RenderedChessTaskContext:
    """Render a Chess board scene with shared panel styling and noise."""

    render_params = resolve_chess_render_params(params, instance_seed=int(instance_seed), checkmate=bool(move_options))
    if move_options:
        render_params = replace(
            render_params,
            option_panel_height_px=max(132, int(render_params.option_panel_height_px)),
            option_panel_gap_px=max(42, int(render_params.option_panel_gap_px)),
            option_panel_font_size_px=max(16, int(render_params.option_panel_font_size_px)),
            coordinate_label_font_size_px=max(14, int(render_params.coordinate_label_font_size_px)),
        )
    allowed_raw = params.get("panel_scene_treatments", group_default(_RENDER_DEFAULTS, "panel_scene_treatments", None))
    if isinstance(allowed_raw, str):
        allowed = (str(allowed_raw),)
    elif allowed_raw is None:
        allowed = None
    else:
        allowed = tuple(str(item) for item in allowed_raw)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.chess.panel_scene_style",
        treatments=allowed,
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(_RENDER_DEFAULTS, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(_RENDER_DEFAULTS, "panel_scene_palette_weights", None)),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_chess_board_scene(
        board=board,
        background=background,
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        badge_text=str(badge_text),
        marked_coord=marked_coord,
        params=render_params,
        target_coord=target_coord,
        panel_style=panel_style,
        show_coordinates=bool(show_coordinates),
        move_options=tuple(move_options),
    )
    image, post_noise_meta = apply_post_image_noise(rendered_scene.image, instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_NOISE_DEFAULTS)
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
        "piece_symbol_font_family": "system_fallback",
    }
    return RenderedChessTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=text_style_meta,
    )

__all__ = ["RenderedChessTaskContext", "render_chess_task_scene", "resolve_chess_render_params"]
