"""Shared nine-men's-morris board renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family

from ...shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.style import NineMensMorrisTheme, build_games_nine_mens_morris_theme
from .state import POSITION_LAYOUT, NineMensMorrisBoardState


@dataclass(frozen=True)
class NineMensMorrisRenderParams:
    """Resolved render controls for one nine-men's-morris scene."""

    canvas_width: int
    canvas_height: int
    board_width_px: int
    board_height_px: int
    board_corner_radius_px: int
    panel_margin_px: int
    title_font_size_px: int
    title_band_height_px: int
    board_padding_px: int
    piece_radius_px: int
    node_radius_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class _RenderFallbacks:
    """Stable fallback defaults for Nine Men's Morris rendering."""

    canvas_width: int = 1180
    canvas_height: int = 820
    board_width_px: int = 860
    board_height_px: int = 660
    board_corner_radius_px: int = 24
    panel_margin_px: int = 56
    title_font_size_px: int = 34
    title_band_height_px: int = 62
    board_padding_px: int = 72
    piece_radius_px: int = 22
    node_radius_px: int = 5


_FALLBACKS = _RenderFallbacks()


@dataclass(frozen=True)
class RenderedNineMensMorrisPieceSpec:
    """One rendered nine-men's-morris piece."""

    piece_id: str
    node_index: int
    node_label: str
    color: str
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedNineMensMorrisScene:
    """Rendered nine-men's-morris scene plus trace-friendly metadata."""

    image: Image.Image
    piece_specs: Tuple[RenderedNineMensMorrisPieceSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def resolve_nine_mens_morris_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> NineMensMorrisRenderParams:
    """Resolve stable render parameters for one Morris scene."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.unit_size",
        fallback_min=0.55,
        fallback_max=1.10,
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
    board_width_px = scale_games_px(
        params.get("board_width_px", group_default(render_defaults, "board_width_px", _FALLBACKS.board_width_px)),
        unit_scale,
        min_px=470,
    )
    board_height_px = scale_games_px(
        params.get("board_height_px", group_default(render_defaults, "board_height_px", _FALLBACKS.board_height_px)),
        unit_scale,
        min_px=360,
    )
    default_canvas_width = int(group_default(render_defaults, "canvas_width", _FALLBACKS.canvas_width))
    default_canvas_height = int(group_default(render_defaults, "canvas_height", _FALLBACKS.canvas_height))
    canvas_width = int(max(620, min(default_canvas_width, int(board_width_px) + 250)))
    canvas_height = int(max(500, min(default_canvas_height, int(board_height_px) + 190)))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font_family",
        params=params,
    )
    return NineMensMorrisRenderParams(
        canvas_width=int(params.get("canvas_width", canvas_width)),
        canvas_height=int(params.get("canvas_height", canvas_height)),
        board_width_px=int(board_width_px),
        board_height_px=int(board_height_px),
        board_corner_radius_px=scale_games_px(
            params.get(
                "board_corner_radius_px",
                group_default(render_defaults, "board_corner_radius_px", _FALLBACKS.board_corner_radius_px),
            ),
            unit_scale,
            min_px=12,
        ),
        panel_margin_px=scale_games_px(
            params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", _FALLBACKS.panel_margin_px)),
            unit_scale,
            min_px=30,
        ),
        title_font_size_px=scale_games_px(
            params.get("title_font_size_px", group_default(render_defaults, "title_font_size_px", _FALLBACKS.title_font_size_px)),
            unit_scale,
            min_px=18,
        ),
        title_band_height_px=scale_games_px(
            params.get(
                "title_band_height_px",
                group_default(render_defaults, "title_band_height_px", _FALLBACKS.title_band_height_px),
            ),
            unit_scale,
            min_px=38,
        ),
        board_padding_px=scale_games_px(
            params.get("board_padding_px", group_default(render_defaults, "board_padding_px", _FALLBACKS.board_padding_px)),
            unit_scale,
            min_px=38,
        ),
        piece_radius_px=scale_games_px(
            params.get("piece_radius_px", group_default(render_defaults, "piece_radius_px", _FALLBACKS.piece_radius_px)),
            unit_scale,
            min_px=13,
        ),
        node_radius_px=scale_games_px(
            params.get("node_radius_px", group_default(render_defaults, "node_radius_px", _FALLBACKS.node_radius_px)),
            unit_scale,
            min_px=3,
        ),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


def _draw_shadow(
    image: Image.Image,
    *,
    bbox_px: Tuple[float, float, float, float],
    radius_px: int,
    theme: NineMensMorrisTheme,
) -> None:
    """Draw one soft panel shadow for the board."""

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = bbox_px
    dx, dy = theme.shadow_offset_px
    draw.rounded_rectangle(
        [left + dx, top + dy, right + dx, bottom + dy],
        radius=int(radius_px),
        fill=(
            int(theme.shadow_rgb[0]),
            int(theme.shadow_rgb[1]),
            int(theme.shadow_rgb[2]),
            int(theme.shadow_alpha),
        ),
    )
    image.alpha_composite(overlay)


def _node_xy(
    *,
    node_index: int,
    board_left: float,
    board_top: float,
    board_size_px: float,
) -> Tuple[float, float]:
    """Return the pixel center for one Morris board node."""

    _, x_frac, y_frac = POSITION_LAYOUT[int(node_index)]
    return (
        round(float(board_left + (float(x_frac) * board_size_px)), 3),
        round(float(board_top + (float(y_frac) * board_size_px)), 3),
    )


def render_nine_mens_morris_scene(
    *,
    board_state: NineMensMorrisBoardState,
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    params: NineMensMorrisRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedNineMensMorrisScene:
    """Render one Morris board from state while preserving node projections.

    The node-coordinate layout is the single source for both visible pieces and
    `render_map` centers, so annotation projection stays aligned with the
    rendered board after panel jitter and unit-size scaling.
    """

    if str(scene_variant) != "single_board":
        raise ValueError(f"unsupported nine-men's-morris scene_variant: {scene_variant}")

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image)
    theme = build_games_nine_mens_morris_theme(style_variant=str(style_variant))

    board_left = float((int(params.canvas_width) - int(params.board_width_px)) / 2)
    board_top = float((int(params.canvas_height) - int(params.board_height_px)) / 2)
    board_right = float(board_left + int(params.board_width_px))
    board_bottom = float(board_top + int(params.board_height_px))
    panel_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=(board_left, board_top, board_right, board_bottom),
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left, board_top, board_right, board_bottom = [float(value) for value in panel_bbox]

    if panel_style is not None:
        chrome_pad_x = max(14, int(round(float(params.panel_margin_px) * 0.34)))
        chrome_pad_y = max(14, int(round(float(params.panel_margin_px) * 0.28)))
        chrome_bbox = (
            max(4, int(round(board_left)) - chrome_pad_x),
            max(4, int(round(board_top)) - chrome_pad_y),
            min(int(params.canvas_width) - 4, int(round(board_right)) + chrome_pad_x),
            min(int(params.canvas_height) - 4, int(round(board_bottom)) + chrome_pad_y),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=chrome_bbox,
            style=panel_style,
            radius=int(params.board_corner_radius_px) + 8,
            border_width=max(1, min(3, int(theme.board_border_width_px))),
        )

    _draw_shadow(
        image,
        bbox_px=panel_bbox,
        radius_px=int(params.board_corner_radius_px),
        theme=theme,
    )
    draw.rounded_rectangle(
        panel_bbox,
        radius=int(params.board_corner_radius_px),
        fill=tuple(int(value) for value in theme.board_fill_rgb),
        outline=tuple(int(value) for value in theme.board_border_rgb),
        width=int(theme.board_border_width_px),
    )

    inner_left = float(board_left + int(params.board_padding_px))
    inner_top = float(board_top + int(params.board_padding_px))
    inner_right = float(board_right - int(params.board_padding_px))
    inner_bottom = float(board_bottom - int(params.board_padding_px))
    board_size_px = float(min(inner_right - inner_left, inner_bottom - inner_top))
    board_left_px = float(inner_left + ((inner_right - inner_left - board_size_px) / 2.0))
    board_top_px = float(inner_top + ((inner_bottom - inner_top - board_size_px) / 2.0))

    def p(node_index: int) -> Tuple[float, float]:
        return _node_xy(node_index=int(node_index), board_left=board_left_px, board_top=board_top_px, board_size_px=board_size_px)

    line_rgb = tuple(int(value) for value in theme.line_rgb)
    line_width = int(theme.line_width_px)

    draw.rectangle([p(0), p(23)], outline=line_rgb, width=line_width)
    draw.rectangle([p(3), p(20)], outline=line_rgb, width=line_width)
    draw.rectangle([p(6), p(17)], outline=line_rgb, width=line_width)
    draw.line([p(1), p(7)], fill=line_rgb, width=line_width)
    draw.line([p(16), p(22)], fill=line_rgb, width=line_width)
    draw.line([p(9), p(11)], fill=line_rgb, width=line_width)
    draw.line([p(12), p(14)], fill=line_rgb, width=line_width)

    node_centers_px: Dict[str, List[float]] = {}
    for node_index, (node_label, _, _) in enumerate(POSITION_LAYOUT):
        cx, cy = p(node_index)
        node_centers_px[str(node_label)] = [float(cx), float(cy)]
        radius = float(params.node_radius_px)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=tuple(int(value) for value in theme.node_rgb),
        )

    piece_specs: List[RenderedNineMensMorrisPieceSpec] = []
    piece_bboxes_px: Dict[str, List[float]] = {}
    piece_centers_px: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    for piece in board_state.piece_specs:
        cx, cy = p(int(piece.node_index))
        radius = float(params.piece_radius_px)
        bbox = (
            round(float(cx - radius), 3),
            round(float(cy - radius), 3),
            round(float(cx + radius), 3),
            round(float(cy + radius), 3),
        )
        if str(piece.color) == "white":
            fill_rgb = tuple(int(value) for value in theme.white_piece_fill_rgb)
            outline_rgb = tuple(int(value) for value in theme.white_piece_outline_rgb)
        else:
            fill_rgb = tuple(int(value) for value in theme.black_piece_fill_rgb)
            outline_rgb = tuple(int(value) for value in theme.black_piece_outline_rgb)
        draw.ellipse(bbox, fill=fill_rgb, outline=outline_rgb, width=3)
        piece_specs.append(
            RenderedNineMensMorrisPieceSpec(
                piece_id=str(piece.piece_id),
                node_index=int(piece.node_index),
                node_label=str(piece.node_label),
                color=str(piece.color),
                bbox_px=bbox,
            )
        )
        piece_bboxes_px[str(piece.piece_id)] = [float(value) for value in bbox]
        piece_centers_px[str(piece.piece_id)] = [round(float(cx), 3), round(float(cy), 3)]
        scene_entities.append(
            {
                "entity_id": str(piece.piece_id),
                "kind": "nine_mens_morris_piece",
                "bbox": [float(value) for value in bbox],
                "point": list(piece_centers_px[str(piece.piece_id)]),
                "node_index": int(piece.node_index),
                "node_label": str(piece.node_label),
                "color": str(piece.color),
            }
        )

    return RenderedNineMensMorrisScene(
        image=image.convert("RGB"),
        piece_specs=tuple(piece_specs),
        scene_entities=tuple(scene_entities),
        render_map={
            "board_bbox_px": [float(value) for value in panel_bbox],
            "piece_bboxes_px": piece_bboxes_px,
            "piece_centers_px": piece_centers_px,
            "node_centers_px": node_centers_px,
            "layout_jitter": dict(layout_jitter),
            "style_variant": str(style_variant),
            "font_family": str(params.font_family),
            "text_style": {
                "font_family": str(params.font_family),
            },
            "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        },
    )


__all__ = [
    "NineMensMorrisRenderParams",
    "RenderedNineMensMorrisPieceSpec",
    "RenderedNineMensMorrisScene",
    "render_nine_mens_morris_scene",
    "resolve_nine_mens_morris_render_params",
]
