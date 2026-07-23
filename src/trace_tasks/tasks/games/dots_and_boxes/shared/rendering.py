"""Shared dots-and-boxes board renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record

from ...shared.text import draw_game_text_traced as draw_text_traced
from ...shared.layout import apply_games_layout_jitter_to_bbox
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from ...shared.style import DotsAndBoxesTheme, build_games_dots_and_boxes_theme
from .defaults import DOTS_AND_BOXES_NAMESPACE, SCENE_ID
from .state import DotsAndBoxesBoardState, DotsAndBoxesBoxInstance, DotsAndBoxesEdgeInstance


POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


@dataclass(frozen=True)
class DotsAndBoxesRenderParams:
    """Resolved render controls for one dots-and-boxes board scene."""

    canvas_width: int
    canvas_height: int
    board_width_px: int
    board_height_px: int
    board_corner_radius_px: int
    panel_margin_px: int
    title_font_size_px: int
    title_band_height_px: int
    board_padding_px: int
    dot_radius_px: int
    dash_length_px: int
    dash_gap_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RenderedDotsAndBoxesBoxSpec:
    """One rendered dots-and-boxes box region."""

    box_id: str
    row_index: int
    column_index: int
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedDotsAndBoxesScene:
    """Rendered dots-and-boxes scene plus trace-friendly metadata."""

    image: Image.Image
    box_specs: Tuple[RenderedDotsAndBoxesBoxSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedDotsAndBoxesTaskContext:
    """Rendered dots-and-boxes image plus common style/noise metadata."""

    image: Image.Image
    rendered_scene: RenderedDotsAndBoxesScene
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def _allowed_panel_treatments(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Resolve optional panel-scene treatment restrictions for this render pass."""

    raw = params.get("panel_scene_treatments", group_default(render_defaults, "panel_scene_treatments", None))
    if isinstance(raw, str):
        return (str(raw),)
    if raw is None:
        return None
    return tuple(str(item) for item in raw)


def _draw_shadow(
    image: Image.Image,
    *,
    bbox_px: Tuple[float, float, float, float],
    radius_px: int,
    theme: DotsAndBoxesTheme,
) -> None:
    """Draw one soft panel shadow for the dots-and-boxes board."""

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


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    *,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    dash_length_px: int,
    dash_gap_px: int,
    width_px: int,
    fill_rgb: Tuple[int, int, int],
) -> None:
    """Draw one dashed highlight line between two points."""

    x1, y1 = start_xy
    x2, y2 = end_xy
    total_length = math.hypot(float(x2 - x1), float(y2 - y1))
    if total_length <= 0.0:
        return
    dx = float(x2 - x1) / total_length
    dy = float(y2 - y1) / total_length
    distance = 0.0
    while distance < total_length:
        segment_end = min(total_length, distance + float(dash_length_px))
        sx = float(x1 + (dx * distance))
        sy = float(y1 + (dy * distance))
        ex = float(x1 + (dx * segment_end))
        ey = float(y1 + (dy * segment_end))
        draw.line(
            [(sx, sy), (ex, ey)],
            fill=tuple(int(value) for value in fill_rgb),
            width=int(width_px),
        )
        distance = segment_end + float(dash_gap_px)


def _draw_board_treatment(
    image: Image.Image,
    *,
    board_bbox: Tuple[float, float, float, float],
    radius_px: int,
    theme: DotsAndBoxesTheme,
) -> None:
    """Draw optional inner fill and surface pattern for one board style."""

    draw = ImageDraw.Draw(image)
    left, top, right, bottom = board_bbox
    if theme.board_inner_fill_rgb is not None:
        inset = 10.0
        draw.rounded_rectangle(
            [left + inset, top + inset, right - inset, bottom - inset],
            radius=max(6, int(radius_px) - 8),
            fill=tuple(int(value) for value in theme.board_inner_fill_rgb),
        )
    if theme.board_pattern_rgb is None or int(theme.board_pattern_alpha) <= 0:
        return

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    color = (
        int(theme.board_pattern_rgb[0]),
        int(theme.board_pattern_rgb[1]),
        int(theme.board_pattern_rgb[2]),
        int(theme.board_pattern_alpha),
    )
    pattern = str(theme.board_rendering)
    if pattern == "notebook":
        spacing = 38.0
        y = float(top + 78.0)
        while y < float(bottom - 18.0):
            overlay_draw.line([(left + 18.0, y), (right - 18.0, y)], fill=color, width=1)
            y += spacing
        x = float(left + 86.0)
        overlay_draw.line([(x, top + 18.0), (x, bottom - 18.0)], fill=color, width=2)
    elif pattern == "wood":
        spacing = 58.0
        x = float(left + 24.0)
        while x < float(right - 18.0):
            overlay_draw.line([(x, top + 18.0), (x + 24.0, bottom - 18.0)], fill=color, width=3)
            x += spacing
    else:
        overlay_draw.rounded_rectangle(
            [left + 12.0, top + 12.0, right - 12.0, bottom - 12.0],
            radius=max(6, int(radius_px) - 10),
            outline=color,
            width=2,
        )
    image.alpha_composite(overlay)


def _edge_bbox(
    edge: DotsAndBoxesEdgeInstance,
    *,
    dot_xy: Mapping[Tuple[int, int], Tuple[float, float]],
    pad_px: float,
) -> Tuple[float, float, float, float]:
    """Return one padded bbox for a rendered edge."""

    start_xy = dot_xy[(int(edge.dot_start[0]), int(edge.dot_start[1]))]
    end_xy = dot_xy[(int(edge.dot_end[0]), int(edge.dot_end[1]))]
    x1, y1 = start_xy
    x2, y2 = end_xy
    return (
        round(min(x1, x2) - pad_px, 3),
        round(min(y1, y2) - pad_px, 3),
        round(max(x1, x2) + pad_px, 3),
        round(max(y1, y2) + pad_px, 3),
    )


def _edge_point_pair(
    edge: DotsAndBoxesEdgeInstance,
    *,
    dot_xy: Mapping[Tuple[int, int], Tuple[float, float]],
) -> List[List[float]]:
    """Return the two rendered endpoint pixels for one edge."""

    start_xy = dot_xy[(int(edge.dot_start[0]), int(edge.dot_start[1]))]
    end_xy = dot_xy[(int(edge.dot_end[0]), int(edge.dot_end[1]))]
    return [
        [round(float(start_xy[0]), 3), round(float(start_xy[1]), 3)],
        [round(float(end_xy[0]), 3), round(float(end_xy[1]), 3)],
    ]


def render_dots_and_boxes_scene(
    *,
    board_state: DotsAndBoxesBoardState,
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    params: DotsAndBoxesRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedDotsAndBoxesScene:
    """Render one dots-and-boxes board with a highlighted starting edge."""

    if str(scene_variant) != "single_board":
        raise ValueError(f"unsupported dots-and-boxes scene_variant: {scene_variant}")

    image = background.convert("RGBA")
    theme = build_games_dots_and_boxes_theme(style_variant=str(style_variant))
    draw = ImageDraw.Draw(image)
    board_left = float((int(params.canvas_width) - int(params.board_width_px)) / 2)
    board_top = float((int(params.canvas_height) - int(params.board_height_px)) / 2)
    board_right = float(board_left + int(params.board_width_px))
    board_bottom = float(board_top + int(params.board_height_px))
    board_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=(board_left, board_top, board_right, board_bottom),
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left, board_top, board_right, board_bottom = [float(value) for value in board_bbox]

    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_pad = max(16, int(round(float(params.panel_margin_px) * 0.55)))
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
            radius=max(18, int(params.board_corner_radius_px) + 8),
            border_width=max(2, int(round(float(theme.board_border_width_px) * 0.55))),
        )

    _draw_shadow(
        image,
        bbox_px=board_bbox,
        radius_px=int(params.board_corner_radius_px),
        theme=theme,
    )
    draw.rounded_rectangle(
        board_bbox,
        radius=int(params.board_corner_radius_px),
        fill=tuple(int(value) for value in theme.board_fill_rgb),
        outline=tuple(int(value) for value in theme.board_border_rgb),
        width=int(theme.board_border_width_px),
    )
    _draw_board_treatment(
        image,
        board_bbox=board_bbox,
        radius_px=int(params.board_corner_radius_px),
        theme=theme,
    )

    inner_left = float(board_left + int(params.board_padding_px))
    inner_top = float(board_top + int(params.board_padding_px))
    inner_right = float(board_right - int(params.board_padding_px))
    inner_bottom = float(board_bottom - int(params.board_padding_px))

    cell_size = min(
        float((inner_right - inner_left) / float(board_state.box_cols)),
        float((inner_bottom - inner_top) / float(board_state.box_rows)),
    )
    grid_width = float(cell_size * float(board_state.box_cols))
    grid_height = float(cell_size * float(board_state.box_rows))
    grid_left = float(inner_left + ((inner_right - inner_left - grid_width) / 2.0))
    grid_top = float(inner_top + ((inner_bottom - inner_top - grid_height) / 2.0))

    dot_xy: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for dot_row in range(int(board_state.box_rows) + 1):
        for dot_col in range(int(board_state.box_cols) + 1):
            dot_xy[(dot_row, dot_col)] = (
                round(float(grid_left + (dot_col * cell_size)), 3),
                round(float(grid_top + (dot_row * cell_size)), 3),
            )

    box_bboxes_px: Dict[str, List[float]] = {}
    box_specs: List[RenderedDotsAndBoxesBoxSpec] = []
    box_by_id = {str(box.box_id): box for box in board_state.boxes}
    for box in board_state.boxes:
        left = float(dot_xy[(int(box.row_index), int(box.column_index))][0])
        top = float(dot_xy[(int(box.row_index), int(box.column_index))][1])
        right = float(dot_xy[(int(box.row_index), int(box.column_index) + 1)][0])
        bottom = float(dot_xy[(int(box.row_index) + 1, int(box.column_index))][1])
        bbox = (
            round(left, 3),
            round(top, 3),
            round(right, 3),
            round(bottom, 3),
        )
        box_bboxes_px[str(box.box_id)] = [float(value) for value in bbox]
        box_specs.append(
            RenderedDotsAndBoxesBoxSpec(
                box_id=str(box.box_id),
                row_index=int(box.row_index),
                column_index=int(box.column_index),
                bbox_px=bbox,
            )
        )

    owner_fill_rgba = {
        "A": (48, 112, 214, 132),
        "B": (220, 126, 42, 132),
    }
    owner_preferred_text_rgb = {
        "A": (14, 31, 62),
        "B": (68, 37, 10),
    }
    owner_font = load_font(
        max(18, int(round(float(cell_size) * 0.28))),
        bold=True,
        font_family=str(params.font_family) or None,
    )
    owner_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    owner_draw = ImageDraw.Draw(owner_overlay)
    for box in board_state.boxes:
        owner = str(getattr(box, "owner", "") or "")
        if owner not in owner_fill_rgba:
            continue
        left, top, right, bottom = [float(value) for value in box_bboxes_px[str(box.box_id)]]
        inset = max(8.0, float(cell_size) * 0.11)
        fill_rgba = tuple(int(value) for value in owner_fill_rgba[str(owner)])
        owner_draw.rounded_rectangle(
            [left + inset, top + inset, right - inset, bottom - inset],
            radius=max(4, int(round(float(cell_size) * 0.12))),
            fill=fill_rgba,
        )
    image.alpha_composite(owner_overlay)
    draw = ImageDraw.Draw(image)
    for box in board_state.boxes:
        owner = str(getattr(box, "owner", "") or "")
        if owner not in owner_fill_rgba:
            continue
        left, top, right, bottom = [float(value) for value in box_bboxes_px[str(box.box_id)]]
        text_bbox = draw.textbbox((0, 0), owner, font=owner_font, stroke_width=1)
        text_width = float(text_bbox[2] - text_bbox[0])
        text_height = float(text_bbox[3] - text_bbox[1])
        text_xy = (
            float(left + ((right - left - text_width) / 2.0)),
            float(top + ((bottom - top - text_height) / 2.0)),
        )
        fill_rgb = tuple(int(value) for value in owner_preferred_text_rgb[str(owner)])
        surface_rgb = tuple(int(value) for value in owner_fill_rgba[str(owner)][:3])
        draw_text_traced(
            draw,
            text_xy,
            owner,
            font=owner_font,
            fill=fill_rgb,
            stroke_width=1,
            stroke_fill=(255, 255, 255),
            role="board_mark",
            required=True,
            surface_rgbs=[surface_rgb],
            preferred_rgbs=[fill_rgb],
        )

    edge_bboxes_px: Dict[str, List[float]] = {}
    edge_point_pairs_px: Dict[str, List[List[float]]] = {}
    for edge in board_state.edges:
        edge_bboxes_px[str(edge.edge_id)] = list(
            _edge_bbox(
                edge,
                dot_xy=dot_xy,
                pad_px=float(max(theme.edge_width_px, theme.highlight_width_px) + 8),
            )
        )
        edge_point_pairs_px[str(edge.edge_id)] = _edge_point_pair(edge, dot_xy=dot_xy)
        if not bool(edge.is_drawn) and not bool(edge.is_highlighted):
            continue
        start_xy = dot_xy[(int(edge.dot_start[0]), int(edge.dot_start[1]))]
        end_xy = dot_xy[(int(edge.dot_end[0]), int(edge.dot_end[1]))]
        if bool(edge.is_drawn):
            draw.line(
                [start_xy, end_xy],
                fill=tuple(int(value) for value in theme.edge_rgb),
                width=int(theme.edge_width_px),
            )
        if bool(edge.is_highlighted):
            _draw_dashed_line(
                draw,
                start_xy=start_xy,
                end_xy=end_xy,
                dash_length_px=int(params.dash_length_px),
                dash_gap_px=int(params.dash_gap_px),
                width_px=int(theme.highlight_width_px),
                fill_rgb=tuple(int(value) for value in theme.highlight_rgb),
            )

    for dot_row in range(int(board_state.box_rows) + 1):
        for dot_col in range(int(board_state.box_cols) + 1):
            cx, cy = dot_xy[(dot_row, dot_col)]
            radius = float(params.dot_radius_px)
            dot_bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            if str(theme.dot_rendering) == "outlined" and theme.dot_outline_rgb is not None:
                draw.ellipse(
                    [dot_bbox[0] - 1.5, dot_bbox[1] - 1.5, dot_bbox[2] + 1.5, dot_bbox[3] + 1.5],
                    fill=tuple(int(value) for value in theme.dot_outline_rgb),
                )
            draw.ellipse(
                dot_bbox,
                fill=tuple(int(value) for value in theme.dot_rgb),
            )

    option_label_by_box_id = {str(box_id): str(label) for box_id, label in board_state.option_label_by_box_id}
    option_label_bboxes_px: Dict[str, List[float]] = {}
    option_label_centers_px: Dict[str, List[float]] = {}
    option_box_id_by_label: Dict[str, str] = {}
    if option_label_by_box_id:
        option_font_size = max(18, int(round(float(cell_size) * 0.24)))
        option_font = load_font(
            option_font_size,
            bold=True,
            font_family=str(params.font_family) or None,
        )
        badge_radius = max(15.0, float(option_font_size) * 0.78)
        for box_id, label in sorted(option_label_by_box_id.items(), key=lambda item: str(item[1])):
            if not str(label) or str(box_id) not in box_bboxes_px:
                continue
            left, top, right, bottom = [float(value) for value in box_bboxes_px[str(box_id)]]
            cx = float((left + right) / 2.0)
            cy = float((top + bottom) / 2.0)
            badge_bbox = (
                round(cx - badge_radius, 3),
                round(cy - badge_radius, 3),
                round(cx + badge_radius, 3),
                round(cy + badge_radius, 3),
            )
            draw.ellipse(
                badge_bbox,
                fill=(255, 255, 255),
                outline=tuple(int(value) for value in theme.highlight_rgb),
                width=max(2, int(round(float(theme.highlight_width_px) * 0.28))),
            )
            text_bbox = draw.textbbox((0, 0), str(label), font=option_font, stroke_width=1)
            text_width = float(text_bbox[2] - text_bbox[0])
            text_height = float(text_bbox[3] - text_bbox[1])
            text_left = float(text_bbox[0])
            text_top = float(text_bbox[1])
            text_xy = (
                float(cx - (text_width / 2.0) - text_left),
                float(cy - (text_height / 2.0) - text_top),
            )
            draw_text_traced(
                draw,
                text_xy,
                str(label),
                font=option_font,
                fill=(24, 28, 34),
                stroke_width=1,
                stroke_fill=(255, 255, 255),
                role="option_label",
                required=True,
                surface_rgbs=[(255, 255, 255)],
                preferred_rgbs=[(24, 28, 34)],
            )
            label_bbox = draw.textbbox(text_xy, str(label), font=option_font, stroke_width=1)
            option_label_bboxes_px[str(label)] = [round(float(value), 3) for value in label_bbox]
            option_label_centers_px[str(label)] = [round(cx, 3), round(cy, 3)]
            option_box_id_by_label[str(label)] = str(box_id)

    scene_entities: List[Dict[str, Any]] = []
    for box_spec in box_specs:
        box = box_by_id[str(box_spec.box_id)]
        scene_entities.append(
            {
                "entity_id": str(box_spec.box_id),
                "kind": "dots_and_boxes_box",
                "bbox": [float(value) for value in box_spec.bbox_px],
                "row_index": int(box.row_index),
                "column_index": int(box.column_index),
                "owner": str(getattr(box, "owner", "") or ""),
            }
        )
    highlighted_edge_ids = tuple(
        str(edge_id)
        for edge_id in getattr(board_state, "highlighted_edge_ids", ())
        if str(edge_id)
    )
    if not highlighted_edge_ids and str(board_state.highlighted_edge_id):
        highlighted_edge_ids = (str(board_state.highlighted_edge_id),)
    for edge_id in highlighted_edge_ids:
        scene_entities.append(
            {
                "entity_id": str(edge_id),
                "kind": "dots_and_boxes_highlighted_edge",
                "bbox": list(edge_bboxes_px[str(edge_id)]),
            }
        )
    for label, box_id in sorted(option_box_id_by_label.items()):
        scene_entities.append(
            {
                "entity_id": f"option_{str(label)}",
                "kind": "dots_and_boxes_option_label",
                "bbox": list(option_label_bboxes_px[str(label)]),
                "label": str(label),
                "box_id": str(box_id),
            }
        )

    return RenderedDotsAndBoxesScene(
        image=image.convert("RGB"),
        box_specs=tuple(box_specs),
        scene_entities=tuple(scene_entities),
        render_map={
            "board_bbox_px": [float(value) for value in board_bbox],
            "box_bboxes_px": box_bboxes_px,
            "edge_bboxes_px": edge_bboxes_px,
            "edge_point_pairs_px": edge_point_pairs_px,
            "box_owner_by_id": {
                str(box.box_id): str(getattr(box, "owner", "") or "")
                for box in board_state.boxes
                if str(getattr(box, "owner", "") or "")
            },
            "highlighted_edge_id": str(board_state.highlighted_edge_id),
            "highlighted_edge_ids": [str(edge_id) for edge_id in highlighted_edge_ids],
            "option_label_bboxes_px": dict(option_label_bboxes_px),
            "option_label_centers_px": dict(option_label_centers_px),
            "option_box_id_by_label": dict(option_box_id_by_label),
            "layout_jitter": dict(layout_jitter),
            "scene_panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
            "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
            "font_family": str(params.font_family),
        },
    )


def render_dots_and_boxes_task_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    board_state: DotsAndBoxesBoardState,
    scene_variant: str,
    style_variant: str,
    render_params: DotsAndBoxesRenderParams,
) -> RenderedDotsAndBoxesTaskContext:
    """Render one dots-and-boxes board with shared games canvas styling."""

    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{DOTS_AND_BOXES_NAMESPACE}.panel_scene_style",
        treatments=_allowed_panel_treatments(params, render_defaults),
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(render_defaults, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(render_defaults, "panel_scene_palette_weights", None)),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_dots_and_boxes_scene(
        board_state=board_state,
        background=background,
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        params=render_params,
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
    return RenderedDotsAndBoxesTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "DotsAndBoxesRenderParams",
    "RenderedDotsAndBoxesBoxSpec",
    "RenderedDotsAndBoxesScene",
    "RenderedDotsAndBoxesTaskContext",
    "render_dots_and_boxes_scene",
    "render_dots_and_boxes_task_context",
]
