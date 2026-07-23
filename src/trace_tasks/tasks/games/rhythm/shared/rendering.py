"""Shared rhythm-lanes renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_game_text_traced as draw_text_traced
from ...shared.layout import apply_games_layout_jitter_to_bbox
from .rules import lane_entity_id, lane_label
from .state import RhythmNote, SUPPORTED_COLOR_KEYS
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata


@dataclass(frozen=True)
class RhythmRenderParams:
    """Resolved render controls for one rhythm-lanes scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    grid_width_px: int
    grid_height_px: int
    grid_border_width_px: int
    row_gap_px: int
    lane_gap_px: int
    note_radius_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RhythmTheme:
    """Resolved rhythm-lanes palette for one style variant."""

    grid_fill_rgb: Tuple[int, int, int]
    grid_outline_rgb: Tuple[int, int, int]
    lane_line_rgb: Tuple[int, int, int]
    row_line_rgb: Tuple[int, int, int]
    hit_line_rgb: Tuple[int, int, int]
    label_fill_rgb: Tuple[int, int, int]
    label_text_rgb: Tuple[int, int, int]
    note_outline_rgb: Tuple[int, int, int]
    note_text_rgb: Tuple[int, int, int]
    note_palette_rgb: Dict[str, Tuple[int, int, int]]


@dataclass(frozen=True)
class RenderedRhythmScene:
    """Rendered rhythm-lanes image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def build_games_rhythm_theme(*, style_variant: str) -> RhythmTheme:
    """Return the palette for one rhythm-lanes visual style."""

    style = str(style_variant)
    if style == "neon":
        return RhythmTheme(
            grid_fill_rgb=(12, 14, 32),
            grid_outline_rgb=(147, 127, 255),
            lane_line_rgb=(70, 77, 155),
            row_line_rgb=(42, 49, 111),
            hit_line_rgb=(255, 225, 77),
            label_fill_rgb=(36, 33, 84),
            label_text_rgb=(238, 241, 255),
            note_outline_rgb=(248, 250, 255),
            note_text_rgb=(20, 20, 28),
            note_palette_rgb={
                "yellow": (255, 230, 83),
                "cyan": (67, 225, 239),
                "magenta": (248, 80, 179),
                "green": (122, 239, 94),
            },
        )
    if style == "paper":
        return RhythmTheme(
            grid_fill_rgb=(246, 240, 222),
            grid_outline_rgb=(86, 78, 64),
            lane_line_rgb=(186, 173, 145),
            row_line_rgb=(214, 204, 177),
            hit_line_rgb=(193, 71, 61),
            label_fill_rgb=(232, 222, 196),
            label_text_rgb=(46, 39, 32),
            note_outline_rgb=(70, 62, 52),
            note_text_rgb=(36, 31, 25),
            note_palette_rgb={
                "yellow": (232, 190, 74),
                "cyan": (76, 162, 186),
                "magenta": (192, 91, 142),
                "green": (96, 156, 97),
            },
        )
    if style == "dark":
        return RhythmTheme(
            grid_fill_rgb=(22, 27, 35),
            grid_outline_rgb=(115, 132, 155),
            lane_line_rgb=(62, 74, 92),
            row_line_rgb=(42, 51, 64),
            hit_line_rgb=(255, 119, 92),
            label_fill_rgb=(41, 51, 66),
            label_text_rgb=(237, 243, 248),
            note_outline_rgb=(224, 231, 238),
            note_text_rgb=(15, 22, 28),
            note_palette_rgb={
                "yellow": (246, 207, 76),
                "cyan": (79, 190, 219),
                "magenta": (218, 100, 181),
                "green": (111, 196, 119),
            },
        )
    if style == "pastel":
        return RhythmTheme(
            grid_fill_rgb=(237, 241, 247),
            grid_outline_rgb=(91, 105, 124),
            lane_line_rgb=(184, 196, 212),
            row_line_rgb=(212, 220, 230),
            hit_line_rgb=(82, 114, 209),
            label_fill_rgb=(221, 228, 238),
            label_text_rgb=(37, 49, 65),
            note_outline_rgb=(86, 98, 116),
            note_text_rgb=(31, 41, 55),
            note_palette_rgb={
                "yellow": (245, 213, 118),
                "cyan": (133, 215, 226),
                "magenta": (232, 143, 194),
                "green": (152, 216, 148),
            },
        )
    return RhythmTheme(
        grid_fill_rgb=(27, 22, 44),
        grid_outline_rgb=(222, 119, 82),
        lane_line_rgb=(90, 58, 104),
        row_line_rgb=(64, 46, 81),
        hit_line_rgb=(255, 218, 94),
        label_fill_rgb=(72, 45, 85),
        label_text_rgb=(255, 238, 219),
        note_outline_rgb=(255, 238, 215),
        note_text_rgb=(36, 23, 24),
        note_palette_rgb={
            "yellow": (247, 209, 67),
            "cyan": (73, 197, 222),
            "magenta": (228, 75, 158),
            "green": (99, 198, 103),
        },
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[float, float, float, float],
    text: str,
    fill: Tuple[int, int, int],
    max_size_px: int,
    bold: bool = True,
    font_family: str = "",
) -> None:
    """Draw centered text inside one bbox."""

    left, top, right, bottom = bbox
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left)),
        max_height=max(1.0, float(bottom - top)),
        bold=bool(bold),
        min_size_px=8,
        max_size_px=int(max_size_px),
        font_family=str(font_family),
        fill_ratio=0.76,
    )
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    draw_text_traced(draw,
        (
            float(left + (0.5 * (float(right - left) - text_w)) - float(text_bbox[0])),
            float(top + (0.5 * (float(bottom - top) - text_h)) - float(text_bbox[1])),
        ),
        str(text),
        fill=tuple(int(v) for v in fill),
        font=font,
     role="readout", required=False,)


_SCORE_PALETTE_WIDTH_PX = 150.0
_SCORE_PALETTE_GAP_PX = 20.0


def _grid_bbox(
    params: RhythmRenderParams,
    *,
    side_panel_width_px: float = 0.0,
    side_panel_gap_px: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Return the rhythm grid bbox before jitter."""

    total_width = float(params.grid_width_px) + max(0.0, float(side_panel_width_px))
    if float(side_panel_width_px) > 0:
        total_width += max(0.0, float(side_panel_gap_px))
    left = float((int(params.canvas_width) - total_width) / 2.0)
    top = float((int(params.canvas_height) - int(params.grid_height_px)) / 2.0) - 14.0
    return (
        left,
        top,
        left + float(params.grid_width_px),
        top + float(params.grid_height_px),
    )


def _lane_bbox(
    *,
    play_bbox: Tuple[float, float, float, float],
    lane: int,
    lane_count: int,
) -> Tuple[float, float, float, float]:
    """Return one lane body bbox."""

    left, top, right, bottom = play_bbox
    lane_w = float((right - left) / max(1, int(lane_count)))
    return (
        round(float(left + (int(lane) * lane_w)), 3),
        round(float(top), 3),
        round(float(left + ((int(lane) + 1) * lane_w)), 3),
        round(float(bottom), 3),
    )


def _note_bbox(
    *,
    play_bbox: Tuple[float, float, float, float],
    note: RhythmNote,
    lane_count: int,
    row_count: int,
    lane_gap_px: int,
    row_gap_px: int,
) -> Tuple[float, float, float, float]:
    """Return a note bbox in image coordinates."""

    left, top, right, bottom = play_bbox
    lane_w = float((right - left) / max(1, int(lane_count)))
    row_h = float((bottom - top) / max(1, int(row_count)))
    x0 = float(left + (int(note.lane_index) * lane_w) + float(lane_gap_px))
    x1 = float(left + ((int(note.lane_index) + 1) * lane_w) - float(lane_gap_px))
    top_row = int(note.bottom_row) + int(note.length) - 1
    y0 = float(bottom - (top_row * row_h) + float(row_gap_px))
    y1 = float(bottom - ((int(note.bottom_row) - 1) * row_h) - float(row_gap_px))
    return (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))


def _draw_score_palette(
    draw: ImageDraw.ImageDraw,
    *,
    palette_bbox: Tuple[float, float, float, float],
    score_values_by_color: Mapping[str, int],
    theme: RhythmTheme,
    params: RhythmRenderParams,
) -> Dict[str, Any]:
    """Draw the side score palette used by score-value objectives."""

    left, top, right, bottom = palette_bbox
    draw.rounded_rectangle(
        palette_bbox,
        radius=18,
        fill=tuple(int(v) for v in theme.grid_fill_rgb) + (238,),
        outline=tuple(int(v) for v in theme.grid_outline_rgb) + (240,),
        width=max(2, int(params.grid_border_width_px) - 1),
    )
    heading_bbox = (left + 12.0, top + 12.0, right - 12.0, top + 46.0)
    _draw_centered_text(
        draw,
        bbox=heading_bbox,
        text="POINTS",
        fill=tuple(int(v) for v in theme.label_text_rgb),
        max_size_px=22,
        font_family=str(params.font_family),
    )

    colors = [str(color) for color in SUPPORTED_COLOR_KEYS if str(color) in score_values_by_color]
    available_h = max(1.0, float(bottom - top - 62.0))
    row_h = available_h / max(1, len(colors))
    entry_bboxes: Dict[str, list[float]] = {}
    swatch_bboxes: Dict[str, list[float]] = {}
    value_bboxes: Dict[str, list[float]] = {}
    for index, color_key in enumerate(colors):
        row_top = float(top + 52.0 + (index * row_h))
        row_bottom = float(top + 52.0 + ((index + 1) * row_h) - 6.0)
        entry_bbox = (left + 12.0, row_top, right - 12.0, row_bottom)
        swatch_side = min(34.0, max(20.0, float(row_bottom - row_top - 4.0)))
        swatch_bbox = (
            entry_bbox[0],
            row_top + ((row_bottom - row_top - swatch_side) / 2.0),
            entry_bbox[0] + swatch_side,
            row_top + ((row_bottom - row_top + swatch_side) / 2.0),
        )
        value_bbox = (swatch_bbox[2] + 12.0, row_top, entry_bbox[2], row_bottom)
        draw.rounded_rectangle(
            swatch_bbox,
            radius=8,
            fill=tuple(int(v) for v in theme.note_palette_rgb[str(color_key)]) + (248,),
            outline=tuple(int(v) for v in theme.note_outline_rgb) + (235,),
            width=2,
        )
        _draw_centered_text(
            draw,
            bbox=value_bbox,
            text=str(int(score_values_by_color[str(color_key)])),
            fill=tuple(int(v) for v in theme.label_text_rgb),
            max_size_px=26,
            font_family=str(params.font_family),
        )
        entry_bboxes[str(color_key)] = [round(float(v), 3) for v in entry_bbox]
        swatch_bboxes[str(color_key)] = [round(float(v), 3) for v in swatch_bbox]
        value_bboxes[str(color_key)] = [round(float(v), 3) for v in value_bbox]

    return {
        "bbox_px": [round(float(v), 3) for v in palette_bbox],
        "values_by_color": {str(color): int(value) for color, value in score_values_by_color.items()},
        "entry_bboxes_px": entry_bboxes,
        "swatch_bboxes_px": swatch_bboxes,
        "value_bboxes_px": value_bboxes,
    }


def render_rhythm_lanes_scene(
    *,
    lane_count: int,
    row_count: int,
    beat_window: int,
    notes: Tuple[RhythmNote, ...],
    score_values_by_color: Mapping[str, int] | None = None,
    background: Image.Image,
    style_variant: str,
    params: RhythmRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedRhythmScene:
    """Render the complete rhythm-lanes playfield and trace geometry."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_rhythm_theme(style_variant=str(style_variant))
    score_palette_values = (
        None
        if score_values_by_color is None
        else {str(color): int(value) for color, value in score_values_by_color.items()}
    )
    palette_width = _SCORE_PALETTE_WIDTH_PX if score_palette_values else 0.0
    palette_gap = _SCORE_PALETTE_GAP_PX if score_palette_values else 0.0

    grid_bbox = _grid_bbox(params, side_panel_width_px=palette_width, side_panel_gap_px=palette_gap)
    group_bbox = (
        grid_bbox[0],
        grid_bbox[1],
        grid_bbox[2] + palette_gap + palette_width,
        grid_bbox[3],
    )
    if isinstance(params.layout_jitter_meta, Mapping):
        jittered_group_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=group_bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
        grid_bbox = (
            float(grid_bbox[0] + dx),
            float(grid_bbox[1] + dy),
            float(grid_bbox[2] + dx),
            float(grid_bbox[3] + dy),
        )
        group_bbox = jittered_group_bbox
    else:
        layout_jitter = {}

    grid_left, grid_top, grid_right, grid_bottom = grid_bbox
    group_left, group_top, group_right, group_bottom = group_bbox
    score_palette_bbox = (
        round(float(grid_right + palette_gap), 3),
        round(float(grid_top + 44.0), 3),
        round(float(grid_right + palette_gap + palette_width), 3),
        round(float(grid_top + 300.0), 3),
    ) if score_palette_values else None
    label_band_h = 52.0
    hit_label_w = 54.0
    play_bbox = (
        round(float(grid_left + hit_label_w), 3),
        round(float(grid_top + 24.0), 3),
        round(float(grid_right - 24.0), 3),
        round(float(grid_bottom - label_band_h), 3),
    )
    play_left, play_top, play_right, play_bottom = play_bbox

    if panel_style is not None:
        panel_pad = 22.0
        panel_bbox = (
            int(round(max(6.0, float(group_left) - panel_pad))),
            int(round(max(6.0, float(group_top) - panel_pad))),
            int(round(min(float(params.canvas_width) - 6.0, float(group_right) + panel_pad))),
            int(round(min(float(params.canvas_height) - 6.0, float(group_bottom) + panel_pad))),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=24,
            border_width=2,
        )

    draw.rounded_rectangle(
        grid_bbox,
        radius=18,
        fill=tuple(int(v) for v in theme.grid_fill_rgb) + (240,),
        outline=tuple(int(v) for v in theme.grid_outline_rgb) + (255,),
        width=int(params.grid_border_width_px),
    )
    score_palette_meta: Dict[str, Any] = {}
    if score_palette_values and score_palette_bbox is not None:
        score_palette_meta = _draw_score_palette(
            draw,
            palette_bbox=score_palette_bbox,
            score_values_by_color=score_palette_values,
            theme=theme,
            params=params,
        )

    lane_w = float((play_right - play_left) / max(1, int(lane_count)))
    row_h = float((play_bottom - play_top) / max(1, int(row_count)))
    for lane in range(int(lane_count) + 1):
        x = float(play_left + (lane * lane_w))
        draw.line(
            (x, play_top, x, play_bottom),
            fill=tuple(int(v) for v in theme.lane_line_rgb) + (180,),
            width=2,
        )
    for row in range(int(row_count) + 1):
        y = float(play_bottom - (row * row_h))
        draw.line(
            (play_left, y, play_right, y),
            fill=tuple(int(v) for v in theme.row_line_rgb) + (150,),
            width=1,
        )

    hit_y = float(play_bottom)
    draw.line(
        (play_left - 10.0, hit_y, play_right + 10.0, hit_y),
        fill=tuple(int(v) for v in theme.hit_line_rgb) + (255,),
        width=6,
    )
    _draw_centered_text(
        draw,
        bbox=(grid_left + 8.0, hit_y - 28.0, play_left - 8.0, hit_y + 22.0),
        text="HIT",
        fill=tuple(int(v) for v in theme.hit_line_rgb),
        max_size_px=int(params.label_font_size_px),
        font_family=str(params.font_family),
    )

    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    lane_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    note_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []

    for lane in range(int(lane_count)):
        body_bbox = _lane_bbox(play_bbox=play_bbox, lane=lane, lane_count=int(lane_count))
        label_bbox = (
            round(float(body_bbox[0] + max(4.0, 0.12 * lane_w)), 3),
            round(float(play_bottom + 13.0), 3),
            round(float(body_bbox[2] - max(4.0, 0.12 * lane_w)), 3),
            round(float(play_bottom + 43.0), 3),
        )
        lane_id = lane_entity_id(lane)
        lane_bboxes[str(lane_id)] = label_bbox
        entity_bboxes[str(lane_id)] = label_bbox
        draw.rounded_rectangle(
            label_bbox,
            radius=8,
            fill=tuple(int(v) for v in theme.label_fill_rgb) + (235,),
            outline=tuple(int(v) for v in theme.grid_outline_rgb) + (210,),
            width=2,
        )
        _draw_centered_text(
            draw,
            bbox=(label_bbox[0] + 2, label_bbox[1] + 2, label_bbox[2] - 2, label_bbox[3] - 2),
            text=lane_label(lane),
            fill=tuple(int(v) for v in theme.label_text_rgb),
            max_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
        )
        scene_entities.append(
            {
                "entity_id": str(lane_id),
                "entity_type": "rhythm_lane_label",
                "label": lane_label(lane),
                "lane": int(lane),
                "bbox_px": list(label_bbox),
            }
        )

    for note in notes:
        bbox = _note_bbox(
            play_bbox=play_bbox,
            note=note,
            lane_count=int(lane_count),
            row_count=int(row_count),
            lane_gap_px=int(params.lane_gap_px),
            row_gap_px=int(params.row_gap_px),
        )
        fill = theme.note_palette_rgb[str(note.color_key)]
        radius = max(5, min(int(params.note_radius_px), int((bbox[2] - bbox[0]) * 0.28)))
        draw.rounded_rectangle(
            bbox,
            radius=radius,
            fill=tuple(int(v) for v in fill) + (248,),
            outline=tuple(int(v) for v in theme.note_outline_rgb) + (245,),
            width=3,
        )
        if int(note.length) > 1:
            inner_x = float((bbox[0] + bbox[2]) / 2.0)
            draw.line(
                (inner_x, bbox[1] + 7.0, inner_x, bbox[3] - 7.0),
                fill=tuple(int(v) for v in theme.note_outline_rgb) + (145,),
                width=3,
            )
        else:
            shine_bbox = (
                float(bbox[0] + 0.16 * (bbox[2] - bbox[0])),
                float(bbox[1] + 0.18 * (bbox[3] - bbox[1])),
                float(bbox[0] + 0.42 * (bbox[2] - bbox[0])),
                float(bbox[1] + 0.45 * (bbox[3] - bbox[1])),
            )
            draw.ellipse(shine_bbox, fill=(255, 255, 255, 92))
        note_bboxes[str(note.note_id)] = bbox
        entity_bboxes[str(note.note_id)] = bbox
        scene_entities.append(
            {
                "entity_id": str(note.note_id),
                "entity_type": "rhythm_note",
                "lane": int(note.lane_index),
                "lane_label": lane_label(int(note.lane_index)),
                "bottom_row_from_hit_line": int(note.bottom_row),
                "length_rows": int(note.length),
                "color_key": str(note.color_key),
                "kind": str(note.kind),
                "bbox_px": list(bbox),
            }
        )

    render_map = {
        "grid_bbox_px": [round(float(v), 3) for v in grid_bbox],
        "playfield_bbox_px": [round(float(v), 3) for v in play_bbox],
        "hit_line_px": {
            "start": [round(float(play_left), 3), round(float(hit_y), 3)],
            "end": [round(float(play_right), 3), round(float(hit_y), 3)],
        },
        "lane_bboxes_px": {str(key): list(value) for key, value in lane_bboxes.items()},
        "note_bboxes_px": {str(key): list(value) for key, value in note_bboxes.items()},
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "lane_count": int(lane_count),
        "row_count": int(row_count),
        "beat_window": int(beat_window),
        "layout_jitter": dict(layout_jitter),
        "font_family": str(params.font_family),
        "text_style": {"font_family": str(params.font_family)},
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "score_palette": score_palette_meta,
    }
    return RenderedRhythmScene(
        image=image.convert("RGB"),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "RhythmRenderParams",
    "RhythmTheme",
    "RenderedRhythmScene",
    "build_games_rhythm_theme",
    "render_rhythm_lanes_scene",
]
