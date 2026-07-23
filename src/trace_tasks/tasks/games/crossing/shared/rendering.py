"""Shared lane-crossing renderer for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from ....shared.text_rendering import fit_font_to_box
from ...shared.text import draw_game_text_traced as draw_text_traced
from .state import (
    CrossingRouteOption,
    CrossingVehicle,
    route_cell_entity_id,
    route_entity_id,
    start_entity_id,
)
from ...shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from ...shared.scene_style import (
    GamePanelSceneStyle,
    draw_panel_scene_chrome,
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from ...shared.visual_defaults import load_games_scene_noise_defaults
from .defaults import FALLBACK_RENDERING_DEFAULTS, SCENE_ID
from .state import CrossingSample


@dataclass(frozen=True)
class CrossingRenderParams:
    """Resolved render controls for one lane-crossing scene."""

    canvas_width: int
    canvas_height: int
    playfield_width_px: int
    playfield_height_px: int
    panel_margin_px: int
    border_width_px: int
    safe_band_height_px: int
    vehicle_width_px: int
    vehicle_height_px: int
    path_width_px: int
    label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class CrossingTheme:
    """Resolved lane-crossing visual palette."""

    background_rgb: Tuple[int, int, int]
    road_rgb: Tuple[int, int, int]
    road_alt_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    safe_rgb: Tuple[int, int, int]
    safe_alt_rgb: Tuple[int, int, int]
    start_rgb: Tuple[int, int, int]
    start_outline_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
    vehicle_rgbs: Tuple[Tuple[int, int, int], ...]
    vehicle_outline_rgb: Tuple[int, int, int]
    path_rgbs: Tuple[Tuple[int, int, int], ...]
    marked_path_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedCrossingScene:
    """Rendered crossing image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedCrossingTaskContext:
    """Rendered crossing scene context shared by objective-owned tasks."""

    image: Image.Image
    rendered_scene: RenderedCrossingScene
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def resolve_crossing_render_params(params: Mapping[str, Any], *, instance_seed: int) -> CrossingRenderParams:
    """Resolve crossing rendering parameters from scene config/defaults."""

    fallback = FALLBACK_RENDERING_DEFAULTS
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.crossing.font_family",
        params=params,
    )
    return CrossingRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", fallback["canvas_width"]))),
        canvas_height=int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", fallback["canvas_height"]))),
        playfield_width_px=int(params.get("playfield_width_px", group_default(_RENDER_DEFAULTS, "playfield_width_px", fallback["playfield_width_px"]))),
        playfield_height_px=int(params.get("playfield_height_px", group_default(_RENDER_DEFAULTS, "playfield_height_px", fallback["playfield_height_px"]))),
        panel_margin_px=int(params.get("panel_margin_px", group_default(_RENDER_DEFAULTS, "panel_margin_px", fallback["panel_margin_px"]))),
        border_width_px=int(params.get("border_width_px", group_default(_RENDER_DEFAULTS, "border_width_px", fallback["border_width_px"]))),
        safe_band_height_px=int(params.get("safe_band_height_px", group_default(_RENDER_DEFAULTS, "safe_band_height_px", fallback["safe_band_height_px"]))),
        vehicle_width_px=int(params.get("vehicle_width_px", group_default(_RENDER_DEFAULTS, "vehicle_width_px", fallback["vehicle_width_px"]))),
        vehicle_height_px=int(params.get("vehicle_height_px", group_default(_RENDER_DEFAULTS, "vehicle_height_px", fallback["vehicle_height_px"]))),
        path_width_px=int(params.get("path_width_px", group_default(_RENDER_DEFAULTS, "path_width_px", fallback["path_width_px"]))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", fallback["label_font_size_px"]))),
        font_family=str(font_family),
        layout_jitter_meta=resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.crossing.layout",
        ),
    )


def build_games_crossing_theme(*, style_variant: str) -> CrossingTheme:
    """Resolve lane-crossing colors while keeping route and vehicle contrast stable."""

    style = str(style_variant)
    if style == "night":
        return CrossingTheme(
            background_rgb=(14, 24, 34),
            road_rgb=(31, 36, 43),
            road_alt_rgb=(39, 45, 54),
            grid_rgb=(122, 136, 150),
            safe_rgb=(22, 82, 62),
            safe_alt_rgb=(18, 70, 55),
            start_rgb=(40, 126, 90),
            start_outline_rgb=(184, 238, 208),
            text_rgb=(240, 245, 238),
            vehicle_rgbs=((232, 83, 83), (82, 166, 232), (246, 194, 88), (196, 105, 232), (92, 212, 160)),
            vehicle_outline_rgb=(246, 247, 240),
            path_rgbs=((255, 220, 90), (98, 210, 255), (255, 126, 190), (156, 240, 135), (225, 160, 255), (255, 156, 90)),
            marked_path_rgb=(255, 226, 66),
        )
    if style == "retro":
        return CrossingTheme(
            background_rgb=(23, 18, 36),
            road_rgb=(46, 39, 63),
            road_alt_rgb=(57, 47, 75),
            grid_rgb=(163, 139, 210),
            safe_rgb=(38, 120, 72),
            safe_alt_rgb=(35, 103, 65),
            start_rgb=(66, 174, 104),
            start_outline_rgb=(251, 235, 128),
            text_rgb=(255, 241, 168),
            vehicle_rgbs=((255, 91, 119), (97, 219, 255), (255, 198, 72), (182, 114, 255), (120, 244, 150)),
            vehicle_outline_rgb=(35, 28, 50),
            path_rgbs=((255, 236, 87), (66, 235, 255), (255, 111, 198), (132, 255, 128), (207, 135, 255), (255, 155, 82)),
            marked_path_rgb=(255, 244, 83),
        )
    if style == "paper":
        return CrossingTheme(
            background_rgb=(232, 226, 204),
            road_rgb=(190, 190, 178),
            road_alt_rgb=(205, 204, 190),
            grid_rgb=(93, 91, 80),
            safe_rgb=(173, 205, 151),
            safe_alt_rgb=(158, 190, 138),
            start_rgb=(216, 232, 183),
            start_outline_rgb=(70, 80, 55),
            text_rgb=(42, 45, 38),
            vehicle_rgbs=((194, 72, 63), (62, 126, 176), (208, 158, 62), (137, 90, 167), (73, 147, 105)),
            vehicle_outline_rgb=(48, 45, 38),
            path_rgbs=((219, 126, 42), (54, 132, 178), (174, 74, 126), (76, 145, 82), (132, 91, 164), (187, 101, 54)),
            marked_path_rgb=(223, 132, 38),
        )
    if style == "construction":
        return CrossingTheme(
            background_rgb=(38, 42, 39),
            road_rgb=(58, 61, 57),
            road_alt_rgb=(68, 70, 65),
            grid_rgb=(228, 188, 73),
            safe_rgb=(72, 104, 63),
            safe_alt_rgb=(63, 91, 55),
            start_rgb=(238, 188, 62),
            start_outline_rgb=(38, 38, 32),
            text_rgb=(250, 241, 210),
            vehicle_rgbs=((228, 82, 58), (69, 149, 198), (236, 183, 55), (143, 101, 190), (74, 171, 118)),
            vehicle_outline_rgb=(27, 28, 25),
            path_rgbs=((255, 233, 92), (91, 202, 250), (255, 118, 174), (130, 232, 125), (202, 150, 255), (255, 158, 89)),
            marked_path_rgb=(255, 231, 55),
        )
    return CrossingTheme(
        background_rgb=(84, 132, 105),
        road_rgb=(74, 79, 83),
        road_alt_rgb=(86, 91, 94),
        grid_rgb=(225, 226, 214),
        safe_rgb=(85, 156, 98),
        safe_alt_rgb=(76, 139, 90),
        start_rgb=(186, 225, 143),
        start_outline_rgb=(37, 77, 43),
        text_rgb=(28, 38, 31),
        vehicle_rgbs=((210, 64, 56), (55, 129, 201), (236, 182, 64), (148, 87, 192), (64, 176, 119)),
        vehicle_outline_rgb=(245, 248, 236),
        path_rgbs=((242, 208, 57), (61, 153, 220), (225, 76, 146), (80, 185, 89), (156, 101, 207), (229, 124, 60)),
        marked_path_rgb=(250, 215, 44),
    )


def _fit_text(
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
        font_family=str(font_family) or None,
        min_size_px=8,
        max_size_px=int(max_size_px),
        fill_ratio=0.78,
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


def _union_bbox(boxes: Tuple[Tuple[float, float, float, float], ...]) -> Tuple[float, float, float, float]:
    """Return the union of bboxes."""

    left = min(float(box[0]) for box in boxes)
    top = min(float(box[1]) for box in boxes)
    right = max(float(box[2]) for box in boxes)
    bottom = max(float(box[3]) for box in boxes)
    return (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    direction: int,
    color: Tuple[int, int, int],
    size: float,
) -> None:
    """Draw a small horizontal direction arrow."""

    cx, cy = float(center[0]), float(center[1])
    span = float(size)
    if int(direction) > 0:
        line = (cx - span, cy, cx + span, cy)
        head = [(cx + span, cy), (cx + (0.45 * span), cy - (0.40 * span)), (cx + (0.45 * span), cy + (0.40 * span))]
    else:
        line = (cx + span, cy, cx - span, cy)
        head = [(cx - span, cy), (cx - (0.45 * span), cy - (0.40 * span)), (cx - (0.45 * span), cy + (0.40 * span))]
    draw.line(line, fill=tuple(int(v) for v in color) + (210,), width=max(2, int(round(span * 0.16))))
    draw.polygon(head, fill=tuple(int(v) for v in color) + (210,))


def render_crossing_scene(
    *,
    lane_count: int,
    row_count: int,
    row_directions: Tuple[int, ...],
    vehicles: Tuple[CrossingVehicle, ...],
    start_labels: Tuple[str, ...],
    route_options: Tuple[CrossingRouteOption, ...],
    marked_route_label: str | None,
    background: Image.Image,
    style_variant: str,
    params: CrossingRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedCrossingScene:
    """Render the full playfield and record every task-addressable entity geometry."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_games_crossing_theme(style_variant=str(style_variant))

    left = float((int(params.canvas_width) - int(params.playfield_width_px)) / 2.0)
    top = float((int(params.canvas_height) - int(params.playfield_height_px)) / 2.0)
    playfield_bbox = (
        left,
        top,
        left + float(params.playfield_width_px),
        top + float(params.playfield_height_px),
    )
    if isinstance(params.layout_jitter_meta, Mapping):
        playfield_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
            bbox_px=playfield_bbox,
            canvas_width=int(params.canvas_width),
            canvas_height=int(params.canvas_height),
            jitter=params.layout_jitter_meta,
        )
    else:
        layout_jitter = {}
    left, top, right, bottom = [float(v) for v in playfield_bbox]
    width = float(right - left)
    height = float(bottom - top)
    safe_h = float(params.safe_band_height_px)
    goal_h = max(26.0, float(safe_h) * 0.45)
    road_top = float(top + goal_h)
    road_bottom = float(bottom - safe_h)
    row_h = float((road_bottom - road_top) / max(1, int(row_count)))
    lane_w = float(width / max(1, int(lane_count)))
    panel_bbox = (
        int(max(8, round(left - 24.0))),
        int(max(8, round(top - 24.0))),
        int(min(int(params.canvas_width) - 8, round(right + 24.0))),
        int(min(int(params.canvas_height) - 8, round(bottom + 24.0))),
    )
    if panel_style is not None:
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=28,
            border_width=max(1, int(params.border_width_px)),
        )

    draw.rounded_rectangle(
        playfield_bbox,
        radius=22,
        fill=tuple(int(v) for v in theme.background_rgb) + (245,),
        outline=tuple(int(v) for v in theme.grid_rgb) + (255,),
        width=max(2, int(params.border_width_px)),
    )
    draw.rectangle((left, top, right, road_top), fill=tuple(int(v) for v in theme.safe_alt_rgb) + (246,))
    draw.rectangle((left, road_bottom, right, bottom), fill=tuple(int(v) for v in theme.safe_rgb) + (246,))
    draw.line(
        (left + 10.0, road_top - 2.0, right - 10.0, road_top - 2.0),
        fill=tuple(int(v) for v in theme.grid_rgb) + (150,),
        width=max(2, int(params.border_width_px) - 2),
    )

    cell_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    scene_entities: list[Dict[str, Any]] = []

    for row in range(int(row_count)):
        y0 = float(road_bottom - ((row + 1) * row_h))
        y1 = float(road_bottom - (row * row_h))
        fill = theme.road_rgb if int(row) % 2 == 0 else theme.road_alt_rgb
        draw.rectangle((left, y0, right, y1), fill=tuple(int(v) for v in fill) + (248,))
        draw.line((left, y0, right, y0), fill=tuple(int(v) for v in theme.grid_rgb) + (120,), width=2)
        _draw_arrow(
            draw,
            center=(left + 24.0, (y0 + y1) / 2.0),
            direction=int(row_directions[row]),
            color=theme.grid_rgb,
            size=13.0,
        )
        _draw_arrow(
            draw,
            center=(right - 24.0, (y0 + y1) / 2.0),
            direction=int(row_directions[row]),
            color=theme.grid_rgb,
            size=13.0,
        )
        for col in range(int(lane_count)):
            x0 = float(left + (col * lane_w))
            x1 = float(left + ((col + 1) * lane_w))
            cell_bboxes[f"row_{row}_col_{col}"] = (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))

    for col in range(1, int(lane_count)):
        x = float(left + (col * lane_w))
        draw.line((x, road_top, x, road_bottom), fill=tuple(int(v) for v in theme.grid_rgb) + (95,), width=1)

    start_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for index, label in enumerate(start_labels):
        if index >= int(lane_count):
            break
        pad_margin_x = max(8.0, lane_w * 0.15)
        pad_left = float(left + (index * lane_w) + pad_margin_x)
        pad_right = float(left + ((index + 1) * lane_w) - pad_margin_x)
        pad_top = float(road_bottom + 0.20 * safe_h)
        pad_bottom = float(bottom - 0.18 * safe_h)
        bbox = (round(pad_left, 3), round(pad_top, 3), round(pad_right, 3), round(pad_bottom, 3))
        draw.rounded_rectangle(
            bbox,
            radius=9,
            fill=tuple(int(v) for v in theme.start_rgb) + (245,),
            outline=tuple(int(v) for v in theme.start_outline_rgb) + (255,),
            width=3,
        )
        _fit_text(
            draw,
            bbox=bbox,
            text=str(label),
            fill=theme.text_rgb,
            max_size_px=int(params.label_font_size_px),
            font_family=str(params.font_family),
        )
        entity_id = start_entity_id(index)
        start_bboxes[str(entity_id)] = bbox
        entity_bboxes[str(entity_id)] = bbox
        scene_entities.append(
            {
                "entity_id": str(entity_id),
                "entity_type": "crossing_start_pad",
                "label": str(label),
                "start_index": int(index),
                "bbox_px": list(bbox),
            }
        )

    route_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    route_cell_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for route_index, route in enumerate(route_options):
        color = theme.marked_path_rgb if route.label == marked_route_label else theme.path_rgbs[int(route.color_index) % len(theme.path_rgbs)]
        points: list[Tuple[float, float]] = []
        cell_boxes: list[Tuple[float, float, float, float]] = []
        start_col = int(route.path_cols[0]) if route.path_cols else 0
        points.append((left + ((start_col + 0.5) * lane_w), road_bottom + (0.08 * safe_h)))
        for row, col in enumerate(route.path_cols):
            cx = float(left + ((int(col) + 0.5) * lane_w))
            cy = float(road_bottom - ((int(row) + 0.5) * row_h))
            points.append((cx, cy))
            box_size = min(lane_w, row_h) * 0.56
            bbox = (
                round(cx - (0.5 * box_size), 3),
                round(cy - (0.5 * box_size), 3),
                round(cx + (0.5 * box_size), 3),
                round(cy + (0.5 * box_size), 3),
            )
            entity_id = route_cell_entity_id(route.label, row)
            route_cell_bboxes[str(entity_id)] = bbox
            entity_bboxes[str(entity_id)] = bbox
            scene_entities.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": "crossing_route_cell",
                    "route_label": str(route.label),
                    "row": int(row),
                    "col": int(col),
                    "bbox_px": list(bbox),
                }
            )
            cell_boxes.append(bbox)
        goal_col = int(route.path_cols[-1]) if route.path_cols else start_col
        points.append((left + ((goal_col + 0.5) * lane_w), road_top - max(4.0, 0.12 * goal_h)))
        if len(points) >= 2:
            line_width = max(3, int(params.path_width_px) + (5 if route.label == marked_route_label else 0))
            draw.line(points, fill=tuple(int(v) for v in color) + (220,), width=line_width, joint="curve")
        for row, point in enumerate(points[1:-1]):
            radius_scale = 0.18 if route.label == marked_route_label else 0.12
            r = max(6.0, min(lane_w, row_h) * radius_scale)
            draw.ellipse(
                (point[0] - r, point[1] - r, point[0] + r, point[1] + r),
                fill=tuple(int(v) for v in color) + (238,),
                outline=tuple(int(v) for v in theme.vehicle_outline_rgb) + (210,),
                width=2,
            )
            if route.label == marked_route_label:
                _fit_text(
                    draw,
                    bbox=(point[0] - (1.6 * r), point[1] - (1.6 * r), point[0] + (1.6 * r), point[1] + (1.6 * r)),
                    text=str(row + 1),
                    fill=theme.text_rgb,
                    max_size_px=max(8, int(params.label_font_size_px * 0.55)),
                    font_family=str(params.font_family),
                )
        label_center = points[0]
        label_box = (
            label_center[0] - 18.0,
            label_center[1] + 1.0,
            label_center[0] + 18.0,
            label_center[1] + 34.0,
        )
        start_pad_label_matches = bool(
            int(start_col) < len(start_labels) and str(start_labels[int(start_col)]) == str(route.label)
        )
        if route.label != marked_route_label and not start_pad_label_matches:
            draw.rounded_rectangle(
                label_box,
                radius=7,
                fill=tuple(int(v) for v in color) + (230,),
                outline=tuple(int(v) for v in theme.vehicle_outline_rgb) + (210,),
                width=2,
            )
            _fit_text(
                draw,
                bbox=label_box,
                text=str(route.label),
                fill=theme.text_rgb,
                max_size_px=int(params.label_font_size_px),
                font_family=str(params.font_family),
            )
        if cell_boxes:
            bbox = _union_bbox(tuple(cell_boxes))
            entity_id = route_entity_id(route.label)
            route_bboxes[str(entity_id)] = bbox
            entity_bboxes[str(entity_id)] = bbox
            scene_entities.append(
                {
                    "entity_id": str(entity_id),
                    "entity_type": "crossing_route",
                    "route_label": str(route.label),
                    "route_index": int(route_index),
                    "bbox_px": list(bbox),
                }
            )

    vehicle_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for vehicle in vehicles:
        row = int(vehicle.row)
        col = int(vehicle.start_col)
        cx = float(left + ((col + 0.5) * lane_w))
        cy = float(road_bottom - ((row + 0.5) * row_h))
        w = min(float(params.vehicle_width_px), lane_w * 0.72)
        h = min(float(params.vehicle_height_px), row_h * 0.62)
        bbox = (round(cx - (w / 2.0), 3), round(cy - (h / 2.0), 3), round(cx + (w / 2.0), 3), round(cy + (h / 2.0), 3))
        fill = theme.vehicle_rgbs[int(vehicle.color_index) % len(theme.vehicle_rgbs)]
        draw.rounded_rectangle(
            bbox,
            radius=max(6, int(round(h * 0.20))),
            fill=tuple(int(v) for v in fill) + (248,),
            outline=tuple(int(v) for v in theme.vehicle_outline_rgb) + (245,),
            width=3,
        )
        wheel_r = max(3.0, min(w, h) * 0.10)
        for wx in (bbox[0] + 0.24 * w, bbox[2] - 0.24 * w):
            draw.ellipse((wx - wheel_r, bbox[3] - wheel_r, wx + wheel_r, bbox[3] + wheel_r), fill=(24, 24, 24, 255))
        if vehicle.option_label is not None:
            label_w = max(22.0, w * 0.36)
            label_h = max(20.0, h * 0.58)
            label_bbox = (
                round(cx - (label_w / 2.0), 3),
                round(cy - (label_h / 2.0), 3),
                round(cx + (label_w / 2.0), 3),
                round(cy + (label_h / 2.0), 3),
            )
            draw.rounded_rectangle(
                label_bbox,
                radius=max(5, int(round(label_h * 0.22))),
                fill=(255, 255, 246, 226),
                outline=tuple(int(v) for v in theme.vehicle_outline_rgb) + (235,),
                width=2,
            )
            _fit_text(
                draw,
                bbox=label_bbox,
                text=str(vehicle.option_label),
                fill=(22, 28, 32),
                max_size_px=max(14, int(params.label_font_size_px)),
                font_family=str(params.font_family),
            )
            arrow_x = float(cx + (int(vehicle.direction) * w * 0.30))
            _draw_arrow(
                draw,
                center=(arrow_x, cy - (h * 0.25)),
                direction=int(vehicle.direction),
                color=theme.vehicle_outline_rgb,
                size=min(w, h) * 0.13,
            )
        else:
            _draw_arrow(
                draw,
                center=(cx, cy),
                direction=int(vehicle.direction),
                color=theme.vehicle_outline_rgb,
                size=min(w, h) * 0.22,
            )
        vehicle_bboxes[str(vehicle.vehicle_id)] = bbox
        entity_bboxes[str(vehicle.vehicle_id)] = bbox
        scene_entities.append(
            {
                "entity_id": str(vehicle.vehicle_id),
                "entity_type": "crossing_vehicle",
                "row": int(vehicle.row),
                "start_col": int(vehicle.start_col),
                "direction": int(vehicle.direction),
                "option_label": None if vehicle.option_label is None else str(vehicle.option_label),
                "bbox_px": list(bbox),
            }
        )

    render_map = {
        "scene_panel_bbox_px": [round(float(v), 3) for v in panel_bbox],
        "playfield_bbox_px": [round(float(v), 3) for v in playfield_bbox],
        "road_bbox_px": [round(left, 3), round(road_top, 3), round(right, 3), round(road_bottom, 3)],
        "cell_bboxes_px": {str(key): list(value) for key, value in cell_bboxes.items()},
        "start_bboxes_px": {str(key): list(value) for key, value in start_bboxes.items()},
        "vehicle_bboxes_px": {str(key): list(value) for key, value in vehicle_bboxes.items()},
        "vehicle_option_labels": {
            str(vehicle.vehicle_id): str(vehicle.option_label)
            for vehicle in vehicles
            if vehicle.option_label is not None
        },
        "route_bboxes_px": {str(key): list(value) for key, value in route_bboxes.items()},
        "route_cell_bboxes_px": {str(key): list(value) for key, value in route_cell_bboxes.items()},
        "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
        "layout_jitter": dict(layout_jitter),
        "panel_scene_style": {} if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "font_family": str(params.font_family),
    }
    return RenderedCrossingScene(
        image=image.convert("RGB"),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


def render_crossing_sample(
    *,
    sample: CrossingSample,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedCrossingTaskContext:
    """Render one crossing sample without binding a public answer."""

    render_params = resolve_crossing_render_params(params, instance_seed=int(instance_seed))
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.crossing.panel_scene_style",
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_crossing_scene(
        lane_count=int(sample.lane_count),
        row_count=int(sample.row_count),
        row_directions=tuple(int(value) for value in sample.row_directions),
        vehicles=tuple(sample.vehicles),
        start_labels=tuple(sample.start_labels),
        route_options=tuple(sample.route_options),
        marked_route_label=sample.marked_route_label,
        background=background,
        style_variant=str(sample.style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCrossingTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "CrossingRenderParams",
    "CrossingTheme",
    "RenderedCrossingScene",
    "RenderedCrossingTaskContext",
    "build_games_crossing_theme",
    "render_crossing_scene",
    "render_crossing_sample",
    "resolve_crossing_render_params",
]
