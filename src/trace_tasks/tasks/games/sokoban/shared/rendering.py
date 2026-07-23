"""Rendering-only helpers for Sokoban game scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.games.shared.layout import resolve_games_unit_size_scale, scale_games_px
from trace_tasks.tasks.shared.color_distance import coerce_rgb
from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import RENDER_DEFAULTS
from .rules import box_id, cell_id, option_id, target_id
from .state import (
    PATH_CONTRACT_KIND,
    PUSH_STAND_CONTRACT_KIND,
    RELATION_CONTRACT_KIND,
    RELATION_MODE_RANKED_PAIR,
    BBox,
    Cell,
    Color,
    RenderedSokobanScene,
    SokobanRenderParams,
)


SCENE_STYLES: Mapping[str, Dict[str, Color]] = {
    "warehouse_classic": {
        "panel": (248, 242, 231),
        "floor": (225, 194, 150),
        "floor_alt": (230, 203, 165),
        "wall": (190, 122, 58),
        "wall_dark": (130, 80, 42),
        "grid": (151, 105, 70),
        "border": (92, 70, 54),
        "box": (128, 72, 37),
        "box_light": (159, 91, 47),
        "target": (44, 157, 66),
        "player": (16, 18, 22),
        "option": (253, 250, 244),
        "accent": (54, 105, 178),
    },
    "paper_grid": {
        "panel": (248, 249, 252),
        "floor": (250, 247, 239),
        "floor_alt": (253, 250, 244),
        "wall": (150, 158, 170),
        "wall_dark": (94, 104, 118),
        "grid": (190, 198, 207),
        "border": (82, 91, 105),
        "box": (145, 100, 65),
        "box_light": (180, 128, 82),
        "target": (51, 149, 112),
        "player": (22, 25, 30),
        "option": (252, 253, 255),
        "accent": (176, 79, 86),
    },
    "cool_room": {
        "panel": (242, 248, 252),
        "floor": (224, 235, 243),
        "floor_alt": (232, 242, 248),
        "wall": (94, 132, 158),
        "wall_dark": (50, 79, 102),
        "grid": (156, 181, 199),
        "border": (58, 78, 94),
        "box": (135, 91, 72),
        "box_light": (171, 120, 91),
        "target": (60, 153, 119),
        "player": (18, 24, 30),
        "option": (250, 253, 255),
        "accent": (205, 132, 55),
    },
}


def _int_value(mapping: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(mapping.get(str(key), int(fallback)))


def resolve_sokoban_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int | None = None,
) -> SokobanRenderParams:
    """Resolve Sokoban render parameters from scene config and params."""

    merged = dict(RENDER_DEFAULTS)
    merged.update(dict(params))
    unit_scale, unit_meta = resolve_games_unit_size_scale(
        params,
        RENDER_DEFAULTS,
        instance_seed=instance_seed,
        namespace="games.sokoban.unit_size",
    )
    return SokobanRenderParams(
        canvas_width=_int_value(merged, "canvas_width", 1080),
        canvas_height=_int_value(merged, "canvas_height", 740),
        scene_margin_left_px=_int_value(merged, "scene_margin_left_px", 52),
        scene_margin_top_px=_int_value(merged, "scene_margin_top_px", 48),
        board_panel_width_px=_int_value(merged, "board_panel_width_px", 590),
        board_panel_height_px=_int_value(merged, "board_panel_height_px", 620),
        option_panel_width_px=_int_value(merged, "option_panel_width_px", 170),
        option_panel_height_px=_int_value(merged, "option_panel_height_px", 150),
        option_gap_px=_int_value(merged, "option_gap_px", 18),
        option_row_gap_px=_int_value(merged, "option_row_gap_px", 18),
        panel_corner_radius_px=scale_games_px(_int_value(merged, "panel_corner_radius_px", 22), unit_scale, min_px=8),
        board_border_width_px=scale_games_px(_int_value(merged, "board_border_width_px", 4), unit_scale, min_px=2),
        grid_width_px=scale_games_px(_int_value(merged, "grid_width_px", 2), unit_scale, min_px=1),
        coord_gutter_px=scale_games_px(_int_value(merged, "coord_gutter_px", 34), unit_scale, min_px=16),
        main_cell_size_px=scale_games_px(_int_value(merged, "main_cell_size_px", 54), unit_scale, min_px=48),
        mini_cell_size_px=scale_games_px(_int_value(merged, "mini_cell_size_px", 15), unit_scale, min_px=7),
        option_label_font_size_px=scale_games_px(_int_value(merged, "option_label_font_size_px", 28), unit_scale, min_px=14),
        cell_label_font_size_px=scale_games_px(_int_value(merged, "cell_label_font_size_px", 18), unit_scale, min_px=10),
        sequence_font_size_px=scale_games_px(_int_value(merged, "sequence_font_size_px", 22), unit_scale, min_px=11),
        text_color_rgb=coerce_rgb(merged.get("text_color_rgb"), (28, 32, 38)),
        text_stroke_rgb=coerce_rgb(merged.get("text_stroke_rgb"), (255, 255, 255)),
        style_overrides={},
        unit_size_jitter=dict(unit_meta),
    )


def apply_panel_style(render_params: SokobanRenderParams, scene_style: Any) -> SokobanRenderParams:
    """Adapt shared games panel colors into Sokoban-specific style slots."""

    return replace(
        render_params,
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        style_overrides={
            "panel": tuple(int(value) for value in scene_style.panel_fill_rgb),
            "floor": tuple(int(value) for value in scene_style.option_fill_rgb),
            "floor_alt": tuple(int(value) for value in scene_style.step_fill_rgb),
            "wall": tuple(int(value) for value in scene_style.grid_rgb),
            "wall_dark": tuple(int(value) for value in scene_style.panel_border_rgb),
            "grid": tuple(int(value) for value in scene_style.notebook_line_rgb),
            "border": tuple(int(value) for value in scene_style.panel_border_rgb),
            "box": tuple(int(value) for value in scene_style.panel_accent_rgb),
            "box_light": tuple(int(value) for value in scene_style.agent_rgb),
            "target": tuple(int(value) for value in scene_style.mark_rgb),
            "player": tuple(int(value) for value in scene_style.text_rgb),
            "option": tuple(int(value) for value in scene_style.panel_fill_rgb),
            "accent": tuple(int(value) for value in scene_style.mark_rgb),
        },
    )


def _bbox_union(boxes: Iterable[Sequence[float]]) -> BBox:
    items = [tuple(float(value) for value in box) for box in boxes]
    if not items:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        round(min(box[0] for box in items), 3),
        round(min(box[1] for box in items), 3),
        round(max(box[2] for box in items), 3),
        round(max(box[3] for box in items), 3),
    )


def _cell_bbox(origin: Tuple[float, float], cell_size: float, cell: Cell) -> BBox:
    x0 = float(origin[0] + (cell[1] * cell_size))
    y0 = float(origin[1] + (cell[0] * cell_size))
    return (round(x0, 3), round(y0, 3), round(x0 + cell_size, 3), round(y0 + cell_size, 3))


def _draw_x(draw: ImageDraw.ImageDraw, bbox: BBox, *, fill: Color, width: int) -> None:
    pad = max(3.0, min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])) * 0.22)
    draw.line((bbox[0] + pad, bbox[1] + pad, bbox[2] - pad, bbox[3] - pad), fill=fill, width=max(1, int(width)))
    draw.line((bbox[0] + pad, bbox[3] - pad, bbox[2] - pad, bbox[1] + pad), fill=fill, width=max(1, int(width)))


def _mix_color(a: Color, b: Color, amount: float) -> Color:
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(int(round((float(a[index]) * (1.0 - amount)) + (float(b[index]) * amount))) for index in range(3))


def _color_map(raw: Any) -> Dict[str, Color]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): coerce_rgb(value, (120, 120, 120)) for key, value in raw.items()}


def _draw_check(draw: ImageDraw.ImageDraw, bbox: BBox, *, fill: Color, shadow: Color, width: int) -> None:
    size = min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    points = (
        (float(bbox[0]) + size * 0.26, float(bbox[1]) + size * 0.54),
        (float(bbox[0]) + size * 0.43, float(bbox[1]) + size * 0.69),
        (float(bbox[0]) + size * 0.74, float(bbox[1]) + size * 0.33),
    )
    draw.line(points, fill=shadow, width=max(2, int(width) + 2), joint="curve")
    draw.line(points, fill=fill, width=max(1, int(width)), joint="curve")


def _draw_player(draw: ImageDraw.ImageDraw, bbox: BBox, *, fill: Color, width: int) -> None:
    cx = (float(bbox[0]) + float(bbox[2])) * 0.5
    cy = (float(bbox[1]) + float(bbox[3])) * 0.5
    size = min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    head_r = size * 0.13
    draw.ellipse((cx - head_r, cy - size * 0.29, cx + head_r, cy - size * 0.03), fill=fill)
    draw.line((cx, cy - size * 0.02, cx, cy + size * 0.24), fill=fill, width=max(1, int(width)))
    draw.line((cx - size * 0.18, cy + size * 0.06, cx + size * 0.18, cy + size * 0.06), fill=fill, width=max(1, int(width)))
    draw.line((cx, cy + size * 0.24, cx - size * 0.16, cy + size * 0.42), fill=fill, width=max(1, int(width)))
    draw.line((cx, cy + size * 0.24, cx + size * 0.16, cy + size * 0.42), fill=fill, width=max(1, int(width)))


def _draw_option_badge(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    label: str,
    style: Mapping[str, Color],
    params: SokobanRenderParams,
) -> None:
    size = min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    radius = size * 0.24
    cx = (float(bbox[0]) + float(bbox[2])) * 0.5
    cy = (float(bbox[1]) + float(bbox[3])) * 0.5
    badge = (cx - radius, cy - radius, cx + radius, cy + radius)
    font = load_font(max(10, int(size * 0.34)), bold=True)
    draw.ellipse(badge, fill=(255, 255, 255), outline=style["accent"], width=max(2, int(size * 0.06)))
    draw_centered_text(
        draw,
        text=str(label),
        center=(cx, cy),
        font=font,
        fill=params.text_color_rgb,
        stroke_fill=params.text_stroke_rgb,
        stroke_width=1,
    )


def _draw_relation_option_overlays(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    cell_bbox_map: Mapping[str, BBox],
    style: Mapping[str, Color],
    params: SokobanRenderParams,
) -> None:
    for option in list(dataset.get("option_specs", [])):
        cells = [tuple(cell) for cell in option.get("candidate_cells", [])]
        for cell in cells:
            bbox = cell_bbox_map.get(cell_id(cell))
            if bbox is not None:
                _draw_option_badge(
                    draw,
                    bbox=bbox,
                    label=str(option.get("option_label", "")),
                    style=style,
                    params=params,
                )


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: BBox,
    *,
    fill: Color,
    stroke_fill: Color,
    max_size: int,
    min_size: int = 10,
    bold: bool = True,
) -> None:
    """Fit short option text inside a fixed panel without changing layout geometry."""

    words = str(text).split()
    if not words:
        return
    for size in range(int(max_size), int(min_size) - 1, -1):
        font = load_font(size, bold=bold)
        lines: List[str] = []
        current = ""
        max_width = float(bbox[2] - bbox[0]) - 10.0
        for word in words:
            candidate = f"{current} {word}".strip()
            width = draw.textbbox((0, 0), candidate, font=font, stroke_width=1)[2]
            if width <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        line_height = max(10, int(size * 1.18))
        total_height = line_height * len(lines)
        if total_height <= float(bbox[3] - bbox[1]) - 8:
            y = float(bbox[1]) + (float(bbox[3] - bbox[1]) - total_height) * 0.5
            for line in lines:
                text_bbox = draw.textbbox((0, 0), line, font=font, stroke_width=1)
                x = float(bbox[0]) + (float(bbox[2] - bbox[0]) - float(text_bbox[2] - text_bbox[0])) * 0.5
                draw_text_traced(
                    draw,
                    (x, y),
                    line,
                    fill=fill,
                    font=font,
                    stroke_width=1,
                    stroke_fill=stroke_fill,
                    role="readout",
                    required=False,
                )
                y += line_height
            return


def _draw_board(
    draw: ImageDraw.ImageDraw,
    *,
    rows: int,
    cols: int,
    walls: set[Cell],
    boxes: Mapping[str, Cell],
    targets: Mapping[str, Cell],
    box_colors: Mapping[str, Color] | None = None,
    target_colors: Mapping[str, Color] | None = None,
    matching_targets: Mapping[str, str] | None = None,
    player: Cell | None,
    origin: Tuple[float, float],
    cell_size: float,
    style: Mapping[str, Color],
    params: SokobanRenderParams,
    show_coordinates: bool,
    show_labels: bool,
    show_box_labels: bool = False,
    marked_box_label: str = "",
    marked_target_label: str = "",
    start_cell: Cell | None = None,
    goal_cell: Cell | None = None,
    candidate_cell: Cell | None = None,
) -> Dict[str, BBox]:
    """Draw the primary grid and return every cell bbox."""

    cell_bbox_map: Dict[str, BBox] = {}
    label_font = load_font(max(9, int(cell_size * 0.30)), bold=True)
    small_font = load_font(max(8, int(cell_size * 0.24)), bold=True)
    box_color_map = {str(key): tuple(value) for key, value in dict(box_colors or {}).items()}
    target_color_map = {str(key): tuple(value) for key, value in dict(target_colors or {}).items()}
    matching_target_map = {str(key): str(value) for key, value in dict(matching_targets or {}).items()}
    targets_by_label = {str(label): tuple(cell) for label, cell in targets.items()}
    for row in range(int(rows)):
        for col in range(int(cols)):
            cell = (row, col)
            bbox = _cell_bbox(origin, cell_size, cell)
            cell_bbox_map[cell_id(cell)] = bbox
            fill = style["wall"] if cell in walls else style["floor"]
            outline = style["grid"] if cell not in walls else style["wall_dark"]
            draw.rectangle(bbox, fill=fill, outline=outline, width=max(1, int(params.grid_width_px)))
    if show_coordinates:
        coord_font = load_font(max(9, int(params.cell_label_font_size_px)), bold=True)
        for col in range(int(cols)):
            bbox = _cell_bbox(origin, cell_size, (0, col))
            draw_centered_text(
                draw,
                text=str(col),
                center=((bbox[0] + bbox[2]) * 0.5, float(origin[1]) - 15),
                font=coord_font,
                fill=params.text_color_rgb,
                stroke_fill=params.text_stroke_rgb,
                stroke_width=1,
            )
        for row in range(int(rows)):
            bbox = _cell_bbox(origin, cell_size, (row, 0))
            draw_centered_text(
                draw,
                text=str(row),
                center=(float(origin[0]) - 16, (bbox[1] + bbox[3]) * 0.5),
                font=coord_font,
                fill=params.text_color_rgb,
                stroke_fill=params.text_stroke_rgb,
                stroke_width=1,
            )
    for target_label, cell in targets.items():
        bbox = cell_bbox_map[cell_id(tuple(cell))]
        target_color = target_color_map.get(str(target_label), style["target"])
        ring = (
            bbox[0] + cell_size * 0.22,
            bbox[1] + cell_size * 0.22,
            bbox[2] - cell_size * 0.22,
            bbox[3] - cell_size * 0.22,
        )
        dot = (
            bbox[0] + cell_size * 0.38,
            bbox[1] + cell_size * 0.38,
            bbox[2] - cell_size * 0.38,
            bbox[3] - cell_size * 0.38,
        )
        draw.ellipse(ring, outline=target_color, width=max(2, int(cell_size * 0.08)))
        draw.ellipse(dot, fill=target_color)
        if show_labels:
            draw_centered_text(
                draw,
                text=str(target_label),
                center=(bbox[2] - cell_size * 0.22, bbox[1] + cell_size * 0.22),
                font=small_font,
                fill=params.text_color_rgb,
                stroke_fill=params.text_stroke_rgb,
                stroke_width=1,
            )
        if str(target_label) == str(marked_target_label):
            draw.rectangle(bbox, outline=style["accent"], width=max(3, int(cell_size * 0.08)))
    if start_cell is not None:
        bbox = cell_bbox_map[cell_id(tuple(start_cell))]
        draw.ellipse(
            (bbox[0] + cell_size * 0.24, bbox[1] + cell_size * 0.24, bbox[2] - cell_size * 0.24, bbox[3] - cell_size * 0.24),
            fill=(255, 255, 255),
            outline=style["accent"],
            width=max(2, int(cell_size * 0.07)),
        )
        draw_centered_text(draw, text="S", center=((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5), font=label_font, fill=style["accent"], stroke_fill=(255, 255, 255), stroke_width=1)
    if goal_cell is not None:
        bbox = cell_bbox_map[cell_id(tuple(goal_cell))]
        draw.ellipse(
            (bbox[0] + cell_size * 0.20, bbox[1] + cell_size * 0.20, bbox[2] - cell_size * 0.20, bbox[3] - cell_size * 0.20),
            fill=(255, 255, 255),
            outline=style["target"],
            width=max(2, int(cell_size * 0.07)),
        )
        draw_centered_text(draw, text="G", center=((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5), font=label_font, fill=style["target"], stroke_fill=(255, 255, 255), stroke_width=1)
    for box_label, cell in boxes.items():
        bbox = cell_bbox_map[cell_id(tuple(cell))]
        pad = max(3.0, cell_size * 0.12)
        box_bbox = (bbox[0] + pad, bbox[1] + pad, bbox[2] - pad, bbox[3] - pad)
        box_color = box_color_map.get(str(box_label), style["box"])
        box_light = _mix_color(box_color, (255, 255, 255), 0.28)
        box_dark = _mix_color(box_color, (25, 28, 34), 0.38)
        draw.rectangle(box_bbox, fill=box_light, outline=box_dark, width=max(2, int(cell_size * 0.05)))
        inner = (
            box_bbox[0] + cell_size * 0.10,
            box_bbox[1] + cell_size * 0.10,
            box_bbox[2] - cell_size * 0.10,
            box_bbox[3] - cell_size * 0.10,
        )
        draw.rectangle(inner, outline=box_color, width=max(2, int(cell_size * 0.04)))
        draw.line((box_bbox[0], box_bbox[1], box_bbox[2], box_bbox[3]), fill=box_dark, width=max(1, int(cell_size * 0.035)))
        draw.line((box_bbox[0], box_bbox[3], box_bbox[2], box_bbox[1]), fill=box_dark, width=max(1, int(cell_size * 0.035)))
        target_label = matching_target_map.get(str(box_label))
        on_matching_goal = bool(
            target_label
            and target_label in targets_by_label
            and tuple(targets_by_label[target_label]) == tuple(cell)
        )
        if on_matching_goal:
            target_color = target_color_map.get(str(target_label), box_color)
            center = ((box_bbox[0] + box_bbox[2]) * 0.5, (box_bbox[1] + box_bbox[3]) * 0.5)
            radius = cell_size * 0.23
            dot = (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            )
            draw.ellipse(dot, fill=target_color, outline=(255, 255, 255), width=max(3, int(cell_size * 0.07)))
        if show_labels or show_box_labels:
            draw_centered_text(
                draw,
                text=str(box_label),
                center=((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5),
                font=label_font if show_box_labels else small_font,
                fill=(255, 255, 255),
                stroke_fill=box_dark,
                stroke_width=max(1, int(cell_size * 0.045)),
            )
        if str(box_label) == str(marked_box_label):
            draw.rectangle(bbox, outline=style["accent"], width=max(3, int(cell_size * 0.08)))
    if player is not None:
        bbox = cell_bbox_map[cell_id(tuple(player))]
        _draw_player(draw, bbox, fill=style["player"], width=max(2, int(cell_size * 0.05)))
    if candidate_cell is not None:
        bbox = cell_bbox_map[cell_id(tuple(candidate_cell))]
        draw.ellipse(
            (bbox[0] + cell_size * 0.18, bbox[1] + cell_size * 0.18, bbox[2] - cell_size * 0.18, bbox[3] - cell_size * 0.18),
            outline=style["accent"],
            width=max(3, int(cell_size * 0.10)),
        )
    return cell_bbox_map


def _draw_sokoban_option(
    draw: ImageDraw.ImageDraw,
    *,
    option: Mapping[str, Any],
    bbox: BBox,
    dataset: Mapping[str, Any],
    style: Mapping[str, Color],
    params: SokobanRenderParams,
) -> None:
    """Draw one MCQ option panel as either a mini-board snapshot or wrapped move text."""

    draw_rounded_rect(draw, bbox, radius=int(params.panel_corner_radius_px), fill=style["option"], outline=style["border"], width=2)
    label = str(option["option_label"])
    label_font = load_font(int(params.option_label_font_size_px), bold=True)
    draw_centered_text(draw, text=label, center=(float(bbox[0]) + 24, float(bbox[1]) + 24), font=label_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=1)
    kind = str(option.get("kind", ""))
    if kind == "cell_snapshot":
        rows, cols = int(dataset["rows"]), int(dataset["cols"])
        mini_size = min(
            (float(bbox[2] - bbox[0]) - 26) / max(1, cols),
            (float(bbox[3] - bbox[1]) - 58) / max(1, rows),
            float(params.mini_cell_size_px),
        )
        origin = (
            float(bbox[0]) + (float(bbox[2] - bbox[0]) - mini_size * cols) * 0.5,
            float(bbox[1]) + 48,
        )
        _draw_board(
            draw,
            rows=rows,
            cols=cols,
            walls={tuple(cell) for cell in dataset["walls"]},
            boxes={},
            targets={},
            player=None,
            origin=origin,
            cell_size=mini_size,
            style=style,
            params=params,
            show_coordinates=False,
            show_labels=False,
            show_box_labels=False,
            candidate_cell=tuple(option["candidate_cell"]),
        )
    else:
        _draw_wrapped_text(
            draw,
            text=str(option.get("display_text", "")),
            bbox=(bbox[0] + 12, bbox[1] + 44, bbox[2] - 12, bbox[3] - 12),
            fill=params.text_color_rgb,
            stroke_fill=params.text_stroke_rgb,
            max_size=int(params.sequence_font_size_px),
            min_size=10,
            bold=True,
        )


def render_sokoban_scene(
    base_image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    scene_variant: str,
    render_params: SokobanRenderParams,
) -> RenderedSokobanScene:
    """Render one Sokoban scene and return traceable bboxes."""

    image = base_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    style = dict(SCENE_STYLES.get(str(scene_variant), SCENE_STYLES["warehouse_classic"]))
    style.update({str(key): tuple(value) for key, value in dict(render_params.style_overrides or {}).items()})
    rows, cols = int(dataset["rows"]), int(dataset["cols"])
    contract_kind = str(dataset.get("contract_kind"))
    is_relation_family = contract_kind == RELATION_CONTRACT_KIND
    is_push_stand_family = contract_kind == PUSH_STAND_CONTRACT_KIND
    uses_side_options = contract_kind == PATH_CONTRACT_KIND
    uses_board_options = bool(is_relation_family or is_push_stand_family)
    board_x0 = (
        float(render_params.scene_margin_left_px)
        if uses_side_options
        else float(render_params.scene_margin_left_px)
    )
    if not uses_side_options:
        board_x0 = float(render_params.canvas_width - render_params.board_panel_width_px) * 0.5
    board_panel = (
        board_x0,
        float(render_params.scene_margin_top_px),
        float(board_x0 + render_params.board_panel_width_px),
        float(render_params.scene_margin_top_px + render_params.board_panel_height_px),
    )
    available_w = float(render_params.board_panel_width_px)
    available_h = float(render_params.board_panel_height_px)
    cell_size = min(float(render_params.main_cell_size_px), available_w / max(1, cols), available_h / max(1, rows))
    board_origin = (
        float(board_panel[0] + (available_w - (cell_size * cols)) * 0.5),
        float(board_panel[1] + (available_h - (cell_size * rows)) * 0.5),
    )
    start_cell = tuple(dataset["path_start"]) if "path_start" in dataset else None
    goal_cell = tuple(dataset["path_goal"]) if "path_goal" in dataset else None
    cell_bbox_map = _draw_board(
        draw,
        rows=rows,
        cols=cols,
        walls={tuple(cell) for cell in dataset["walls"]},
        boxes={str(k): tuple(v) for k, v in dict(dataset.get("boxes_start", {})).items()},
        targets={str(k): tuple(v) for k, v in dict(dataset.get("targets", {})).items()},
        box_colors=_color_map(dataset.get("box_colors")),
        target_colors=_color_map(dataset.get("target_colors")),
        matching_targets={str(k): str(v) for k, v in dict(dataset.get("matching_targets", {})).items()},
        player=tuple(dataset["player_start"]) if "player_start" in dataset else None,
        origin=board_origin,
        cell_size=cell_size,
        style=style,
        params=render_params,
        show_coordinates=False,
        show_labels=False,
        show_box_labels=bool(dataset.get("show_box_labels", False)),
        marked_box_label=str(dataset.get("marked_box_label", "")),
        marked_target_label=str(dataset.get("marked_target_label", "")),
        start_cell=start_cell,
        goal_cell=goal_cell,
    )
    if uses_board_options:
        _draw_relation_option_overlays(draw, dataset=dataset, cell_bbox_map=cell_bbox_map, style=style, params=render_params)
    board_bbox = _bbox_union(cell_bbox_map.values())
    subtitle_font = load_font(16, bold=False)
    if str(dataset.get("contract_kind")) == PATH_CONTRACT_KIND:
        draw_text_traced(draw, (board_panel[0] + 22, board_panel[3] - 56), "Move codes: U=up, D=down, L=left, R=right.", fill=render_params.text_color_rgb, font=subtitle_font, role="readout", required=False)
        draw_text_traced(draw, (board_panel[0] + 22, board_panel[3] - 82), "Boxes count as blockers for these path options.", fill=render_params.text_color_rgb, font=subtitle_font, role="readout", required=False)
    elif str(dataset.get("relation_mode")) == RELATION_MODE_RANKED_PAIR:
        rank_word = str(dataset.get("relation_support", {}).get("rank_word", "requested"))
        draw_text_traced(draw, (board_panel[0] + 22, board_panel[3] - 82), f"Compare same-letter box-target pairs; find the {rank_word} closest pair.", fill=render_params.text_color_rgb, font=subtitle_font, role="readout", required=False)

    option_panel_bbox_map: Dict[str, BBox] = {}
    if not uses_board_options:
        option_x0 = float(board_panel[2] + 34)
        option_y0 = float(board_panel[1] + 16)
        for index, option in enumerate(dataset["option_specs"]):
            col = index % 2
            row = index // 2
            x0 = option_x0 + col * (render_params.option_panel_width_px + render_params.option_gap_px)
            y0 = option_y0 + row * (render_params.option_panel_height_px + render_params.option_row_gap_px)
            bbox = (
                round(x0, 3),
                round(y0, 3),
                round(x0 + render_params.option_panel_width_px, 3),
                round(y0 + render_params.option_panel_height_px, 3),
            )
            _draw_sokoban_option(draw, option=option, bbox=bbox, dataset=dataset, style=style, params=render_params)
            option_panel_bbox_map[option_id(str(option["option_label"]))] = bbox

    entities: List[Dict[str, Any]] = []
    for cell_key, bbox in cell_bbox_map.items():
        entities.append({"entity_id": cell_key, "type": "sokoban_cell", "bbox_px": list(bbox)})
    for label, cell in dict(dataset.get("boxes_start", {})).items():
        entity = {"entity_id": box_id(str(label)), "type": "sokoban_box", "cell": list(cell), "bbox_px": list(cell_bbox_map[cell_id(tuple(cell))])}
        if str(label) in dict(dataset.get("box_colors", {})):
            entity["color_rgb"] = list(dataset["box_colors"][str(label)])
        if str(label) in dict(dataset.get("matching_targets", {})):
            entity["matching_target"] = str(dataset["matching_targets"][str(label)])
        entities.append(entity)
    for label, cell in dict(dataset.get("targets", {})).items():
        entity = {"entity_id": target_id(str(label)), "type": "sokoban_target", "cell": list(cell), "bbox_px": list(cell_bbox_map[cell_id(tuple(cell))])}
        if str(label) in dict(dataset.get("target_colors", {})):
            entity["color_rgb"] = list(dataset["target_colors"][str(label)])
        entities.append(entity)
    if "player_start" in dataset:
        entities.append({"entity_id": "player", "type": "sokoban_player", "cell": list(dataset["player_start"]), "bbox_px": list(cell_bbox_map[cell_id(tuple(dataset["player_start"]))])})
    if uses_board_options:
        for option in dataset.get("option_specs", []):
            cells = [tuple(cell) for cell in option.get("candidate_cells", [])]
            if not cells:
                continue
            bboxes = [cell_bbox_map[cell_id(cell)] for cell in cells]
            entities.append(
                {
                    "entity_id": option_id(str(option["option_label"])),
                    "type": "sokoban_board_option",
                    "option_label": str(option["option_label"]),
                    "candidate_cells": [list(cell) for cell in cells],
                    "bbox_px": list(_bbox_union(bboxes)),
                }
            )
    for option_entity_id, bbox in option_panel_bbox_map.items():
        entities.append({"entity_id": option_entity_id, "type": "sokoban_option_panel", "bbox_px": list(bbox)})
    scene_bbox = _bbox_union([board_bbox, *option_panel_bbox_map.values()])
    return RenderedSokobanScene(
        image=image,
        entities=tuple(entities),
        scene_bbox_px=scene_bbox,
        board_bbox_px=board_bbox,
        option_panel_bbox_map=dict(option_panel_bbox_map),
        cell_bbox_map=dict(cell_bbox_map),
    )


__all__ = [
    "SCENE_STYLES",
    "apply_panel_style",
    "render_sokoban_scene",
    "resolve_sokoban_render_params",
]
