"""Rendering helpers for slot-machine games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from trace_tasks.tasks.games.shared.scene_style import (
    game_panel_scene_style_metadata,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_centered_game_text
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.core.visual.noise import apply_post_image_noise

from .defaults import DEFAULTS, PAYLINE_IDS, POST_IMAGE_NOISE_DEFAULTS, REEL_COUNT, ROW_COUNT, SCENE_NAMESPACE
from .state import (
    PAYLINE_CELLS_BY_ID,
    SlotCompletionScene,
    SlotMachineScene,
    cell_grid,
    completion_option_grid,
    payline_entity_id,
    slot_cell_id,
    validate_slot_completion_scene,
    validate_slot_machine_scene,
)


@dataclass(frozen=True)
class SlotMachineRenderParams:
    """Resolved slot-machine render dimensions."""

    canvas_width: int
    canvas_height: int
    cabinet_width_px: int
    cabinet_height_px: int
    reel_cell_width_px: int
    reel_cell_height_px: int
    reel_gap_px: int
    row_gap_px: int
    cabinet_pad_px: int
    label_font_size_px: int
    symbol_font_size_px: int


@dataclass(frozen=True)
class RenderedSlotMachineScene:
    """Rendered slot-machine image plus projection data."""

    image: Image.Image
    render_map: dict[str, Any]
    scene_entities: tuple[dict[str, Any], ...]
    panel_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]


def resolve_slot_machine_render_params(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> SlotMachineRenderParams:
    """Resolve pixel dimensions from params, config defaults, and fallbacks."""

    return SlotMachineRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))),
        canvas_height=int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))),
        cabinet_width_px=int(params.get("cabinet_width_px", group_default(render_defaults, "cabinet_width_px", DEFAULTS.cabinet_width_px))),
        cabinet_height_px=int(params.get("cabinet_height_px", group_default(render_defaults, "cabinet_height_px", DEFAULTS.cabinet_height_px))),
        reel_cell_width_px=int(params.get("reel_cell_width_px", group_default(render_defaults, "reel_cell_width_px", DEFAULTS.reel_cell_width_px))),
        reel_cell_height_px=int(params.get("reel_cell_height_px", group_default(render_defaults, "reel_cell_height_px", DEFAULTS.reel_cell_height_px))),
        reel_gap_px=int(params.get("reel_gap_px", group_default(render_defaults, "reel_gap_px", DEFAULTS.reel_gap_px))),
        row_gap_px=int(params.get("row_gap_px", group_default(render_defaults, "row_gap_px", DEFAULTS.row_gap_px))),
        cabinet_pad_px=int(params.get("cabinet_pad_px", group_default(render_defaults, "cabinet_pad_px", DEFAULTS.cabinet_pad_px))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px))),
        symbol_font_size_px=int(params.get("symbol_font_size_px", group_default(render_defaults, "symbol_font_size_px", DEFAULTS.symbol_font_size_px))),
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    try:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(name, int(size))
    except OSError:
        return ImageFont.load_default()


def _style_colors(style_variant: str) -> dict[str, tuple[int, int, int]]:
    """Return slot-cabinet colors for one visual style variant."""

    palettes = {
        "classic_red": {
            "cabinet": (154, 37, 45),
            "cabinet_dark": (87, 22, 31),
            "trim": (247, 198, 69),
            "reel": (250, 246, 231),
            "payline": (230, 42, 58),
        },
        "chrome_blue": {
            "cabinet": (59, 92, 137),
            "cabinet_dark": (28, 43, 68),
            "trim": (205, 214, 224),
            "reel": (239, 245, 249),
            "payline": (224, 61, 77),
        },
        "neon_night": {
            "cabinet": (38, 31, 76),
            "cabinet_dark": (15, 13, 34),
            "trim": (61, 218, 209),
            "reel": (232, 236, 244),
            "payline": (255, 72, 133),
        },
        "candy_arcade": {
            "cabinet": (205, 83, 132),
            "cabinet_dark": (116, 45, 83),
            "trim": (255, 215, 108),
            "reel": (255, 248, 238),
            "payline": (61, 131, 214),
        },
        "paper_ticket": {
            "cabinet": (118, 97, 78),
            "cabinet_dark": (73, 58, 46),
            "trim": (226, 188, 116),
            "reel": (251, 244, 224),
            "payline": (179, 47, 58),
        },
    }
    return palettes.get(str(style_variant), palettes["classic_red"])


def _draw_symbol(draw: ImageDraw.ImageDraw, bbox: Sequence[float], symbol_key: str, *, colors: Mapping[str, tuple[int, int, int]]) -> None:
    """Draw one simple slot symbol inside a reel cell."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    size = min(x1 - x0, y1 - y0) * 0.56
    key = str(symbol_key)
    outline = (42, 42, 48)
    if key == "seven":
        font = _font(int(size * 0.82), bold=True)
        draw_centered_game_text(
            draw,
            text="7",
            center=(cx, cy - size * 0.03),
            font=font,
            fill=(204, 34, 45),
            stroke_fill=(255, 245, 210),
            surface_rgbs=(colors["reel"],),
            role="game_symbol",
            required=True,
            stroke_width=1,
        )
    elif key == "bar":
        rect = [cx - size * 0.48, cy - size * 0.22, cx + size * 0.48, cy + size * 0.22]
        draw.rounded_rectangle(rect, radius=int(size * 0.08), fill=(32, 36, 42), outline=(228, 193, 72), width=3)
        draw_centered_game_text(
            draw,
            text="BAR",
            center=(cx, cy),
            font=_font(int(size * 0.28), bold=True),
            fill=(248, 236, 180),
            stroke_fill=(32, 36, 42),
            surface_rgbs=((32, 36, 42),),
            role="game_symbol",
            required=True,
        )
    elif key == "gem":
        points = [(cx, cy - size * 0.45), (cx + size * 0.45, cy), (cx, cy + size * 0.45), (cx - size * 0.45, cy)]
        draw.polygon(points, fill=(50, 145, 214), outline=outline)
    elif key == "star":
        r1 = size * 0.45
        r2 = size * 0.20
        pts = []
        import math

        for index in range(10):
            angle = -math.pi / 2.0 + index * math.pi / 5.0
            radius = r1 if index % 2 == 0 else r2
            pts.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        draw.polygon(pts, fill=(245, 181, 46), outline=outline)
    elif key == "cherry":
        left_cherry = [cx - size * 0.44, cy - size * 0.08, cx - size * 0.04, cy + size * 0.34]
        right_cherry = [cx + size * 0.02, cy - size * 0.02, cx + size * 0.42, cy + size * 0.40]
        draw.line([(cx - size * 0.22, cy - size * 0.08), (cx, cy - size * 0.46), (cx + size * 0.22, cy - size * 0.04)], fill=(52, 113, 59), width=max(2, int(size * 0.06)))
        draw.ellipse(left_cherry, fill=(203, 38, 49), outline=outline, width=2)
        draw.ellipse(right_cherry, fill=(221, 50, 60), outline=outline, width=2)
        draw.ellipse([cx + size * 0.02, cy - size * 0.54, cx + size * 0.34, cy - size * 0.28], fill=(72, 151, 77), outline=outline, width=1)
    else:
        draw.ellipse([cx - size * 0.42, cy - size * 0.42, cx + size * 0.42, cy + size * 0.42], fill=(236, 149, 48), outline=outline, width=3)
        draw.ellipse([cx - size * 0.22, cy - size * 0.22, cx + size * 0.22, cy + size * 0.22], fill=(255, 210, 88), outline=(166, 89, 26), width=2)


def _draw_paytable(
    draw: ImageDraw.ImageDraw,
    *,
    scene: SlotMachineScene,
    cabinet_right: float,
    cabinet_top: float,
    colors: Mapping[str, tuple[int, int, int]],
) -> tuple[dict[str, list[float]], list[dict[str, Any]], list[float] | None]:
    """Draw a side paytable when the scene carries score entries."""

    if not scene.paytable_entries:
        return {}, [], None
    panel_left = float(cabinet_right) + 26.0
    panel_top = float(cabinet_top) + 84.0
    panel_width = 170.0
    title_height = 44.0
    row_height = 48.0
    panel_height = title_height + row_height * len(scene.paytable_entries) + 18.0
    panel_bbox = [panel_left, panel_top, panel_left + panel_width, panel_top + panel_height]
    draw.rounded_rectangle(panel_bbox, radius=18, fill=(250, 246, 231), outline=colors["trim"], width=4)
    draw_centered_game_text(
        draw,
        text="PAYTABLE",
        center=(panel_left + panel_width / 2.0, panel_top + title_height / 2.0 + 2.0),
        font=_font(18, bold=True),
        fill=colors["cabinet_dark"],
        stroke_fill=(250, 246, 231),
        surface_rgbs=((250, 246, 231),),
        role="readout",
        required=True,
    )

    entry_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    for index, entry in enumerate(scene.paytable_entries):
        row_top = panel_top + title_height + index * row_height + 4.0
        row_bbox = [panel_left + 10.0, row_top, panel_left + panel_width - 10.0, row_top + row_height - 6.0]
        draw.rounded_rectangle(row_bbox, radius=10, fill=(255, 252, 241), outline=(208, 193, 160), width=1)
        symbol_bbox = [row_bbox[0] + 8.0, row_bbox[1] + 4.0, row_bbox[0] + 46.0, row_bbox[3] - 4.0]
        _draw_symbol(draw, symbol_bbox, str(entry.symbol_key), colors=colors)
        draw_centered_game_text(
            draw,
            text=f"= {int(entry.score_value)}",
            center=(row_bbox[0] + 104.0, (row_bbox[1] + row_bbox[3]) / 2.0),
            font=_font(22, bold=True),
            fill=colors["cabinet_dark"],
            stroke_fill=(255, 252, 241),
            surface_rgbs=((255, 252, 241),),
            role="readout",
            required=True,
        )
        key = f"paytable_{entry.symbol_key}"
        bbox = [round(float(value), 3) for value in row_bbox]
        entry_bboxes[key] = bbox
        entities.append(
            {
                "id": key,
                "kind": "paytable_entry",
                "symbol_key": str(entry.symbol_key),
                "score_value": int(entry.score_value),
                "bbox_px": bbox,
            }
        )
    return entry_bboxes, entities, [round(float(value), 3) for value in panel_bbox]


def render_slot_machine_scene(
    *,
    scene: SlotMachineScene,
    render_params: SlotMachineRenderParams,
    instance_seed: int,
) -> RenderedSlotMachineScene:
    """Render a front-view slot machine and record conceptual payline projections."""

    validate_slot_machine_scene(scene)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_style",
        treatments=("bare_canvas", "plain_sheet", "soft_panel", "game_table", "arcade_screen"),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    draw = ImageDraw.Draw(image)
    colors = _style_colors(str(scene.style_variant))
    cabinet_w = float(render_params.cabinet_width_px)
    cabinet_h = float(render_params.cabinet_height_px)
    left = (float(render_params.canvas_width) - cabinet_w) / 2.0
    top = (float(render_params.canvas_height) - cabinet_h) / 2.0
    right = left + cabinet_w
    bottom = top + cabinet_h

    draw.rounded_rectangle([left, top, right, bottom], radius=34, fill=colors["cabinet_dark"], outline=colors["trim"], width=6)
    inset = 18
    draw.rounded_rectangle([left + inset, top + inset, right - inset, bottom - inset], radius=24, fill=colors["cabinet"], outline=colors["trim"], width=3)
    title_h = 76
    draw.rounded_rectangle([left + 44, top + 24, right - 44, top + title_h], radius=18, fill=colors["trim"], outline=colors["cabinet_dark"], width=3)
    draw_centered_game_text(
        draw,
        text="SLOTS",
        center=((left + right) / 2.0, top + title_h / 2.0 + 12),
        font=_font(int(render_params.label_font_size_px) + 8, bold=True),
        fill=colors["cabinet_dark"],
        stroke_fill=colors["trim"],
        surface_rgbs=(colors["trim"],),
        role="readout",
        required=True,
    )

    grid_w = REEL_COUNT * render_params.reel_cell_width_px + (REEL_COUNT - 1) * render_params.reel_gap_px
    grid_h = ROW_COUNT * render_params.reel_cell_height_px + (ROW_COUNT - 1) * render_params.row_gap_px
    grid_left = left + (cabinet_w - grid_w) / 2.0
    grid_top = top + 116
    window_bbox = [grid_left - 18, grid_top - 18, grid_left + grid_w + 18, grid_top + grid_h + 18]
    draw.rounded_rectangle(window_bbox, radius=22, fill=(28, 30, 36), outline=colors["trim"], width=5)

    grid = cell_grid(scene)
    cell_bboxes: dict[str, list[float]] = {}
    cell_centers: dict[str, list[float]] = {}
    scene_entities: list[dict[str, Any]] = []
    for row in range(ROW_COUNT):
        for col in range(REEL_COUNT):
            x0 = grid_left + col * (render_params.reel_cell_width_px + render_params.reel_gap_px)
            y0 = grid_top + row * (render_params.reel_cell_height_px + render_params.row_gap_px)
            x1 = x0 + render_params.reel_cell_width_px
            y1 = y0 + render_params.reel_cell_height_px
            cell_id = slot_cell_id(row, col)
            bbox = [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]
            center = [round((x0 + x1) / 2.0, 3), round((y0 + y1) / 2.0, 3)]
            cell_bboxes[cell_id] = bbox
            cell_centers[cell_id] = center
            draw.rounded_rectangle(bbox, radius=12, fill=colors["reel"], outline=(76, 76, 82), width=2)
            _draw_symbol(draw, bbox, grid[row][col], colors=colors)
            scene_entities.append(
                {
                    "id": cell_id,
                    "kind": "slot_cell",
                    "row": int(row),
                    "col": int(col),
                    "symbol_key": str(grid[row][col]),
                    "bbox_px": list(bbox),
                    "center_px": list(center),
                }
            )

    payline_segments: dict[str, list[list[float]]] = {}
    for payline_key in PAYLINE_IDS:
        cells = PAYLINE_CELLS_BY_ID[str(payline_key)]
        first_row, first_col = cells[0]
        last_row, last_col = cells[-1]
        first_center = cell_centers[slot_cell_id(int(first_row), int(first_col))]
        last_center = cell_centers[slot_cell_id(int(last_row), int(last_col))]
        segment = [list(first_center), list(last_center)]
        entity_id = payline_entity_id(str(payline_key))
        payline_segments[entity_id] = segment
        scene_entities.append(
            {
                "id": entity_id,
                "kind": "conceptual_payline",
                "payline_key": str(payline_key),
                "cells": [[int(row), int(col)] for row, col in cells],
                "segment_px": segment,
                "is_winning": bool(str(payline_key) in scene.winning_payline_ids),
                "visible": False,
            }
        )

    lever_x = right - 8
    lever_y0 = top + 152
    lever_y1 = top + 312
    draw.line([(lever_x, lever_y0), (lever_x + 54, lever_y1)], fill=colors["cabinet_dark"], width=11)
    draw.ellipse([lever_x + 34, lever_y1 - 18, lever_x + 76, lever_y1 + 24], fill=colors["trim"], outline=colors["cabinet_dark"], width=3)
    paytable_bboxes, paytable_entities, paytable_panel_bbox = _draw_paytable(
        draw,
        scene=scene,
        cabinet_right=right,
        cabinet_top=top,
        colors=colors,
    )
    scene_entities.extend(paytable_entities)

    noisy_image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params={},
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedSlotMachineScene(
        image=noisy_image,
        render_map={
            "cell_bboxes_px": cell_bboxes,
            "cell_centers_px": cell_centers,
            "payline_segments_px": payline_segments,
            "paytable_entry_bboxes_px": paytable_bboxes,
            "paytable_panel_bbox_px": paytable_panel_bbox,
            "cabinet_bbox_px": [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)],
            "window_bbox_px": [round(float(v), 3) for v in window_bbox],
            "style_colors": {key: list(value) for key, value in colors.items()},
        },
        scene_entities=tuple(scene_entities),
        panel_style_meta=game_panel_scene_style_metadata(panel_style),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def render_slot_completion_scene(
    *,
    scene: SlotCompletionScene,
    render_params: SlotMachineRenderParams,
    instance_seed: int,
) -> RenderedSlotMachineScene:
    """Render two fixed reels with four labeled third-reel option panels."""

    validate_slot_completion_scene(scene)
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_style",
        treatments=("bare_canvas", "plain_sheet", "soft_panel", "game_table", "arcade_screen"),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    draw = ImageDraw.Draw(image)
    colors = _style_colors(str(scene.style_variant))
    source_cell_w = float(render_params.reel_cell_width_px)
    source_cell_h = float(render_params.reel_cell_height_px)
    source_gap_x = float(render_params.reel_gap_px)
    source_gap_y = float(render_params.row_gap_px)
    source_grid_w = 2.0 * source_cell_w + source_gap_x
    source_grid_h = ROW_COUNT * source_cell_h + (ROW_COUNT - 1) * source_gap_y
    source_pad_x = 32.0
    source_pad_y = 36.0
    source_w = source_grid_w + source_pad_x * 2.0
    source_h = source_grid_h + source_pad_y * 2.0 + 58.0
    source_left = 58.0
    source_top = (float(render_params.canvas_height) - source_h) / 2.0
    source_right = source_left + source_w
    source_bottom = source_top + source_h

    draw.rounded_rectangle(
        [source_left, source_top, source_right, source_bottom],
        radius=28,
        fill=colors["cabinet_dark"],
        outline=colors["trim"],
        width=5,
    )
    draw.rounded_rectangle(
        [source_left + 16, source_top + 16, source_right - 16, source_bottom - 16],
        radius=20,
        fill=colors["cabinet"],
        outline=colors["trim"],
        width=2,
    )
    title_bbox = [source_left + 42, source_top + 24, source_right - 42, source_top + 70]
    draw.rounded_rectangle(title_bbox, radius=15, fill=colors["trim"], outline=colors["cabinet_dark"], width=2)
    draw_centered_game_text(
        draw,
        text="SLOTS",
        center=((title_bbox[0] + title_bbox[2]) / 2.0, (title_bbox[1] + title_bbox[3]) / 2.0 + 2.0),
        font=_font(26, bold=True),
        fill=colors["cabinet_dark"],
        stroke_fill=colors["trim"],
        surface_rgbs=(colors["trim"],),
        role="readout",
        required=True,
    )
    source_grid_left = source_left + source_pad_x
    source_grid_top = source_top + 94.0
    window_bbox = [
        source_grid_left - 14.0,
        source_grid_top - 14.0,
        source_grid_left + source_grid_w + 14.0,
        source_grid_top + source_grid_h + 14.0,
    ]
    draw.rounded_rectangle(window_bbox, radius=20, fill=(28, 30, 36), outline=colors["trim"], width=4)

    base_cell_bboxes: dict[str, list[float]] = {}
    base_cell_centers: dict[str, list[float]] = {}
    scene_entities: list[dict[str, Any]] = []
    base_by_position = {(int(cell.row), int(cell.col)): str(cell.symbol_key) for cell in scene.base_cells}
    for row in range(ROW_COUNT):
        for col in range(REEL_COUNT - 1):
            x0 = source_grid_left + col * (source_cell_w + source_gap_x)
            y0 = source_grid_top + row * (source_cell_h + source_gap_y)
            bbox = [round(x0, 3), round(y0, 3), round(x0 + source_cell_w, 3), round(y0 + source_cell_h, 3)]
            center = [round((bbox[0] + bbox[2]) / 2.0, 3), round((bbox[1] + bbox[3]) / 2.0, 3)]
            cell_id = slot_cell_id(row, col)
            base_cell_bboxes[cell_id] = bbox
            base_cell_centers[cell_id] = center
            draw.rounded_rectangle(bbox, radius=12, fill=colors["reel"], outline=(76, 76, 82), width=2)
            _draw_symbol(draw, bbox, base_by_position[(row, col)], colors=colors)
            scene_entities.append(
                {
                    "id": cell_id,
                    "kind": "slot_completion_base_cell",
                    "row": int(row),
                    "col": int(col),
                    "symbol_key": base_by_position[(row, col)],
                    "bbox_px": list(bbox),
                    "center_px": list(center),
                }
            )

    option_cell_w = 76.0
    option_cell_h = 64.0
    option_gap_y = 8.0
    option_panel_w = 126.0
    option_panel_h = 270.0
    option_gap_x = 30.0
    option_gap_block_y = 26.0
    option_block_w = option_panel_w * 2.0 + option_gap_x
    option_block_h = option_panel_h * 2.0 + option_gap_block_y
    options_left = source_right + 64.0
    if options_left + option_block_w > float(render_params.canvas_width) - 24.0:
        options_left = float(render_params.canvas_width) - option_block_w - 32.0
    options_top = (float(render_params.canvas_height) - option_block_h) / 2.0

    option_bboxes: dict[str, list[float]] = {}
    option_cell_bboxes: dict[str, list[float]] = {}
    option_completed_paylines: dict[str, list[str]] = {}
    for index, option in enumerate(scene.options):
        row_index = int(index // 2)
        col_index = int(index % 2)
        panel_left = options_left + col_index * (option_panel_w + option_gap_x)
        panel_top = options_top + row_index * (option_panel_h + option_gap_block_y)
        panel_bbox = [panel_left, panel_top, panel_left + option_panel_w, panel_top + option_panel_h]
        rounded_panel_bbox = [round(float(value), 3) for value in panel_bbox]
        option_bboxes[str(option.label)] = rounded_panel_bbox
        draw.rounded_rectangle(
            panel_bbox,
            radius=20,
            fill=colors["cabinet_dark"],
            outline=colors["trim"],
            width=4,
        )
        label_center = (panel_left + option_panel_w / 2.0, panel_top + 28.0)
        draw.ellipse(
            [
                label_center[0] - 18.0,
                label_center[1] - 18.0,
                label_center[0] + 18.0,
                label_center[1] + 18.0,
            ],
            fill=(252, 246, 231),
            outline=colors["trim"],
            width=3,
        )
        draw_centered_game_text(
            draw,
            text=str(option.label),
            center=(label_center[0], label_center[1] + 1.0),
            font=_font(22, bold=True),
            fill=colors["cabinet_dark"],
            stroke_fill=(252, 246, 231),
            surface_rgbs=((252, 246, 231),),
            role="option_label",
            required=True,
        )
        cells_left = panel_left + (option_panel_w - option_cell_w) / 2.0
        cells_top = panel_top + 58.0
        option_symbols = {int(cell.row): str(cell.symbol_key) for cell in option.cells}
        option_completed_paylines[str(option.label)] = [str(payline_id) for payline_id in option.completed_payline_ids]
        for option_row in range(ROW_COUNT):
            y0 = cells_top + option_row * (option_cell_h + option_gap_y)
            cell_bbox = [cells_left, y0, cells_left + option_cell_w, y0 + option_cell_h]
            rounded_cell_bbox = [round(float(value), 3) for value in cell_bbox]
            key = f"option_{option.label}_cell_{option_row}_2"
            option_cell_bboxes[key] = rounded_cell_bbox
            draw.rounded_rectangle(rounded_cell_bbox, radius=10, fill=colors["reel"], outline=(76, 76, 82), width=2)
            _draw_symbol(draw, rounded_cell_bbox, option_symbols[option_row], colors=colors)
            scene_entities.append(
                {
                    "id": key,
                    "kind": "slot_completion_option_cell",
                    "option_label": str(option.label),
                    "row": int(option_row),
                    "col": REEL_COUNT - 1,
                    "symbol_key": option_symbols[option_row],
                    "bbox_px": list(rounded_cell_bbox),
                    "center_px": [
                        round((rounded_cell_bbox[0] + rounded_cell_bbox[2]) / 2.0, 3),
                        round((rounded_cell_bbox[1] + rounded_cell_bbox[3]) / 2.0, 3),
                    ],
                }
            )
        scene_entities.append(
            {
                "id": f"option_{option.label}",
                "kind": "slot_completion_option",
                "label": str(option.label),
                "bbox_px": list(rounded_panel_bbox),
                "completed_payline_ids": [str(payline_id) for payline_id in option.completed_payline_ids],
                "completed_grid": [list(row) for row in completion_option_grid(scene.base_cells, option.cells)],
            }
        )

    noisy_image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params={},
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedSlotMachineScene(
        image=noisy_image,
        render_map={
            "base_cell_bboxes_px": base_cell_bboxes,
            "base_cell_centers_px": base_cell_centers,
            "option_bboxes_px": option_bboxes,
            "option_cell_bboxes_px": option_cell_bboxes,
            "option_completed_payline_ids": option_completed_paylines,
            "answer_label": str(scene.answer_label),
            "answer_completed_payline_ids": [str(payline_id) for payline_id in scene.answer_completed_payline_ids],
            "source_cabinet_bbox_px": [round(source_left, 3), round(source_top, 3), round(source_right, 3), round(source_bottom, 3)],
            "source_window_bbox_px": [round(float(value), 3) for value in window_bbox],
            "style_colors": {key: list(value) for key, value in colors.items()},
        },
        scene_entities=tuple(scene_entities),
        panel_style_meta=game_panel_scene_style_metadata(panel_style),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "RenderedSlotMachineScene",
    "SlotMachineRenderParams",
    "render_slot_completion_scene",
    "render_slot_machine_scene",
    "resolve_slot_machine_render_params",
]
