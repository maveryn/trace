"""Rendering helpers for Star Battle puzzle scenes."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.common import get_int_param
from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.color_distance import coerce_rgb
from trace_tasks.tasks.shared.text_rendering import load_font

from .rules import candidate_key, cell_key, region_key
from .state import BBox, RenderedStarBattleScene, StarBattleDataset, StarBattleRenderParams


REGION_PALETTES = {
    "star_battle_classic": (
        (246, 217, 215),
        (219, 234, 249),
        (224, 241, 217),
        (250, 235, 190),
        (235, 222, 246),
        (221, 241, 239),
        (248, 224, 202),
        (229, 232, 240),
        (240, 224, 212),
    ),
    "star_battle_pastel": (
        (248, 226, 232),
        (223, 238, 255),
        (230, 246, 225),
        (255, 243, 206),
        (239, 229, 252),
        (220, 245, 241),
        (255, 231, 210),
        (231, 235, 246),
        (243, 231, 220),
    ),
    "star_battle_blueprint": (
        (219, 233, 246),
        (229, 241, 250),
        (220, 239, 238),
        (236, 241, 221),
        (229, 230, 246),
        (218, 229, 239),
        (242, 236, 218),
        (226, 236, 244),
        (236, 225, 235),
    ),
}


def round_bbox(bbox: Sequence[float]) -> BBox:
    """Round one pixel bbox for trace metadata."""

    return [round(float(value), 3) for value in bbox[:4]]


def bbox_union(boxes: Sequence[Sequence[float]]) -> BBox:
    """Return the enclosing bbox for one non-empty bbox sequence."""

    if not boxes:
        raise ValueError("bbox_union requires at least one bbox")
    return round_bbox(
        (
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        )
    )


def resolve_render_params(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> StarBattleRenderParams:
    """Resolve canvas dimensions and size jitter for one Star Battle instance."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.star_battle.unit_size",
    )
    return StarBattleRenderParams(
        canvas_width=get_int_param(params, rendering_defaults, key="canvas_width", fallback=1080),
        canvas_height=get_int_param(params, rendering_defaults, key="canvas_height", fallback=900),
        cell_size_px=scale_puzzle_px(rendering_defaults.get("cell_size_px", 64), unit_scale, min_px=24),
        panel_padding_px=scale_puzzle_px(rendering_defaults.get("panel_padding_px", 28), unit_scale, min_px=12),
        panel_corner_radius_px=scale_puzzle_px(
            rendering_defaults.get("panel_corner_radius_px", 18),
            unit_scale,
            min_px=7,
        ),
        grid_line_width_px=scale_puzzle_px(rendering_defaults.get("grid_line_width_px", 2), unit_scale, min_px=1),
        heavy_line_width_px=scale_puzzle_px(rendering_defaults.get("heavy_line_width_px", 4), unit_scale, min_px=2),
        clue_size_px=scale_puzzle_px(rendering_defaults.get("clue_size_px", 54), unit_scale, min_px=26),
        candidate_font_size_px=scale_puzzle_px(
            rendering_defaults.get("candidate_font_size_px", 28),
            unit_scale,
            min_px=14,
        ),
        clue_font_size_px=scale_puzzle_px(rendering_defaults.get("clue_font_size_px", 24), unit_scale, min_px=12),
        text_color_rgb=coerce_rgb(rendering_defaults.get("text_color_rgb"), (28, 32, 38)),
        text_stroke_rgb=coerce_rgb(rendering_defaults.get("text_stroke_rgb"), (255, 255, 255)),
        style_overrides={},
        unit_size_jitter=dict(unit_meta),
    )


def apply_scene_style(
    *,
    render_params: StarBattleRenderParams,
    instance_seed: int,
    namespace: str,
) -> tuple[StarBattleRenderParams, Image.Image, Dict[str, Any], Dict[str, Any]]:
    """Apply shared puzzle treatment colors and create the background image."""

    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    resolved_params = replace(
        render_params,
        text_color_rgb=tuple(int(value) for value in scene_style.text_rgb),
        text_stroke_rgb=tuple(int(value) for value in scene_style.text_stroke_rgb),
        style_overrides={
            "region_palette": tuple(tuple(int(channel) for channel in color) for color in scene_style.state_colors),
            "border": tuple(int(value) for value in scene_style.panel_border_rgb),
            "grid_line": tuple(int(value) for value in scene_style.grid_rgb),
            "clue_fill": tuple(int(value) for value in scene_style.option_fill_rgb),
            "accent": tuple(int(value) for value in scene_style.mark_rgb),
            "accent_backdrop": tuple(int(value) for value in scene_style.text_stroke_rgb),
            "highlight_fill": tuple(int(value) for value in scene_style.step_fill_rgb),
            "candidate_fill": tuple(int(value) for value in scene_style.option_fill_rgb),
            "panel_fill": tuple(int(value) for value in scene_style.panel_fill_rgb),
        },
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(resolved_params.canvas_width),
        canvas_height=int(resolved_params.canvas_height),
        style=scene_style,
    )
    return resolved_params, background, dict(background_meta), dict(scene_style_meta)


def draw_star(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> None:
    """Draw one five-point Star Battle star inside a cell bbox."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    r_outer = min(x1 - x0, y1 - y0) * 0.34
    r_inner = r_outer * 0.43
    points = []
    for idx in range(10):
        radius = r_outer if idx % 2 == 0 else r_inner
        angle = -1.5708 + idx * 0.628318
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=outline)


def render_star_battle_scene(
    image: Image.Image,
    *,
    dataset: StarBattleDataset,
    scene_variant: str,
    render_params: StarBattleRenderParams,
) -> RenderedStarBattleScene:
    """Render one complete Star Battle board with clues, regions, stars, and candidates."""

    draw = ImageDraw.Draw(image)
    size = int(dataset.size)
    style_overrides = dict(render_params.style_overrides or {})
    palette = [
        tuple(int(channel) for channel in color)
        for color in style_overrides.get("region_palette", REGION_PALETTES[str(scene_variant)])
    ]
    cell_size = int(render_params.cell_size_px)
    clue = int(render_params.clue_size_px)
    grid_w = int(size * cell_size)
    total_w = int(clue + grid_w)
    total_h = int(clue + grid_w)
    panel_x0 = int((int(render_params.canvas_width) - total_w) // 2) - int(render_params.panel_padding_px)
    panel_y0 = int((int(render_params.canvas_height) - total_h) // 2) - int(render_params.panel_padding_px)
    grid_x0 = int(panel_x0 + int(render_params.panel_padding_px) + clue)
    grid_y0 = int(panel_y0 + int(render_params.panel_padding_px) + clue)
    panel_x1 = int(panel_x0 + total_w + (2 * int(render_params.panel_padding_px)))
    panel_y1 = int(panel_y0 + total_h + (2 * int(render_params.panel_padding_px)))
    border = tuple(style_overrides.get("border", (74, 82, 96)))
    grid_line = tuple(style_overrides.get("grid_line", (126, 132, 144)))
    clue_fill = tuple(style_overrides.get("clue_fill", (245, 247, 250)))
    accent = tuple(style_overrides.get("accent", (45, 91, 176)))
    accent_backdrop = tuple(style_overrides.get("accent_backdrop", (255, 255, 255)))
    highlight_fill = tuple(style_overrides.get("highlight_fill", (255, 241, 142)))
    candidate_fill = tuple(style_overrides.get("candidate_fill", (255, 250, 218)))
    panel_fill = tuple(style_overrides.get("panel_fill", (250, 250, 247)))

    draw_rounded_rect(
        draw,
        (panel_x0, panel_y0, panel_x1, panel_y1),
        radius=int(render_params.panel_corner_radius_px),
        fill=panel_fill,
        outline=border,
        width=max(1, int(render_params.grid_line_width_px)),
    )
    clue_font = load_font(int(render_params.clue_font_size_px), bold=True)
    candidate_font = load_font(int(render_params.candidate_font_size_px), bold=True)
    cell_bbox_map: Dict[str, BBox] = {}
    row_bbox_map: Dict[str, BBox] = {}
    col_bbox_map: Dict[str, BBox] = {}
    item_bbox_map: Dict[str, BBox] = {}
    entities: list[dict[str, Any]] = [
        {
            "entity_id": "star_battle_panel",
            "entity_type": "puzzle_star_battle_panel",
            "bbox_px": round_bbox((panel_x0, panel_y0, panel_x1, panel_y1)),
            "scene_variant": str(scene_variant),
        }
    ]

    for row in range(size):
        bbox = (grid_x0 - clue, grid_y0 + row * cell_size, grid_x0, grid_y0 + (row + 1) * cell_size)
        row_bbox_map[f"row_{row}"] = round_bbox(bbox)
        item_bbox_map[f"row_{row}"] = round_bbox(bbox)
        is_marked = bool(dataset.marked_row_index is not None and int(dataset.marked_row_index) == int(row))
        draw.rectangle(
            bbox,
            fill=highlight_fill if is_marked else clue_fill,
            outline=accent if is_marked else grid_line,
            width=max(3 if is_marked else 1, int(render_params.grid_line_width_px)),
        )
        draw_centered_text(
            draw,
            text="1",
            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            font=clue_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        entities.append(
            {
                "entity_id": f"row_{row}",
                "entity_type": "puzzle_star_battle_row_clue",
                "bbox_px": round_bbox(bbox),
                "value": 1,
                "row": int(row),
            }
        )
    for col in range(size):
        bbox = (grid_x0 + col * cell_size, grid_y0 - clue, grid_x0 + (col + 1) * cell_size, grid_y0)
        col_bbox_map[f"col_{col}"] = round_bbox(bbox)
        item_bbox_map[f"col_{col}"] = round_bbox(bbox)
        is_marked = bool(dataset.marked_col_index is not None and int(dataset.marked_col_index) == int(col))
        draw.rectangle(
            bbox,
            fill=highlight_fill if is_marked else clue_fill,
            outline=accent if is_marked else grid_line,
            width=max(3 if is_marked else 1, int(render_params.grid_line_width_px)),
        )
        draw_centered_text(
            draw,
            text="1",
            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            font=clue_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        entities.append(
            {
                "entity_id": f"col_{col}",
                "entity_type": "puzzle_star_battle_col_clue",
                "bbox_px": round_bbox(bbox),
                "value": 1,
                "col": int(col),
            }
        )

    for row in range(size):
        for col in range(size):
            region_index = int(dataset.region_grid[row][col])
            bbox = (
                grid_x0 + col * cell_size,
                grid_y0 + row * cell_size,
                grid_x0 + (col + 1) * cell_size,
                grid_y0 + (row + 1) * cell_size,
            )
            fill = palette[region_index % len(palette)]
            draw.rectangle(
                bbox,
                fill=fill,
                outline=grid_line,
                width=max(1, int(render_params.grid_line_width_px)),
            )
            key = cell_key((row, col))
            cell_bbox_map[key] = round_bbox(bbox)
            item_bbox_map[key] = round_bbox(bbox)
            entities.append(
                {
                    "entity_id": key,
                    "entity_type": "puzzle_star_battle_cell",
                    "bbox_px": round_bbox(bbox),
                    "row": int(row),
                    "col": int(col),
                    "region_index": int(region_index),
                }
            )

    _draw_region_borders(
        draw=draw,
        dataset=dataset,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        cell_size=cell_size,
        border=border,
        width=max(2, int(render_params.heavy_line_width_px)),
    )
    draw.rectangle(
        (grid_x0, grid_y0, grid_x0 + grid_w, grid_y0 + grid_w),
        outline=border,
        width=max(3, int(render_params.heavy_line_width_px)),
    )
    _draw_marked_scope_outlines(
        draw=draw,
        dataset=dataset,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        grid_w=grid_w,
        cell_size=cell_size,
        accent=accent,
        accent_backdrop=accent_backdrop,
        heavy_width=int(render_params.heavy_line_width_px),
    )

    region_bbox_map: Dict[str, BBox] = {}
    for region_index, cells in dict(dataset.regions).items():
        boxes = [cell_bbox_map[cell_key(tuple(cell))] for cell in cells]
        region_bbox_map[region_key(int(region_index))] = bbox_union(boxes)
        item_bbox_map[region_key(int(region_index))] = list(region_bbox_map[region_key(int(region_index))])
        entities.append(
            {
                "entity_id": region_key(int(region_index)),
                "entity_type": "puzzle_star_battle_region",
                "bbox_px": list(region_bbox_map[region_key(int(region_index))]),
                "region_index": int(region_index),
                "cell_count": len(cells),
            }
        )
    _draw_marked_region_outline(
        draw=draw,
        dataset=dataset,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        accent=accent,
        accent_backdrop=accent_backdrop,
        cell_size=cell_size,
        heavy_width=int(render_params.heavy_line_width_px),
    )
    _draw_visible_stars(
        draw=draw,
        dataset=dataset,
        cell_bbox_map=cell_bbox_map,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )
    _draw_candidate_labels(
        draw=draw,
        dataset=dataset,
        cell_bbox_map=cell_bbox_map,
        item_bbox_map=item_bbox_map,
        entities=entities,
        cell_size=cell_size,
        candidate_fill=candidate_fill,
        accent=accent,
        candidate_font=candidate_font,
    )

    return RenderedStarBattleScene(
        image=image,
        entities=entities,
        scene_bbox_px=round_bbox((panel_x0, panel_y0, panel_x1, panel_y1)),
        cell_bbox_map=cell_bbox_map,
        row_bbox_map=row_bbox_map,
        col_bbox_map=col_bbox_map,
        region_bbox_map=region_bbox_map,
        item_bbox_map=item_bbox_map,
    )


def _draw_region_borders(
    *,
    draw: ImageDraw.ImageDraw,
    dataset: StarBattleDataset,
    grid_x0: int,
    grid_y0: int,
    cell_size: int,
    border: Tuple[int, int, int],
    width: int,
) -> None:
    """Draw heavy boundaries where adjacent cells belong to different regions."""

    size = int(dataset.size)
    for row in range(size):
        for col in range(size):
            region_index = int(dataset.region_grid[row][col])
            x0 = grid_x0 + col * cell_size
            y0 = grid_y0 + row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            if row == 0 or int(dataset.region_grid[row - 1][col]) != region_index:
                draw.line((x0, y0, x1, y0), fill=border, width=width)
            if row == size - 1 or int(dataset.region_grid[row + 1][col]) != region_index:
                draw.line((x0, y1, x1, y1), fill=border, width=width)
            if col == 0 or int(dataset.region_grid[row][col - 1]) != region_index:
                draw.line((x0, y0, x0, y1), fill=border, width=width)
            if col == size - 1 or int(dataset.region_grid[row][col + 1]) != region_index:
                draw.line((x1, y0, x1, y1), fill=border, width=width)


def _draw_marked_scope_outlines(
    *,
    draw: ImageDraw.ImageDraw,
    dataset: StarBattleDataset,
    grid_x0: int,
    grid_y0: int,
    grid_w: int,
    cell_size: int,
    accent: Tuple[int, int, int],
    accent_backdrop: Tuple[int, int, int],
    heavy_width: int,
) -> None:
    """Draw row or column scope outlines when a task marks one."""

    backdrop_width = max(10, int(heavy_width) + 6)
    accent_width = max(7, int(heavy_width) + 3)
    if dataset.marked_row_index is not None:
        bbox = (
            grid_x0,
            grid_y0 + int(dataset.marked_row_index) * cell_size,
            grid_x0 + grid_w,
            grid_y0 + (int(dataset.marked_row_index) + 1) * cell_size,
        )
        draw.rectangle(bbox, outline=accent_backdrop, width=backdrop_width)
        draw.rectangle(bbox, outline=accent, width=accent_width)
    if dataset.marked_col_index is not None:
        bbox = (
            grid_x0 + int(dataset.marked_col_index) * cell_size,
            grid_y0,
            grid_x0 + (int(dataset.marked_col_index) + 1) * cell_size,
            grid_y0 + grid_w,
        )
        draw.rectangle(bbox, outline=accent_backdrop, width=backdrop_width)
        draw.rectangle(bbox, outline=accent, width=accent_width)


def _draw_marked_region_outline(
    *,
    draw: ImageDraw.ImageDraw,
    dataset: StarBattleDataset,
    grid_x0: int,
    grid_y0: int,
    accent: Tuple[int, int, int],
    accent_backdrop: Tuple[int, int, int],
    cell_size: int,
    heavy_width: int,
) -> None:
    """Draw a boundary outline for a marked colored region without changing cell fills."""

    if dataset.marked_region_index is None:
        return

    selected_region = int(dataset.marked_region_index)
    size = int(dataset.size)

    def draw_external_edges(*, fill: Tuple[int, int, int], width: int) -> None:
        for row in range(size):
            for col in range(size):
                if int(dataset.region_grid[row][col]) != selected_region:
                    continue
                x0 = grid_x0 + col * cell_size
                y0 = grid_y0 + row * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                if row == 0 or int(dataset.region_grid[row - 1][col]) != selected_region:
                    draw.line((x0, y0, x1, y0), fill=fill, width=width)
                if row == size - 1 or int(dataset.region_grid[row + 1][col]) != selected_region:
                    draw.line((x0, y1, x1, y1), fill=fill, width=width)
                if col == 0 or int(dataset.region_grid[row][col - 1]) != selected_region:
                    draw.line((x0, y0, x0, y1), fill=fill, width=width)
                if col == size - 1 or int(dataset.region_grid[row][col + 1]) != selected_region:
                    draw.line((x1, y0, x1, y1), fill=fill, width=width)

    draw_external_edges(fill=accent_backdrop, width=max(4, int(heavy_width) + 1))
    draw_external_edges(fill=accent, width=max(2, int(heavy_width) - 1))


def _draw_visible_stars(
    *,
    draw: ImageDraw.ImageDraw,
    dataset: StarBattleDataset,
    cell_bbox_map: Mapping[str, Sequence[float]],
    item_bbox_map: Dict[str, BBox],
    entities: list[dict[str, Any]],
) -> None:
    """Draw all currently visible fixed stars and record their bboxes."""

    for index, star in enumerate(dataset.visible_stars):
        bbox = cell_bbox_map[cell_key(tuple(star))]
        draw_star(draw, bbox, fill=(35, 39, 46), outline=(255, 255, 255))
        entity_id = f"star_{index}"
        item_bbox_map[entity_id] = list(round_bbox(bbox))
        entities.append(
            {
                "entity_id": entity_id,
                "entity_type": "puzzle_star_battle_star",
                "bbox_px": list(round_bbox(bbox)),
                "row": int(star[0]),
                "col": int(star[1]),
            }
        )


def _draw_candidate_labels(
    *,
    draw: ImageDraw.ImageDraw,
    dataset: StarBattleDataset,
    cell_bbox_map: Mapping[str, Sequence[float]],
    item_bbox_map: Dict[str, BBox],
    entities: list[dict[str, Any]],
    cell_size: int,
    candidate_fill: Tuple[int, int, int],
    accent: Tuple[int, int, int],
    candidate_font: Any,
) -> None:
    """Draw labeled candidate cells for option-label tasks."""

    for spec in dataset.candidate_specs:
        bbox = cell_bbox_map[cell_key(spec.cell)]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        label_bbox = (
            x0 + cell_size * 0.22,
            y0 + cell_size * 0.22,
            x1 - cell_size * 0.22,
            y1 - cell_size * 0.22,
        )
        draw.rounded_rectangle(
            label_bbox,
            radius=max(5, int(cell_size * 0.10)),
            fill=candidate_fill,
            outline=accent,
            width=max(2, int(cell_size * 0.04)),
        )
        draw_centered_text(
            draw,
            text=str(spec.label),
            center=((x0 + x1) / 2, (y0 + y1) / 2),
            font=candidate_font,
            fill=(26, 28, 32),
            stroke_fill=(255, 255, 255),
            stroke_width=1,
        )
        item_bbox_map[candidate_key(spec.label)] = list(round_bbox(bbox))
        entities.append(
            {
                "entity_id": candidate_key(spec.label),
                "entity_type": "puzzle_star_battle_candidate_cell",
                "bbox_px": list(round_bbox(bbox)),
                "label": str(spec.label),
                "row": int(spec.row),
                "col": int(spec.col),
                "is_correct": bool(spec.is_correct),
                "is_legal": bool(spec.is_legal),
            }
        )


__all__ = [
    "apply_scene_style",
    "bbox_union",
    "render_star_battle_scene",
    "resolve_render_params",
    "round_bbox",
]
