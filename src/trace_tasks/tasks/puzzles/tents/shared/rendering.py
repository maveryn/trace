"""Rendering helpers for Tents grids, clues, trees, tents, and candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.bbox_projection import round_bbox
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect

from .defaults import PALETTE_COLORS, STYLE_COLORS
from .state import (
    CandidateCellSpec,
    Cell,
    LabeledTentSpec,
    RenderedTentsScene,
    TentsRenderParams,
)


def render_tents_scene(
    image: Image.Image,
    *,
    scene_variant: str,
    palette_variant: str,
    rows: int,
    cols: int,
    row_clues: Sequence[int],
    col_clues: Sequence[int],
    marked_tree: Cell | None,
    tree_cells: Sequence[Cell],
    visible_tents: Sequence[Cell],
    candidate_specs: Sequence[CandidateCellSpec],
    labeled_tent_specs: Sequence[LabeledTentSpec] | None = None,
    render_params: TentsRenderParams,
) -> RenderedTentsScene:
    """Render the complete Tents scene and return item-to-pixel maps."""

    draw = ImageDraw.Draw(image)
    style = dict(STYLE_COLORS[str(scene_variant)])
    style.update(PALETTE_COLORS[str(palette_variant)])
    style.update(
        {
            str(key): tuple(int(component) for component in value[:3])
            for key, value in dict(render_params.style_overrides or {}).items()
        }
    )
    cell_size = int(render_params.cell_size_px)
    grid_w = int(cols) * int(cell_size)
    grid_h = int(rows) * int(cell_size)
    total_w = int(render_params.left_clue_width_px) + int(grid_w)
    total_h = int(render_params.top_clue_height_px) + int(grid_h)
    labeled_tents = list(labeled_tent_specs or [])
    panel_x0 = int((int(render_params.canvas_width) - total_w) // 2) - int(
        render_params.panel_padding_px
    )
    panel_y0 = int((int(render_params.canvas_height) - total_h) // 2) - int(
        render_params.panel_padding_px
    )
    panel_x1 = int(panel_x0 + total_w + (2 * int(render_params.panel_padding_px)))
    panel_y1 = int(panel_y0 + total_h + (2 * int(render_params.panel_padding_px)))
    grid_x0 = int(
        panel_x0
        + int(render_params.panel_padding_px)
        + int(render_params.left_clue_width_px)
    )
    grid_y0 = int(
        panel_y0
        + int(render_params.panel_padding_px)
        + int(render_params.top_clue_height_px)
    )

    draw_rounded_rect(
        draw,
        (panel_x0, panel_y0, panel_x1, panel_y1),
        radius=int(render_params.panel_corner_radius_px),
        fill=style["panel_fill"],
        outline=style["heavy_line"],
        width=max(1, int(render_params.grid_line_width_px)),
    )

    clue_font = load_font(int(render_params.clue_font_size_px), bold=True)
    candidate_font = load_font(int(render_params.candidate_font_size_px), bold=True)
    cell_bbox_map: Dict[str, List[float]] = {}
    clue_bbox_map: Dict[str, List[float]] = {}
    option_panel_bbox_map: Dict[str, List[float]] = {}
    item_bbox_map: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "tents_panel",
            "entity_type": "puzzle_tents_panel",
            "bbox_px": round_bbox((panel_x0, panel_y0, panel_x1, panel_y1)),
            "scene_variant": str(scene_variant),
            "palette_variant": str(palette_variant),
        }
    ]

    _draw_clue_corner(
        draw,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        render_params=render_params,
        style=style,
    )
    _draw_row_clues(
        draw,
        row_clues=row_clues,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        cell_size=cell_size,
        clue_font=clue_font,
        render_params=render_params,
        style=style,
        clue_bbox_map=clue_bbox_map,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )
    _draw_col_clues(
        draw,
        col_clues=col_clues,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        cell_size=cell_size,
        clue_font=clue_font,
        render_params=render_params,
        style=style,
        clue_bbox_map=clue_bbox_map,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )
    _draw_grid_cells(
        draw,
        rows=int(rows),
        cols=int(cols),
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        cell_size=cell_size,
        candidate_specs=candidate_specs,
        render_params=render_params,
        style=style,
        cell_bbox_map=cell_bbox_map,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )

    draw.rectangle(
        (grid_x0, grid_y0, grid_x0 + grid_w, grid_y0 + grid_h),
        outline=style["heavy_line"],
        width=int(render_params.heavy_line_width_px),
    )
    _draw_tree_items(
        draw,
        tree_cells=tree_cells,
        marked_tree=marked_tree,
        cell_bbox_map=cell_bbox_map,
        style=style,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )
    _draw_tent_items(
        draw,
        visible_tents=visible_tents,
        cell_bbox_map=cell_bbox_map,
        style=style,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )
    _draw_labeled_tent_labels(
        draw,
        labeled_tent_specs=labeled_tents,
        cell_size=cell_size,
        label_font=candidate_font,
        cell_bbox_map=cell_bbox_map,
        style=style,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )
    _draw_candidate_labels(
        draw,
        candidate_specs=candidate_specs,
        cell_size=cell_size,
        candidate_font=candidate_font,
        cell_bbox_map=cell_bbox_map,
        style=style,
        item_bbox_map=item_bbox_map,
        entities=entities,
    )

    return RenderedTentsScene(
        image=image,
        entities=entities,
        scene_bbox_px=round_bbox((panel_x0, panel_y0, panel_x1, panel_y1)),
        cell_bbox_map=cell_bbox_map,
        clue_bbox_map=clue_bbox_map,
        option_panel_bbox_map=option_panel_bbox_map,
        item_bbox_map=item_bbox_map,
    )


def _draw_clue_corner(
    draw: ImageDraw.ImageDraw,
    *,
    grid_x0: int,
    grid_y0: int,
    render_params: TentsRenderParams,
    style: Dict[str, Tuple[int, int, int]],
) -> None:
    bbox = (
        grid_x0 - int(render_params.left_clue_width_px),
        grid_y0 - int(render_params.top_clue_height_px),
        grid_x0,
        grid_y0,
    )
    draw.rectangle(
        bbox,
        fill=style["clue_fill"],
        outline=style["grid_line"],
        width=int(render_params.grid_line_width_px),
    )


def _draw_row_clues(
    draw: ImageDraw.ImageDraw,
    *,
    row_clues: Sequence[int],
    grid_x0: int,
    grid_y0: int,
    cell_size: int,
    clue_font: Any,
    render_params: TentsRenderParams,
    style: Dict[str, Tuple[int, int, int]],
    clue_bbox_map: Dict[str, List[float]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
) -> None:
    """Draw row clue rail cells and record their bboxes."""

    for row in range(len(row_clues)):
        y0 = int(grid_y0 + (row * int(cell_size)))
        y1 = int(y0 + int(cell_size))
        bbox = (grid_x0 - int(render_params.left_clue_width_px), y0, grid_x0, y1)
        clue_bbox_map[f"row_clue_{row}"] = round_bbox(bbox)
        item_bbox_map[f"row_clue_{row}"] = round_bbox(bbox)
        draw.rectangle(
            bbox,
            fill=style["clue_fill"],
            outline=style["grid_line"],
            width=int(render_params.grid_line_width_px),
        )
        draw_centered_text(
            draw,
            text=str(int(row_clues[row])),
            center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            font=clue_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        entities.append(
            {
                "entity_id": f"row_clue_{row}",
                "entity_type": "puzzle_tents_row_clue",
                "bbox_px": round_bbox(bbox),
                "value": int(row_clues[row]),
            }
        )


def _draw_col_clues(
    draw: ImageDraw.ImageDraw,
    *,
    col_clues: Sequence[int],
    grid_x0: int,
    grid_y0: int,
    cell_size: int,
    clue_font: Any,
    render_params: TentsRenderParams,
    style: Dict[str, Tuple[int, int, int]],
    clue_bbox_map: Dict[str, List[float]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
) -> None:
    """Draw column clue rail cells and record their bboxes."""

    for col in range(len(col_clues)):
        x0 = int(grid_x0 + (col * int(cell_size)))
        x1 = int(x0 + int(cell_size))
        bbox = (x0, grid_y0 - int(render_params.top_clue_height_px), x1, grid_y0)
        clue_bbox_map[f"col_clue_{col}"] = round_bbox(bbox)
        item_bbox_map[f"col_clue_{col}"] = round_bbox(bbox)
        draw.rectangle(
            bbox,
            fill=style["clue_fill"],
            outline=style["grid_line"],
            width=int(render_params.grid_line_width_px),
        )
        draw_centered_text(
            draw,
            text=str(int(col_clues[col])),
            center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            font=clue_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
        entities.append(
            {
                "entity_id": f"col_clue_{col}",
                "entity_type": "puzzle_tents_col_clue",
                "bbox_px": round_bbox(bbox),
                "value": int(col_clues[col]),
            }
        )


def _draw_grid_cells(
    draw: ImageDraw.ImageDraw,
    *,
    rows: int,
    cols: int,
    grid_x0: int,
    grid_y0: int,
    cell_size: int,
    candidate_specs: Sequence[CandidateCellSpec],
    render_params: TentsRenderParams,
    style: Dict[str, Tuple[int, int, int]],
    cell_bbox_map: Dict[str, List[float]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
) -> None:
    """Draw board cells before icon and label overlays."""

    candidate_by_cell = {tuple(spec.cell): spec for spec in candidate_specs}
    for row in range(int(rows)):
        for col in range(int(cols)):
            x0 = int(grid_x0 + (col * int(cell_size)))
            y0 = int(grid_y0 + (row * int(cell_size)))
            x1 = int(x0 + int(cell_size))
            y1 = int(y0 + int(cell_size))
            bbox = (x0, y0, x1, y1)
            fill = style["cell_a"] if (row + col) % 2 == 0 else style["cell_b"]
            if (row, col) in candidate_by_cell:
                fill = style["candidate_fill"]
            draw.rectangle(
                bbox,
                fill=fill,
                outline=style["grid_line"],
                width=int(render_params.grid_line_width_px),
            )
            cell_key = f"cell_{row}_{col}"
            cell_bbox_map[cell_key] = round_bbox(bbox)
            item_bbox_map[cell_key] = round_bbox(bbox)
            entities.append(
                {
                    "entity_id": cell_key,
                    "entity_type": "puzzle_tents_cell",
                    "bbox_px": round_bbox(bbox),
                    "row": int(row),
                    "col": int(col),
                }
            )


def _draw_tree_items(
    draw: ImageDraw.ImageDraw,
    *,
    tree_cells: Sequence[Cell],
    marked_tree: Cell | None,
    cell_bbox_map: Dict[str, List[float]],
    style: Dict[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
) -> None:
    """Draw tree icons, using a square outline only for the marked tree."""

    tree_index = 0
    for tree_cell in tree_cells:
        row, col = int(tree_cell[0]), int(tree_cell[1])
        bbox = cell_bbox_map[f"cell_{row}_{col}"]
        is_marked = marked_tree is not None and tuple(tree_cell) == tuple(marked_tree)
        entity_id = "marked_tree" if bool(is_marked) else f"tree_{tree_index}"
        _draw_tree_icon(
            draw,
            bbox=bbox,
            fill=style["tree_fill"],
            outline=style["tree_outline"],
            trunk_fill=style.get("trunk_fill", (126, 88, 54)),
            trunk_outline=style.get("trunk_outline", (83, 58, 38)),
            ring_fill=style.get("marked_tree_outline", (214, 48, 49)),
            marked=bool(is_marked),
        )
        item_bbox_map[entity_id] = list(bbox)
        entities.append(
            {
                "entity_id": entity_id,
                "entity_type": "puzzle_tents_tree",
                "bbox_px": list(bbox),
                "row": int(row),
                "col": int(col),
                "marked": bool(is_marked),
            }
        )
        tree_index += 1


def _draw_tent_items(
    draw: ImageDraw.ImageDraw,
    *,
    visible_tents: Sequence[Cell],
    cell_bbox_map: Dict[str, List[float]],
    style: Dict[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
) -> None:
    """Draw visible tent icons already fixed on the board."""

    for tent_index, tent_cell in enumerate(visible_tents):
        row, col = int(tent_cell[0]), int(tent_cell[1])
        bbox = cell_bbox_map[f"cell_{row}_{col}"]
        entity_id = f"tent_{tent_index}"
        _draw_tent_icon(
            draw,
            bbox=bbox,
            fill=style["tent_fill"],
            shadow=style["tent_shadow"],
            flap_fill=style["tent_flap_fill"],
        )
        item_bbox_map[entity_id] = list(bbox)
        entities.append(
            {
                "entity_id": entity_id,
                "entity_type": "puzzle_tents_tent",
                "bbox_px": list(bbox),
                "row": int(row),
                "col": int(col),
            }
        )


def _draw_labeled_tent_labels(
    draw: ImageDraw.ImageDraw,
    *,
    labeled_tent_specs: Sequence[LabeledTentSpec],
    cell_size: int,
    label_font: Any,
    cell_bbox_map: Dict[str, List[float]],
    style: Dict[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
) -> None:
    """Draw option letters on visible tent cells without visible answer boxes."""

    for spec in labeled_tent_specs:
        row, col = int(spec.row), int(spec.col)
        bbox = cell_bbox_map[f"cell_{row}_{col}"]
        x0, y0, x1, y1 = [float(value) for value in bbox]
        draw_centered_text(
            draw,
            text=str(spec.label),
            center=((x0 + x1) / 2.0, y0 + (float(cell_size) * 0.52)),
            font=label_font,
            fill=style.get("candidate_label_fill", (255, 255, 255)),
            stroke_fill=style.get("tent_shadow", (40, 40, 40)),
            stroke_width=max(2, int(float(cell_size) * 0.05)),
        )
        tent_id = f"labeled_tent_{spec.label}"
        item_bbox_map[tent_id] = list(bbox)
        entities.append(
            {
                "entity_id": tent_id,
                "entity_type": "puzzle_tents_labeled_tent",
                "bbox_px": list(bbox),
                "label": str(spec.label),
                "row": int(row),
                "col": int(col),
                "is_correct": bool(spec.is_correct),
                "violation_type": str(spec.violation_type),
            }
        )


def _draw_candidate_labels(
    draw: ImageDraw.ImageDraw,
    *,
    candidate_specs: Sequence[CandidateCellSpec],
    cell_size: int,
    candidate_font: Any,
    cell_bbox_map: Dict[str, List[float]],
    style: Dict[str, Tuple[int, int, int]],
    item_bbox_map: Dict[str, List[float]],
    entities: List[Dict[str, Any]],
) -> None:
    """Draw candidate letter badges over labeled cells."""

    for spec in candidate_specs:
        row, col = int(spec.row), int(spec.col)
        bbox = cell_bbox_map[f"cell_{row}_{col}"]
        x0, y0, x1, y1 = [float(value) for value in bbox]
        draw_centered_text(
            draw,
            text=str(spec.label),
            center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            font=candidate_font,
            fill=(26, 28, 32),
            stroke_fill=(255, 255, 255),
            stroke_width=1,
        )
        candidate_id = f"candidate_{spec.label}"
        item_bbox_map[candidate_id] = list(bbox)
        entities.append(
            {
                "entity_id": candidate_id,
                "entity_type": "puzzle_tents_candidate_cell",
                "bbox_px": list(bbox),
                "label": str(spec.label),
                "row": int(row),
                "col": int(col),
                "is_correct": bool(spec.is_correct),
                "is_legal": bool(spec.is_legal),
            }
        )


def _draw_tree_icon(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    trunk_fill: Tuple[int, int, int],
    trunk_outline: Tuple[int, int, int],
    ring_fill: Tuple[int, int, int],
    marked: bool,
) -> None:
    """Draw a stylized tree inside one grid cell; marked trees get a square outline."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    w = float(x1 - x0)
    h = float(y1 - y0)
    trunk = (
        x0 + (0.43 * w),
        y0 + (0.55 * h),
        x0 + (0.57 * w),
        y0 + (0.82 * h),
    )
    canopy = (
        x0 + (0.20 * w),
        y0 + (0.16 * h),
        x0 + (0.80 * w),
        y0 + (0.70 * h),
    )
    draw.rounded_rectangle(
        trunk,
        radius=max(2, int(w * 0.04)),
        fill=tuple(trunk_fill),
        outline=tuple(trunk_outline),
        width=1,
    )
    draw.ellipse(canopy, fill=tuple(fill), outline=tuple(outline), width=2)
    if bool(marked):
        mark = (
            x0 + (0.08 * w),
            y0 + (0.07 * h),
            x0 + (0.92 * w),
            y0 + (0.91 * h),
        )
        draw.rectangle(mark, outline=tuple(ring_fill), width=max(3, int(w * 0.07)))


def _draw_tent_icon(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    fill: Tuple[int, int, int],
    shadow: Tuple[int, int, int],
    flap_fill: Tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    w = float(x1 - x0)
    h = float(y1 - y0)
    base_y = y0 + (0.78 * h)
    left = (x0 + (0.17 * w), base_y)
    peak = (x0 + (0.50 * w), y0 + (0.21 * h))
    right = (x0 + (0.83 * w), base_y)
    draw.polygon([left, peak, right], fill=tuple(fill), outline=tuple(shadow))
    draw.line([left, right], fill=tuple(shadow), width=max(2, int(w * 0.05)))
    draw.line(
        [peak, (x0 + (0.50 * w), base_y)],
        fill=tuple(shadow),
        width=max(2, int(w * 0.035)),
    )
    flap = [
        (x0 + (0.50 * w), base_y),
        (x0 + (0.61 * w), base_y),
        (x0 + (0.50 * w), y0 + (0.46 * h)),
    ]
    draw.polygon(flap, fill=tuple(flap_fill), outline=tuple(shadow))


__all__ = ["render_tents_scene"]
