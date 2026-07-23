"""Layout primitives for symbolic Life automaton scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.unit_size_jitter import scale_symbolic_px

from .state import LifeRenderParams, LifeSceneSpec


def grid_bbox(*, left: int, top: int, rows: int, cols: int, cell_size: int, gap: int) -> Tuple[int, int, int, int]:
    """Return one grid bbox."""

    width = int(cols * cell_size + max(0, cols - 1) * gap)
    height = int(rows * cell_size + max(0, rows - 1) * gap)
    return (int(left), int(top), int(left + width), int(top + height))


def cell_bbox(*, left: int, top: int, row: int, col: int, cell_size: int, gap: int) -> Tuple[int, int, int, int]:
    """Return one cell bbox."""

    x0 = int(left + col * (cell_size + gap))
    y0 = int(top + row * (cell_size + gap))
    return (x0, y0, int(x0 + cell_size), int(y0 + cell_size))


def inset_bbox(bbox: Sequence[int], inset: int) -> Tuple[int, int, int, int]:
    """Inset one bbox without inverting it."""

    x0, y0, x1, y1 = [int(value) for value in bbox]
    inset_px = max(0, int(inset))
    return (
        min(x1, x0 + inset_px),
        min(y1, y0 + inset_px),
        max(x0, x1 - inset_px),
        max(y0, y1 - inset_px),
    )


def marked_cells_bbox(*, cells: Sequence[Tuple[int, int]], grid_left: int, grid_top: int, cell_size: int, gap: int) -> Tuple[int, int, int, int]:
    """Return the bbox spanning marked cells."""

    boxes = [
        cell_bbox(left=grid_left, top=grid_top, row=int(row), col=int(col), cell_size=cell_size, gap=gap)
        for row, col in cells
    ]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def option_grid_gap(render_params: LifeRenderParams) -> int:
    """Return the gap between option-grid cells."""

    return max(1, int(render_params.grid_gap_px))


def option_vertical_gap(render_params: LifeRenderParams) -> int:
    """Return vertical gap between source panel and options."""

    unit_scale = float(render_params.unit_size_jitter.get("scale", 1.0))
    return scale_symbolic_px(50, unit_scale, min_px=28)


def source_header_height(render_params: LifeRenderParams, *, enabled: bool) -> int:
    """Return reserved panel header height for the marked source grid."""

    if not bool(enabled):
        return 0
    return max(24, int(round(render_params.label_font_size_px * 1.35)))


def option_card_size(*, rows: int, cols: int, render_params: LifeRenderParams) -> Tuple[int, int]:
    """Return one option card size."""

    cell = int(render_params.option_grid_cell_px)
    gap = option_grid_gap(render_params)
    grid_w = int(cols * cell + max(0, cols - 1) * gap)
    grid_h = int(rows * cell + max(0, rows - 1) * gap)
    pad_x = max(10, int(round(render_params.panel_padding_px * 0.45)))
    header_h = max(30, int(round(render_params.label_font_size_px * 1.45)))
    pad_bottom = max(10, int(round(render_params.panel_padding_px * 0.40)))
    return (
        max(int(render_params.option_card_width_px), int(grid_w + 2 * pad_x)),
        max(int(render_params.option_card_height_px) + 38, int(header_h + grid_h + pad_bottom)),
    )


def content_metrics(*, scene: LifeSceneSpec, render_params: LifeRenderParams) -> Dict[str, int]:
    """Return content extents used by fitting and rendering."""

    rows, cols = int(scene.rows), int(scene.cols)
    cell = int(render_params.cell_size_px)
    gap = int(render_params.grid_gap_px)
    grid_box = grid_bbox(left=0, top=0, rows=rows, cols=cols, cell_size=cell, gap=gap)
    grid_w = int(grid_box[2] - grid_box[0])
    grid_h = int(grid_box[3] - grid_box[1])
    panel_w = int(grid_w + 2 * int(render_params.panel_padding_px))
    header_h = source_header_height(render_params, enabled=bool(str(scene.source_marker_label).strip()))
    panel_h = int(grid_h + 2 * int(render_params.panel_padding_px) + header_h)
    option_w = 0
    option_h = 0
    option_gap_y = 0
    if scene.option_specs:
        card_w, card_h = option_card_size(rows=rows, cols=cols, render_params=render_params)
        gap_x = int(render_params.option_gap_px)
        option_w = int(len(scene.option_specs) * card_w + max(0, len(scene.option_specs) - 1) * gap_x)
        option_h = int(card_h)
        option_gap_y = int(option_vertical_gap(render_params))
    content_w = int(max(panel_w, option_w))
    content_h = int(panel_h + (option_gap_y + option_h if option_h else 0))
    return {
        "grid_width_px": grid_w,
        "grid_height_px": grid_h,
        "panel_width_px": panel_w,
        "panel_height_px": panel_h,
        "option_width_px": option_w,
        "option_height_px": option_h,
        "option_vertical_gap_px": option_gap_y,
        "content_width_px": content_w,
        "content_height_px": content_h,
    }


def fit_life_render_params(*, scene: LifeSceneSpec, render_params: LifeRenderParams) -> LifeRenderParams:
    """Shrink/fit Life grids into the configured canvas."""

    if scene.option_specs:
        safe_margin = max(18, int(round(render_params.panel_padding_px * 0.85)))
        min_cell = max(18, min(int(render_params.cell_size_px), 22))
        for cell in range(int(render_params.cell_size_px), min_cell - 1, -1):
            candidate = replace(render_params, cell_size_px=int(cell), option_grid_cell_px=int(cell))
            metrics = content_metrics(scene=scene, render_params=candidate)
            if (
                int(metrics["content_width_px"]) <= int(render_params.canvas_width - 2 * safe_margin)
                and int(metrics["content_height_px"]) <= int(render_params.canvas_height - 2 * safe_margin)
            ):
                render_params = candidate
                break
        else:
            render_params = replace(render_params, cell_size_px=int(min_cell), option_grid_cell_px=int(min_cell))

    metrics = content_metrics(scene=scene, render_params=render_params)
    rng = spawn_rng(instance_seed=int(render_params.layout_seed), namespace="life_automaton_canvas")
    min_margin = max(24, int(round(render_params.panel_padding_px * 1.05)))
    max_margin = max(min_margin, int(round(render_params.panel_padding_px * 2.25)))
    left_margin = rng.randint(min_margin, max_margin)
    right_margin = rng.randint(min_margin, max_margin)
    top_margin = rng.randint(min_margin, max_margin)
    bottom_margin = rng.randint(min_margin, max_margin)
    min_canvas_w = 520 if scene.option_specs else 420
    min_canvas_h = 420 if scene.option_specs else 360
    target_w = int(metrics["content_width_px"] + left_margin + right_margin)
    target_h = int(metrics["content_height_px"] + top_margin + bottom_margin)
    canvas_w = min(int(render_params.canvas_width), max(int(min_canvas_w), target_w))
    canvas_h = min(int(render_params.canvas_height), max(int(min_canvas_h), target_h))
    return replace(render_params, canvas_width=int(canvas_w), canvas_height=int(canvas_h))
