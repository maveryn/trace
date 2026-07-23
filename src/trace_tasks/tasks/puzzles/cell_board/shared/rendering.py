"""Rendering helpers for rectangular cell-board puzzles."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.puzzles.shared.common import resolve_puzzle_axis_variant
from trace_tasks.tasks.puzzles.shared.scene_style import (
    draw_puzzle_chrome_by_mode,
    make_puzzle_scene_background,
    resolve_panel_chrome_mode,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.shared.bbox_projection import BBox
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font

from .layout import resolve_board_layout, resolve_tile_size
from .state import Color, NamedColor, RenderedCellBoard
from .topology import Coord, cell_id

SUPPORTED_CELL_TILE_STYLES: Tuple[str, ...] = (
    "classic_grid",
    "rounded_tiles",
    "inset_tiles",
    "lab_matrix",
)


def _style_color(value: Sequence[int]) -> Color:
    return (int(value[0]), int(value[1]), int(value[2]))


def _resolve_tile_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    return resolve_puzzle_axis_variant(
        params=params,
        gen_defaults=rendering_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_CELL_TILE_STYLES,
        task_id=str(namespace),
        explicit_key="cell_board_tile_style",
        weights_key="cell_board_tile_style_weights",
        balance_flag_key="balanced_cell_board_tile_style_sampling",
        axis_namespace="cell_board_tile_style",
    )


def _bbox_map_for_layout(layout) -> Dict[str, BBox]:
    """Project every logical board coordinate into final image pixels."""

    bbox_map: Dict[str, BBox] = {}
    for row in range(int(layout.rows)):
        for col in range(int(layout.cols)):
            x0 = int(layout.board_origin_x_px) + (int(col) * int(layout.tile_width_px))
            y0 = int(layout.board_origin_y_px) + (int(row) * int(layout.tile_height_px))
            x1 = int(x0) + int(layout.tile_width_px)
            y1 = int(y0) + int(layout.tile_height_px)
            bbox_map[cell_id((row, col))] = (
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            )
    return bbox_map


def _draw_cell_text(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    text: str,
    font_family: str,
    fill: Color,
    stroke: Color,
) -> None:
    """Draw compact per-cell labels such as start/goal markers."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    size = max(12, int(round(min(float(x1 - x0), float(y1 - y0)) * 0.46)))
    font = load_font(size, bold=True, font_family=font_family)
    draw_text_centered(
        draw,
        text=str(text),
        center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        font=font,
        fill=fill,
        stroke_fill=stroke,
        stroke_width=max(1, int(round(size * 0.10))),
    )


def _draw_clean_semantic_cell(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    fill: Color,
    outline: Color,
    width: int,
    tile_style: str,
) -> None:
    """Draw a semantic color cell without treatment-specific interior marks."""

    x0, y0, x1, y1 = [int(round(value)) for value in bbox]
    cell_w = max(1, int(x1 - x0))
    cell_h = max(1, int(y1 - y0))
    style = str(tile_style)

    if style == "rounded_tiles":
        radius = max(3, int(round(min(cell_w, cell_h) * 0.10)))
        gap = max(1, int(round(min(cell_w, cell_h) * 0.02)))
        draw.rounded_rectangle(
            (x0 + gap, y0 + gap, x1 - gap, y1 - gap),
            radius=radius,
            fill=fill,
            outline=outline,
            width=max(1, int(width)),
        )
        return

    if style == "inset_tiles":
        gap = max(2, int(round(min(cell_w, cell_h) * 0.04)))
        draw.rectangle(
            (x0 + gap, y0 + gap, x1 - gap, y1 - gap),
            fill=fill,
            outline=outline,
            width=max(1, int(width)),
        )
    elif style == "lab_matrix":
        draw.rectangle(
            (x0, y0, x1, y1),
            fill=fill,
            outline=outline,
            width=max(2, int(width)),
        )
    else:
        draw.rectangle(
            (x0, y0, x1, y1),
            fill=fill,
            outline=outline,
            width=max(1, int(width)),
        )


def render_cell_board(
    *,
    rows: int,
    cols: int,
    board_colors: Mapping[Coord, NamedColor],
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    visual_defaults: Mapping[str, Any],
    coordinate_labels: bool = False,
    cell_text: Mapping[Coord, str] | None = None,
) -> RenderedCellBoard:
    """Render one styled rectangular cell board and expose bbox projections."""

    layout_rng = spawn_rng(int(instance_seed), "puzzles.cell_board.layout")
    tile_w, tile_h, tile_meta = resolve_tile_size(
        rng=layout_rng,
        params=params,
        rendering_defaults=rendering_defaults,
    )
    layout = resolve_board_layout(
        rng=layout_rng,
        rows=int(rows),
        cols=int(cols),
        tile_width_px=int(tile_w),
        tile_height_px=int(tile_h),
        coordinate_labels=bool(coordinate_labels),
        params=params,
        rendering_defaults=rendering_defaults,
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace="puzzles.cell_board.scene_style",
    )
    chrome_mode, chrome_meta = resolve_panel_chrome_mode(
        instance_seed=int(instance_seed),
        namespace="puzzles.cell_board.chrome",
    )
    tile_style, tile_style_probs = _resolve_tile_style(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
        namespace="puzzles.cell_board",
    )
    image, background_meta = make_puzzle_scene_background(
        canvas_width=int(layout.canvas_width_px),
        canvas_height=int(layout.canvas_height_px),
        style=scene_style,
    )
    draw = ImageDraw.Draw(image)
    panel_pad = max(7, int(round(min(tile_w, tile_h) * 0.22)))
    panel_bbox = (
        int(layout.board_origin_x_px - panel_pad),
        int(layout.board_origin_y_px - panel_pad),
        int(layout.board_origin_x_px + layout.board_width_px + panel_pad),
        int(layout.board_origin_y_px + layout.board_height_px + panel_pad),
    )
    draw_puzzle_chrome_by_mode(
        draw,
        bbox=panel_bbox,
        style=scene_style,
        radius=max(6, int(round(min(tile_w, tile_h) * 0.18))),
        border_width=max(1, int(round(min(tile_w, tile_h) * 0.05))),
        mode=str(chrome_mode),
    )

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="puzzles.cell_board",
        params=params,
    )
    font_record = get_font_family_record(font_family)
    bbox_map = _bbox_map_for_layout(layout)
    outline_width = max(1, int(round(min(tile_w, tile_h) * 0.05)))
    entities = []
    text_map = {
        (int(row), int(col)): str(value)
        for (row, col), value in dict(cell_text or {}).items()
    }
    for row in range(int(rows)):
        for col in range(int(cols)):
            coord = (int(row), int(col))
            color_name, rgb = board_colors[coord]
            bbox = bbox_map[cell_id(coord)]
            _draw_clean_semantic_cell(
                draw,
                bbox=bbox,
                fill=rgb,
                outline=scene_style.grid_rgb,
                width=int(outline_width),
                tile_style=str(tile_style),
            )
            if coord in text_map:
                _draw_cell_text(
                    draw,
                    bbox=bbox,
                    text=text_map[coord],
                    font_family=str(font_family),
                    fill=_style_color(scene_style.text_rgb),
                    stroke=_style_color(scene_style.text_stroke_rgb),
                )
            entities.append(
                {
                    "id": cell_id(coord),
                    "type": "cell",
                    "row": int(row),
                    "col": int(col),
                    "color_name": str(color_name),
                    "rgb": [int(value) for value in rgb],
                    "bbox": [round(float(value), 3) for value in bbox],
                }
            )

    if bool(coordinate_labels):
        label_font = load_font(
            int(layout.label_font_size_px),
            bold=True,
            font_family=str(font_family),
        )
        for col in range(int(cols)):
            center_x = float(layout.board_origin_x_px) + (
                (float(col) + 0.5) * float(layout.tile_width_px)
            )
            center_y = float(layout.board_origin_y_px) - max(14, tile_h * 0.45)
            draw_text_centered(
                draw,
                text=str(col + 1),
                center=(center_x, center_y),
                font=label_font,
                fill=scene_style.text_rgb,
                stroke_fill=scene_style.text_stroke_rgb,
            )
        for row in range(int(rows)):
            center_x = float(layout.board_origin_x_px) - max(14, tile_w * 0.45)
            center_y = float(layout.board_origin_y_px) + (
                (float(row) + 0.5) * float(layout.tile_height_px)
            )
            draw_text_centered(
                draw,
                text=str(row + 1),
                center=(center_x, center_y),
                font=label_font,
                fill=scene_style.text_rgb,
                stroke_fill=scene_style.text_stroke_rgb,
            )

    noisy_image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=visual_defaults,
    )
    render_meta = {
        "scene_id": "cell_board",
        "rows": int(rows),
        "cols": int(cols),
        "tile_width_px": int(tile_w),
        "tile_height_px": int(tile_h),
        "tile_size_sampling": dict(tile_meta),
        "coordinate_labels": bool(coordinate_labels),
        "tile_style": str(tile_style),
        "tile_style_probabilities": dict(tile_style_probs),
        "chrome": dict(chrome_meta),
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_record": font_record.to_trace(),
        "scene_style": dict(scene_style_meta),
    }
    background_meta = {
        **dict(background_meta),
        "scene_style": dict(scene_style_meta),
        "cell_board": {
            "tile_style": str(tile_style),
            "tile_style_probabilities": dict(tile_style_probs),
            "chrome": dict(chrome_meta),
        },
    }
    return RenderedCellBoard(
        image=noisy_image,
        bbox_map=bbox_map,
        layout=layout,
        entities=entities,
        render_meta=render_meta,
        background_meta=background_meta,
        post_noise_meta=post_noise_meta,
    )


__all__ = ["SUPPORTED_CELL_TILE_STYLES", "render_cell_board"]
