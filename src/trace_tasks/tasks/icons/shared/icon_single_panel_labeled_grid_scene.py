"""Reusable single-panel labeled-grid chrome for icon pattern tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.text_rendering import load_font
from ...shared.text_legibility import draw_text_traced
from .icon_grid_scene import centered_square_bbox, resolve_fixed_grid_cell_slots
from .icon_scene import SingleIconPanelLayout, draw_single_panel, resolve_single_panel_layout
from .scene_style import IconCanvasStyle


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PreparedSinglePanelGridCell:
    """Resolved frame and content rectangles for one labeled grid cell."""

    label: str
    cell_bbox_xyxy: BBox
    content_bbox_xyxy: BBox


@dataclass(frozen=True)
class PreparedSinglePanelLabeledGridScene:
    """Prepared canvas + grid geometry for one single-panel labeled-grid scene."""

    image: Image.Image
    layout: SingleIconPanelLayout
    scene_cells: Tuple[PreparedSinglePanelGridCell, ...]


def resolve_single_panel_labeled_grid_canvas_size(
    *,
    rows: int,
    cols: int,
    cell_box_width_px: int,
    cell_box_height_px: int,
    render_params: Mapping[str, int | float | Sequence[int] | Sequence[float]],
) -> Tuple[int, int]:
    """Derive one single-panel canvas size that fits the sampled grid cell geometry."""

    rows_i = max(1, int(rows))
    cols_i = max(1, int(cols))
    cell_padding_px = int(render_params["cell_padding_px"])
    panel_padding_px = int(render_params["panel_padding_px"])
    outer_margin_px = int(render_params["outer_margin_px"])
    title_font_size_px = int(render_params["panel_title_font_size_px"])
    title_band_height = max(40, int(round(float(title_font_size_px) * 1.8)))

    content_width = int(cols_i * (int(cell_box_width_px) + (2 * cell_padding_px)))
    content_height = int(rows_i * (int(cell_box_height_px) + (2 * cell_padding_px)))
    panel_width = int(content_width + (2 * panel_padding_px))
    panel_height = int(content_height + title_band_height + panel_padding_px + (panel_padding_px // 2))
    canvas_width = int(panel_width + (2 * outer_margin_px))
    canvas_height = int(panel_height + (2 * outer_margin_px))
    return canvas_width, canvas_height


def prepare_single_panel_labeled_grid_scene(
    *,
    scene_labels: Sequence[str],
    grid_rows: int,
    grid_cols: int,
    canvas_width: int,
    canvas_height: int,
    outer_margin_px: int,
    panel_padding_px: int,
    panel_corner_radius_px: int,
    panel_title_font_size_px: int,
    background_rgb: Tuple[int, int, int],
    panel_fill_rgb: Tuple[int, int, int],
    panel_border_rgb: Tuple[int, int, int],
    title_color_rgb: Tuple[int, int, int],
    cell_padding_px: int,
    cell_border_rgb: Tuple[int, int, int],
    cell_label_color_rgb: Tuple[int, int, int],
    cell_label_stroke_rgb: Tuple[int, int, int] | None = None,
    cell_label_stroke_width_px: int = 1,
    cell_label_font_size_px: int,
    cell_corner_radius_px: int = 12,
    scene_content_side_padding_px: int = 12,
    scene_content_bottom_padding_px: int = 12,
    scene_content_top_offset_px: int = 40,
    scene_square_cells: bool = False,
    scene_title: str = "Pattern",
    icon_canvas_style: IconCanvasStyle | None = None,
) -> PreparedSinglePanelLabeledGridScene:
    """Draw single-panel labeled-grid chrome and return resolved cell geometry."""

    labels = [str(value) for value in scene_labels]
    if not labels:
        raise ValueError("scene_labels must contain at least one label")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid_rows and grid_cols must be positive")
    if len(labels) != int(grid_rows) * int(grid_cols):
        raise ValueError("scene_labels must match the explicit grid_rows * grid_cols layout")

    layout = resolve_single_panel_layout(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        outer_margin_px=int(outer_margin_px),
        panel_padding_px=int(panel_padding_px),
        title_font_size_px=int(panel_title_font_size_px),
        reserve_title=False,
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(int(v) for v in background_rgb),
        panel_fill_rgb=tuple(int(v) for v in panel_fill_rgb),
        panel_border_rgb=tuple(int(v) for v in panel_border_rgb),
        title_color_rgb=tuple(int(v) for v in title_color_rgb),
        corner_radius_px=int(panel_corner_radius_px),
        title_font_size_px=int(panel_title_font_size_px),
        scene_title="",
        icon_canvas_style=icon_canvas_style,
    )

    draw = ImageDraw.Draw(image)
    label_font = load_font(int(cell_label_font_size_px), bold=True)
    scene_cell_slots = resolve_fixed_grid_cell_slots(
        tuple(int(value) for value in layout.scene_content_xyxy),
        rows=int(grid_rows),
        cols=int(grid_cols),
        cell_padding_px=int(cell_padding_px),
    )
    scene_cells = []
    for label, cell_bbox in zip(labels, scene_cell_slots):
        resolved_cell_bbox = centered_square_bbox(cell_bbox) if bool(scene_square_cells) else tuple(
            int(value) for value in cell_bbox
        )
        draw.rounded_rectangle(
            resolved_cell_bbox,
            radius=max(0, int(cell_corner_radius_px)),
            outline=tuple(int(v) for v in cell_border_rgb),
            width=2,
            fill=tuple(int(v) for v in panel_fill_rgb),
        )
        draw_text_traced(draw,
            (int(resolved_cell_bbox[0] + 16), int(resolved_cell_bbox[1] + 14)),
            str(label),
            font=label_font,
            fill=tuple(int(v) for v in cell_label_color_rgb),
            stroke_fill=tuple(int(v) for v in (cell_label_stroke_rgb or panel_fill_rgb)),
            stroke_width=max(0, int(cell_label_stroke_width_px)),
            role="icon_cell_label_text",
            required=False,
        )
        scene_cells.append(
            PreparedSinglePanelGridCell(
                label=str(label),
                cell_bbox_xyxy=tuple(int(value) for value in resolved_cell_bbox),
                content_bbox_xyxy=tuple(
                    int(value)
                    for value in (
                        centered_square_bbox(
                            (
                                int(resolved_cell_bbox[0] + int(scene_content_side_padding_px)),
                                int(resolved_cell_bbox[1] + int(scene_content_top_offset_px)),
                                int(resolved_cell_bbox[2] - int(scene_content_side_padding_px)),
                                int(resolved_cell_bbox[3] - int(scene_content_bottom_padding_px)),
                            )
                        )
                        if bool(scene_square_cells)
                        else (
                            int(resolved_cell_bbox[0] + int(scene_content_side_padding_px)),
                            int(resolved_cell_bbox[1] + int(scene_content_top_offset_px)),
                            int(resolved_cell_bbox[2] - int(scene_content_side_padding_px)),
                            int(resolved_cell_bbox[3] - int(scene_content_bottom_padding_px)),
                        )
                    )
                ),
            )
        )

    return PreparedSinglePanelLabeledGridScene(
        image=image,
        layout=layout,
        scene_cells=tuple(scene_cells),
    )


__all__ = [
    "PreparedSinglePanelGridCell",
    "PreparedSinglePanelLabeledGridScene",
    "prepare_single_panel_labeled_grid_scene",
    "resolve_single_panel_labeled_grid_canvas_size",
]
