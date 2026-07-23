"""Two-row sequence-completion rendering for icon sequence-strip tasks."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Any, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_legibility import draw_centered_traced_text
from ....shared.text_rendering import load_font
from ...shared.icon_assets import render_icon_rgba
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import (
    IconInstanceSpec,
    RenderedIconInstance,
    SingleIconPanelLayout,
    draw_single_panel,
    resolve_single_panel_layout,
)
from ...shared.scene_style import IconCanvasStyle


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class IconSequenceCellSpec:
    """One sequence or option cell containing zero or more icon instances."""

    icon_instances: Tuple[IconInstanceSpec, ...] = ()
    is_missing: bool = False
    cell_label_text: str | None = None


@dataclass(frozen=True)
class RenderedSequenceCell:
    """Rendered metadata for one sequence-completion cell."""

    row_id: str
    cell_index: int
    cell_bbox_xyxy: BBox
    cell_label_text: str | None
    is_missing: bool
    icon_instances: Tuple[RenderedIconInstance, ...]


@dataclass(frozen=True)
class RenderedIconSequenceCompletionScene:
    """Complete rendered output for one sequence-completion scene."""

    image: Image.Image
    layout: SingleIconPanelLayout
    sequence_cells: Tuple[RenderedSequenceCell, ...]
    option_cells: Tuple[RenderedSequenceCell, ...]


def resolve_completion_canvas_size(
    *,
    cell_count: int,
    cell_box_width_px: int,
    cell_box_height_px: int,
    row_gap_px: int,
    render_params: Mapping[str, Any],
) -> Tuple[int, int]:
    """Derive a canvas size that fits two rows of fixed-width cells."""

    cell_padding_px = int(render_params["cell_padding_px"])
    panel_padding_px = int(render_params["panel_padding_px"])
    outer_margin_px = int(render_params["outer_margin_px"])
    content_width = int(cell_count) * int(int(cell_box_width_px) + (2 * cell_padding_px))
    content_height = (2 * int(int(cell_box_height_px) + (2 * cell_padding_px))) + int(row_gap_px)
    return (
        int(content_width + (2 * panel_padding_px) + (2 * outer_margin_px)),
        int(content_height + (2 * panel_padding_px) + (2 * outer_margin_px)),
    )


def validate_sequence_cell_box_bounds(render_params: Mapping[str, Any]) -> None:
    """Validate sampled sequence-completion cell width/height bounds."""

    if int(render_params["cell_box_width_min_px"]) > int(render_params["cell_box_width_max_px"]):
        raise ValueError("cell_box_width_min_px must be <= cell_box_width_max_px")
    if int(render_params["cell_box_height_min_px"]) > int(render_params["cell_box_height_max_px"]):
        raise ValueError("cell_box_height_min_px must be <= cell_box_height_max_px")


def _resolve_row_slots(
    content_bbox: BBox,
    *,
    cell_count: int,
    cell_box_width_px: int,
    cell_box_height_px: int,
    row_gap_px: int,
) -> Tuple[Tuple[BBox, ...], Tuple[BBox, ...]]:
    """Return centered top and bottom row slots."""

    x0, y0, x1, y1 = (int(value) for value in content_bbox)
    row_width = int(cell_count) * int(cell_box_width_px)
    gap_x = 0
    if int(cell_count) > 1:
        gap_x = max(8, int((int(x1 - x0) - row_width) // int(cell_count - 1)))
    total_width = row_width + (gap_x * max(0, int(cell_count) - 1))
    start_x = int(round(0.5 * float(x0 + x1 - total_width)))
    total_height = (2 * int(cell_box_height_px)) + int(row_gap_px)
    start_y = int(round(0.5 * float(y0 + y1 - total_height)))
    if start_x < x0 or start_y < y0 or start_y + total_height > y1:
        raise ValueError("sequence-completion canvas is too small for sampled cell geometry")

    rows: List[Tuple[BBox, ...]] = []
    for row_offset in (0, int(cell_box_height_px) + int(row_gap_px)):
        row_slots: List[BBox] = []
        cy0 = int(start_y + row_offset)
        for index in range(int(cell_count)):
            cx0 = int(start_x + index * (int(cell_box_width_px) + int(gap_x)))
            row_slots.append((cx0, cy0, int(cx0 + cell_box_width_px), int(cy0 + cell_box_height_px)))
        rows.append(tuple(row_slots))
    return tuple(rows[0]), tuple(rows[1])


def _grid_icon_bboxes(
    content_bbox: BBox,
    *,
    count: int,
    min_size_px: int,
    max_size_px: int,
    gap_px: int = 4,
) -> Tuple[Tuple[int, BBox], ...]:
    """Return deterministic icon grid bboxes for a cell."""

    count = int(count)
    if count <= 0:
        return ()
    x0, y0, x1, y1 = (int(value) for value in content_bbox)
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    cols = max(1, int(ceil(sqrt(float(count)))))
    rows = max(1, int(ceil(float(count) / float(cols))))
    size = min(
        int(max_size_px),
        max(int(min_size_px), int((width - (cols - 1) * int(gap_px)) // cols)),
        max(int(min_size_px), int((height - (rows - 1) * int(gap_px)) // rows)),
    )
    total_w = (cols * size) + ((cols - 1) * int(gap_px))
    total_h = (rows * size) + ((rows - 1) * int(gap_px))
    start_x = int(round(float(x0) + 0.5 * float(width - total_w)))
    start_y = int(round(float(y0) + 0.5 * float(height - total_h)))
    bboxes: List[Tuple[int, BBox]] = []
    for icon_index in range(count):
        row = int(icon_index // cols)
        col = int(icon_index % cols)
        ix0 = int(start_x + col * (size + int(gap_px)))
        iy0 = int(start_y + row * (size + int(gap_px)))
        bboxes.append((int(size), (ix0, iy0, int(ix0 + size), int(iy0 + size))))
    return tuple(bboxes)


def _draw_cell(
    *,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    cell_spec: IconSequenceCellSpec,
    cell_bbox: BBox,
    row_id: str,
    cell_index: int,
    render_params: Mapping[str, Any],
) -> RenderedSequenceCell:
    """Draw one sequence or option cell."""

    panel_fill_rgb = tuple(int(value) for value in render_params["panel_fill_rgb"])
    draw.rounded_rectangle(
        tuple(int(value) for value in cell_bbox),
        radius=max(0, int(render_params["cell_corner_radius_px"])),
        outline=tuple(int(value) for value in render_params["cell_border_rgb"]),
        width=2,
        fill=panel_fill_rgb,
    )

    label_text = None if cell_spec.cell_label_text is None else str(cell_spec.cell_label_text).strip()
    label_band_height = 0
    if label_text:
        label_font_size = max(10, int(render_params["cell_label_font_size_px"]))
        label_band_height = max(24, int(round(float(label_font_size) * 1.45)))
        draw_centered_traced_text(
            draw,
            text=str(label_text),
            center=(
                0.5 * float(cell_bbox[0] + cell_bbox[2]),
                float(cell_bbox[1]) + (0.5 * float(label_band_height)),
            ),
            font=load_font(label_font_size, bold=True),
            fill_rgb=tuple(int(value) for value in render_params["cell_label_color_rgb"]),
            stroke_rgb=panel_fill_rgb,
            stroke_width=2,
            role="icon_option_label_text",
            required=False,
        )

    inner_padding = max(0, int(render_params["cell_icon_padding_px"]))
    icon_content_bbox = (
        int(cell_bbox[0] + inner_padding),
        int(cell_bbox[1] + label_band_height + inner_padding),
        int(cell_bbox[2] - inner_padding),
        int(cell_bbox[3] - inner_padding),
    )
    if int(icon_content_bbox[1]) >= int(icon_content_bbox[3]):
        raise ValueError("sequence-completion cell is too short for label and icon content")

    if bool(cell_spec.is_missing):
        draw_centered_traced_text(
            draw,
            text="?",
            center=(
                0.5 * float(icon_content_bbox[0] + icon_content_bbox[2]),
                0.5 * float(icon_content_bbox[1] + icon_content_bbox[3]),
            ),
            font=load_font(int(render_params["missing_mark_font_size_px"]), bold=True),
            fill_rgb=tuple(int(value) for value in render_params["missing_mark_color_rgb"]),
            stroke_rgb=panel_fill_rgb,
            stroke_width=2,
            role="icon_missing_mark_text",
            required=False,
        )
        return RenderedSequenceCell(
            row_id=str(row_id),
            cell_index=int(cell_index),
            cell_bbox_xyxy=tuple(int(value) for value in cell_bbox),
            cell_label_text=label_text,
            is_missing=True,
            icon_instances=(),
        )

    min_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
    rendered_instances: List[RenderedIconInstance] = []
    for icon_index, (icon_spec, (resolved_size, paste_bbox)) in enumerate(
        zip(
            cell_spec.icon_instances,
            _grid_icon_bboxes(
                icon_content_bbox,
                count=len(cell_spec.icon_instances),
                min_size_px=min_size,
                max_size_px=max_size,
            ),
        )
    ):
        nominal_size = int(icon_spec.nominal_size_px or resolved_size)
        sprite = render_icon_rgba(
            icon_id=str(icon_spec.icon_id),
            size_px=int(nominal_size),
            tint_rgb=tuple(int(value) for value in icon_spec.tint_rgb),
            rotation_degrees=int(icon_spec.rotation_degrees),
            mirror_x=bool(icon_spec.mirror_x),
            noise_edits=tuple(icon_spec.noise_edits),
            noise_seed=icon_spec.noise_seed,
        )
        cx = 0.5 * float(paste_bbox[0] + paste_bbox[2])
        cy = 0.5 * float(paste_bbox[1] + paste_bbox[3])
        px0 = int(round(cx - (0.5 * float(sprite.size[0]))))
        py0 = int(round(cy - (0.5 * float(sprite.size[1]))))
        paste_bbox = (px0, py0, int(px0 + sprite.size[0]), int(py0 + sprite.size[1]))
        image.alpha_composite(sprite, (int(px0), int(py0)))
        rendered_instances.append(
            RenderedIconInstance(
                instance_id=f"{str(row_id)}_{int(cell_index)}_icon_{int(icon_index)}",
                icon_id=str(icon_spec.icon_id),
                panel="scene",
                bbox_xyxy=tuple(int(value) for value in paste_bbox),
                nominal_size_px=int(nominal_size),
                rotation_degrees=int(icon_spec.rotation_degrees) % 360,
                mirror_x=bool(icon_spec.mirror_x),
                tint_rgb=tuple(int(value) for value in icon_spec.tint_rgb),
                noise_edits=serialize_icon_noise_edits(icon_spec.noise_edits),
                noise_seed=None if icon_spec.noise_seed is None else int(icon_spec.noise_seed),
            )
        )

    return RenderedSequenceCell(
        row_id=str(row_id),
        cell_index=int(cell_index),
        cell_bbox_xyxy=tuple(int(value) for value in cell_bbox),
        cell_label_text=label_text,
        is_missing=False,
        icon_instances=tuple(rendered_instances),
    )


def render_sequence_completion_scene_from_params(
    *,
    sequence_cells: Sequence[IconSequenceCellSpec],
    option_cells: Sequence[IconSequenceCellSpec],
    canvas_width: int,
    canvas_height: int,
    render_params: Mapping[str, Any],
) -> RenderedIconSequenceCompletionScene:
    """Render fixed sequence and option rows with stable cell geometry."""

    if len(sequence_cells) != 4 or len(option_cells) != 4:
        raise ValueError("sequence completion scenes require exactly four sequence cells and four option cells")
    layout = resolve_single_panel_layout(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(int(value) for value in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(value) for value in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(value) for value in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        scene_title="",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    top_slots, bottom_slots = _resolve_row_slots(
        tuple(int(value) for value in layout.scene_content_xyxy),
        cell_count=4,
        cell_box_width_px=int(render_params["cell_box_width_px"]),
        cell_box_height_px=int(render_params["cell_box_height_px"]),
        row_gap_px=int(render_params["row_gap_px"]),
    )
    draw = ImageDraw.Draw(image)
    rendered_sequence = tuple(
        _draw_cell(
            image=image,
            draw=draw,
            cell_spec=cell_spec,
            cell_bbox=cell_bbox,
            row_id="sequence",
            cell_index=cell_index,
            render_params=render_params,
        )
        for cell_index, (cell_spec, cell_bbox) in enumerate(zip(sequence_cells, top_slots))
    )
    rendered_options = tuple(
        _draw_cell(
            image=image,
            draw=draw,
            cell_spec=cell_spec,
            cell_bbox=cell_bbox,
            row_id="option",
            cell_index=cell_index,
            render_params=render_params,
        )
        for cell_index, (cell_spec, cell_bbox) in enumerate(zip(option_cells, bottom_slots))
    )
    return RenderedIconSequenceCompletionScene(
        image=image.convert("RGB"),
        layout=layout,
        sequence_cells=rendered_sequence,
        option_cells=rendered_options,
    )


__all__ = [
    "IconSequenceCellSpec",
    "RenderedIconSequenceCompletionScene",
    "RenderedSequenceCell",
    "render_sequence_completion_scene_from_params",
    "resolve_completion_canvas_size",
    "validate_sequence_cell_box_bounds",
]
