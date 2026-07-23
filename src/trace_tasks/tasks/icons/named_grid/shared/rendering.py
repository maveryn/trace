"""Neutral renderer for the named-grid icons scene."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import draw_text_centered, load_font
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import BBox, draw_single_panel, resolve_single_panel_layout, single_panel_geometry_to_trace
from ...shared.icon_style import sample_icon_palette
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.procedural_named_icons import (
    procedural_named_icon_display_name,
    render_procedural_named_icon_rgba,
    sample_procedural_named_icon_fill_style,
)
from ...shared.procedural_named_icon_field_scene import rotation_for_named_shape

from .defaults import NamedGridDefaults
from .state import NamedGridRenderSpec, NamedGridScenePayload, RenderedGridIcon
from .styles import render_int, render_rgb


_DEFAULTS = NamedGridDefaults()
_NOISE_NAMESPACE = "icons.named_grid.icon_noise"


def resolve_grid_bboxes(
    *,
    content_bbox: BBox,
    rows: int,
    cols: int,
    row_label_band_width_px: int,
    column_label_band_height_px: int,
    grid_label_gap_px: int,
    grid_cell_max_size_px: int,
) -> Tuple[BBox, Tuple[Tuple[BBox, ...], ...], int]:
    x0, y0, x1, y1 = tuple(int(value) for value in content_bbox)
    grid_area_x0 = int(x0 + int(row_label_band_width_px) + int(grid_label_gap_px))
    grid_area_y0 = int(y0 + int(column_label_band_height_px) + int(grid_label_gap_px))
    available_w = max(1, int(x1 - grid_area_x0))
    available_h = max(1, int(y1 - grid_area_y0))
    cell_size = min(int(grid_cell_max_size_px), int(available_w // max(1, int(cols))), int(available_h // max(1, int(rows))))
    if int(cell_size) < 42:
        raise ValueError("named-grid content area is too small for clear cells")
    grid_w = int(cell_size) * int(cols)
    grid_h = int(cell_size) * int(rows)
    gx0 = int(grid_area_x0 + max(0, (available_w - grid_w) // 2))
    gy0 = int(grid_area_y0 + max(0, (available_h - grid_h) // 2))
    grid_bbox = (int(gx0), int(gy0), int(gx0 + grid_w), int(gy0 + grid_h))
    cell_rows: List[Tuple[BBox, ...]] = []
    for row in range(int(rows)):
        row_boxes: List[BBox] = []
        for col in range(int(cols)):
            cx0 = int(gx0 + int(col) * int(cell_size))
            cy0 = int(gy0 + int(row) * int(cell_size))
            row_boxes.append((int(cx0), int(cy0), int(cx0 + cell_size), int(cy0 + cell_size)))
        cell_rows.append(tuple(row_boxes))
    return tuple(int(value) for value in grid_bbox), tuple(cell_rows), int(cell_size)


def render_named_grid_scene(
    *,
    sample: NamedGridRenderSpec,
    instance_seed: int,
    render_params: Mapping[str, Any],
    params: Mapping[str, Any],
    rng,
) -> NamedGridScenePayload:
    """Render one complete grid while preserving cell/icon projection state.

    The renderer is scene-level only: it draws numbered row/column labels,
    places one procedural icon at each cell center, and records both cell and
    icon bboxes so task files can bind objective-specific annotations.
    """

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
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
    row_label_band_width_px = int(render_params.get("row_label_band_width_px", render_int(params, "row_label_band_width_px", _DEFAULTS.row_label_band_width_px)))
    column_label_band_height_px = int(render_params.get("column_label_band_height_px", render_int(params, "column_label_band_height_px", _DEFAULTS.column_label_band_height_px)))
    grid_label_gap_px = int(render_params.get("grid_label_gap_px", render_int(params, "grid_label_gap_px", _DEFAULTS.grid_label_gap_px)))
    grid_cell_max_size_px = int(render_params.get("grid_cell_max_size_px", render_int(params, "grid_cell_max_size_px", _DEFAULTS.grid_cell_max_size_px)))
    grid_cell_padding_px = int(render_params.get("grid_cell_padding_px", render_int(params, "grid_cell_padding_px", _DEFAULTS.grid_cell_padding_px)))
    grid_line_width_px = int(render_params.get("grid_line_width_px", render_int(params, "grid_line_width_px", _DEFAULTS.grid_line_width_px)))
    grid_border_width_px = int(render_params.get("grid_border_width_px", render_int(params, "grid_border_width_px", _DEFAULTS.grid_border_width_px)))
    axis_label_font_size_px = int(render_params.get("axis_label_font_size_px", render_int(params, "axis_label_font_size_px", _DEFAULTS.axis_label_font_size_px)))
    grid_line_rgb = tuple(int(value) for value in render_params.get("grid_line_rgb", render_rgb(params, "grid_line_rgb", _DEFAULTS.grid_line_rgb)))
    cell_fill_rgb = tuple(int(value) for value in render_params.get("cell_fill_rgb", render_rgb(params, "cell_fill_rgb", _DEFAULTS.cell_fill_rgb)))
    alternate_cell_fill_rgb = tuple(int(value) for value in render_params.get("alternate_cell_fill_rgb", render_rgb(params, "alternate_cell_fill_rgb", _DEFAULTS.alternate_cell_fill_rgb)))
    axis_label_rgb = tuple(int(value) for value in render_params.get("axis_label_rgb", render_rgb(params, "axis_label_rgb", _DEFAULTS.axis_label_rgb)))
    axis_label_stroke_rgb = tuple(int(value) for value in render_params.get("axis_label_stroke_rgb", render_params["panel_fill_rgb"]))

    grid_bbox, cell_bboxes, cell_size_px = resolve_grid_bboxes(
        content_bbox=tuple(int(value) for value in layout.scene_content_xyxy),
        rows=int(sample.grid_rows),
        cols=int(sample.grid_cols),
        row_label_band_width_px=int(row_label_band_width_px),
        column_label_band_height_px=int(column_label_band_height_px),
        grid_label_gap_px=int(grid_label_gap_px),
        grid_cell_max_size_px=int(grid_cell_max_size_px),
    )

    draw = ImageDraw.Draw(image)
    axis_font = load_font(int(axis_label_font_size_px), bold=True)
    for col in range(int(sample.grid_cols)):
        cell_bbox = cell_bboxes[0][int(col)]
        draw_text_centered(
            draw,
            text=str(int(col) + 1),
            center=(0.5 * float(cell_bbox[0] + cell_bbox[2]), float(grid_bbox[1] - (0.5 * column_label_band_height_px))),
            font=axis_font,
            fill=tuple(int(value) for value in axis_label_rgb),
            stroke_fill=tuple(int(value) for value in axis_label_stroke_rgb),
            stroke_width=2,
        )
    for row in range(int(sample.grid_rows)):
        cell_bbox = cell_bboxes[int(row)][0]
        draw_text_centered(
            draw,
            text=str(int(row) + 1),
            center=(float(grid_bbox[0] - (0.5 * row_label_band_width_px)), 0.5 * float(cell_bbox[1] + cell_bbox[3])),
            font=axis_font,
            fill=tuple(int(value) for value in axis_label_rgb),
            stroke_fill=tuple(int(value) for value in axis_label_stroke_rgb),
            stroke_width=2,
        )

    for row in range(int(sample.grid_rows)):
        for col in range(int(sample.grid_cols)):
            cell_bbox = cell_bboxes[int(row)][int(col)]
            fill = alternate_cell_fill_rgb if (int(row) + int(col)) % 2 else cell_fill_rgb
            draw.rectangle(
                cell_bbox,
                fill=tuple(int(value) for value in fill),
                outline=tuple(int(value) for value in grid_line_rgb),
                width=max(1, int(grid_line_width_px)),
            )
    draw.rectangle(
        grid_bbox,
        outline=tuple(int(value) for value in grid_line_rgb),
        width=max(1, int(grid_border_width_px)),
    )

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    palette = sample_icon_palette(
        rng,
        palette_size=int(palette_size),
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(value) for value in render_params["background_color_rgb"]),
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["panel_border_rgb"]),
            tuple(int(value) for value in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    min_icon_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_icon_size = max(min_icon_size, min(int(render_params["scene_icon_size_max_px"]), int(cell_size_px) - (2 * int(grid_cell_padding_px))))
    if max_icon_size < min_icon_size:
        min_icon_size = max_icon_size
    counted_set = set((int(row), int(col)) for row, col in sample.counted_cells)
    icons: List[RenderedGridIcon] = []
    for row in range(int(sample.grid_rows)):
        for col in range(int(sample.grid_cols)):
            shape_id = str(sample.shape_ids_by_cell[int(row)][int(col)])
            fill_style = sample_procedural_named_icon_fill_style(
                rng,
                support=sample.fill_style_support,
                probabilities=sample.fill_style_probabilities,
            )
            tint_rgb = tuple(int(value) for value in rng.choice(palette))
            nominal_size_px = int(rng.randint(int(min_icon_size), int(max_icon_size)))
            noise_edits, noise_seed = sample_icon_instance_noise(
                instance_seed=int(instance_seed),
                namespace=f"{_NOISE_NAMESPACE}:r{int(row)}c{int(col)}",
                render_params=render_params,
            )
            rotation_degrees = rotation_for_named_shape(rng, str(shape_id))
            sprite = render_procedural_named_icon_rgba(
                shape_id=str(shape_id),
                size_px=int(nominal_size_px),
                tint_rgb=tint_rgb,
                fill_style=str(fill_style),
                rotation_degrees=int(rotation_degrees),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
            cell_bbox = cell_bboxes[int(row)][int(col)]
            cx = 0.5 * float(cell_bbox[0] + cell_bbox[2])
            cy = 0.5 * float(cell_bbox[1] + cell_bbox[3])
            x0 = int(round(cx - (0.5 * float(sprite.size[0]))))
            y0 = int(round(cy - (0.5 * float(sprite.size[1]))))
            bbox = (int(x0), int(y0), int(x0 + sprite.size[0]), int(y0 + sprite.size[1]))
            image.alpha_composite(sprite, (int(x0), int(y0)))
            instance_id = f"grid_r{int(row) + 1}_c{int(col) + 1}"
            icons.append(
                RenderedGridIcon(
                    instance_id=str(instance_id),
                    row_index=int(row),
                    col_index=int(col),
                    row_number=int(row) + 1,
                    column_number=int(col) + 1,
                    shape_id=str(shape_id),
                    shape_name=procedural_named_icon_display_name(str(shape_id)),
                    bbox_xyxy=tuple(int(value) for value in bbox),
                    cell_bbox_xyxy=tuple(int(value) for value in cell_bbox),
                    nominal_size_px=int(nominal_size_px),
                    tint_rgb=tuple(int(value) for value in tint_rgb),
                    fill_style=str(fill_style),
                    rotation_degrees=int(rotation_degrees),
                    noise_edits=tuple(serialize_icon_noise_edits(noise_edits)),
                    noise_seed=int(noise_seed),
                    is_target_shape=str(shape_id) == str(sample.target_shape_id),
                    is_counted=(int(row), int(col)) in counted_set,
                )
            )
    return NamedGridScenePayload(
        image=image.convert("RGB"),
        panel_geometry=single_panel_geometry_to_trace(layout),
        grid_bbox_xyxy=tuple(int(value) for value in grid_bbox),
        cell_bboxes_xyxy=tuple(tuple(tuple(int(value) for value in box) for box in row) for row in cell_bboxes),
        icons=tuple(icons),
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in palette),
        cell_size_px=int(cell_size_px),
    )


def serialize_named_grid_icon(icon: RenderedGridIcon) -> Dict[str, Any]:
    return {
        "instance_id": str(icon.instance_id),
        "row_index": int(icon.row_index),
        "col_index": int(icon.col_index),
        "row_number": int(icon.row_number),
        "column_number": int(icon.column_number),
        "shape_id": str(icon.shape_id),
        "shape_name": str(icon.shape_name),
        "bbox_xyxy": [int(value) for value in icon.bbox_xyxy],
        "cell_bbox_xyxy": [int(value) for value in icon.cell_bbox_xyxy],
        "nominal_size_px": int(icon.nominal_size_px),
        "rotation_degrees": int(icon.rotation_degrees),
        "tint_rgb": [int(value) for value in icon.tint_rgb],
        "fill_style": str(icon.fill_style),
        "noise_edits": [dict(edit) for edit in icon.noise_edits],
        "noise_seed": int(icon.noise_seed) if icon.noise_seed is not None else None,
        "is_target_shape": bool(icon.is_target_shape),
        "is_counted": bool(icon.is_counted),
    }
