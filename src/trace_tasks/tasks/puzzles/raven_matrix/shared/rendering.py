"""Renderer for Raven-style 3 by 3 matrix option puzzles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.puzzles.shared.symbol_rendering import draw_puzzle_shape_icon
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import RavenRenderParams, RenderedRavenScene, SUPPORTED_SCENE_VARIANTS


def _rgb(value: Sequence[int]) -> tuple[int, int, int]:
    """Normalize one RGB sequence."""

    return tuple(int(component) for component in value[:3])


def _matrix_content_pad(cell_size: float) -> float:
    """Return the matrix-cell inset used before Raven panel content is drawn."""

    return float(max(8.0, 0.08 * float(cell_size)))


def _centered_square(
    bbox: tuple[float, float, float, float] | list[float],
    side: float,
) -> tuple[float, float, float, float]:
    """Return a square with the requested side centered inside a bbox."""

    left, top, right, bottom = [float(value) for value in bbox]
    resolved_side = float(min(float(side), right - left, bottom - top))
    cx = float(0.5 * (left + right))
    cy = float(0.5 * (top + bottom))
    half = float(0.5 * resolved_side)
    return (cx - half, cy - half, cx + half, cy + half)


def _draw_attribute_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[float, float, float, float],
    panel_spec: Mapping[str, Any],
    outline_rgb: Sequence[int],
    border_width_px: int,
) -> None:
    """Draw one shape/color/size attribute panel."""

    left, top, right, bottom = [float(value) for value in bbox]
    size_scale = max(0.32, min(0.90, float(panel_spec.get("size_scale", 0.78))))
    side = min(right - left, bottom - top)
    inset = 0.5 * side * (1.0 - size_scale)
    draw_puzzle_shape_icon(
        draw,
        bbox=(left + inset, top + inset, right - inset, bottom - inset),
        object_type=str(panel_spec["object_type"]),
        fill_rgb=_rgb(panel_spec["fill_rgb"]),
        outline_rgb=_rgb(outline_rgb),
        width=max(2, int(border_width_px)),
        inset_px=max(2.0, 0.05 * side),
    )


def _draw_raven_option_cell(
    draw: ImageDraw.ImageDraw,
    *,
    slot_bbox: tuple[float, float, float, float],
    option_label: str,
    label_font,
    label_center_y_px: float,
    cell_side_px: float,
    label_gap_px: float,
    cell_fill_rgb: Sequence[int],
    border_color_rgb: Sequence[int],
    text_color_rgb: Sequence[int],
    text_stroke_rgb: Sequence[int],
    cell_corner_radius_px: int,
    border_width_px: int,
) -> tuple[list[float], list[float]]:
    """Draw one Raven option label and candidate cell without card chrome."""

    left, top, right, bottom = [float(value) for value in slot_bbox]
    slot_width = float(right - left)
    label_bbox = draw_centered_text(
        draw,
        text=str(option_label),
        center=(float(left + 0.5 * slot_width), float(label_center_y_px)),
        font=label_font,
        fill=text_color_rgb,
        stroke_fill=text_stroke_rgb,
        stroke_width=1,
    )
    cell_top_min = float(label_bbox[3] + float(label_gap_px))
    available_height = float(bottom - cell_top_min)
    side = float(min(float(cell_side_px), slot_width, max(1.0, available_height)))
    cell_left = float(left + 0.5 * (slot_width - side))
    cell_top = float(cell_top_min + max(0.0, 0.5 * (available_height - side)))
    cell_bbox = (
        float(cell_left),
        float(cell_top),
        float(cell_left + side),
        float(cell_top + side),
    )
    draw_rounded_rect(
        draw,
        cell_bbox,
        radius=int(cell_corner_radius_px),
        fill=cell_fill_rgb,
        outline=border_color_rgb,
        width=int(border_width_px),
    )
    return (
        [round(float(value), 3) for value in label_bbox],
        [round(float(value), 3) for value in cell_bbox],
    )


def _count_cells(count: int) -> tuple[tuple[int, int], ...]:
    """Return a stable filled-cell pattern for a visible count."""

    support = (
        (1, 1),
        (0, 0),
        (2, 2),
        (0, 2),
        (2, 0),
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    )
    return tuple(support[: max(0, min(9, int(count)))])


def _draw_count_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[float, float, float, float],
    panel_spec: Mapping[str, Any],
    outline_rgb: Sequence[int],
    border_width_px: int,
) -> None:
    """Draw one count as filled cells in a Raven mini-grid."""

    _draw_filled_cell_grid(
        draw,
        bbox=bbox,
        grid_size=3,
        selected_cells=_count_cells(int(panel_spec["count"])),
        fill_rgb=panel_spec["fill_rgb"],
        outline_rgb=outline_rgb,
        border_width_px=int(border_width_px),
    )


def _draw_filled_cell_grid(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[float, float, float, float],
    grid_size: int,
    selected_cells: Sequence[Sequence[int]],
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    border_width_px: int,
) -> None:
    """Draw a nested Raven mini-grid with selected cells filled."""

    left, top, right, bottom = [float(value) for value in bbox]
    if grid_size < 2:
        raise ValueError("Raven pattern panels require grid_size >= 2")
    pad = float(0.14 * min(right - left, bottom - top))
    grid_left = float(left + pad)
    grid_top = float(top + pad)
    grid_right = float(right - pad)
    grid_bottom = float(bottom - pad)
    cell_size = float(min(grid_right - grid_left, grid_bottom - grid_top) / grid_size)
    grid_width = float(cell_size * grid_size)
    grid_left = float(0.5 * (left + right - grid_width))
    grid_top = float(0.5 * (top + bottom - grid_width))
    fill = _rgb(fill_rgb)
    outline = _rgb(outline_rgb)
    selected = {(int(row), int(col)) for row, col in selected_cells}
    for row_index in range(grid_size):
        for col_index in range(grid_size):
            x0 = float(grid_left + col_index * cell_size)
            y0 = float(grid_top + row_index * cell_size)
            x1 = float(x0 + cell_size)
            y1 = float(y0 + cell_size)
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=fill if (row_index, col_index) in selected else (255, 255, 255),
                outline=outline,
                width=max(1, int(border_width_px) - 1),
            )


def _draw_pattern_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[float, float, float, float],
    panel_spec: Mapping[str, Any],
    outline_rgb: Sequence[int],
    border_width_px: int,
) -> None:
    """Draw one filled-cell spatial pattern panel."""

    _draw_filled_cell_grid(
        draw,
        bbox=bbox,
        grid_size=int(panel_spec.get("grid_size", 3)),
        selected_cells=panel_spec["cells"],
        fill_rgb=panel_spec["fill_rgb"],
        outline_rgb=outline_rgb,
        border_width_px=int(border_width_px),
    )


def draw_raven_panel_content(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[float, float, float, float],
    panel_spec: Mapping[str, Any],
    outline_rgb: Sequence[int],
    border_width_px: int,
) -> None:
    """Draw a Raven mini-panel from its symbolic panel spec."""

    panel_kind = str(panel_spec["panel_kind"])
    if panel_kind == "attribute":
        _draw_attribute_panel(
            draw,
            bbox=bbox,
            panel_spec=panel_spec,
            outline_rgb=outline_rgb,
            border_width_px=int(border_width_px),
        )
        return
    if panel_kind == "count":
        _draw_count_panel(
            draw,
            bbox=bbox,
            panel_spec=panel_spec,
            outline_rgb=outline_rgb,
            border_width_px=int(border_width_px),
        )
        return
    if panel_kind == "pattern":
        _draw_pattern_panel(
            draw,
            bbox=bbox,
            panel_spec=panel_spec,
            outline_rgb=outline_rgb,
            border_width_px=int(border_width_px),
        )
        return
    raise ValueError(f"unsupported Raven panel_kind: {panel_kind}")


def render_raven_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    matrix_rows: Sequence[Sequence[Mapping[str, Any]]],
    option_specs: Sequence[Mapping[str, Any]],
    render_params: RavenRenderParams,
) -> RenderedRavenScene:
    """Render one Raven-style 3 by 3 matrix and labeled image options."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(SUPPORTED_SCENE_VARIANTS):
        raise ValueError(f"unsupported Raven scene_variant: {scene_variant}")
    rows = [list(row) for row in matrix_rows]
    options = [dict(option) for option in option_specs]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Raven scenes require a 3 by 3 matrix")
    if len(options) < 2:
        raise ValueError("Raven scenes require at least two option panels")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    value_font = load_font(int(render_params.value_font_size_px), bold=True)
    option_label_font = load_font(
        int(render_params.option_label_font_size_px),
        bold=True,
    )

    cell_size = float(render_params.cell_size_px)
    cell_gap = float(render_params.cell_gap_px)
    matrix_width = float((3 * cell_size) + 2 * cell_gap)
    matrix_height = float(matrix_width)
    board_panel_pad = float(render_params.board_panel_padding_px)
    option_panel_width = float(render_params.option_panel_width_px)
    option_panel_height = float(render_params.option_panel_height_px)
    option_gap = float(render_params.option_gap_px)
    options_width = float(
        (len(options) * option_panel_width) + max(0, len(options) - 1) * option_gap
    )
    options_height = float(option_panel_height)
    board_to_options_gap = float(render_params.board_to_options_gap_px)

    content_width = float(max(matrix_width, options_width))
    content_height = float(matrix_height + board_to_options_gap + options_height)
    usable_width = float(
        render_params.canvas_width
        - render_params.scene_margin_left_px
        - render_params.scene_margin_right_px
    )
    usable_height = float(
        render_params.canvas_height
        - render_params.scene_margin_top_px
        - render_params.scene_margin_bottom_px
    )
    content_left = float(
        render_params.scene_margin_left_px
        + max(0.0, 0.5 * (usable_width - content_width))
    )
    content_top = float(
        render_params.scene_margin_top_px
        + max(0.0, 0.5 * (usable_height - content_height))
    )

    matrix_left = float(content_left + 0.5 * (content_width - matrix_width))
    matrix_top = float(content_top)
    options_left = float(content_left + 0.5 * (content_width - options_width))
    options_top = float(matrix_top + matrix_height + board_to_options_gap)

    matrix_panel_bbox = (
        float(matrix_left - board_panel_pad),
        float(matrix_top - board_panel_pad),
        float(matrix_left + matrix_width + board_panel_pad),
        float(matrix_top + matrix_height + board_panel_pad),
    )
    options_panel_bbox = (
        float(options_left - board_panel_pad),
        float(options_top - board_panel_pad),
        float(options_left + options_width + board_panel_pad),
        float(options_top + options_height + board_panel_pad),
    )

    entities: list[dict[str, Any]] = []
    matrix_cell_bbox_map: dict[str, list[float]] = {}
    option_panel_bbox_map: dict[str, list[float]] = {}
    option_cell_bbox_map: dict[str, list[float]] = {}

    for row_index, row in enumerate(rows):
        for col_index, cell in enumerate(row):
            cell_left = float(matrix_left + col_index * (cell_size + cell_gap))
            cell_top = float(matrix_top + row_index * (cell_size + cell_gap))
            cell_bbox = (
                float(cell_left),
                float(cell_top),
                float(cell_left + cell_size),
                float(cell_top + cell_size),
            )
            cell_id = str(cell["cell_id"])
            is_unknown = bool(cell.get("is_unknown", False))
            draw_rounded_rect(
                draw,
                cell_bbox,
                radius=int(render_params.slot_corner_radius_px),
                fill=(
                    render_params.unknown_cell_fill_rgb
                    if is_unknown
                    else render_params.cell_fill_rgb
                ),
                outline=render_params.border_color_rgb,
                width=int(render_params.border_width_px),
            )
            if is_unknown:
                draw_centered_text(
                    draw,
                    text="?",
                    center=(
                        float(cell_left + 0.5 * cell_size),
                        float(cell_top + 0.5 * cell_size),
                    ),
                    font=value_font,
                    fill=render_params.accent_color_rgb,
                    stroke_fill=render_params.text_stroke_rgb,
                    stroke_width=1,
                )
            else:
                content_pad = _matrix_content_pad(cell_size)
                draw_raven_panel_content(
                    draw,
                    bbox=(
                        float(cell_bbox[0] + content_pad),
                        float(cell_bbox[1] + content_pad),
                        float(cell_bbox[2] - content_pad),
                        float(cell_bbox[3] - content_pad),
                    ),
                    panel_spec=cell["panel_spec"],
                    outline_rgb=render_params.border_color_rgb,
                    border_width_px=int(render_params.border_width_px),
                )
            bbox_list = [round(float(value), 3) for value in cell_bbox]
            matrix_cell_bbox_map[cell_id] = list(bbox_list)
            entities.append(
                {
                    "entity_id": cell_id,
                    "entity_type": "puzzle_raven_matrix_cell",
                    "bbox_px": list(bbox_list),
                    "attrs": {
                        "row_index": int(row_index),
                        "col_index": int(col_index),
                        "is_unknown": bool(is_unknown),
                        "panel_kind": None
                        if is_unknown
                        else str(cell["panel_spec"]["panel_kind"]),
                    },
                }
            )

    matrix_content_side = float(cell_size - 2.0 * _matrix_content_pad(cell_size))
    symbol_box_size = float(
        max(
            float(render_params.option_symbol_box_size_px),
            matrix_content_side + 12.0,
        )
    )
    option_label_gap = float(render_params.option_label_gap_px)
    for option_index, option in enumerate(options):
        panel_left = float(options_left + option_index * (option_panel_width + option_gap))
        panel_top = float(options_top)
        panel_bbox = (
            float(panel_left),
            float(panel_top),
            float(panel_left + option_panel_width),
            float(panel_top + option_panel_height),
        )
        option_panel_id = str(option["option_panel_id"])
        label_bbox, cell_bbox = _draw_raven_option_cell(
            draw,
            slot_bbox=panel_bbox,
            option_label=str(option["option_label"]),
            label_font=option_label_font,
            label_center_y_px=float(panel_top + 28.0),
            cell_side_px=float(symbol_box_size),
            label_gap_px=float(option_label_gap),
            cell_fill_rgb=render_params.option_symbol_fill_rgb,
            border_color_rgb=render_params.border_color_rgb,
            text_color_rgb=render_params.text_color_rgb,
            text_stroke_rgb=render_params.text_stroke_rgb,
            cell_corner_radius_px=int(max(8, render_params.slot_corner_radius_px - 4)),
            border_width_px=int(render_params.border_width_px),
        )
        draw_raven_panel_content(
            draw,
            bbox=_centered_square(cell_bbox, matrix_content_side),
            panel_spec=option["panel_spec"],
            outline_rgb=render_params.border_color_rgb,
            border_width_px=int(render_params.border_width_px),
        )

        panel_bbox_list = [round(float(value), 3) for value in panel_bbox]
        cell_bbox_list = [round(float(value), 3) for value in cell_bbox]
        option_panel_bbox_map[option_panel_id] = list(panel_bbox_list)
        option_cell_bbox_map[option_panel_id] = list(cell_bbox_list)
        entities.append(
            {
                "entity_id": option_panel_id,
                "entity_type": "puzzle_raven_option_slot",
                "bbox_px": list(panel_bbox_list),
                "attrs": {
                    "option_index": int(option_index),
                    "option_label": str(option["option_label"]),
                    "panel_kind": str(option["panel_spec"]["panel_kind"]),
                    "is_correct": bool(option.get("is_correct", False)),
                },
            }
        )
        entities.append(
            {
                "entity_id": f"{option_panel_id}_label",
                "entity_type": "puzzle_raven_option_label",
                "bbox_px": list(label_bbox),
                "attrs": {
                    "option_index": int(option_index),
                    "option_label": str(option["option_label"]),
                },
            }
        )
        entities.append(
            {
                "entity_id": f"{option_panel_id}_cell",
                "entity_type": "puzzle_raven_option_cell",
                "bbox_px": list(cell_bbox_list),
                "attrs": {
                    "option_index": int(option_index),
                    "option_label": str(option["option_label"]),
                    "panel_kind": str(option["panel_spec"]["panel_kind"]),
                },
            }
        )

    scene_bbox = [
        round(float(min(matrix_panel_bbox[0], options_panel_bbox[0], matrix_left)), 3),
        round(float(min(matrix_panel_bbox[1], options_panel_bbox[1], matrix_top)), 3),
        round(
            float(max(matrix_panel_bbox[2], options_panel_bbox[2], options_left + options_width)),
            3,
        ),
        round(
            float(max(matrix_panel_bbox[3], options_panel_bbox[3], options_top + options_height)),
            3,
        ),
    ]
    return RenderedRavenScene(
        image=image,
        entities=entities,
        scene_bbox_px=scene_bbox,
        matrix_cell_bbox_map=matrix_cell_bbox_map,
        option_panel_bbox_map=option_panel_bbox_map,
        option_cell_bbox_map=option_cell_bbox_map,
    )


__all__ = ["draw_raven_panel_content", "render_raven_scene"]
