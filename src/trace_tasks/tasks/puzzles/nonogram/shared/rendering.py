"""Rendering helpers for nonogram puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.scene_style import PuzzleSceneStyle
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import NonogramRenderParams
from .rules import format_clue

_FILLED_RGB = (36, 42, 52)
_EMPTY_RGB = (252, 252, 252)
_UNKNOWN_RGB = (238, 242, 248)
_GRID_RGB = (76, 84, 98)
_ACCENT_RGB = (46, 111, 173)
_TEXT_RGB = (28, 32, 38)
_PANEL_RGB = (248, 249, 252)
_OPTION_SIDE_MARGIN_PX = 32.0
_OPTION_PANEL_X_PADDING_PX = 28.0
_OPTION_PANEL_LABEL_AREA_PX = 42.0
_OPTION_PANEL_BOTTOM_PADDING_PX = 10.0
_OPTION_TO_GRID_GAP_PX = 56.0
_OPTION_BOTTOM_MARGIN_PX = 38.0


@dataclass(frozen=True)
class RenderedNonogramScene:
    """Rendered nonogram image and projection maps."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    cell_bbox_map: Dict[str, List[float]]
    clue_bbox_map: Dict[str, List[float]]
    option_panel_bbox_map: Dict[str, List[float]]
    item_bbox_map: Dict[str, List[float]]


def _bbox(left: float, top: float, right: float, bottom: float) -> List[float]:
    return [
        round(float(left), 3),
        round(float(top), 3),
        round(float(right), 3),
        round(float(bottom), 3),
    ]


def _variant_palette(
    scene_variant: str,
    *,
    scene_style: PuzzleSceneStyle | None,
) -> Dict[str, Tuple[int, int, int]]:
    """Resolve palette colors while preserving filled, empty, and clue contrast."""

    if str(scene_variant) == "nonogram_card":
        palette = {
            "panel": (246, 250, 248),
            "cell": (253, 253, 250),
            "grid": (71, 100, 92),
            "accent": (46, 128, 109),
            "filled": (34, 54, 50),
            "unknown": (237, 246, 242),
            "text": _TEXT_RGB,
            "text_stroke": (255, 255, 255),
        }
    elif str(scene_variant) == "nonogram_blueprint":
        palette = {
            "panel": (244, 248, 253),
            "cell": (251, 253, 255),
            "grid": (67, 89, 126),
            "accent": (63, 103, 178),
            "filled": (32, 45, 76),
            "unknown": (235, 241, 250),
            "text": _TEXT_RGB,
            "text_stroke": (255, 255, 255),
        }
    else:
        palette = {
            "panel": _PANEL_RGB,
            "cell": _EMPTY_RGB,
            "grid": _GRID_RGB,
            "accent": _ACCENT_RGB,
            "filled": _FILLED_RGB,
            "unknown": _UNKNOWN_RGB,
            "text": _TEXT_RGB,
            "text_stroke": (255, 255, 255),
        }
    if scene_style is None:
        return palette
    return {
        **palette,
        "panel": tuple(scene_style.panel_fill_rgb),
        "cell": tuple(scene_style.option_fill_rgb),
        "grid": tuple(scene_style.grid_rgb),
        "accent": tuple(scene_style.mark_rgb),
        "filled": tuple(scene_style.mark_rgb),
        "unknown": tuple(scene_style.step_fill_rgb),
        "text": tuple(scene_style.text_rgb),
        "text_stroke": tuple(scene_style.text_stroke_rgb),
    }


def _draw_cell(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    value: int | None,
    palette: Mapping[str, Tuple[int, int, int]],
    line_width: int,
    show_empty_marks: bool,
) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    if value is None:
        fill = tuple(palette["unknown"])
    elif int(value) == 1:
        fill = tuple(palette["filled"])
    else:
        fill = tuple(palette["cell"])
    draw.rectangle(
        (left, top, right, bottom),
        fill=fill,
        outline=tuple(palette["grid"]),
        width=max(1, int(line_width)),
    )
    if value is not None and int(value) == 0 and bool(show_empty_marks):
        pad = max(4.0, 0.25 * float(right - left))
        draw.line(
            (left + pad, top + pad, right - pad, bottom - pad),
            fill=tuple(palette["grid"]),
            width=max(1, int(line_width)),
        )
        draw.line(
            (right - pad, top + pad, left + pad, bottom - pad),
            fill=tuple(palette["grid"]),
            width=max(1, int(line_width)),
        )


def _draw_row_clue(
    draw: ImageDraw.ImageDraw,
    *,
    clue: Sequence[int],
    bbox: Sequence[float],
    palette: Mapping[str, Tuple[int, int, int]],
    font,
) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    draw_centered_text(
        draw,
        text=format_clue(clue),
        center=((left + right) / 2.0, (top + bottom) / 2.0),
        font=font,
        fill=tuple(palette["text"]),
        stroke_fill=tuple(palette["text_stroke"]),
        stroke_width=1,
    )


def _draw_col_clue(
    draw: ImageDraw.ImageDraw,
    *,
    clue: Sequence[int],
    bbox: Sequence[float],
    palette: Mapping[str, Tuple[int, int, int]],
    font,
) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    clue_values = [int(value) for value in clue]
    spacing = min(
        24.0,
        max(17.0, float(bottom - top - 16.0) / max(1, len(clue_values))),
    )
    base_y = float(bottom - 14.0 - ((len(clue_values) - 1) * spacing))
    for index, value in enumerate(clue_values):
        draw_centered_text(
            draw,
            text=str(int(value)),
            center=((left + right) / 2.0, base_y + (float(index) * spacing)),
            font=font,
            fill=tuple(palette["text"]),
            stroke_fill=tuple(palette["text_stroke"]),
            stroke_width=1,
        )


def _draw_main_nonogram(
    draw: ImageDraw.ImageDraw,
    *,
    grid: Sequence[Sequence[int | None]],
    row_clues: Sequence[Sequence[int]],
    col_clues: Sequence[Sequence[int]],
    params: NonogramRenderParams,
    palette: Mapping[str, Tuple[int, int, int]],
    marked_axis: str | None,
    marked_index: int | None,
    show_empty_marks: bool,
) -> tuple[Dict[str, List[float]], Dict[str, List[float]], List[float], List[float] | None]:
    """Draw the clue rails and main grid, returning projection maps."""

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    grid_width = int(cols) * int(params.cell_size_px)
    grid_height = int(rows) * int(params.cell_size_px)
    total_width = int(params.left_clue_width_px) + int(grid_width)
    grid_left = round(
        (float(params.canvas_width) - float(total_width)) / 2.0
        + float(params.left_clue_width_px),
        3,
    )
    grid_top = float(params.margin_top_px + params.top_clue_height_px)
    left_rail_left = float(grid_left - params.left_clue_width_px)
    top_rail_top = float(params.margin_top_px)
    cell_size = float(params.cell_size_px)
    clue_font = load_font(int(params.clue_font_size_px), bold=True)

    scene_bbox = _bbox(
        left_rail_left,
        top_rail_top,
        grid_left + grid_width,
        grid_top + grid_height,
    )
    draw.rounded_rectangle(
        tuple(scene_bbox),
        radius=int(params.panel_corner_radius_px),
        fill=tuple(palette["panel"]),
        outline=tuple(palette["grid"]),
        width=max(1, int(params.grid_line_width_px)),
    )

    cell_bboxes: Dict[str, List[float]] = {}
    clue_bboxes: Dict[str, List[float]] = {
        "row_clue_panel": _bbox(left_rail_left, grid_top, grid_left, grid_top + grid_height),
        "col_clue_panel": _bbox(grid_left, top_rail_top, grid_left + grid_width, grid_top),
    }

    for row_index, clue in enumerate(row_clues):
        top = float(grid_top + (row_index * cell_size))
        clue_id = f"row_clue_{row_index}"
        clue_bbox = _bbox(left_rail_left, top, grid_left, top + cell_size)
        clue_bboxes[clue_id] = clue_bbox
        _draw_row_clue(
            draw,
            clue=clue,
            bbox=clue_bbox,
            palette=palette,
            font=clue_font,
        )

    for col_index, clue in enumerate(col_clues):
        left = float(grid_left + (col_index * cell_size))
        clue_id = f"col_clue_{col_index}"
        clue_bbox = _bbox(left, top_rail_top, left + cell_size, grid_top)
        clue_bboxes[clue_id] = clue_bbox
        _draw_col_clue(
            draw,
            clue=clue,
            bbox=clue_bbox,
            palette=palette,
            font=clue_font,
        )

    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            left = float(grid_left + (col_index * cell_size))
            top = float(grid_top + (row_index * cell_size))
            cell_id = f"cell_{row_index}_{col_index}"
            bbox = _bbox(left, top, left + cell_size, top + cell_size)
            cell_bboxes[cell_id] = bbox
            _draw_cell(
                draw,
                bbox=bbox,
                value=value,
                palette=palette,
                line_width=int(params.grid_line_width_px),
                show_empty_marks=bool(show_empty_marks),
            )

    for offset in range(0, int(cols) + 1):
        width = (
            int(params.heavy_line_width_px)
            if offset % 5 == 0
            else int(params.grid_line_width_px)
        )
        x = float(grid_left + (offset * cell_size))
        draw.line(
            (x, grid_top, x, grid_top + grid_height),
            fill=tuple(palette["grid"]),
            width=max(1, width),
        )
    for offset in range(0, int(rows) + 1):
        width = (
            int(params.heavy_line_width_px)
            if offset % 5 == 0
            else int(params.grid_line_width_px)
        )
        y = float(grid_top + (offset * cell_size))
        draw.line(
            (grid_left, y, grid_left + grid_width, y),
            fill=tuple(palette["grid"]),
            width=max(1, width),
        )

    line_bbox: List[float] | None = None
    if marked_axis is not None and marked_index is not None:
        top = float(grid_top + (int(marked_index) * cell_size))
        line_bbox = _bbox(grid_left, top, grid_left + grid_width, top + cell_size)
        draw.rectangle(
            tuple(line_bbox),
            outline=tuple(palette["accent"]),
            width=max(3, int(params.heavy_line_width_px)),
        )
        clue_id = f"row_clue_{int(marked_index)}"
        if clue_id in clue_bboxes:
            draw.rectangle(
                tuple(clue_bboxes[clue_id]),
                outline=tuple(palette["accent"]),
                width=max(3, int(params.heavy_line_width_px)),
            )

    return cell_bboxes, clue_bboxes, scene_bbox, line_bbox


def _draw_option_label(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    panel_bbox: Sequence[float],
    params: NonogramRenderParams,
    palette: Mapping[str, Tuple[int, int, int]],
) -> None:
    left, top, right, _bottom = [float(value) for value in panel_bbox]
    font = load_font(int(params.option_label_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text=str(label),
        center=((left + right) / 2.0, top + 18.0),
        font=font,
        fill=tuple(palette["text"]),
        stroke_fill=tuple(palette["text_stroke"]),
        stroke_width=1,
    )


def _option_panel_bboxes(
    option_count: int,
    params: NonogramRenderParams,
    *,
    panel_width: float,
    panel_height: float,
    option_top: float,
) -> Dict[str, List[float]]:
    total_width = (int(option_count) * float(panel_width)) + (
        max(0, int(option_count) - 1) * int(params.option_gap_px)
    )
    start_x = round((float(params.canvas_width) - float(total_width)) / 2.0, 3)
    boxes: Dict[str, List[float]] = {}
    for index in range(int(option_count)):
        left = float(start_x + (index * (float(panel_width) + int(params.option_gap_px))))
        top = float(option_top)
        boxes[str(index)] = _bbox(
            left,
            top,
            left + float(panel_width),
            top + float(panel_height),
        )
    return boxes


def _option_panel_size(
    *,
    mode: str,
    rows: int,
    cols: int,
    params: NonogramRenderParams,
) -> tuple[float, float]:
    """Return a panel size that can contain option cells at main-grid scale."""

    cell_size = float(params.cell_size_px)
    panel_width = max(
        float(params.option_panel_width_px),
        (float(cols) * cell_size) + _OPTION_PANEL_X_PADDING_PX,
    )
    option_rows = int(rows) if str(mode) == "candidate_solution" else 1
    panel_height = max(
        float(params.option_panel_height_px),
        _OPTION_PANEL_LABEL_AREA_PX
        + (float(option_rows) * cell_size)
        + _OPTION_PANEL_BOTTOM_PADDING_PX,
    )
    return float(panel_width), float(panel_height)


def _option_top(
    *,
    scene_bbox: Sequence[float],
    panel_height: float,
    params: NonogramRenderParams,
) -> float:
    """Place option panels close to the clue grid without clipping the canvas."""

    latest_top = float(params.canvas_height) - float(panel_height) - _OPTION_BOTTOM_MARGIN_PX
    min_top = float(scene_bbox[3]) + _OPTION_TO_GRID_GAP_PX
    if float(min_top) <= float(latest_top):
        return float(min_top)
    return float(latest_top)


def _resolve_layout_params(
    *,
    mode: str,
    rows: int,
    cols: int,
    option_count: int,
    params: NonogramRenderParams,
) -> NonogramRenderParams:
    """Use one cell size for the main grid and MCQ cells while preserving fit."""

    desired = float(params.cell_size_px)
    width_cap = desired
    if int(option_count) > 0:
        available_width = (
            float(params.canvas_width)
            - (2.0 * _OPTION_SIDE_MARGIN_PX)
            - (max(0, int(option_count) - 1) * float(params.option_gap_px))
        )
        width_cap = (
            (available_width / float(option_count)) - _OPTION_PANEL_X_PADDING_PX
        ) / max(1.0, float(cols))

    main_static_height = float(params.margin_top_px + params.top_clue_height_px)
    if int(option_count) <= 0:
        height_cap = desired
    elif str(mode) == "candidate_solution":
        option_static_height = _OPTION_PANEL_LABEL_AREA_PX + _OPTION_PANEL_BOTTOM_PADDING_PX
        height_cap = (
            float(params.canvas_height)
            - main_static_height
            - _OPTION_TO_GRID_GAP_PX
            - _OPTION_BOTTOM_MARGIN_PX
            - option_static_height
        ) / max(1.0, float(rows * 2))
    else:
        variable_height_cap = (
            float(params.canvas_height)
            - main_static_height
            - _OPTION_TO_GRID_GAP_PX
            - _OPTION_BOTTOM_MARGIN_PX
            - _OPTION_PANEL_LABEL_AREA_PX
            - _OPTION_PANEL_BOTTOM_PADDING_PX
        ) / max(1.0, float(rows + 1))
        fixed_panel_cap = (
            float(params.canvas_height)
            - main_static_height
            - _OPTION_TO_GRID_GAP_PX
            - _OPTION_BOTTOM_MARGIN_PX
            - float(params.option_panel_height_px)
        ) / max(1.0, float(rows))
        height_cap = min(float(variable_height_cap), float(fixed_panel_cap))

    resolved_cell_size = max(18, int(min(desired, width_cap, height_cap)))
    if int(resolved_cell_size) == int(params.cell_size_px):
        return params
    return replace(params, cell_size_px=int(resolved_cell_size))


def _draw_line_option(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    label: str,
    line: Sequence[int],
    palette: Mapping[str, Tuple[int, int, int]],
    params: NonogramRenderParams,
) -> None:
    """Draw one row-strip option panel; the whole panel is the annotation target."""

    left, top, right, bottom = [float(value) for value in panel_bbox]
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=int(params.panel_corner_radius_px),
        fill=tuple(palette["panel"]),
        outline=tuple(palette["grid"]),
        width=max(1, int(params.grid_line_width_px)),
    )
    _draw_option_label(
        draw,
        label=str(label),
        panel_bbox=panel_bbox,
        params=params,
        palette=palette,
    )
    grid_area_left = float(left + 14.0)
    grid_area_right = float(right - 14.0)
    grid_area_top = float(top + _OPTION_PANEL_LABEL_AREA_PX)
    grid_area_bottom = float(bottom - _OPTION_PANEL_BOTTOM_PADDING_PX)
    available_width = max(1.0, float(grid_area_right - grid_area_left))
    available_height = max(1.0, float(grid_area_bottom - grid_area_top))
    cell_size = min(
        float(params.cell_size_px),
        available_width / max(1, len(line)),
        available_height,
    )
    strip_width = float(len(line)) * float(cell_size)
    strip_left = float(grid_area_left + ((available_width - strip_width) / 2.0))
    strip_top = float(grid_area_top + ((available_height - cell_size) / 2.0))
    for index, value in enumerate(line):
        cell_bbox = _bbox(
            strip_left + (index * cell_size),
            strip_top,
            strip_left + ((index + 1) * cell_size),
            strip_top + cell_size,
        )
        _draw_cell(
            draw,
            bbox=cell_bbox,
            value=int(value),
            palette=palette,
            line_width=1,
            show_empty_marks=False,
        )


def _draw_candidate_option(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    label: str,
    grid: Sequence[Sequence[int]],
    palette: Mapping[str, Tuple[int, int, int]],
    params: NonogramRenderParams,
) -> None:
    """Draw one full-grid option panel; mini cells are not separate annotations."""

    left, top, right, bottom = [float(value) for value in panel_bbox]
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=int(params.panel_corner_radius_px),
        fill=tuple(palette["panel"]),
        outline=tuple(palette["grid"]),
        width=max(1, int(params.grid_line_width_px)),
    )
    _draw_option_label(
        draw,
        label=str(label),
        panel_bbox=panel_bbox,
        params=params,
        palette=palette,
    )
    grid_area_left = float(left + 14.0)
    grid_area_right = float(right - 14.0)
    grid_area_top = float(top + 42.0)
    grid_area_bottom = float(bottom - 10.0)
    available_width = max(1.0, float(grid_area_right - grid_area_left))
    available_height = max(1.0, float(grid_area_bottom - grid_area_top))
    cell_size = min(
        float(params.cell_size_px),
        available_width / max(1, cols),
        available_height / max(1, rows),
    )
    grid_width = float(cols) * float(cell_size)
    grid_height = float(rows) * float(cell_size)
    grid_left = float(grid_area_left + ((available_width - grid_width) / 2.0))
    grid_top = float(grid_area_top + ((available_height - grid_height) / 2.0))
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            cell_bbox = _bbox(
                grid_left + (col_index * cell_size),
                grid_top + (row_index * cell_size),
                grid_left + ((col_index + 1) * cell_size),
                grid_top + ((row_index + 1) * cell_size),
            )
            _draw_cell(
                draw,
                bbox=cell_bbox,
                value=int(value),
                palette=palette,
                line_width=1,
                show_empty_marks=False,
            )


def render_nonogram_scene(
    image: Image.Image,
    *,
    scene_variant: str,
    mode: str,
    display_grid: Sequence[Sequence[int | None]],
    row_clues: Sequence[Sequence[int]],
    col_clues: Sequence[Sequence[int]],
    render_params: NonogramRenderParams,
    marked_axis: str | None = None,
    marked_index: int | None = None,
    option_specs: Sequence[Mapping[str, Any]] | None = None,
    show_empty_marks: bool = False,
    scene_style: PuzzleSceneStyle | None = None,
) -> RenderedNonogramScene:
    """Render one nonogram clue grid with optional visual answer choices."""

    if not display_grid or not display_grid[0]:
        raise ValueError("nonogram scene requires a non-empty grid")
    draw = ImageDraw.Draw(image)
    option_count = len(list(option_specs or []))
    render_params = _resolve_layout_params(
        mode=str(mode),
        rows=int(len(display_grid)),
        cols=int(len(display_grid[0])),
        option_count=int(option_count),
        params=render_params,
    )
    palette = _variant_palette(str(scene_variant), scene_style=scene_style)
    cell_bboxes, clue_bboxes, scene_bbox, line_bbox = _draw_main_nonogram(
        draw,
        grid=display_grid,
        row_clues=row_clues,
        col_clues=col_clues,
        params=render_params,
        palette=palette,
        marked_axis=marked_axis,
        marked_index=marked_index,
        show_empty_marks=bool(show_empty_marks),
    )

    option_panel_bbox_map: Dict[str, List[float]] = {}
    item_bbox_map: Dict[str, List[float]] = {
        "row_clue_panel": list(clue_bboxes["row_clue_panel"]),
        "col_clue_panel": list(clue_bboxes["col_clue_panel"]),
    }
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "nonogram",
            "entity_type": "nonogram",
            "bbox_px": list(scene_bbox),
            "metadata": {
                "rows": int(len(display_grid)),
                "cols": int(len(display_grid[0])),
                "scene_variant": str(scene_variant),
            },
        },
        {
            "entity_id": "row_clue_panel",
            "entity_type": "nonogram_clue_panel",
            "bbox_px": list(clue_bboxes["row_clue_panel"]),
            "metadata": {"axis": "row"},
        },
        {
            "entity_id": "col_clue_panel",
            "entity_type": "nonogram_clue_panel",
            "bbox_px": list(clue_bboxes["col_clue_panel"]),
            "metadata": {"axis": "column"},
        },
    ]
    for clue_id, bbox in clue_bboxes.items():
        if str(clue_id).endswith("_panel"):
            continue
        item_bbox_map[str(clue_id)] = list(bbox)
        entities.append(
            {
                "entity_id": str(clue_id),
                "entity_type": "nonogram_clue",
                "bbox_px": list(bbox),
                "metadata": {
                    "axis": "row" if str(clue_id).startswith("row_") else "column",
                },
            }
        )
    for cell_id, bbox in cell_bboxes.items():
        item_bbox_map[str(cell_id)] = list(bbox)
        entities.append(
            {
                "entity_id": str(cell_id),
                "entity_type": "nonogram_cell",
                "bbox_px": list(bbox),
                "metadata": {},
            }
        )
    if line_bbox is not None:
        item_bbox_map["marked_line"] = list(line_bbox)
        entities.append(
            {
                "entity_id": "marked_line",
                "entity_type": "nonogram_marked_line",
                "bbox_px": list(line_bbox),
                "metadata": {"axis": str(marked_axis), "index": int(marked_index or 0)},
            }
        )

    options = list(option_specs or [])
    if options:
        panel_width, panel_height = _option_panel_size(
            mode=str(mode),
            rows=int(len(display_grid)),
            cols=int(len(display_grid[0])),
            params=render_params,
        )
        option_top = _option_top(
            scene_bbox=scene_bbox,
            panel_height=float(panel_height),
            params=render_params,
        )
        indexed_bboxes = _option_panel_bboxes(
            len(options),
            render_params,
            panel_width=float(panel_width),
            panel_height=float(panel_height),
            option_top=float(option_top),
        )
        for option_index, option in enumerate(options):
            label = str(option.get("option_label", ""))
            panel_id = str(option.get("option_panel_id", f"option_{label}"))
            panel_bbox = indexed_bboxes[str(option_index)]
            option_panel_bbox_map[panel_id] = list(panel_bbox)
            item_bbox_map[panel_id] = list(panel_bbox)
            if str(mode) == "line_completion":
                _draw_line_option(
                    draw,
                    panel_bbox=panel_bbox,
                    label=label,
                    line=[int(value) for value in option.get("line", [])],
                    palette=palette,
                    params=render_params,
                )
            elif str(mode) == "candidate_solution":
                _draw_candidate_option(
                    draw,
                    panel_bbox=panel_bbox,
                    label=label,
                    grid=[
                        [int(value) for value in row]
                        for row in option.get("grid", [])
                    ],
                    palette=palette,
                    params=render_params,
                )
            else:
                raise ValueError(f"unsupported nonogram option mode: {mode}")
            entities.append(
                {
                    "entity_id": panel_id,
                    "entity_type": "nonogram_option_panel",
                    "bbox_px": list(panel_bbox),
                    "metadata": {
                        "option_label": label,
                        "is_correct": bool(option.get("is_correct", False)),
                    },
                }
            )

    return RenderedNonogramScene(
        image=image,
        entities=entities,
        scene_bbox_px=list(scene_bbox),
        cell_bbox_map=dict(cell_bboxes),
        clue_bbox_map=dict(clue_bboxes),
        option_panel_bbox_map=dict(option_panel_bbox_map),
        item_bbox_map=dict(item_bbox_map),
    )


__all__ = ["RenderedNonogramScene", "render_nonogram_scene"]
