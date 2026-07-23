"""Sudoku-grid renderer for the puzzle scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.puzzles.shared.layout import apply_puzzle_layout_jitter_to_bbox
from trace_tasks.tasks.puzzles.shared.marking import (
    draw_semantic_bbox_marker,
    resolve_semantic_marker_style,
)
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.puzzles.shared.scene_style import (
    PuzzleSceneStyle,
    draw_puzzle_panel_chrome,
    make_puzzle_scene_background,
    puzzle_scene_style_metadata,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.shared.bbox_projection import bbox_union
from trace_tasks.tasks.shared.text_legibility import (
    draw_centered_readable_text,
    draw_text_traced,
    resolve_readable_text_style,
)
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box

from .rules import coord_to_cell_id, unit_coords
from .sampling import SudokuRenderParams
from .state import SIZE, Board, Coord
from .state import SudokuSample
from .styles import SudokuTheme, build_puzzle_sudoku_theme


@dataclass(frozen=True)
class SudokuCellSpec:
    """One rendered Sudoku cell."""

    cell_id: str
    row: int
    col: int
    value: int
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedSudokuScene:
    """Rendered Sudoku image plus trace-friendly cell geometry."""

    image: Image.Image
    cell_specs: Tuple[SudokuCellSpec, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class SudokuVisualArtifacts:
    """Rendered Sudoku image plus reusable scene-level render metadata."""

    image: Image.Image
    rendered_scene: RenderedSudokuScene
    panel_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def _cell_bbox(
    *,
    board_left: float,
    board_top: float,
    cell_size: float,
    row: int,
    col: int,
    padding_px: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Return the bbox for one Sudoku cell, with optional inset padding."""

    left = float(board_left + (int(col) * float(cell_size)) + float(padding_px))
    top = float(board_top + (int(row) * float(cell_size)) + float(padding_px))
    right = float(board_left + ((int(col) + 1) * float(cell_size)) - float(padding_px))
    bottom = float(board_top + ((int(row) + 1) * float(cell_size)) - float(padding_px))
    return (round(left, 3), round(top, 3), round(right, 3), round(bottom, 3))


def _draw_digit(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    digit: int,
    theme: SudokuTheme,
    font_size_px: int,
    conflict: bool,
    font_family: str = "",
) -> None:
    """Draw one centered Sudoku digit with traced text metadata."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    text = str(int(digit))
    font = fit_font_to_box(
        draw,
        text=text,
        max_width=float(width),
        max_height=float(height),
        bold=True,
        min_size_px=16,
        max_size_px=int(font_size_px),
        fill_ratio=0.72,
        font_family=str(font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    text_x = float(left + (0.5 * (width - text_w)) - float(text_bbox[0]))
    text_y = float(top + (0.5 * (height - text_h)) - float(text_bbox[1]))
    fill = theme.conflict_digit_rgb if bool(conflict) else theme.digit_rgb
    draw_text_traced(
        draw,
        (text_x, text_y),
        text,
        fill=tuple(int(v) for v in fill),
        font=font,
        role="readout",
        required=False,
    )


def _draw_option_label(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Tuple[float, float, float, float],
    label: str,
    theme: SudokuTheme,
    font_family: str = "",
    filled_cell: bool,
    instance_seed: int,
) -> Tuple[float, float, float, float]:
    """Draw an in-cell option label badge and return its bbox."""

    left, top, right, bottom = bbox_px
    width = float(right - left)
    height = float(bottom - top)
    badge_size = float(min(width, height) * (0.34 if bool(filled_cell) else 0.54))
    pad = float(min(width, height) * 0.08)
    if bool(filled_cell):
        badge_left = float(left + pad)
        badge_top = float(top + pad)
    else:
        badge_left = float(left + (0.5 * (width - badge_size)))
        badge_top = float(top + (0.5 * (height - badge_size)))
    badge_bbox = (
        badge_left,
        badge_top,
        float(badge_left + badge_size),
        float(badge_top + badge_size),
    )
    badge_fill_rgb = (250, 252, 255)
    badge_fill = (*badge_fill_rgb, 255)
    badge_outline = tuple(int(v) for v in theme.box_line_rgb) + (255,)
    draw.rounded_rectangle(
        badge_bbox,
        radius=max(4, int(round(badge_size * 0.22))),
        fill=badge_fill,
        outline=badge_outline,
        width=max(1, int(round(badge_size * 0.06))),
    )
    text = str(label)
    font = fit_font_to_box(
        draw,
        text=text,
        max_width=float(badge_size * 0.72),
        max_height=float(badge_size * 0.72),
        bold=True,
        min_size_px=12,
        max_size_px=max(14, int(round(badge_size * 0.62))),
        fill_ratio=0.86,
        font_family=str(font_family) or None,
    )
    style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"puzzles.sudoku.option_label.{str(label)}",
        role="option_label",
        surface_rgbs=(badge_fill_rgb,),
        preferred_rgbs=((10, 14, 22), (24, 31, 44)),
        required=True,
    )
    draw_centered_readable_text(
        draw,
        center=(
            float(badge_bbox[0] + (0.5 * badge_size)),
            float(badge_bbox[1] + (0.5 * badge_size)),
        ),
        text=text,
        font=font,
        style=style,
        stroke_width=max(1, int(round(badge_size * 0.045))),
        extra_metadata={"option_label": str(label), "badge_bbox_px": list(badge_bbox)},
    )
    return tuple(round(float(value), 3) for value in badge_bbox)


def render_sudoku_grid_scene(
    *,
    board: Board,
    background: Image.Image,
    style_variant: str,
    params: SudokuRenderParams,
    highlighted_unit_type: str | None = None,
    highlighted_unit_index: int | None = None,
    marked_cell: Coord | None = None,
    conflict_coords: Sequence[Coord] = (),
    option_specs: Sequence[Mapping[str, Any]] = (),
    panel_style: PuzzleSceneStyle | None = None,
) -> RenderedSudokuScene:
    """Render one Sudoku grid with optional highlighted unit and marked cell."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = build_puzzle_sudoku_theme(style_variant=str(style_variant))

    board_size_px = min(
        int(params.max_board_size_px),
        int(params.canvas_width) - (2 * int(params.panel_margin_px)),
        int(params.canvas_height) - (2 * int(params.panel_margin_px)),
    )
    board_left = int(0.5 * (int(params.canvas_width) - int(board_size_px)))
    board_top = int(0.5 * (int(params.canvas_height) - int(board_size_px)))
    board_bbox = (
        round(float(board_left), 3),
        round(float(board_top), 3),
        round(float(board_left + board_size_px), 3),
        round(float(board_top + board_size_px), 3),
    )
    board_bbox, _dx, _dy, layout_jitter = apply_puzzle_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    board_left = float(board_bbox[0])
    board_top = float(board_bbox[1])
    cell_size = float((float(board_bbox[2]) - float(board_bbox[0])) / float(SIZE))

    if panel_style is not None:
        panel_pad = max(18.0, float(params.board_border_width_px) * 2.5)
        panel_bbox = (
            int(round(max(6.0, float(board_bbox[0]) - panel_pad))),
            int(round(max(6.0, float(board_bbox[1]) - panel_pad))),
            int(
                round(
                    min(
                        float(params.canvas_width) - 6.0,
                        float(board_bbox[2]) + panel_pad,
                    )
                )
            ),
            int(
                round(
                    min(
                        float(params.canvas_height) - 6.0,
                        float(board_bbox[3]) + panel_pad,
                    )
                )
            ),
        )
        draw_puzzle_panel_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=20,
            border_width=2,
        )

    draw.rectangle(board_bbox, fill=tuple(int(v) for v in theme.board_fill_rgb))
    inner_board_bbox = (
        round(float(board_bbox[0] + int(params.board_border_width_px)), 3),
        round(float(board_bbox[1] + int(params.board_border_width_px)), 3),
        round(float(board_bbox[2] - int(params.board_border_width_px)), 3),
        round(float(board_bbox[3] - int(params.board_border_width_px)), 3),
    )
    draw.rectangle(inner_board_bbox, fill=tuple(int(v) for v in theme.cell_fill_rgb))

    highlighted_coords: set[Coord] = set()
    if highlighted_unit_type is not None and highlighted_unit_index is not None:
        highlighted_coords = set(
            unit_coords(str(highlighted_unit_type), int(highlighted_unit_index))
        )
    highlighted_cell_ids = [
        coord_to_cell_id(coord) for coord in sorted(highlighted_coords)
    ]
    highlighted_unit_bboxes = [
        _cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=row,
            col=col,
        )
        for row, col in sorted(highlighted_coords)
    ]
    highlighted_unit_bbox_px = (
        bbox_union(highlighted_unit_bboxes) if highlighted_unit_bboxes else None
    )
    for row, col in sorted(highlighted_coords):
        draw.rectangle(
            _cell_bbox(
                board_left=board_left,
                board_top=board_top,
                cell_size=cell_size,
                row=row,
                col=col,
            ),
            fill=tuple(int(v) for v in theme.highlighted_cell_fill_rgba),
        )
    if marked_cell is not None:
        mark_row, mark_col = int(marked_cell[0]), int(marked_cell[1])
        draw.rectangle(
            _cell_bbox(
                board_left=board_left,
                board_top=board_top,
                cell_size=cell_size,
                row=mark_row,
                col=mark_col,
            ),
            fill=tuple(int(v) for v in theme.marked_cell_fill_rgba),
        )

    cell_bboxes_px: Dict[str, list[float]] = {}
    scene_entities: list[Dict[str, Any]] = []
    cell_specs: list[SudokuCellSpec] = []
    conflict_set = {(int(row), int(col)) for row, col in conflict_coords}
    option_label_by_coord = {
        (int(spec["row"]), int(spec["col"])): str(spec["label"])
        for spec in option_specs
    }
    option_cell_ids_by_label = {
        str(spec["label"]): coord_to_cell_id((int(spec["row"]), int(spec["col"])))
        for spec in option_specs
    }
    for row in range(SIZE):
        for col in range(SIZE):
            cell_id = coord_to_cell_id((row, col))
            bbox_px = _cell_bbox(
                board_left=board_left,
                board_top=board_top,
                cell_size=cell_size,
                row=row,
                col=col,
                padding_px=float(params.cell_padding_px),
            )
            cell_bboxes_px[cell_id] = list(bbox_px)
            value = int(board[row][col])
            if value != 0:
                _draw_digit(
                    draw,
                    bbox_px=bbox_px,
                    digit=int(value),
                    theme=theme,
                    font_size_px=int(params.digit_font_size_px),
                    conflict=(row, col) in conflict_set,
                    font_family=str(params.font_family),
                )
            scene_entities.append(
                {
                    "entity_id": str(cell_id),
                    "entity_type": "sudoku_cell",
                    "row": int(row),
                    "col": int(col),
                    "value": int(value),
                    "filled": bool(value != 0),
                    "highlighted": bool((row, col) in highlighted_coords),
                    "option_label": option_label_by_coord.get((row, col)),
                    "marked": bool(
                        marked_cell is not None
                        and (row, col) == (int(marked_cell[0]), int(marked_cell[1]))
                    ),
                    "conflict": bool((row, col) in conflict_set),
                    "bbox_px": list(bbox_px),
                }
            )
            cell_specs.append(
                SudokuCellSpec(
                    cell_id=str(cell_id),
                    row=int(row),
                    col=int(col),
                    value=int(value),
                    bbox_px=bbox_px,
                )
            )

    # Draw grid lines after fills and digits so 3 by 3 structure remains clear.
    for index in range(SIZE + 1):
        x = float(board_left + (index * cell_size))
        y = float(board_top + (index * cell_size))
        is_box_line = int(index) % 3 == 0
        line_rgb = theme.box_line_rgb if bool(is_box_line) else theme.grid_line_rgb
        line_width = int(
            params.box_line_width_px if bool(is_box_line) else params.grid_line_width_px
        )
        draw.line(
            [(x, board_top), (x, float(board_bbox[3]))],
            fill=tuple(int(v) for v in line_rgb),
            width=int(line_width),
        )
        draw.line(
            [(board_left, y), (float(board_bbox[2]), y)],
            fill=tuple(int(v) for v in line_rgb),
            width=int(line_width),
        )
    draw.rectangle(
        board_bbox,
        outline=tuple(int(v) for v in theme.board_border_rgb),
        width=int(params.board_border_width_px),
    )

    if marked_cell is not None:
        mark_row, mark_col = int(marked_cell[0]), int(marked_cell[1])
        marker_bbox = _cell_bbox(
            board_left=board_left,
            board_top=board_top,
            cell_size=cell_size,
            row=mark_row,
            col=mark_col,
            padding_px=0.5 * float(params.marked_cell_outline_width_px),
        )
        marker_style = resolve_semantic_marker_style(
            instance_seed=int(params.instance_seed),
            namespace=f"puzzles.sudoku.marked_cell.{mark_row}.{mark_col}",
            role="sudoku_marked_cell",
            surface_rgbs=(
                tuple(int(v) for v in theme.marked_cell_fill_rgba[:3]),
                tuple(int(v) for v in theme.cell_fill_rgb),
            ),
            preferred_rgbs=(tuple(int(v) for v in theme.marked_cell_outline_rgb),),
        )
        draw_semantic_bbox_marker(
            draw,
            marker_bbox,
            style=marker_style,
            width=int(params.marked_cell_outline_width_px),
            marker_kind="sudoku_marked_cell_outline",
            extra_metadata={"cell_id": coord_to_cell_id((mark_row, mark_col))},
        )

    option_badge_bboxes_px: Dict[str, list[float]] = {}
    for spec in option_specs:
        label = str(spec["label"])
        row, col = int(spec["row"]), int(spec["col"])
        cell_id = coord_to_cell_id((row, col))
        badge_bbox = _draw_option_label(
            draw,
            bbox_px=tuple(float(value) for value in cell_bboxes_px[cell_id]),
            label=label,
            theme=theme,
            font_family=str(params.font_family),
            filled_cell=bool(int(board[row][col]) != 0),
            instance_seed=int(params.instance_seed),
        )
        option_badge_bboxes_px[label] = list(badge_bbox)

    render_map = {
        "board_bbox_px": list(board_bbox),
        "cell_bboxes_px": dict(cell_bboxes_px),
        "option_cell_ids_by_label": dict(option_cell_ids_by_label),
        "option_cell_bboxes_px": {
            str(label): list(cell_bboxes_px[str(cell_id)])
            for label, cell_id in option_cell_ids_by_label.items()
        },
        "option_badge_bboxes_px": dict(option_badge_bboxes_px),
        "highlighted_cell_ids": list(highlighted_cell_ids),
        "highlighted_unit_bbox_px": highlighted_unit_bbox_px,
        "marked_cell_id": (
            coord_to_cell_id(marked_cell) if marked_cell is not None else None
        ),
        "conflict_cell_ids": [
            coord_to_cell_id(coord) for coord in sorted(conflict_set)
        ],
        "style_variant": str(style_variant),
        "text_style": {"font_family": str(params.font_family)},
        "font_family": str(params.font_family),
        "panel_scene_style": (
            None if panel_style is None else puzzle_scene_style_metadata(panel_style)
        ),
        "layout_jitter": dict(layout_jitter),
    }
    return RenderedSudokuScene(
        image=image.convert("RGB"),
        cell_specs=tuple(cell_specs),
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


def _repeated_digit_conflict_coords(sample: SudokuSample) -> tuple[Coord, ...]:
    """Return highlighted-unit cells whose visible digit is one of the repeats."""

    if sample.highlighted_unit_type is None or sample.highlighted_unit_index is None:
        return ()
    repeated = {int(value) for value in sample.repeated_digit_values}
    if not repeated:
        return ()
    return tuple(
        coord
        for coord in unit_coords(
            str(sample.highlighted_unit_type),
            int(sample.highlighted_unit_index),
        )
        if int(sample.board[int(coord[0])][int(coord[1])]) in repeated
    )


def render_sudoku_visual_artifacts(
    *,
    sample: SudokuSample,
    style_variant: str,
    render_params: SudokuRenderParams,
    instance_seed: int,
    params: dict[str, Any],
    noise_defaults: dict[str, Any],
) -> SudokuVisualArtifacts:
    """Render one Sudoku sample on the shared puzzle background treatment."""

    panel_style, panel_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace="puzzles.sudoku.panel_scene_style",
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_sudoku_grid_scene(
        board=sample.board,
        background=background,
        style_variant=str(style_variant),
        params=render_params,
        highlighted_unit_type=sample.highlighted_unit_type,
        highlighted_unit_index=sample.highlighted_unit_index,
        marked_cell=sample.marked_cell,
        conflict_coords=(
            _repeated_digit_conflict_coords(sample)
            if sample.construction_mode == "highlighted_unit_repeats"
            else ()
        ),
        option_specs=sample.option_specs,
        panel_style=panel_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return SudokuVisualArtifacts(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "RenderedSudokuScene",
    "SudokuCellSpec",
    "SudokuVisualArtifacts",
    "render_sudoku_grid_scene",
    "render_sudoku_visual_artifacts",
]
