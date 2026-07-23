"""Rendering primitives for symbolic Turing tape scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from ....shared.drawing import draw_arrow, draw_centered_text, draw_rounded_rect
from ....shared.text_rendering import load_font, temporary_default_font_family
from ...shared.scene_style import (
    SymbolicSceneStyle,
    draw_symbolic_chrome_by_mode,
    draw_symbolic_grid_cell,
)

from .state import RenderedTuringScene, TuringDataset, TuringRenderParams


def grid_bbox(
    *,
    left: int,
    top: int,
    rows: int,
    cols: int,
    cell_size: int,
    gap: int,
) -> Tuple[int, int, int, int]:
    """Return a bbox around a regular grid."""

    width = int(cols * cell_size + max(0, cols - 1) * gap)
    height = int(rows * cell_size + max(0, rows - 1) * gap)
    return (int(left), int(top), int(left + width), int(top + height))


def cell_bbox(
    *,
    left: int,
    top: int,
    row: int,
    col: int,
    cell_size: int,
    gap: int,
) -> Tuple[int, int, int, int]:
    """Return the bbox for one grid cell."""

    x0 = int(left + col * (cell_size + gap))
    y0 = int(top + row * (cell_size + gap))
    return (x0, y0, int(x0 + cell_size), int(y0 + cell_size))


def decorate_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Tuple[int, int, int, int],
    radius: int,
    border_width: int,
    style: SymbolicSceneStyle,
    chrome_mode: str,
) -> None:
    """Draw a symbolic panel frame."""

    draw_symbolic_chrome_by_mode(
        draw,
        bbox=bbox,
        style=style,
        radius=int(radius),
        border_width=int(border_width),
        mode=str(chrome_mode),
    )


def _draw_turing_tape(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: TuringDataset,
    left: int,
    top: int,
    cell_size: int,
    gap: int,
    item_bboxes: Dict[str, Tuple[int, int, int, int]],
    style: SymbolicSceneStyle,
) -> Tuple[int, int, int, int]:
    """Draw the initial tape panel and record reusable bboxes for task annotation."""

    font = load_font(max(16, int(cell_size * 0.42)), bold=True)
    small_font = load_font(max(11, int(cell_size * 0.22)), bold=True)
    symbol_to_color = {
        symbol: tuple(style.state_colors[index % len(style.state_colors)])
        for index, symbol in enumerate(dataset.symbols)
    }
    for index, symbol in enumerate(dataset.initial_tape):
        bbox = (
            int(left + index * (cell_size + gap)),
            int(top),
            int(left + index * (cell_size + gap) + cell_size),
            int(top + cell_size),
        )
        item_bboxes[f"tape_cell_{index}"] = bbox
        draw_symbolic_grid_cell(
            draw,
            bbox=bbox,
            fill=symbol_to_color[str(symbol)],
            style=style,
            outline=style.grid_rgb,
            width=2,
            selected=int(index) == int(dataset.start_head),
            selected_width=max(3, int(cell_size * 0.09)),
        )
        draw_centered_text(
            draw,
            text=str(symbol),
            center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            font=font,
            fill=style.text_rgb,
            stroke_fill=style.text_stroke_rgb,
            stroke_width=1,
        )
        draw_centered_text(
            draw,
            text=str(index + 1),
            center=((bbox[0] + bbox[2]) / 2.0, bbox[3] + max(11, int(cell_size * 0.18))),
            font=small_font,
            fill=style.text_rgb,
            stroke_fill=style.text_stroke_rgb,
            stroke_width=1,
        )
    tape_bbox = grid_bbox(left=left, top=top, rows=1, cols=int(dataset.tape_length), cell_size=cell_size, gap=gap)
    item_bboxes["source_tape"] = (
        int(tape_bbox[0]),
        int(tape_bbox[1]),
        int(tape_bbox[2]),
        int(tape_bbox[3] + max(18, int(cell_size * 0.30))),
    )
    head_cell = cell_bbox(left=left, top=top, row=0, col=int(dataset.start_head), cell_size=cell_size, gap=gap)
    head_cx = int((head_cell[0] + head_cell[2]) / 2)
    arrow_top = int(head_cell[1] - max(32, int(cell_size * 0.48)))
    arrow_end = int(head_cell[1] - 5)
    draw_arrow(
        draw,
        start=(head_cx, arrow_top),
        end=(head_cx, arrow_end),
        fill=style.agent_rgb,
        width=max(3, int(cell_size * 0.08)),
        head_length_px=max(12, int(cell_size * 0.22)),
        head_width_px=max(14, int(cell_size * 0.24)),
    )
    head_label_bbox = (
        int(head_cx - cell_size * 0.65),
        int(arrow_top - max(22, int(cell_size * 0.30))),
        int(head_cx + cell_size * 0.65),
        int(arrow_top - 2),
    )
    draw_rounded_rect(draw, head_label_bbox, radius=8, fill=style.panel_accent_rgb, outline=style.panel_border_rgb, width=1)
    draw_centered_text(
        draw,
        text=f"HEAD {dataset.start_state}",
        center=((head_label_bbox[0] + head_label_bbox[2]) / 2.0, (head_label_bbox[1] + head_label_bbox[3]) / 2.0),
        font=small_font,
        fill=style.text_rgb,
        stroke_fill=style.text_stroke_rgb,
        stroke_width=1,
    )
    item_bboxes["start_head"] = (
        int(min(head_label_bbox[0], head_cell[0])),
        int(head_label_bbox[1]),
        int(max(head_label_bbox[2], head_cell[2])),
        int(head_cell[1]),
    )
    return item_bboxes["source_tape"]


def _draw_transition_table(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: TuringDataset,
    left: int,
    top: int,
    row_height: int,
    style: SymbolicSceneStyle,
) -> Tuple[int, int, int, int]:
    """Draw the complete transition table, highlighting rules used by the simulation trace."""

    col_widths = (76, 70, 72, 70, 84)
    headers = ("State", "Read", "Write", "Move", "Next")
    table_width = int(sum(col_widths))
    header_font = load_font(max(12, int(row_height * 0.42)), bold=True)
    cell_font = load_font(max(12, int(row_height * 0.42)), bold=False)
    x = int(left)
    y = int(top)
    header_bbox = (x, y, x + table_width, y + row_height)
    draw_rounded_rect(draw, header_bbox, radius=10, fill=style.panel_accent_rgb, outline=style.panel_border_rgb, width=2)
    cursor = x
    for width, header in zip(col_widths, headers):
        draw_centered_text(
            draw,
            text=header,
            center=(cursor + width / 2, y + row_height / 2),
            font=header_font,
            fill=style.text_rgb,
            stroke_fill=style.text_stroke_rgb,
            stroke_width=1,
        )
        cursor += int(width)
    used_keys = {(trace.state, trace.read_symbol) for trace in dataset.traces}
    for row_index, transition in enumerate(dataset.transitions, 1):
        y0 = int(top + row_index * row_height)
        row_bbox = (x, y0, x + table_width, y0 + row_height)
        row_fill = style.option_marker_fill_rgb if (transition.state, transition.read_symbol) in used_keys else style.option_fill_rgb
        draw.rectangle(row_bbox, fill=row_fill, outline=style.grid_rgb, width=1)
        values = (
            transition.state,
            transition.read_symbol,
            transition.write_symbol,
            transition.move,
            transition.next_state,
        )
        cursor = x
        for width, value in zip(col_widths, values):
            draw_centered_text(
                draw,
                text=str(value),
                center=(cursor + width / 2, y0 + row_height / 2),
                font=cell_font,
                fill=style.text_rgb,
                stroke_fill=style.text_stroke_rgb,
                stroke_width=1,
            )
            cursor += int(width)
    return (x, y, x + table_width, int(top + (len(dataset.transitions) + 1) * row_height))


def render_turing_scene(
    *,
    background: Image.Image,
    dataset: TuringDataset,
    scene_variant: str,
    render_params: TuringRenderParams,
    style: SymbolicSceneStyle,
    style_meta: Mapping[str, Any],
) -> RenderedTuringScene:
    """Render one Turing tape plus transition table."""

    image = background.copy()
    with temporary_default_font_family(render_params.font_family):
        draw = ImageDraw.Draw(image)
        item_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        cell = max(38, min(60, int(render_params.cell_size_px)))
        gap = max(2, int(render_params.grid_gap_px))
        row_height = max(28, min(36, int(render_params.cell_size_px * 0.58)))
        layout_gap = 34
        tape_width = int(dataset.tape_length * cell + max(0, dataset.tape_length - 1) * gap)
        tape_left = int((render_params.canvas_width - tape_width) // 2)
        machine_panel_height = int(cell + 2 * render_params.panel_padding_px + 108)
        table_panel_height = int((len(dataset.transitions) + 1) * row_height + 36)
        total_layout_height = int(machine_panel_height + layout_gap + table_panel_height)
        machine_panel_top = max(16, int((render_params.canvas_height - total_layout_height) // 2))
        tape_top = int(machine_panel_top + render_params.panel_padding_px + 62)
        machine_panel = (
            int(tape_left - render_params.panel_padding_px),
            int(machine_panel_top),
            int(tape_left + tape_width + render_params.panel_padding_px),
            int(machine_panel_top + machine_panel_height),
        )
        decorate_panel(
            draw,
            bbox=machine_panel,
            radius=int(render_params.panel_corner_radius_px),
            border_width=int(render_params.panel_border_width_px),
            style=style,
            chrome_mode=str(style_meta.get("panel_chrome_mode", "accent_frame")),
        )
        _draw_turing_tape(
            draw,
            dataset=dataset,
            left=tape_left,
            top=tape_top,
            cell_size=cell,
            gap=gap,
            item_bboxes=item_bboxes,
            style=style,
        )
        chip_font = load_font(max(13, int(render_params.small_font_size_px)), bold=True)
        chip_y = int(machine_panel[3] - render_params.panel_padding_px - 22)
        chips = (
            f"steps {dataset.steps}",
            f"symbols {' / '.join(str(symbol) for symbol in dataset.symbols)}",
        )
        chip_x = int(machine_panel[0] + render_params.panel_padding_px)
        symbols_chip_bbox = (0, 0, 0, 0)
        for chip in chips:
            chip_w = max(82, 16 + len(chip) * max(8, int(render_params.small_font_size_px * 0.52)))
            bbox = (chip_x, chip_y, int(chip_x + chip_w), int(chip_y + 28))
            draw_rounded_rect(draw, bbox, radius=10, fill=style.step_fill_rgb, outline=style.panel_border_rgb, width=1)
            draw_centered_text(
                draw,
                text=chip,
                center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                font=chip_font,
                fill=style.text_rgb,
                stroke_fill=style.text_stroke_rgb,
                stroke_width=1,
            )
            if chip.startswith("symbols "):
                symbols_chip_bbox = bbox
            chip_x += int(chip_w + 12)
        item_bboxes["machine_panel"] = machine_panel
        item_bboxes["symbols_chip"] = symbols_chip_bbox

        table_width = 372
        table_left = int((render_params.canvas_width - table_width) // 2)
        table_top = int(machine_panel[3] + layout_gap)
        table_bbox_raw = _draw_transition_table(
            draw,
            dataset=dataset,
            left=table_left,
            top=table_top,
            row_height=row_height,
            style=style,
        )
        table_panel = (
            int(table_bbox_raw[0] - 18),
            int(table_bbox_raw[1] - 18),
            int(table_bbox_raw[2] + 18),
            int(table_bbox_raw[3] + 18),
        )
        table_crop = image.crop(table_bbox_raw)
        decorate_panel(
            draw,
            bbox=table_panel,
            radius=int(render_params.panel_corner_radius_px),
            border_width=2,
            style=style,
            chrome_mode="plain_panel",
        )
        image.paste(table_crop, table_bbox_raw)
    item_bboxes["transition_table"] = table_panel
    scene_bbox = (
        int(min(machine_panel[0], table_panel[0])),
        int(min(machine_panel[1], table_panel[1])),
        int(max(machine_panel[2], table_panel[2])),
        int(max(machine_panel[3], table_panel[3])),
    )
    entities = tuple(
        {
            "entity_id": key,
            "bbox_px": list(value),
            "entity_type": "turing_machine_item",
        }
        for key, value in sorted(item_bboxes.items())
    )
    return RenderedTuringScene(
        image=image,
        scene_bbox_px=scene_bbox,
        item_bboxes=item_bboxes,
        entities=entities,
        layout_jitter={
            "enabled": False,
            "reason": "single_tape_and_table_layout",
            "canvas_size_px": [int(render_params.canvas_width), int(render_params.canvas_height)],
            "tape_cell_size_px": int(cell),
            "transition_row_height_px": int(row_height),
        },
        style_metadata=dict(style_meta),
    )
