"""Renderer for symbolic Braille-cell scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle

from .state import BRAILLE_POSITIONS, BrailleCellSpec, BraillePlateSpec, BrailleRenderParams, RenderedBrailleScene


def _rounded_bbox(values: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def _position_center(
    bbox: Sequence[float],
    *,
    position: int,
) -> tuple[float, float]:
    left, top, right, bottom = [float(value) for value in bbox]
    column = 0 if int(position) in {1, 2, 3} else 1
    row = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}[int(position)]
    x_pad = 0.34 * float(right - left)
    y_pad = 0.25 * float(bottom - top)
    col_gap = float(right - left) - (2.0 * x_pad)
    row_gap = (float(bottom - top) - (2.0 * y_pad)) / 2.0
    return float(left + x_pad + (column * col_gap)), float(top + y_pad + (row * row_gap))


def _draw_braille_cell(
    draw: ImageDraw.ImageDraw,
    *,
    spec: BrailleCellSpec,
    bbox: Sequence[float],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
    label_position: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, Any]]:
    """Draw one six-dot Braille cell and return all projected dot witnesses."""

    cell_bbox = _rounded_bbox(bbox)
    raised_set = {int(pos) for pos in spec.raised_positions}
    outline = tuple(int(value) for value in style.panel_border_rgb)
    fill = tuple(int(value) for value in style.panel_fill_rgb)
    if spec.marked:
        outline = tuple(int(value) for value in style.panel_accent_rgb)
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in cell_bbox),
        radius=int(params.cell_corner_radius_px),
        fill=fill,
        outline=outline,
        width=int(params.marked_border_width_px if spec.marked else params.cell_border_width_px),
    )

    dot_centers: dict[str, list[float]] = {}
    raised_dot_centers: dict[str, list[float]] = {}
    for position in BRAILLE_POSITIONS:
        cx, cy = _position_center(cell_bbox, position=int(position))
        dot_id = f"{spec.item_id}_dot_{position}"
        center = [round(float(cx), 3), round(float(cy), 3)]
        dot_centers[dot_id] = center
        if int(position) in raised_set:
            radius = int(params.dot_radius_px)
            dot_fill = tuple(int(value) for value in style.text_rgb)
            dot_outline = tuple(int(value) for value in style.text_rgb)
            raised_dot_centers[dot_id] = center
        else:
            radius = int(params.empty_dot_radius_px)
            dot_fill = tuple(int(value) for value in style.panel_fill_rgb)
            dot_outline = tuple(int(value) for value in style.grid_rgb)
        draw.ellipse(
            (float(cx - radius), float(cy - radius), float(cx + radius), float(cy + radius)),
            fill=dot_fill,
            outline=dot_outline,
            width=2,
        )

    label_bbox: list[float] | None = None
    if str(spec.label).strip():
        font = load_font(int(params.option_label_font_size_px), bold=True)
        label_y = float(cell_bbox[1] - 24) if str(label_position) == "above" else float(cell_bbox[3] + 28)
        label_bbox = draw_centered_text(
            draw,
            text=str(spec.label),
            center=(0.5 * (float(cell_bbox[0]) + float(cell_bbox[2])), label_y),
            font=font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )

    entity = {
        "item_id": str(spec.item_id),
        "entity_type": "braille_cell",
        "role": str(spec.role),
        "label": str(spec.label),
        "raised_positions": [int(pos) for pos in spec.raised_positions],
        "bbox_px": list(cell_bbox),
        "marked": bool(spec.marked),
        "dot_center_ids": [f"{spec.item_id}_dot_{position}" for position in BRAILLE_POSITIONS],
        "raised_dot_center_ids": [f"{spec.item_id}_dot_{position}" for position in spec.raised_positions],
    }
    if label_bbox is not None:
        entity["label_bbox_px"] = list(label_bbox)
    return dot_centers, raised_dot_centers, entity


def _scaled_params(
    params: BrailleRenderParams,
    *,
    cell_width: int,
    cell_height: int,
    dot_radius: int,
    empty_dot_radius: int,
) -> BrailleRenderParams:
    """Return temporary render params for compact word-plate cells."""

    return replace(
        params,
        cell_width_px=int(cell_width),
        cell_height_px=int(cell_height),
        dot_radius_px=int(dot_radius),
        empty_dot_radius_px=int(empty_dot_radius),
        cell_corner_radius_px=max(5, int(round(0.16 * int(cell_width)))),
        cell_border_width_px=1,
        marked_border_width_px=2,
    )


def _draw_card(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
    marked: bool,
) -> list[float]:
    """Draw one rounded option/source card."""

    card_bbox = _rounded_bbox(bbox)
    outline_rgb = tuple(int(value) for value in (style.panel_accent_rgb if bool(marked) else style.panel_border_rgb))
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in card_bbox),
        radius=max(12, int(params.cell_corner_radius_px)),
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=outline_rgb,
        width=int(params.marked_border_width_px if bool(marked) else params.cell_border_width_px),
    )
    return card_bbox


def _draw_braille_word_plate(
    draw: ImageDraw.ImageDraw,
    *,
    plate: BraillePlateSpec,
    bbox: Sequence[float],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
    cell_width: int,
    cell_height: int,
    dot_radius: int,
    empty_dot_radius: int,
    label_position: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], dict[str, dict[str, list[float]]], list[dict[str, Any]]]:
    """Draw a multi-cell Braille word plate inside a card bbox."""

    card_bbox = _draw_card(draw, bbox=bbox, params=params, style=style, marked=bool(plate.marked))
    compact_params = _scaled_params(
        params,
        cell_width=int(cell_width),
        cell_height=int(cell_height),
        dot_radius=int(dot_radius),
        empty_dot_radius=int(empty_dot_radius),
    )
    cells = tuple(plate.cells)
    if not cells:
        raise ValueError("Braille word plate requires at least one cell")

    label_bbox: list[float] | None = None
    if str(plate.label).strip():
        font = load_font(int(params.option_label_font_size_px), bold=True)
        label_y = float(card_bbox[1] - 22) if str(label_position) == "above" else float(card_bbox[3] + 24)
        label_bbox = draw_centered_text(
            draw,
            text=str(plate.label),
            center=(0.5 * (float(card_bbox[0]) + float(card_bbox[2])), label_y),
            font=font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )

    max_gap = max(4, int(round(0.18 * int(cell_width))))
    available_width = max(1.0, float(card_bbox[2] - card_bbox[0] - 34.0))
    gap = min(max_gap, int(max(4.0, (available_width - (len(cells) * int(cell_width))) / max(1, len(cells) - 1))))
    total_width = (len(cells) * int(cell_width)) + ((len(cells) - 1) * int(gap))
    start_x = float(0.5 * (card_bbox[0] + card_bbox[2]) - (0.5 * total_width))
    cell_top = float(0.5 * (card_bbox[1] + card_bbox[3]) - (0.5 * int(cell_height)))

    item_bboxes: dict[str, list[float]] = {str(plate.item_id): list(card_bbox)}
    dot_centers: dict[str, list[float]] = {}
    raised_dot_centers: dict[str, list[float]] = {}
    cell_dot_centers: dict[str, dict[str, list[float]]] = {}
    entities: list[dict[str, Any]] = [
        {
            "item_id": str(plate.item_id),
            "entity_type": "braille_plate",
            "role": str(plate.role),
            "label": str(plate.label),
            "raised_positions": [],
            "bbox_px": list(card_bbox),
            "marked": bool(plate.marked),
            "cell_ids": [str(cell.item_id) for cell in cells],
        }
    ]
    if label_bbox is not None:
        entities[0]["label_bbox_px"] = list(label_bbox)

    for index, cell in enumerate(cells):
        left = float(start_x + (index * (int(cell_width) + int(gap))))
        cell_bbox = [left, cell_top, left + int(cell_width), cell_top + int(cell_height)]
        cell_dots, cell_raised, cell_entity = _draw_braille_cell(
            draw,
            spec=cell,
            bbox=cell_bbox,
            params=compact_params,
            style=style,
            label_position="below",
        )
        item_bboxes[str(cell.item_id)] = _rounded_bbox(cell_bbox)
        dot_centers.update(cell_dots)
        raised_dot_centers.update(cell_raised)
        cell_dot_centers[str(cell.item_id)] = dict(cell_dots)
        entities.append(cell_entity)

    return item_bboxes, dot_centers, raised_dot_centers, cell_dot_centers, entities


def _draw_word_card(
    draw: ImageDraw.ImageDraw,
    *,
    item_id: str,
    label: str,
    word: str,
    bbox: Sequence[float],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
    marked: bool,
    font_size: int,
    label_position: str,
) -> tuple[list[float], dict[str, Any]]:
    """Draw one word option/source card and return its bbox/entity record."""

    card_bbox = _draw_card(draw, bbox=bbox, params=params, style=style, marked=bool(marked))
    if str(label).strip():
        label_font = load_font(int(params.option_label_font_size_px), bold=True)
        label_y = float(card_bbox[1] - 22) if str(label_position) == "above" else float(card_bbox[3] + 24)
        draw_centered_text(
            draw,
            text=str(label),
            center=(0.5 * (float(card_bbox[0]) + float(card_bbox[2])), label_y),
            font=label_font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )
    word_font = load_font(int(font_size), bold=True)
    word_bbox = draw_centered_text(
        draw,
        text=str(word),
        center=(0.5 * (float(card_bbox[0]) + float(card_bbox[2])), 0.5 * (float(card_bbox[1]) + float(card_bbox[3]))),
        font=word_font,
        fill=style.text_rgb,
        stroke_fill=style.panel_fill_rgb,
        stroke_width=2,
    )
    entity = {
        "item_id": str(item_id),
        "entity_type": "word_card",
        "role": "word_option" if str(label).strip() else "source_word",
        "label": str(label),
        "word": str(word),
        "bbox_px": list(card_bbox),
        "text_bbox_px": list(word_bbox),
        "raised_positions": [],
        "marked": bool(marked),
    }
    return list(card_bbox), entity


def render_braille_count_scene(
    image: Image.Image,
    *,
    cells: Sequence[BrailleCellSpec],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedBrailleScene:
    """Render a row of Braille cells with one target cell marked."""

    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    cell_count = len(cells)
    if int(cell_count) <= 0:
        raise ValueError("Braille count scene requires at least one cell")
    cell_gap = 34
    if int(cell_count) > 1:
        max_gap = int((max(0, width - 64 - (cell_count * int(params.cell_width_px)))) / (cell_count - 1))
        cell_gap = max(18, min(cell_gap, int(max_gap)))
    total_width = (cell_count * int(params.cell_width_px)) + ((cell_count - 1) * int(cell_gap))
    start_x = max(24, int(round((width - total_width) / 2)))
    top = int(round(0.5 * height - 0.5 * int(params.cell_height_px)))

    entities: list[dict[str, Any]] = []
    item_bboxes: dict[str, list[float]] = {}
    dot_centers: dict[str, list[float]] = {}
    raised_dot_centers: dict[str, list[float]] = {}
    cell_dot_centers: dict[str, dict[str, list[float]]] = {}
    for index, spec in enumerate(cells):
        left = float(start_x + (index * (int(params.cell_width_px) + int(cell_gap))))
        bbox = [left, float(top), left + int(params.cell_width_px), float(top + int(params.cell_height_px))]
        cell_dots, cell_raised, entity = _draw_braille_cell(
            draw,
            spec=spec,
            bbox=bbox,
            params=params,
            style=style,
            label_position="below",
        )
        entities.append(entity)
        item_bboxes[str(spec.item_id)] = _rounded_bbox(bbox)
        dot_centers.update(cell_dots)
        raised_dot_centers.update(cell_raised)
        cell_dot_centers[str(spec.item_id)] = dict(cell_dots)

    scene_bbox = [30.0, float(top - 72), float(width - 30), float(top + int(params.cell_height_px) + 72)]
    return RenderedBrailleScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        dot_centers=dot_centers,
        raised_dot_centers=raised_dot_centers,
        cell_dot_centers=cell_dot_centers,
        scene_bbox_px=_rounded_bbox(scene_bbox),
        style_metadata={"renderer": "braille_cell_v1", "layout": "marked_row"},
    )


def render_braille_match_scene(
    image: Image.Image,
    *,
    reference: BrailleCellSpec,
    options: Sequence[BrailleCellSpec],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedBrailleScene:
    """Render one reference Braille cell and six labeled option cells."""

    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    option_count = len(options)
    if int(option_count) != 6:
        raise ValueError("Braille matching scene requires exactly six options")

    item_bboxes: dict[str, list[float]] = {}
    dot_centers: dict[str, list[float]] = {}
    raised_dot_centers: dict[str, list[float]] = {}
    cell_dot_centers: dict[str, dict[str, list[float]]] = {}
    entities: list[dict[str, Any]] = []

    ref_left = 88.0
    ref_top = float(round(0.5 * height - 0.5 * int(params.cell_height_px)))
    ref_bbox = [ref_left, ref_top, ref_left + int(params.cell_width_px), ref_top + int(params.cell_height_px)]
    ref_dots, ref_raised, ref_entity = _draw_braille_cell(
        draw,
        spec=reference,
        bbox=ref_bbox,
        params=params,
        style=style,
        label_position="above",
    )
    item_bboxes[str(reference.item_id)] = _rounded_bbox(ref_bbox)
    dot_centers.update(ref_dots)
    raised_dot_centers.update(ref_raised)
    cell_dot_centers[str(reference.item_id)] = dict(ref_dots)
    entities.append(ref_entity)

    option_gap_x = 42
    option_gap_y = 52
    grid_cols = 3
    grid_left = 370.0
    grid_top = float(round(0.5 * height - int(params.cell_height_px) - (0.5 * option_gap_y)))
    for index, spec in enumerate(options):
        row = index // grid_cols
        col = index % grid_cols
        left = grid_left + (col * (int(params.cell_width_px) + option_gap_x))
        top = grid_top + (row * (int(params.cell_height_px) + option_gap_y))
        bbox = [left, top, left + int(params.cell_width_px), top + int(params.cell_height_px)]
        option_dots, option_raised, entity = _draw_braille_cell(
            draw,
            spec=spec,
            bbox=bbox,
            params=params,
            style=style,
            label_position="above",
        )
        item_bboxes[str(spec.item_id)] = _rounded_bbox(bbox)
        dot_centers.update(option_dots)
        raised_dot_centers.update(option_raised)
        cell_dot_centers[str(spec.item_id)] = dict(option_dots)
        entities.append(entity)

    scene_bbox = [40.0, 50.0, float(width - 40), float(height - 48)]
    return RenderedBrailleScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        dot_centers=dot_centers,
        raised_dot_centers=raised_dot_centers,
        cell_dot_centers=cell_dot_centers,
        scene_bbox_px=_rounded_bbox(scene_bbox),
        style_metadata={"renderer": "braille_cell_v1", "layout": "reference_and_six_options"},
    )


def render_braille_word_read_scene(
    image: Image.Image,
    *,
    source_plate: BraillePlateSpec,
    word_options: Sequence[tuple[str, str]],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedBrailleScene:
    """Render a Braille word plate with four candidate word labels."""

    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    if len(word_options) != 4:
        raise ValueError("Braille word read scene requires exactly four word options")

    item_bboxes: dict[str, list[float]] = {}
    dot_centers: dict[str, list[float]] = {}
    raised_dot_centers: dict[str, list[float]] = {}
    cell_dot_centers: dict[str, dict[str, list[float]]] = {}
    entities: list[dict[str, Any]] = []

    source_width = min(680, max(420, int(len(source_plate.cells) * 88 + 72)))
    source_bbox = [
        float(round(0.5 * width - 0.5 * source_width)),
        88.0,
        float(round(0.5 * width + 0.5 * source_width)),
        248.0,
    ]
    source_bboxes, source_dots, source_raised, source_cell_dots, source_entities = _draw_braille_word_plate(
        draw,
        plate=source_plate,
        bbox=source_bbox,
        params=params,
        style=style,
        cell_width=int(params.word_source_cell_width_px),
        cell_height=int(params.word_source_cell_height_px),
        dot_radius=int(params.word_source_dot_radius_px),
        empty_dot_radius=int(params.word_source_empty_dot_radius_px),
        label_position="above",
    )
    item_bboxes.update(source_bboxes)
    dot_centers.update(source_dots)
    raised_dot_centers.update(source_raised)
    cell_dot_centers.update(source_cell_dots)
    entities.extend(source_entities)

    option_width = 260
    option_height = 70
    gap_x = 46
    gap_y = 52
    grid_left = float(round(0.5 * width - 0.5 * ((2 * option_width) + gap_x)))
    grid_top = 370.0
    for index, (label, word) in enumerate(word_options):
        row = index // 2
        col = index % 2
        left = grid_left + (col * (option_width + gap_x))
        top = grid_top + (row * (option_height + gap_y))
        option_id = f"option_{label}"
        bbox, entity = _draw_word_card(
            draw,
            item_id=option_id,
            label=str(label),
            word=str(word),
            bbox=[left, top, left + option_width, top + option_height],
            params=params,
            style=style,
            marked=False,
            font_size=int(params.word_option_word_font_size_px),
            label_position="above",
        )
        item_bboxes[option_id] = _rounded_bbox(bbox)
        entities.append(entity)

    scene_bbox = [40.0, 48.0, float(width - 40), float(height - 48)]
    return RenderedBrailleScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        dot_centers=dot_centers,
        raised_dot_centers=raised_dot_centers,
        cell_dot_centers=cell_dot_centers,
        scene_bbox_px=_rounded_bbox(scene_bbox),
        style_metadata={"renderer": "braille_cell_v1", "layout": "braille_word_to_word_options"},
    )


def render_braille_word_match_scene(
    image: Image.Image,
    *,
    source_word: str,
    braille_options: Sequence[BraillePlateSpec],
    params: BrailleRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedBrailleScene:
    """Render a source word with four candidate Braille word plates."""

    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    if len(braille_options) != 4:
        raise ValueError("Braille word match scene requires exactly four Braille options")

    item_bboxes: dict[str, list[float]] = {}
    dot_centers: dict[str, list[float]] = {}
    raised_dot_centers: dict[str, list[float]] = {}
    cell_dot_centers: dict[str, dict[str, list[float]]] = {}
    entities: list[dict[str, Any]] = []

    source_width = min(520, max(300, int((len(source_word) * 44) + 132)))
    source_bbox = [
        float(round(0.5 * width - 0.5 * source_width)),
        56.0,
        float(round(0.5 * width + 0.5 * source_width)),
        136.0,
    ]
    bbox, entity = _draw_word_card(
        draw,
        item_id="source_word",
        label="",
        word=str(source_word),
        bbox=source_bbox,
        params=params,
        style=style,
        marked=True,
        font_size=int(params.word_source_word_font_size_px),
        label_position="above",
    )
    item_bboxes["source_word"] = _rounded_bbox(bbox)
    entities.append(entity)

    option_width = int(params.word_option_card_width_px)
    option_height = int(params.word_option_card_height_px)
    gap_x = 28
    gap_y = 48
    grid_left = float(round(0.5 * width - 0.5 * ((2 * option_width) + gap_x)))
    grid_top = 220.0
    for index, plate in enumerate(braille_options):
        row = index // 2
        col = index % 2
        left = grid_left + (col * (option_width + gap_x))
        top = grid_top + (row * (option_height + gap_y))
        plate_bboxes, plate_dots, plate_raised, plate_cell_dots, plate_entities = _draw_braille_word_plate(
            draw,
            plate=plate,
            bbox=[left, top, left + option_width, top + option_height],
            params=params,
            style=style,
            cell_width=int(params.word_option_cell_width_px),
            cell_height=int(params.word_option_cell_height_px),
            dot_radius=int(params.word_option_dot_radius_px),
            empty_dot_radius=int(params.word_option_empty_dot_radius_px),
            label_position="above",
        )
        item_bboxes.update(plate_bboxes)
        dot_centers.update(plate_dots)
        raised_dot_centers.update(plate_raised)
        cell_dot_centers.update(plate_cell_dots)
        entities.extend(plate_entities)

    scene_bbox = [40.0, 32.0, float(width - 40), float(height - 38)]
    return RenderedBrailleScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        dot_centers=dot_centers,
        raised_dot_centers=raised_dot_centers,
        cell_dot_centers=cell_dot_centers,
        scene_bbox_px=_rounded_bbox(scene_bbox),
        style_metadata={"renderer": "braille_cell_v1", "layout": "word_to_braille_word_options"},
    )


__all__ = [
    "render_braille_count_scene",
    "render_braille_match_scene",
    "render_braille_word_match_scene",
    "render_braille_word_read_scene",
]
