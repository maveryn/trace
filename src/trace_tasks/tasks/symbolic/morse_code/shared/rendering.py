"""Renderer for symbolic Morse-code scenes."""

from __future__ import annotations

from typing import Any, Sequence

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle

from .state import MorseRenderParams, MorseWordSpec, RenderedMorseScene


def _rounded_bbox(values: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def _draw_card(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    params: MorseRenderParams,
    style: SymbolicSceneStyle,
    marked: bool,
) -> list[float]:
    card_bbox = _rounded_bbox(bbox)
    outline = tuple(int(value) for value in (style.panel_accent_rgb if bool(marked) else style.panel_border_rgb))
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in card_bbox),
        radius=int(params.card_corner_radius_px),
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=outline,
        width=int(params.marked_border_width_px if bool(marked) else params.card_border_width_px),
    )
    return card_bbox


def _symbol_width(symbol: str, *, dot_radius: int, dash_width: int) -> int:
    return int(dot_radius) * 2 if str(symbol) == "." else int(dash_width)


def _letter_width(symbols: Sequence[str], *, dot_radius: int, dash_width: int, symbol_gap: int) -> int:
    if not symbols:
        return 0
    return sum(_symbol_width(symbol, dot_radius=dot_radius, dash_width=dash_width) for symbol in symbols) + (
        (len(symbols) - 1) * int(symbol_gap)
    )


def _code_width(letter_widths: Sequence[int], *, letter_gap: int) -> int:
    if not letter_widths:
        return 0
    return sum(int(width) for width in letter_widths) + ((len(letter_widths) - 1) * int(letter_gap))


def _line_groups(
    letters_with_widths: Sequence[tuple[Any, int]],
    *,
    letter_gap: int,
    available_width: float,
    force_single_line: bool,
) -> tuple[tuple[tuple[Any, int], ...], ...]:
    """Group Morse letters into centered lines without changing source-code semantics."""

    if bool(force_single_line):
        return (tuple(letters_with_widths),)
    total_width = _code_width([width for _letter, width in letters_with_widths], letter_gap=int(letter_gap))
    if float(total_width) <= float(available_width):
        return (tuple(letters_with_widths),)

    groups: list[list[tuple[Any, int]]] = []
    current: list[tuple[Any, int]] = []
    current_width = 0
    for letter, width in letters_with_widths:
        added_width = int(width) if not current else int(letter_gap) + int(width)
        if current and (current_width + added_width) > float(available_width):
            groups.append(current)
            current = [(letter, int(width))]
            current_width = int(width)
        else:
            current.append((letter, int(width)))
            current_width += int(added_width)
    if current:
        groups.append(current)
    return tuple(tuple(group) for group in groups)


def _draw_morse_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    symbol: str,
    left: float,
    center_y: float,
    dot_radius: int,
    dash_width: int,
    dash_height: int,
    fill: Sequence[int],
) -> list[float]:
    if str(symbol) == ".":
        bbox = [
            float(left),
            float(center_y - int(dot_radius)),
            float(left + (2 * int(dot_radius))),
            float(center_y + int(dot_radius)),
        ]
        draw.ellipse(tuple(bbox), fill=tuple(int(value) for value in fill))
        return _rounded_bbox(bbox)
    bbox = [
        float(left),
        float(center_y - (0.5 * int(dash_height))),
        float(left + int(dash_width)),
        float(center_y + (0.5 * int(dash_height))),
    ]
    draw.rounded_rectangle(tuple(bbox), radius=max(2, int(round(int(dash_height) / 2))), fill=tuple(int(value) for value in fill))
    return _rounded_bbox(bbox)


def _draw_morse_word_code(
    draw: ImageDraw.ImageDraw,
    *,
    spec: MorseWordSpec,
    bbox: Sequence[float],
    params: MorseRenderParams,
    style: SymbolicSceneStyle,
    source: bool,
    label_position: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[dict[str, Any]]]:
    """Draw one Morse-code card and keep every symbol projected after final placement."""

    card_bbox = _draw_card(draw, bbox=bbox, params=params, style=style, marked=bool(spec.marked))
    if str(spec.label).strip():
        label_font = load_font(int(params.option_label_font_size_px), bold=True)
        label_y = float(card_bbox[1] - 22) if str(label_position) == "above" else float(card_bbox[3] + 24)
        draw_centered_text(
            draw,
            text=str(spec.label),
            center=(0.5 * (float(card_bbox[0]) + float(card_bbox[2])), label_y),
            font=label_font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )

    dot_radius = int(params.code_symbol_dot_radius_px if source else params.option_symbol_dot_radius_px)
    dash_width = int(params.code_symbol_dash_width_px if source else params.option_symbol_dash_width_px)
    dash_height = int(params.code_symbol_dash_height_px if source else params.option_symbol_dash_height_px)
    symbol_gap = int(params.code_symbol_gap_px if source else params.option_symbol_gap_px)
    letter_gap = int(params.code_letter_gap_px if source else params.option_letter_gap_px)
    letters = tuple(spec.letters)
    letter_widths = tuple(
        _letter_width(
            [symbol.symbol for symbol in letter.symbols],
            dot_radius=dot_radius,
            dash_width=dash_width,
            symbol_gap=symbol_gap,
        )
        for letter in letters
    )
    available_width = float(card_bbox[2] - card_bbox[0] - 36.0)
    groups = _line_groups(
        tuple(zip(letters, letter_widths)),
        letter_gap=int(letter_gap),
        available_width=available_width,
        force_single_line=bool(source),
    )
    line_gap_y = max(24.0, float(3.2 * max(dot_radius, dash_height)))
    first_center_y = float(0.5 * (card_bbox[1] + card_bbox[3]) - (0.5 * (len(groups) - 1) * line_gap_y))

    item_bboxes: dict[str, list[float]] = {str(spec.item_id): list(card_bbox)}
    symbol_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = [
        {
            "item_id": str(spec.item_id),
            "entity_type": "morse_word_code",
            "role": str(spec.role),
            "label": str(spec.label),
            "word": str(spec.word),
            "code": " / ".join("".join(symbol.symbol for symbol in letter.symbols) for letter in letters),
            "bbox_px": list(card_bbox),
            "marked": bool(spec.marked),
            "letter_ids": [str(letter.item_id) for letter in letters],
        }
    ]
    for line_index, group in enumerate(groups):
        line_width = _code_width([width for _letter, width in group], letter_gap=int(letter_gap))
        cursor_x = float(0.5 * (card_bbox[0] + card_bbox[2]) - (0.5 * line_width))
        center_y = float(first_center_y + (line_index * line_gap_y))
        for letter, letter_width in group:
            letter_left = float(cursor_x)
            for symbol in letter.symbols:
                bbox_px = _draw_morse_symbol(
                    draw,
                    symbol=str(symbol.symbol),
                    left=cursor_x,
                    center_y=center_y,
                    dot_radius=dot_radius,
                    dash_width=dash_width,
                    dash_height=dash_height,
                    fill=style.text_rgb,
                )
                symbol_bboxes[str(symbol.item_id)] = bbox_px
                entities.append(
                    {
                        "item_id": str(symbol.item_id),
                        "entity_type": "morse_symbol",
                        "role": str(symbol.role),
                        "label": "",
                        "word": str(spec.word),
                        "code": str(symbol.symbol),
                        "bbox_px": list(bbox_px),
                        "marked": bool(spec.marked),
                    }
                )
                cursor_x += _symbol_width(str(symbol.symbol), dot_radius=dot_radius, dash_width=dash_width) + symbol_gap
            letter_bbox = [
                letter_left,
                center_y - max(dot_radius, dash_height),
                letter_left + letter_width,
                center_y + max(dot_radius, dash_height),
            ]
            item_bboxes[str(letter.item_id)] = _rounded_bbox(letter_bbox)
            entities.append(
                {
                    "item_id": str(letter.item_id),
                    "entity_type": "morse_letter",
                    "role": str(letter.role),
                    "label": "",
                    "word": str(spec.word),
                    "code": "".join(symbol.symbol for symbol in letter.symbols),
                    "bbox_px": _rounded_bbox(letter_bbox),
                    "marked": bool(spec.marked),
                }
            )
            cursor_x = letter_left + letter_width + letter_gap
    return item_bboxes, symbol_bboxes, entities


def _draw_word_card(
    draw: ImageDraw.ImageDraw,
    *,
    item_id: str,
    label: str,
    word: str,
    bbox: Sequence[float],
    params: MorseRenderParams,
    style: SymbolicSceneStyle,
    marked: bool,
    font_size: int,
    label_position: str,
) -> tuple[list[float], dict[str, Any]]:
    """Draw one source/option word card; callers decide whether its bbox is prompt-facing annotation."""

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
    text_bbox = draw_centered_text(
        draw,
        text=str(word).upper(),
        center=(0.5 * (float(card_bbox[0]) + float(card_bbox[2])), 0.5 * (float(card_bbox[1]) + float(card_bbox[3]))),
        font=word_font,
        fill=style.text_rgb,
        stroke_fill=style.panel_fill_rgb,
        stroke_width=2,
    )
    return list(card_bbox), {
        "item_id": str(item_id),
        "entity_type": "word_card",
        "role": "word_option" if str(label).strip() else "source_word",
        "label": str(label),
        "word": str(word),
        "bbox_px": list(card_bbox),
        "text_bbox_px": list(text_bbox),
        "marked": bool(marked),
    }


def render_morse_word_read_scene(
    image: Image.Image,
    *,
    source_code: MorseWordSpec,
    word_options: Sequence[tuple[str, str]],
    params: MorseRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedMorseScene:
    """Render a Morse word code with four candidate word labels."""

    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    if len(word_options) != 4:
        raise ValueError("Morse word-read scene requires exactly four word options")

    item_bboxes: dict[str, list[float]] = {}
    symbol_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    source_letter_widths = tuple(
        _letter_width(
            [symbol.symbol for symbol in letter.symbols],
            dot_radius=int(params.code_symbol_dot_radius_px),
            dash_width=int(params.code_symbol_dash_width_px),
            symbol_gap=int(params.code_symbol_gap_px),
        )
        for letter in source_code.letters
    )
    source_code_width = _code_width(source_letter_widths, letter_gap=int(params.code_letter_gap_px))
    source_width = min(900, max(420, int(source_code_width + 96)))
    source_bbox = [float(round(0.5 * width - 0.5 * source_width)), 88.0, float(round(0.5 * width + 0.5 * source_width)), 230.0]
    source_items, source_symbols, source_entities = _draw_morse_word_code(
        draw,
        spec=source_code,
        bbox=source_bbox,
        params=params,
        style=style,
        source=True,
        label_position="above",
    )
    item_bboxes.update(source_items)
    symbol_bboxes.update(source_symbols)
    entities.extend(source_entities)

    option_width = int(params.word_option_card_width_px)
    option_height = int(params.word_option_card_height_px)
    gap_x = 64
    gap_y = 58
    grid_left = float(round(0.5 * width - 0.5 * ((2 * option_width) + gap_x)))
    grid_top = 360.0
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
            font_size=int(params.option_word_font_size_px),
            label_position="above",
        )
        item_bboxes[option_id] = _rounded_bbox(bbox)
        entities.append(entity)

    return RenderedMorseScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        symbol_bboxes=symbol_bboxes,
        scene_bbox_px=[40.0, 48.0, float(width - 40), float(height - 48)],
        style_metadata={"renderer": "morse_code_v1", "layout": "morse_code_to_word_options"},
    )


def render_word_morse_match_scene(
    image: Image.Image,
    *,
    source_word: str,
    code_options: Sequence[MorseWordSpec],
    params: MorseRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedMorseScene:
    """Render a source word with four candidate Morse-code word cards."""

    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    if len(code_options) != 4:
        raise ValueError("Morse word-match scene requires exactly four Morse options")

    item_bboxes: dict[str, list[float]] = {}
    symbol_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    source_width = min(520, max(300, int((len(source_word) * 46) + 132)))
    source_bbox = [float(round(0.5 * width - 0.5 * source_width)), 56.0, float(round(0.5 * width + 0.5 * source_width)), 136.0]
    bbox, entity = _draw_word_card(
        draw,
        item_id="source_word",
        label="",
        word=str(source_word),
        bbox=source_bbox,
        params=params,
        style=style,
        marked=True,
        font_size=int(params.source_word_font_size_px),
        label_position="above",
    )
    item_bboxes["source_word"] = _rounded_bbox(bbox)
    entities.append(entity)

    option_width = int(params.option_card_width_px)
    option_height = int(params.option_card_height_px)
    gap_x = 60
    gap_y = 54
    grid_left = float(round(0.5 * width - 0.5 * ((2 * option_width) + gap_x)))
    grid_top = 220.0
    for index, spec in enumerate(code_options):
        row = index // 2
        col = index % 2
        left = grid_left + (col * (option_width + gap_x))
        top = grid_top + (row * (option_height + gap_y))
        option_items, option_symbols, option_entities = _draw_morse_word_code(
            draw,
            spec=spec,
            bbox=[left, top, left + option_width, top + option_height],
            params=params,
            style=style,
            source=False,
            label_position="above",
        )
        item_bboxes.update(option_items)
        symbol_bboxes.update(option_symbols)
        entities.extend(option_entities)

    return RenderedMorseScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        symbol_bboxes=symbol_bboxes,
        scene_bbox_px=[40.0, 32.0, float(width - 40), float(height - 38)],
        style_metadata={"renderer": "morse_code_v1", "layout": "word_to_morse_code_options"},
    )


__all__ = ["render_morse_word_read_scene", "render_word_morse_match_scene"]
