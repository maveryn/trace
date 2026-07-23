"""Renderer for symbolic chemical-equation scenes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from ....shared.bbox_projection import bbox_union, round_bbox
from ....shared.text_rendering import load_font
from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle
from .state import (
    ChemicalEquationDataset,
    ChemicalEquationRenderParams,
    ChemicalTermSpec,
    RenderedChemicalEquationScene,
)
from .styles import ELEMENT_COLORS, chemical_variant_palette


def _tuple_rgb(value: Sequence[int]) -> tuple[int, int, int]:
    return tuple(int(item) for item in value[:3])  # type: ignore[return-value]


def _rounded(values: Sequence[float]) -> tuple[float, float, float, float]:
    rounded = round_bbox(values)
    return (float(rounded[0]), float(rounded[1]), float(rounded[2]), float(rounded[3]))


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    return (
        0.5 * (float(bbox[0]) + float(bbox[2])),
        0.5 * (float(bbox[1]) + float(bbox[3])),
    )


def _contrast_text(fill: Sequence[int]) -> tuple[int, int, int]:
    r, g, b = [int(value) for value in fill[:3]]
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return (28, 32, 36) if luminance > 155 else (255, 255, 255)


def _draw_panel_background(
    draw: ImageDraw.ImageDraw,
    *,
    params: ChemicalEquationRenderParams,
    palette: Mapping[str, Sequence[int]],
    scene_variant: str,
) -> tuple[float, float, float, float]:
    panel_bbox = _rounded(
        (
            params.panel_left_px,
            params.panel_top_px,
            params.panel_left_px + params.panel_width_px,
            params.panel_top_px + params.panel_height_px,
        )
    )
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=18,
        fill=palette["panel_fill"],
        outline=palette["panel_border"],
        width=2,
    )
    if str(scene_variant) in {"worksheet", "notebook_scan"}:
        line_rgb = tuple(int(value) for value in palette["line"])
        y = float(panel_bbox[1]) + 46.0
        while y < float(panel_bbox[3]) - 24.0:
            draw.line(
                (panel_bbox[0] + 24.0, y, panel_bbox[2] - 24.0, y),
                fill=line_rgb,
                width=1,
            )
            y += 38.0 if str(scene_variant) == "notebook_scan" else 54.0
    return panel_bbox


def _expand_atoms(term: ChemicalTermSpec) -> list[str]:
    atoms: list[str] = []
    for element in term.element_order:
        atoms.extend([str(element)] * int(term.atom_counts[str(element)]))
    return atoms


def _draw_coefficient_slot(
    draw: ImageDraw.ImageDraw,
    *,
    term: ChemicalTermSpec,
    bbox: Sequence[float],
    params: ChemicalEquationRenderParams,
    palette: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    bbox_tuple = _rounded(bbox)
    outline = palette["accent"] if term.hidden_coefficient else palette["panel_border"]
    draw_rounded_rect(
        draw,
        bbox_tuple,
        radius=8,
        fill=palette["slot_fill"],
        outline=outline,
        width=2,
    )
    text = "?" if term.hidden_coefficient else str(int(term.coefficient))
    font = load_font(int(params.coefficient_font_size_px), bold=False)
    draw_centered_text(
        draw,
        text=text,
        center=_center(bbox_tuple),
        font=font,
        fill=palette["accent"] if term.hidden_coefficient else palette["text"],
        stroke_fill=palette["slot_fill"],
        stroke_width=0,
    )
    return {
        "entity_id": str(term.coefficient_slot_id),
        "entity_type": "chemical_coefficient_slot",
        "term_index": int(term.term_index),
        "side": str(term.side),
        "formula": str(term.formula),
        "coefficient": None if term.hidden_coefficient else int(term.coefficient),
        "hidden": bool(term.hidden_coefficient),
        "bbox_px": list(bbox_tuple),
    }


def _draw_atom_chips(
    draw: ImageDraw.ImageDraw,
    *,
    term: ChemicalTermSpec,
    card_bbox: Sequence[float],
    params: ChemicalEquationRenderParams,
    palette: Mapping[str, Sequence[int]],
) -> tuple[dict[str, tuple[float, float, float, float]], list[dict[str, Any]]]:
    """Draw visible atom chips while preserving one bbox per counted atom."""

    atoms = _expand_atoms(term)
    chip_bboxes: dict[str, tuple[float, float, float, float]] = {}
    entities: list[dict[str, Any]] = []
    if not atoms:
        return chip_bboxes, entities

    card_left, card_top, card_right, card_bottom = [float(value) for value in card_bbox]
    diameter = int(params.atom_chip_diameter_px)
    columns = min(4, len(atoms))
    rows = int(math.ceil(float(len(atoms)) / float(columns)))
    gap = 6.0 if len(atoms) <= 8 else 4.0
    total_w = float(columns * diameter) + float(columns - 1) * gap
    total_h = float(rows * diameter) + float(rows - 1) * gap
    start_x = card_left + ((card_right - card_left) - total_w) / 2.0
    start_y = card_top + ((card_bottom - card_top) - total_h) / 2.0
    atom_font = load_font(int(params.atom_font_size_px), bold=False)
    for atom_index, element in enumerate(atoms):
        row = atom_index // columns
        col = atom_index % columns
        x0 = start_x + float(col) * (float(diameter) + gap)
        y0 = start_y + float(row) * (float(diameter) + gap)
        bbox = _rounded((x0, y0, x0 + diameter, y0 + diameter))
        fill = ELEMENT_COLORS.get(str(element), (192, 192, 192))
        outline = (40, 48, 54)
        draw.ellipse(bbox, fill=fill, outline=outline, width=1)
        draw_centered_text(
            draw,
            text=str(element),
            center=_center(bbox),
            font=atom_font,
            fill=_contrast_text(fill),
            stroke_fill=fill,
            stroke_width=0,
        )
        item_id = f"{term.item_id}_atom_{atom_index + 1}"
        chip_bboxes[item_id] = bbox
        entities.append(
            {
                "entity_id": item_id,
                "entity_type": "chemical_atom_chip",
                "term_id": str(term.item_id),
                "term_index": int(term.term_index),
                "element": str(element),
                "bbox_px": list(bbox),
            }
        )
    return chip_bboxes, entities


def _draw_molecule_card(
    draw: ImageDraw.ImageDraw,
    *,
    term: ChemicalTermSpec,
    bbox: Sequence[float],
    params: ChemicalEquationRenderParams,
    palette: Mapping[str, Sequence[int]],
) -> tuple[dict[str, tuple[float, float, float, float]], list[dict[str, Any]]]:
    """Draw one molecule card as repeated atom chips inside one term bbox."""

    bbox_tuple = _rounded(bbox)
    draw_rounded_rect(
        draw,
        bbox_tuple,
        radius=int(params.card_corner_radius_px),
        fill=palette["card_fill"],
        outline=palette["panel_border"],
        width=int(params.card_border_width_px),
    )
    atom_bboxes, atom_entities = _draw_atom_chips(
        draw,
        term=term,
        card_bbox=bbox_tuple,
        params=params,
        palette=palette,
    )
    entity = {
        "entity_id": str(term.molecule_card_id),
        "entity_type": "chemical_molecule_card",
        "term_id": str(term.item_id),
        "term_index": int(term.term_index),
        "side": str(term.side),
        "side_index": int(term.side_index),
        "formula": str(term.formula),
        "coefficient": int(term.coefficient),
        "atom_counts": {str(key): int(value) for key, value in term.atom_counts.items()},
        "bbox_px": list(bbox_tuple),
    }
    return atom_bboxes, [entity, *atom_entities]


def _operator_after_term(dataset: ChemicalEquationDataset, term: ChemicalTermSpec) -> str | None:
    left_last = len(dataset.reaction.left_formulas) - 1
    right_last = len(dataset.terms) - 1
    if term.term_index < left_last:
        return "+"
    if term.term_index == left_last:
        return "->"
    if term.term_index < right_last:
        return "+"
    return None


def _draw_equation(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: ChemicalEquationDataset,
    params: ChemicalEquationRenderParams,
    palette: Mapping[str, Sequence[int]],
    top_y: float,
    panel_bbox: Sequence[float],
) -> tuple[
    dict[str, tuple[float, float, float, float]],
    dict[str, tuple[float, float, float, float]],
    dict[str, tuple[float, float, float, float]],
    list[dict[str, Any]],
]:
    """Draw the equation row left-to-right while keeping term/operator bboxes stable."""

    item_bboxes: dict[str, tuple[float, float, float, float]] = {}
    coefficient_bboxes: dict[str, tuple[float, float, float, float]] = {}
    molecule_bboxes: dict[str, tuple[float, float, float, float]] = {}
    atom_bboxes: dict[str, tuple[float, float, float, float]] = {}
    entities: list[dict[str, Any]] = []

    group_width = (
        int(params.coefficient_box_width_px)
        + int(params.term_gap_px)
        + int(params.molecule_card_width_px)
    )
    operator_count = (
        (len(dataset.reaction.left_formulas) - 1)
        + (len(dataset.reaction.right_formulas) - 1)
        + 1
    )
    total_width = (len(dataset.terms) * group_width) + (
        operator_count * int(params.operator_gap_px)
    )
    start_x = (float(panel_bbox[0]) + float(panel_bbox[2]) - float(total_width)) / 2.0
    cursor_x = float(start_x)
    coeff_y = float(top_y) + (
        float(params.molecule_card_height_px) - float(params.coefficient_box_height_px)
    ) / 2.0
    operator_font = load_font(int(params.operator_font_size_px), bold=False)

    for term in dataset.terms:
        coeff_bbox = _rounded(
            (
                cursor_x,
                coeff_y,
                cursor_x + params.coefficient_box_width_px,
                coeff_y + params.coefficient_box_height_px,
            )
        )
        card_x0 = cursor_x + params.coefficient_box_width_px + params.term_gap_px
        card_bbox = _rounded(
            (
                card_x0,
                top_y,
                card_x0 + params.molecule_card_width_px,
                top_y + params.molecule_card_height_px,
            )
        )
        coefficient_entity = _draw_coefficient_slot(
            draw,
            term=term,
            bbox=coeff_bbox,
            params=params,
            palette=palette,
        )
        molecule_atom_bboxes, molecule_entities = _draw_molecule_card(
            draw,
            term=term,
            bbox=card_bbox,
            params=params,
            palette=palette,
        )
        coefficient_bboxes[str(term.coefficient_slot_id)] = coeff_bbox
        molecule_bboxes[str(term.molecule_card_id)] = card_bbox
        item_bboxes[str(term.coefficient_slot_id)] = coeff_bbox
        item_bboxes[str(term.molecule_card_id)] = card_bbox
        item_bboxes[str(term.item_id)] = _rounded(
            bbox_union((coeff_bbox, card_bbox), padding=0.0)
        )
        atom_bboxes.update(molecule_atom_bboxes)
        item_bboxes.update(molecule_atom_bboxes)
        entities.append(coefficient_entity)
        entities.extend(molecule_entities)
        entities.append(
            {
                "entity_id": str(term.item_id),
                "entity_type": "chemical_equation_term",
                "coefficient_slot_id": str(term.coefficient_slot_id),
                "molecule_card_id": str(term.molecule_card_id),
                "term_index": int(term.term_index),
                "side": str(term.side),
                "formula": str(term.formula),
                "coefficient": int(term.coefficient),
                "hidden_coefficient": bool(term.hidden_coefficient),
                "bbox_px": list(item_bboxes[str(term.item_id)]),
            }
        )

        operator = _operator_after_term(dataset, term)
        cursor_x += float(group_width)
        if operator is not None:
            operator_bbox = _rounded(
                (
                    cursor_x,
                    top_y + 58.0,
                    cursor_x + params.operator_gap_px,
                    top_y + 158.0,
                )
            )
            draw_centered_text(
                draw,
                text=str(operator),
                center=_center(operator_bbox),
                font=operator_font,
                fill=palette["accent"],
                stroke_fill=palette["panel_fill"],
                stroke_width=0,
            )
            operator_id = "operator_arrow" if operator == "->" else f"operator_plus_{term.term_index + 1}"
            item_bboxes[operator_id] = operator_bbox
            entities.append(
                {
                    "entity_id": operator_id,
                    "entity_type": "chemical_equation_operator",
                    "operator": str(operator),
                    "after_term_index": int(term.term_index),
                    "bbox_px": list(operator_bbox),
                }
            )
            cursor_x += float(params.operator_gap_px)

    return item_bboxes, coefficient_bboxes, molecule_bboxes, entities


def _draw_options(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: ChemicalEquationDataset,
    params: ChemicalEquationRenderParams,
    palette: Mapping[str, Sequence[int]],
    top_y: float,
    panel_bbox: Sequence[float],
) -> tuple[
    dict[str, tuple[float, float, float, float]],
    list[dict[str, Any]],
]:
    """Draw answer option cards with coefficient tuples as selected-answer witnesses."""

    option_bboxes: dict[str, tuple[float, float, float, float]] = {}
    entities: list[dict[str, Any]] = []
    if not dataset.options:
        return option_bboxes, entities

    gap_x = 34.0
    gap_y = 20.0
    card_w = (float(panel_bbox[2]) - float(panel_bbox[0]) - 120.0 - gap_x) / 2.0
    card_h = 90.0
    start_x = float(panel_bbox[0]) + 60.0
    label_font = load_font(int(params.option_label_font_size_px), bold=False)
    option_font = load_font(int(params.option_text_font_size_px), bold=False)

    for index, option in enumerate(dataset.options):
        row = index // 2
        col = index % 2
        x0 = start_x + float(col) * (card_w + gap_x)
        y0 = float(top_y) + float(row) * (card_h + gap_y)
        bbox = _rounded((x0, y0, x0 + card_w, y0 + card_h))
        draw_rounded_rect(
            draw,
            bbox,
            radius=12,
            fill=palette["option_fill"],
            outline=palette["option_border"],
            width=2,
        )
        label_bbox = (bbox[0] + 18.0, bbox[1] + 18.0, bbox[0] + 66.0, bbox[3] - 18.0)
        draw_centered_text(
            draw,
            text=str(option.label),
            center=_center(label_bbox),
            font=label_font,
            fill=palette["accent"],
            stroke_fill=palette["option_fill"],
            stroke_width=0,
        )
        coeff_text = ", ".join(str(int(value)) for value in option.coefficients)
        coeff_bbox = (bbox[0] + 82.0, bbox[1] + 14.0, bbox[2] - 22.0, bbox[3] - 14.0)
        draw_centered_text(
            draw,
            text=coeff_text,
            center=_center(coeff_bbox),
            font=option_font,
            fill=palette["text"],
            stroke_fill=palette["option_fill"],
            stroke_width=0,
        )
        option_bboxes[str(option.item_id)] = bbox
        entities.append(
            {
                "entity_id": str(option.item_id),
                "entity_type": "chemical_coefficient_option",
                "label": str(option.label),
                "coefficients": [int(value) for value in option.coefficients],
                "balances_equation": bool(option.balances_equation),
                "bbox_px": list(bbox),
            }
        )
    return option_bboxes, entities


def render_chemical_equation_scene(
    image: Image.Image,
    *,
    dataset: ChemicalEquationDataset,
    params: ChemicalEquationRenderParams,
    style: SymbolicSceneStyle,
    scene_variant: str,
) -> RenderedChemicalEquationScene:
    """Render one chemical-equation scene and return projection maps."""

    _ = style
    draw = ImageDraw.Draw(image)
    palette = chemical_variant_palette(str(scene_variant))
    panel_bbox = _draw_panel_background(
        draw,
        params=params,
        palette=palette,
        scene_variant=str(scene_variant),
    )
    equation_top = (
        float(panel_bbox[1]) + 90.0
        if str(dataset.task_kind) == "coefficient_option_selection"
        else float(panel_bbox[1]) + 222.0
    )
    item_bboxes, coefficient_bboxes, molecule_bboxes, equation_entities = _draw_equation(
        draw,
        dataset=dataset,
        params=params,
        palette=palette,
        top_y=float(equation_top),
        panel_bbox=panel_bbox,
    )
    option_bboxes, option_entities = _draw_options(
        draw,
        dataset=dataset,
        params=params,
        palette=palette,
        top_y=float(panel_bbox[1]) + 448.0,
        panel_bbox=panel_bbox,
    )
    item_bboxes.update(option_bboxes)
    entities = [*equation_entities, *option_entities]
    return RenderedChemicalEquationScene(
        image=image,
        entities=tuple(entities),
        scene_bbox_px=panel_bbox,
        item_bboxes=dict(item_bboxes),
        coefficient_slot_bboxes=dict(coefficient_bboxes),
        molecule_card_bboxes=dict(molecule_bboxes),
        option_bboxes=dict(option_bboxes),
        atom_chip_bboxes={
            key: value
            for key, value in item_bboxes.items()
            if "_atom_" in str(key)
        },
        style_metadata={
            "renderer": "chemical_equation_v2",
            "scene_variant": str(scene_variant),
            "element_colors": {
                str(key): [int(value) for value in rgb]
                for key, rgb in sorted(ELEMENT_COLORS.items())
            },
            "palette": {str(key): list(_tuple_rgb(value)) for key, value in palette.items()},
        },
    )


__all__ = ["render_chemical_equation_scene"]
