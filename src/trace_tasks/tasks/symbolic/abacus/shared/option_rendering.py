"""Renderer for compact abacus option cards."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle
from ....shared.text_rendering import load_font

from .layout import bead_bbox, option_card_bboxes, rounded_bbox
from .rules import ABACUS_COLUMN_ROLES, digit_active_counts, digits_for_abacus_value
from .state import AbacusOptionPanelRenderParams, AbacusOptionSpec, RenderedAbacusOptionPanelScene
from .styles import variant_colors


def _draw_bead(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    fill: Sequence[int],
    outline: Sequence[int],
    shadow: Sequence[int],
) -> None:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    draw.ellipse((x0 + 3.0, y0 + 4.0, x1 + 3.0, y1 + 4.0), fill=tuple(int(value) for value in shadow))
    draw.ellipse((x0, y0, x1, y1), fill=tuple(int(value) for value in fill), outline=tuple(int(value) for value in outline), width=2)
    draw.arc((x0 + 8.0, y0 + 6.0, x1 - 8.0, y1 - 6.0), start=200, end=340, fill=tuple(int(value) for value in outline), width=1)


def _draw_compact_abacus_option(
    draw: ImageDraw.ImageDraw,
    *,
    card_bbox: Sequence[float],
    option: AbacusOptionSpec,
    params: AbacusOptionPanelRenderParams,
    colors: Mapping[str, tuple[int, int, int]],
) -> tuple[list[float], list[dict[str, Any]], dict[str, list[float]]]:
    """Draw one option card and return its abacus bbox, entities, and bead boxes."""

    label = str(option.label)
    value = int(option.value)
    x0, y0, x1, y1 = (float(value_) for value_ in card_bbox)
    card_bbox_out = rounded_bbox((x0, y0, x1, y1))
    shadow_bbox = (x0 + 4.0, y0 + 5.0, x1 + 4.0, y1 + 5.0)
    draw_rounded_rect(
        draw,
        shadow_bbox,
        radius=int(params.option_card_corner_radius_px),
        fill=colors["shadow"],
        outline=colors["shadow"],
        width=1,
    )
    draw_rounded_rect(
        draw,
        (x0, y0, x1, y1),
        radius=int(params.option_card_corner_radius_px),
        fill=colors["panel_fill"],
        outline=colors["panel_outline"],
        width=2,
    )

    badge_bbox = (x0 + 13.0, y0 + 13.0, x0 + 49.0, y0 + 49.0)
    draw.rounded_rectangle(badge_bbox, radius=10, fill=colors["frame"], outline=colors["frame"], width=1)
    option_label_font = load_font(int(params.option_label_font_size_px), bold=True)
    badge_text_bbox = draw_centered_text(
        draw,
        text=label,
        center=(0.5 * (badge_bbox[0] + badge_bbox[2]), 0.5 * (badge_bbox[1] + badge_bbox[3])),
        font=option_label_font,
        fill=colors["panel_fill"],
        stroke_fill=colors["frame"],
        stroke_width=0,
    )

    frame_bbox = [x0 + 58.0, y0 + 58.0, x1 - 32.0, y1 - 44.0]
    frame_left, frame_top, frame_right, frame_bottom = (float(value_) for value_ in frame_bbox)
    draw_rounded_rect(
        draw,
        tuple(frame_bbox),
        radius=12,
        fill=colors["panel_fill"],
        outline=colors["frame"],
        width=4,
    )
    beam_y = float(frame_top + (0.36 * (frame_bottom - frame_top)))
    beam_bbox = (
        frame_left + 5.0,
        beam_y - (0.5 * int(params.option_beam_height_px)),
        frame_right - 5.0,
        beam_y + (0.5 * int(params.option_beam_height_px)),
    )
    draw.rounded_rectangle(beam_bbox, radius=4, fill=colors["beam"], outline=colors["frame"], width=1)

    rod_top = frame_top + 11.0
    rod_bottom = frame_bottom - 11.0
    rod_xs = (
        frame_left + 54.0,
        0.5 * (frame_left + frame_right),
        frame_right - 54.0,
    )
    upper_inactive_y = beam_y - 44.0
    upper_active_y = beam_y - 20.0
    lower_active_start_y = beam_y + 22.0
    lower_spacing = 24.0
    lower_inactive_bottom_y = rod_bottom - 6.0
    place_label_y = frame_bottom + 22.0
    place_font = load_font(int(params.option_place_label_font_size_px), bold=True)
    bead_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = [
        {
            "item_id": f"option_{label}_card",
            "entity_type": "abacus_option_card",
            "option_label": label,
            "value": value,
            "is_correct": bool(option.is_correct),
            "bbox_px": list(card_bbox_out),
        },
        {
            "item_id": f"option_{label}_abacus_frame",
            "entity_type": "abacus_frame",
            "option_label": label,
            "value": value,
            "bbox_px": rounded_bbox(frame_bbox),
        },
        {
            "item_id": f"option_{label}_label_badge",
            "entity_type": "option_label_badge",
            "option_label": label,
            "bbox_px": rounded_bbox(badge_bbox),
            "text_bbox_px": list(badge_text_bbox),
        },
    ]

    for role, place_label, digit, cx in zip(ABACUS_COLUMN_ROLES, ("100", "10", "1"), digits_for_abacus_value(value), rod_xs):
        draw.line((cx, rod_top, cx, rod_bottom), fill=colors["rod"], width=int(params.option_rod_width_px))
        draw_centered_text(
            draw,
            text=str(place_label),
            center=(float(cx), float(place_label_y)),
            font=place_font,
            fill=colors["label"],
            stroke_fill=colors["panel_fill"],
            stroke_width=1,
        )
        upper_active, lower_count = digit_active_counts(int(digit))
        upper_id = f"option_{label}_column_{role}_upper"
        upper_bbox = bead_bbox(
            cx,
            upper_active_y if upper_active else upper_inactive_y,
            width=int(params.option_bead_width_px),
            height=int(params.option_bead_height_px),
        )
        _draw_bead(
            draw,
            bbox=upper_bbox,
            fill=colors["active_bead"],
            outline=colors["bead_outline"],
            shadow=colors["shadow"],
        )
        bead_bboxes[str(upper_id)] = list(upper_bbox)

        for bead_index in range(4):
            is_active = int(bead_index) < int(lower_count)
            if is_active:
                bead_y = float(lower_active_start_y + (bead_index * lower_spacing))
            else:
                remaining_index = int(bead_index - lower_count)
                inactive_count = int(4 - lower_count)
                bead_y = float(lower_inactive_bottom_y - ((inactive_count - remaining_index - 1) * lower_spacing))
            bead_id = f"option_{label}_column_{role}_lower_{bead_index + 1}"
            bead_box = bead_bbox(
                cx,
                bead_y,
                width=int(params.option_bead_width_px),
                height=int(params.option_bead_height_px),
            )
            _draw_bead(
                draw,
                bbox=bead_box,
                fill=colors["active_bead"],
                outline=colors["bead_outline"],
                shadow=colors["shadow"],
            )
            bead_bboxes[str(bead_id)] = list(bead_box)

        entities.append(
            {
                "item_id": f"option_{label}_column_{role}",
                "entity_type": "abacus_column",
                "option_label": label,
                "place_label": str(place_label),
                "place_value": {"hundreds": 100, "tens": 10, "ones": 1}[str(role)],
                "digit": int(digit),
                "bbox_px": rounded_bbox((cx - 8.0, rod_top, cx + 8.0, rod_bottom)),
                "active_upper_bead": bool(upper_active),
                "active_lower_bead_count": int(lower_count),
            }
        )

    for bead_id, bead_box in bead_bboxes.items():
        entities.append(
            {
                "item_id": str(bead_id),
                "entity_type": "abacus_bead",
                "option_label": label,
                "bbox_px": list(bead_box),
            }
        )
    return rounded_bbox(frame_bbox), entities, bead_bboxes


def render_abacus_option_panel_scene(
    image: Image.Image,
    *,
    options: Sequence[AbacusOptionSpec],
    correct_label: str,
    params: AbacusOptionPanelRenderParams,
    scene_variant: str,
    style: SymbolicSceneStyle,
) -> RenderedAbacusOptionPanelScene:
    """Render six labeled compact abacus options for target-value matching."""

    if len(options) != 6:
        raise ValueError("abacus option panel requires exactly six options")
    labels = tuple(str(option.label) for option in options)
    if len(set(labels)) != len(labels):
        raise ValueError("abacus option labels must be unique")
    if str(correct_label) not in set(labels):
        raise ValueError("correct label must be one of the visible options")

    draw = ImageDraw.Draw(image)
    colors = variant_colors(str(scene_variant), style)
    card_bboxes = option_card_bboxes(option_labels=labels, params=params)
    entities: list[dict[str, Any]] = []
    item_bboxes: dict[str, list[float]] = {}
    option_card_boxes: dict[str, list[float]] = {}
    option_abacus_boxes: dict[str, list[float]] = {}
    option_values_by_label: dict[str, int] = {}

    for option in options:
        label = str(option.label)
        option_card_boxes[label] = list(card_bboxes[label])
        option_values_by_label[label] = int(option.value)
        abacus_box, option_entities, bead_bboxes = _draw_compact_abacus_option(
            draw,
            card_bbox=card_bboxes[label],
            option=option,
            params=params,
            colors=colors,
        )
        option_abacus_boxes[label] = list(abacus_box)
        item_bboxes[f"option_{label}_card"] = list(card_bboxes[label])
        item_bboxes[f"option_{label}_abacus_frame"] = list(abacus_box)
        item_bboxes.update({str(key): list(value) for key, value in bead_bboxes.items()})
        entities.extend(option_entities)

    selected_card_bbox = list(option_card_boxes[str(correct_label)])
    selected_abacus_bbox = list(option_abacus_boxes[str(correct_label)])
    scene_bbox = rounded_bbox(
        (
            min(float(bbox[0]) for bbox in option_card_boxes.values()) - 8.0,
            min(float(bbox[1]) for bbox in option_card_boxes.values()) - 8.0,
            max(float(bbox[2]) for bbox in option_card_boxes.values()) + 12.0,
            max(float(bbox[3]) for bbox in option_card_boxes.values()) + 12.0,
        )
    )
    return RenderedAbacusOptionPanelScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=dict(item_bboxes),
        option_card_bboxes={str(key): list(value) for key, value in option_card_boxes.items()},
        option_abacus_bboxes={str(key): list(value) for key, value in option_abacus_boxes.items()},
        option_values_by_label=dict(option_values_by_label),
        selected_option_card_bbox=list(selected_card_bbox),
        selected_option_abacus_bbox=list(selected_abacus_bbox),
        scene_bbox_px=list(scene_bbox),
        style_metadata={
            "renderer": "abacus_option_panel_v1",
            "scene_variant": str(scene_variant),
            "layout": "six_option_cards_3x2",
            "option_labels": [str(label) for label in labels],
            "active_inactive_bead_color_shared": tuple(colors["active_bead"]) == tuple(colors["inactive_bead"]),
            "active_bead_rgb": [int(value) for value in colors["active_bead"]],
            "inactive_bead_rgb": [int(value) for value in colors["inactive_bead"]],
            "upper_bead_value": 5,
            "lower_bead_value": 1,
        },
    )


__all__ = [
    "render_abacus_option_panel_scene",
]
