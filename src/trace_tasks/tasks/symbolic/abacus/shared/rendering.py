"""Renderer for one three-column soroban-style abacus board."""

from __future__ import annotations

from typing import Any, Sequence

from PIL import Image, ImageDraw

from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle
from ....shared.text_rendering import load_font

from .layout import bbox_center, bead_bbox, readout_option_bboxes, rounded_bbox
from .rules import ABACUS_ANNOTATION_KEYS, ABACUS_COLUMN_ROLES, digit_active_counts
from .state import AbacusColumnSpec, AbacusReadoutOptionSpec, AbacusReadoutRenderParams, RenderedAbacusReadoutScene
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


def render_abacus_single_board_scene(
    image: Image.Image,
    *,
    columns: Sequence[AbacusColumnSpec],
    params: AbacusReadoutRenderParams,
    scene_variant: str,
    style: SymbolicSceneStyle,
    options: Sequence[AbacusReadoutOptionSpec] | None = None,
    correct_label: str | None = None,
) -> RenderedAbacusReadoutScene:
    """Render one three-column abacus readout scene."""

    if len(columns) != 3:
        raise ValueError("symbolic abacus readout currently renders exactly three columns")
    if options is not None:
        if len(options) != 6:
            raise ValueError("abacus readout option row requires exactly six options")
        option_labels = tuple(str(option.label) for option in options)
        if len(set(option_labels)) != len(option_labels):
            raise ValueError("abacus readout option labels must be unique")
        if correct_label is None or str(correct_label) not in set(option_labels):
            raise ValueError("correct_label must be one of the visible readout options")
    else:
        option_labels = ()
    draw = ImageDraw.Draw(image)
    colors = variant_colors(str(scene_variant), style)
    width, height = int(params.canvas_width), int(params.canvas_height)
    panel_w = int(params.panel_width_px)
    panel_h = int(params.panel_height_px)
    panel_left = float(round(0.5 * (width - panel_w)))
    panel_top = float(round(0.5 * (height - panel_h) + 20))
    panel_bbox = [panel_left, panel_top, panel_left + panel_w, panel_top + panel_h]
    shadow_bbox = [panel_bbox[0] + 5.0, panel_bbox[1] + 7.0, panel_bbox[2] + 5.0, panel_bbox[3] + 7.0]

    draw_rounded_rect(
        draw,
        tuple(shadow_bbox),
        radius=int(params.panel_corner_radius_px),
        fill=colors["shadow"],
        outline=colors["shadow"],
        width=1,
    )
    draw_rounded_rect(
        draw,
        tuple(panel_bbox),
        radius=int(params.panel_corner_radius_px),
        fill=colors["panel_fill"],
        outline=colors["panel_outline"],
        width=2,
    )
    if str(scene_variant) == "worksheet":
        for offset in range(28, int(panel_h), 36):
            y = float(panel_top + offset)
            draw.line((panel_left + 18.0, y, panel_left + panel_w - 18.0, y), fill=colors["guide"], width=1)

    frame_pad = 48.0
    frame_bbox = [panel_left + frame_pad, panel_top + 62.0, panel_left + panel_w - frame_pad, panel_top + panel_h - 74.0]
    draw_rounded_rect(
        draw,
        tuple(frame_bbox),
        radius=16,
        fill=(0, 0, 0, 0) if image.mode == "RGBA" else colors["panel_fill"],
        outline=colors["frame"],
        width=int(params.frame_width_px),
    )

    # The lower deck is deliberately tall so active beads near the beam remain
    # separated from inactive beads parked at the bottom.
    beam_y = float(panel_top + 0.42 * panel_h)
    beam_bbox = [frame_bbox[0] + 6.0, beam_y - (0.5 * int(params.beam_height_px)), frame_bbox[2] - 6.0, beam_y + (0.5 * int(params.beam_height_px))]
    draw.rounded_rectangle(tuple(beam_bbox), radius=8, fill=colors["beam"], outline=colors["frame"], width=2)

    title_font = load_font(int(params.title_font_size_px), bold=True)
    label_font = load_font(int(params.label_font_size_px), bold=True)
    label_bboxes: dict[str, list[float]] = {}
    title_bbox = draw_centered_text(
        draw,
        text="Abacus",
        center=(0.5 * width, panel_top + 34.0),
        font=title_font,
        fill=colors["label"],
        stroke_fill=colors["panel_fill"],
        stroke_width=2,
    )
    label_bboxes["title"] = list(title_bbox)

    usable_left = frame_bbox[0] + 96.0
    usable_right = frame_bbox[2] - 96.0
    rod_gap = (usable_right - usable_left) / 2.0
    rod_top = frame_bbox[1] + 14.0
    rod_bottom = frame_bbox[3] - 14.0
    upper_inactive_y = beam_y - 116.0
    upper_active_y = beam_y - 34.0
    lower_active_start_y = beam_y + 34.0
    lower_spacing = 38.0
    lower_inactive_bottom_y = rod_bottom - 10.0

    entities: list[dict[str, Any]] = []
    item_bboxes: dict[str, list[float]] = {"abacus_frame": rounded_bbox(frame_bbox), "beam": rounded_bbox(beam_bbox)}
    bead_bboxes: dict[str, list[float]] = {}
    column_bboxes: dict[str, list[float]] = {}
    active_bead_bboxes_by_column: dict[str, list[list[float]]] = {}
    active_bead_points_by_column: dict[str, list[list[float]]] = {}
    active_bead_ids_by_column: dict[str, list[str]] = {}
    option_card_boxes: dict[str, list[float]] = {}
    option_values_by_label: dict[str, int] = {}
    selected_option_card_bbox: list[float] | None = None

    for index, column in enumerate(columns):
        role = str(column.role)
        cx = float(usable_left + (index * rod_gap))
        rod_bbox = rounded_bbox((cx - 12.0, rod_top, cx + 12.0, rod_bottom))
        column_bboxes[role] = list(rod_bbox)
        draw.line((cx, rod_top, cx, rod_bottom), fill=colors["rod"], width=int(params.rod_width_px))
        label_bbox = draw_centered_text(
            draw,
            text=str(column.place_label),
            center=(cx, frame_bbox[3] + 38.0),
            font=label_font,
            fill=colors["label"],
            stroke_fill=colors["panel_fill"],
            stroke_width=2,
        )
        label_bboxes[f"{role}_place_label"] = list(label_bbox)
        upper_active, lower_count = digit_active_counts(int(column.digit))
        active_ids: list[str] = []
        active_bboxes: list[list[float]] = []
        upper_id = f"{column.item_id}_upper"
        upper_bbox = bead_bbox(
            cx,
            upper_active_y if upper_active else upper_inactive_y,
            width=int(params.bead_width_px),
            height=int(params.bead_height_px),
        )
        _draw_bead(
            draw,
            bbox=upper_bbox,
            fill=colors["active_bead"] if upper_active else colors["inactive_bead"],
            outline=colors["bead_outline"],
            shadow=colors["shadow"],
        )
        bead_bboxes[upper_id] = list(upper_bbox)
        if upper_active:
            active_ids.append(str(upper_id))
            active_bboxes.append(list(upper_bbox))

        for bead_index in range(4):
            is_active = int(bead_index) < int(lower_count)
            if is_active:
                bead_y = float(lower_active_start_y + (bead_index * lower_spacing))
            else:
                remaining_index = int(bead_index - lower_count)
                inactive_count = int(4 - lower_count)
                bead_y = float(lower_inactive_bottom_y - ((inactive_count - remaining_index - 1) * lower_spacing))
            bead_id = f"{column.item_id}_lower_{bead_index + 1}"
            bbox = bead_bbox(
                cx,
                bead_y,
                width=int(params.bead_width_px),
                height=int(params.bead_height_px),
            )
            _draw_bead(
                draw,
                bbox=bbox,
                fill=colors["active_bead"] if is_active else colors["inactive_bead"],
                outline=colors["bead_outline"],
                shadow=colors["shadow"],
            )
            bead_bboxes[str(bead_id)] = list(bbox)
            if is_active:
                active_ids.append(str(bead_id))
                active_bboxes.append(list(bbox))

        active_bead_ids_by_column[role] = [str(item) for item in active_ids]
        active_bead_bboxes_by_column[f"{role}_active_beads"] = [list(bbox) for bbox in active_bboxes]
        active_bead_points_by_column[f"{role}_active_beads"] = [bbox_center(bbox) for bbox in active_bboxes]
        entities.append(
            {
                "item_id": str(column.item_id),
                "entity_type": "abacus_column",
                "role": str(role),
                "place_label": str(column.place_label),
                "place_value": int(column.place_value),
                "digit": int(column.digit),
                "bbox_px": list(rod_bbox),
                "active_upper_bead": bool(upper_active),
                "active_lower_bead_count": int(lower_count),
                "active_bead_ids": [str(item) for item in active_ids],
            }
        )

    for bead_id, bbox in bead_bboxes.items():
        item_bboxes[str(bead_id)] = list(bbox)
        role = "active_bead" if any(str(bead_id) in ids for ids in active_bead_ids_by_column.values()) else "inactive_bead"
        entities.append(
            {
                "item_id": str(bead_id),
                "entity_type": "abacus_bead",
                "role": str(role),
                "bbox_px": list(bbox),
            }
        )

    if options is not None:
        option_card_boxes = readout_option_bboxes(option_labels=option_labels, params=params, panel_bbox=panel_bbox)
        label_font = load_font(int(params.readout_option_label_font_size_px), bold=True)
        value_font = load_font(int(params.readout_option_value_font_size_px), bold=True)
        for option in options:
            label = str(option.label)
            value_text = str(option.text)
            value = int(option.value)
            option_bbox = list(option_card_boxes[label])
            x0, y0, x1, y1 = (float(v) for v in option_bbox)
            shadow_bbox = (x0 + 3.0, y0 + 4.0, x1 + 3.0, y1 + 4.0)
            draw_rounded_rect(
                draw,
                shadow_bbox,
                radius=12,
                fill=colors["shadow"],
                outline=colors["shadow"],
                width=1,
            )
            draw_rounded_rect(
                draw,
                (x0, y0, x1, y1),
                radius=12,
                fill=colors["panel_fill"],
                outline=colors["panel_outline"],
                width=2,
            )
            label_bbox = draw_centered_text(
                draw,
                text=f"{label}.",
                center=(x0 + 24.0, 0.5 * (y0 + y1)),
                font=label_font,
                fill=colors["label"],
                stroke_fill=colors["panel_fill"],
                stroke_width=1,
            )
            value_bbox = draw_centered_text(
                draw,
                text=value_text,
                center=(x0 + 80.0, 0.5 * (y0 + y1)),
                font=value_font,
                fill=colors["label"],
                stroke_fill=colors["panel_fill"],
                stroke_width=1,
            )
            item_bboxes[f"option_{label}_card"] = list(option_bbox)
            label_bboxes[f"option_{label}_label"] = list(label_bbox)
            label_bboxes[f"option_{label}_value"] = list(value_bbox)
            option_values_by_label[label] = int(value)
            if str(label) == str(correct_label):
                selected_option_card_bbox = list(option_bbox)
            entities.append(
                {
                    "item_id": f"option_{label}_card",
                    "entity_type": "abacus_readout_option_card",
                    "option_label": str(label),
                    "text": str(value_text),
                    "value": int(value),
                    "is_correct": bool(option.is_correct),
                    "bbox_px": list(option_bbox),
                    "label_bbox_px": list(label_bbox),
                    "value_bbox_px": list(value_bbox),
                }
            )

    scene_bottom = panel_bbox[3] + 12.0
    if option_card_boxes:
        scene_bottom = max(float(scene_bottom), max(float(bbox[3]) for bbox in option_card_boxes.values()) + 12.0)
    scene_bbox = rounded_bbox((panel_bbox[0] - 8.0, panel_bbox[1] - 8.0, panel_bbox[2] + 10.0, scene_bottom))
    return RenderedAbacusReadoutScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        bead_bboxes=bead_bboxes,
        active_bead_bboxes_by_column=active_bead_bboxes_by_column,
        active_bead_points_by_column=active_bead_points_by_column,
        active_bead_ids_by_column=active_bead_ids_by_column,
        column_bboxes=column_bboxes,
        label_bboxes=label_bboxes,
        option_card_bboxes={str(key): list(value) for key, value in option_card_boxes.items()},
        option_values_by_label=dict(option_values_by_label),
        selected_option_card_bbox=(list(selected_option_card_bbox) if selected_option_card_bbox is not None else None),
        scene_bbox_px=list(scene_bbox),
        style_metadata={
            "renderer": "abacus_single_board_v1",
            "scene_variant": str(scene_variant),
            "column_roles": [str(role) for role in ABACUS_COLUMN_ROLES],
            "option_labels": [str(label) for label in option_labels],
            "readout_options_visible": bool(options is not None),
            "annotation_keys": [str(key) for key in ABACUS_ANNOTATION_KEYS],
            "active_beads_touch_center_beam": True,
            "active_inactive_bead_color_shared": tuple(colors["active_bead"]) == tuple(colors["inactive_bead"]),
            "active_bead_rgb": [int(value) for value in colors["active_bead"]],
            "inactive_bead_rgb": [int(value) for value in colors["inactive_bead"]],
            "upper_bead_value": 5,
            "lower_bead_value": 1,
        },
    )


__all__ = [
    "render_abacus_single_board_scene",
]
