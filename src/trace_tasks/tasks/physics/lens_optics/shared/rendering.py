"""Identity-free renderer for the lens-optics scene."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.physics.shared.option_cards import (
    OptionCardRenderResult,
    draw_lettered_option_cards,
)
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text, draw_dashed_line
from trace_tasks.tasks.shared.named_colors import named_color
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .state import OPTION_LETTERS, LensOpticsVisualScenario, RenderedLensOpticsScene


def _bbox(values: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in values]


def _clip_bbox(bbox: Sequence[float], *, width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return _bbox(
        (
            max(0.0, min(float(width), min(x0, x1))),
            max(0.0, min(float(height), min(y0, y1))),
            max(0.0, min(float(width), max(x0, x1))),
            max(0.0, min(float(height), max(y0, y1))),
        )
    )


def _draw_option_cards(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: LensOpticsVisualScenario,
    option_left: float,
    option_top: float,
    card_width: float,
    card_height: float,
    card_gap: float,
    font_family: str,
    style: Any,
) -> OptionCardRenderResult:
    """Draw the visible image-property option cards."""

    letter_font = load_font(22, bold=True, font_family=font_family)
    option_font = load_font(21, bold=True, font_family=font_family)
    label_rgb = tuple(int(v) for v in style.label_rgb)
    return draw_lettered_option_cards(
        draw,
        options=[
            (str(letter), str(scenario.option_text_map[str(letter)]))
            for letter in OPTION_LETTERS
        ],
        option_left=option_left,
        option_top=option_top,
        card_width=card_width,
        card_height=card_height,
        card_gap_x=0.0,
        card_gap_y=card_gap,
        columns=1,
        option_font=option_font,
        letter_font=letter_font,
        text_rgb=label_rgb,
        card_fill_rgb=tuple(int(v) for v in style.panel_alt_fill_rgb),
        card_outline_rgb=tuple(int(v) for v in style.panel_border_rgb),
        label_fill_rgb=tuple(int(v) for v in style.label_fill_rgb),
        label_outline_rgb=tuple(int(v) for v in style.label_border_rgb),
        label_text_rgb=label_rgb,
        card_radius_px=14.0,
        text_align="center",
        text_left_offset_px=76.0,
        text_right_padding_px=16.0,
        line_spacing_px=5.0,
    )


def render_lens_optics_scene(
    *,
    image: Image.Image,
    scenario: LensOpticsVisualScenario,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
) -> RenderedLensOpticsScene:
    """Render one converging-lens diagram and project role-keyed boxes."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    label_rgb = tuple(int(v) for v in style.label_rgb)
    stroke_rgb = tuple(int(v) for v in style.stroke_rgb)
    axis_rgb = tuple(int(v) for v in style.axis_rgb)
    accent_rgb = tuple(int(v) for v in named_color(str(scenario.accent_color_name)))
    title_font = load_font(
        int(render_defaults.get("title_font_size_px", 28)),
        bold=True,
        font_family=font_family,
    )
    label_font = load_font(
        int(render_defaults.get("label_font_size_px", 22)),
        bold=True,
        font_family=font_family,
    )
    small_font = load_font(
        int(render_defaults.get("small_font_size_px", 18)),
        bold=True,
        font_family=font_family,
    )

    diagram = _bbox(
        (
            float(render_defaults.get("diagram_left_px", 54)),
            float(render_defaults.get("diagram_top_px", 54)),
            float(render_defaults.get("diagram_right_px", 760)),
            float(render_defaults.get("diagram_bottom_px", 666)),
        )
    )
    draw.rounded_rectangle(
        tuple(diagram),
        radius=18,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    if str(scenario.scene_variant) in {"paper_grid", "lab_card"}:
        spacing = float(render_defaults.get("grid_spacing_px", 42))
        x = diagram[0] + spacing
        while x < diagram[2]:
            draw.line(
                (x, diagram[1], x, diagram[3]),
                fill=tuple(int(v) for v in style.grid_minor_rgb),
                width=1,
            )
            x += spacing
        y = diagram[1] + spacing
        while y < diagram[3]:
            draw.line(
                (diagram[0], y, diagram[2], y),
                fill=tuple(int(v) for v in style.grid_minor_rgb),
                width=1,
            )
            y += spacing

    draw_centered_text(
        draw,
        text="Converging lens diagram",
        center=((diagram[0] + diagram[2]) * 0.5, diagram[1] + 32.0),
        font=title_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )

    axis_y = float(render_defaults.get("axis_y_px", 370))
    lens_x = float(render_defaults.get("lens_x_px", 430))
    focal = float(scenario.focal_length_px)
    axis_left = diagram[0] + 42.0
    axis_right = diagram[2] - 42.0
    draw.line(
        (axis_left, axis_y, axis_right, axis_y),
        fill=axis_rgb,
        width=int(render_defaults.get("axis_width_px", 4)),
    )
    draw.polygon(
        [
            (axis_right + 14, axis_y),
            (axis_right - 5, axis_y - 8),
            (axis_right - 5, axis_y + 8),
        ],
        fill=axis_rgb,
    )

    lens_height = float(render_defaults.get("lens_height_px", 294))
    lens_width = float(render_defaults.get("lens_width_px", 34))
    lens_bbox = _bbox(
        (
            lens_x - lens_width,
            axis_y - lens_height * 0.5,
            lens_x + lens_width,
            axis_y + lens_height * 0.5,
        )
    )
    draw.ellipse(
        tuple(lens_bbox),
        fill=tuple(int(v) for v in style.panel_alt_fill_rgb),
        outline=accent_rgb,
        width=4,
    )
    draw.line(
        (lens_x, lens_bbox[1] - 12, lens_x, lens_bbox[3] + 12),
        fill=accent_rgb,
        width=3,
    )
    lens_label_bbox = draw_centered_text(
        draw,
        text="lens",
        center=(lens_x, lens_bbox[1] - 28),
        font=small_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    lens_annotation_bbox = bbox_union_many(lens_bbox, lens_label_bbox, padding=3.0)

    focal_boxes: List[List[float]] = []
    for side_sign in (-1, 1):
        for factor, label in ((1.0, "F"), (2.0, "2F")):
            mark_x = lens_x + side_sign * factor * focal
            tick = _bbox((mark_x - 3.0, axis_y - 14.0, mark_x + 3.0, axis_y + 14.0))
            draw.line((mark_x, axis_y - 14.0, mark_x, axis_y + 14.0), fill=axis_rgb, width=3)
            text_bbox = draw_centered_text(
                draw,
                text=str(label),
                center=(mark_x, axis_y + 34.0),
                font=small_font,
                fill=label_rgb,
                stroke_fill=resolve_text_stroke_fill(label_rgb),
                stroke_width=1,
            )
            focal_boxes.append(bbox_union_many(tick, text_bbox, padding=3.0))
    focal_marks_bbox = bbox_union_many(*focal_boxes, padding=5.0)

    object_x = lens_x - float(scenario.object_x_factor) * focal
    object_height = float(render_defaults.get("object_arrow_height_px", 122))
    object_base = (object_x, axis_y)
    object_tip = (object_x, axis_y - object_height)
    draw_arrow(
        draw,
        start=object_base,
        end=object_tip,
        fill=stroke_rgb,
        width=int(render_defaults.get("object_arrow_width_px", 8)),
        head_length_px=float(render_defaults.get("object_arrow_head_length_px", 22)),
        head_width_px=float(render_defaults.get("object_arrow_head_width_px", 22)),
    )
    draw.line((object_x - 24, axis_y, object_x + 24, axis_y), fill=stroke_rgb, width=4)
    object_label_bbox = draw_centered_text(
        draw,
        text="object",
        center=(object_x, object_tip[1] - 26.0),
        font=label_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    object_arrow_bbox = _bbox(
        (object_x - 31.0, object_tip[1] - 11.0, object_x + 31.0, axis_y + 10.0)
    )
    object_annotation_bbox = bbox_union_many(object_arrow_bbox, object_label_bbox, padding=3.0)

    guide_y = object_tip[1]
    draw_dashed_line(
        draw,
        start=(object_x, guide_y),
        end=(lens_x, guide_y),
        fill=accent_rgb,
        width=2,
        dash_px=9,
        gap_px=7,
    )
    draw_dashed_line(
        draw,
        start=(object_x, object_tip[1]),
        end=(lens_x, axis_y),
        fill=accent_rgb,
        width=2,
        dash_px=9,
        gap_px=7,
    )

    option_cards = _draw_option_cards(
        draw,
        scenario=scenario,
        option_left=float(render_defaults.get("option_left_px", 796)),
        option_top=float(render_defaults.get("option_top_px", 82)),
        card_width=float(render_defaults.get("option_card_width_px", 278)),
        card_height=float(render_defaults.get("option_card_height_px", 116)),
        card_gap=float(render_defaults.get("option_card_gap_px", 18)),
        font_family=str(font_family),
        style=style,
    )
    option_bboxes = option_cards.option_bboxes

    annotation = {
        "lens": _clip_bbox(lens_annotation_bbox, width=width, height=height),
        "object_arrow": _clip_bbox(object_annotation_bbox, width=width, height=height),
        "focal_marks": _clip_bbox(focal_marks_bbox, width=width, height=height),
    }
    scene_entities = [
        {
            "entity_id": "lens",
            "entity_type": "converging_lens",
            "bbox_px": list(annotation["lens"]),
            "meta": {"lens_type": "converging"},
        },
        {
            "entity_id": "object_arrow",
            "entity_type": "object_arrow",
            "bbox_px": list(annotation["object_arrow"]),
            "meta": {"object_position_case": str(scenario.object_position_case)},
        },
        {
            "entity_id": "focal_marks",
            "entity_type": "focal_mark_set",
            "bbox_px": list(annotation["focal_marks"]),
            "meta": {"focal_length_px": float(focal)},
        },
    ]
    render_map = {
        "diagram_bbox_px": list(diagram),
        "axis_y_px": round(float(axis_y), 3),
        "lens_x_px": round(float(lens_x), 3),
        "focal_length_px": round(float(focal), 3),
        "object_x_px": round(float(object_x), 3),
        "object_position_case": str(scenario.object_position_case),
        "image_property": str(scenario.image_property),
        "correct_option_letter": str(scenario.correct_option_letter),
        "option_text_map": dict(scenario.option_text_map),
        "option_bboxes_px": {str(letter): list(bbox) for letter, bbox in option_bboxes.items()},
        "option_letter_bboxes_px": {
            str(letter): list(bbox)
            for letter, bbox in option_cards.option_letter_bboxes.items()
        },
        "option_text_bboxes_px": {
            str(letter): list(bbox)
            for letter, bbox in option_cards.option_text_bboxes.items()
        },
        "annotation_keyed_bboxes_px": dict(annotation),
    }
    return RenderedLensOpticsScene(
        image=image,
        annotation_bbox_map=dict(annotation),
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )
