"""Rendering helpers for symbolic clock-display scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import draw_text_centered, load_font
from ....shared.time_artifact_style import TimeArtifactClockTheme
from ....shared.time_format import split_clock_total_minutes, split_clock_total_seconds

from .state import ClockRenderParams, RenderedClockGeometry, RenderedClockScene


def _clock_angles(total_minutes: int, *, total_seconds: int | None = None) -> Tuple[float, float, float | None]:
    """Return hand angles in image coordinates."""

    second: int | None = None
    if total_seconds is not None:
        hour_12, minute, second = split_clock_total_seconds(int(total_seconds))
    else:
        hour_12, minute = split_clock_total_minutes(int(total_minutes))
    second_fraction = (float(second) / 60.0) if second is not None else 0.0
    hour_progress = (float(hour_12 % 12) + (float(minute) / 60.0) + (float(second_fraction) / 60.0)) / 12.0
    minute_progress = (float(minute) + float(second_fraction)) / 60.0
    hour_angle = (360.0 * hour_progress) - 90.0
    minute_angle = (360.0 * minute_progress) - 90.0
    second_angle = ((360.0 * (float(second) / 60.0)) - 90.0) if second is not None else None
    return float(hour_angle), float(minute_angle), (float(second_angle) if second_angle is not None else None)


def _endpoint(center: Tuple[float, float], radius: float, angle_deg: float) -> Tuple[float, float]:
    """Return one endpoint on the analog-clock circle."""

    rad = math.radians(float(angle_deg))
    return (
        float(center[0] + (float(radius) * math.cos(rad))),
        float(center[1] + (float(radius) * math.sin(rad))),
    )


def _segment_bbox(
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    width_px: float,
    padding_px: float,
) -> Tuple[float, float, float, float]:
    """Return a padded axis-aligned bbox for one rendered hand segment."""

    pad = float(max(0.0, float(padding_px)) + (0.5 * float(width_px)))
    return (
        float(min(start[0], end[0]) - pad),
        float(min(start[1], end[1]) - pad),
        float(max(start[0], end[0]) + pad),
        float(max(start[1], end[1]) + pad),
    )


def _merge_bboxes(*bboxes: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Return one bbox enclosing all provided bboxes."""

    if not bboxes:
        raise ValueError("at least one bbox is required")
    return (
        float(min(float(bbox[0]) for bbox in bboxes)),
        float(min(float(bbox[1]) for bbox in bboxes)),
        float(max(float(bbox[2]) for bbox in bboxes)),
        float(max(float(bbox[3]) for bbox in bboxes)),
    )


def _variant_style(scene_variant: str, params: ClockRenderParams, theme: TimeArtifactClockTheme) -> Dict[str, Any]:
    """Return visual toggles for one clock scene variant."""

    if str(scene_variant) == "minimal":
        return {
            "show_minor_ticks": False,
            "face_fill_rgb": tuple(int(v) for v in theme.face_fill_rgb),
            "face_outline_rgb": tuple(int(v) for v in theme.face_outline_rgb),
            "tick_color_rgb": tuple(int(v) for v in theme.tick_color_rgb),
            "numeral_color_rgb": tuple(int(v) for v in theme.numeral_color_rgb),
            "inner_ring_rgb": None,
            "minor_tick_mode": str(theme.minor_tick_mode),
            "bezel_width_px": max(2, int(round(0.7 * float(params.bezel_width_px)))),
        }
    if str(scene_variant) == "outline":
        return {
            "show_minor_ticks": False,
            "face_fill_rgb": None,
            "face_outline_rgb": tuple(int(v) for v in theme.face_outline_rgb),
            "tick_color_rgb": tuple(int(v) for v in theme.tick_color_rgb),
            "numeral_color_rgb": tuple(int(v) for v in theme.numeral_color_rgb),
            "inner_ring_rgb": tuple(int(v) for v in theme.inner_ring_rgb) if theme.inner_ring_rgb is not None else None,
            "minor_tick_mode": str(theme.minor_tick_mode),
            "bezel_width_px": max(3, int(round(0.85 * float(params.bezel_width_px)))),
        }
    return {
        "show_minor_ticks": True,
        "face_fill_rgb": tuple(int(v) for v in theme.face_fill_rgb),
        "face_outline_rgb": tuple(int(v) for v in theme.face_outline_rgb),
        "tick_color_rgb": tuple(int(v) for v in theme.tick_color_rgb),
        "numeral_color_rgb": tuple(int(v) for v in theme.numeral_color_rgb),
        "inner_ring_rgb": tuple(int(v) for v in theme.inner_ring_rgb) if theme.inner_ring_rgb is not None else None,
        "minor_tick_mode": str(theme.minor_tick_mode),
        "bezel_width_px": int(params.bezel_width_px),
    }


def draw_clock_geometry(
    image: Image.Image,
    *,
    center_px: Tuple[float, float],
    face_radius_px: float,
    scene_variant: str,
    shown_total_minutes: int,
    shown_total_seconds: int | None = None,
    show_second_hand: bool = False,
    render_params: ClockRenderParams,
    visual_theme: TimeArtifactClockTheme,
    entity_prefix: str = "",
    extra_face_attrs: Dict[str, Any] | None = None,
    alarm_hour_12: int | None = None,
    alarm_hand_color_rgb: Tuple[int, int, int] = (210, 40, 40),
    alarm_hand_width_px: int | None = None,
    force_show_minor_ticks: bool = False,
) -> RenderedClockGeometry:
    """Draw one analog clock into an existing image and return its geometry."""

    draw = ImageDraw.Draw(image)
    center = (float(center_px[0]), float(center_px[1]))
    face_radius = float(face_radius_px)
    face_bbox = (
        float(center[0] - face_radius),
        float(center[1] - face_radius),
        float(center[0] + face_radius),
        float(center[1] + face_radius),
    )
    style = _variant_style(str(scene_variant), render_params, visual_theme)
    if bool(force_show_minor_ticks):
        style["show_minor_ticks"] = True

    if style["face_fill_rgb"] is not None:
        draw.ellipse(
            face_bbox,
            fill=tuple(int(v) for v in style["face_fill_rgb"]),
            outline=tuple(int(v) for v in style["face_outline_rgb"]),
            width=int(style["bezel_width_px"]),
        )
    else:
        draw.ellipse(
            face_bbox,
            outline=tuple(int(v) for v in style["face_outline_rgb"]),
            width=int(style["bezel_width_px"]),
        )
    if style["inner_ring_rgb"] is not None:
        inner_inset = float(max(6, int(render_params.inner_ring_inset_px)))
        inner_bbox = (
            float(face_bbox[0] + inner_inset),
            float(face_bbox[1] + inner_inset),
            float(face_bbox[2] - inner_inset),
            float(face_bbox[3] - inner_inset),
        )
        draw.ellipse(
            inner_bbox,
            outline=tuple(int(v) for v in style["inner_ring_rgb"]),
            width=int(render_params.inner_ring_width_px),
        )

    major_tick_inner = float(face_radius - render_params.major_tick_length_px)
    minor_tick_inner = float(face_radius - render_params.minor_tick_length_px)
    numeral_radius = float(face_radius - max(36, int(round(0.18 * float(face_radius)))))
    numeral_font = load_font(int(render_params.numeral_font_size_px), bold=False)

    for tick_index in range(60):
        angle_deg = (6.0 * float(tick_index)) - 90.0
        outer = _endpoint(center, face_radius - 4.0, angle_deg)
        is_major = (tick_index % 5) == 0
        if (not is_major) and (not bool(style["show_minor_ticks"])):
            continue
        if (not is_major) and str(style["minor_tick_mode"]) == "dot":
            dot_center = _endpoint(center, face_radius - 8.0, angle_deg)
            dot_radius = float(max(1, int(render_params.minor_tick_dot_radius_px)))
            draw.ellipse(
                (
                    float(dot_center[0] - dot_radius),
                    float(dot_center[1] - dot_radius),
                    float(dot_center[0] + dot_radius),
                    float(dot_center[1] + dot_radius),
                ),
                fill=tuple(int(v) for v in style["tick_color_rgb"]),
            )
            continue
        inner_radius = major_tick_inner if is_major else minor_tick_inner
        inner = _endpoint(center, inner_radius, angle_deg)
        draw.line(
            [inner, outer],
            fill=tuple(int(v) for v in style["tick_color_rgb"]),
            width=int(render_params.major_tick_width_px if is_major else render_params.minor_tick_width_px),
        )

    for numeral in range(1, 13):
        angle_deg = (30.0 * float(numeral)) - 90.0
        numeral_center = _endpoint(center, numeral_radius, angle_deg)
        draw_text_centered(
            draw,
            text=str(int(numeral)),
            center=numeral_center,
            font=numeral_font,
            fill=tuple(int(v) for v in style["numeral_color_rgb"]),
        )

    hour_angle, minute_angle, second_angle = _clock_angles(
        int(shown_total_minutes),
        total_seconds=(int(shown_total_seconds) if bool(show_second_hand) else None),
    )
    alarm_angle: float | None = None
    alarm_tip: Tuple[float, float] | None = None
    alarm_hand_bbox: Tuple[float, float, float, float] | None = None
    if alarm_hour_12 is not None:
        alarm_hour = int(alarm_hour_12)
        if not 1 <= alarm_hour <= 12:
            raise ValueError("alarm_hour_12 must be in 1..12")
        alarm_angle = (30.0 * float(alarm_hour % 12)) - 90.0
        alarm_tip = _endpoint(center, 0.68 * face_radius, float(alarm_angle))
        alarm_width = int(alarm_hand_width_px) if alarm_hand_width_px is not None else max(4, int(round(0.55 * float(render_params.hour_hand_width_px))))
        alarm_color = tuple(int(v) for v in alarm_hand_color_rgb)
        draw.line(
            [center, alarm_tip],
            fill=alarm_color,
            width=int(alarm_width),
        )
        tip_radius = float(max(3, int(round(0.55 * float(alarm_width)))))
        tip_bbox = (
            float(alarm_tip[0] - tip_radius),
            float(alarm_tip[1] - tip_radius),
            float(alarm_tip[0] + tip_radius),
            float(alarm_tip[1] + tip_radius),
        )
        draw.ellipse(tip_bbox, fill=alarm_color)
        alarm_hand_bbox = _merge_bboxes(
            _segment_bbox(
                center,
                alarm_tip,
                width_px=float(alarm_width),
                padding_px=float(render_params.hand_bbox_padding_px),
            ),
            tip_bbox,
        )
    hour_tip = _endpoint(center, 0.55 * face_radius, hour_angle)
    minute_tip = _endpoint(center, 0.82 * face_radius, minute_angle)
    second_tip = _endpoint(center, 0.88 * face_radius, float(second_angle)) if second_angle is not None else None
    second_tail = _endpoint(center, -0.12 * face_radius, float(second_angle)) if second_angle is not None else None

    draw.line(
        [center, hour_tip],
        fill=tuple(int(v) for v in visual_theme.hour_hand_color_rgb),
        width=int(render_params.hour_hand_width_px),
    )
    draw.line(
        [center, minute_tip],
        fill=tuple(int(v) for v in visual_theme.minute_hand_color_rgb),
        width=int(render_params.minute_hand_width_px),
    )
    if second_tip is not None and second_tail is not None:
        draw.line(
            [second_tail, second_tip],
            fill=tuple(int(v) for v in visual_theme.second_hand_color_rgb),
            width=int(render_params.second_hand_width_px),
        )
    draw.ellipse(
        (
            float(center[0] - render_params.center_dot_radius_px),
            float(center[1] - render_params.center_dot_radius_px),
            float(center[0] + render_params.center_dot_radius_px),
            float(center[1] + render_params.center_dot_radius_px),
        ),
        fill=tuple(int(v) for v in visual_theme.center_dot_color_rgb),
    )

    hour_hand_bbox = _segment_bbox(
        center,
        hour_tip,
        width_px=float(render_params.hour_hand_width_px),
        padding_px=float(render_params.hand_bbox_padding_px),
    )
    minute_hand_bbox = _segment_bbox(
        center,
        minute_tip,
        width_px=float(render_params.minute_hand_width_px),
        padding_px=float(render_params.hand_bbox_padding_px),
    )
    second_hand_bbox = (
        _segment_bbox(
            second_tail if second_tail is not None else center,
            second_tip if second_tip is not None else center,
            width_px=float(render_params.second_hand_width_px),
            padding_px=float(render_params.hand_bbox_padding_px),
        )
        if second_tip is not None and second_tail is not None
        else None
    )

    prefix = str(entity_prefix).strip()
    if prefix:
        prefix = f"{prefix}_"
    face_attrs: Dict[str, Any] = {
        "scene_variant": str(scene_variant),
        "shown_total_minutes": int(shown_total_minutes),
        "accent_color_name": str(visual_theme.accent_color_name),
        "style_variant": str(visual_theme.style_variant),
    }
    if extra_face_attrs:
        face_attrs.update({str(key): value for key, value in extra_face_attrs.items()})

    entities: List[Dict[str, Any]] = [
        {
            "entity_id": f"{prefix}clock_face",
            "entity_kind": "clock_face",
            "bbox_px": [float(value) for value in face_bbox],
            "attrs": dict(face_attrs),
        },
        {
            "entity_id": f"{prefix}hour_hand",
            "entity_kind": "clock_hand",
            "bbox_px": [float(value) for value in hour_hand_bbox],
            "attrs": {
                "hand_kind": "hour",
                "tip_px": [float(hour_tip[0]), float(hour_tip[1])],
                "center_px": [float(center[0]), float(center[1])],
                "angle_deg": float(hour_angle),
            },
        },
        {
            "entity_id": f"{prefix}minute_hand",
            "entity_kind": "clock_hand",
            "bbox_px": [float(value) for value in minute_hand_bbox],
            "attrs": {
                "hand_kind": "minute",
                "tip_px": [float(minute_tip[0]), float(minute_tip[1])],
                "center_px": [float(center[0]), float(center[1])],
                "angle_deg": float(minute_angle),
            },
        },
    ]
    if second_hand_bbox is not None and second_tip is not None:
        entities.append(
            {
                "entity_id": f"{prefix}second_hand",
                "entity_kind": "clock_hand",
                "bbox_px": [float(value) for value in second_hand_bbox],
                "attrs": {
                    "hand_kind": "second",
                    "tip_px": [float(second_tip[0]), float(second_tip[1])],
                    "center_px": [float(center[0]), float(center[1])],
                    "angle_deg": float(second_angle),
                },
            }
        )
    if alarm_hand_bbox is not None and alarm_tip is not None:
        entities.append(
            {
                "entity_id": f"{prefix}alarm_hand",
                "entity_kind": "clock_hand",
                "bbox_px": [float(value) for value in alarm_hand_bbox],
                "attrs": {
                    "hand_kind": "alarm",
                    "scale": "hour",
                    "alarm_hour": int(alarm_hour_12),
                    "alarm_minute": 0,
                    "semantic_color": "red",
                    "color_rgb": [int(value) for value in alarm_hand_color_rgb],
                    "tip_px": [float(alarm_tip[0]), float(alarm_tip[1])],
                    "center_px": [float(center[0]), float(center[1])],
                    "angle_deg": float(alarm_angle),
                },
            }
        )

    return RenderedClockGeometry(
        face_bbox_px=face_bbox,
        center_px=(float(center[0]), float(center[1])),
        hour_hand_bbox_px=hour_hand_bbox,
        minute_hand_bbox_px=minute_hand_bbox,
        second_hand_bbox_px=second_hand_bbox,
        alarm_hand_bbox_px=alarm_hand_bbox,
        hour_hand_tip_px=(float(hour_tip[0]), float(hour_tip[1])),
        minute_hand_tip_px=(float(minute_tip[0]), float(minute_tip[1])),
        second_hand_tip_px=((float(second_tip[0]), float(second_tip[1])) if second_tip is not None else None),
        alarm_hand_tip_px=((float(alarm_tip[0]), float(alarm_tip[1])) if alarm_tip is not None else None),
        entities=entities,
    )


def render_clock_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    shown_total_minutes: int,
    shown_total_seconds: int | None = None,
    show_second_hand: bool = False,
    render_params: ClockRenderParams,
    visual_theme: TimeArtifactClockTheme,
    alarm_hour_12: int | None = None,
    alarm_hand_color_rgb: Tuple[int, int, int] = (210, 40, 40),
    alarm_hand_width_px: int | None = None,
    center_px: Tuple[float, float] | None = None,
    force_show_minor_ticks: bool = False,
) -> RenderedClockScene:
    """Render one analog clock and return witness geometry."""

    image = background.copy().convert("RGB")
    show_seconds = bool(show_second_hand)
    extra_face_attrs = (
        {"shown_total_seconds": int(shown_total_seconds)}
        if show_seconds and shown_total_seconds is not None
        else None
    )
    geometry = draw_clock_geometry(
        image,
        center_px=(
            tuple(float(value) for value in center_px)
            if center_px is not None
            else (
                0.5 * float(render_params.canvas_width),
                0.5 * float(render_params.canvas_height),
            )
        ),
        face_radius_px=float(render_params.face_radius_px),
        scene_variant=str(scene_variant),
        shown_total_minutes=int(shown_total_minutes),
        shown_total_seconds=(
            int(shown_total_seconds)
            if show_seconds and shown_total_seconds is not None
            else None
        ),
        show_second_hand=show_seconds,
        render_params=render_params,
        visual_theme=visual_theme,
        extra_face_attrs=extra_face_attrs,
        alarm_hour_12=(int(alarm_hour_12) if alarm_hour_12 is not None else None),
        alarm_hand_color_rgb=tuple(int(value) for value in alarm_hand_color_rgb),
        alarm_hand_width_px=(int(alarm_hand_width_px) if alarm_hand_width_px is not None else None),
        force_show_minor_ticks=bool(force_show_minor_ticks),
    )
    return RenderedClockScene(
        image=image,
        scene_bbox_px=tuple(float(value) for value in geometry.face_bbox_px),
        face_bbox_px=tuple(float(value) for value in geometry.face_bbox_px),
        center_px=tuple(float(value) for value in geometry.center_px),
        hour_hand_bbox_px=tuple(float(value) for value in geometry.hour_hand_bbox_px),
        minute_hand_bbox_px=tuple(float(value) for value in geometry.minute_hand_bbox_px),
        second_hand_bbox_px=(
            tuple(float(value) for value in geometry.second_hand_bbox_px)
            if geometry.second_hand_bbox_px is not None
            else None
        ),
        alarm_hand_bbox_px=(
            tuple(float(value) for value in geometry.alarm_hand_bbox_px)
            if geometry.alarm_hand_bbox_px is not None
            else None
        ),
        hour_hand_tip_px=tuple(float(value) for value in geometry.hour_hand_tip_px),
        minute_hand_tip_px=tuple(float(value) for value in geometry.minute_hand_tip_px),
        second_hand_tip_px=(
            tuple(float(value) for value in geometry.second_hand_tip_px)
            if geometry.second_hand_tip_px is not None
            else None
        ),
        alarm_hand_tip_px=(
            tuple(float(value) for value in geometry.alarm_hand_tip_px)
            if geometry.alarm_hand_tip_px is not None
            else None
        ),
        entities=[dict(entity) for entity in geometry.entities],
    )


def text_option_card_bboxes(
    *,
    canvas_width: int,
    canvas_height: int,
    labels: Sequence[str],
    outer_margin_px: int = 24,
    gap_px: int = 8,
    card_height_px: int = 66,
    y0_px: int | None = None,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return one row of fixed-width text option card boxes."""

    option_labels = tuple(str(label) for label in labels)
    if not option_labels:
        raise ValueError("at least one option label is required")
    card_count = len(option_labels)
    total_gap = float(max(0, card_count - 1) * int(gap_px))
    available_width = float(canvas_width) - (2.0 * float(outer_margin_px)) - total_gap
    if available_width <= 0:
        raise ValueError("canvas is too narrow for option cards")
    card_width = available_width / float(card_count)
    y0 = (
        float(y0_px)
        if y0_px is not None
        else float(canvas_height) - float(outer_margin_px) - float(card_height_px)
    )
    return {
        str(label): (
            float(outer_margin_px) + float(index) * (card_width + float(gap_px)),
            float(y0),
            float(outer_margin_px) + float(index) * (card_width + float(gap_px)) + card_width,
            float(y0) + float(card_height_px),
        )
        for index, label in enumerate(option_labels)
    }


def option_cards_y_below_bbox(
    content_bbox_px: Sequence[float],
    *,
    canvas_height: int,
    gap_px: int = 24,
    card_height_px: int = 66,
    bottom_margin_px: int = 24,
) -> int:
    """Place text option cards under the clock content instead of page-bottom pinning."""

    proposed = int(round(float(content_bbox_px[3]) + float(gap_px)))
    max_y = int(canvas_height) - int(bottom_margin_px) - int(card_height_px)
    return max(0, min(int(proposed), int(max_y)))


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return float(bbox[2] - bbox[0])


def _fit_option_font(draw: ImageDraw.ImageDraw, texts: Sequence[str], *, max_width: float, start_size: int) -> Any:
    """Load the largest option font that fits all option values."""

    size = int(start_size)
    while size > 10:
        font = load_font(int(size), bold=False)
        if all(_text_width(draw, str(text), font) <= float(max_width) for text in texts):
            return font
        size -= 1
    return load_font(10, bold=False)


def draw_text_option_cards(
    image: Image.Image,
    *,
    text_by_label: Mapping[str, str],
    correct_label: str,
    y0_px: int | None = None,
    outer_margin_px: int = 24,
    gap_px: int = 8,
    card_height_px: int = 66,
    option_font_size_px: int = 18,
    label_font_size_px: int = 16,
) -> tuple[Dict[str, Tuple[float, float, float, float]], List[Dict[str, Any]]]:
    """Draw labeled text option cards and return their boxes/entities."""

    labels = tuple(str(label) for label in text_by_label)
    bboxes = text_option_card_bboxes(
        canvas_width=int(image.width),
        canvas_height=int(image.height),
        labels=labels,
        outer_margin_px=int(outer_margin_px),
        gap_px=int(gap_px),
        card_height_px=int(card_height_px),
        y0_px=y0_px,
    )
    draw = ImageDraw.Draw(image)
    max_text_width = min(float(box[2] - box[0]) - 12.0 for box in bboxes.values())
    option_font = _fit_option_font(
        draw,
        [str(text_by_label[str(label)]) for label in labels],
        max_width=float(max_text_width),
        start_size=int(option_font_size_px),
    )
    label_font = load_font(int(label_font_size_px), bold=True)
    entities: List[Dict[str, Any]] = []
    for label in labels:
        bbox = tuple(float(value) for value in bboxes[str(label)])
        draw.rounded_rectangle(
            bbox,
            radius=12,
            fill=(252, 252, 249),
            outline=(124, 132, 144),
            width=2,
        )
        label_box = (bbox[0] + 7.0, bbox[1] + 7.0, bbox[0] + 29.0, bbox[1] + 29.0)
        draw.rounded_rectangle(label_box, radius=6, fill=(46, 54, 66))
        draw_text_centered(
            draw,
            text=str(label),
            center=((label_box[0] + label_box[2]) / 2.0, (label_box[1] + label_box[3]) / 2.0),
            font=label_font,
            fill=(255, 255, 255),
        )
        draw_text_centered(
            draw,
            text=str(text_by_label[str(label)]),
            center=((bbox[0] + bbox[2]) / 2.0, bbox[1] + 43.0),
            font=option_font,
            fill=(30, 36, 44),
        )
        entities.append(
            {
                "entity_id": f"option_{str(label).lower()}",
                "entity_kind": "answer_option",
                "bbox_px": [float(value) for value in bbox],
                "attrs": {
                    "option_label": str(label),
                    "option_text": str(text_by_label[str(label)]),
                    "is_correct": bool(str(label) == str(correct_label)),
                },
            }
        )
    return bboxes, entities


__all__ = [
    "draw_clock_geometry",
    "draw_text_option_cards",
    "option_cards_y_below_bbox",
    "render_clock_scene",
    "text_option_card_bboxes",
]
