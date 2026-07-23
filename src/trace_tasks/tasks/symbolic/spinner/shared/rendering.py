"""Shared spinner-panel renderer for symbolic probability tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.bbox_projection import round_bbox as _round_bbox
from ....shared.color_distance import normalize_rgb as _rgb
from ....shared.drawing import draw_centered_text, draw_rounded_rect
from ....shared.text_rendering import load_font


SUPPORTED_SPINNER_SCENE_VARIANTS: Tuple[str, ...] = (
    "spinner_clean",
    "spinner_card",
    "spinner_notebook",
)


@dataclass(frozen=True)
class SpinnerRenderParams:
    """Pixel geometry and style controls for spinner scenes."""

    canvas_width: int = 1100
    canvas_height: int = 780
    single_center_x_px: int = 550
    single_center_y_px: int = 360
    single_radius_px: int = 235
    pair_left_center_x_px: int = 350
    pair_right_center_x_px: int = 750
    pair_center_y_px: int = 360
    pair_radius_px: int = 180
    panel_padding_px: int = 34
    panel_corner_radius_px: int = 22
    sector_outline_width_px: int = 3
    pointer_width_px: int = 5
    hub_radius_px: int = 18
    badge_width_px: int = 54
    badge_height_px: int = 46
    number_font_size_px: int = 21
    title_font_size_px: int = 28
    subtitle_font_size_px: int = 17
    style_overrides: Dict[str, Tuple[int, int, int]] | None = None


@dataclass(frozen=True)
class RenderedSpinnerScene:
    """Rendered spinner scene plus traceable geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    item_bbox_map: Dict[str, List[float]]
    sector_bbox_map: Dict[str, List[float]]
    panel_bbox_map: Dict[str, List[float]]
    scene_bbox_px: List[float]


def _sector_bbox(
    *,
    center: Tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
) -> List[float]:
    cx, cy = float(center[0]), float(center[1])
    points = [(cx, cy)]
    steps = max(4, int(abs(float(end_deg) - float(start_deg)) // 8) + 2)
    for step in range(int(steps) + 1):
        t = float(start_deg) + ((float(end_deg) - float(start_deg)) * float(step) / float(steps))
        radians = math.radians(float(t))
        points.append((float(cx + float(radius) * math.cos(radians)), float(cy + float(radius) * math.sin(radians))))
    return _round_bbox(
        [
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        ]
    )


def _shape_points(
    shape: str,
    *,
    center: Tuple[float, float],
    radius: float,
) -> List[Tuple[float, float]]:
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    if str(shape) == "triangle":
        return [
            (cx, cy - r),
            (cx + 0.88 * r, cy + 0.5 * r),
            (cx - 0.88 * r, cy + 0.5 * r),
        ]
    if str(shape) == "square":
        return [
            (cx - 0.74 * r, cy - 0.74 * r),
            (cx + 0.74 * r, cy - 0.74 * r),
            (cx + 0.74 * r, cy + 0.74 * r),
            (cx - 0.74 * r, cy + 0.74 * r),
        ]
    if str(shape) == "diamond":
        return [
            (cx, cy - r),
            (cx + r, cy),
            (cx, cy + r),
            (cx - r, cy),
        ]
    if str(shape) == "star":
        points: List[Tuple[float, float]] = []
        for index in range(10):
            angle = math.radians(-90.0 + 36.0 * float(index))
            local_r = r if index % 2 == 0 else 0.44 * r
            points.append((float(cx + local_r * math.cos(angle)), float(cy + local_r * math.sin(angle))))
        return points
    return []


def _draw_shape_marker(
    draw: ImageDraw.ImageDraw,
    *,
    shape: str,
    center: Tuple[float, float],
    radius: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int,
) -> None:
    if str(shape) == "circle":
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=max(1, int(width)))
        return
    points = _shape_points(str(shape), center=center, radius=float(radius))
    if points:
        draw.polygon(points, fill=fill, outline=outline)
        draw.line([*points, points[0]], fill=outline, width=max(1, int(width)), joint="curve")


def _draw_pointer(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    fill: Tuple[int, int, int],
    width: int,
) -> List[float]:
    cx, cy = float(center[0]), float(center[1])
    top_y = float(cy - radius + 8.0)
    tip = (cx, float(cy - radius + 34.0))
    points = [
        (cx - 16.0, top_y),
        (cx + 16.0, top_y),
        tip,
    ]
    draw.polygon(points, fill=fill, outline=fill)
    draw.line([(cx, top_y - 24.0), (cx, top_y)], fill=fill, width=max(1, int(width)))
    return _round_bbox([cx - 17.0, top_y - 24.0, cx + 17.0, tip[1]])


def _draw_spinner(
    draw: ImageDraw.ImageDraw,
    *,
    spinner_id: str,
    title: str,
    center: Tuple[float, float],
    radius: float,
    sectors: Sequence[Mapping[str, Any]],
    params: SpinnerRenderParams,
    style: Mapping[str, Tuple[int, int, int]],
    entities: List[Dict[str, Any]],
    item_bbox_map: Dict[str, List[float]],
    sector_bbox_map: Dict[str, List[float]],
    panel_bbox_map: Dict[str, List[float]],
) -> None:
    """Draw one complete spinner panel and record trace bboxes for panel, sectors, and pointer."""

    cx, cy = float(center[0]), float(center[1])
    panel_bbox = _round_bbox(
        [
            cx - float(radius) - float(params.panel_padding_px),
            cy - float(radius) - float(params.panel_padding_px) - 44.0,
            cx + float(radius) + float(params.panel_padding_px),
            cy + float(radius) + float(params.panel_padding_px),
        ]
    )
    panel_id = f"{spinner_id}_panel"
    panel_bbox_map[panel_id] = list(panel_bbox)
    item_bbox_map[panel_id] = list(panel_bbox)

    draw_rounded_rect(
        draw,
        tuple(panel_bbox),
        radius=int(params.panel_corner_radius_px),
        fill=style["panel_fill"],
        outline=style["panel_outline"],
        width=2,
    )
    title_font = load_font(int(params.title_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text=str(title),
        center=(cx, float(panel_bbox[1]) + 28.0),
        font=title_font,
        fill=style["text"],
        stroke_fill=style["text_stroke"],
        stroke_width=1,
    )

    sector_count = max(1, len(sectors))
    spinner_bbox = [cx - float(radius), cy - float(radius), cx + float(radius), cy + float(radius)]
    show_numbers = bool(sectors[0].get("show_number", True)) if sectors else True
    show_shapes = bool(sectors[0].get("show_shape", True)) if sectors else True
    for index, sector in enumerate(sectors):
        start_deg = -90.0 + (360.0 * float(index) / float(sector_count))
        end_deg = -90.0 + (360.0 * float(index + 1) / float(sector_count))
        fill = _rgb(sector.get("color_rgb", (210, 210, 210)))
        draw.pieslice(
            spinner_bbox,
            start=float(start_deg),
            end=float(end_deg),
            fill=fill,
            outline=style["sector_outline"],
            width=max(1, int(params.sector_outline_width_px)),
        )
        sector_id = str(sector["sector_id"])
        sector_bbox = _sector_bbox(center=(cx, cy), radius=float(radius), start_deg=float(start_deg), end_deg=float(end_deg))
        sector_bbox_map[sector_id] = list(sector_bbox)
        item_bbox_map[sector_id] = list(sector_bbox)
        entities.append(
            {
                "entity_id": sector_id,
                "entity_type": "spinner_sector",
                "spinner_id": str(spinner_id),
                "sector_index": int(index),
                "bbox_px": list(sector_bbox),
                "color_name": str(sector["color_name"]),
                "shape": str(sector["shape"]),
                "number": int(sector["number"]),
            }
        )

        if show_numbers or show_shapes:
            mid_deg = 0.5 * (float(start_deg) + float(end_deg))
            mid = math.radians(float(mid_deg))
            label_radius = float(radius) * 0.62
            lx = float(cx + label_radius * math.cos(mid))
            ly = float(cy + label_radius * math.sin(mid))
            if show_numbers and show_shapes:
                badge_w = float(params.badge_width_px)
                badge_h = float(params.badge_height_px)
                number_center = (lx - 10.0, ly)
                shape_center = (lx + 15.0, ly)
                shape_radius = 10.0
            elif show_numbers:
                badge_w = float(params.badge_width_px) * 0.76
                badge_h = float(params.badge_height_px)
                number_center = (lx, ly)
                shape_center = (lx, ly)
                shape_radius = 0.0
            else:
                badge_w = float(params.badge_height_px)
                badge_h = float(params.badge_height_px)
                number_center = (lx, ly)
                shape_center = (lx, ly)
                shape_radius = 13.0
            badge_bbox = [lx - 0.5 * badge_w, ly - 0.5 * badge_h, lx + 0.5 * badge_w, ly + 0.5 * badge_h]
            draw_rounded_rect(
                draw,
                tuple(badge_bbox),
                radius=12,
                fill=style["badge_fill"],
                outline=style["badge_outline"],
                width=1,
            )
            if show_numbers:
                number_font = load_font(int(params.number_font_size_px), bold=True)
                draw_centered_text(
                    draw,
                    text=str(int(sector["number"])),
                    center=number_center,
                    font=number_font,
                    fill=style["text"],
                    stroke_fill=style["text_stroke"],
                    stroke_width=0,
                )
            if show_shapes:
                _draw_shape_marker(
                    draw,
                    shape=str(sector["shape"]),
                    center=shape_center,
                    radius=float(shape_radius),
                    fill=style["shape_fill"],
                    outline=style["shape_outline"],
                    width=2,
                )

    draw.ellipse(
        [
            cx - float(params.hub_radius_px),
            cy - float(params.hub_radius_px),
            cx + float(params.hub_radius_px),
            cy + float(params.hub_radius_px),
        ],
        fill=style["hub_fill"],
        outline=style["sector_outline"],
        width=2,
    )
    pointer_bbox = _draw_pointer(
        draw,
        center=(cx, cy),
        radius=float(radius),
        fill=style["pointer"],
        width=int(params.pointer_width_px),
    )
    pointer_id = f"{spinner_id}_pointer"
    item_bbox_map[pointer_id] = list(pointer_bbox)
    entities.append(
        {
            "entity_id": panel_id,
            "entity_type": "spinner_panel",
            "spinner_id": str(spinner_id),
            "bbox_px": list(panel_bbox),
        }
    )
    entities.append(
        {
            "entity_id": pointer_id,
            "entity_type": "spinner_pointer",
            "spinner_id": str(spinner_id),
            "bbox_px": list(pointer_bbox),
        }
    )


def _style_for_variant(
    scene_variant: str,
    *,
    style_overrides: Mapping[str, Tuple[int, int, int]] | None = None,
) -> Dict[str, Tuple[int, int, int]]:
    base = {
        "panel_fill": (252, 252, 250),
        "panel_outline": (70, 78, 91),
        "sector_outline": (43, 48, 58),
        "text": (25, 29, 36),
        "muted_text": (91, 99, 112),
        "text_stroke": (255, 255, 255),
        "badge_fill": (255, 255, 255),
        "badge_outline": (58, 64, 76),
        "shape_fill": (28, 32, 40),
        "shape_outline": (255, 255, 255),
        "hub_fill": (255, 255, 255),
        "pointer": (30, 35, 44),
    }
    if str(scene_variant) == "spinner_card":
        base.update(
            {
                "panel_fill": (247, 250, 248),
                "panel_outline": (54, 103, 99),
                "sector_outline": (42, 65, 68),
                "muted_text": (70, 105, 103),
            }
        )
    elif str(scene_variant) == "spinner_notebook":
        base.update(
            {
                "panel_fill": (252, 249, 240),
                "panel_outline": (117, 91, 67),
                "sector_outline": (80, 62, 50),
                "muted_text": (123, 98, 76),
            }
        )
    base.update({str(key): _rgb(value) for key, value in dict(style_overrides or {}).items()})
    return base


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return float(bbox[2] - bbox[0])


def _fit_option_font(draw: ImageDraw.ImageDraw, texts: Sequence[str], *, max_width: float, start_size: int):
    size = int(start_size)
    while int(size) > 10:
        font = load_font(int(size), bold=False)
        if all(_text_width(draw, str(text), font) <= float(max_width) for text in texts):
            return font
        size -= 1
    return load_font(10, bold=False)


def option_cards_y_for_scene(
    scene_bbox_px: Sequence[float],
    *,
    canvas_height: int,
    gap_px: int = 28,
    card_height_px: int = 66,
    bottom_margin_px: int = 28,
) -> int:
    """Place answer cards below the rendered spinner panels."""

    proposed = int(round(float(scene_bbox_px[3]) + float(gap_px)))
    max_y = int(canvas_height) - int(bottom_margin_px) - int(card_height_px)
    return max(0, min(int(proposed), int(max_y)))


def probability_option_card_bboxes(
    *,
    canvas_width: int,
    labels: Sequence[str],
    y0_px: int,
    outer_margin_px: int = 44,
    gap_px: int = 10,
    card_height_px: int = 66,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return one fixed A-F row of reduced-fraction option cards."""

    option_labels = tuple(str(label) for label in labels)
    if len(option_labels) != 6:
        raise ValueError("spinner probability option cards require exactly six labels")
    total_gap = float(max(0, len(option_labels) - 1) * int(gap_px))
    available_width = float(canvas_width) - (2.0 * float(outer_margin_px)) - total_gap
    if available_width <= 0.0:
        raise ValueError("canvas is too narrow for spinner probability option cards")
    card_width = float(available_width) / float(len(option_labels))
    return {
        str(label): (
            float(outer_margin_px) + float(index) * (card_width + float(gap_px)),
            float(y0_px),
            float(outer_margin_px) + float(index) * (card_width + float(gap_px)) + card_width,
            float(y0_px) + float(card_height_px),
        )
        for index, label in enumerate(option_labels)
    }


def draw_probability_option_cards(
    image: Image.Image,
    *,
    text_by_label: Mapping[str, str],
    correct_label: str,
    y0_px: int,
    outer_margin_px: int = 44,
    gap_px: int = 10,
    card_height_px: int = 66,
    option_font_size_px: int = 24,
    label_font_size_px: int = 16,
) -> Tuple[Dict[str, Tuple[float, float, float, float]], List[Dict[str, Any]]]:
    """Draw visible A-F fraction answer cards below the spinner panels."""

    labels = tuple(str(label) for label in text_by_label)
    bboxes = probability_option_card_bboxes(
        canvas_width=int(image.width),
        labels=labels,
        y0_px=int(y0_px),
        outer_margin_px=int(outer_margin_px),
        gap_px=int(gap_px),
        card_height_px=int(card_height_px),
    )
    draw = ImageDraw.Draw(image, "RGBA")
    option_font = _fit_option_font(
        draw,
        [str(text_by_label[str(label)]) for label in labels],
        max_width=min(float(box[2] - box[0]) - 14.0 for box in bboxes.values()),
        start_size=int(option_font_size_px),
    )
    label_font = load_font(int(label_font_size_px), bold=True)
    entities: List[Dict[str, Any]] = []
    for label in labels:
        bbox = tuple(float(value) for value in bboxes[str(label)])
        draw_rounded_rect(
            draw,
            bbox,
            radius=11,
            fill=(253, 252, 248),
            outline=(104, 112, 126),
            width=2,
        )
        label_box = (bbox[0] + 7.0, bbox[1] + 8.0, bbox[0] + 31.0, bbox[1] + 32.0)
        draw_rounded_rect(
            draw,
            label_box,
            radius=6,
            fill=(42, 50, 63),
            outline=(42, 50, 63),
            width=1,
        )
        draw_centered_text(
            draw,
            text=str(label),
            center=((label_box[0] + label_box[2]) / 2.0, (label_box[1] + label_box[3]) / 2.0),
            font=label_font,
            fill=(255, 255, 255),
            stroke_fill=(42, 50, 63),
            stroke_width=0,
        )
        draw_centered_text(
            draw,
            text=str(text_by_label[str(label)]),
            center=((bbox[0] + bbox[2]) / 2.0, bbox[1] + 44.0),
            font=option_font,
            fill=(28, 34, 44),
            stroke_fill=(253, 252, 248),
            stroke_width=0,
        )
        entities.append(
            {
                "entity_id": f"option_{str(label).lower()}",
                "entity_type": "answer_option",
                "bbox_px": [round(float(value), 3) for value in bbox],
                "attrs": {
                    "option_label": str(label),
                    "option_text": str(text_by_label[str(label)]),
                    "is_correct": bool(str(label) == str(correct_label)),
                },
            }
        )
    return bboxes, entities


def render_spinner_scene(
    image: Image.Image,
    *,
    scene_variant: str,
    mode: str,
    spinner_specs: Sequence[Mapping[str, Any]],
    render_params: SpinnerRenderParams,
) -> RenderedSpinnerScene:
    """Render one single- or pair-spinner panel."""

    if str(scene_variant) not in SUPPORTED_SPINNER_SCENE_VARIANTS:
        raise ValueError(f"unsupported spinner scene variant: {scene_variant}")
    if str(mode) not in {"single", "pair"}:
        raise ValueError(f"unsupported spinner mode: {mode}")

    draw = ImageDraw.Draw(image)
    if str(scene_variant) == "spinner_notebook":
        grid_color = (232, 225, 211)
        for x in range(34, int(render_params.canvas_width), 34):
            draw.line([(x, 0), (x, int(render_params.canvas_height))], fill=grid_color, width=1)
        for y in range(34, int(render_params.canvas_height), 34):
            draw.line([(0, y), (int(render_params.canvas_width), y)], fill=grid_color, width=1)

    entities: List[Dict[str, Any]] = []
    item_bbox_map: Dict[str, List[float]] = {}
    sector_bbox_map: Dict[str, List[float]] = {}
    panel_bbox_map: Dict[str, List[float]] = {}
    style = _style_for_variant(str(scene_variant), style_overrides=render_params.style_overrides)

    if str(mode) == "single":
        if len(spinner_specs) != 1:
            raise ValueError("single spinner scene requires exactly one spinner spec")
        spinner = spinner_specs[0]
        _draw_spinner(
            draw,
            spinner_id=str(spinner["spinner_id"]),
            title=str(spinner.get("title", "Spinner")),
            center=(float(render_params.single_center_x_px), float(render_params.single_center_y_px)),
            radius=float(render_params.single_radius_px),
            sectors=list(spinner["sectors"]),
            params=render_params,
            style=style,
            entities=entities,
            item_bbox_map=item_bbox_map,
            sector_bbox_map=sector_bbox_map,
            panel_bbox_map=panel_bbox_map,
        )
    else:
        if len(spinner_specs) != 2:
            raise ValueError("pair spinner scene requires exactly two spinner specs")
        centers = [
            (float(render_params.pair_left_center_x_px), float(render_params.pair_center_y_px)),
            (float(render_params.pair_right_center_x_px), float(render_params.pair_center_y_px)),
        ]
        for spinner, center in zip(spinner_specs, centers):
            _draw_spinner(
                draw,
                spinner_id=str(spinner["spinner_id"]),
                title=str(spinner.get("title", str(spinner["spinner_id"]))),
                center=center,
                radius=float(render_params.pair_radius_px),
                sectors=list(spinner["sectors"]),
                params=render_params,
                style=style,
                entities=entities,
                item_bbox_map=item_bbox_map,
                sector_bbox_map=sector_bbox_map,
                panel_bbox_map=panel_bbox_map,
            )

    bboxes = list(panel_bbox_map.values())
    scene_bbox = _round_bbox(
        [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ]
    )
    return RenderedSpinnerScene(
        image=image,
        entities=entities,
        item_bbox_map=item_bbox_map,
        sector_bbox_map=sector_bbox_map,
        panel_bbox_map=panel_bbox_map,
        scene_bbox_px=list(scene_bbox),
    )


__all__ = [
    "RenderedSpinnerScene",
    "SUPPORTED_SPINNER_SCENE_VARIANTS",
    "SpinnerRenderParams",
    "draw_probability_option_cards",
    "option_cards_y_for_scene",
    "render_spinner_scene",
]
