"""Shared dice-tray renderer for symbolic probability tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.bbox_projection import round_bbox as _round_bbox
from ....shared.color_distance import normalize_rgb as _rgb
from ....shared.drawing import draw_centered_text, draw_rounded_rect
from ....shared.text_rendering import load_font
from ...shared.scene_style import SymbolicSceneStyle


SUPPORTED_DICE_SCENE_VARIANTS: Tuple[str, ...] = (
    "dice_tray_clean",
    "dice_tray_felt",
    "dice_tray_notebook",
)
SUPPORTED_DICE_VISUAL_STYLES: Tuple[str, ...] = (
    "classic_rounded",
    "flat_print",
    "beveled_tokens",
    "inked_pips",
    "soft_shadow",
)


@dataclass(frozen=True)
class DiceRenderParams:
    """Pixel geometry and style controls for dice probability scenes."""

    canvas_width: int = 1100
    canvas_height: int = 780
    single_tray_bbox_px: Tuple[int, int, int, int] = (145, 112, 955, 650)
    pair_left_tray_bbox_px: Tuple[int, int, int, int] = (68, 142, 522, 628)
    pair_right_tray_bbox_px: Tuple[int, int, int, int] = (578, 142, 1032, 628)
    die_size_px: int = 72
    die_gap_px: int = 18
    tray_corner_radius_px: int = 24
    tray_outline_width_px: int = 3
    die_corner_radius_px: int = 14
    die_outline_width_px: int = 3
    pip_radius_px: int = 6
    title_font_size_px: int = 28


@dataclass(frozen=True)
class RenderedDiceScene:
    """Rendered dice scene plus traceable geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    item_bbox_map: Dict[str, List[float]]
    die_bbox_map: Dict[str, List[float]]
    tray_bbox_map: Dict[str, List[float]]
    scene_bbox_px: List[float]


def _luminance(rgb: Sequence[int]) -> float:
    red, green, blue = _rgb(rgb)
    return 0.2126 * float(red) + 0.7152 * float(green) + 0.0722 * float(blue)


def _pip_centers(bbox: Sequence[float], value: int) -> List[Tuple[float, float]]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    width = float(x1 - x0)
    height = float(y1 - y0)
    left = float(x0 + 0.30 * width)
    mid_x = float(x0 + 0.50 * width)
    right = float(x0 + 0.70 * width)
    top = float(y0 + 0.30 * height)
    mid_y = float(y0 + 0.50 * height)
    bottom = float(y0 + 0.70 * height)
    value = int(value)
    if value == 1:
        return [(mid_x, mid_y)]
    if value == 2:
        return [(left, top), (right, bottom)]
    if value == 3:
        return [(left, top), (mid_x, mid_y), (right, bottom)]
    if value == 4:
        return [(left, top), (right, top), (left, bottom), (right, bottom)]
    if value == 5:
        return [(left, top), (right, top), (mid_x, mid_y), (left, bottom), (right, bottom)]
    return [(left, top), (right, top), (left, mid_y), (right, mid_y), (left, bottom), (right, bottom)]


def _draw_die(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    value: int,
    params: DiceRenderParams,
    visual_style: str = "classic_rounded",
) -> None:
    """Draw one die while preserving semantic color and visible pip value.

    Visual-style branches only alter chrome, shadowing, and pip treatment; the
    filled face color and pip count remain the metadata-grounded semantics for
    probability questions.
    """

    x0, y0, x1, y1 = [float(v) for v in bbox]
    style_id = str(visual_style)
    radius = int(params.die_corner_radius_px)
    outline_width = int(params.die_outline_width_px)
    if style_id in {"classic_rounded", "beveled_tokens", "soft_shadow"}:
        shadow_alpha = 42 if style_id == "classic_rounded" else (72 if style_id == "soft_shadow" else 56)
        shadow_offset = 6.0 if style_id == "soft_shadow" else 4.0
        shadow = (38, 42, 50, shadow_alpha)
        draw_rounded_rect(
            draw,
            (x0 + shadow_offset, y0 + shadow_offset + 1.0, x1 + shadow_offset, y1 + shadow_offset + 1.0),
            radius=radius,
            fill=shadow,
            outline=shadow,
            width=1,
        )
    elif style_id == "flat_print":
        radius = max(5, int(params.die_corner_radius_px * 0.55))
        outline_width = max(1, int(params.die_outline_width_px) - 1)
    elif style_id == "inked_pips":
        radius = max(4, int(params.die_corner_radius_px * 0.45))
        outline_width = max(2, int(params.die_outline_width_px))
    draw_rounded_rect(
        draw,
        (x0, y0, x1, y1),
        radius=radius,
        fill=fill,
        outline=outline,
        width=outline_width,
    )
    if style_id in {"classic_rounded", "beveled_tokens"}:
        highlight = tuple(min(255, int(channel) + (48 if style_id == "beveled_tokens" else 34)) for channel in fill)
        draw.arc(
            [x0 + 8.0, y0 + 8.0, x1 - 8.0, y1 - 8.0],
            start=202,
            end=276,
            fill=highlight,
            width=2 if style_id == "classic_rounded" else 3,
        )
    if style_id == "beveled_tokens":
        lowlight = tuple(max(0, int(channel) - 38) for channel in fill)
        draw.arc([x0 + 7.0, y0 + 7.0, x1 - 7.0, y1 - 7.0], start=28, end=92, fill=lowlight, width=2)
    pip_fill = (255, 255, 255) if _luminance(fill) < 145.0 else (24, 28, 36)
    pip_outline = (24, 28, 36) if _luminance(fill) < 145.0 else (255, 255, 255)
    pip_radius = float(params.pip_radius_px)
    if style_id == "flat_print":
        pip_radius = max(3.0, float(params.pip_radius_px) * 0.82)
    elif style_id == "inked_pips":
        pip_radius = max(4.0, float(params.pip_radius_px) * 0.92)
    elif style_id == "beveled_tokens":
        pip_radius = float(params.pip_radius_px) * 1.08
    for cx, cy in _pip_centers((x0, y0, x1, y1), int(value)):
        if style_id == "inked_pips":
            draw.ellipse(
                [cx - pip_radius - 2.0, cy - pip_radius - 2.0, cx + pip_radius + 2.0, cy + pip_radius + 2.0],
                fill=pip_outline,
                outline=pip_outline,
                width=1,
            )
        draw.ellipse(
            [cx - pip_radius, cy - pip_radius, cx + pip_radius, cy + pip_radius],
            fill=pip_fill,
            outline=pip_outline,
            width=1,
        )


def _layout_dice(*, count: int, tray_bbox: Sequence[float], params: DiceRenderParams) -> List[List[float]]:
    x0, y0, x1, y1 = [float(v) for v in tray_bbox]
    title_pad = 66.0
    inner_pad = 32.0
    usable_x0 = float(x0 + inner_pad)
    usable_y0 = float(y0 + title_pad)
    usable_x1 = float(x1 - inner_pad)
    usable_y1 = float(y1 - inner_pad)
    die = float(params.die_size_px)
    gap = float(params.die_gap_px)
    count = max(1, int(count))
    max_cols = max(1, int((usable_x1 - usable_x0 + gap) // (die + gap)))
    cols = max(2, int(math.ceil(math.sqrt(float(count)))))
    cols = min(max_cols, max(1, cols))
    while int(math.ceil(float(count) / float(cols))) * (die + gap) - gap > (usable_y1 - usable_y0) and cols < max_cols:
        cols += 1
    rows = int(math.ceil(float(count) / float(cols)))
    block_w = float(cols * die + max(0, cols - 1) * gap)
    block_h = float(rows * die + max(0, rows - 1) * gap)
    start_x = float(usable_x0 + 0.5 * ((usable_x1 - usable_x0) - block_w))
    start_y = float(usable_y0 + 0.5 * ((usable_y1 - usable_y0) - block_h))
    bboxes: List[List[float]] = []
    for index in range(count):
        row = int(index // cols)
        col = int(index % cols)
        dx0 = float(start_x + col * (die + gap))
        dy0 = float(start_y + row * (die + gap))
        bboxes.append(_round_bbox([dx0, dy0, dx0 + die, dy0 + die]))
    return bboxes


def _style_for_variant(scene_variant: str, scene_style: SymbolicSceneStyle | None = None) -> Dict[str, Tuple[int, int, int]]:
    base = {
        "tray_fill": (250, 251, 249),
        "tray_outline": (67, 75, 88),
        "title": (29, 34, 43),
        "title_stroke": (255, 255, 255),
        "die_outline": (28, 34, 44),
        "notebook_grid": (232, 225, 211),
    }
    if str(scene_variant) == "dice_tray_felt":
        base.update(
            {
                "tray_fill": (227, 242, 233),
                "tray_outline": (48, 105, 82),
                "title": (22, 63, 50),
            }
        )
    elif str(scene_variant) == "dice_tray_notebook":
        base.update(
            {
                "tray_fill": (252, 249, 240),
                "tray_outline": (117, 91, 67),
                "title": (83, 65, 51),
            }
        )
    if scene_style is not None:
        base.update(
            {
                "tray_fill": tuple(int(value) for value in scene_style.panel_fill_rgb),
                "tray_outline": tuple(int(value) for value in scene_style.panel_border_rgb),
                "title": tuple(int(value) for value in scene_style.text_rgb),
                "title_stroke": tuple(int(value) for value in scene_style.text_stroke_rgb),
                "die_outline": tuple(int(value) for value in scene_style.grid_rgb),
                "notebook_grid": tuple(int(value) for value in scene_style.notebook_line_rgb),
            }
        )
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
    """Place answer cards below the rendered dice trays, not at the page bottom."""

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
        raise ValueError("dice probability option cards require exactly six labels")
    total_gap = float(max(0, len(option_labels) - 1) * int(gap_px))
    available_width = float(canvas_width) - (2.0 * float(outer_margin_px)) - total_gap
    if available_width <= 0.0:
        raise ValueError("canvas is too narrow for dice probability option cards")
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
    """Draw visible A-F fraction answer cards below the dice trays."""

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


def _draw_tray(
    draw: ImageDraw.ImageDraw,
    *,
    tray: Mapping[str, Any],
    tray_bbox: Sequence[float],
    params: DiceRenderParams,
    style: Mapping[str, Tuple[int, int, int]],
    entities: List[Dict[str, Any]],
    item_bbox_map: Dict[str, List[float]],
    die_bbox_map: Dict[str, List[float]],
    tray_bbox_map: Dict[str, List[float]],
    visual_style: str = "classic_rounded",
) -> None:
    """Draw one tray and register tray/die boxes for annotation projection.

    The renderer writes tray bboxes and die bboxes into the supplied maps after
    final layout, so task annotations are projected from rendered geometry.
    """

    tray_id = str(tray["tray_id"])
    title = str(tray.get("title", "Dice tray"))
    tray_bbox_rounded = _round_bbox(tray_bbox)
    tray_bbox_map[tray_id] = list(tray_bbox_rounded)
    item_bbox_map[tray_id] = list(tray_bbox_rounded)
    draw_rounded_rect(
        draw,
        tuple(tray_bbox_rounded),
        radius=int(params.tray_corner_radius_px),
        fill=style["tray_fill"],
        outline=style["tray_outline"],
        width=int(params.tray_outline_width_px),
    )
    title_font = load_font(int(params.title_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text=title,
        center=(0.5 * (float(tray_bbox_rounded[0]) + float(tray_bbox_rounded[2])), float(tray_bbox_rounded[1]) + 34.0),
        font=title_font,
        fill=style["title"],
        stroke_fill=style["title_stroke"],
        stroke_width=1,
    )
    dice = [dict(die) for die in tray.get("dice", [])]
    for die, bbox in zip(dice, _layout_dice(count=len(dice), tray_bbox=tray_bbox_rounded, params=params)):
        die_id = str(die["die_id"])
        fill = _rgb(die.get("color_rgb", (230, 230, 230)))
        _draw_die(
            draw,
            bbox=bbox,
            fill=fill,
            outline=style["die_outline"],
            value=int(die["value"]),
            params=params,
            visual_style=str(visual_style),
        )
        die_bbox_map[die_id] = list(bbox)
        item_bbox_map[die_id] = list(bbox)
        entities.append(
            {
                "entity_id": die_id,
                "entity_type": "probability_die",
                "tray_id": tray_id,
                "die_index": int(die.get("die_index", 0)),
                "value": int(die["value"]),
                "color_name": str(die["color_name"]),
                "bbox_px": list(bbox),
            }
        )
    entities.append(
        {
            "entity_id": tray_id,
            "entity_type": "dice_tray",
            "tray_id": tray_id,
            "bbox_px": list(tray_bbox_rounded),
        }
    )


def render_dice_probability_scene(
    image: Image.Image,
    *,
    scene_variant: str,
    mode: str,
    tray_specs: Sequence[Mapping[str, Any]],
    render_params: DiceRenderParams,
    scene_style: SymbolicSceneStyle | None = None,
    visual_style: str = "classic_rounded",
) -> RenderedDiceScene:
    """Render one single-, pair-, or conditional-dice probability panel."""

    if str(scene_variant) not in SUPPORTED_DICE_SCENE_VARIANTS:
        raise ValueError(f"unsupported dice scene variant: {scene_variant}")
    if str(mode) not in {"single", "pair", "conditional"}:
        raise ValueError(f"unsupported dice probability mode: {mode}")

    draw = ImageDraw.Draw(image, "RGBA")
    style = _style_for_variant(str(scene_variant), scene_style=scene_style)
    if str(scene_variant) == "dice_tray_notebook":
        grid_color = tuple(int(value) for value in style["notebook_grid"]) + (140,)
        for x in range(34, int(render_params.canvas_width), 34):
            draw.line([(x, 0), (x, int(render_params.canvas_height))], fill=grid_color, width=1)
        for y in range(34, int(render_params.canvas_height), 34):
            draw.line([(0, y), (int(render_params.canvas_width), y)], fill=grid_color, width=1)

    entities: List[Dict[str, Any]] = []
    item_bbox_map: Dict[str, List[float]] = {}
    die_bbox_map: Dict[str, List[float]] = {}
    tray_bbox_map: Dict[str, List[float]] = {}

    if str(mode) in {"single", "conditional"}:
        if len(tray_specs) != 1:
            raise ValueError("single or conditional dice scene requires exactly one tray spec")
        _draw_tray(
            draw,
            tray=tray_specs[0],
            tray_bbox=render_params.single_tray_bbox_px,
            params=render_params,
            style=style,
            entities=entities,
            item_bbox_map=item_bbox_map,
            die_bbox_map=die_bbox_map,
            tray_bbox_map=tray_bbox_map,
            visual_style=str(visual_style),
        )
    else:
        if len(tray_specs) != 2:
            raise ValueError("pair dice scene requires exactly two tray specs")
        for tray, bbox in zip(tray_specs, [render_params.pair_left_tray_bbox_px, render_params.pair_right_tray_bbox_px]):
            _draw_tray(
                draw,
                tray=tray,
                tray_bbox=bbox,
                params=render_params,
                style=style,
                entities=entities,
                item_bbox_map=item_bbox_map,
                die_bbox_map=die_bbox_map,
                tray_bbox_map=tray_bbox_map,
                visual_style=str(visual_style),
            )

    bboxes = list(tray_bbox_map.values())
    scene_bbox = _round_bbox(
        [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ]
    )
    return RenderedDiceScene(
        image=image,
        entities=entities,
        item_bbox_map=item_bbox_map,
        die_bbox_map=die_bbox_map,
        tray_bbox_map=tray_bbox_map,
        scene_bbox_px=list(scene_bbox),
    )


__all__ = [
    "DiceRenderParams",
    "RenderedDiceScene",
    "SUPPORTED_DICE_SCENE_VARIANTS",
    "SUPPORTED_DICE_VISUAL_STYLES",
    "draw_probability_option_cards",
    "option_cards_y_for_scene",
    "probability_option_card_bboxes",
    "render_dice_probability_scene",
]
