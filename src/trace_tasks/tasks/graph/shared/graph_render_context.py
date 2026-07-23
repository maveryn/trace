"""Context text and content reservation helpers for node-link graphs."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ...shared.text_legibility import draw_traced_text
from ...shared.text_rendering import load_font
from .graph_render_types import BBox, GraphRenderParams


def _apply_content_layout_jitter(
    panel_geometry: Dict[str, Any],
    *,
    render_params: GraphRenderParams,
    layout_seed: int,
) -> None:
    """Apply bounded content-area jitter before graph layout projection."""

    base = tuple(int(value) for value in panel_geometry["scene_content_xyxy"])
    width = int(base[2] - base[0])
    height = int(base[3] - base[1])
    requested = max(0, int(render_params.content_jitter_max_px))
    safe_jitter = min(int(requested), max(0, int(width // 12)), max(0, int(height // 12)))
    metadata: Dict[str, Any] = {
        "enabled": bool(safe_jitter > 0),
        "requested_max_px": int(requested),
        "resolved_max_px": int(safe_jitter),
        "base_content_xyxy": [int(value) for value in base],
        "final_content_xyxy": [int(value) for value in base],
        "insets_px": {"left": 0, "top": 0, "right": 0, "bottom": 0},
    }
    if int(safe_jitter) <= 0:
        panel_geometry["layout_jitter"] = metadata
        return

    rng = random.Random(int(layout_seed) + 104729)
    left = int(rng.randint(0, int(safe_jitter)))
    top = int(rng.randint(0, int(safe_jitter)))
    right = int(rng.randint(0, int(safe_jitter)))
    bottom = int(rng.randint(0, int(safe_jitter)))
    jittered = (
        int(base[0] + left),
        int(base[1] + top),
        int(base[2] - right),
        int(base[3] - bottom),
    )
    if int(jittered[2] - jittered[0]) < 96 or int(jittered[3] - jittered[1]) < 96:
        jittered = base
        left = top = right = bottom = 0
    panel_geometry["scene_content_unjittered_xyxy"] = [int(value) for value in base]
    panel_geometry["scene_content_xyxy"] = [int(value) for value in jittered]
    metadata["final_content_xyxy"] = [int(value) for value in jittered]
    metadata["insets_px"] = {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)}
    panel_geometry["layout_jitter"] = metadata


_NODE_LINK_CONTEXT_LEFT_TEXTS: Tuple[str, ...] = (
    "NETWORK VIEW",
    "TOPOLOGY DRAFT",
    "RELATION MAP",
    "NODE STUDY",
    "LINK LAYER",
)
_NODE_LINK_CONTEXT_RIGHT_TEXTS: Tuple[str, ...] = (
    "MAP REF",
    "SCHEMA",
    "DRAFT",
    "LAYER",
    "INDEX",
)
_NODE_LINK_CONTEXT_BLOCK_PHRASES: Tuple[str, ...] = (
    "Reference note: this panel is a compact visual appendix.",
    "Display note: spacing, color, and style are presentation details.",
    "Labels are identifiers for the drawing and may use mixed typography.",
    "Decorative notes around the panel are not part of the requested result.",
    "Review copy: use the diagram marks inside the framed area for the task.",
    "Layout memo: the surrounding copy is context text only.",
    "Draft caption: the figure has been resized to fit the worksheet.",
    "Source note: presentation choices can vary across generated panels.",
)
_NODE_LINK_CONTEXT_BLOCK_POSITIONS: Tuple[str, ...] = ("top", "bottom", "left", "right")
_NODE_LINK_CONTEXT_BLOCK_LEVELS: Tuple[str, ...] = ("low", "medium", "high")


def _weighted_context_choice(
    rng: random.Random,
    *,
    values: Sequence[str],
    weights: Mapping[str, float] | None,
) -> str:
    """Choose one context style axis value from optional positive weights."""

    options = tuple(str(value) for value in values)
    if not options:
        raise ValueError("context choice requires at least one value")
    explicit_weights = dict(weights or {})
    missing_weight = 0.0 if explicit_weights else 1.0
    parsed = [max(0.0, float(explicit_weights.get(str(value), missing_weight))) for value in options]
    total = float(sum(parsed))
    if total <= 0.0:
        parsed = [1.0 for _ in options]
        total = float(len(options))
    threshold = rng.random() * total
    cursor = 0.0
    for value, weight in zip(options, parsed):
        cursor += float(weight)
        if threshold <= cursor:
            return str(value)
    return str(options[-1])


def _wrap_context_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: ImageFont.ImageFont,
    max_width_px: int,
    max_lines: int,
) -> List[str]:
    """Wrap muted context copy to a small fixed block."""

    words = [str(word) for word in str(text).split() if str(word)]
    if not words:
        return []
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=0)
        if int(bbox[2] - bbox[0]) <= int(max_width_px) or not current:
            current = candidate
            continue
        lines.append(str(current))
        current = str(word)
        if len(lines) >= max(1, int(max_lines)):
            break
    if current and len(lines) < max(1, int(max_lines)):
        lines.append(str(current))
    if len(lines) > max(1, int(max_lines)):
        lines = lines[: max(1, int(max_lines))]
    if words and lines:
        joined_prefix = " ".join(lines).replace("...", "").strip()
        original = " ".join(words)
        if not original.startswith(joined_prefix) and len(lines[-1]) > 3:
            lines[-1] = str(lines[-1]).rstrip(". ") + "..."
        elif len(" ".join(lines).split()) < len(words) and len(lines[-1]) > 3:
            lines[-1] = str(lines[-1]).rstrip(". ") + "..."
    return lines


def _apply_context_block_reservation(
    panel_geometry: Dict[str, Any],
    *,
    position: str,
    block_bbox: BBox,
) -> None:
    """Reserve panel space for one context block before graph projection."""

    base = tuple(int(value) for value in panel_geometry["scene_content_xyxy"])
    block = tuple(int(value) for value in block_bbox)
    gutter = 12
    final = list(base)
    if str(position) == "top":
        final[1] = max(int(final[1]), int(block[3] + gutter))
    elif str(position) == "bottom":
        final[3] = min(int(final[3]), int(block[1] - gutter))
    elif str(position) == "left":
        final[0] = max(int(final[0]), int(block[2] + gutter))
    elif str(position) == "right":
        final[2] = min(int(final[2]), int(block[0] - gutter))
    if int(final[2] - final[0]) < 220 or int(final[3] - final[1]) < 190:
        final = list(base)
    panel_geometry["context_block_reservation"] = {
        "base_content_xyxy": [int(value) for value in base],
        "final_content_xyxy": [int(value) for value in final],
        "position": str(position),
        "reserved_bbox_xyxy": [int(value) for value in block],
    }
    panel_geometry["scene_content_xyxy"] = [int(value) for value in final]


def _draw_context_text_blocks(
    image: Image.Image,
    *,
    panel_geometry: Dict[str, Any],
    render_params: GraphRenderParams,
    layout_seed: int,
) -> Tuple[Dict[str, Any], ...]:
    """Draw optional longer non-answer context text outside graph content."""

    probability = max(0.0, min(1.0, float(render_params.context_block_probability)))
    max_elements = max(0, int(render_params.context_block_max_elements))
    if probability <= 0.0 or max_elements <= 0:
        panel_geometry["context_block_reservation"] = {
            "base_content_xyxy": [int(value) for value in panel_geometry["scene_content_xyxy"]],
            "final_content_xyxy": [int(value) for value in panel_geometry["scene_content_xyxy"]],
            "position": "none",
            "reserved_bbox_xyxy": [],
        }
        return ()
    rng = random.Random(int(layout_seed) + 158003)
    if rng.random() > float(probability):
        panel_geometry["context_block_reservation"] = {
            "base_content_xyxy": [int(value) for value in panel_geometry["scene_content_xyxy"]],
            "final_content_xyxy": [int(value) for value in panel_geometry["scene_content_xyxy"]],
            "position": "none",
            "reserved_bbox_xyxy": [],
        }
        return ()

    draw = ImageDraw.Draw(image)
    panel = tuple(int(value) for value in panel_geometry["scene_panel_xyxy"])
    content = tuple(int(value) for value in panel_geometry["scene_content_xyxy"])
    title_band = tuple(int(value) for value in panel_geometry["title_band_xyxy"])
    position = _weighted_context_choice(
        rng,
        values=_NODE_LINK_CONTEXT_BLOCK_POSITIONS,
        weights=render_params.context_block_position_weights,
    )
    clutter_level = _weighted_context_choice(
        rng,
        values=_NODE_LINK_CONTEXT_BLOCK_LEVELS,
        weights=render_params.context_block_clutter_level_weights,
    )
    line_targets = {"low": 2, "medium": 3, "high": 4}
    line_count = int(line_targets.get(str(clutter_level), 3))
    font_size = {"low": 10, "medium": 10, "high": 9}.get(str(clutter_level), 10)
    font = load_font(int(font_size), bold=False, font_family=str(render_params.font_family or ""))
    pad_x = 10
    pad_y = 8

    if str(position) in {"left", "right"}:
        block_w = int({"low": 128, "medium": 156, "high": 184}.get(str(clutter_level), 156))
        block_h = int({"low": 58, "medium": 78, "high": 98}.get(str(clutter_level), 78))
        x0 = int(content[0]) if str(position) == "left" else int(content[2] - block_w)
        min_y = int(content[1] + 10)
        max_y = int(max(min_y, content[3] - block_h - 10))
        y0 = int(rng.randint(min_y, max_y)) if max_y > min_y else min_y
    else:
        block_w = int({"low": 260, "medium": 340, "high": 430}.get(str(clutter_level), 340))
        block_h = int({"low": 48, "medium": 62, "high": 78}.get(str(clutter_level), 62))
        min_x = int(content[0] + 12)
        max_x = int(max(min_x, content[2] - block_w - 12))
        x0 = int(rng.randint(min_x, max_x)) if max_x > min_x else min_x
        y0 = int(content[1]) if str(position) == "top" else int(content[3] - block_h)
        if str(position) == "top":
            y0 = max(int(y0), int(title_band[3] + 6))
    block = (
        int(max(panel[0] + 10, min(x0, panel[2] - block_w - 10))),
        int(max(title_band[3] + 4, min(y0, panel[3] - block_h - 10))),
        int(max(panel[0] + 10, min(x0, panel[2] - block_w - 10)) + block_w),
        int(max(title_band[3] + 4, min(y0, panel[3] - block_h - 10)) + block_h),
    )

    phrase_count = {"low": 1, "medium": 2, "high": 3}.get(str(clutter_level), 2)
    phrases = rng.sample(
        list(_NODE_LINK_CONTEXT_BLOCK_PHRASES),
        k=min(int(phrase_count), len(_NODE_LINK_CONTEXT_BLOCK_PHRASES)),
    )
    text = " ".join(str(phrase) for phrase in phrases)
    lines = _wrap_context_text(
        draw,
        text=str(text),
        font=font,
        max_width_px=max(24, int(block_w - (2 * pad_x))),
        max_lines=int(line_count),
    )

    fill = tuple(int(round((2 * int(a) + int(b)) / 3.0)) for a, b in zip(render_params.panel_fill_rgb, render_params.background_color_rgb))
    outline = tuple(int(value) for value in render_params.panel_border_rgb)
    text_fill = tuple(int(round((2 * int(a) + int(b)) / 3.0)) for a, b in zip(render_params.title_color_rgb, render_params.panel_border_rgb))
    draw.rounded_rectangle(block, radius=5, fill=fill, outline=outline, width=1)
    y_cursor = int(block[1] + pad_y)
    line_step = max(11, int(font_size + 4))
    for line in lines:
        draw_traced_text(
            draw,
            xy=(int(block[0] + pad_x), int(y_cursor)),
            text=str(line),
            font=font,
            fill_rgb=text_fill,
            role="non_answer_context_text",
            required=False,
            extra_metadata={"answer_excluded": True, "kind": "context_block_line"},
        )
        y_cursor += int(line_step)

    _apply_context_block_reservation(panel_geometry, position=str(position), block_bbox=block)
    return (
        {
            "role": "non_answer_context_text",
            "kind": "context_block",
            "position": str(position),
            "clutter_level": str(clutter_level),
            "text": str(text),
            "bbox_xyxy": [int(value) for value in block],
            "font_family": str(render_params.font_family or ""),
        },
    )


def _draw_context_text_chips(
    image: Image.Image,
    *,
    panel_geometry: Mapping[str, Any],
    render_params: GraphRenderParams,
    layout_seed: int,
) -> Tuple[Dict[str, Any], ...]:
    """Draw small non-answer context chips in the title band."""

    probability = max(0.0, min(1.0, float(render_params.context_text_probability)))
    max_elements = max(0, int(render_params.context_text_max_elements))
    if probability <= 0.0 or max_elements <= 0:
        return ()
    rng = random.Random(int(layout_seed) + 130363)
    if rng.random() > float(probability):
        return ()

    title_band = tuple(int(value) for value in panel_geometry["title_band_xyxy"])
    panel = tuple(int(value) for value in panel_geometry["scene_panel_xyxy"])
    font_size = max(10, min(13, int(round(float(render_params.panel_title_font_size_px) * 0.50))))
    font = load_font(int(font_size), bold=True, font_family=str(render_params.font_family or ""))
    draw = ImageDraw.Draw(image)
    chip_fill = tuple(int(value) for value in render_params.panel_fill_rgb)
    chip_border = tuple(int(value) for value in render_params.panel_border_rgb)
    chip_text = tuple(int(value) for value in render_params.title_color_rgb)
    chip_positions = ("left", "right")[: int(max_elements)]
    elements: List[Dict[str, Any]] = []

    def draw_chip(text: str, side: str) -> None:
        raw_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
        text_w = int(raw_bbox[2] - raw_bbox[0])
        text_h = int(raw_bbox[3] - raw_bbox[1])
        pad_x = 9
        pad_y = 5
        chip_w = int(text_w + (2 * pad_x))
        chip_h = int(text_h + (2 * pad_y))
        y0 = int(round(0.5 * float(title_band[1] + title_band[3] - chip_h)))
        if str(side) == "right":
            x0 = int(panel[2] - 16 - chip_w)
        else:
            x0 = int(panel[0] + 16)
        box = (int(x0), int(y0), int(x0 + chip_w), int(y0 + chip_h))
        draw.rounded_rectangle(
            box,
            radius=max(5, int(round(float(chip_h) * 0.28))),
            fill=chip_fill,
            outline=chip_border,
            width=1,
        )
        draw_traced_text(
            draw,
            xy=(int(x0 + pad_x), int(y0 + pad_y - 1)),
            text=str(text),
            font=font,
            fill_rgb=chip_text,
            role="non_answer_context_text",
            required=False,
            extra_metadata={"answer_excluded": True, "kind": "title_band_chip"},
        )
        elements.append(
            {
                "role": "non_answer_context_text",
                "kind": "title_band_chip",
                "text": str(text),
                "bbox_xyxy": [int(value) for value in box],
                "font_family": str(render_params.font_family or ""),
            }
        )

    if "left" in chip_positions:
        draw_chip(str(rng.choice(_NODE_LINK_CONTEXT_LEFT_TEXTS)), "left")
    if "right" in chip_positions:
        suffix = int(rng.randint(2, 98))
        draw_chip(f"{str(rng.choice(_NODE_LINK_CONTEXT_RIGHT_TEXTS))} {suffix:02d}", "right")
    return tuple(elements)


def apply_graph_content_layout_jitter(
    panel_geometry: Dict[str, Any],
    *,
    render_params: GraphRenderParams,
    layout_seed: int,
) -> None:
    """Apply the shared bounded graph content-area jitter before projection."""

    _apply_content_layout_jitter(
        panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )


def draw_graph_context_text_blocks(
    image: Image.Image,
    *,
    panel_geometry: Dict[str, Any],
    render_params: GraphRenderParams,
    layout_seed: int,
) -> Tuple[Dict[str, Any], ...]:
    """Draw optional shared graph context blocks and reserve layout space."""

    return _draw_context_text_blocks(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )


def draw_graph_context_text_chips(
    image: Image.Image,
    *,
    panel_geometry: Mapping[str, Any],
    render_params: GraphRenderParams,
    layout_seed: int,
) -> Tuple[Dict[str, Any], ...]:
    """Draw optional shared graph context chips in the title band."""

    return _draw_context_text_chips(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )


__all__ = [
    "apply_graph_content_layout_jitter",
    "draw_graph_context_text_blocks",
    "draw_graph_context_text_chips",
    "_apply_content_layout_jitter",
    "_draw_context_text_blocks",
    "_draw_context_text_chips",
]
