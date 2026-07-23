"""Trace-backed non-answer context text layer for structured visual scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ..color_distance import coerce_rgb as _rgb
from ..context_text_assets import context_text_asset_version, sample_context_text
from ..font_assets import font_asset_version, font_role_trace, sample_font_family
from ..text_legibility import draw_text_traced
from ..text_rendering import load_font, resolve_text_stroke_fill


Color = Tuple[int, int, int]
BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class ContextTextElement:
    """One drawn non-answer context text element."""

    context_id: str
    role: str
    text: str
    bbox_xyxy: BBox
    manifest_path: str
    source_ids: Tuple[str, ...]
    row_index: int
    layout_mode: str
    font_family: str = ""
    font_role: str = "context"
    excluded_from_answer: bool = True

    def to_trace(self) -> dict[str, Any]:
        trace = {
            "context_id": str(self.context_id),
            "role": str(self.role),
            "text": str(self.text),
            "bbox_xyxy": [int(value) for value in self.bbox_xyxy],
            "manifest_path": str(self.manifest_path),
            "source_ids": [str(source_id) for source_id in self.source_ids],
            "row_index": int(self.row_index),
            "layout_mode": str(self.layout_mode),
            "font_family": str(self.font_family),
            "font_role": str(self.font_role),
            "excluded_from_answer": bool(self.excluded_from_answer),
        }
        if self.font_family:
            trace.update({k: v for k, v in font_role_trace(str(self.font_family), role=str(self.font_role)).items() if k != "font_family"})
        return trace


def _weighted_choice(rng: Any, weights: Mapping[str, Any], *, fallback: Sequence[str]) -> str:
    weighted: list[tuple[str, float]] = []
    for key, value in weights.items():
        try:
            weight = float(value)
        except Exception:
            continue
        if weight > 0:
            weighted.append((str(key), float(weight)))
    if not weighted:
        weighted = [(str(key), 1.0) for key in fallback]
    total = sum(weight for _, weight in weighted)
    cursor = rng.random() * float(total)
    running = 0.0
    for key, weight in weighted:
        running += float(weight)
        if cursor <= running:
            return str(key)
    return str(weighted[-1][0])


def _int_range_sample(rng: Any, *, params: Mapping[str, Any], min_key: str, max_key: str, fallback_min: int, fallback_max: int) -> int:
    low = int(params.get(str(min_key), int(fallback_min)))
    high = int(params.get(str(max_key), int(fallback_max)))
    if high < low:
        high = low
    return int(rng.randrange(int(low), int(high) + 1))


def _text_bbox_tuple(box: Sequence[float]) -> BBox:
    return (
        int(round(float(box[0]))),
        int(round(float(box[1]))),
        int(round(float(box[2]))),
        int(round(float(box[3]))),
    )


def _clip_bbox(box: Sequence[int], *, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = [int(value) for value in box[:4]]
    clipped_x0 = min(max(0, x0), max(0, int(width) - 1))
    clipped_y0 = min(max(0, y0), max(0, int(height) - 1))
    clipped_x1 = min(max(clipped_x0 + 1, x1), int(width))
    clipped_y1 = min(max(clipped_y0 + 1, y1), int(height))
    return (int(clipped_x0), int(clipped_y0), int(clipped_x1), int(clipped_y1))


def _fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: Any,
    max_width_px: int,
    stroke_width: int = 0,
) -> str:
    max_width = max(10, int(max_width_px))
    raw = " ".join(str(text).split())
    if not raw:
        return ""
    if draw.textbbox((0, 0), raw, font=font, stroke_width=int(stroke_width))[2] <= max_width:
        return raw
    ellipsis = "..."
    words = raw.split()
    fitted = ""
    for word in words:
        candidate = f"{fitted} {word}".strip()
        width = draw.textbbox((0, 0), f"{candidate}{ellipsis}", font=font, stroke_width=int(stroke_width))[2]
        if width > max_width:
            break
        fitted = candidate
    if fitted:
        return f"{fitted}{ellipsis}"
    chars = []
    for char in raw:
        candidate = "".join(chars) + char
        width = draw.textbbox((0, 0), f"{candidate}{ellipsis}", font=font, stroke_width=int(stroke_width))[2]
        if width > max_width:
            break
        chars.append(char)
    return f"{''.join(chars).strip()}{ellipsis}" if chars else ellipsis


def _wrap_text_to_box(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: Any,
    max_width_px: int,
    max_lines: int,
    stroke_width: int = 0,
) -> str:
    words = " ".join(str(text).split()).split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), candidate, font=font, stroke_width=int(stroke_width))[2]
        if width <= max(10, int(max_width_px)):
            current = candidate
            continue
        if current:
            lines.append(current)
        current = str(word)
        if len(lines) >= max(1, int(max_lines)):
            break
    if current and len(lines) < max(1, int(max_lines)):
        lines.append(current)
    if not lines:
        return ""
    if len(lines) >= max(1, int(max_lines)) and words:
        last = lines[-1]
        ellipsis = "..."
        while last and draw.textbbox((0, 0), f"{last}{ellipsis}", font=font, stroke_width=int(stroke_width))[2] > max(10, int(max_width_px)):
            last = " ".join(last.split()[:-1])
        lines[-1] = f"{last}{ellipsis}" if last else ellipsis
    return "\n".join(lines)


def _draw_context_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    fill: Color,
    anchor: str,
    max_width_px: int,
    canvas_width: int,
    canvas_height: int,
    stroke_width: int = 0,
) -> tuple[str, BBox]:
    fitted = _fit_text_to_width(
        draw,
        text=str(text),
        font=font,
        max_width_px=int(max_width_px),
        stroke_width=int(stroke_width),
    )
    stroke_fill = resolve_text_stroke_fill(tuple(fill))
    bbox = draw.textbbox(tuple(xy), fitted, font=font, anchor=str(anchor), stroke_width=int(stroke_width))
    draw_text_traced(
        draw,
        tuple(xy),
        fitted,
        font=font,
        anchor=str(anchor),
        fill=tuple(fill),
        stroke_width=int(stroke_width),
        stroke_fill=tuple(stroke_fill),
        role="non_answer_context_text",
        required=False,
        extra_metadata={"answer_excluded": True, "context_layer": True},
    )
    return fitted, _clip_bbox(
        _text_bbox_tuple(bbox),
        width=int(canvas_width),
        height=int(canvas_height),
    )


def _draw_wrapped_context_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    fill: Color,
    anchor: str,
    max_width_px: int,
    max_lines: int,
    canvas_width: int,
    canvas_height: int,
    stroke_width: int = 0,
    spacing_px: int = 3,
) -> tuple[str, BBox]:
    wrapped = _wrap_text_to_box(
        draw,
        text=str(text),
        font=font,
        max_width_px=int(max_width_px),
        max_lines=int(max_lines),
        stroke_width=int(stroke_width),
    )
    stroke_fill = resolve_text_stroke_fill(tuple(fill))
    bbox = draw.multiline_textbbox(
        tuple(xy),
        wrapped,
        font=font,
        anchor=str(anchor),
        spacing=int(spacing_px),
        stroke_width=int(stroke_width),
    )
    draw.multiline_text(
        tuple(xy),
        wrapped,
        font=font,
        anchor=str(anchor),
        fill=tuple(fill),
        spacing=int(spacing_px),
        stroke_width=int(stroke_width),
        stroke_fill=tuple(stroke_fill),
    )
    return wrapped, _clip_bbox(
        _text_bbox_tuple(bbox),
        width=int(canvas_width),
        height=int(canvas_height),
    )


def _draw_metric_chip(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    text: str,
    font: Any,
    fill_rgb: Color,
    border_rgb: Color,
    text_rgb: Color,
    canvas_width: int,
    canvas_height: int,
) -> tuple[str, BBox]:
    clipped = _clip_bbox(bbox, width=int(canvas_width), height=int(canvas_height))
    draw.rounded_rectangle(
        clipped,
        radius=max(4, int((clipped[3] - clipped[1]) / 3)),
        fill=tuple(fill_rgb),
        outline=tuple(border_rgb),
        width=1,
    )
    fitted, text_bbox = _draw_context_text(
        draw,
        xy=((clipped[0] + clipped[2]) / 2.0, (clipped[1] + clipped[3]) / 2.0),
        text=str(text),
        font=font,
        fill=tuple(text_rgb),
        anchor="mm",
        max_width_px=max(10, int(clipped[2] - clipped[0] - 12)),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        stroke_width=0,
    )
    combined = (
        min(clipped[0], text_bbox[0]),
        min(clipped[1], text_bbox[1]),
        max(clipped[2], text_bbox[2]),
        max(clipped[3], text_bbox[3]),
    )
    return fitted, _clip_bbox(combined, width=int(canvas_width), height=int(canvas_height))


def context_text_layer_metadata(
    elements: Iterable[ContextTextElement],
    *,
    enabled: bool,
    layout_mode: str,
    layout_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return trace metadata for one context-text layer."""

    element_records = [element.to_trace() for element in elements]
    return {
        "enabled": bool(enabled),
        "layout_mode": str(layout_mode),
        "layout_spec": dict(layout_spec or {}),
        "asset_version": context_text_asset_version(),
        "font_asset_version": font_asset_version(),
        "element_count": int(len(element_records)),
        "elements": element_records,
        "policy": {
            "answer_excluded_by_default": True,
            "requires_trace_ids_and_bboxes": True,
            "source": "assets/context_text",
        },
    }


def resolve_dashboard_context_layout(
    *,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any] | None = None,
    canvas_width: int,
    canvas_height: int,
    top_reserved_px: int = 64,
    bottom_reserved_px: int = 26,
    left_margin_px: int = 24,
    right_margin_px: int = 24,
) -> dict[str, Any]:
    """Resolve non-answer dashboard context-box placement for one instance."""

    resolved_params = params or {}
    enabled = bool(resolved_params.get("context_text_enabled", True))
    layout_mode = str(resolved_params.get("context_text_layout_mode", "reserved_context"))
    top_reserved = max(36, int(resolved_params.get("context_text_top_reserved_px", top_reserved_px)))
    bottom_reserved = max(18, int(resolved_params.get("context_text_bottom_reserved_px", bottom_reserved_px)))
    left_margin = max(8, int(resolved_params.get("context_text_left_margin_px", left_margin_px)))
    right_margin = max(8, int(resolved_params.get("context_text_right_margin_px", right_margin_px)))
    if not enabled:
        return {
            "enabled": False,
            "layout_mode": "none",
            "placement": "none",
            "box_count": 0,
            "top_reserved_px": int(top_reserved),
            "bottom_reserved_px": int(bottom_reserved),
            "left_margin_px": int(left_margin),
            "right_margin_px": int(right_margin),
        }
    rng = spawn_rng(int(instance_seed), f"{namespace}.context_layout")
    explicit_placement = resolved_params.get("context_text_placement")
    supported = ("right_sidebar", "left_sidebar", "bottom_band")
    if explicit_placement is not None:
        placement = str(explicit_placement)
        if placement not in set(supported):
            placement = "right_sidebar"
    else:
        raw_weights = resolved_params.get(
            "context_text_placement_weights",
            {"right_sidebar": 0.45, "left_sidebar": 0.25, "bottom_band": 0.30},
        )
        placement = _weighted_choice(
            rng,
            raw_weights if isinstance(raw_weights, Mapping) else {},
            fallback=supported,
        )
    box_count = _int_range_sample(
        rng,
        params=resolved_params,
        min_key="context_text_box_count_min",
        max_key="context_text_box_count_max",
        fallback_min=1,
        fallback_max=3,
    )
    if str(placement) == "bottom_band":
        box_count = min(int(box_count), 2)
    else:
        box_count = min(int(box_count), 3)
    sidebar_width = _int_range_sample(
        rng,
        params=resolved_params,
        min_key="context_text_sidebar_width_min_px",
        max_key="context_text_sidebar_width_max_px",
        fallback_min=210,
        fallback_max=280,
    )
    bottom_band_height = _int_range_sample(
        rng,
        params=resolved_params,
        min_key="context_text_bottom_band_height_min_px",
        max_key="context_text_bottom_band_height_max_px",
        fallback_min=118,
        fallback_max=178,
    )
    sidebar_gap = int(resolved_params.get("context_text_sidebar_gap_px", 14))
    bottom_gap = int(resolved_params.get("context_text_bottom_band_gap_px", 14))
    max_sidebar = max(180, int(canvas_width) - int(left_margin) - int(right_margin) - 720)
    sidebar_width = min(int(sidebar_width), int(max_sidebar))
    max_bottom_band = max(90, int(canvas_height) - int(top_reserved) - int(bottom_reserved) - 620)
    bottom_band_height = min(int(bottom_band_height), int(max_bottom_band))
    return {
        "enabled": True,
        "layout_mode": str(layout_mode),
        "placement": str(placement),
        "box_count": int(max(1, box_count)),
        "top_reserved_px": int(top_reserved),
        "bottom_reserved_px": int(bottom_reserved),
        "left_margin_px": int(left_margin),
        "right_margin_px": int(right_margin),
        "sidebar_width_px": int(sidebar_width),
        "sidebar_gap_px": int(sidebar_gap),
        "bottom_band_height_px": int(bottom_band_height),
        "bottom_band_gap_px": int(bottom_gap),
    }


def sample_dashboard_title(
    *,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic optional dashboard title selection."""

    resolved_params = params or {}
    if not bool(resolved_params.get("dashboard_title_enabled", True)):
        return {
            "enabled": False,
            "text": "",
            "manifest_path": "",
            "row_index": -1,
            "source_ids": [],
        }
    rng = spawn_rng(int(instance_seed), f"{namespace}.dashboard_title")
    drop_probability = float(resolved_params.get("dashboard_title_drop_probability", 0.18))
    if rng.random() < max(0.0, min(1.0, float(drop_probability))):
        return {
            "enabled": False,
            "text": "",
            "manifest_path": "",
            "row_index": -1,
            "source_ids": [],
        }
    selection = sample_context_text("phrases/headlines.txt", rng=rng)
    return {
        "enabled": True,
        "text": str(selection.text),
        "manifest_path": str(selection.manifest_path),
        "row_index": int(selection.row_index),
        "source_ids": [str(source_id) for source_id in selection.source_ids],
    }


def draw_dashboard_reserved_margin_context(
    image: Image.Image,
    *,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any] | None = None,
    text_rgb: Color = (35, 40, 48),
    muted_text_rgb: Color = (90, 96, 108),
    panel_fill_rgb: Color = (255, 255, 255),
    panel_border_rgb: Color = (200, 207, 216),
    accent_rgb: Color = (35, 99, 180),
    top_reserved_px: int = 64,
    bottom_reserved_px: int = 26,
    left_margin_px: int = 24,
    right_margin_px: int = 24,
    layout_spec: Mapping[str, Any] | None = None,
) -> Tuple[ContextTextElement, ...]:
    """Draw dashboard context text only in reserved non-chart regions."""

    resolved_params = params or {}
    if not bool(resolved_params.get("context_text_enabled", True)):
        return tuple()

    width, height = image.size
    resolved_layout = dict(
        layout_spec
        or resolve_dashboard_context_layout(
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            params=resolved_params,
            canvas_width=int(width),
            canvas_height=int(height),
            top_reserved_px=int(top_reserved_px),
            bottom_reserved_px=int(bottom_reserved_px),
            left_margin_px=int(left_margin_px),
            right_margin_px=int(right_margin_px),
        )
    )
    top_reserved = max(36, int(resolved_layout.get("top_reserved_px", top_reserved_px)))
    bottom_reserved = max(18, int(resolved_layout.get("bottom_reserved_px", bottom_reserved_px)))
    left_margin = max(8, int(resolved_layout.get("left_margin_px", left_margin_px)))
    right_margin = max(8, int(resolved_layout.get("right_margin_px", right_margin_px)))
    layout_mode = f"{resolved_layout.get('layout_mode', 'reserved_context')}:{resolved_layout.get('placement', 'none')}"
    placement = str(resolved_layout.get("placement", "right_sidebar"))
    box_count = max(1, int(resolved_layout.get("box_count", 1)))
    side_reserved = placement in {"right_sidebar", "left_sidebar"}
    bottom_reserved_box = placement == "bottom_band"
    sidebar_width = int(resolved_layout.get("sidebar_width_px", 240))
    sidebar_gap = int(resolved_layout.get("sidebar_gap_px", 14))
    bottom_band_height = int(resolved_layout.get("bottom_band_height_px", 140))
    bottom_band_gap = int(resolved_layout.get("bottom_band_gap_px", 14))
    rng = spawn_rng(int(instance_seed), f"{namespace}.context_text_layer")
    draw = ImageDraw.Draw(image)

    text_color = _rgb(text_rgb, (35, 40, 48))
    muted_color = _rgb(muted_text_rgb, (90, 96, 108))
    fill_color = _rgb(panel_fill_rgb, (255, 255, 255))
    border_color = _rgb(panel_border_rgb, (200, 207, 216))
    accent_color = _rgb(accent_rgb, (35, 99, 180))

    chrome_font_family = sample_font_family(
        role="context",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.context_chrome_font",
        params=resolved_params,
        exclude_tags=("mono", "display", "script", "handwriting"),
        explicit_key="context_text_chrome_font_family",
        weights_key="context_text_font_family_weights",
    )
    chip_font_family = sample_font_family(
        role="context",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.context_chip_font",
        params=resolved_params,
        exclude_tags=("display", "script", "handwriting"),
        explicit_key="context_text_chip_font_family",
        weights_key="context_text_font_family_weights",
    )
    header_font = load_font(13, bold=True, font_family=chrome_font_family)
    small_font = load_font(11, bold=False, font_family=chrome_font_family)
    chip_font = load_font(11, bold=True, font_family=chip_font_family)

    elements: list[ContextTextElement] = []

    def add_text(
        role: str,
        manifest: str,
        xy: Tuple[float, float],
        anchor: str,
        font: Any,
        fill: Color,
        max_width: int,
        font_family: str,
    ) -> None:
        selection = sample_context_text(manifest, rng=rng)
        fitted, bbox = _draw_context_text(
            draw,
            xy=xy,
            text=str(selection.text),
            font=font,
            fill=fill,
            anchor=str(anchor),
            max_width_px=int(max_width),
            canvas_width=int(width),
            canvas_height=int(height),
            stroke_width=1 if str(role) in {"header", "source_note"} else 0,
        )
        elements.append(
            ContextTextElement(
                context_id=f"context_{len(elements):02d}",
                role=str(role),
                text=str(fitted),
                bbox_xyxy=tuple(bbox),
                manifest_path=str(selection.manifest_path),
                source_ids=tuple(selection.source_ids),
                row_index=int(selection.row_index),
                layout_mode=str(layout_mode),
                font_family=str(font_family),
            )
        )

    add_text(
        "header",
        "phrases/headlines.txt",
        (float(left_margin), 18.0),
        "lm",
        header_font,
        text_color,
        max(160, int(width * 0.31)),
        chrome_font_family,
    )
    add_text(
        "source_note",
        "phrases/source_notes.txt",
        (float(width - right_margin), 18.0),
        "rm",
        small_font,
        muted_color,
        max(160, int(width * 0.31)),
        chrome_font_family,
    )
    add_text(
        "footer",
        "phrases/footers.txt",
        (float(left_margin), float(height - max(10, bottom_reserved // 2))),
        "lm",
        small_font,
        muted_color,
        max(220, int(width * 0.42)),
        chrome_font_family,
    )

    chip_selection = sample_context_text("phrases/metric_snippets.txt", rng=rng)
    usable_left = int(left_margin + (sidebar_width + sidebar_gap if placement == "left_sidebar" else 0))
    usable_right = int(width - right_margin - (sidebar_width + sidebar_gap if placement == "right_sidebar" else 0))
    chip_width = min(210, max(130, int(width * 0.16)))
    chip_height = 20
    chip_bbox = (
        int(max(usable_left + 220, usable_right - chip_width)),
        int(height - bottom_reserved + max(2, (bottom_reserved - chip_height) // 2)),
        int(usable_right),
        int(height - bottom_reserved + max(2, (bottom_reserved - chip_height) // 2) + chip_height),
    )
    chip_text, chip_trace_bbox = _draw_metric_chip(
        draw,
        bbox=chip_bbox,
        text=str(chip_selection.text),
        font=chip_font,
        fill_rgb=fill_color,
        border_rgb=border_color,
        text_rgb=accent_color,
        canvas_width=int(width),
        canvas_height=int(height),
    )
    elements.append(
        ContextTextElement(
            context_id=f"context_{len(elements):02d}",
            role="metric_snippet",
            text=str(chip_text),
            bbox_xyxy=tuple(chip_trace_bbox),
            manifest_path=str(chip_selection.manifest_path),
            source_ids=tuple(chip_selection.source_ids),
            row_index=int(chip_selection.row_index),
            layout_mode=str(layout_mode),
            font_family=str(chip_font_family),
        )
    )

    context_box_bboxes: list[BBox] = []
    if side_reserved:
        if placement == "left_sidebar":
            x0 = int(left_margin)
            x1 = int(x0 + max(180, sidebar_width))
        else:
            x1 = int(width - right_margin)
            x0 = int(x1 - max(180, sidebar_width))
        y0 = int(max(72, top_reserved + 6))
        y1 = int(max(y0 + 160, height - bottom_reserved - 10))
        total_height = max(80, int(y1 - y0))
        gap = 12
        selected_box_count = min(int(box_count), 3)
        slot_height = max(70, int((total_height - gap * (selected_box_count - 1)) / selected_box_count))
        for index in range(selected_box_count):
            by0 = y0 + index * (slot_height + gap)
            by1 = y1 if index == selected_box_count - 1 else min(y1, by0 + slot_height)
            context_box_bboxes.append(_clip_bbox((x0, by0, x1, by1), width=int(width), height=int(height)))
    elif bottom_reserved_box:
        x0 = int(left_margin)
        x1 = int(width - right_margin)
        y1 = int(height - bottom_reserved - 8)
        y0 = int(y1 - max(90, bottom_band_height))
        selected_box_count = min(int(box_count), 2)
        gap = 14
        box_width = max(220, int((x1 - x0 - gap * (selected_box_count - 1)) / selected_box_count))
        for index in range(selected_box_count):
            bx0 = x0 + index * (box_width + gap)
            bx1 = x1 if index == selected_box_count - 1 else min(x1, bx0 + box_width)
            context_box_bboxes.append(_clip_bbox((bx0, y0, bx1, y1), width=int(width), height=int(height)))

    for box_index, sidebar_bbox in enumerate(context_box_bboxes):
        box_font_family = sample_font_family(
            role="context",
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.context_box_font.{box_index}",
            params=resolved_params,
            exclude_tags=("mono", "display", "script", "handwriting"),
            explicit_key="context_text_box_font_family",
            weights_key="context_text_font_family_weights",
        )
        paragraph_title_font = load_font(12, bold=True, font_family=box_font_family)
        paragraph_font = load_font(11, bold=False, font_family=box_font_family)
        box_small_font = load_font(11, bold=False, font_family=box_font_family)
        draw.rounded_rectangle(
            sidebar_bbox,
            radius=8,
            fill=tuple(fill_color),
            outline=tuple(border_color),
            width=1,
        )
        draw.line(
            (
                sidebar_bbox[0] + 12,
                sidebar_bbox[1] + 34,
                sidebar_bbox[2] - 12,
                sidebar_bbox[1] + 34,
            ),
            fill=tuple(border_color),
            width=1,
        )
        title_manifest = "phrases/callout_phrases.txt" if box_index % 2 == 0 else "phrases/sidebar_notes.txt"
        title_selection = sample_context_text(title_manifest, rng=rng)
        title_text, title_bbox = _draw_context_text(
            draw,
            xy=(sidebar_bbox[0] + 12, sidebar_bbox[1] + 18),
            text=str(title_selection.text),
            font=paragraph_title_font,
            fill=tuple(text_color),
            anchor="lm",
            max_width_px=max(60, sidebar_bbox[2] - sidebar_bbox[0] - 24),
            canvas_width=int(width),
            canvas_height=int(height),
            stroke_width=0,
        )
        elements.append(
            ContextTextElement(
                context_id=f"context_{len(elements):02d}",
                role="context_box_heading",
                text=str(title_text),
                bbox_xyxy=tuple(title_bbox),
                manifest_path=str(title_selection.manifest_path),
                source_ids=tuple(title_selection.source_ids),
                row_index=int(title_selection.row_index),
                layout_mode=str(layout_mode),
                font_family=str(box_font_family),
            )
        )
        box_height = int(sidebar_bbox[3] - sidebar_bbox[1])
        has_note = box_height >= 150
        if box_height >= 190:
            body_manifest = "paragraphs/context_long_blocks.txt"
        elif box_index == 0:
            body_manifest = "paragraphs/context_template_blocks.txt"
        else:
            body_manifest = _weighted_choice(
                rng,
                {
                    "sentences/context_template_sentences.txt": 1.0,
                    "phrases/sidebar_notes.txt": 1.0,
                    "phrases/captions.txt": 0.8,
                    "phrases/legend_notes.txt": 0.6,
                    "phrases/metric_snippets.txt": 0.4,
                },
                fallback=("sentences/context_template_sentences.txt",),
            )
        body_selection = sample_context_text(body_manifest, rng=rng)
        body_line_budget = max(3, int((box_height - (112 if has_note else 70)) / 18))
        body_text, body_bbox = _draw_wrapped_context_text(
            draw,
            xy=(sidebar_bbox[0] + 12, sidebar_bbox[1] + 48),
            text=str(body_selection.text),
            font=paragraph_font,
            fill=tuple(muted_color),
            anchor="la",
            max_width_px=max(80, sidebar_bbox[2] - sidebar_bbox[0] - 24),
            max_lines=int(body_line_budget),
            canvas_width=int(width),
            canvas_height=int(height),
            stroke_width=0,
            spacing_px=4,
        )
        elements.append(
            ContextTextElement(
                context_id=f"context_{len(elements):02d}",
                role="context_box_body",
                text=str(body_text),
                bbox_xyxy=tuple(body_bbox),
                manifest_path=str(body_selection.manifest_path),
                source_ids=tuple(body_selection.source_ids),
                row_index=int(body_selection.row_index),
                layout_mode=str(layout_mode),
                font_family=str(box_font_family),
            )
        )
        if has_note:
            note_selection = sample_context_text("phrases/legend_notes.txt", rng=rng)
            note_text, note_bbox = _draw_wrapped_context_text(
                draw,
                xy=(sidebar_bbox[0] + 12, sidebar_bbox[3] - 38),
                text=str(note_selection.text),
                font=box_small_font,
                fill=tuple(accent_color),
                anchor="la",
                max_width_px=max(80, sidebar_bbox[2] - sidebar_bbox[0] - 24),
                max_lines=2,
                canvas_width=int(width),
                canvas_height=int(height),
                stroke_width=0,
                spacing_px=2,
            )
            elements.append(
                ContextTextElement(
                    context_id=f"context_{len(elements):02d}",
                    role="context_box_note",
                    text=str(note_text),
                    bbox_xyxy=tuple(note_bbox),
                    manifest_path=str(note_selection.manifest_path),
                    source_ids=tuple(note_selection.source_ids),
                    row_index=int(note_selection.row_index),
                    layout_mode=str(layout_mode),
                    font_family=str(box_font_family),
                )
            )

    # Thin separators make the added text read as dashboard chrome rather than
    # extra plotted data.
    line_y_top = max(42, min(int(top_reserved) - 4, 66))
    line_y_bottom = min(int(height) - int(bottom_reserved), int(height) - 22)
    line_left = int(left_margin + (sidebar_width + sidebar_gap if placement == "left_sidebar" else 0))
    right_line = int(width - right_margin - (sidebar_width + sidebar_gap if placement == "right_sidebar" else 0))
    if bottom_reserved_box:
        line_y_bottom = int(height - bottom_reserved - bottom_band_height - bottom_band_gap)
    if right_line > line_left + 20:
        draw.line((line_left, line_y_top, right_line, line_y_top), fill=tuple(border_color), width=1)
        draw.line((line_left, line_y_bottom, right_line, line_y_bottom), fill=tuple(border_color), width=1)

    return tuple(elements)


__all__ = [
    "ContextTextElement",
    "context_text_layer_metadata",
    "draw_dashboard_reserved_margin_context",
    "resolve_dashboard_context_layout",
    "sample_dashboard_title",
]
