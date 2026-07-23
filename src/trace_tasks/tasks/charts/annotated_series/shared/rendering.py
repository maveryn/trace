"""Rendering helpers for the annotated-series chart scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.charts.annotated_series.shared.layout import (
    apply_context_margin_overrides,
    resolve_context_layout,
)
from trace_tasks.tasks.charts.annotated_series.shared.defaults import (
    FALLBACK_CHART_DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    SCENE_ID,
    SCENE_NAMESPACE,
    rendering_value,
)
from trace_tasks.tasks.charts.annotated_series.shared.state import (
    FinalRender,
    MarkupRender,
    RenderedBaseSeries,
    SeriesSample,
)
from trace_tasks.tasks.charts.shared.chart_scene_labeled import render_labeled_chart_scene
from trace_tasks.tasks.charts.shared.chart_scene_primitives import value_axis_render_metadata
from trace_tasks.tasks.charts.shared.chart_scene_types import RenderedChartScene
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.tasks.charts.shared.labeled_chart_marks import (
    build_chart_mark_specs,
    resolve_chart_mark_colors,
)
from trace_tasks.tasks.charts.shared.labeled_chart_render_params import resolve_chart_render_params_for_task
from trace_tasks.tasks.shared.bbox_projection import bbox_union, round_bbox
from trace_tasks.tasks.shared.context_text_assets import sample_context_text
from trace_tasks.tasks.shared.font_assets import font_asset_version, sample_font_family
from trace_tasks.tasks.shared.render_variation import resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.text_rendering import (
    draw_text_centered,
    load_font,
    temporary_default_font_family,
)
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.visual_style.context_layer import (
    ContextTextElement,
    context_text_layer_metadata,
    draw_dashboard_reserved_margin_context,
)


RGB = tuple[int, int, int]


def _fit_context_text(draw: ImageDraw.ImageDraw, text: str, *, font: Any, max_width_px: int) -> str:
    raw = " ".join(str(text).split())
    if not raw:
        return ""
    max_width = max(20, int(max_width_px))
    if draw.textbbox((0, 0), raw, font=font)[2] <= int(max_width):
        return raw
    suffix = "..."
    words = raw.split()
    fitted = ""
    for word in words:
        candidate = f"{fitted} {word}".strip()
        if draw.textbbox((0, 0), f"{candidate}{suffix}", font=font)[2] > int(max_width):
            break
        fitted = candidate
    if fitted:
        return f"{fitted}{suffix}"
    chars: list[str] = []
    for char in raw:
        candidate = "".join(chars) + str(char)
        if draw.textbbox((0, 0), f"{candidate}{suffix}", font=font)[2] > int(max_width):
            break
        chars.append(str(char))
    return f"{''.join(chars).strip()}{suffix}" if chars else suffix


def _draw_light_context_text(
    draw: ImageDraw.ImageDraw,
    *,
    elements: list[ContextTextElement],
    rng: Any,
    role: str,
    manifest_path: str,
    xy: tuple[float, float],
    anchor: str,
    font: Any,
    font_family: str,
    fill_rgb: RGB,
    max_width_px: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """Draw one light-context text element and record its clipped bbox.

    The helper owns visual placement/provenance for non-answer context text
    only; it must not influence chart targets, answers, or annotations.
    """
    selection = sample_context_text(str(manifest_path), rng=rng)
    fitted = _fit_context_text(draw, str(selection.text), font=font, max_width_px=int(max_width_px))
    try:
        bbox_raw = draw.textbbox(tuple(xy), str(fitted), font=font, anchor=str(anchor))
    except TypeError:
        bbox_raw = draw.textbbox(tuple(xy), str(fitted), font=font)
    draw_text_traced(
        draw,
        tuple(xy),
        str(fitted),
        font=font,
        fill=tuple(fill_rgb),
        anchor=str(anchor),
        role="readout",
        required=False,
    )
    bbox = (
        max(0, min(int(canvas_width) - 1, int(round(float(bbox_raw[0]))))),
        max(0, min(int(canvas_height) - 1, int(round(float(bbox_raw[1]))))),
        max(1, min(int(canvas_width), int(round(float(bbox_raw[2]))))),
        max(1, min(int(canvas_height), int(round(float(bbox_raw[3]))))),
    )
    elements.append(
        ContextTextElement(
            context_id=f"context_{len(elements):02d}",
            role=str(role),
            text=str(fitted),
            bbox_xyxy=tuple(int(value) for value in bbox),
            manifest_path=str(selection.manifest_path),
            source_ids=tuple(str(source_id) for source_id in selection.source_ids),
            row_index=int(selection.row_index),
            layout_mode="light_context:top_bottom_notes",
            font_family=str(font_family),
        )
    )


def _rgb_role(information_style_meta: Mapping[str, Any], role: str, fallback: RGB) -> RGB:
    roles = information_style_meta.get("roles_rgb", {})
    if isinstance(roles, Mapping):
        value = roles.get(str(role))
        if isinstance(value, Sequence) and len(value) >= 3:
            return (int(value[0]), int(value[1]), int(value[2]))
    return tuple(int(value) for value in fallback)


def draw_context_layer(
    image: Image.Image,
    *,
    context_layout: Mapping[str, Any],
    information_style_meta: Mapping[str, Any],
    instance_seed: int,
) -> tuple[ContextTextElement, ...]:
    """Draw optional non-answer context text and return traceable records.

    Context text is visual clutter only. It documents rendered surroundings for
    audit/trace purposes without changing answer or annotation binding.
    """
    if not bool(context_layout.get("enabled", False)):
        return tuple()

    raw_context_params = context_layout.get("context_params", {})
    resolved_context_params = dict(raw_context_params) if isinstance(raw_context_params, Mapping) else {}
    text_rgb = _rgb_role(information_style_meta, "text", (35, 40, 48))
    muted_rgb = _rgb_role(information_style_meta, "muted_text", (90, 96, 108))
    panel_fill_rgb = _rgb_role(information_style_meta, "panel_fill", (255, 255, 255))
    panel_border_rgb = _rgb_role(information_style_meta, "panel_border", (200, 207, 216))
    accent_rgb = _rgb_role(information_style_meta, "accent", (35, 99, 180))

    mode = str(context_layout.get("mode", context_layout.get("layout_mode", "clean")))
    if mode == "paragraph_box":
        return draw_dashboard_reserved_margin_context(
            image,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.context",
            params=resolved_context_params,
            text_rgb=text_rgb,
            muted_text_rgb=muted_rgb,
            panel_fill_rgb=panel_fill_rgb,
            panel_border_rgb=panel_border_rgb,
            accent_rgb=accent_rgb,
            top_reserved_px=int(context_layout.get("top_reserved_px", 64)),
            bottom_reserved_px=int(context_layout.get("bottom_reserved_px", 28)),
            left_margin_px=int(context_layout.get("left_margin_px", 24)),
            right_margin_px=int(context_layout.get("right_margin_px", 24)),
            layout_spec=context_layout,
        )

    draw = ImageDraw.Draw(image)
    font_family = sample_font_family(
        role="context",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.context_light_font",
        params=resolved_context_params,
        exclude_tags=("mono", "display", "script", "handwriting"),
        explicit_key="context_text_light_font_family",
        weights_key="context_text_font_family_weights",
    )
    header_font = load_font(14, bold=True, font_family=font_family)
    small_font = load_font(12, bold=False, font_family=font_family)
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.context_light")
    width, height = image.size
    elements: list[ContextTextElement] = []
    left_margin = 24
    right_margin = 24
    _draw_light_context_text(
        draw,
        elements=elements,
        rng=rng,
        role="header",
        manifest_path="phrases/headlines.txt",
        xy=(float(left_margin), 12.0),
        anchor="la",
        font=header_font,
        font_family=str(font_family),
        fill_rgb=text_rgb,
        max_width_px=max(180, int(width * 0.34)),
        canvas_width=int(width),
        canvas_height=int(height),
    )
    _draw_light_context_text(
        draw,
        elements=elements,
        rng=rng,
        role="source_note",
        manifest_path="phrases/source_notes.txt",
        xy=(float(width - right_margin), 12.0),
        anchor="ra",
        font=small_font,
        font_family=str(font_family),
        fill_rgb=muted_rgb,
        max_width_px=max(180, int(width * 0.34)),
        canvas_width=int(width),
        canvas_height=int(height),
    )
    _draw_light_context_text(
        draw,
        elements=elements,
        rng=rng,
        role="footer",
        manifest_path="phrases/footers.txt",
        xy=(float(left_margin), float(height - 18)),
        anchor="lm",
        font=small_font,
        font_family=str(font_family),
        fill_rgb=muted_rgb,
        max_width_px=max(220, int(width * 0.45)),
        canvas_width=int(width),
        canvas_height=int(height),
    )
    draw.line((left_margin, 40, width - right_margin, 40), fill=panel_border_rgb, width=1)
    draw.line((left_margin, height - 38, width - right_margin, height - 38), fill=panel_border_rgb, width=1)
    return tuple(elements)


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, center: tuple[float, float], font: Any) -> list[float]:
    try:
        raw = draw.textbbox((0, 0), str(text), font=font)
        width = float(raw[2] - raw[0])
        height = float(raw[3] - raw[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        width = float(width)
        height = float(height)
    return round_bbox(
        [
            float(center[0]) - 0.5 * float(width),
            float(center[1]) - 0.5 * float(height),
            float(center[0]) + 0.5 * float(width),
            float(center[1]) + 0.5 * float(height),
        ]
    )


def _sample_short_markup_label(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    manifest_key: str,
    max_chars_key: str,
    sample_attempts_key: str,
    fallback_text: str,
    fallback_max_chars: int,
    draw: ImageDraw.ImageDraw | None = None,
    font: Any | None = None,
    max_width_px: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Sample compact decorative annotation text with provenance metadata.

    The sampled text may label visible markup, but it never determines the
    answer or annotation target selected by a public task.
    """
    manifest_path = str(rendering_value(params, str(manifest_key), "phrases/callout_phrases.txt"))
    max_chars = max(4, int(rendering_value(params, str(max_chars_key), int(fallback_max_chars))))
    sample_attempts = max(1, int(rendering_value(params, str(sample_attempts_key), 128)))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{str(namespace)}")
    width_limit = int(max_width_px) if max_width_px is not None else None
    last_short: tuple[str, dict[str, Any]] | None = None
    for draw_index in range(int(sample_attempts)):
        selection = sample_context_text(manifest_path, rng=rng)
        candidate = " ".join(str(selection.text).split())
        if not candidate or len(candidate) > int(max_chars):
            continue
        trace = {
            "text": str(candidate),
            "manifest_path": str(selection.manifest_path),
            "row_index": int(selection.row_index),
            "source_ids": [str(source_id) for source_id in selection.source_ids],
            "max_chars": int(max_chars),
            "sample_attempts": int(sample_attempts),
            "selected_after_draws": int(draw_index + 1),
        }
        last_short = (str(candidate), dict(trace))
        if draw is not None and font is not None and width_limit is not None:
            try:
                text_width = int(draw.textbbox((0, 0), str(candidate), font=font)[2])
            except Exception:
                text_width = int(draw.textsize(str(candidate), font=font)[0])
            if int(text_width) > int(width_limit):
                continue
        return str(candidate), dict(trace)
    if last_short is not None:
        text, trace = last_short
        trace = dict(trace)
        trace["selected_after_draws"] = int(sample_attempts)
        trace["width_fallback"] = bool(draw is not None and font is not None and width_limit is not None)
        return str(text), trace
    fallback = str(fallback_text)[: int(max_chars)].strip() or "Note"
    return fallback, {
        "text": str(fallback),
        "manifest_path": str(manifest_path),
        "row_index": -1,
        "source_ids": [],
        "max_chars": int(max_chars),
        "sample_attempts": int(sample_attempts),
        "selected_after_draws": int(sample_attempts),
        "fallback_used": True,
    }


def _trace_by_label(rendered_scene: RenderedChartScene) -> dict[str, dict[str, Any]]:
    return {str(item["label"]): dict(item) for item in rendered_scene.mark_traces}


def _bbox_overlap_area(box_a: Sequence[float], box_b: Sequence[float], *, padding: float = 0.0) -> float:
    b0 = float(box_b[0]) - float(padding)
    b1 = float(box_b[1]) - float(padding)
    b2 = float(box_b[2]) + float(padding)
    b3 = float(box_b[3]) + float(padding)
    width = min(float(box_a[2]), b2) - max(float(box_a[0]), b0)
    height = min(float(box_a[3]), b3) - max(float(box_a[1]), b1)
    return max(0.0, float(width)) * max(0.0, float(height))


def _choose_callout_box(
    *,
    rendered_scene: RenderedChartScene,
    anchor_label: str,
    endpoint_label: str,
    anchor_center: Sequence[float],
    box_width: float,
    box_height: float,
    gap: float,
) -> list[float]:
    """Choose a callout box that does not obscure task-critical marks.

    Candidate boxes stay inside the plot area. The scoring first protects the
    anchor and endpoint marks, then all chart marks, and only then prefers a
    shorter leader line to the anchor.
    """
    traces = _trace_by_label(rendered_scene)
    plot_left, plot_top, plot_right, plot_bottom = [float(value) for value in rendered_scene.plot_bbox_px]
    left_x = float(plot_left) + float(gap)
    right_x = float(plot_right) - float(box_width) - float(gap)
    top_y = float(plot_top) + float(gap)
    bottom_y = float(plot_bottom) - float(box_height) - float(gap)
    mid_y = max(top_y, min(bottom_y, float(anchor_center[1]) - 0.5 * float(box_height)))
    center_y = max(top_y, min(bottom_y, 0.5 * (float(plot_top) + float(plot_bottom) - float(box_height))))

    side_order = (
        (right_x, left_x)
        if float(anchor_center[0]) < 0.5 * (float(plot_left) + float(plot_right))
        else (left_x, right_x)
    )
    y_values = (top_y, bottom_y, mid_y, center_y)
    candidates: list[list[float]] = []
    seen: set[tuple[int, int]] = set()
    for x0 in side_order:
        for y0 in y_values:
            key = (int(round(float(x0))), int(round(float(y0))))
            if key in seen:
                continue
            seen.add(key)
            candidates.append([float(x0), float(y0), float(x0) + float(box_width), float(y0) + float(box_height)])

    all_mark_boxes = [list(item["mark_bbox_px"]) for item in traces.values()]
    critical_boxes = [
        list(traces[str(anchor_label)]["mark_bbox_px"]),
        list(traces[str(endpoint_label)]["mark_bbox_px"]),
    ]

    def score(box: Sequence[float]) -> tuple[float, float, float]:
        critical_overlap = sum(_bbox_overlap_area(box, mark_box, padding=8.0) for mark_box in critical_boxes)
        mark_overlap = sum(_bbox_overlap_area(box, mark_box, padding=4.0) for mark_box in all_mark_boxes)
        center = (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))
        leader_length = ((center[0] - float(anchor_center[0])) ** 2 + (center[1] - float(anchor_center[1])) ** 2) ** 0.5
        return (float(critical_overlap), float(mark_overlap), float(leader_length))

    return round_bbox(min(candidates, key=score))


def render_base_series(
    sample: SeriesSample,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedBaseSeries:
    """Render the unannotated chart scene shared by all scene objectives.

    This function owns chart style, background, fonts, axis layout, and base
    mark geometry. It deliberately does not choose task targets or answers.
    """
    mark_style = resolve_chart_mark_colors(
        params,
        render_defaults=RENDERING_DEFAULTS,
        defaults=FALLBACK_CHART_DEFAULTS,
        instance_seed=int(instance_seed),
        scene_variant=str(sample.scene_variant),
        mark_count=len(sample.labels),
    )
    marks = build_chart_mark_specs(
        labels=sample.labels,
        values=sample.values,
        scene_variant=str(sample.scene_variant),
        mark_style=mark_style,
    )
    context_layout = resolve_context_layout(
        params=params,
        instance_seed=int(instance_seed),
        canvas_width=int(rendering_value(params, "canvas_width", FALLBACK_CHART_DEFAULTS.canvas_width)),
        canvas_height=int(rendering_value(params, "canvas_height", FALLBACK_CHART_DEFAULTS.canvas_height)),
    )
    render_input_params = apply_context_margin_overrides(
        {**dict(params), **mark_style},
        context_layout=context_layout,
    )
    render_params = resolve_chart_render_params_for_task(
        render_input_params,
        render_defaults=RENDERING_DEFAULTS,
        defaults=FALLBACK_CHART_DEFAULTS,
        instance_seed=int(instance_seed),
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        render_params=render_params,
    )
    chart_font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
        explicit_key="chart_font_family",
        weights_key="chart_font_family_weights",
    )
    annotation_font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.annotation_font",
        params=params,
        explicit_key="annotation_font_family",
        weights_key="annotation_font_family_weights",
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_labeled_chart_scene(
            background,
            scene_variant=str(sample.scene_variant),
            marks=marks,
            render_params=render_params,
            instance_seed=int(instance_seed),
        )
    return RenderedBaseSeries(
        image=rendered_scene.image,
        rendered_scene=rendered_scene,
        context_layout=dict(context_layout),
        render_params=render_params,
        mark_style=dict(mark_style),
        background_meta=dict(background_meta),
        information_style_meta=dict(information_style_meta),
        chart_font_family=str(chart_font_family),
        annotation_font_family=str(annotation_font_family),
    )


def draw_callout_markup(
    *,
    base: RenderedBaseSeries,
    anchor_label: str,
    endpoint_label: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> MarkupRender:
    """Draw a callout anchored to a task-selected chart mark.

    The public task owns the anchor/endpoint semantics. This function only
    places the callout box, leader line, and geometry for selected labels.
    """
    traces = _trace_by_label(base.rendered_scene)
    anchor_trace = traces[str(anchor_label)]
    anchor_center = [float(value) for value in anchor_trace["mark_center_px"]]
    box_width = float(params.get("callout_box_width_px", group_default(RENDERING_DEFAULTS, "callout_box_width_px", 138)))
    box_height = float(params.get("callout_box_height_px", group_default(RENDERING_DEFAULTS, "callout_box_height_px", 48)))
    gap = 18.0
    callout_box = _choose_callout_box(
        rendered_scene=base.rendered_scene,
        anchor_label=str(anchor_label),
        endpoint_label=str(endpoint_label),
        anchor_center=anchor_center,
        box_width=float(box_width),
        box_height=float(box_height),
        gap=float(gap),
    )
    callout_center = (0.5 * (callout_box[0] + callout_box[2]), 0.5 * (callout_box[1] + callout_box[3]))

    fill_rgb = resolve_render_rgb(
        params,
        RENDERING_DEFAULTS,
        "callout_fill_rgb",
        (255, 255, 255),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )
    outline_rgb = resolve_render_rgb(
        params,
        RENDERING_DEFAULTS,
        "callout_outline_rgb",
        (48, 98, 170),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )
    text_rgb = resolve_render_rgb(
        params,
        RENDERING_DEFAULTS,
        "callout_text_rgb",
        (38, 42, 50),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )
    text_stroke_rgb = _rgb_role(base.information_style_meta, "text_stroke", (255, 255, 255))
    corner_radius = resolve_render_int(
        params,
        RENDERING_DEFAULTS,
        "callout_corner_radius_px",
        8,
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )
    arrow_width = resolve_render_int(
        params,
        RENDERING_DEFAULTS,
        "callout_arrow_width_px",
        3,
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )
    annotated = base.image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    draw.rounded_rectangle(
        tuple(float(value) for value in callout_box),
        radius=int(corner_radius),
        fill=fill_rgb,
        outline=outline_rgb,
        width=max(2, int(arrow_width)),
    )
    font = load_font(
        int(params.get("callout_font_size_px", group_default(RENDERING_DEFAULTS, "callout_font_size_px", 20))),
        bold=True,
        font_family=base.annotation_font_family,
    )
    callout_text, callout_source_trace = _sample_short_markup_label(
        params,
        instance_seed=int(instance_seed),
        namespace="callout_label",
        manifest_key="callout_label_manifest_path",
        max_chars_key="callout_label_max_chars",
        sample_attempts_key="callout_label_sample_attempts",
        fallback_text="Callout",
        fallback_max_chars=14,
        draw=draw,
        font=font,
        max_width_px=max(48, int(box_width - 18)),
    )
    draw_text_centered(
        draw,
        text=str(callout_text),
        center=callout_center,
        font=font,
        fill=text_rgb,
        stroke_fill=text_stroke_rgb,
        stroke_width=0,
    )
    line_start = (
        float(callout_box[0] if float(anchor_center[0]) < float(callout_center[0]) else callout_box[2]),
        float(callout_center[1]),
    )
    line_end = (float(anchor_center[0]), float(anchor_center[1]))
    draw.line((line_start, line_end), fill=outline_rgb, width=max(2, int(arrow_width)))
    arrow_radius = float(max(5, int(arrow_width) + 3))
    draw.ellipse(
        (
            float(anchor_center[0] - arrow_radius),
            float(anchor_center[1] - arrow_radius),
            float(anchor_center[0] + arrow_radius),
            float(anchor_center[1] + arrow_radius),
        ),
        fill=outline_rgb,
        outline=(255, 255, 255),
        width=2,
    )
    arrow_bbox = bbox_union(
        [
            [float(line_start[0]), float(line_start[1]), float(line_end[0]), float(line_end[1])],
            [
                float(anchor_center[0] - arrow_radius),
                float(anchor_center[1] - arrow_radius),
                float(anchor_center[0] + arrow_radius),
                float(anchor_center[1] + arrow_radius),
            ],
        ],
        padding=3.0,
    )
    label_bbox = _text_bbox(draw, str(callout_text), callout_center, font)
    entities = (
        {
            "entity_id": "annotation_callout",
            "entity_type": "annotation_callout",
            "attrs": {
                "bbox_px": list(callout_box),
                "label": str(callout_text),
                "label_bbox_px": list(label_bbox),
                "arrow_bbox_px": list(arrow_bbox),
                "anchor_label": str(anchor_label),
                "endpoint_label": str(endpoint_label),
                "label_source": dict(callout_source_trace),
                "callout_fill_rgb": list(fill_rgb),
                "callout_outline_rgb": list(outline_rgb),
                "callout_corner_radius_px": int(corner_radius),
                "callout_arrow_width_px": int(arrow_width),
            },
        },
    )
    return MarkupRender(
        image=annotated,
        entities=tuple(dict(item) for item in entities),
        annotation_bboxes={
            "annotation_callout": list(callout_box),
            "annotation_callout_label": list(label_bbox),
            "annotation_callout_arrow": list(arrow_bbox),
        },
    )


def finish_rendered_image(
    markup: MarkupRender,
    *,
    base: RenderedBaseSeries,
    params: Mapping[str, Any],
    instance_seed: int,
) -> FinalRender:
    context_elements = draw_context_layer(
        markup.image,
        context_layout=base.context_layout,
        information_style_meta=base.information_style_meta,
        instance_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        markup.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return FinalRender(
        image=image,
        context_elements=tuple(context_elements),
        post_noise_meta=dict(post_noise_meta),
    )


def context_entities(final: FinalRender) -> list[dict[str, Any]]:
    element_traces = [element.to_trace() for element in final.context_elements]
    return [
        {
            "entity_id": str(element["context_id"]),
            "entity_type": "non_answer_context_text",
            "attrs": dict(element),
        }
        for element in element_traces
    ]


def label_maps(rendered_scene: RenderedChartScene) -> dict[str, dict[str, Any]]:
    return {
        "label_centers_px": {
            str(mark["label"]): list(mark["label_center_px"]) for mark in rendered_scene.mark_traces
        },
        "values_by_label": {
            str(mark["label"]): int(mark["value"]) for mark in rendered_scene.mark_traces
        },
        "mark_bbox_by_label": {
            str(mark["label"]): list(mark["mark_bbox_px"]) for mark in rendered_scene.mark_traces
        },
        "mark_center_by_label": {
            str(mark["label"]): list(mark["mark_center_px"]) for mark in rendered_scene.mark_traces
        },
    }


def render_trace_sections(
    *,
    base: RenderedBaseSeries,
    markup: MarkupRender,
    final: FinalRender,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert renderer state into shared render_spec and render_map sections.

    The result records canvas, style, context, mark, and annotation projection
    metadata. Public task files add objective-specific answer semantics.
    """
    render_params = base.render_params
    rendered_scene = base.rendered_scene
    context_mode = str(base.context_layout.get("mode", "clean"))
    context_layout_mode = (
        f"chart_context:{context_mode}"
        if context_mode in {"clean", "minimal", "paragraph_box"}
        else f"{base.context_layout.get('layout_mode', context_mode)}:{base.context_layout.get('placement', 'none')}"
    )
    render_spec = {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "information_scene_style": dict(base.information_style_meta),
        "background_style": dict(base.background_meta),
        "post_image_noise": dict(final.post_noise_meta),
        "layout_jitter": dict(render_params.layout_jitter_meta or {}),
        "context_text_layer": context_text_layer_metadata(
            final.context_elements,
            enabled=bool(base.context_layout.get("enabled", False)),
            layout_mode=str(context_layout_mode),
            layout_spec={
                str(key): value
                for key, value in dict(base.context_layout).items()
                if str(key) != "context_params"
            },
        ),
        "text_style": {
            "label_font_size_px": int(render_params.label_font_size_px),
            "tick_font_size_px": int(render_params.tick_font_size_px),
            "label_stroke_width_px": int(render_params.label_stroke_width_px),
            "font_asset_version": str(font_asset_version()),
            "chart_font_family": str(base.chart_font_family),
            "annotation_font_family": str(base.annotation_font_family),
            "chart_font_exclude_tags": [],
            "annotation_font_exclude_tags": [],
            "context_font_exclude_tags": ["mono", "display", "script", "handwriting"],
        },
        "axis_style": {
            "axis_line_width_px": int(render_params.axis_line_width_px),
            "grid_line_width_px": int(render_params.grid_line_width_px),
            "tick_length_px": int(render_params.tick_length_px),
        },
        "mark_style": {
            "sampling_policy": str(base.mark_style["sampling_policy"]),
            "mark_fill_rgb": list(base.mark_style["mark_fill_rgb"]),
            "mark_outline_rgb": list(base.mark_style["mark_outline_rgb"]),
            **{
                str(key): value
                for key, value in base.mark_style.items()
                if key not in {"sampling_policy", "mark_fill_rgb", "mark_outline_rgb"}
            },
        },
        "annotation_bboxes": dict(markup.annotation_bboxes),
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "y_axis_max": int(rendered_scene.y_axis_max),
        "y_ticks": [int(value) for value in rendered_scene.y_ticks],
        **value_axis_render_metadata(rendered_scene),
    }
    maps = label_maps(rendered_scene)
    render_map = {
        "image_id": "img0",
        "plot_bbox_px": list(rendered_scene.plot_bbox_px),
        "label_centers_px": dict(maps["label_centers_px"]),
        "mark_bbox_by_label": dict(maps["mark_bbox_by_label"]),
        "mark_center_by_label": dict(maps["mark_center_by_label"]),
        "annotation_bboxes": dict(markup.annotation_bboxes),
    }
    return render_spec, render_map
