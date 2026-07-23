"""Rendering primitives for Vernier-caliper diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.physics.shared.diagram_style import (
    prepare_physics_diagram_style_and_background,
)
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many as _bbox_union_many
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)
from trace_tasks.tasks.shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    contrast_ratio,
    draw_text_traced,
)
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, normalize_annotation_point_map
from .state import (
    DEFAULTS,
    OPTION_LETTERS,
    SCENE_ID,
    SCENE_NAMESPACE,
    VERNIER_DIVISIONS,
    VERNIER_RESOLUTION_MM,
    CaliperScenario,
    RenderedScene,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _precomputed_text_contrast_metadata(
    *,
    fill_rgb: Tuple[int, int, int],
    surface_rgb: Tuple[int, int, int],
    surface_sample_method: str,
) -> Dict[str, Any]:
    """Return validation-ready contrast metadata for text on a known fill."""

    ratio = float(contrast_ratio(fill_rgb, surface_rgb))
    lab_distance = float(
        color_distance(fill_rgb, surface_rgb, distance_space="lab")
    )
    return {
        "surface_rgbs": [[int(value) for value in surface_rgb]],
        "surface_sample_method": str(surface_sample_method),
        "min_contrast_ratio": round(float(ratio), 3),
        "min_lab_distance": round(float(lab_distance), 3),
        "min_contrast_required": round(float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO), 3),
        "min_lab_distance_required": round(float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE), 3),
        "passes": bool(
            ratio >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO)
            and lab_distance >= float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE)
        ),
    }


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    fill: Tuple[int, int, int],
    *,
    anchor: str = "mm",
    backing_fill: Tuple[int, int, int] | None = None,
) -> List[float]:
    """Draw one required readout label and return its pixel box."""

    stroke_width = 1
    backing_bbox = None
    if backing_fill is not None:
        text_bbox = draw.textbbox(
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            stroke_width=stroke_width,
            anchor=anchor,
        )
        pad_x = 7.0
        pad_y = 4.0
        backing_bbox = (
            float(text_bbox[0]) - pad_x,
            float(text_bbox[1]) - pad_y,
            float(text_bbox[2]) + pad_x,
            float(text_bbox[3]) + pad_y,
        )
        draw.rounded_rectangle(
            backing_bbox,
            radius=5,
            fill=tuple(int(value) for value in backing_fill),
            outline=tuple(int(value) for value in backing_fill),
            width=1,
        )
    extra_metadata: Dict[str, Any] = {}
    if backing_bbox is not None:
        extra_metadata["glyph_bbox_px"] = [
            round(float(value), 3)
            for value in draw.textbbox(
                (float(xy[0]), float(xy[1])),
                str(text),
                font=font,
                stroke_width=stroke_width,
                anchor=anchor,
            )
        ]
        extra_metadata.update(
            _precomputed_text_contrast_metadata(
                fill_rgb=tuple(int(value) for value in fill),
                surface_rgb=tuple(int(value) for value in backing_fill),
                surface_sample_method="declared_label_backing_fill",
            )
        )
    record = draw_text_traced(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=resolve_text_stroke_fill(fill),
        role="readout",
        required=True,
        anchor=anchor,
        extra_metadata=extra_metadata,
    )
    return bbox(record["bbox_px"])


def draw_main_scale(
    draw: ImageDraw.ImageDraw,
    *,
    left_x: float,
    scale_y: float,
    mm_px: float,
    max_mm: int,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
) -> Tuple[Dict[int, List[float]], Dict[int, float], List[float]]:
    """Draw the fixed main scale and return tick boxes and x positions."""

    stroke = tuple(int(value) for value in style.stroke_rgb)
    guide = tuple(int(value) for value in style.guide_rgb)
    label_rgb = tuple(int(value) for value in style.label_rgb)
    backing_fill = tuple(int(value) for value in style.label_fill_rgb)
    label_font = load_font(
        int(render_defaults.get("label_font_size_px", DEFAULTS.label_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    small_font = load_font(
        int(render_defaults.get("small_font_size_px", DEFAULTS.small_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    body_top = float(scale_y - 46.0)
    body_bottom = float(scale_y + 42.0)
    right_x = float(left_x + max_mm * mm_px)
    label_y = float(body_top - 18.0)
    draw.rounded_rectangle(
        (left_x - 22.0, body_top, right_x + 24.0, body_bottom),
        radius=13,
        fill=tuple(int(value) for value in style.panel_alt_fill_rgb),
        outline=stroke,
        width=3,
    )
    draw.line((left_x, scale_y, right_x, scale_y), fill=stroke, width=4)

    tick_bboxes: List[List[float]] = []
    tick_bbox_map: Dict[int, List[float]] = {}
    label_bboxes: List[List[float]] = []
    tick_xs: Dict[int, float] = {}
    for tick in range(0, int(max_mm) + 1):
        x = float(left_x + tick * mm_px)
        tick_xs[int(tick)] = x
        if tick % 10 == 0:
            tick_len = 37.0
            tick_width = 3
        elif tick % 5 == 0:
            tick_len = 28.0
            tick_width = 2
        else:
            tick_len = 18.0
            tick_width = 1
        draw.line(
            (x, scale_y, x, scale_y - tick_len),
            fill=stroke if tick % 5 == 0 else guide,
            width=tick_width,
        )
        tick_bbox = bbox((x - 2.0, scale_y - tick_len - 2.0, x + 2.0, scale_y + 2.0))
        tick_bboxes.append(tick_bbox)
        tick_bbox_map[int(tick)] = tick_bbox
        if tick % 10 == 0:
            label_bboxes.append(
                _draw_label(
                    draw,
                    (x, label_y),
                    str(tick),
                    label_font,
                    label_rgb,
                    backing_fill=backing_fill,
                )
            )
    unit_bbox = _draw_label(
        draw,
        (right_x + 38.0, label_y),
        "mm",
        small_font,
        label_rgb,
        backing_fill=backing_fill,
    )
    scale_bbox = bbox(_bbox_union_many(*(tick_bboxes + label_bboxes + [unit_bbox]), padding=8.0))
    return tick_bbox_map, tick_xs, scale_bbox


def draw_option_panel(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: CaliperScenario,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    """Draw six visible numeric answer choices and return their boxes."""

    canvas_width = int(render_defaults.get("canvas_width", DEFAULTS.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", DEFAULTS.canvas_height))
    panel_left = float(render_defaults.get("panel_left_px", DEFAULTS.panel_left_px))
    panel_right = float(
        canvas_width
        - int(render_defaults.get("panel_right_margin_px", DEFAULTS.panel_right_margin_px))
    )
    option_top = float(
        render_defaults.get("option_panel_top_px", DEFAULTS.option_panel_top_px)
    )
    cell_height = float(
        render_defaults.get("option_cell_height_px", DEFAULTS.option_cell_height_px)
    )
    cell_gap_x = 16.0
    cell_gap_y = 10.0
    outer_pad = 22.0
    columns = 3
    cell_width = float(
        (panel_right - panel_left - 2.0 * outer_pad - (columns - 1) * cell_gap_x)
        / columns
    )
    stroke = tuple(int(value) for value in style.stroke_rgb)
    label_rgb = tuple(int(value) for value in style.label_rgb)
    fill = tuple(int(value) for value in style.panel_alt_fill_rgb)
    letter_fill = tuple(int(value) for value in style.panel_fill_rgb)
    value_font = load_font(
        int(render_defaults.get("small_font_size_px", DEFAULTS.small_font_size_px)) + 3,
        bold=False,
        font_family=font_family,
    )
    letter_font = load_font(
        int(render_defaults.get("small_font_size_px", DEFAULTS.small_font_size_px)) + 3,
        bold=True,
        font_family=font_family,
    )

    option_bboxes: Dict[str, List[float]] = {}
    option_letter_bboxes: Dict[str, List[float]] = {}
    option_text_bboxes: Dict[str, List[float]] = {}
    for index, letter in enumerate(OPTION_LETTERS):
        row = int(index // columns)
        col = int(index % columns)
        x0 = float(panel_left + outer_pad + col * (cell_width + cell_gap_x))
        y0 = float(option_top + row * (cell_height + cell_gap_y))
        x1 = float(x0 + cell_width)
        y1 = float(y0 + cell_height)
        cell_bbox = bbox(
            (
                max(0.0, x0),
                max(0.0, y0),
                min(float(canvas_width), x1),
                min(float(canvas_height), y1),
            )
        )
        draw.rounded_rectangle(
            tuple(cell_bbox),
            radius=8,
            fill=fill,
            outline=stroke,
            width=2,
        )
        letter_center = (float(cell_bbox[0] + 27.0), float((cell_bbox[1] + cell_bbox[3]) / 2.0))
        draw.ellipse(
            (
                letter_center[0] - 15.0,
                letter_center[1] - 15.0,
                letter_center[0] + 15.0,
                letter_center[1] + 15.0,
            ),
            fill=letter_fill,
            outline=stroke,
            width=2,
        )
        letter_bbox = draw_centered_text(
            draw,
            text=str(letter),
            center=letter_center,
            font=letter_font,
            fill=label_rgb,
            stroke_fill=resolve_text_stroke_fill(label_rgb),
            stroke_width=1,
        )
        value_text = f"{float(scenario.option_values_mm[str(letter)]):.1f} mm"
        value_record = draw_text_traced(
            draw,
            (float(cell_bbox[0] + 54.0), float((cell_bbox[1] + cell_bbox[3]) / 2.0)),
            value_text,
            font=value_font,
            fill=label_rgb,
            stroke_width=1,
            stroke_fill=resolve_text_stroke_fill(label_rgb),
            role="option_value",
            required=True,
            anchor="lm",
            extra_metadata=_precomputed_text_contrast_metadata(
                fill_rgb=label_rgb,
                surface_rgb=fill,
                surface_sample_method="declared_option_cell_fill",
            ),
        )
        option_bboxes[str(letter)] = list(cell_bbox)
        option_letter_bboxes[str(letter)] = bbox(letter_bbox)
        option_text_bboxes[str(letter)] = bbox(value_record["bbox_px"])

    panel_bbox = bbox(
        _bbox_union_many(*option_bboxes.values(), padding=8.0)
    )
    return {
        "option_values_mm": {
            str(letter): float(scenario.option_values_mm[str(letter)])
            for letter in OPTION_LETTERS
        },
        "option_bboxes_px": {str(key): list(value) for key, value in option_bboxes.items()},
        "option_letter_bboxes_px": {
            str(key): list(value) for key, value in option_letter_bboxes.items()
        },
        "option_text_bboxes_px": {
            str(key): list(value) for key, value in option_text_bboxes.items()
        },
        "option_panel_bbox_px": list(panel_bbox),
        "correct_option_letter": str(scenario.correct_option_letter),
        "correct_option_bbox_px": list(option_bboxes[str(scenario.correct_option_letter)]),
    }


def draw_caliper(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: CaliperScenario,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
    layout_shift: Tuple[float, float],
    accent_rgb: Tuple[int, int, int],
) -> Tuple[Dict[str, List[float]], Dict[str, Any], List[Dict[str, Any]]]:
    """Draw the caliper apparatus and return witness points plus render metadata."""

    dx, dy = float(layout_shift[0]), float(layout_shift[1])
    left_x = float(render_defaults.get("main_scale_left_px", DEFAULTS.main_scale_left_px)) + dx
    scale_y = float(render_defaults.get("main_scale_y_px", DEFAULTS.main_scale_y_px)) + dy
    mm_px = float(render_defaults.get("mm_px", DEFAULTS.mm_px))
    max_mm = int(render_defaults.get("main_scale_max_mm", DEFAULTS.main_scale_max_mm))
    jaw_top = float(render_defaults.get("jaw_top_px", DEFAULTS.jaw_top_px)) + dy
    jaw_bottom = float(render_defaults.get("jaw_bottom_px", DEFAULTS.jaw_bottom_px)) + dy
    stroke = tuple(int(value) for value in style.stroke_rgb)
    guide = tuple(int(value) for value in style.guide_rgb)
    label_rgb = tuple(int(value) for value in style.label_rgb)
    backing_fill = tuple(int(value) for value in style.label_fill_rgb)
    title_font = load_font(
        int(render_defaults.get("title_font_size_px", DEFAULTS.title_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    label_font = load_font(
        int(render_defaults.get("small_font_size_px", DEFAULTS.small_font_size_px)),
        bold=True,
        font_family=font_family,
    )

    tick_bbox_map, _tick_xs, full_main_scale_region = draw_main_scale(
        draw,
        left_x=left_x,
        scale_y=scale_y,
        mm_px=mm_px,
        max_mm=max_mm,
        font_family=font_family,
        style=style,
        render_defaults=render_defaults,
    )
    vernier_zero_x = float(
        left_x
        + (
            float(scenario.main_mm)
            + float(scenario.aligned_vernier_tick) * VERNIER_RESOLUTION_MM
        )
        * mm_px
    )
    moving_body_top = float(scale_y + 72.0)
    moving_body_bottom = float(scale_y + 174.0)
    vernier_span_px = float((VERNIER_DIVISIONS - 1) * mm_px)
    vernier_right = float(vernier_zero_x + vernier_span_px)

    title_bbox = draw_centered_text(
        draw,
        text="Vernier caliper",
        center=(float(left_x + max_mm * mm_px * 0.5), float(jaw_top - 40.0)),
        font=title_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )

    rail_y = float(scale_y + 8.0)
    draw.rounded_rectangle(
        (left_x - 68.0, rail_y - 18.0, left_x + max_mm * mm_px + 66.0, rail_y + 16.0),
        radius=12,
        fill=tuple(int(value) for value in style.muted_fill_rgb),
        outline=stroke,
        width=3,
    )

    fixed_jaw_x = left_x
    jaw_fill = tuple(int(value) for value in style.panel_alt_fill_rgb)
    draw.polygon(
        [
            (fixed_jaw_x - 45.0, jaw_top),
            (fixed_jaw_x + 14.0, jaw_top),
            (fixed_jaw_x + 14.0, jaw_bottom),
            (fixed_jaw_x - 15.0, jaw_bottom),
            (fixed_jaw_x - 15.0, scale_y + 52.0),
            (fixed_jaw_x - 45.0, scale_y + 26.0),
        ],
        fill=jaw_fill,
        outline=stroke,
    )
    draw.line(
        (fixed_jaw_x + 14.0, jaw_top, fixed_jaw_x + 14.0, jaw_bottom),
        fill=stroke,
        width=4,
    )

    draw.rounded_rectangle(
        (vernier_zero_x - 18.0, moving_body_top, vernier_right + 26.0, moving_body_bottom),
        radius=12,
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=stroke,
        width=3,
    )
    draw.polygon(
        [
            (vernier_zero_x - 10.0, jaw_top + 10.0),
            (vernier_zero_x + 42.0, jaw_top + 10.0),
            (vernier_zero_x + 42.0, scale_y + 78.0),
            (vernier_zero_x + 15.0, scale_y + 78.0),
            (vernier_zero_x + 15.0, jaw_bottom),
            (vernier_zero_x - 10.0, jaw_bottom),
        ],
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=stroke,
    )
    draw.line((vernier_zero_x, jaw_top + 8.0, vernier_zero_x, jaw_bottom), fill=stroke, width=4)

    measured_object_bbox = bbox(
        (fixed_jaw_x + 16.0, jaw_bottom - 104.0, vernier_zero_x - 4.0, jaw_bottom - 34.0)
    )
    object_fill = tuple(int(value) for value in accent_rgb)
    if measured_object_bbox[2] - measured_object_bbox[0] > 24.0:
        draw.rounded_rectangle(tuple(measured_object_bbox), radius=10, fill=object_fill, outline=stroke, width=3)

    vernier_y = float(moving_body_top + 24.0)
    vernier_label_bboxes: List[List[float]] = []
    vernier_tick_bboxes: List[List[float]] = []
    vernier_tick_segments: Dict[int, List[List[float]]] = {}
    vernier_tick_points: Dict[int, List[float]] = {}
    for tick in range(0, VERNIER_DIVISIONS):
        x = float(vernier_zero_x + tick * (0.9 * mm_px))
        tick_len = 42.0 if tick in (0, VERNIER_DIVISIONS - 1) else 30.0
        segment = [
            [round(float(x), 3), round(float(vernier_y), 3)],
            [round(float(x), 3), round(float(vernier_y + tick_len), 3)],
        ]
        vernier_tick_segments[int(tick)] = [list(point) for point in segment]
        vernier_tick_points[int(tick)] = [
            round(float(x), 3),
            round(float(vernier_y + tick_len / 2.0), 3),
        ]
        draw.line(
            (x, vernier_y, x, vernier_y + tick_len),
            fill=stroke if tick in (0, VERNIER_DIVISIONS - 1) else guide,
            width=2,
        )
        vernier_tick_bboxes.append(
            bbox((x - 2.0, vernier_y - 2.0, x + 2.0, vernier_y + tick_len + 2.0))
        )
        if tick in (0, 5, 9):
            label_x = x + 7.0 if tick == 0 else x
            vernier_label_bboxes.append(
                _draw_label(
                    draw,
                    (label_x, vernier_y + tick_len + 20.0),
                    str(tick),
                    label_font,
                    label_rgb,
                    backing_fill=backing_fill,
                )
            )
    vernier_zero_bbox = bbox((vernier_zero_x - 8.0, vernier_y - 8.0, vernier_zero_x + 8.0, vernier_y + 52.0))
    aligned_tick_x = float(vernier_zero_x + int(scenario.aligned_vernier_tick) * (0.9 * mm_px))
    aligned_tick_bbox = bbox((aligned_tick_x - 8.0, vernier_y - 8.0, aligned_tick_x + 8.0, vernier_y + 56.0))
    vernier_scale_region = bbox(
        _bbox_union_many(*(vernier_tick_bboxes + vernier_label_bboxes), padding=8.0)
    )

    nearest_main_tick = int(scenario.main_mm + scenario.aligned_vernier_tick)
    local_tick_keys = [
        tick
        for tick in range(
            max(0, int(scenario.main_mm) - 2),
            min(max_mm, int(scenario.main_mm + scenario.aligned_vernier_tick) + 2) + 1,
        )
        if tick in tick_bbox_map
    ]
    local_scale_boxes = [tick_bbox_map[tick] for tick in local_tick_keys]
    main_scale_region = (
        bbox(_bbox_union_many(*local_scale_boxes, padding=8.0))
        if local_scale_boxes
        else full_main_scale_region
    )

    _draw_label(
        draw,
        (float((vernier_zero_x + vernier_right) / 2.0), float(moving_body_bottom + 34.0)),
        "0.1 mm vernier",
        label_font,
        label_rgb,
        backing_fill=backing_fill,
    )

    annotation_map = normalize_annotation_point_map(
        {
            "vernier_zero_tick": vernier_tick_points[0],
            "aligned_vernier_tick": vernier_tick_points[int(scenario.aligned_vernier_tick)],
        }
    )
    option_map = draw_option_panel(
        draw,
        scenario=scenario,
        font_family=str(font_family),
        style=style,
        render_defaults=render_defaults,
    )
    context_bbox_map = {
        "main_scale_region": list(main_scale_region),
        "vernier_zero": list(vernier_zero_bbox),
        "vernier_scale_region": list(vernier_scale_region),
        "aligned_vernier_tick": list(aligned_tick_bbox),
    }
    scene_entities = [
        {"id": key, "point_px": list(value)}
        for key, value in annotation_map.items()
    ]
    render_map = {
        "main_mm": int(scenario.main_mm),
        "aligned_vernier_tick": int(scenario.aligned_vernier_tick),
        "vernier_resolution_mm": float(VERNIER_RESOLUTION_MM),
        "answer_mm": float(scenario.answer_mm),
        "main_scale_max_mm": int(max_mm),
        "mm_px": round(float(mm_px), 3),
        "vernier_zero_x_px": round(float(vernier_zero_x), 3),
        "aligned_vernier_tick_x_px": round(float(aligned_tick_x), 3),
        "nearest_aligned_main_tick": int(nearest_main_tick),
        "annotation_source": "selected_option_bbox_px",
        "readout_witness_point_map_px": {str(key): list(value) for key, value in annotation_map.items()},
        "vernier_tick_points_px": {str(key): list(value) for key, value in vernier_tick_points.items()},
        "vernier_tick_segments_px": {
            str(key): [list(point) for point in value]
            for key, value in vernier_tick_segments.items()
        },
        "context_bbox_map_px": {str(key): list(value) for key, value in context_bbox_map.items()},
        "full_main_scale_region_px": list(full_main_scale_region),
        "title_bbox_px": list(title_bbox),
        "measured_object_bbox_px": measured_object_bbox,
    }
    render_map.update(option_map)
    return annotation_map, render_map, scene_entities


def render_caliper(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: CaliperScenario,
    render_defaults: Mapping[str, Any],
) -> RenderedScene:
    """Render one caliper diagram and project keyed annotation points."""

    canvas_width = int(render_defaults.get("canvas_width", DEFAULTS.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", DEFAULTS.canvas_height))
    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            require_grid=True,
        )
    )
    draw = ImageDraw.Draw(background)
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.render")
    panel = (
        float(render_defaults.get("panel_left_px", DEFAULTS.panel_left_px)),
        float(render_defaults.get("panel_top_px", DEFAULTS.panel_top_px)),
        float(canvas_width - int(render_defaults.get("panel_right_margin_px", DEFAULTS.panel_right_margin_px))),
        float(canvas_height - int(render_defaults.get("panel_bottom_margin_px", DEFAULTS.panel_bottom_margin_px))),
    )
    draw.rounded_rectangle(
        panel,
        radius=20,
        fill=tuple(int(value) for value in diagram_style.panel_fill_rgb),
        outline=tuple(int(value) for value in diagram_style.panel_border_rgb),
        width=3,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font",
        params=params,
    )
    font_record = get_font_family_record(str(font_family))
    object_palette = (
        (215, 90, 74),
        (75, 146, 216),
        (63, 159, 112),
        (190, 129, 55),
        (143, 99, 190),
    )
    object_rgb = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.object_color").choice(object_palette)
    annotation_map, render_map, scene_entities = draw_caliper(
        draw,
        scenario=scenario,
        font_family=str(font_family),
        style=diagram_style,
        render_defaults=render_defaults,
        layout_shift=(float(rng.randint(-18, 18)), float(rng.randint(-8, 12))),
        accent_rgb=object_rgb,
    )
    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map.update(
        {
            "object_rgb": [int(value) for value in object_rgb],
            "font": {
                "font_family": str(font_family),
                "font_asset_version": font_asset_version(),
                "font_asset": font_record.to_trace(),
            },
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
        }
    )
    return RenderedScene(
        image=image,
        annotation_point_map={str(key): list(value) for key, value in annotation_map.items()},
        scene_entities=list(scene_entities),
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = [
    "draw_caliper",
    "draw_main_scale",
    "draw_option_panel",
    "render_caliper",
]
