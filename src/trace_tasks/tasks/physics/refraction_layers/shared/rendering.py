"""Identity-free renderer for the refraction-layers scene."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text, draw_dashed_line
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .ray_model import ray_geometry
from .state import MEDIUM_LABELS, OPTION_LABELS, RefractionScenario, RenderedRefractionScene


def _bbox(values: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in values]


def _clip_bbox(values: Sequence[float], *, width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in values]
    return _bbox(
        (
            max(0.0, min(float(width), x0)),
            max(0.0, min(float(height), y0)),
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
        )
    )


def _blend_rgb(a: Sequence[int], b: Sequence[int], alpha: float) -> Tuple[int, int, int]:
    return tuple(
        int(round((float(1.0 - alpha) * int(a[idx])) + (float(alpha) * int(b[idx]))))
        for idx in range(3)
    )


def _draw_media_regions(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: RefractionScenario,
    medium_box: Tuple[float, float, float, float],
    style: Any,
    font: Any,
    instance_seed: int,
) -> Dict[str, List[float]]:
    """Draw the three physical media while preserving label-region projections."""

    left, top, right, bottom = [float(value) for value in medium_box]
    fill_seeds = [
        (219, 238, 255),
        (225, 244, 225),
        (255, 237, 213),
        (239, 229, 255),
        (225, 246, 246),
        (255, 228, 233),
    ]
    rng = spawn_rng(int(instance_seed), "refraction_layers.medium_fills")
    rng.shuffle(fill_seeds)
    label_bboxes: Dict[str, List[float]] = {}
    outline = tuple(int(v) for v in style.panel_border_rgb)
    text_rgb = tuple(int(v) for v in style.label_rgb)
    if scenario.orientation == "horizontal":
        y_values = [top, top + ((bottom - top) / 3.0), top + (2.0 * (bottom - top) / 3.0), bottom]
        for idx, label in enumerate(MEDIUM_LABELS):
            rect = (left, y_values[idx], right, y_values[idx + 1])
            fill = _blend_rgb(fill_seeds[idx], style.panel_fill_rgb, 0.18)
            draw.rectangle(rect, fill=fill, outline=outline, width=1)
            label_bboxes[str(label)] = draw_centered_text(
                draw,
                text=str(label),
                center=(left + 48.0, (rect[1] + rect[3]) * 0.5),
                font=font,
                fill=text_rgb,
                stroke_fill=resolve_text_stroke_fill(text_rgb),
                stroke_width=2,
            )
        for y in y_values[1:-1]:
            draw.line((left, y, right, y), fill=tuple(int(v) for v in style.stroke_rgb), width=3)
    else:
        x_values = [left, left + ((right - left) / 3.0), left + (2.0 * (right - left) / 3.0), right]
        for idx, label in enumerate(MEDIUM_LABELS):
            rect = (x_values[idx], top, x_values[idx + 1], bottom)
            fill = _blend_rgb(fill_seeds[idx], style.panel_fill_rgb, 0.18)
            draw.rectangle(rect, fill=fill, outline=outline, width=1)
            label_bboxes[str(label)] = draw_centered_text(
                draw,
                text=str(label),
                center=((rect[0] + rect[2]) * 0.5, top + 38.0),
                font=font,
                fill=text_rgb,
                stroke_fill=resolve_text_stroke_fill(text_rgb),
                stroke_width=2,
            )
        for x in x_values[1:-1]:
            draw.line((x, top, x, bottom), fill=tuple(int(v) for v in style.stroke_rgb), width=3)
    return label_bboxes


def _draw_ray_and_normals(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: RefractionScenario,
    geometry: Any,
    style: Any,
) -> None:
    """Draw normals first, then the refracted ray path on top."""

    ray_rgb = tuple(int(v) for v in style.secondary_accent_rgb)
    guide_rgb = tuple(int(v) for v in style.guide_rgb)
    stroke_rgb = tuple(int(v) for v in style.stroke_rgb)
    normal_half = 74.0
    for point in geometry.bend_points:
        x, y = float(point[0]), float(point[1])
        if scenario.orientation == "horizontal":
            start = (x, y - normal_half)
            end = (x, y + normal_half)
        else:
            start = (x - normal_half, y)
            end = (x + normal_half, y)
        draw_dashed_line(draw, start=start, end=end, fill=guide_rgb, width=3, dash_px=9, gap_px=7)
    draw.line(list(geometry.points), fill=stroke_rgb, width=11)
    draw.line(list(geometry.points), fill=ray_rgb, width=7)
    for point in geometry.bend_points:
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=ray_rgb, outline=stroke_rgb, width=2)
    pre_end = geometry.points[-2]
    final_end = geometry.points[-1]
    arrow_start = (
        float(pre_end[0] + 0.52 * (final_end[0] - pre_end[0])),
        float(pre_end[1] + 0.52 * (final_end[1] - pre_end[1])),
    )
    draw_arrow(
        draw,
        start=arrow_start,
        end=final_end,
        fill=ray_rgb,
        width=7,
        head_length_px=18,
        head_width_px=17,
    )


def _draw_option_panel(
    draw: ImageDraw.ImageDraw,
    *,
    option_box: Tuple[float, float, float, float],
    scenario: RefractionScenario,
    style: Any,
    title_font: Any,
    option_font: Any,
) -> Dict[str, List[float]]:
    """Draw fixed option cards without exposing which option is correct."""

    left, top, right, _bottom = [float(value) for value in option_box]
    draw.rounded_rectangle(
        option_box,
        radius=18,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    text_rgb = tuple(int(v) for v in style.label_rgb)
    draw_centered_text(
        draw,
        text="Fastest to slowest",
        center=((left + right) * 0.5, top + 34.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    option_bboxes: Dict[str, List[float]] = {}
    box_left = left + 22.0
    box_right = right - 22.0
    box_h = 66.0
    gap = 14.0
    y = top + 76.0
    for label in OPTION_LABELS:
        box = (box_left, y, box_right, y + box_h)
        draw.rounded_rectangle(
            box,
            radius=12,
            fill=tuple(int(v) for v in style.option_fill_rgb),
            outline=tuple(int(v) for v in style.panel_border_rgb),
            width=2,
        )
        draw_centered_text(
            draw,
            text=str(label),
            center=(box_left + 25.0, y + (box_h * 0.5)),
            font=option_font,
            fill=text_rgb,
            stroke_fill=resolve_text_stroke_fill(text_rgb),
            stroke_width=1,
        )
        draw_centered_text(
            draw,
            text=str(scenario.option_map[str(label)]),
            center=(box_left + 132.0, y + (box_h * 0.5)),
            font=option_font,
            fill=text_rgb,
            stroke_fill=resolve_text_stroke_fill(text_rgb),
            stroke_width=1,
        )
        option_bboxes[str(label)] = _bbox(box)
        y += box_h + gap
    return option_bboxes


def render_refraction_layers_scene(
    *,
    image: Image.Image,
    scenario: RefractionScenario,
    font_family: str,
    style: Any,
    instance_seed: int,
    render_defaults: Mapping[str, Any],
) -> RenderedRefractionScene:
    """Render one refraction diagram and project the bend witness boxes."""

    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    title_font = load_font(25, bold=True, font_family=font_family)
    label_font = load_font(24, bold=True, font_family=font_family)
    option_font = load_font(19, bold=True, font_family=font_family)

    diagram_box = (
        float(render_defaults.get("diagram_left_px", 54)),
        float(render_defaults.get("diagram_top_px", 54)),
        float(render_defaults.get("diagram_right_px", 744)),
        float(render_defaults.get("diagram_bottom_px", 646)),
    )
    option_box = (
        float(render_defaults.get("option_left_px", 774)),
        float(render_defaults.get("option_top_px", 64)),
        float(render_defaults.get("option_right_px", canvas_width - 50)),
        float(render_defaults.get("option_bottom_px", canvas_height - 64)),
    )
    draw.rounded_rectangle(
        diagram_box,
        radius=20,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    text_rgb = tuple(int(v) for v in style.label_rgb)
    draw_centered_text(
        draw,
        text="Refraction through labeled media",
        center=((diagram_box[0] + diagram_box[2]) * 0.5, diagram_box[1] + 30.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    medium_box = (
        diagram_box[0] + 36.0,
        diagram_box[1] + 68.0,
        diagram_box[2] - 36.0,
        diagram_box[3] - 32.0,
    )
    label_bboxes = _draw_media_regions(
        draw,
        scenario=scenario,
        medium_box=medium_box,
        style=style,
        font=label_font,
        instance_seed=int(instance_seed),
    )
    geometry = ray_geometry(scenario=scenario, medium_box=medium_box)
    _draw_ray_and_normals(draw, scenario=scenario, geometry=geometry, style=style)
    option_bboxes = _draw_option_panel(
        draw,
        option_box=option_box,
        scenario=scenario,
        style=style,
        title_font=title_font,
        option_font=option_font,
    )
    annotation_map: Dict[str, List[float]] = {}
    for idx, point in enumerate(geometry.bend_points, start=1):
        x, y = float(point[0]), float(point[1])
        if scenario.orientation == "horizontal":
            box = (x - 72.0, y - 88.0, x + 72.0, y + 88.0)
        else:
            box = (x - 88.0, y - 72.0, x + 88.0, y + 72.0)
        annotation_map[f"interface_{idx}_bend"] = _clip_bbox(box, width=canvas_width, height=canvas_height)
    render_map = {
        "diagram_box": _bbox(diagram_box),
        "medium_box": _bbox(medium_box),
        "option_bboxes": option_bboxes,
        "medium_label_bboxes": label_bboxes,
        "ray_points": [[round(float(x), 3), round(float(y), 3)] for x, y in geometry.points],
        "bend_points": [[round(float(x), 3), round(float(y), 3)] for x, y in geometry.bend_points],
        "segment_mediums": list(geometry.segment_mediums),
        "segment_angles_deg": [round(float(value), 3) for value in geometry.segment_angles_deg],
        "medium_speeds": {str(k): round(float(v), 3) for k, v in scenario.medium_speeds.items()},
        "angle_by_medium_deg": {str(k): round(float(v), 3) for k, v in scenario.angle_by_medium_deg.items()},
        "speed_order": list(scenario.speed_order),
        "option_map": dict(scenario.option_map),
        "correct_label": str(scenario.correct_label),
    }
    return RenderedRefractionScene(
        image=image,
        annotation_bbox_map={str(k): list(v) for k, v in annotation_map.items()},
        render_map=render_map,
    )
