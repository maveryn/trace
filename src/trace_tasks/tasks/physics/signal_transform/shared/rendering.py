"""Rendering helpers for signal-transform scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .sampling import bbox, spectrum_payload
from .state import OPTION_LABELS, RenderedSignalTransformScene, SignalScenario, SignalTransformAxes, SpectrumSpec


def draw_panel_frame(
    draw: ImageDraw.ImageDraw,
    *,
    box: Sequence[float],
    label: str,
    label_font: Any,
    axis_font: Any,
    style: Any,
    scene_variant: str,
) -> None:
    """Draw one waveform or spectrum panel frame."""

    left, top, right, bottom = [float(value) for value in box]
    panel_fill = tuple(int(value) for value in style.panel_alt_fill_rgb)
    panel_border = tuple(int(value) for value in style.panel_border_rgb)
    text_rgb = tuple(int(value) for value in style.label_rgb)
    grid_rgb = tuple(int(value) for value in style.grid_minor_rgb)
    axis_rgb = tuple(int(value) for value in style.axis_rgb)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=12,
        fill=panel_fill,
        outline=panel_border,
        width=2,
    )
    if str(scene_variant) in {"grid_match", "lab_sheet"}:
        for idx in range(1, 6):
            x = left + (idx * (right - left) / 6.0)
            draw.line((x, top + 12.0, x, bottom - 12.0), fill=grid_rgb, width=1)
        for idx in range(1, 4):
            y = top + (idx * (bottom - top) / 4.0)
            draw.line((left + 12.0, y, right - 12.0, y), fill=grid_rgb, width=1)
    draw.line((left + 42.0, bottom - 34.0, right - 24.0, bottom - 34.0), fill=axis_rgb, width=2)
    draw.line((left + 42.0, top + 24.0, left + 42.0, bottom - 34.0), fill=axis_rgb, width=2)
    label_center_x = left + (52.0 if str(label).lower().startswith("input") else 23.0)
    draw_centered_text(
        draw,
        text=str(label),
        center=(label_center_x, top + 24.0),
        font=label_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    x_axis = "t" if str(label).lower().startswith("input") else "f"
    draw_centered_text(
        draw,
        text=x_axis,
        center=(right - 20.0, bottom - 22.0),
        font=axis_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )


def draw_time_waveform(
    draw: ImageDraw.ImageDraw,
    *,
    box: Sequence[float],
    scenario: SignalScenario,
    style: Any,
    line_width: int,
) -> List[float]:
    """Draw the input waveform from symbolic family parameters.

    This function is the only place that turns the sampled signal family into
    visible time-domain geometry. The returned bbox must cover the final drawn
    waveform so annotation and render metadata stay aligned with the image.
    """

    left, top, right, bottom = [float(value) for value in box]
    wave_left = left + 58.0
    wave_right = right - 34.0
    mid_y = top + ((bottom - top) * 0.52)
    amp = (bottom - top) * 0.28
    points: List[Tuple[float, float]] = []
    sample_count = 420
    family = str(scenario.waveform_family)
    for idx in range(sample_count + 1):
        t = float(idx) / float(sample_count)
        if family == "square_wave":
            y_unit = 1.0 if math.sin(2.0 * math.pi * float(scenario.time_cycles) * t) >= 0.0 else -1.0
        elif family == "triangle_wave":
            phase = (float(scenario.time_cycles) * t) % 1.0
            y_unit = 1.0 - (4.0 * abs(phase - 0.5))
        elif family == "sawtooth_wave":
            phase = (float(scenario.time_cycles) * t) % 1.0
            y_unit = (2.0 * phase) - 1.0
        else:
            raise ValueError(f"unsupported signal-transform waveform family: {family}")
        points.append((wave_left + (t * (wave_right - wave_left)), mid_y - (amp * y_unit)))
    shadow = tuple(int(value) for value in style.stroke_rgb)
    wave_rgb = tuple(int(value) for value in style.secondary_accent_rgb)
    draw.line(points, fill=shadow, width=max(1, int(line_width) + 2), joint="curve")
    draw.line(points, fill=wave_rgb, width=max(1, int(line_width)), joint="curve")
    ys = [point[1] for point in points]
    return bbox((wave_left, min(ys), wave_right, max(ys)))


def draw_spectrum(
    draw: ImageDraw.ImageDraw,
    *,
    box: Sequence[float],
    spec: SpectrumSpec,
    style: Any,
    line_width: int,
) -> List[float]:
    """Draw one candidate magnitude spectrum from a symbolic spec.

    Spikes, sinc envelopes, and bell distractors share the same panel coordinate
    system. The returned bbox covers only the drawn spectrum marks, while option
    annotation uses the enclosing panel bbox for stable visual selection.
    """

    left, top, right, bottom = [float(value) for value in box]
    plot_left = left + 50.0
    plot_right = right - 28.0
    baseline = bottom - 34.0
    plot_top = top + 34.0
    plot_height = baseline - plot_top
    stroke = tuple(int(value) for value in style.stroke_rgb)
    spectrum_rgb = tuple(int(value) for value in style.accent_rgb)
    max_bin = 8.0
    used_points: List[Tuple[float, float]] = []
    if spec.kind == "spikes":
        for bin_value, amplitude in zip(spec.bins, spec.amplitudes):
            x = plot_left + ((float(bin_value) / max_bin) * (plot_right - plot_left))
            y = baseline - (float(amplitude) * plot_height * 0.86)
            draw.line((x, baseline, x, y), fill=stroke, width=max(1, int(line_width) + 2))
            draw.line((x, baseline, x, y), fill=spectrum_rgb, width=max(1, int(line_width)))
            draw.ellipse(
                (x - 5.0, y - 5.0, x + 5.0, y + 5.0),
                fill=spectrum_rgb,
                outline=stroke,
                width=1,
            )
            used_points.append((x, y))
    elif spec.kind == "sinc":
        points: List[Tuple[float, float]] = []
        for idx in range(260):
            u = float(idx) / 259.0
            f = u * max_bin
            z = (math.pi * f) / max(0.25, float(spec.lobe_width))
            mag = 1.0 if abs(z) < 1e-6 else abs(math.sin(z) / z)
            x = plot_left + (u * (plot_right - plot_left))
            y = baseline - (mag * plot_height * 0.9)
            points.append((x, y))
        draw.line(points, fill=stroke, width=max(1, int(line_width) + 2), joint="curve")
        draw.line(points, fill=spectrum_rgb, width=max(1, int(line_width)), joint="curve")
        used_points = points
    elif spec.kind == "bell":
        points = []
        for idx in range(220):
            u = float(idx) / 219.0
            mag = math.exp(-((u - 0.36) ** 2) / 0.045)
            x = plot_left + (u * (plot_right - plot_left))
            y = baseline - (mag * plot_height * 0.82)
            points.append((x, y))
        draw.line(points, fill=stroke, width=max(1, int(line_width) + 2), joint="curve")
        draw.line(points, fill=spectrum_rgb, width=max(1, int(line_width)), joint="curve")
        used_points = points
    if not used_points:
        return bbox((plot_left, baseline, plot_right, baseline))
    xs = [point[0] for point in used_points]
    ys = [point[1] for point in used_points]
    return bbox((min(xs) - 6.0, min(ys) - 6.0, max(xs) + 6.0, baseline + 6.0))


def option_box(index: int, render_defaults: Mapping[str, int]) -> Tuple[float, float, float, float]:
    """Return the bbox for one spectrum option panel."""

    option_columns = 2 if len(OPTION_LABELS) <= 4 else 3
    col = int(index) % int(option_columns)
    row = int(index) // int(option_columns)
    left = float(render_defaults["options_left_px"]) + float(col) * (
        float(render_defaults["option_width_px"]) + float(render_defaults["option_gap_x_px"])
    )
    top = float(render_defaults["options_top_px"]) + float(row) * (
        float(render_defaults["option_height_px"]) + float(render_defaults["option_gap_y_px"])
    )
    return (
        left,
        top,
        left + float(render_defaults["option_width_px"]),
        top + float(render_defaults["option_height_px"]),
    )


def render_signal_transform_scene(
    *,
    image: Image.Image,
    axes: SignalTransformAxes,
    scenario: SignalScenario,
    render_defaults: Mapping[str, int],
    font_family: str,
    style: Any,
) -> RenderedSignalTransformScene:
    """Render one full signal-transform option-matching scene.

    The renderer owns only visual layout and projection. It receives the already
    sampled correct option mapping and returns role bboxes plus option metadata
    without selecting answers or branching on public task identity.
    """

    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    axis_font = load_font(int(render_defaults["axis_font_size_px"]), bold=False, font_family=font_family)
    text_rgb = tuple(int(value) for value in style.label_rgb)
    panel_fill = tuple(int(value) for value in style.panel_fill_rgb)
    panel_border = tuple(int(value) for value in style.panel_border_rgb)

    sheet_box = (
        float(render_defaults["sheet_left_px"]),
        float(render_defaults["sheet_top_px"]),
        float(canvas_width - int(render_defaults["sheet_right_margin_px"])),
        float(canvas_height - int(render_defaults["sheet_bottom_margin_px"])),
    )
    draw.rounded_rectangle(sheet_box, radius=20, fill=panel_fill, outline=panel_border, width=3)
    draw_centered_text(
        draw,
        text="Signal spectrum match",
        center=((sheet_box[0] + sheet_box[2]) * 0.5, sheet_box[1] + 34.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )

    input_box = (
        float(render_defaults["input_left_px"]),
        float(render_defaults["input_top_px"]),
        float(render_defaults["input_left_px"] + render_defaults["input_width_px"]),
        float(render_defaults["input_top_px"] + render_defaults["input_height_px"]),
    )
    draw_panel_frame(
        draw,
        box=input_box,
        label="input",
        label_font=label_font,
        axis_font=axis_font,
        style=style,
        scene_variant=str(axes.scene_variant),
    )
    input_wave_bbox = draw_time_waveform(
        draw,
        box=input_box,
        scenario=scenario,
        style=style,
        line_width=int(render_defaults["waveform_line_width_px"]),
    )

    option_bboxes: Dict[str, List[float]] = {}
    spectrum_bboxes: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, object]] = [
        {
            "entity_id": "input_waveform",
            "entity_type": "time_domain_waveform",
            "bbox_px": bbox(input_box),
            "meta": {
                "waveform_family": str(scenario.waveform_family),
                "tone_bins": list(scenario.tone_bins),
                "pulse_width": round(float(scenario.pulse_width), 4),
            },
        }
    ]
    for index, label in enumerate(OPTION_LABELS):
        box = option_box(index, render_defaults)
        draw_panel_frame(
            draw,
            box=box,
            label=str(label),
            label_font=label_font,
            axis_font=axis_font,
            style=style,
            scene_variant=str(axes.scene_variant),
        )
        spec = scenario.option_specs[str(label)]
        spectrum_bbox = draw_spectrum(
            draw,
            box=box,
            spec=spec,
            style=style,
            line_width=int(render_defaults["spectrum_line_width_px"]),
        )
        option_bboxes[str(label)] = bbox(box)
        spectrum_bboxes[str(label)] = list(spectrum_bbox)
        scene_entities.append(
            {
                "entity_id": f"spectrum_{label}",
                "entity_type": "spectrum_option",
                "bbox_px": bbox(box),
                "meta": {
                    "label": str(label),
                    "is_correct": str(label) == str(axes.correct_option_letter),
                    "spectrum": spectrum_payload(spec),
                },
            }
        )

    selected_box = option_bboxes[str(axes.correct_option_letter)]
    annotation_bbox_map = {
        "input_waveform": bbox(input_box),
        "selected_spectrum": list(selected_box),
    }
    render_map = {
        "scene_variant": str(axes.scene_variant),
        "waveform_family": str(scenario.waveform_family),
        "time_cycles": int(scenario.time_cycles),
        "tone_bins": list(scenario.tone_bins),
        "pulse_width": round(float(scenario.pulse_width), 4),
        "correct_option_letter": str(axes.correct_option_letter),
        "input_panel_bbox_px": bbox(input_box),
        "input_wave_bbox_px": list(input_wave_bbox),
        "option_bboxes": dict(option_bboxes),
        "spectrum_bboxes": dict(spectrum_bboxes),
        "option_map": {str(label): spectrum_payload(spec) for label, spec in scenario.option_specs.items()},
        "correct_spectrum": spectrum_payload(scenario.correct_spectrum),
    }
    return RenderedSignalTransformScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_bbox_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )
