"""Rendering primitives for fluid-flow continuity diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many as bbox_union
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, clamp_bbox
from .formulas import continuity_product, pipe_size
from .state import SCENE_ID, SCENE_NAMESPACE, FlowScenario, FluidFlowRenderDefaults, RenderedFlowScene


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
DEFAULTS = FluidFlowRenderDefaults()


def draw_label_box(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: tuple[float, float],
    font: Any,
    style: Any,
    missing: bool = False,
) -> list[float]:
    """Draw one rounded readout label and return its bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    pad_x = 13.0
    pad_y = 8.0
    cx, cy = float(center[0]), float(center[1])
    box = bbox(
        (
            cx - text_w / 2.0 - pad_x,
            cy - text_h / 2.0 - pad_y,
            cx + text_w / 2.0 + pad_x,
            cy + text_h / 2.0 + pad_y,
        )
    )
    if bool(missing):
        fill = (255, 235, 235)
        outline = (170, 42, 42)
        text_rgb = (178, 38, 38)
    else:
        fill = tuple(int(value) for value in style.label_fill_rgb)
        outline = tuple(int(value) for value in style.label_border_rgb)
        text_rgb = tuple(int(value) for value in style.label_rgb)
    draw.rounded_rectangle(tuple(box), radius=10, fill=fill, outline=outline, width=3)
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    return bbox(bbox_union(box, text_draw_bbox))


def draw_station_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    label: str,
    font: Any,
    style: Any,
) -> list[float]:
    """Draw a numbered station marker and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    radius = 21.0
    marker_bbox = bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    draw.ellipse(
        tuple(marker_bbox),
        fill=tuple(int(value) for value in style.label_fill_rgb),
        outline=tuple(int(value) for value in style.stroke_rgb),
        width=3,
    )
    text_bbox = draw_centered_text(
        draw,
        text=str(label),
        center=(cx, cy),
        font=font,
        fill=tuple(int(value) for value in style.label_rgb),
        stroke_fill=resolve_text_stroke_fill(tuple(int(value) for value in style.label_rgb)),
        stroke_width=1,
    )
    return bbox(bbox_union(marker_bbox, text_bbox))


def draw_horizontal_flow(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: FlowScenario,
    style: Any,
    font: Any,
    station_font: Any,
    fluid_rgb: tuple[int, int, int],
    panel: Sequence[float],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Draw the horizontal apparatus while preserving station-role annotation boxes."""

    x1 = float(panel[0] + 255.0)
    x2 = float(panel[2] - 255.0)
    cy = float((panel[1] + panel[3]) * 0.52)
    h1 = pipe_size(int(scenario.area_1_cm2))
    h2 = pipe_size(int(scenario.area_2_cm2))
    stroke = tuple(int(value) for value in style.stroke_rgb)
    pipe_fill = tuple(min(255, int(value) + 42) for value in fluid_rgb)
    body_poly = [
        (x1, cy - h1 / 2.0),
        (x2, cy - h2 / 2.0),
        (x2, cy + h2 / 2.0),
        (x1, cy + h1 / 2.0),
    ]
    draw.polygon(body_poly, fill=pipe_fill, outline=stroke)
    draw.line(body_poly + [body_poly[0]], fill=stroke, width=5)
    draw.ellipse(
        (x1 - 16.0, cy - h1 / 2.0, x1 + 16.0, cy + h1 / 2.0),
        fill=tuple(int(value) for value in fluid_rgb),
        outline=stroke,
        width=4,
    )
    draw.ellipse(
        (x2 - 16.0, cy - h2 / 2.0, x2 + 16.0, cy + h2 / 2.0),
        fill=tuple(int(value) for value in fluid_rgb),
        outline=stroke,
        width=4,
    )

    arrow_start = (float(x1 + 102.0), cy)
    arrow_end = (float(x2 - 102.0), cy)
    draw_arrow(
        draw,
        start=arrow_start,
        end=arrow_end,
        fill=stroke,
        width=7,
        head_length_px=24,
        head_width_px=22,
    )
    arrow_bbox = bbox(
        (
            arrow_start[0] - 12.0,
            arrow_start[1] - 18.0,
            arrow_end[0] + 30.0,
            arrow_end[1] + 18.0,
        )
    )

    marker_1 = draw_station_marker(
        draw,
        center=(x1, cy - h1 / 2.0 - 46.0),
        label="1",
        font=station_font,
        style=style,
    )
    marker_2 = draw_station_marker(
        draw,
        center=(x2, cy - h2 / 2.0 - 46.0),
        label="2",
        font=station_font,
        style=style,
    )
    label_y = cy + max(h1, h2) / 2.0
    area_1 = draw_label_box(
        draw,
        text=f"A1 = {int(scenario.area_1_cm2)} cm^2",
        center=(x1, label_y + 58.0),
        font=font,
        style=style,
    )
    area_2 = draw_label_box(
        draw,
        text=f"A2 = {int(scenario.area_2_cm2)} cm^2",
        center=(x2, label_y + 58.0),
        font=font,
        style=style,
    )
    speed_1_text = "v1 = ?" if str(scenario.missing_station) == "v1" else (
        f"v1 = {int(scenario.speed_1_m_s)} m/s"
    )
    speed_2_text = "v2 = ?" if str(scenario.missing_station) == "v2" else (
        f"v2 = {int(scenario.speed_2_m_s)} m/s"
    )
    speed_1 = draw_label_box(
        draw,
        text=speed_1_text,
        center=(x1, label_y + 112.0),
        font=font,
        style=style,
        missing=str(scenario.missing_station) == "v1",
    )
    speed_2 = draw_label_box(
        draw,
        text=speed_2_text,
        center=(x2, label_y + 112.0),
        font=font,
        style=style,
        missing=str(scenario.missing_station) == "v2",
    )
    pipe_bbox = bbox((x1 - 20.0, cy - max(h1, h2) / 2.0 - 4.0, x2 + 20.0, cy + max(h1, h2) / 2.0 + 4.0))
    station_1_bbox = bbox(
        bbox_union(
            marker_1,
            area_1,
            speed_1,
            (x1 - 24.0, cy - h1 / 2.0, x1 + 24.0, cy + h1 / 2.0),
            padding=6.0,
        )
    )
    station_2_bbox = bbox(
        bbox_union(
            marker_2,
            area_2,
            speed_2,
            (x2 - 24.0, cy - h2 / 2.0, x2 + 24.0, cy + h2 / 2.0),
            padding=6.0,
        )
    )
    flow_path_bbox = bbox(bbox_union(pipe_bbox, arrow_bbox, padding=8.0))
    missing_speed_bbox = speed_1 if str(scenario.missing_station) == "v1" else speed_2
    return (
        {"missing_speed_label": bbox(missing_speed_bbox)},
        {
            "pipe_bbox_px": pipe_bbox,
            "flow_arrow_bbox_px": arrow_bbox,
            "flow_path_bbox_px": flow_path_bbox,
            "station_bboxes_px": {
                "station_1": list(station_1_bbox),
                "station_2": list(station_2_bbox),
            },
            "area_label_bboxes_px": {
                "area_1": list(area_1),
                "area_2": list(area_2),
            },
            "speed_label_bboxes_px": {
                "speed_1": list(speed_1),
                "speed_2": list(speed_2),
            },
            "missing_speed_label_bbox_px": list(missing_speed_bbox),
            "station_1_center_px": [round(float(x1), 3), round(float(cy), 3)],
            "station_2_center_px": [round(float(x2), 3), round(float(cy), 3)],
        },
    )


def draw_vertical_flow(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: FlowScenario,
    style: Any,
    font: Any,
    station_font: Any,
    fluid_rgb: tuple[int, int, int],
    panel: Sequence[float],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Draw the vertical apparatus while preserving station-role annotation boxes."""

    cx = float((panel[0] + panel[2]) * 0.5)
    y1 = float(panel[1] + 204.0)
    y2 = float(panel[3] - 126.0)
    w1 = pipe_size(int(scenario.area_1_cm2))
    w2 = pipe_size(int(scenario.area_2_cm2))
    stroke = tuple(int(value) for value in style.stroke_rgb)
    pipe_fill = tuple(min(255, int(value) + 42) for value in fluid_rgb)
    body_poly = [
        (cx - w1 / 2.0, y1),
        (cx + w1 / 2.0, y1),
        (cx + w2 / 2.0, y2),
        (cx - w2 / 2.0, y2),
    ]
    draw.polygon(body_poly, fill=pipe_fill, outline=stroke)
    draw.line(body_poly + [body_poly[0]], fill=stroke, width=5)
    draw.ellipse(
        (cx - w1 / 2.0, y1 - 16.0, cx + w1 / 2.0, y1 + 16.0),
        fill=tuple(int(value) for value in fluid_rgb),
        outline=stroke,
        width=4,
    )
    draw.ellipse(
        (cx - w2 / 2.0, y2 - 16.0, cx + w2 / 2.0, y2 + 16.0),
        fill=tuple(int(value) for value in fluid_rgb),
        outline=stroke,
        width=4,
    )
    arrow_start = (cx, float(y1 + 88.0))
    arrow_end = (cx, float(y2 - 88.0))
    draw_arrow(
        draw,
        start=arrow_start,
        end=arrow_end,
        fill=stroke,
        width=7,
        head_length_px=24,
        head_width_px=22,
    )
    arrow_bbox = bbox((cx - 18.0, arrow_start[1] - 12.0, cx + 18.0, arrow_end[1] + 30.0))

    marker_1 = draw_station_marker(
        draw,
        center=(cx - w1 / 2.0 - 48.0, y1),
        label="1",
        font=station_font,
        style=style,
    )
    marker_2 = draw_station_marker(
        draw,
        center=(cx + w2 / 2.0 + 48.0, y2),
        label="2",
        font=station_font,
        style=style,
    )
    area_1 = draw_label_box(
        draw,
        text=f"A1 = {int(scenario.area_1_cm2)} cm^2",
        center=(cx - 230.0, y1 + 48.0),
        font=font,
        style=style,
    )
    area_2 = draw_label_box(
        draw,
        text=f"A2 = {int(scenario.area_2_cm2)} cm^2",
        center=(cx + 230.0, y2 - 48.0),
        font=font,
        style=style,
    )
    speed_1_text = "v1 = ?" if str(scenario.missing_station) == "v1" else (
        f"v1 = {int(scenario.speed_1_m_s)} m/s"
    )
    speed_2_text = "v2 = ?" if str(scenario.missing_station) == "v2" else (
        f"v2 = {int(scenario.speed_2_m_s)} m/s"
    )
    speed_1 = draw_label_box(
        draw,
        text=speed_1_text,
        center=(cx - 230.0, y1 + 102.0),
        font=font,
        style=style,
        missing=str(scenario.missing_station) == "v1",
    )
    speed_2 = draw_label_box(
        draw,
        text=speed_2_text,
        center=(cx + 230.0, y2 + 6.0),
        font=font,
        style=style,
        missing=str(scenario.missing_station) == "v2",
    )
    pipe_bbox = bbox(
        (
            cx - max(w1, w2) / 2.0 - 4.0,
            y1 - 20.0,
            cx + max(w1, w2) / 2.0 + 4.0,
            y2 + 20.0,
        )
    )
    station_1_bbox = bbox(
        bbox_union(
            marker_1,
            area_1,
            speed_1,
            (cx - w1 / 2.0, y1 - 24.0, cx + w1 / 2.0, y1 + 24.0),
            padding=6.0,
        )
    )
    station_2_bbox = bbox(
        bbox_union(
            marker_2,
            area_2,
            speed_2,
            (cx - w2 / 2.0, y2 - 24.0, cx + w2 / 2.0, y2 + 24.0),
            padding=6.0,
        )
    )
    flow_path_bbox = bbox(bbox_union(pipe_bbox, arrow_bbox, padding=8.0))
    missing_speed_bbox = speed_1 if str(scenario.missing_station) == "v1" else speed_2
    return (
        {"missing_speed_label": bbox(missing_speed_bbox)},
        {
            "pipe_bbox_px": pipe_bbox,
            "flow_arrow_bbox_px": arrow_bbox,
            "flow_path_bbox_px": flow_path_bbox,
            "station_bboxes_px": {
                "station_1": list(station_1_bbox),
                "station_2": list(station_2_bbox),
            },
            "area_label_bboxes_px": {
                "area_1": list(area_1),
                "area_2": list(area_2),
            },
            "speed_label_bboxes_px": {
                "speed_1": list(speed_1),
                "speed_2": list(speed_2),
            },
            "missing_speed_label_bbox_px": list(missing_speed_bbox),
            "station_1_center_px": [round(float(cx), 3), round(float(y1), 3)],
            "station_2_center_px": [round(float(cx), 3), round(float(y2), 3)],
        },
    )


def render_fluid_flow(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: FlowScenario,
    rendering_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedFlowScene:
    """Render one fluid-flow continuity diagram from a resolved scenario."""

    canvas_width = int(rendering_defaults.get("canvas_width", DEFAULTS.canvas_width))
    canvas_height = int(rendering_defaults.get("canvas_height", DEFAULTS.canvas_height))
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
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    label_font = load_font(
        int(rendering_defaults.get("label_font_size_px", DEFAULTS.label_font_size_px)),
        bold=True,
        font_family=str(font_family),
    )
    station_font = load_font(
        int(rendering_defaults.get("station_font_size_px", DEFAULTS.station_font_size_px)),
        bold=True,
        font_family=str(font_family),
    )
    title_font = load_font(
        int(rendering_defaults.get("title_font_size_px", DEFAULTS.title_font_size_px)),
        bold=True,
        font_family=str(font_family),
    )
    panel = (
        float(rendering_defaults.get("panel_margin_x_px", DEFAULTS.panel_margin_x_px)),
        float(rendering_defaults.get("panel_margin_top_px", DEFAULTS.panel_margin_top_px)),
        float(canvas_width - rendering_defaults.get("panel_margin_x_px", DEFAULTS.panel_margin_x_px)),
        float(canvas_height - rendering_defaults.get("panel_margin_bottom_px", DEFAULTS.panel_margin_bottom_px)),
    )
    draw.rounded_rectangle(
        panel,
        radius=18,
        fill=tuple(int(value) for value in diagram_style.panel_fill_rgb),
        outline=tuple(int(value) for value in diagram_style.panel_border_rgb),
        width=3,
    )
    draw_label_box(
        draw,
        text="Steady incompressible flow",
        center=(float(canvas_width * 0.5), float(panel[1] + 48.0)),
        font=title_font,
        style=diagram_style,
    )
    fluid_palette = (
        (85, 168, 220),
        (72, 184, 154),
        (126, 149, 217),
        (213, 143, 77),
        (153, 123, 196),
    )
    fluid_rgb = spawn_rng(int(instance_seed), f"{namespace}.fluid_color").choice(fluid_palette)
    if str(scenario.orientation) == "vertical_pipe":
        annotation_map, render_geometry = draw_vertical_flow(
            draw,
            scenario=scenario,
            style=diagram_style,
            font=label_font,
            station_font=station_font,
            fluid_rgb=fluid_rgb,
            panel=panel,
        )
    else:
        annotation_map, render_geometry = draw_horizontal_flow(
            draw,
            scenario=scenario,
            style=diagram_style,
            font=label_font,
            station_font=station_font,
            fluid_rgb=fluid_rgb,
            panel=panel,
        )

    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_map = {
        str(key): clamp_bbox(value, width=int(image.size[0]), height=int(image.size[1]))
        for key, value in annotation_map.items()
    }
    station_bboxes = {
        str(key): clamp_bbox(value, width=int(image.size[0]), height=int(image.size[1]))
        for key, value in dict(render_geometry.get("station_bboxes_px", {})).items()
    }
    flow_path_bbox = clamp_bbox(
        render_geometry.get("flow_path_bbox_px", render_geometry.get("pipe_bbox_px", (0, 0, 1, 1))),
        width=int(image.size[0]),
        height=int(image.size[1]),
    )
    scene_entities = [
        {
            "id": "station_1",
            "area_cm2": int(scenario.area_1_cm2),
            "speed_m_s": int(scenario.speed_1_m_s),
            "bbox_px": list(station_bboxes["station_1"]),
        },
        {
            "id": "station_2",
            "area_cm2": int(scenario.area_2_cm2),
            "speed_m_s": int(scenario.speed_2_m_s),
            "bbox_px": list(station_bboxes["station_2"]),
        },
        {
            "id": "missing_speed_label",
            "missing_station": str(scenario.missing_station),
            "bbox_px": list(annotation_map["missing_speed_label"]),
        },
        {"id": "flow_path", "bbox_px": list(flow_path_bbox)},
    ]
    render_map = {
        "orientation": str(scenario.orientation),
        "missing_station": str(scenario.missing_station),
        "area_1_cm2": int(scenario.area_1_cm2),
        "area_2_cm2": int(scenario.area_2_cm2),
        "speed_1_m_s": int(scenario.speed_1_m_s),
        "speed_2_m_s": int(scenario.speed_2_m_s),
        "target_answer": int(scenario.target_answer),
        "continuity_lhs": continuity_product(
            area_cm2=int(scenario.area_1_cm2),
            speed_m_s=int(scenario.speed_1_m_s),
        ),
        "continuity_rhs": continuity_product(
            area_cm2=int(scenario.area_2_cm2),
            speed_m_s=int(scenario.speed_2_m_s),
        ),
        "fluid_rgb": [int(value) for value in fluid_rgb],
        "technical_diagram_style": dict(diagram_style_meta),
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    render_map.update(render_geometry)
    return RenderedFlowScene(
        image=image,
        annotation_bbox_map=annotation_map,
        scene_entities=scene_entities,
        render_map=render_map,
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )
