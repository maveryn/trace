"""Rendering primitives for two-source wave-interference diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw

from trace_tasks.tasks.physics.shared.style import build_physics_waves_theme
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many as _bbox_union_many
from trace_tasks.tasks.shared.drawing import (
    draw_centered_text,
    draw_dashed_line,
    draw_rounded_rect,
)
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, bbox_union, point, segment_set
from .state import (
    SOURCE_SEPARATION_STEPS,
    RenderedWaveInterferenceScene,
    SceneSpec,
    WaveInterferenceDefaults,
)


def line_bbox(
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    padding_px: float,
) -> List[float]:
    """Return a conservative bbox for one line segment."""

    return bbox(
        (
            min(float(start[0]), float(end[0])) - float(padding_px),
            min(float(start[1]), float(end[1])) - float(padding_px),
            max(float(start[0]), float(end[0])) + float(padding_px),
            max(float(start[1]), float(end[1])) + float(padding_px),
        )
    )


def board_bbox(render_defaults: Mapping[str, Any]) -> List[float]:
    """Return the tank board bbox in final image pixels."""

    return bbox(
        (
            float(render_defaults["board_left_px"]),
            float(render_defaults["board_top_px"]),
            float(render_defaults["board_left_px"]) + float(render_defaults["board_width_px"]),
            float(render_defaults["board_top_px"]) + float(render_defaults["board_height_px"]),
        )
    )


def wave_content_bbox(render_defaults: Mapping[str, Any]) -> List[float]:
    """Return a conservative bbox for the whole wave tank before placement."""

    board = board_bbox(render_defaults)
    return bbox((board[0] - 8.0, board[1] - 8.0, board[2] + 8.0, board[3] + 8.0))


def resolve_wave_render_defaults(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    fallback_defaults: WaveInterferenceDefaults,
    instance_seed: int,
    namespace: str,
) -> Dict[str, int]:
    """Resolve integer render defaults for a wave-interference scene."""

    resolved = {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(fallback_defaults, key)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in (
            "canvas_width",
            "canvas_height",
            "board_left_px",
            "board_top_px",
            "board_width_px",
            "board_height_px",
            "half_wavelength_px",
            "ring_count",
            "grid_line_width_px",
            "source_radius_px",
            "candidate_radius_px",
            "point_radius_px",
            "wavefront_width_px",
            "guide_width_px",
            "label_font_size_px",
            "source_font_size_px",
            "candidate_font_size_px",
            "note_font_size_px",
        )
    }
    return resolved


def resolve_wave_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve whole-tank placement before annotation projection."""

    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    content = wave_content_bbox(render_defaults)
    content_left, content_top, content_right, content_bottom = [float(value) for value in content]
    jitter = resolve_layout_jitter(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.wave_layout",
    )
    min_margin = int(jitter.get("min_margin_px", 18))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if min_dx > max_dx:
        min_dx = 0
        max_dx = 0
    if min_dy > max_dy:
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))

    adjusted = dict(render_defaults)
    adjusted["board_left_px"] = int(adjusted["board_left_px"]) + int(dx)
    adjusted["board_top_px"] = int(adjusted["board_top_px"]) + int(dy)

    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    final_bbox = bbox(
        (
            float(content_left) + float(dx),
            float(content_top) + float(dy),
            float(content_right) + float(dx),
            float(content_bottom) + float(dy),
        )
    )
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_wave_tank_offset",
            "content_bbox_px": list(content),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement


def source_positions(
    board: Sequence[float],
    *,
    unit_px: float,
) -> Dict[str, Tuple[float, float]]:
    """Return the two source centers in pixels."""

    cx = (float(board[0]) + float(board[2])) / 2.0
    cy = (float(board[1]) + float(board[3])) / 2.0
    half_sep = 0.5 * float(SOURCE_SEPARATION_STEPS) * float(unit_px)
    return {"S1": (float(cx - half_sep), float(cy)), "S2": (float(cx + half_sep), float(cy))}


def point_to_px(
    board: Sequence[float],
    *,
    unit_px: float,
    x_steps: float,
    y_steps: float,
) -> Tuple[float, float]:
    """Project half-wavelength coordinates into pixels."""

    cx = (float(board[0]) + float(board[2])) / 2.0
    cy = (float(board[1]) + float(board[3])) / 2.0
    return (
        float(cx + (float(x_steps) * float(unit_px))),
        float(cy - (float(y_steps) * float(unit_px))),
    )


def _draw_text_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: Any,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    text_rgb: Sequence[int],
    pad_x: float = 11.0,
    pad_y: float = 7.0,
) -> List[float]:
    """Draw a small rounded label and return its bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    width = float(text_bbox[2] - text_bbox[0])
    height = float(text_bbox[3] - text_bbox[1])
    cx, cy = float(center[0]), float(center[1])
    tag_bbox = bbox(
        (
            cx - (width / 2.0) - float(pad_x),
            cy - (height / 2.0) - float(pad_y),
            cx + (width / 2.0) + float(pad_x),
            cy + (height / 2.0) + float(pad_y),
        )
    )
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in tag_bbox),
        radius=9,
        fill=fill_rgb,
        outline=outline_rgb,
        width=2,
    )
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=text_rgb,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=1,
    )
    return bbox(_bbox_union_many(tag_bbox, text_draw_bbox))


def _draw_tank(
    draw: ImageDraw.ImageDraw,
    *,
    board: Sequence[float],
    scene_variant: str,
    theme: Any,
    render_defaults: Mapping[str, Any],
) -> None:
    """Draw the ripple-tank panel and optional grid."""

    fill_rgb = theme.tank_alt_fill_rgb if str(scene_variant) == "lab_sheet" else theme.tank_fill_rgb
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in board),
        radius=12,
        fill=fill_rgb,
        outline=theme.tank_outline_rgb,
        width=3,
    )
    left, top, right, bottom = [float(value) for value in board[:4]]
    if str(scene_variant) in {"grid_tank", "lab_sheet"}:
        unit_px = float(render_defaults["half_wavelength_px"])
        x = left + unit_px
        while x < right:
            draw.line(
                [(x, top), (x, bottom)],
                fill=theme.grid_rgb,
                width=max(1, int(render_defaults["grid_line_width_px"])),
            )
            x += unit_px
        y = top + unit_px
        while y < bottom:
            draw.line(
                [(left, y), (right, y)],
                fill=theme.grid_rgb,
                width=max(1, int(render_defaults["grid_line_width_px"])),
            )
            y += unit_px
    if str(scene_variant) == "lab_sheet":
        for offset in range(0, int(bottom - top), 34):
            y = top + float(offset)
            draw.line([(left, y), (right, y)], fill=(236, 240, 245), width=1)


def _draw_dashed_ellipse(
    draw: ImageDraw.ImageDraw,
    ellipse_bbox: Sequence[float],
    *,
    fill: Sequence[int],
    width: int,
    dash_degrees: int = 11,
    gap_degrees: int = 9,
) -> None:
    """Draw a dashed ellipse using short arc segments."""

    angle = 0
    while angle < 360:
        draw.arc(
            tuple(float(value) for value in ellipse_bbox),
            start=int(angle),
            end=int(min(360, angle + dash_degrees)),
            fill=tuple(int(value) for value in fill),
            width=max(1, int(width)),
        )
        angle += int(dash_degrees + gap_degrees)


def _draw_wavefronts(
    image: Image.Image,
    *,
    board: Sequence[float],
    source_centers: Mapping[str, Tuple[float, float]],
    phase_relation: str,
    theme: Any,
    render_defaults: Mapping[str, Any],
) -> Image.Image:
    """Draw clipped crest/trough rings for the two wave sources."""

    unit_px = float(render_defaults["half_wavelength_px"])
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    ring_count = int(render_defaults["ring_count"])
    width = max(1, int(render_defaults["wavefront_width_px"]))
    offsets = {"S1": 0, "S2": 0 if str(phase_relation) == "in_phase" else 1}
    for source_id, center in source_centers.items():
        offset = int(offsets[str(source_id)])
        for step in range(1, ring_count + 1):
            radius = float(step) * unit_px
            ring_bbox = [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ]
            if (int(step) + int(offset)) % 2 == 0:
                layer_draw.ellipse(
                    tuple(ring_bbox),
                    outline=tuple(int(value) for value in theme.crest_rgb) + (112,),
                    width=width,
                )
            else:
                _draw_dashed_ellipse(
                    layer_draw,
                    ring_bbox,
                    fill=tuple(int(value) for value in theme.trough_rgb) + (92,),
                    width=max(1, width - 1),
                )

    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(tuple(float(value) for value in board), radius=12, fill=255)
    alpha = ImageChops.multiply(layer.getchannel("A"), mask)
    layer.putalpha(alpha)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def _draw_source(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    source_id: str,
    theme: Any,
    render_defaults: Mapping[str, Any],
    font: Any,
) -> List[float]:
    """Draw one source marker and return its bbox."""

    radius = float(render_defaults["source_radius_px"])
    source_bbox = bbox((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius))
    draw.ellipse(tuple(source_bbox), fill=theme.source_fill_rgb, outline=theme.source_outline_rgb, width=3)
    text_bbox = draw_centered_text(
        draw,
        text=str(source_id),
        center=center,
        font=font,
        fill=theme.source_text_rgb,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.source_text_rgb)),
        stroke_width=1,
    )
    return bbox(_bbox_union_many(source_bbox, text_bbox))


def _draw_candidate(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    letter: str,
    theme: Any,
    render_defaults: Mapping[str, Any],
    font: Any,
) -> List[float]:
    """Draw one labeled candidate point and return its bbox."""

    radius = float(render_defaults["candidate_radius_px"])
    candidate_bbox = bbox((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius))
    halo = bbox((center[0] - radius - 5.0, center[1] - radius - 5.0, center[0] + radius + 5.0, center[1] + radius + 5.0))
    draw.ellipse(tuple(halo), fill=theme.label_fill_rgb, outline=theme.candidate_outline_rgb, width=1)
    draw.ellipse(tuple(candidate_bbox), fill=theme.candidate_fill_rgb, outline=theme.candidate_outline_rgb, width=3)
    text_bbox = draw_centered_text(
        draw,
        text=str(letter),
        center=center,
        font=font,
        fill=theme.candidate_text_rgb,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.candidate_text_rgb)),
        stroke_width=1,
    )
    return bbox(_bbox_union_many(candidate_bbox, text_bbox))


def _draw_point_p(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    theme: Any,
    render_defaults: Mapping[str, Any],
    font: Any,
) -> List[float]:
    """Draw highlighted point P and return its bbox."""

    radius = float(render_defaults["point_radius_px"])
    point_bbox = bbox((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius))
    halo = bbox((center[0] - radius - 5.0, center[1] - radius - 5.0, center[0] + radius + 5.0, center[1] + radius + 5.0))
    draw.ellipse(tuple(halo), fill=theme.label_fill_rgb, outline=theme.point_outline_rgb, width=1)
    draw.ellipse(tuple(point_bbox), fill=theme.point_fill_rgb, outline=theme.point_outline_rgb, width=3)
    text_bbox = draw_centered_text(
        draw,
        text="P",
        center=center,
        font=font,
        fill=theme.candidate_text_rgb,
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.candidate_text_rgb)),
        stroke_width=1,
    )
    return bbox(_bbox_union_many(point_bbox, text_bbox))


def _phase_label(phase_relation: str) -> str:
    """Return a short source-phase label."""

    return "sources in phase" if str(phase_relation) == "in_phase" else "S2 opposite phase"


def render_wave_interference_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    scene_spec: SceneSpec,
    diagram_style: Any | None = None,
    font_family: str | None = None,
) -> RenderedWaveInterferenceScene:
    """Render the full tank and project annotations from final geometry.

    The renderer is intentionally task-identity-free: the scene spec selects
    whether candidate points or a marked point P is drawn, while this function
    owns only visual construction, entity bboxes, and final pixel projection.
    """

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_physics_waves_theme(str(accent_color_name), diagram_style=diagram_style)
    resolved_font_family = None if font_family is None else str(font_family)
    label_font = load_font(
        int(render_defaults["label_font_size_px"]),
        bold=False,
        font_family=resolved_font_family,
    )
    source_font = load_font(
        int(render_defaults["source_font_size_px"]),
        bold=True,
        font_family=resolved_font_family,
    )
    candidate_font = load_font(
        int(render_defaults["candidate_font_size_px"]),
        bold=True,
        font_family=resolved_font_family,
    )
    note_font = load_font(
        int(render_defaults["note_font_size_px"]),
        bold=False,
        font_family=resolved_font_family,
    )
    board = board_bbox(render_defaults)
    _draw_tank(
        draw,
        board=board,
        scene_variant=str(scene_spec.scene_variant),
        theme=theme,
        render_defaults=render_defaults,
    )
    source_centers = source_positions(
        board,
        unit_px=float(render_defaults["half_wavelength_px"]),
    )
    image = _draw_wavefronts(
        image,
        board=board,
        source_centers=source_centers,
        phase_relation=str(scene_spec.phase_relation),
        theme=theme,
        render_defaults=render_defaults,
    )
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(tuple(float(value) for value in board), radius=12, outline=theme.tank_outline_rgb, width=3)
    scene_entities: list[dict[str, Any]] = []
    render_map: dict[str, Any] = {
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "board_bbox_px": list(board),
    }

    source_bboxes: dict[str, list[float]] = {}
    for source_id, center in source_centers.items():
        source_bbox = _draw_source(
            draw,
            center=center,
            source_id=str(source_id),
            theme=theme,
            render_defaults=render_defaults,
            font=source_font,
        )
        source_bboxes[str(source_id)] = list(source_bbox)
        scene_entities.append(
            {
                "entity_id": str(source_id),
                "entity_type": "wave_source",
                "bbox": list(source_bbox),
                "meta": {
                    "source_id": str(source_id),
                    "phase_relation": str(scene_spec.phase_relation),
                },
            }
        )

    phase_bbox = _draw_text_tag(
        draw,
        text=_phase_label(str(scene_spec.phase_relation)),
        center=(board[0] + 150.0, board[1] + 34.0),
        font=label_font,
        fill_rgb=theme.label_fill_rgb,
        outline_rgb=theme.label_outline_rgb,
        text_rgb=theme.label_text_rgb,
    )
    legend_bbox = _draw_text_tag(
        draw,
        text="ring gap = lambda/2",
        center=(board[2] - 170.0, board[3] - 34.0),
        font=note_font,
        fill_rgb=theme.label_fill_rgb,
        outline_rgb=theme.label_outline_rgb,
        text_rgb=theme.label_text_rgb,
    )
    scene_entities.append(
        {
            "entity_id": "phase_relation_label",
            "entity_type": "phase_label",
            "bbox": list(phase_bbox),
            "meta": {"phase_relation": str(scene_spec.phase_relation)},
        }
    )
    scene_entities.append(
        {
            "entity_id": "ring_spacing_label",
            "entity_type": "scale_label",
            "bbox": list(legend_bbox),
            "meta": {"spacing": "lambda/2"},
        }
    )

    annotation_point: list[float] | None = None
    annotation_segments: list[list[list[float]]] = []
    unit_px = float(render_defaults["half_wavelength_px"])

    if scene_spec.choice_scenario is not None:
        candidate_bboxes: dict[str, list[float]] = {}
        candidate_centers: dict[str, list[float]] = {}
        for candidate in scene_spec.choice_scenario.candidates:
            center = point_to_px(
                board,
                unit_px=unit_px,
                x_steps=float(candidate.x_steps),
                y_steps=float(candidate.y_steps),
            )
            candidate_bbox = _draw_candidate(
                draw,
                center=center,
                letter=str(candidate.letter),
                theme=theme,
                render_defaults=render_defaults,
                font=candidate_font,
            )
            candidate_bboxes[str(candidate.letter)] = list(candidate_bbox)
            candidate_centers[str(candidate.letter)] = point(center)
            scene_entities.append(
                {
                    "entity_id": f"candidate_{str(candidate.letter)}",
                    "entity_type": "interference_candidate_point",
                    "bbox": list(candidate_bbox),
                    "meta": {
                        "option_letter": str(candidate.letter),
                        "x_steps": float(candidate.x_steps),
                        "y_steps": float(candidate.y_steps),
                        "s1_distance_steps": int(candidate.r1_steps),
                        "s2_distance_steps": int(candidate.r2_steps),
                        "condition": str(candidate.condition),
                        "is_correct": bool(candidate.is_correct),
                    },
                }
            )
        correct_letter = str(scene_spec.choice_scenario.correct_option_letter)
        annotation_point = list(candidate_centers[correct_letter])
        render_map.update(
            {
                "source_bboxes_px": dict(source_bboxes),
                "candidate_bboxes_px": dict(candidate_bboxes),
                "candidate_centers_px": dict(candidate_centers),
                "annotation_point_px": list(annotation_point),
            }
        )
        annotation_type = "point"

    elif scene_spec.path_scenario is not None:
        scenario = scene_spec.path_scenario
        point_center = point_to_px(
            board,
            unit_px=unit_px,
            x_steps=float(scenario.point_x_steps),
            y_steps=float(scenario.point_y_steps),
        )
        s1_center = source_centers["S1"]
        s2_center = source_centers["S2"]
        draw_dashed_line(
            draw,
            start=s1_center,
            end=point_center,
            fill=theme.guide_rgb,
            width=max(1, int(render_defaults["guide_width_px"])),
            dash_px=16.0,
            gap_px=9.0,
        )
        draw_dashed_line(
            draw,
            start=s2_center,
            end=point_center,
            fill=theme.guide_rgb,
            width=max(1, int(render_defaults["guide_width_px"])),
            dash_px=16.0,
            gap_px=9.0,
        )
        line_1_bbox = line_bbox(s1_center, point_center, padding_px=18.0)
        line_2_bbox = line_bbox(s2_center, point_center, padding_px=18.0)
        point_bbox = _draw_point_p(
            draw,
            center=point_center,
            theme=theme,
            render_defaults=render_defaults,
            font=candidate_font,
        )
        tag_1_bbox = _draw_text_tag(
            draw,
            text="S1P",
            center=((s1_center[0] + point_center[0]) / 2.0, (s1_center[1] + point_center[1]) / 2.0 - 20.0),
            font=note_font,
            fill_rgb=theme.label_fill_rgb,
            outline_rgb=theme.label_outline_rgb,
            text_rgb=theme.label_text_rgb,
        )
        tag_2_bbox = _draw_text_tag(
            draw,
            text="S2P",
            center=((s2_center[0] + point_center[0]) / 2.0, (s2_center[1] + point_center[1]) / 2.0 + 20.0),
            font=note_font,
            fill_rgb=theme.label_fill_rgb,
            outline_rgb=theme.label_outline_rgb,
            text_rgb=theme.label_text_rgb,
        )
        path_s1p_bbox = bbox_union(line_1_bbox, tag_1_bbox, point_bbox)
        path_s2p_bbox = bbox_union(line_2_bbox, tag_2_bbox, point_bbox)
        witness_bbox = bbox_union(
            source_bboxes["S1"],
            source_bboxes["S2"],
            path_s1p_bbox,
            path_s2p_bbox,
        )
        annotation_segments = segment_set(((s1_center, point_center), (s2_center, point_center)))
        scene_entities.append(
            {
                "entity_id": "point_P",
                "entity_type": "path_difference_point",
                "bbox": list(point_bbox),
                "meta": {
                    "x_steps": float(scenario.point_x_steps),
                    "y_steps": float(scenario.point_y_steps),
                    "s1_distance_steps": int(scenario.r1_steps),
                    "s2_distance_steps": int(scenario.r2_steps),
                },
            }
        )
        scene_entities.append(
            {
                "entity_id": "path_S1P",
                "entity_type": "path_difference_guide",
                "bbox": list(path_s1p_bbox),
                "meta": {
                    "path_key": "S1P",
                    "source_id": "S1",
                    "target_id": "P",
                    "distance_steps": int(scenario.r1_steps),
                    "segment_px": list(annotation_segments[0]),
                },
            }
        )
        scene_entities.append(
            {
                "entity_id": "path_S2P",
                "entity_type": "path_difference_guide",
                "bbox": list(path_s2p_bbox),
                "meta": {
                    "path_key": "S2P",
                    "source_id": "S2",
                    "target_id": "P",
                    "distance_steps": int(scenario.r2_steps),
                    "segment_px": list(annotation_segments[1]),
                },
            }
        )
        scene_entities.append(
            {
                "entity_id": "path_difference_witness_region",
                "entity_type": "path_difference_witness_region",
                "bbox": list(witness_bbox),
                "meta": {"path_difference_steps": int(scenario.path_difference_steps)},
            }
        )
        render_map.update(
            {
                "source_bboxes_px": dict(source_bboxes),
                "point_p_bbox_px": list(point_bbox),
                "path_s1_bbox_px": list(line_1_bbox),
                "path_s2_bbox_px": list(line_2_bbox),
                "path_s1_label_bbox_px": list(tag_1_bbox),
                "path_s2_label_bbox_px": list(tag_2_bbox),
                "path_s1p_bbox_px": list(path_s1p_bbox),
                "path_s2p_bbox_px": list(path_s2p_bbox),
                "path_difference_witness_region_bbox_px": list(witness_bbox),
                "annotation_segment_set_px": [list(segment) for segment in annotation_segments],
            }
        )
        annotation_type = "segment_set"
    else:
        raise ValueError("wave scene requires a choice or path-difference scenario")

    return RenderedWaveInterferenceScene(
        image=image,
        annotation_type=str(annotation_type),
        annotation_point=annotation_point,
        annotation_segments=[list(value) for value in annotation_segments],
        annotation_entity_ids=[str(entity_id) for entity_id in scene_spec.annotation_entity_ids],
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )


__all__ = [
    "board_bbox",
    "point_to_px",
    "render_wave_interference_scene",
    "resolve_wave_layout_placement",
    "resolve_wave_render_defaults",
    "source_positions",
]
