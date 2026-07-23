"""Rendering helpers for 3D landmark correspondence panels."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font
from .layout import (
    CANDIDATE_VIEW_KEY,
    REFERENCE_VIEW_KEY,
    offset_point,
    offset_point_map,
    panel_layout,
    render_two_view_object_scene,
)


LANDMARK_MARKER_RADIUS_PX = 12.0
LANDMARK_MARKER_RGB = (220, 36, 44)
_Q_FIELD = "query" + "_id"


def _bbox_union(*bboxes: Sequence[float]) -> List[float]:
    return [
        round(float(min(float(bbox[0]) for bbox in bboxes)), 3),
        round(float(min(float(bbox[1]) for bbox in bboxes)), 3),
        round(float(max(float(bbox[2]) for bbox in bboxes)), 3),
        round(float(max(float(bbox[3]) for bbox in bboxes)), 3),
    ]


def render_landmark_view_dataset(dataset: Mapping[str, Any], view_key: str) -> Dict[str, Any]:
    view = dataset["views"][str(view_key)]
    return {
        _Q_FIELD: str(dataset[_Q_FIELD]),
        "scene_variant": str(dataset["scene_variant"]),
        "point_specs": [dict(spec) for spec in view["point_specs"]],
        "context_object_specs": [],
        "answer_label": "",
        "answer_point_id": "",
        "camera": dict(view["camera"]),
        "projection_frame": dict(view["projection_frame"]),
    }


def _landmark_render_options(_view_key: str) -> Dict[str, Any]:
    return {"draw_candidate_labels": False, "compute_single_annotation": False}


def _draw_landmark_marker(
    draw: ImageDraw.ImageDraw,
    *,
    point: Sequence[float],
    label: str,
    canvas_width: int,
    canvas_height: int,
    role: str,
) -> List[float]:
    x = float(point[0])
    y = float(point[1])
    radius = float(LANDMARK_MARKER_RADIUS_PX)
    ring_bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(tuple(ring_bbox), outline=LANDMARK_MARKER_RGB, width=4)
    draw.ellipse((x - 2.2, y - 2.2, x + 2.2, y + 2.2), fill=LANDMARK_MARKER_RGB)
    font = load_font(24 if str(label) == "REF" else 25, bold=True)
    try:
        text_bbox_raw = draw.textbbox((0.0, 0.0), str(label), font=font, stroke_width=3)
    except Exception:
        text_bbox_raw = (0.0, 0.0, 26.0, 24.0)
    text_width = float(text_bbox_raw[2] - text_bbox_raw[0])
    text_height = float(text_bbox_raw[3] - text_bbox_raw[1])
    label_x = x + radius + 5.0
    if label_x + text_width + 4.0 > float(canvas_width):
        label_x = x - radius - text_width - 7.0
    label_y = y - text_height * 0.66
    label_y = max(3.0, min(float(canvas_height) - text_height - 4.0, label_y))
    text_record = draw_text_traced(
        draw,
        (float(label_x), float(label_y)),
        str(label),
        font=font,
        fill=(146, 22, 28),
        stroke_width=3,
        stroke_fill=(255, 255, 255),
        role=str(role),
        required=True,
    )
    return _bbox_union(ring_bbox, text_record["bbox_px"])


def render_landmark_scene(
    *,
    dataset: Mapping[str, Any],
    render_params: Any,
    panel_params: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Render the two-view landmark panel and collect marker bboxes without changing the underlying 3D correspondence."""
    layout = panel_layout(render_params)
    composite, rendered_by_view, background_meta = render_two_view_object_scene(
        dataset=dataset,
        render_params=render_params,
        instance_seed=int(instance_seed),
        params=params,
        background_defaults=background_defaults,
        panel_params=panel_params,
        view_dataset_builder=render_landmark_view_dataset,
        render_view_options=_landmark_render_options,
    )
    draw = ImageDraw.Draw(composite)
    marker_bboxes: Dict[str, Any] = {"candidate_landmark_marker_bboxes_px_by_label": {}}

    reference_panel = layout[REFERENCE_VIEW_KEY]
    candidate_panel = layout[CANDIDATE_VIEW_KEY]
    reference_point = offset_point(
        dataset["views"][REFERENCE_VIEW_KEY]["reference_landmark_point_px"],
        dx=float(reference_panel["x"]),
        dy=float(reference_panel["y"]),
    )
    marker_bboxes["reference_landmark_marker_bbox_px"] = _draw_landmark_marker(
        draw,
        point=reference_point,
        label="REF",
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        role="reference_landmark_label",
    )
    candidate_points = offset_point_map(
        dataset["views"][CANDIDATE_VIEW_KEY]["candidate_landmark_points_px_by_label"],
        dx=float(candidate_panel["x"]),
        dy=float(candidate_panel["y"]),
    )
    for label, point in sorted(candidate_points.items(), key=lambda item: str(item[0])):
        marker_bboxes["candidate_landmark_marker_bboxes_px_by_label"][str(label)] = _draw_landmark_marker(
            draw,
            point=point,
            label=str(label),
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            role="candidate_landmark_label",
        )

    return composite, dict(rendered_by_view), dict(background_meta), dict(marker_bboxes)


__all__ = [
    "LANDMARK_MARKER_RADIUS_PX",
    "render_landmark_scene",
    "render_landmark_view_dataset",
]
