"""Rendering helpers for synthetic 3D conveyor carousel scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.three_d.shared.camera_projection import (
    CameraSpec,
    ProjectionFrame,
    project_xy,
)
from trace_tasks.tasks.three_d.shared.object_rendering import (
    ThreeDObjectSpec,
    ThreeDRenderContext,
    render_three_d_object,
)
from trace_tasks.tasks.three_d.shared.object_scene_rendering import _bbox_union, _draw_line
from trace_tasks.tasks.shared.text_legibility import draw_text_traced, text_legibility_metadata_for_surfaces
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import BELT_GEOMETRY, BELT_KEYS, BELT_LABELS


FALLBACK_FLOOR_RGB = (244, 247, 249)
FALLBACK_BELT_ARROW_RGB = (128, 142, 156)


@dataclass(frozen=True)
class RenderedConveyor:
    """Rendered conveyor carousel scene with projected object geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    conveyor_bbox_px: List[float]
    object_bboxes_px: Dict[str, List[float]]
    object_centers_px: Dict[str, List[float]]
    target_object_bboxes_px: Dict[str, List[float]]
    target_object_centers_px: Dict[str, List[float]]
    belt_bboxes_px: Dict[str, List[float]]


def _camera_from_dataset(dataset: Mapping[str, Any]) -> CameraSpec:
    raw = dict(dataset["camera"])
    return CameraSpec(
        camera_position=tuple(float(value) for value in raw["camera_position"]),
        target=tuple(float(value) for value in raw["target"]),
        right=tuple(float(value) for value in raw["right"]),
        up=tuple(float(value) for value in raw["up"]),
        forward=tuple(float(value) for value in raw["forward"]),
        yaw_degrees=float(raw["yaw_degrees"]),
        pitch_degrees=float(raw["pitch_degrees"]),
        distance=float(raw["distance"]),
    )


def _frame_from_dataset(dataset: Mapping[str, Any]) -> ProjectionFrame:
    raw = dict(dataset["projection_frame"])
    return ProjectionFrame(
        scale=float(raw["scale"]),
        center_x=float(raw["center_x"]),
        center_y=float(raw["center_y"]),
        normalized_center_u=float(raw["normalized_center_u"]),
        normalized_center_v=float(raw["normalized_center_v"]),
    )


def _projected_bbox(points: Sequence[Sequence[float]]) -> List[float]:
    return [
        round(float(min(point[0] for point in points)), 3),
        round(float(min(point[1] for point in points)), 3),
        round(float(max(point[0] for point in points)), 3),
        round(float(max(point[1] for point in points)), 3),
    ]


def _ellipse_points(
    *,
    belt_key: str,
    radial_offset: float,
    z: float,
    steps: int = 96,
) -> list[tuple[float, float, float]]:
    geometry = BELT_GEOMETRY[str(belt_key)]
    radius_x = float(geometry["radius_x"]) + float(radial_offset)
    radius_y = float(geometry["radius_y"]) + float(radial_offset) * 0.62
    return [
        (
            float(radius_x * math.cos(2.0 * math.pi * float(index) / float(steps))),
            float(radius_y * math.sin(2.0 * math.pi * float(index) / float(steps))),
            float(z),
        )
        for index in range(int(steps))
    ]


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    fill: Tuple[int, int, int],
) -> None:
    sx, sy = float(start_xy[0]), float(start_xy[1])
    ex, ey = float(end_xy[0]), float(end_xy[1])
    _draw_line(draw, (sx, sy), (ex, ey), fill=fill, width=3)
    angle = math.atan2(ey - sy, ex - sx)
    length = 11.0
    spread = 0.58
    left = (
        ex - length * math.cos(angle - spread),
        ey - length * math.sin(angle - spread),
    )
    right = (
        ex - length * math.cos(angle + spread),
        ey - length * math.sin(angle + spread),
    )
    draw.polygon([(ex, ey), left, right], fill=fill)


def _draw_belt(
    draw: ImageDraw.ImageDraw,
    *,
    belt_key: str,
    camera: CameraSpec,
    frame: ProjectionFrame,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    arrow_rgb: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> tuple[List[float], Dict[str, Any]]:
    """Draw one annular ellipse belt from shared belt geometry.

    The belt bbox and rendered surface come from the same projected
    centerline/band-width geometry used by the sampler, so inner/outer belt
    scope stays visually bound to metadata without drawing text labels.
    """

    width = float(BELT_GEOMETRY[str(belt_key)]["band_width"])
    outer_world = _ellipse_points(belt_key=str(belt_key), radial_offset=0.5 * width, z=0.035)
    inner_world = _ellipse_points(belt_key=str(belt_key), radial_offset=-0.5 * width, z=0.04)
    outer_screen = [project_xy(point, camera, frame) for point in outer_world]
    inner_screen = [project_xy(point, camera, frame) for point in inner_world]
    draw.polygon(outer_screen, fill=fill)
    draw.polygon(inner_screen, fill=floor_rgb)
    draw.line([*outer_screen, outer_screen[0]], fill=outline, width=3)
    draw.line([*inner_screen, inner_screen[0]], fill=outline, width=3)
    for theta in (0.42 * math.pi, 1.06 * math.pi, 1.64 * math.pi):
        radius_x = float(BELT_GEOMETRY[str(belt_key)]["radius_x"])
        radius_y = float(BELT_GEOMETRY[str(belt_key)]["radius_y"])
        start = (radius_x * math.cos(theta - 0.055), radius_y * math.sin(theta - 0.055), 0.065)
        end = (radius_x * math.cos(theta + 0.055), radius_y * math.sin(theta + 0.055), 0.065)
        _draw_arrow(draw, start_xy=project_xy(start, camera, frame), end_xy=project_xy(end, camera, frame), fill=arrow_rgb)
    bbox = _projected_bbox([*outer_screen, *inner_screen])
    entity = {
        "entity_id": f"belt_{belt_key}",
        "entity_type": "three_d_conveyor_belt",
        "bbox_px": list(bbox),
        "attrs": {
            "belt_key": str(belt_key),
            "belt_label": str(BELT_LABELS[str(belt_key)]),
            "geometry": dict(BELT_GEOMETRY[str(belt_key)]),
        },
    }
    return list(bbox), entity


def _draw_conveyor_belts(
    image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    camera: CameraSpec,
    frame: ProjectionFrame,
    render_params: Any,
) -> tuple[Image.Image, List[float], Dict[str, List[float]], List[Dict[str, Any]]]:
    """Draw the full two-belt carousel surface.

    Invariant: both belt entities are generated from shared belt geometry in
    fixed outer-then-inner draw order, so object scope, visual belt shape, and
    trace bboxes agree even though no text labels are drawn on the belts.
    """

    draw = ImageDraw.Draw(image)
    belt_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    floor_rgb = tuple(int(value) for value in getattr(render_params, "floor_rgb", FALLBACK_FLOOR_RGB))
    fills = {
        "outer": tuple(int(value) for value in (getattr(render_params, "conveyor_belt_fill_rgb", None) or (198, 214, 226))),
        "inner": tuple(int(value) for value in (getattr(render_params, "conveyor_belt_fill_alt_rgb", None) or (211, 224, 234))),
    }
    outlines = {
        "outer": tuple(int(value) for value in (getattr(render_params, "conveyor_belt_outline_rgb", None) or (82, 97, 115))),
        "inner": tuple(int(value) for value in (getattr(render_params, "conveyor_belt_outline_secondary_rgb", None) or (92, 108, 126))),
    }
    arrow_rgb = tuple(int(value) for value in (getattr(render_params, "conveyor_belt_arrow_rgb", None) or FALLBACK_BELT_ARROW_RGB))
    for belt_key in ("outer", "inner"):
        bbox, entity = _draw_belt(
            draw,
            belt_key=str(belt_key),
            camera=camera,
            frame=frame,
            fill=fills[str(belt_key)],
            outline=outlines[str(belt_key)],
            arrow_rgb=arrow_rgb,
            floor_rgb=floor_rgb,
        )
        belt_bboxes[str(belt_key)] = list(bbox)
        entities.append(dict(entity))
    conveyor_bbox = _bbox_union(*belt_bboxes.values())
    return image, list(conveyor_bbox), belt_bboxes, entities


def _draw_anchor_markers(
    image: Image.Image,
    *,
    object_bboxes: Mapping[str, Sequence[float]],
    entities: List[Dict[str, Any]],
    dataset: Mapping[str, Any],
) -> None:
    """Draw red A/B reference boxes around marked anchor objects."""

    marker_records = [dict(record) for record in dataset.get("marked_anchor_records", [])]
    if not marker_records:
        return
    draw = ImageDraw.Draw(image)
    font = load_font(26, bold=True)
    outline = (214, 28, 48)
    badge_fill = (23, 28, 36)
    badge_text = (255, 255, 255)
    image_w, image_h = int(image.width), int(image.height)
    for record in marker_records:
        object_id = str(record.get("object_id", ""))
        if object_id not in object_bboxes:
            continue
        x0, y0, x1, y1 = (float(value) for value in object_bboxes[str(object_id)])
        pad = max(5.0, 0.08 * max(float(x1 - x0), float(y1 - y0)))
        box = [
            max(2.0, x0 - pad),
            max(2.0, y0 - pad),
            min(float(image_w - 2), x1 + pad),
            min(float(image_h - 2), y1 + pad),
        ]
        draw.rectangle(tuple(box), outline=outline, width=4)
        label = str(record.get("anchor_label", ""))
        if label:
            text_bbox = draw.textbbox((0, 0), label, font=font, stroke_width=1)
            text_w = float(text_bbox[2] - text_bbox[0])
            text_h = float(text_bbox[3] - text_bbox[1])
            badge_w = max(28.0, text_w + 14.0)
            badge_h = max(28.0, text_h + 10.0)
            bx0 = min(max(2.0, box[0]), float(image_w) - badge_w - 2.0)
            by0 = box[1] - badge_h - 4.0
            if by0 < 2.0:
                by0 = min(box[3] + 4.0, float(image_h) - badge_h - 2.0)
            badge = [bx0, by0, bx0 + badge_w, by0 + badge_h]
            draw.rounded_rectangle(tuple(badge), radius=6, fill=badge_fill, outline=outline, width=2)
            draw_text_traced(
                draw,
                (badge[0] + badge_w * 0.5, badge[1] + badge_h * 0.5),
                label,
                font=font,
                fill=badge_text,
                stroke_width=0,
                stroke_fill=badge_fill,
                anchor="mm",
                role="three_d_carousel_anchor_label",
                required=True,
                extra_metadata=text_legibility_metadata_for_surfaces(
                    fill_rgb=badge_text,
                    surface_rgbs=(badge_fill,),
                ),
            )
        entities.append(
            {
                "entity_id": f"anchor_marker_{object_id}",
                "entity_type": "three_d_carousel_anchor_marker",
                "bbox_px": [round(float(value), 3) for value in box],
                "attrs": {
                    "object_id": str(object_id),
                    "anchor_label": str(record.get("anchor_label", "")),
                    "anchor_role": str(record.get("anchor_role", "")),
                },
            }
        )


def render_conveyor(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: Any,
) -> RenderedConveyor:
    """Render one conveyor carousel scene and project object boxes."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    floor_rgb = tuple(int(value) for value in getattr(render_params, "floor_rgb", FALLBACK_FLOOR_RGB))
    draw.rectangle((0, 0, int(image.width), int(image.height)), fill=floor_rgb)
    camera = _camera_from_dataset(dataset)
    frame = _frame_from_dataset(dataset)
    image, conveyor_bbox, belt_bboxes, entities = _draw_conveyor_belts(
        image,
        dataset=dataset,
        camera=camera,
        frame=frame,
        render_params=render_params,
    )
    draw = ImageDraw.Draw(image)
    object_specs = [dict(spec) for spec in dataset["object_specs"]]
    ordered_specs = sorted(
        object_specs,
        key=lambda item: float(item.get("camera_distance", 0.0)) + float(item.get("render_order_bias", 0.0)),
        reverse=True,
    )
    object_bboxes: Dict[str, List[float]] = {}
    object_centers: Dict[str, List[float]] = {}
    target_bboxes: Dict[str, List[float]] = {}
    target_centers: Dict[str, List[float]] = {}
    for spec in ordered_specs:
        fill = tuple(int(channel) for channel in spec["fill_rgb"])
        rendered = render_three_d_object(
            ThreeDObjectSpec.from_mapping(spec, object_type_key="shape_type", default_renderer_id="object_scene_shape"),
            ThreeDRenderContext(
                draw=draw,
                camera=camera,
                frame=frame,
                render_params=render_params,
                fill_rgb=fill,
                scene_variant=str(dataset["scene_variant"]),
                floor_rgb=floor_rgb,
            ),
        )
        object_id = str(spec["object_id"])
        bbox = [round(float(value), 3) for value in rendered.bbox_xyxy]
        center = [round(float(spec["screen_xy"][0]), 3), round(float(spec["screen_xy"][1]), 3)]
        object_bboxes[object_id] = list(bbox)
        object_centers[object_id] = list(center)
        entities.append(
            {
                "entity_id": object_id,
                "entity_type": "three_d_conveyor_object",
                "bbox_px": list(bbox),
                "attrs": {
                    "shape_type": str(spec["shape_type"]),
                    "object_name": str(spec["object_name"]),
                    "color_name": str(spec["color_name"]),
                    "belt_key": str(spec["belt_key"]),
                    "belt_label": str(spec["belt_label"]),
                    "matches_query": bool(spec.get("matches_query", False)),
                },
            }
        )
        if str(object_id) in set(str(value) for value in dataset["target_object_ids"]):
            target_bboxes[str(object_id)] = list(bbox)
            target_centers[str(object_id)] = list(center)
    _draw_anchor_markers(
        image,
        object_bboxes=object_bboxes,
        entities=entities,
        dataset=dataset,
    )
    scene_bbox = _bbox_union(conveyor_bbox, *object_bboxes.values()) if object_bboxes else conveyor_bbox
    return RenderedConveyor(
        image=image,
        entities=entities,
        scene_bbox_px=list(scene_bbox),
        conveyor_bbox_px=list(conveyor_bbox),
        object_bboxes_px=dict(object_bboxes),
        object_centers_px=dict(object_centers),
        target_object_bboxes_px=dict(target_bboxes),
        target_object_centers_px=dict(target_centers),
        belt_bboxes_px=dict(belt_bboxes),
    )


__all__ = ["RenderedConveyor", "render_conveyor"]
