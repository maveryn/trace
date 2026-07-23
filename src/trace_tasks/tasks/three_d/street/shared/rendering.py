"""Final scene rendering orchestration for street-intersection 3D scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from ...shared.camera_projection import project_xy as _project_xy
from ...shared.canvas import (
    bbox_dict_transform,
    bbox_transform,
    entities_transform,
    point_dict_transform,
    resize_image_to_fit_pixel_cap,
)
from ...shared.object_rendering import (
    ThreeDObjectSpec,
    ThreeDRenderContext,
    rendered_three_d_object_from_bbox,
    render_three_d_object,
)
from ...shared.object_scene_rendering import _bbox_union, _draw_option_label
from ...shared.option_panel import append_text_option_panel, empty_option_panel_metadata
from ...shared.street_object_rendering_common import (
    STREET_BUILDING_CONTEXT_OBJECT_TYPES,
    _draw_shadow,
    _street_object_fill_rgb,
)
from .objects import _draw_styled_building_object
from .components import _draw_street_shell
from .state import _StreetRenderParams, _camera_from_dataset, _frame_from_dataset


@dataclass(frozen=True)
class _RenderedStreetScene:
    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    street_bbox_px: List[float]
    object_bboxes_px: Dict[str, List[float]]
    object_centers_px: Dict[str, List[float]]
    candidate_bboxes_px: Dict[str, List[float]]
    candidate_centers_px: Dict[str, List[float]]
    context_object_bboxes_px: Dict[str, List[float]]
    context_object_centers_px: Dict[str, List[float]]
    annotation_bboxes: List[List[float]]
    annotation_entity_ids: List[str]
    option_panel_bbox_px: List[float]
    option_choice_bboxes_px: Dict[str, List[float]]
    option_choices: List[Dict[str, Any]]
    option_panel_height_px: int


def render_street_intersection_scene_3d(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: _StreetRenderParams,
    option_choices: Sequence[Mapping[str, Any]] = (),
) -> _RenderedStreetScene:
    """Render street surface, objects, labels, options, and annotation bboxes."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    camera = _camera_from_dataset(dataset)
    frame = _frame_from_dataset(dataset)
    scene_variant = str(dataset["scene_variant"])
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    street_bbox, entities = _draw_street_shell(
        draw,
        camera=camera,
        frame=frame,
        render_params=render_params,
        scene_variant=str(scene_variant),
        intersection_center_xy=dataset["intersection_center_xy"],
        intersection_layout=str(dataset["intersection_layout"]),
    )
    candidate_specs = [dict(spec) for spec in dataset["candidate_object_specs"]]
    reference_specs = [
        dict(spec)
        for spec in dataset.get("reference_object_specs", [])
        if isinstance(spec, Mapping)
    ]
    context_specs = [dict(spec) for spec in dataset["context_object_specs"]]
    all_specs = [*candidate_specs, *reference_specs, *context_specs]

    for spec in sorted(all_specs, key=lambda item: float(item["camera_distance"]), reverse=True):
        _draw_shadow(draw, spec, camera=camera, frame=frame)
    shape_bboxes: Dict[str, List[float]] = {}
    rendered_objects: Dict[str, Any] = {}
    for spec in sorted(all_specs, key=lambda item: float(item["camera_distance"]), reverse=True):
        object_spec = ThreeDObjectSpec.from_mapping(
            spec,
            object_type_key="object_type",
            default_renderer_id="street_object",
            role=str(spec.get("object_role", "street_object")),
            source_entity_type=(
                "three_d_street_candidate_object"
                if bool(spec.get("is_answer_candidate", False))
                else "three_d_street_context_object"
            ),
        )
        render_context = ThreeDRenderContext(
            draw=draw,
            camera=camera,
            frame=frame,
            render_params=render_params,
            fill_rgb=_street_object_fill_rgb(spec),
            scene_variant=str(scene_variant),
        )
        if str(spec.get("object_type")) in STREET_BUILDING_CONTEXT_OBJECT_TYPES:
            bbox = _draw_styled_building_object(
                draw,
                spec,
                camera=camera,
                frame=frame,
                fill=_street_object_fill_rgb(spec),
            )
            rendered = rendered_three_d_object_from_bbox(object_spec, render_context, bbox_xyxy=bbox)
        else:
            rendered = render_three_d_object(object_spec, render_context)
        shape_bboxes[str(spec["object_id"])] = list(rendered.bbox_xyxy)
        rendered_objects[str(spec["object_id"])] = rendered

    reference_marker_bboxes: Dict[str, List[float]] = {}
    reference_direction_marker_bboxes: Dict[str, List[float]] = {}
    for spec in reference_specs:
        object_id = str(spec["object_id"])
        if object_id not in shape_bboxes:
            continue
        x0, y0, x1, y1 = (float(value) for value in shape_bboxes[object_id])
        pad = 8.0
        marker_bbox = [
            round(x0 - pad, 3),
            round(y0 - pad, 3),
            round(x1 + pad, 3),
            round(y1 + pad, 3),
        ]
        for offset, color in ((2.0, (255, 255, 255)), (0.0, (216, 44, 44))):
            draw.rectangle(
                (
                    marker_bbox[0] - offset,
                    marker_bbox[1] - offset,
                    marker_bbox[2] + offset,
                    marker_bbox[3] + offset,
                ),
                outline=color,
                width=4,
            )
        reference_marker_bboxes[object_id] = list(marker_bbox)
        shape_bboxes[object_id] = _bbox_union(shape_bboxes[object_id], marker_bbox)
        direction = spec.get("travel_direction_vector_xy")
        if isinstance(direction, Sequence) and not isinstance(direction, (str, bytes)) and len(direction) >= 2:
            dx, dy = float(direction[0]), float(direction[1])
            norm = math.hypot(dx, dy)
            if norm > 0.001:
                dx /= norm
                dy /= norm
                x, y, _base_z = (float(value) for value in spec["base_xyz"])
                _width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
                start = _project_xy((x, y, height + 0.18), camera, frame)
                end = _project_xy((x + dx * 0.62, y + dy * 0.62, height + 0.18), camera, frame)
                draw.line([start, end], fill=(255, 255, 255), width=8)
                draw.line([start, end], fill=(216, 44, 44), width=5)
                vx = float(end[0]) - float(start[0])
                vy = float(end[1]) - float(start[1])
                vnorm = math.hypot(vx, vy)
                if vnorm > 0.001:
                    ux, uy = vx / vnorm, vy / vnorm
                    px, py = -uy, ux
                    head_len = 18.0
                    head_w = 12.0
                    head_points = [
                        (float(end[0]), float(end[1])),
                        (
                            float(end[0]) - ux * head_len + px * head_w * 0.5,
                            float(end[1]) - uy * head_len + py * head_w * 0.5,
                        ),
                        (
                            float(end[0]) - ux * head_len - px * head_w * 0.5,
                            float(end[1]) - uy * head_len - py * head_w * 0.5,
                        ),
                    ]
                    outline_points = [
                        (head_points[0][0] + ux * 1.5, head_points[0][1] + uy * 1.5),
                        (
                            head_points[1][0] - ux * 2.0 + px * 1.8,
                            head_points[1][1] - uy * 2.0 + py * 1.8,
                        ),
                        (
                            head_points[2][0] - ux * 2.0 - px * 1.8,
                            head_points[2][1] - uy * 2.0 - py * 1.8,
                        ),
                    ]
                    draw.polygon(outline_points, fill=(255, 255, 255))
                    draw.polygon(head_points, fill=(216, 44, 44))
                    arrow_bbox = [
                        round(min(start[0], end[0], *(point[0] for point in outline_points)) - 5.0, 3),
                        round(min(start[1], end[1], *(point[1] for point in outline_points)) - 5.0, 3),
                        round(max(start[0], end[0], *(point[0] for point in outline_points)) + 5.0, 3),
                        round(max(start[1], end[1], *(point[1] for point in outline_points)) + 5.0, 3),
                    ]
                    reference_direction_marker_bboxes[object_id] = list(arrow_bbox)
                    shape_bboxes[object_id] = _bbox_union(shape_bboxes[object_id], arrow_bbox)

    label_bboxes: Dict[str, List[float]] = {}
    if not option_choices:
        for spec in sorted(candidate_specs, key=lambda item: str(item["point_label"])):
            label = str(spec["point_label"])
            x, y = float(spec["screen_xy"][0]), float(spec["screen_xy"][1])
            label_bboxes[str(spec["object_id"])] = _draw_option_label(
                draw,
                label=str(label),
                center=(x, y),
                font=label_font,
            )

    object_bboxes: Dict[str, List[float]] = {}
    object_centers: Dict[str, List[float]] = {}
    candidate_bboxes: Dict[str, List[float]] = {}
    candidate_centers: Dict[str, List[float]] = {}
    context_bboxes: Dict[str, List[float]] = {}
    context_centers: Dict[str, List[float]] = {}
    for spec in all_specs:
        object_id = str(spec["object_id"])
        rendered_object = rendered_objects[object_id]
        bbox = list(shape_bboxes[object_id])
        if object_id in label_bboxes:
            bbox = _bbox_union(bbox, label_bboxes[object_id])
        center = [round(float(spec["screen_xy"][0]), 3), round(float(spec["screen_xy"][1]), 3)]
        object_bboxes[object_id] = list(bbox)
        object_centers[object_id] = list(center)
        if bool(spec.get("is_answer_candidate", False)):
            label = str(spec["point_label"])
            candidate_bboxes[label] = list(bbox)
            candidate_centers[label] = list(center)
        else:
            context_bboxes[object_id] = list(bbox)
            context_centers[object_id] = list(center)
        fill_rgb = _street_object_fill_rgb(spec)
        entities.append(
            {
                "entity_id": object_id,
                "entity_type": "three_d_street_candidate_object"
                if bool(spec.get("is_answer_candidate", False))
                else "three_d_street_context_object",
                "bbox_px": list(bbox),
                "attrs": {
                    "point_label": spec.get("point_label"),
                    "object_label": spec.get("object_label"),
                    "object_type": str(spec["object_type"]),
                    "object_name": str(spec["object_name"]),
                    "prompt_name": str(spec["prompt_name"]),
                    "building_style": spec.get("building_style"),
                    "building_style_name": spec.get("building_style_name"),
                    "object_role": str(spec["object_role"]),
                    "orientation_axis": str(spec.get("orientation_axis", "")),
                    "is_answer_candidate": bool(spec.get("is_answer_candidate", False)),
                    "fill_rgb": [int(channel) for channel in fill_rgb],
                    "world_xyz": list(spec["world_xyz"]),
                    "base_xyz": list(spec["base_xyz"]),
                    "dimensions_xyz": list(spec["dimensions_xyz"]),
                    "dimension_scale": float(spec.get("dimension_scale", 1.0)),
                    "screen_xy": list(center),
                    "camera_xyz": list(spec["camera_xyz"]),
                    "camera_distance": float(spec["camera_distance"]),
                    "ground_distance_to_intersection": float(spec["ground_distance_to_intersection"]),
                    "intersection_center_xy": list(spec["intersection_center_xy"]),
                    "road_arm": spec.get("road_arm"),
                    "reference_marker": "red_bbox" if object_id in reference_marker_bboxes else None,
                    "reference_marker_bbox_px": reference_marker_bboxes.get(object_id),
                    "reference_direction_marker_bbox_px": reference_direction_marker_bboxes.get(object_id),
                    "travel_direction_vector_xy": spec.get("travel_direction_vector_xy"),
                    "scene_variant": str(scene_variant),
                    "object_record": dict(rendered_object.object_record),
                },
            }
        )
    annotation_ids = [str(value) for value in dataset["target_object_ids"]]
    annotation_bboxes = [list(object_bboxes[object_id]) for object_id in annotation_ids]
    all_bboxes = [list(street_bbox), *[list(value) for value in object_bboxes.values()]]
    scene_bbox = [
        round(float(min(bbox[0] for bbox in all_bboxes)), 3),
        round(float(min(bbox[1] for bbox in all_bboxes)), 3),
        round(float(max(bbox[2] for bbox in all_bboxes)), 3),
        round(float(max(bbox[3] for bbox in all_bboxes)), 3),
    ]
    option_metadata = empty_option_panel_metadata()
    if option_choices:
        image, option_metadata, option_entities = append_text_option_panel(
            image,
            option_choices=option_choices,
            font_size_px=int(render_params.label_font_size_px),
            text_rgb=render_params.text_rgb,
            stroke_rgb=render_params.text_stroke_rgb,
        )
        entities.extend(option_entities)
    image, image_scale = resize_image_to_fit_pixel_cap(image)
    if image_scale.changed:
        scale_x = float(image_scale.scale_x)
        scale_y = float(image_scale.scale_y)
        scene_bbox = bbox_transform(scene_bbox, scale_x=scale_x, scale_y=scale_y)
        street_bbox = bbox_transform(street_bbox, scale_x=scale_x, scale_y=scale_y)
        object_bboxes = bbox_dict_transform(object_bboxes, scale_x=scale_x, scale_y=scale_y)
        object_centers = point_dict_transform(object_centers, scale_x=scale_x, scale_y=scale_y)
        candidate_bboxes = bbox_dict_transform(candidate_bboxes, scale_x=scale_x, scale_y=scale_y)
        candidate_centers = point_dict_transform(candidate_centers, scale_x=scale_x, scale_y=scale_y)
        context_bboxes = bbox_dict_transform(context_bboxes, scale_x=scale_x, scale_y=scale_y)
        context_centers = point_dict_transform(context_centers, scale_x=scale_x, scale_y=scale_y)
        annotation_bboxes = [bbox_transform(bbox, scale_x=scale_x, scale_y=scale_y) for bbox in annotation_bboxes]
        option_metadata = dict(option_metadata)
        if option_metadata.get("option_panel_bbox_px"):
            option_metadata["option_panel_bbox_px"] = bbox_transform(
                option_metadata["option_panel_bbox_px"],
                scale_x=scale_x,
                scale_y=scale_y,
            )
        option_metadata["option_choice_bboxes_px"] = bbox_dict_transform(
            option_metadata.get("option_choice_bboxes_px", {}),
            scale_x=scale_x,
            scale_y=scale_y,
        )
        option_metadata["option_panel_height_px"] = int(round(float(option_metadata.get("option_panel_height_px", 0)) * scale_y))
        entities = entities_transform(entities, scale_x=scale_x, scale_y=scale_y)
    return _RenderedStreetScene(
        image=image,
        entities=list(entities),
        scene_bbox_px=list(scene_bbox),
        street_bbox_px=list(street_bbox),
        object_bboxes_px=dict(object_bboxes),
        object_centers_px=dict(object_centers),
        candidate_bboxes_px=dict(candidate_bboxes),
        candidate_centers_px=dict(candidate_centers),
        context_object_bboxes_px=dict(context_bboxes),
        context_object_centers_px=dict(context_centers),
        annotation_bboxes=[list(bbox) for bbox in annotation_bboxes],
        annotation_entity_ids=list(annotation_ids),
        option_panel_bbox_px=list(option_metadata["option_panel_bbox_px"]),
        option_choice_bboxes_px={str(key): list(value) for key, value in option_metadata["option_choice_bboxes_px"].items()},
        option_choices=[dict(choice) for choice in option_metadata["option_choices"]],
        option_panel_height_px=int(option_metadata["option_panel_height_px"]),
    )


__all__ = ["render_street_intersection_scene_3d"]
