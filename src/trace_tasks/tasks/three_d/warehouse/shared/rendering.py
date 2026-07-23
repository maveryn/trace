"""Warehouse scene rendering helpers for robot option-label tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
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
from ...shared.object_resources import (
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB,
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE,
)
from ...shared.object_scene import (
    _CameraSpec,
    _ProjectionFrame,
    _canvas_floor_polygon_xy,
    _grid_values_for_range,
    _object_screen_bbox,
    _polygon_axis_line_segment,
    _project_xy,
)
from ...shared.object_scene_rendering import _bbox_union, _draw_line, _draw_option_label
from ...shared.option_panel import append_text_option_panel, empty_option_panel_metadata
from ...shared.warehouse_object_rendering import _draw_ground_shadow, _fill_for_object
from .state import (
    _WarehouseRenderParams,
    _camera_from_dataset,
    _frame_from_dataset,
    _projected_bbox,
    _scene_palette,
)
from .components import _draw_shelf_rack_object


@dataclass(frozen=True)
class _RenderedWarehouseScene:
    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    warehouse_bbox_px: List[float]
    object_bboxes_px: Dict[str, List[float]]
    object_centers_px: Dict[str, List[float]]
    candidate_bboxes_px: Dict[str, List[float]]
    candidate_centers_px: Dict[str, List[float]]
    context_object_bboxes_px: Dict[str, List[float]]
    context_object_centers_px: Dict[str, List[float]]
    reference_object_bboxes_px: Dict[str, List[float]]
    reference_object_centers_px: Dict[str, List[float]]
    annotation_bboxes: List[List[float]]
    annotation_entity_ids: List[str]
    option_panel_bbox_px: List[float]
    option_choice_bboxes_px: Dict[str, List[float]]
    option_choices: List[Dict[str, Any]]
    option_panel_height_px: int


def _alpha_polygon(
    image: Image.Image,
    points: Sequence[Sequence[float]],
    *,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon([(float(x), float(y)) for x, y in points], fill=fill)
    if outline is not None and points:
        overlay_draw.line(
            [(float(x), float(y)) for x, y in points] + [(float(points[0][0]), float(points[0][1]))],
            fill=outline,
            width=max(1, int(width)),
        )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _draw_warehouse_floor(
    image: Image.Image,
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    render_params: _WarehouseRenderParams,
    scene_variant: str,
    dataset: Mapping[str, Any],
    include_path: bool,
) -> Tuple[Image.Image, List[float], List[Dict[str, Any]]]:
    """Draw the warehouse floor, grid, aisles, and path context."""
    draw = ImageDraw.Draw(image)
    floor_rgb, grid_rgb, aisle_rgb, shelf_zone_rgb = _scene_palette(str(scene_variant), render_params)
    draw.rectangle((0, 0, int(render_params.canvas_width), int(render_params.canvas_height)), fill=floor_rgb)
    floor_polygon_xy = _canvas_floor_polygon_xy(camera=camera, frame=frame, render_params=render_params)
    grid_world_bbox = None
    if floor_polygon_xy:
        min_x = min(float(point[0]) for point in floor_polygon_xy)
        max_x = max(float(point[0]) for point in floor_polygon_xy)
        min_y = min(float(point[1]) for point in floor_polygon_xy)
        max_y = max(float(point[1]) for point in floor_polygon_xy)
        grid_world_bbox = [round(min_x, 4), round(min_y, 4), round(max_x, 4), round(max_y, 4)]
        for value in _grid_values_for_range(min_y, max_y, float(render_params.grid_step)):
            segment = _polygon_axis_line_segment(floor_polygon_xy, axis="y", value=float(value))
            if segment is None:
                continue
            _draw_line(
                draw,
                _project_xy((segment[0][0], segment[0][1], 0.0), camera, frame),
                _project_xy((segment[1][0], segment[1][1], 0.0), camera, frame),
                fill=grid_rgb,
                width=render_params.line_width_px,
            )
        for value in _grid_values_for_range(min_x, max_x, float(render_params.grid_step)):
            segment = _polygon_axis_line_segment(floor_polygon_xy, axis="x", value=float(value))
            if segment is None:
                continue
            _draw_line(
                draw,
                _project_xy((segment[0][0], segment[0][1], 0.0), camera, frame),
                _project_xy((segment[1][0], segment[1][1], 0.0), camera, frame),
                fill=grid_rgb,
                width=render_params.line_width_px,
            )
    for polygon_world, color in (
        (dataset["shelf_zone_polygons_world"][0], shelf_zone_rgb),
        (dataset["shelf_zone_polygons_world"][1], shelf_zone_rgb),
        (dataset["main_aisle_polygon_world"], aisle_rgb),
    ):
        polygon_screen = [_project_xy(point, camera, frame) for point in polygon_world]
        draw.polygon(polygon_screen, fill=color)
    stage_bbox = [0.0, 0.0, float(render_params.canvas_width), float(render_params.canvas_height)]
    entities = [
        {
            "entity_id": "warehouse_floor",
            "entity_type": "three_d_warehouse_floor",
            "bbox_px": list(stage_bbox),
            "attrs": {
                "scene_variant": str(scene_variant),
                "full_bleed_floor": True,
                "grid_mode": "screen_ray_floor_plane",
                "grid_world_bbox": list(grid_world_bbox) if grid_world_bbox is not None else None,
                "floor_rgb": list(floor_rgb),
            },
        }
    ]
    if include_path:
        path_polygon_screen = [_project_xy(point, camera, frame) for point in dataset["robot_path_corridor_polygon_world"]]
        image = _alpha_polygon(
            image,
            path_polygon_screen,
            fill=(*render_params.path_rgb, 50),
            outline=(*render_params.path_rgb, 120),
            width=max(1, int(render_params.line_width_px)),
        )
        entities.append(
            {
                "entity_id": "robot_forward_path_corridor",
                "entity_type": "three_d_warehouse_robot_path_corridor",
                "bbox_px": _projected_bbox(path_polygon_screen),
                "attrs": {
                    "robot_heading": str(dataset["robot_heading"]),
                    "corridor_half_width": float(dataset["path_corridor_half_width"]),
                    "path_polygon_world": [list(point) for point in dataset["robot_path_corridor_polygon_world"]],
                },
            }
        )
    return image, stage_bbox, entities


def _draw_reference_marker(
    draw: ImageDraw.ImageDraw,
    *,
    robot_spec: Mapping[str, Any],
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    dataset: Mapping[str, Any],
) -> Tuple[List[float], List[float]]:
    bbox = _object_screen_bbox(robot_spec, camera, frame, pad_px=9.0)
    draw.rectangle(tuple(float(value) for value in bbox), outline=(220, 34, 34), width=4)
    start_world = tuple(float(value) for value in dataset["robot_arrow_start_world"])
    end_world = tuple(float(value) for value in dataset["robot_arrow_end_world"])
    start = _project_xy(start_world, camera, frame)
    end = _project_xy(end_world, camera, frame)
    _draw_line(draw, start, end, fill=(220, 34, 34), width=5)
    vx, vy = float(end[0] - start[0]), float(end[1] - start[1])
    length = max(1.0, math.hypot(vx, vy))
    ux, uy = vx / length, vy / length
    px, py = -uy, ux
    arrow = [
        (end[0], end[1]),
        (end[0] - ux * 24 + px * 10, end[1] - uy * 24 + py * 10),
        (end[0] - ux * 24 - px * 10, end[1] - uy * 24 - py * 10),
    ]
    draw.polygon(arrow, fill=(220, 34, 34), outline=(112, 20, 20))
    marker_bbox = _bbox_union(bbox, [start[0], start[1], start[0], start[1]], _projected_bbox(arrow))
    return list(bbox), list(marker_bbox)


def render_warehouse_robot_scene_3d(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: _WarehouseRenderParams,
    option_choices: Sequence[Mapping[str, Any]] = (),
) -> _RenderedWarehouseScene:
    """Render the robot path scene and project candidate annotations."""
    image = background.convert("RGB")
    camera = _camera_from_dataset(dataset)
    frame = _frame_from_dataset(dataset)
    scene_variant = str(dataset["scene_variant"])
    image, warehouse_bbox, entities = _draw_warehouse_floor(
        image,
        camera=camera,
        frame=frame,
        render_params=render_params,
        scene_variant=scene_variant,
        dataset=dataset,
        include_path=True,
    )
    draw = ImageDraw.Draw(image)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    candidate_specs = [dict(spec) for spec in dataset["candidate_object_specs"]]
    context_specs = [dict(spec) for spec in dataset["context_object_specs"]]
    reference_specs = [dict(spec) for spec in dataset["reference_object_specs"]]
    all_specs = [*candidate_specs, *context_specs, *reference_specs]
    shelf_specs = [spec for spec in all_specs if str(spec.get("object_type")) == "shelf_rack"]
    non_shelf_specs = [spec for spec in all_specs if str(spec.get("object_type")) != "shelf_rack"]
    ordered_specs = [
        *sorted(shelf_specs, key=lambda item: float(item["camera_distance"]), reverse=True),
        *sorted(non_shelf_specs, key=lambda item: float(item["camera_distance"]), reverse=True),
    ]
    for spec in ordered_specs:
        _draw_ground_shadow(draw, spec, camera=camera, frame=frame)
    point_bboxes: Dict[str, List[float]] = {}
    point_centers: Dict[str, List[float]] = {}
    object_bboxes: Dict[str, List[float]] = {}
    object_centers: Dict[str, List[float]] = {}
    context_bboxes: Dict[str, List[float]] = {}
    context_centers: Dict[str, List[float]] = {}
    reference_bboxes: Dict[str, List[float]] = {}
    reference_centers: Dict[str, List[float]] = {}

    for spec in ordered_specs:
        fill = _fill_for_object(spec, scene_variant=scene_variant)
        object_spec = ThreeDObjectSpec.from_mapping(
            spec,
            object_type_key="object_type",
            default_renderer_id="warehouse_object",
            role=str(spec.get("object_role", "warehouse_object")),
            source_entity_type=(
                "three_d_warehouse_candidate_object"
                if bool(spec.get("is_answer_candidate", False))
                else "three_d_warehouse_reference_robot"
                if str(spec.get("object_role")) == "warehouse_reference_robot"
                else "three_d_warehouse_context_object"
            ),
        )
        render_context = ThreeDRenderContext(
            draw=draw,
            camera=camera,
            frame=frame,
            render_params=render_params,
            fill_rgb=fill,
            scene_variant=str(scene_variant),
        )
        if str(spec.get("object_type")) == "shelf_rack":
            bbox = _draw_shelf_rack_object(draw, spec, camera=camera, frame=frame, fill=fill)
            rendered_object = rendered_three_d_object_from_bbox(object_spec, render_context, bbox_xyxy=bbox)
        else:
            rendered_object = render_three_d_object(object_spec, render_context)
        bbox = list(rendered_object.bbox_xyxy)
        label = str(spec.get("point_label", ""))
        center = [round(float(spec["screen_xy"][0]), 3), round(float(spec["screen_xy"][1]), 3)]
        if bool(spec.get("is_answer_candidate", False)):
            if not option_choices:
                label_bbox = _draw_option_label(draw, label=label, center=(float(center[0]), float(center[1])), font=label_font)
                bbox = _bbox_union(bbox, label_bbox)
            point_bboxes[str(label)] = list(bbox)
            point_centers[str(label)] = list(center)
        elif str(spec.get("object_role")) == "warehouse_reference_robot":
            reference_bboxes[str(spec["object_id"])] = list(bbox)
            reference_centers[str(spec["object_id"])] = list(center)
        else:
            context_bboxes[str(spec["object_id"])] = list(bbox)
            context_centers[str(spec["object_id"])] = list(center)
        object_bboxes[str(spec["object_id"])] = list(bbox)
        object_centers[str(spec["object_id"])] = list(center)
        entities.append(
            {
                "entity_id": str(spec["object_id"]),
                "entity_type": "three_d_warehouse_candidate_object" if bool(spec.get("is_answer_candidate", False)) else ("three_d_warehouse_reference_robot" if str(spec.get("object_role")) == "warehouse_reference_robot" else "three_d_warehouse_context_object"),
                "bbox_px": list(bbox),
                "attrs": {
                    "point_label": str(label) if label else None,
                    "object_label": str(label) if label else None,
                    "object_type": str(spec["object_type"]),
                    "object_name": str(spec.get("object_name", spec["object_type"])),
                    "object_role": str(spec["object_role"]),
                    "shelf_style": str(spec.get("shelf_style")) if str(spec.get("object_type")) == "shelf_rack" else None,
                    "shelf_levels": int(spec.get("shelf_levels", 0)) if str(spec.get("object_type")) == "shelf_rack" else None,
                    "shelf_load_count": len(spec.get("shelf_load_slots", ())) if str(spec.get("object_type")) == "shelf_rack" else None,
                    "shelf_frame_rgb": list(spec.get("shelf_frame_rgb", ())) if str(spec.get("object_type")) == "shelf_rack" else None,
                    "shelf_height_scale": float(spec.get("shelf_height_scale", 1.0)) if str(spec.get("object_type")) == "shelf_rack" else None,
                    "robot_design": str(spec.get("robot_design")) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "robot_base_rgb": list(spec.get("robot_base_rgb", ())) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "robot_accent_rgb": list(spec.get("robot_accent_rgb", ())) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "is_answer_candidate": bool(spec.get("is_answer_candidate", False)),
                    "is_in_forward_path_corridor": bool(spec.get("is_in_forward_path_corridor", False)),
                    "is_first_reached_object": bool(spec.get("is_first_reached_object", False)),
                    "forward_distance_from_robot": float(spec.get("forward_distance_from_robot", 0.0)),
                    "lateral_offset_from_robot": float(spec.get("lateral_offset_from_robot", 0.0)),
                    "fill_rgb": [int(channel) for channel in fill],
                    "world_xyz": list(spec["world_xyz"]),
                    "base_xyz": list(spec["base_xyz"]),
                    "dimensions_xyz": list(spec["dimensions_xyz"]),
                    "screen_xy": list(center),
                    "camera_distance": float(spec["camera_distance"]),
                    "object_record": dict(rendered_object.object_record),
                },
            }
        )

    robot_spec = reference_specs[0]
    robot_bbox, robot_marker_bbox = _draw_reference_marker(draw, robot_spec=robot_spec, camera=camera, frame=frame, dataset=dataset)
    entities.append(
        {
            "entity_id": "robot_reference_marker",
            "entity_type": "three_d_warehouse_robot_reference_marker",
            "bbox_px": list(robot_marker_bbox),
            "attrs": {
                "reference_object_id": str(robot_spec["object_id"]),
                "reference_marker": "red_bbox",
                "reference_marker_bbox_px": list(robot_bbox),
                "reference_direction_marker_bbox_px": list(robot_marker_bbox),
                "robot_heading": str(dataset["robot_heading"]),
                "travel_direction_vector_xy": list(dataset["travel_direction_vector_xy"]),
            },
        }
    )
    if not option_choices:
        for label, center in sorted(point_centers.items()):
            _draw_option_label(draw, label=str(label), center=(float(center[0]), float(center[1])), font=label_font)

    answer_label = str(dataset["answer_label"])
    annotation_bbox = list(point_bboxes[answer_label])
    scene_bboxes = [list(warehouse_bbox), list(robot_marker_bbox)] + [list(bbox) for bbox in object_bboxes.values()]
    scene_bbox = [
        round(float(min(bbox[0] for bbox in scene_bboxes)), 3),
        round(float(min(bbox[1] for bbox in scene_bboxes)), 3),
        round(float(max(bbox[2] for bbox in scene_bboxes)), 3),
        round(float(max(bbox[3] for bbox in scene_bboxes)), 3),
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
        warehouse_bbox = bbox_transform(warehouse_bbox, scale_x=scale_x, scale_y=scale_y)
        object_bboxes = bbox_dict_transform(object_bboxes, scale_x=scale_x, scale_y=scale_y)
        object_centers = point_dict_transform(object_centers, scale_x=scale_x, scale_y=scale_y)
        point_bboxes = bbox_dict_transform(point_bboxes, scale_x=scale_x, scale_y=scale_y)
        point_centers = point_dict_transform(point_centers, scale_x=scale_x, scale_y=scale_y)
        context_bboxes = bbox_dict_transform(context_bboxes, scale_x=scale_x, scale_y=scale_y)
        context_centers = point_dict_transform(context_centers, scale_x=scale_x, scale_y=scale_y)
        reference_bboxes = bbox_dict_transform(reference_bboxes, scale_x=scale_x, scale_y=scale_y)
        reference_centers = point_dict_transform(reference_centers, scale_x=scale_x, scale_y=scale_y)
        annotation_bbox = bbox_transform(annotation_bbox, scale_x=scale_x, scale_y=scale_y)
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
    return _RenderedWarehouseScene(
        image=image,
        entities=list(entities),
        scene_bbox_px=list(scene_bbox),
        warehouse_bbox_px=list(warehouse_bbox),
        object_bboxes_px=dict(object_bboxes),
        object_centers_px=dict(object_centers),
        candidate_bboxes_px=dict(point_bboxes),
        candidate_centers_px=dict(point_centers),
        context_object_bboxes_px=dict(context_bboxes),
        context_object_centers_px=dict(context_centers),
        reference_object_bboxes_px=dict(reference_bboxes),
        reference_object_centers_px=dict(reference_centers),
        annotation_bboxes=[list(annotation_bbox)],
        annotation_entity_ids=[str(dataset["answer_object_id"])],
        option_panel_bbox_px=list(option_metadata["option_panel_bbox_px"]),
        option_choice_bboxes_px={str(key): list(value) for key, value in option_metadata["option_choice_bboxes_px"].items()},
        option_choices=[dict(choice) for choice in option_metadata["option_choices"]],
        option_panel_height_px=int(option_metadata["option_panel_height_px"]),
    )


def render_warehouse_robot_nearest_scene_3d(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: _WarehouseRenderParams,
    option_choices: Sequence[Mapping[str, Any]] = (),
) -> _RenderedWarehouseScene:
    """Render nearest-reference candidates and project option annotations."""
    image = background.convert("RGB")
    camera = _camera_from_dataset(dataset)
    frame = _frame_from_dataset(dataset)
    scene_variant = str(dataset["scene_variant"])
    image, warehouse_bbox, entities = _draw_warehouse_floor(
        image,
        camera=camera,
        frame=frame,
        render_params=render_params,
        scene_variant=scene_variant,
        dataset=dataset,
        include_path=False,
    )
    draw = ImageDraw.Draw(image)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    candidate_specs = [dict(spec) for spec in dataset["candidate_specs"]]
    reference_specs = [dict(spec) for spec in dataset["reference_specs"]]
    context_specs = [dict(spec) for spec in dataset["context_object_specs"]]
    all_specs = [*candidate_specs, *reference_specs, *context_specs]
    shelf_specs = [spec for spec in all_specs if str(spec.get("object_type")) == "shelf_rack"]
    non_shelf_specs = [spec for spec in all_specs if str(spec.get("object_type")) != "shelf_rack"]
    ordered_specs = [
        *sorted(shelf_specs, key=lambda item: float(item["camera_distance"]), reverse=True),
        *sorted(non_shelf_specs, key=lambda item: float(item["camera_distance"]), reverse=True),
    ]
    for spec in ordered_specs:
        _draw_ground_shadow(draw, spec, camera=camera, frame=frame)

    point_bboxes: Dict[str, List[float]] = {}
    point_centers: Dict[str, List[float]] = {}
    object_bboxes: Dict[str, List[float]] = {}
    object_centers: Dict[str, List[float]] = {}
    context_bboxes: Dict[str, List[float]] = {}
    context_centers: Dict[str, List[float]] = {}
    reference_bboxes: Dict[str, List[float]] = {}
    reference_centers: Dict[str, List[float]] = {}

    for spec in ordered_specs:
        if str(spec.get("object_role")) == "warehouse_reference_object":
            fill = WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB if str(spec.get("object_type")) == WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE else _fill_for_object(spec, scene_variant=scene_variant)
        else:
            fill = _fill_for_object(spec, scene_variant=scene_variant)
        object_spec = ThreeDObjectSpec.from_mapping(
            spec,
            object_type_key="object_type",
            default_renderer_id="warehouse_object",
            role=str(spec.get("object_role", "warehouse_object")),
            source_entity_type=(
                "three_d_warehouse_robot_candidate"
                if bool(spec.get("is_answer_candidate", False)) and str(spec.get("object_type")) == "warehouse_robot"
                else "three_d_warehouse_candidate_object"
                if bool(spec.get("is_answer_candidate", False))
                else "three_d_warehouse_reference_object"
                if str(spec.get("object_role")) == "warehouse_reference_object"
                else "three_d_warehouse_reference_robot"
                if str(spec.get("object_role")) == "warehouse_reference_robot"
                else "three_d_warehouse_context_object"
            ),
        )
        render_context = ThreeDRenderContext(
            draw=draw,
            camera=camera,
            frame=frame,
            render_params=render_params,
            fill_rgb=fill,
            scene_variant=str(scene_variant),
        )
        if str(spec.get("object_type")) == "shelf_rack":
            rack_bbox = _draw_shelf_rack_object(draw, spec, camera=camera, frame=frame, fill=fill)
            rendered_object = rendered_three_d_object_from_bbox(object_spec, render_context, bbox_xyxy=rack_bbox)
        else:
            rendered_object = render_three_d_object(object_spec, render_context)
        bbox = list(rendered_object.bbox_xyxy)
        raw_label = spec.get("point_label")
        label = "" if raw_label is None else str(raw_label)
        center = [round(float(spec["screen_xy"][0]), 3), round(float(spec["screen_xy"][1]), 3)]
        if bool(spec.get("is_answer_candidate", False)):
            if not option_choices:
                label_bbox = _draw_option_label(draw, label=label, center=(float(center[0]), float(center[1])), font=label_font)
                bbox = _bbox_union(bbox, label_bbox)
            point_bboxes[str(label)] = list(bbox)
            point_centers[str(label)] = list(center)
        elif str(spec.get("object_role")) in {"warehouse_reference_object", "warehouse_reference_robot"}:
            reference_bboxes[str(spec["object_id"])] = list(bbox)
            reference_centers[str(spec["object_id"])] = list(center)
        else:
            context_bboxes[str(spec["object_id"])] = list(bbox)
            context_centers[str(spec["object_id"])] = list(center)
        object_bboxes[str(spec["object_id"])] = list(bbox)
        object_centers[str(spec["object_id"])] = list(center)
        entities.append(
            {
                "entity_id": str(spec["object_id"]),
                "entity_type": (
                    ("three_d_warehouse_robot_candidate" if str(spec.get("object_type")) == "warehouse_robot" else "three_d_warehouse_candidate_object")
                    if bool(spec.get("is_answer_candidate", False))
                    else (
                        "three_d_warehouse_reference_object"
                        if str(spec.get("object_role")) == "warehouse_reference_object"
                        else ("three_d_warehouse_reference_robot" if str(spec.get("object_role")) == "warehouse_reference_robot" else "three_d_warehouse_context_object")
                    )
                ),
                "bbox_px": list(bbox),
                "attrs": {
                    "point_label": str(label) if label else None,
                    "object_label": str(label) if label else None,
                    "object_type": str(spec["object_type"]),
                    "object_name": str(spec.get("object_name", spec["object_type"])),
                    "prompt_name": str(spec.get("prompt_name", spec.get("object_name", spec["object_type"]))),
                    "object_role": str(spec["object_role"]),
                    "robot_design": str(spec.get("robot_design")) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "robot_heading": str(spec.get("robot_heading")) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "robot_base_rgb": list(spec.get("robot_base_rgb", ())) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "robot_accent_rgb": list(spec.get("robot_accent_rgb", ())) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "gripper_tip_xyz": list(spec.get("gripper_tip_xyz", ())) if str(spec.get("object_type")) == "warehouse_robot" else None,
                    "shelf_style": str(spec.get("shelf_style")) if str(spec.get("object_type")) == "shelf_rack" else None,
                    "shelf_levels": int(spec.get("shelf_levels", 0)) if str(spec.get("object_type")) == "shelf_rack" else None,
                    "is_answer_candidate": bool(spec.get("is_answer_candidate", False)),
                    "is_nearest_robot_to_reference": bool(spec.get("is_nearest_robot_to_reference", False)),
                    "is_nearest_object_to_reference_robot": bool(spec.get("is_nearest_object_to_reference_robot", False)),
                    "distance_to_reference_object": float(spec.get("distance_to_reference_object", 0.0)),
                    "distance_to_reference_robot": float(spec.get("distance_to_reference_robot", 0.0)),
                    "fill_rgb": [int(channel) for channel in fill],
                    "world_xyz": list(spec["world_xyz"]),
                    "base_xyz": list(spec["base_xyz"]),
                    "dimensions_xyz": list(spec["dimensions_xyz"]),
                    "screen_xy": list(center),
                    "camera_distance": float(spec["camera_distance"]),
                    "object_record": dict(rendered_object.object_record),
                },
            }
        )

    if not option_choices:
        for label, center in sorted(point_centers.items()):
            _draw_option_label(draw, label=str(label), center=(float(center[0]), float(center[1])), font=label_font)

    answer_label = str(dataset["answer_label"])
    annotation_bbox = list(point_bboxes[answer_label])
    scene_bboxes = [list(warehouse_bbox)] + [list(bbox) for bbox in object_bboxes.values()]
    scene_bbox = [
        round(float(min(bbox[0] for bbox in scene_bboxes)), 3),
        round(float(min(bbox[1] for bbox in scene_bboxes)), 3),
        round(float(max(bbox[2] for bbox in scene_bboxes)), 3),
        round(float(max(bbox[3] for bbox in scene_bboxes)), 3),
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
        warehouse_bbox = bbox_transform(warehouse_bbox, scale_x=scale_x, scale_y=scale_y)
        object_bboxes = bbox_dict_transform(object_bboxes, scale_x=scale_x, scale_y=scale_y)
        object_centers = point_dict_transform(object_centers, scale_x=scale_x, scale_y=scale_y)
        point_bboxes = bbox_dict_transform(point_bboxes, scale_x=scale_x, scale_y=scale_y)
        point_centers = point_dict_transform(point_centers, scale_x=scale_x, scale_y=scale_y)
        context_bboxes = bbox_dict_transform(context_bboxes, scale_x=scale_x, scale_y=scale_y)
        context_centers = point_dict_transform(context_centers, scale_x=scale_x, scale_y=scale_y)
        reference_bboxes = bbox_dict_transform(reference_bboxes, scale_x=scale_x, scale_y=scale_y)
        reference_centers = point_dict_transform(reference_centers, scale_x=scale_x, scale_y=scale_y)
        annotation_bbox = bbox_transform(annotation_bbox, scale_x=scale_x, scale_y=scale_y)
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
    return _RenderedWarehouseScene(
        image=image,
        entities=list(entities),
        scene_bbox_px=list(scene_bbox),
        warehouse_bbox_px=list(warehouse_bbox),
        object_bboxes_px=dict(object_bboxes),
        object_centers_px=dict(object_centers),
        candidate_bboxes_px=dict(point_bboxes),
        candidate_centers_px=dict(point_centers),
        context_object_bboxes_px=dict(context_bboxes),
        context_object_centers_px=dict(context_centers),
        reference_object_bboxes_px=dict(reference_bboxes),
        reference_object_centers_px=dict(reference_centers),
        annotation_bboxes=[list(annotation_bbox)],
        annotation_entity_ids=[str(dataset["answer_object_id"])],
        option_panel_bbox_px=list(option_metadata["option_panel_bbox_px"]),
        option_choice_bboxes_px={str(key): list(value) for key, value in option_metadata["option_choice_bboxes_px"].items()},
        option_choices=[dict(choice) for choice in option_metadata["option_choices"]],
        option_panel_height_px=int(option_metadata["option_panel_height_px"]),
    )
