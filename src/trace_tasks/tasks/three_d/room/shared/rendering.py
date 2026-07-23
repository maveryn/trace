"""Final scene rendering orchestration for the 3D room scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font
from ...shared.camera_projection import project_screen, screen_to_floor_xy
from ...shared.canvas import (
    bbox_dict_transform,
    bbox_transform,
    entities_transform,
    point_dict_transform,
    resize_image_to_fit_pixel_cap,
)
from ...shared.object_rendering import ThreeDObjectSpec, ThreeDRenderContext, render_three_d_object
from ...shared.object_scene import ObjectSceneRenderParams
from ...shared.object_scene_primitives import bbox_union, draw_line, shade_rgb, tint_rgb
from ...shared.option_panel import append_text_option_panel, empty_option_panel_metadata
from ...shared.room_wall_rendering_geometry import _draw_poly, _draw_poly_fill_only
from .state import (
    ROOM_FRONT_Y,
    ROOM_HEIGHT,
    ROOM_RENDER_FRONT_MAX_EXTENSION,
    ROOM_RENDER_FRONT_MIN_EXTENSION,
    ROOM_RENDER_SIDE_WALL_MAX_EXTENSION,
    WALL_BACK_Y,
    WALL_X,
    _wall_spec,
)
from ...shared.room_wall_object_rendering import _draw_wall_flat_object


@dataclass(frozen=True)
class _RenderedRoomScene:
    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    object_bboxes_px: Dict[str, List[float]]
    object_centers_px: Dict[str, List[float]]
    wall_object_bboxes_px: Dict[str, List[float]]
    wall_object_centers_px: Dict[str, List[float]]
    floor_object_bboxes_px: Dict[str, List[float]]
    floor_object_centers_px: Dict[str, List[float]]
    room_bbox_px: List[float]
    annotation_bboxes: List[List[float]]
    annotation_entity_ids: List[str]
    option_panel_bbox_px: List[float]
    option_choice_bboxes_px: Dict[str, List[float]]
    option_choices: List[Dict[str, Any]]
    option_panel_height_px: int


def _draw_room_shell(draw: ImageDraw.ImageDraw, *, camera, frame, render_params: ObjectSceneRenderParams, scene_variant: str) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Draw room floor/walls while preserving projected wall bounding boxes.

    The returned boxes define render-map anchors for later object placement and
    must stay consistent with the same camera/projection used for furniture.
    """

    floor_rgb = tuple(int(value) for value in render_params.floor_rgb)
    back_rgb = tint_rgb((202, 210, 216), 0.30)
    side_rgb = shade_rgb(back_rgb, 0.94)
    if str(scene_variant) == "living_room":
        back_rgb = (223, 216, 206)
        side_rgb = (214, 221, 216)
    elif str(scene_variant) == "office_room":
        back_rgb = (216, 222, 228)
        side_rgb = (224, 221, 214)

    render_front_y = _room_render_front_y(camera=camera, frame=frame, render_params=render_params)
    side_wall_front_y = max(
        float(render_front_y),
        float(ROOM_FRONT_Y) - float(ROOM_RENDER_SIDE_WALL_MAX_EXTENSION),
    )
    back_wall = [(-WALL_X, WALL_BACK_Y, 0.0), (WALL_X, WALL_BACK_Y, 0.0), (WALL_X, WALL_BACK_Y, ROOM_HEIGHT), (-WALL_X, WALL_BACK_Y, ROOM_HEIGHT)]
    left_wall = [(-WALL_X, side_wall_front_y, 0.0), (-WALL_X, WALL_BACK_Y, 0.0), (-WALL_X, WALL_BACK_Y, ROOM_HEIGHT), (-WALL_X, side_wall_front_y, ROOM_HEIGHT)]
    right_wall = [(WALL_X, WALL_BACK_Y, 0.0), (WALL_X, side_wall_front_y, 0.0), (WALL_X, side_wall_front_y, ROOM_HEIGHT), (WALL_X, WALL_BACK_Y, ROOM_HEIGHT)]
    floor = [(-WALL_X, render_front_y, 0.0), (WALL_X, render_front_y, 0.0), (WALL_X, WALL_BACK_Y, 0.0), (-WALL_X, WALL_BACK_Y, 0.0)]

    wall_bboxes = [
        _draw_poly_fill_only(draw, left_wall, camera=camera, frame=frame, fill=side_rgb),
        _draw_poly_fill_only(draw, right_wall, camera=camera, frame=frame, fill=shade_rgb(side_rgb, 0.97)),
        _draw_poly(draw, back_wall, camera=camera, frame=frame, fill=back_rgb, outline=(94, 103, 114), width=2),
    ]

    floor_bbox = _draw_poly_fill_only(draw, floor, camera=camera, frame=frame, fill=floor_rgb)

    grid_rgb = tuple(int(value) for value in render_params.grid_rgb)
    step = float(render_params.grid_step)
    x = -WALL_X
    while x <= WALL_X + 1e-6:
        a = project_screen((x, render_front_y, 0.0), camera, frame)
        b = project_screen((x, WALL_BACK_Y, 0.0), camera, frame)
        draw_line(draw, (a[0], a[1]), (b[0], b[1]), fill=grid_rgb, width=1)
        x += step
    y = render_front_y
    while y <= WALL_BACK_Y + 1e-6:
        a = project_screen((-WALL_X, y, 0.0), camera, frame)
        b = project_screen((WALL_X, y, 0.0), camera, frame)
        draw_line(draw, (a[0], a[1]), (b[0], b[1]), fill=grid_rgb, width=1)
        y += step

    bboxes = [floor_bbox, *wall_bboxes]

    door = _wall_spec(object_id="room_door", object_type="door", wall="left", hpos=0.72, z=0.92, width=0.82, height=1.84, counts_for_query=False)
    bboxes.append(_draw_wall_flat_object(draw, door, camera=camera, frame=frame, fill=(150, 116, 84), trim=(74, 60, 48)))

    room_bbox = bbox_union(*bboxes)
    entities = [
        {
            "entity_id": "room_shell",
            "entity_type": "three_d_room_shell",
            "bbox_px": list(room_bbox),
            "attrs": {
                "scene_variant": str(scene_variant),
                "room_extent_xyz": [float(WALL_X), float(WALL_BACK_Y - ROOM_FRONT_Y), float(ROOM_HEIGHT)],
                "render_front_y": round(float(render_front_y), 4),
                "render_side_wall_front_y": round(float(side_wall_front_y), 4),
                "semantic_front_y": float(ROOM_FRONT_Y),
                "has_floor": True,
                "has_walls": True,
            },
        }
    ]
    return list(room_bbox), entities


def _room_render_front_y(*, camera, frame, render_params: ObjectSceneRenderParams) -> float:
    floor_hits = [
        screen_to_floor_xy(screen_x, float(render_params.canvas_height), camera=camera, frame=frame)
        for screen_x in (0.0, float(render_params.canvas_width) * 0.5, float(render_params.canvas_width))
    ]
    valid_y = [float(point[1]) for point in floor_hits if point is not None]
    if valid_y:
        desired_front_y = min(valid_y) - 0.35
    else:
        desired_front_y = float(ROOM_FRONT_Y) - float(ROOM_RENDER_FRONT_MAX_EXTENSION)
    min_front_y = float(ROOM_FRONT_Y) - float(ROOM_RENDER_FRONT_MAX_EXTENSION)
    max_front_y = float(ROOM_FRONT_Y) - float(ROOM_RENDER_FRONT_MIN_EXTENSION)
    return max(float(min_front_y), min(float(max_front_y), float(desired_front_y)))

def _draw_room_option_label(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    center: Sequence[float],
    font,
) -> List[float]:
    x, y = float(center[0]), float(center[1])
    text_bbox = draw.textbbox((0, 0), str(label), font=font, stroke_width=3)
    width = float(text_bbox[2] - text_bbox[0])
    height = float(text_bbox[3] - text_bbox[1])
    label_bbox = [
        round(x - width * 0.5 - 4.0, 3),
        round(y - height * 0.5 - 5.0, 3),
        round(x + width * 0.5 + 4.0, 3),
        round(y + height * 0.5 + 3.0, 3),
    ]
    draw_text_traced(draw,
        (x - width * 0.5, y - height * 0.5 - 1.0),
        str(label),
        font=font,
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(24, 29, 38),
     role="readout", required=False,)
    return list(label_bbox)

def render_room_scene_3d(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: ObjectSceneRenderParams,
    option_choices: Sequence[Mapping[str, Any]] = (),
) -> _RenderedRoomScene:
    """Render the full 3D room scene and collect object/option projections.

    The renderer is the single source for visible bboxes, so task annotations
    and option panels must use the maps produced by this function.
    """

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    camera_spec = dataset["camera"]
    frame_spec = dataset["projection_frame"]
    camera = type("CameraTuple", (), {})()
    camera.camera_position = tuple(float(value) for value in camera_spec["camera_position"])
    camera.target = tuple(float(value) for value in camera_spec["target"])
    camera.right = tuple(float(value) for value in camera_spec["right"])
    camera.up = tuple(float(value) for value in camera_spec["up"])
    camera.forward = tuple(float(value) for value in camera_spec["forward"])
    camera.yaw_degrees = float(camera_spec["yaw_degrees"])
    camera.pitch_degrees = float(camera_spec["pitch_degrees"])
    camera.distance = float(camera_spec["distance"])
    frame = type("FrameTuple", (), {})()
    frame.scale = float(frame_spec["scale"])
    frame.center_x = float(frame_spec["center_x"])
    frame.center_y = float(frame_spec["center_y"])
    frame.normalized_center_u = float(frame_spec["normalized_center_u"])
    frame.normalized_center_v = float(frame_spec["normalized_center_v"])

    scene_variant = str(dataset["scene_variant"])
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    room_bbox, entities = _draw_room_shell(draw, camera=camera, frame=frame, render_params=render_params, scene_variant=scene_variant)
    wall_specs = [dict(spec) for spec in dataset["wall_object_specs"]]
    floor_specs = [dict(spec) for spec in dataset["floor_object_specs"]]

    rendered_objects: Dict[str, Any] = {}
    label_bboxes: Dict[str, List[float]] = {}
    for spec in sorted(wall_specs, key=lambda item: float(item["camera_distance"]), reverse=True):
        rendered = render_three_d_object(
            ThreeDObjectSpec.from_mapping(
                spec,
                object_type_key="object_type",
                default_renderer_id="room_wall_object",
                role=str(spec.get("object_role", "wall_object")),
                source_entity_type="three_d_room_wall_object",
            ),
            ThreeDRenderContext(
                draw=draw,
                camera=camera,
                frame=frame,
                render_params=render_params,
                scene_variant=str(scene_variant),
            ),
        )
        rendered_objects[str(spec["object_id"])] = rendered
    for spec in sorted(floor_specs, key=lambda item: (float(item["camera_distance"]), -int(item.get("draw_order", 10))), reverse=True):
        rendered = render_three_d_object(
            ThreeDObjectSpec.from_mapping(
                spec,
                object_type_key="object_type",
                default_renderer_id="room_floor_object",
                role=str(spec.get("object_role", "floor_object")),
                source_entity_type="three_d_room_floor_object",
            ),
            ThreeDRenderContext(
                draw=draw,
                camera=camera,
                frame=frame,
                render_params=render_params,
                scene_variant=str(scene_variant),
            ),
        )
        rendered_objects[str(spec["object_id"])] = rendered
    for spec in wall_specs:
        label = str(spec.get("point_label", ""))
        if label and not option_choices:
            center = (float(spec["screen_xy"][0]), float(spec["screen_xy"][1]))
            label_bboxes[str(spec["object_id"])] = _draw_room_option_label(draw, label=label, center=center, font=label_font)

    object_bboxes: Dict[str, List[float]] = {}
    object_centers: Dict[str, List[float]] = {}
    wall_object_bboxes: Dict[str, List[float]] = {}
    wall_object_centers: Dict[str, List[float]] = {}
    floor_object_bboxes: Dict[str, List[float]] = {}
    floor_object_centers: Dict[str, List[float]] = {}
    all_specs = [*wall_specs, *floor_specs]
    for spec in all_specs:
        object_id = str(spec["object_id"])
        rendered_object = rendered_objects[object_id]
        bbox = list(rendered_object.bbox_xyxy)
        if object_id in label_bboxes:
            bbox = bbox_union(bbox, label_bboxes[object_id])
        center = [round(float(spec["screen_xy"][0]), 3), round(float(spec["screen_xy"][1]), 3)]
        object_bboxes[object_id] = list(bbox)
        object_centers[object_id] = list(center)
        if str(spec.get("object_role")) == "wall_object":
            wall_object_bboxes[object_id] = list(bbox)
            wall_object_centers[object_id] = list(center)
        else:
            floor_object_bboxes[object_id] = list(bbox)
            floor_object_centers[object_id] = list(center)
        entities.append(
            {
                "entity_id": object_id,
                "entity_type": "three_d_room_wall_object" if str(spec.get("object_role")) == "wall_object" else "three_d_room_floor_object",
                "bbox_px": list(bbox),
                "attrs": {
                    "object_type": str(spec["object_type"]),
                    "object_name": str(spec["object_name"]),
                    "prompt_name": str(spec["prompt_name"]),
                    "object_role": str(spec.get("object_role", "")),
                    "is_wall_mounted": bool(spec.get("is_wall_mounted", False)),
                    "mounting": str(spec.get("mounting", "")),
                    "wall": spec.get("wall"),
                    "world_xyz": list(spec["world_xyz"]),
                    "base_xyz": list(spec["base_xyz"]),
                    "dimensions_xyz": list(spec["dimensions_xyz"]),
                    "screen_xy": list(spec["screen_xy"]),
                    "camera_xyz": list(spec["camera_xyz"]),
                    "camera_distance": float(spec["camera_distance"]),
                    "scene_variant": str(scene_variant),
                    "support_object_id": spec.get("support_object_id"),
                    "support_surface_type": spec.get("support_surface_type"),
                    "is_reference_furniture": bool(spec.get("is_reference_furniture", False)),
                    "adjacent_wall": spec.get("adjacent_wall"),
                    "wall_axis_interval": spec.get("wall_axis_interval"),
                    "wall_gap": spec.get("wall_gap"),
                    "scenery_variant": spec.get("scenery_variant"),
                    "picture_content": spec.get("picture_content"),
                    "point_label": spec.get("point_label"),
                    "is_answer_candidate": bool(spec.get("is_answer_candidate", False)),
                    "object_record": dict(rendered_object.object_record),
                },
            }
        )

    annotation_ids = [str(value) for value in dataset["target_object_ids"]]
    annotation_bboxes = [list(object_bboxes[object_id]) for object_id in annotation_ids]
    all_bboxes = [list(room_bbox), *[list(value) for value in object_bboxes.values()]]
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
        object_bboxes = bbox_dict_transform(object_bboxes, scale_x=scale_x, scale_y=scale_y)
        object_centers = point_dict_transform(object_centers, scale_x=scale_x, scale_y=scale_y)
        wall_object_bboxes = bbox_dict_transform(wall_object_bboxes, scale_x=scale_x, scale_y=scale_y)
        wall_object_centers = point_dict_transform(wall_object_centers, scale_x=scale_x, scale_y=scale_y)
        floor_object_bboxes = bbox_dict_transform(floor_object_bboxes, scale_x=scale_x, scale_y=scale_y)
        floor_object_centers = point_dict_transform(floor_object_centers, scale_x=scale_x, scale_y=scale_y)
        room_bbox = bbox_transform(room_bbox, scale_x=scale_x, scale_y=scale_y)
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
    return _RenderedRoomScene(
        image=image,
        entities=list(entities),
        scene_bbox_px=list(scene_bbox),
        object_bboxes_px=dict(object_bboxes),
        object_centers_px=dict(object_centers),
        wall_object_bboxes_px=dict(wall_object_bboxes),
        wall_object_centers_px=dict(wall_object_centers),
        floor_object_bboxes_px=dict(floor_object_bboxes),
        floor_object_centers_px=dict(floor_object_centers),
        room_bbox_px=list(room_bbox),
        annotation_bboxes=list(annotation_bboxes),
        annotation_entity_ids=list(annotation_ids),
        option_panel_bbox_px=list(option_metadata["option_panel_bbox_px"]),
        option_choice_bboxes_px={str(key): list(value) for key, value in option_metadata["option_choice_bboxes_px"].items()},
        option_choices=[dict(choice) for choice in option_metadata["option_choices"]],
        option_panel_height_px=int(option_metadata["option_panel_height_px"]),
    )

__all__ = ["render_room_scene_3d"]
