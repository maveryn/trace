"""Native preview adapters for the shared three_d object inventory."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from . import object_scene
from .camera_projection import build_projection_frame
from .object_rendering import (
    ThreeDObjectSpec,
    ThreeDRenderContext,
    rendered_three_d_object_from_bbox,
    render_three_d_object,
)
from .object_resources import (
    ThreeDObjectProfile,
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB,
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE,
)
from .object_scene import resolve_object_scene_render_params
from ..surface_fixture.shared.state import ELEMENT_TYPE_BY_SCENE_VARIANT, SEMANTIC_COLOR_RGB, SEMANTIC_COLOR_SUPPORT
from ..surface_fixture.shared.rendering import render_surface_fixture
from .warehouse_object_rendering import _draw_ground_shadow as _draw_warehouse_ground_shadow
from .warehouse_object_rendering import _fill_for_object as _warehouse_fill_for_object
from ..room.shared import state as room_scene
from ..room.shared.rendering import _draw_room_shell
from ..street.shared import state as street_scene
from ..street.shared.objects import _draw_styled_building_object
from ..street.shared.components import _draw_street_shell
from ..warehouse.shared import state as warehouse_scene
from ..warehouse.shared.components import _draw_shelf_rack_object
from .street_object_rendering_common import (
    STREET_BUILDING_CONTEXT_OBJECT_TYPES,
    _draw_shadow as _draw_street_shadow,
    _street_object_fill_rgb,
)


@dataclass(frozen=True)
class ObjectProfilePreview:
    """Rendered preview for one canonical 3D object profile."""

    image: Image.Image
    full_image: Image.Image
    object_bbox_px: List[float]
    crop_bbox_px: List[float] | None
    profile_id: str
    preview_renderer: str
    metadata: Dict[str, Any]


def _seed_for_profile(profile: ThreeDObjectProfile, instance_seed: int, salt: str) -> int:
    text = f"{profile.profile_id}.{salt}"
    return int(instance_seed) + sum((index + 1) * ord(char) for index, char in enumerate(text))


def _profile_rng(profile: ThreeDObjectProfile, instance_seed: int, salt: str) -> random.Random:
    return random.Random(_seed_for_profile(profile, int(instance_seed), str(salt)))


def _bbox_from_points(points: Sequence[Sequence[float]], *, pad_px: float = 0.0) -> List[float]:
    return [
        round(float(min(point[0] for point in points) - float(pad_px)), 3),
        round(float(min(point[1] for point in points) - float(pad_px)), 3),
        round(float(max(point[0] for point in points) + float(pad_px)), 3),
        round(float(max(point[1] for point in points) + float(pad_px)), 3),
    ]


def _crop_to_bbox(image: Image.Image, bbox: Sequence[float], *, padding_px: int) -> Tuple[Image.Image, List[float]]:
    x0 = max(0, int(math.floor(float(bbox[0]) - float(padding_px))))
    y0 = max(0, int(math.floor(float(bbox[1]) - float(padding_px))))
    x1 = min(int(image.width), int(math.ceil(float(bbox[2]) + float(padding_px))))
    y1 = min(int(image.height), int(math.ceil(float(bbox[3]) + float(padding_px))))
    if x1 <= x0 or y1 <= y0:
        return image.copy(), [0.0, 0.0, float(image.width), float(image.height)]
    return image.crop((x0, y0, x1, y1)), [float(x0), float(y0), float(x1), float(y1)]


def _normalise_dimensions(
    dimensions_xyz: Sequence[float] | None,
    *,
    fallback_xyz: Tuple[float, float, float],
    max_extent: float,
) -> Tuple[float, float, float]:
    raw = tuple(float(value) for value in dimensions_xyz) if dimensions_xyz is not None else tuple(fallback_xyz)
    if len(raw) == 1:
        dims = (raw[0], raw[0], raw[0])
    elif len(raw) == 2:
        dims = (raw[0], 0.18, raw[1])
    else:
        dims = raw[:3]
    largest = max(float(value) for value in dims)
    if largest > float(max_extent):
        scale = float(max_extent) / float(largest)
        dims = tuple(round(float(value) * float(scale), 4) for value in dims)
    elif largest < 0.30:
        scale = 0.42 / float(max(largest, 1e-6))
        dims = tuple(round(float(value) * float(scale), 4) for value in dims)
    return tuple(float(value) for value in dims)


def _object_scene_render_params(canvas_width: int, canvas_height: int, *, draw_grid: bool = True) -> object_scene._RenderParams:
    return object_scene._resolve_render_params(
        {
            "canvas_width": int(canvas_width),
            "canvas_height": int(canvas_height),
            "scene_margin_left_px": 48,
            "scene_margin_right_px": 48,
            "scene_margin_top_px": 40,
            "scene_margin_bottom_px": 48,
            "room_extent": 2.25,
            "room_height": 2.2,
            "grid_step": 0.8 if bool(draw_grid) else 0.0,
            "marker_radius_px": 18,
            "label_font_size_px": 1,
            "line_width_px": 2,
            "full_bleed_floor": False,
        },
        render_defaults={},
    )


def _render_object_scene_profile(
    profile: ThreeDObjectProfile,
    *,
    canvas_width: int,
    canvas_height: int,
    instance_seed: int,
) -> Tuple[Image.Image, List[float], Dict[str, Any]]:
    render_params = _object_scene_render_params(
        int(canvas_width),
        int(canvas_height),
        draw_grid=str(profile.source_scene) != "object_cluster",
    )
    rng = _profile_rng(profile, int(instance_seed), "object_scene")
    camera = object_scene._sample_camera(rng, yaw_band_degrees=(38.0, 62.0))
    dimensions = _normalise_dimensions(
        profile.dimensions_xyz,
        fallback_xyz=(0.62, 0.52, 0.62),
        max_extent=1.28,
    )
    target = object_scene._make_object_spec(
        object_id="inventory_target",
        shape_type=str(profile.object_type),
        object_role="context",
        xy=(0.0, 0.0),
        dimensions_xyz=tuple(float(value) for value in dimensions),
        dimension_scale=1.0,
        label=None,
    )
    target.update(
        {
            "object_name": str(profile.display_name),
            "prompt_name": str(profile.display_name),
            "nameable_for_prompt": True,
        }
    )
    reference_points = object_scene._object_reference_points(target) + object_scene._stage_reference_points(1.45)
    frame = object_scene._build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
    target = _finalise_object_scene_spec(target, camera=camera, frame=frame)
    dummy = object_scene._make_object_spec(
        object_id="object_A",
        shape_type="sphere",
        object_role="candidate",
        xy=(2.4, 2.4),
        dimensions_xyz=(0.10, 0.10, 0.10),
        dimension_scale=1.0,
        label="A",
    )
    dummy = _finalise_object_scene_spec(dummy, camera=camera, frame=frame)
    dataset = {
        "query_id": "closest_to_camera",
        "scene_variant": "studio_platform",
        "point_count": 1,
        "candidate_count": 1,
        "context_object_count": 1,
        "object_count": 2,
        "point_specs": [dummy],
        "context_object_specs": [target],
        "object_specs": [dummy, target],
        "answer_label": "A",
        "answer_point_id": "object_A",
        "camera": _camera_metadata(camera),
        "projection_frame": _frame_metadata(frame),
    }
    background = Image.new("RGB", (int(canvas_width), int(canvas_height)), (246, 248, 247))
    rendered = object_scene.render_object_scene_3d(background, dataset=dataset, render_params=render_params)
    bbox = list(rendered.object_bboxes_px["inventory_target"])
    return rendered.image, bbox, {
        "preview_adapter": "object_scene_shape",
        "scene_variant": "studio_platform",
        "dimensions_xyz": [float(value) for value in dimensions],
    }


def _finalise_object_scene_spec(spec: Mapping[str, Any], *, camera: object_scene._CameraSpec, frame: object_scene._ProjectionFrame) -> Dict[str, Any]:
    screen = object_scene._project_screen(spec["world_xyz"], camera, frame)
    updated = dict(spec)
    updated.update(
        {
            "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
            "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
            "camera_distance": round(float(screen[7]), 4),
        }
    )
    return updated


def _camera_metadata(camera: Any) -> Dict[str, Any]:
    return {
        "camera_position": [round(float(value), 4) for value in camera.camera_position],
        "target": [round(float(value), 4) for value in camera.target],
        "yaw_degrees": round(float(camera.yaw_degrees), 4),
        "pitch_degrees": round(float(camera.pitch_degrees), 4),
        "distance": round(float(camera.distance), 4),
        "right": [round(float(value), 5) for value in camera.right],
        "up": [round(float(value), 5) for value in camera.up],
        "forward": [round(float(value), 5) for value in camera.forward],
    }


def _frame_metadata(frame: Any) -> Dict[str, Any]:
    return {
        "scale": round(float(frame.scale), 5),
        "center_x": round(float(frame.center_x), 3),
        "center_y": round(float(frame.center_y), 3),
        "normalized_center_u": round(float(frame.normalized_center_u), 6),
        "normalized_center_v": round(float(frame.normalized_center_v), 6),
    }


def _room_reference_points(*specs: Mapping[str, Any]) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = [
        (-room_scene.WALL_X, room_scene.ROOM_FRONT_Y, 0.0),
        (room_scene.WALL_X, room_scene.ROOM_FRONT_Y, 0.0),
        (-room_scene.WALL_X, room_scene.WALL_BACK_Y, 0.0),
        (room_scene.WALL_X, room_scene.WALL_BACK_Y, 0.0),
        (-room_scene.WALL_X, room_scene.WALL_BACK_Y, room_scene.ROOM_HEIGHT),
        (room_scene.WALL_X, room_scene.WALL_BACK_Y, room_scene.ROOM_HEIGHT),
        (-room_scene.WALL_X, room_scene.ROOM_FRONT_Y, room_scene.ROOM_HEIGHT),
        (room_scene.WALL_X, room_scene.ROOM_FRONT_Y, room_scene.ROOM_HEIGHT),
    ]
    for spec in specs:
        if str(spec.get("object_role")) == "wall_object":
            points.extend(room_scene._wall_reference_points(spec))
        else:
            points.extend(_room_floor_reference_points(spec))
    return points


def _room_floor_reference_points(spec: Mapping[str, Any]) -> List[Tuple[float, float, float]]:
    x, y, _z = (float(value) for value in spec.get("world_xyz", (0.0, 0.0, 0.0)))
    base_z = float(spec.get("base_xyz", (x, y, 0.0))[2])
    width, depth, height = (float(value) for value in spec.get("dimensions_xyz", (0.6, 0.6, 0.6)))
    xs = (x - width * 0.5, x + width * 0.5)
    ys = (y - depth * 0.5, y + depth * 0.5)
    zs = (base_z, base_z + height)
    points = [(px, py, pz) for px in xs for py in ys for pz in zs]
    points.append((x, y, base_z + height * 0.5))
    return points


def _render_room_wall_profile(
    profile: ThreeDObjectProfile,
    *,
    canvas_width: int,
    canvas_height: int,
    instance_seed: int,
) -> Tuple[Image.Image, List[float], Dict[str, Any]]:
    render_params = resolve_object_scene_render_params(
        {
            "canvas_width": int(canvas_width),
            "canvas_height": int(canvas_height),
            "scene_margin_left_px": 42,
            "scene_margin_right_px": 42,
            "scene_margin_top_px": 36,
            "scene_margin_bottom_px": 42,
            "label_font_size_px": 1,
            "full_bleed_floor": False,
        },
        render_defaults={},
    )
    rng = _profile_rng(profile, int(instance_seed), "room_wall")
    camera = room_scene._sample_room_camera(rng, scene_variant="studio_room", yaw_band_degrees=(-24.0, 24.0))
    raw_dimensions = tuple(float(value) for value in (profile.dimensions_xyz or room_scene.ROOM_WALL_BASE_DIMENSIONS.get(str(profile.object_type), (0.52, 0.52))))
    width = raw_dimensions[0]
    height = raw_dimensions[1] if len(raw_dimensions) >= 2 else raw_dimensions[0]
    spec = room_scene._wall_spec(
        object_id="inventory_target",
        object_type=str(profile.object_type),
        wall="back",
        hpos=0.0,
        z=1.62,
        width=float(width),
        height=float(height),
        counts_for_query=False,
    )
    spec = room_scene._with_picture_scenery(spec, rng)
    frame = build_projection_frame(camera=camera, render_params=render_params, point_worlds=_room_reference_points(spec))
    spec = room_scene._finalize_specs([spec], camera=camera, frame=frame)[0]
    image = Image.new("RGB", (int(canvas_width), int(canvas_height)), (246, 248, 247))
    draw = ImageDraw.Draw(image)
    _draw_room_shell(draw, camera=camera, frame=frame, render_params=render_params, scene_variant="studio_room")
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
            scene_variant="studio_room",
        ),
    )
    bbox = list(rendered.bbox_xyxy)
    return image, list(bbox), {
        "preview_adapter": "room_wall_object",
        "scene_variant": "studio_room",
        "wall": "back",
        "wall_width": float(spec["wall_width"]),
        "wall_height": float(spec["wall_height"]),
    }


def _room_floor_target_spec(profile: ThreeDObjectProfile, rng: random.Random) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    object_type = str(profile.object_type)
    if str(profile.role) == "room_surface_variant":
        support = room_scene._make_floor_prop(
            rng=rng,
            object_id="inventory_support_side_table",
            prop_shape="side_table",
            xy=(0.0, 0.0),
        )
        target = room_scene._surface_distractor_for_type(
            rng=rng,
            object_id="inventory_target",
            object_type=object_type,
            support_spec=support,
        )
        target["prompt_name"] = str(profile.display_name)
        target["object_name"] = str(profile.display_name)
        return target, support
    if str(profile.role) == "room_floor_prop":
        target = room_scene._make_floor_prop(
            rng=rng,
            object_id="inventory_target",
            prop_shape=object_type,
            xy=(0.0, 0.0),
        )
        target["prompt_name"] = str(profile.display_name)
        target["object_name"] = str(profile.display_name)
        return target, None
    target = room_scene._floor_distractor_for_type(
        rng=rng,
        object_id="inventory_target",
        object_type=object_type,
        xy=(0.0, 0.0),
    )
    target["prompt_name"] = str(profile.display_name)
    target["object_name"] = str(profile.display_name)
    return target, None


def _render_room_floor_profile(
    profile: ThreeDObjectProfile,
    *,
    canvas_width: int,
    canvas_height: int,
    instance_seed: int,
) -> Tuple[Image.Image, List[float], Dict[str, Any]]:
    render_params = resolve_object_scene_render_params(
        {
            "canvas_width": int(canvas_width),
            "canvas_height": int(canvas_height),
            "scene_margin_left_px": 42,
            "scene_margin_right_px": 42,
            "scene_margin_top_px": 36,
            "scene_margin_bottom_px": 42,
            "label_font_size_px": 1,
            "full_bleed_floor": False,
        },
        render_defaults={},
    )
    rng = _profile_rng(profile, int(instance_seed), "room_floor")
    camera = room_scene._sample_room_camera(rng, scene_variant="studio_room", yaw_band_degrees=(-24.0, 24.0))
    target, support = _room_floor_target_spec(profile, rng)
    specs = [spec for spec in (support, target) if spec is not None]
    frame = build_projection_frame(camera=camera, render_params=render_params, point_worlds=_room_reference_points(*specs))
    final_specs = {str(spec["object_id"]): spec for spec in room_scene._finalize_specs(specs, camera=camera, frame=frame)}
    image = Image.new("RGB", (int(canvas_width), int(canvas_height)), (246, 248, 247))
    draw = ImageDraw.Draw(image)
    _draw_room_shell(draw, camera=camera, frame=frame, render_params=render_params, scene_variant="studio_room")
    if support is not None:
        render_three_d_object(
            ThreeDObjectSpec.from_mapping(
                final_specs[str(support["object_id"])],
                object_type_key="object_type",
                default_renderer_id="room_floor_object",
                role=str(support.get("object_role", "floor_object")),
                source_entity_type="three_d_room_floor_object",
            ),
            ThreeDRenderContext(
                draw=draw,
                camera=camera,
                frame=frame,
                render_params=render_params,
                scene_variant="studio_room",
            ),
        )
    target_spec = final_specs["inventory_target"]
    rendered = render_three_d_object(
        ThreeDObjectSpec.from_mapping(
            target_spec,
            object_type_key="object_type",
            default_renderer_id="room_floor_object",
            role=str(target_spec.get("object_role", "floor_object")),
            source_entity_type="three_d_room_floor_object",
        ),
        ThreeDRenderContext(
            draw=draw,
            camera=camera,
            frame=frame,
            render_params=render_params,
            scene_variant="studio_room",
        ),
    )
    bbox = list(rendered.bbox_xyxy)
    return image, list(bbox), {
        "preview_adapter": "room_floor_object",
        "scene_variant": "studio_room",
        "mounting": str(target_spec.get("mounting", "floor")),
        "support_object_id": target_spec.get("support_object_id"),
    }


def _render_street_profile(
    profile: ThreeDObjectProfile,
    *,
    canvas_width: int,
    canvas_height: int,
    instance_seed: int,
) -> Tuple[Image.Image, List[float], Dict[str, Any]]:
    render_params = street_scene._resolve_render_params(
        {
            "canvas_width": int(canvas_width),
            "canvas_height": int(canvas_height),
            "scene_margin_left_px": 34,
            "scene_margin_right_px": 34,
            "scene_margin_top_px": 32,
            "scene_margin_bottom_px": 36,
            "street_extent": 3.8,
            "road_half_width": 0.84,
            "label_font_size_px": 1,
        },
        render_defaults={},
    )
    rng = _profile_rng(profile, int(instance_seed), "street")
    camera = object_scene._sample_camera(rng, yaw_band_degrees=(32.0, 52.0))
    object_type = str(profile.object_type)
    orientation_axis = "x"
    dimensions = street_scene._dimensions_for_orientation(object_type, orientation_axis=orientation_axis, scale=0.92)
    xy = (0.0, -1.45) if str(profile.role) == "street_candidate" else (0.0, 1.50)
    spec = street_scene._make_street_object_spec(
        object_id="inventory_target",
        object_type=object_type,
        object_role=str(profile.role),
        xy=xy,
        intersection_center_xy=(0.0, 0.0),
        orientation_axis=orientation_axis,
        dimensions_xyz=dimensions,
        label=None,
        dimension_scale=0.92,
    )
    spec["object_name"] = str(profile.display_name)
    spec["prompt_name"] = str(profile.display_name)
    street_support = str(profile.resource_kind) == "scene_support"
    if street_support and object_type in STREET_BUILDING_CONTEXT_OBJECT_TYPES:
        style = street_scene._fixed_building_style_for_street_object(object_type) or "concrete_midrise"
        spec = street_scene._apply_street_building_style(spec, style=style)
    reference_points = [
        (-render_params.street_extent, -render_params.street_extent, 0.0),
        (render_params.street_extent, -render_params.street_extent, 0.0),
        (render_params.street_extent, render_params.street_extent, 0.0),
        (-render_params.street_extent, render_params.street_extent, 0.0),
        (-render_params.street_extent, -render_params.street_extent, 1.7),
        (render_params.street_extent, render_params.street_extent, 1.7),
        *object_scene._object_reference_points(spec),
    ]
    frame = object_scene._build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
    spec = street_scene._finalize_specs([spec], camera=camera, frame=frame)[0]
    image = Image.new("RGB", (int(canvas_width), int(canvas_height)), (216, 224, 220))
    draw = ImageDraw.Draw(image)
    _draw_street_shell(
        draw,
        camera=camera,
        frame=frame,
        render_params=render_params,
        scene_variant="downtown_intersection",
        intersection_center_xy=(0.0, 0.0),
        intersection_layout="four_way",
    )
    _draw_street_shadow(draw, spec, camera=camera, frame=frame)
    object_spec = ThreeDObjectSpec.from_mapping(
        spec,
        object_type_key="object_type",
        default_renderer_id="street_object",
        role=str(spec.get("object_role", profile.role)),
        source_entity_type="three_d_street_context_object" if str(profile.role) == "street_context" else "three_d_street_candidate_object",
    )
    render_context = ThreeDRenderContext(
        draw=draw,
        camera=camera,
        frame=frame,
        render_params=render_params,
        fill_rgb=_street_object_fill_rgb(spec),
        scene_variant="downtown_intersection",
    )
    if street_support and object_type in STREET_BUILDING_CONTEXT_OBJECT_TYPES:
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
    bbox = list(rendered.bbox_xyxy)
    return image, list(bbox), {
        "preview_adapter": "street_object",
        "scene_variant": "downtown_intersection",
        "object_role": str(profile.role),
        "orientation_axis": orientation_axis,
        "dimensions_xyz": [float(value) for value in dimensions],
    }


def _warehouse_target_spec(profile: ThreeDObjectProfile) -> Dict[str, Any]:
    object_type = str(profile.object_type)
    orientation_axis = "x"
    dimensions = warehouse_scene._dimensions_for_object(object_type, orientation_axis=orientation_axis, scale=0.92)
    spec = warehouse_scene._make_object_spec(
        object_id="inventory_target",
        object_type=object_type,
        object_role=str(profile.role),
        xy=(0.0, 0.0),
        orientation_axis=orientation_axis,
        dimensions_xyz=dimensions,
        dimension_scale=0.92,
        label=None,
    )
    spec["object_name"] = str(profile.display_name)
    spec["prompt_name"] = str(profile.display_name)
    if object_type == "warehouse_robot":
        spec.update(
            {
                "robot_design": "low_cart",
                "robot_heading": "east",
                "robot_base_rgb": [int(channel) for channel in warehouse_scene.ROBOT_BASE_COLORS[0]],
                "robot_accent_rgb": [int(channel) for channel in warehouse_scene.ROBOT_ACCENT_COLORS[0]],
            }
        )
    warehouse_support = str(profile.resource_kind) == "scene_support"
    if warehouse_support and object_type == "shelf_rack":
        spec.update(
            {
                "shelf_style": "mixed_crates",
                "shelf_levels": 3,
                "shelf_level_fracs": [0.14, 0.50, 0.84],
                "shelf_frame_rgb": [int(channel) for channel in warehouse_scene.SHELF_FRAME_COLORS[0]],
                "shelf_load_slots": [
                    {"level": 0, "x_frac": -0.36, "color": [176, 116, 72]},
                    {"level": 1, "x_frac": 0.18, "color": [94, 132, 174]},
                    {"level": 2, "x_frac": 0.42, "color": [142, 156, 92]},
                ],
            }
        )
    return spec


def _render_warehouse_profile(
    profile: ThreeDObjectProfile,
    *,
    canvas_width: int,
    canvas_height: int,
    instance_seed: int,
) -> Tuple[Image.Image, List[float], Dict[str, Any]]:
    render_params = warehouse_scene._resolve_render_params(
        {
            "canvas_width": int(canvas_width),
            "canvas_height": int(canvas_height),
            "scene_margin_left_px": 38,
            "scene_margin_right_px": 38,
            "scene_margin_top_px": 34,
            "scene_margin_bottom_px": 40,
            "warehouse_extent": 3.6,
            "grid_step": 0.8,
            "label_font_size_px": 1,
        },
        render_defaults={},
    )
    rng = _profile_rng(profile, int(instance_seed), "warehouse")
    camera = object_scene._sample_camera(rng, yaw_band_degrees=(32.0, 52.0))
    spec = _warehouse_target_spec(profile)
    reference_points = object_scene._stage_reference_points(2.0) + object_scene._object_reference_points(spec)
    frame = object_scene._build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
    spec = warehouse_scene._finalize_specs([spec], camera=camera, frame=frame)[0]
    image = Image.new("RGB", (int(canvas_width), int(canvas_height)), tuple(int(value) for value in render_params.floor_rgb))
    draw = ImageDraw.Draw(image)
    _draw_simple_floor_grid(draw, camera=camera, frame=frame, width=int(canvas_width), height=int(canvas_height))
    _draw_warehouse_ground_shadow(draw, spec, camera=camera, frame=frame)
    fill = (
        WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB
        if str(spec.get("object_type")) == WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE
        else _warehouse_fill_for_object(spec, scene_variant="storage_aisle")
    )
    object_spec = ThreeDObjectSpec.from_mapping(
        spec,
        object_type_key="object_type",
        default_renderer_id="warehouse_object",
        role=str(spec.get("object_role", profile.role)),
        source_entity_type="three_d_warehouse_object",
    )
    render_context = ThreeDRenderContext(
        draw=draw,
        camera=camera,
        frame=frame,
        render_params=render_params,
        fill_rgb=fill,
        scene_variant="storage_aisle",
    )
    warehouse_support = str(profile.resource_kind) == "scene_support"
    if warehouse_support and str(spec.get("object_type")) == "shelf_rack":
        rack_bbox = _draw_shelf_rack_object(draw, spec, camera=camera, frame=frame, fill=fill)
        rendered = rendered_three_d_object_from_bbox(object_spec, render_context, bbox_xyxy=rack_bbox)
    else:
        rendered = render_three_d_object(object_spec, render_context)
    bbox = list(rendered.bbox_xyxy)
    return image, list(bbox), {
        "preview_adapter": "warehouse_object",
        "scene_variant": "storage_aisle",
        "object_role": str(profile.role),
        "orientation_axis": str(spec.get("orientation_axis", "")),
        "dimensions_xyz": list(spec["dimensions_xyz"]),
    }


def _render_surface_fixture_profile(
    profile: ThreeDObjectProfile,
    *,
    canvas_width: int,
    canvas_height: int,
    instance_seed: int,
) -> Tuple[Image.Image, List[float], Dict[str, Any]]:
    scene_variant = str(profile.object_type)
    if scene_variant not in ELEMENT_TYPE_BY_SCENE_VARIANT:
        raise ValueError(f"unsupported surface fixture profile object_type: {scene_variant}")
    element_type = str(ELEMENT_TYPE_BY_SCENE_VARIANT[scene_variant])
    render_params = resolve_object_scene_render_params(
        {
            "canvas_width": int(canvas_width),
            "canvas_height": int(canvas_height),
            "scene_margin_left_px": 24,
            "scene_margin_right_px": 24,
            "scene_margin_top_px": 20,
            "scene_margin_bottom_px": 20,
            "grid_step": 0.0,
        },
        render_defaults={},
    )
    rows = 3
    cols = 4
    rng = _profile_rng(profile, int(instance_seed), "surface_fixture_preview")
    color_pool: List[str] = []
    while len(color_pool) < rows * cols:
        color_cycle = [str(value) for value in SEMANTIC_COLOR_SUPPORT]
        rng.shuffle(color_cycle)
        color_pool.extend(color_cycle)
    cells: List[Dict[str, Any]] = []
    for index in range(rows * cols):
        row = index // cols
        col = index % cols
        color_name = str(color_pool[int(index)])
        state = "normal"
        if scene_variant in {"locker_bank", "mailbox_bank", "door_bank"} and index % 5 == 0:
            state = "open"
        elif scene_variant in {"server_rack", "control_panel", "window_grid", "indicator_light_panel"} and index % 4 == 1:
            state = "lit"
        elif scene_variant == "indicator_light_panel" and index % 4 == 3:
            state = "unlit"
        elif scene_variant == "control_panel" and index % 5 == 2:
            state = "pressed"
        elif scene_variant == "solar_panel_array" and index % 6 == 3:
            state = "cracked"
        u_pad = 0.065
        v_pad = 0.075
        gap = 0.016
        u0 = u_pad + (float(col) / float(cols)) * (1.0 - 2.0 * u_pad)
        u1 = u_pad + (float(col + 1) / float(cols)) * (1.0 - 2.0 * u_pad)
        v0 = v_pad + (float(row) / float(rows)) * (1.0 - 2.0 * v_pad)
        v1 = v_pad + (float(row + 1) / float(rows)) * (1.0 - 2.0 * v_pad)
        cells.append(
            {
                "element_id": f"{element_type}_{index:02d}",
                "cell_id": f"cell_{index:02d}",
                "flat_index": int(index),
                "element_type": str(element_type),
                "row": int(row),
                "column": int(col),
                "u0": float(u0 + gap),
                "u1": float(u1 - gap),
                "v0": float(v0 + gap),
                "v1": float(v1 - gap),
                "present": True,
                "color_name": str(color_name),
                "fill_rgb": list(SEMANTIC_COLOR_RGB[str(color_name)]),
                "state": str(state),
                "count_role": "target",
            }
        )
    dataset = {
        "query_id": "object_inventory_preview",
        "scene_variant": str(scene_variant),
        "target_element_type": str(element_type),
        "target_count": int(rows * cols),
        "surface_cells": list(cells),
        "layout_rows": int(rows),
        "layout_columns": int(cols),
        "layout_style": "uniform_grid",
        "surface_world_corners": [
            [-2.0, 1.35, 2.55],
            [2.0, 1.35, 2.55],
            [2.0, 1.35, 0.15],
            [-2.0, 1.35, 0.15],
        ],
    }
    image = Image.new("RGB", (int(canvas_width), int(canvas_height)), (226, 232, 231))
    rendered = render_surface_fixture(image, dataset=dataset, render_params=render_params)
    return (
        rendered.image,
        list(rendered.fixture_bbox_px),
        {
            "preview_adapter": "surface_fixture",
            "scene_variant": str(scene_variant),
            "element_type": str(element_type),
            "layout_rows": int(rows),
            "layout_columns": int(cols),
            "dimensions_xyz": list(profile.dimensions_xyz or ()),
        },
    )


def _draw_simple_floor_grid(
    draw: ImageDraw.ImageDraw,
    *,
    camera: object_scene._CameraSpec,
    frame: object_scene._ProjectionFrame,
    width: int,
    height: int,
) -> None:
    draw.rectangle((0, 0, int(width), int(height)), fill=(226, 232, 231))
    grid_values = [round(-2.4 + index * 0.6, 3) for index in range(9)]
    for value in grid_values:
        a = object_scene._project_xy((value, -2.4, 0.0), camera, frame)
        b = object_scene._project_xy((value, 2.4, 0.0), camera, frame)
        object_scene._draw_line(draw, a, b, fill=(174, 185, 188), width=1)
        c = object_scene._project_xy((-2.4, value, 0.0), camera, frame)
        d = object_scene._project_xy((2.4, value, 0.0), camera, frame)
        object_scene._draw_line(draw, c, d, fill=(174, 185, 188), width=1)


def render_three_d_object_profile_preview(
    profile: ThreeDObjectProfile,
    *,
    canvas_width: int = 420,
    canvas_height: int = 330,
    instance_seed: int = 0,
    crop_to_object: bool = True,
    crop_padding_px: int = 18,
) -> ObjectProfilePreview:
    """Render an inventory preview using a scene adapter that supports the profile."""

    renderer = str(profile.renderer)
    if renderer == "object_scene_shape":
        image, bbox, metadata = _render_object_scene_profile(
            profile,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
        )
    elif renderer == "room_wall_object":
        image, bbox, metadata = _render_room_wall_profile(
            profile,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
        )
    elif renderer == "room_floor_object":
        image, bbox, metadata = _render_room_floor_profile(
            profile,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
        )
    elif renderer == "street_object":
        image, bbox, metadata = _render_street_profile(
            profile,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
        )
    elif renderer == "warehouse_object":
        image, bbox, metadata = _render_warehouse_profile(
            profile,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
        )
    elif renderer == "surface_fixture":
        image, bbox, metadata = _render_surface_fixture_profile(
            profile,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
        )
    else:
        raise ValueError(f"unsupported three_d inventory preview renderer: {renderer}")

    crop_bbox = None
    preview_image = image
    if bool(crop_to_object):
        preview_image, crop_bbox = _crop_to_bbox(image, bbox, padding_px=int(crop_padding_px))

    metadata = {
        **dict(metadata),
        "profile_renderer": str(profile.renderer),
        "profile_source_scene": str(profile.source_scene),
        "profile_role": str(profile.role),
        "profile_resource_kind": str(profile.resource_kind),
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
        "crop_to_object": bool(crop_to_object),
        "crop_padding_px": int(crop_padding_px),
    }
    return ObjectProfilePreview(
        image=preview_image,
        full_image=image,
        object_bbox_px=[round(float(value), 3) for value in bbox],
        crop_bbox_px=crop_bbox,
        profile_id=str(profile.profile_id),
        preview_renderer=str(metadata["preview_adapter"]),
        metadata=metadata,
    )
