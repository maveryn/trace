import ast
import random
from pathlib import Path

from PIL import Image

from trace_tasks.tasks.three_d.shared import object_scene
from trace_tasks.tasks.three_d.shared import object_rendering
from trace_tasks.tasks.three_d.shared import object_scene_rendering
from trace_tasks.tasks.three_d.shared import object_scene_primitives
from trace_tasks.tasks.three_d.shared.object_resources import (
    OBJECT_SCENE_ID,
    THREE_D_OBJECT_PROFILES,
    object_profile,
    object_profile_by_id,
    object_profile_or_none,
    profile_display_name,
    profile_dimensions_xyz,
    scene_profile_ids,
)
from trace_tasks.tasks.three_d.shared.camera_projection import CameraSpec
from trace_tasks.tasks.three_d.shared.scene_schema import ThreeDPlacementSpec, ThreeDSceneStyleSpec
from trace_tasks.tasks.three_d.object_scene import camera_distance_extremum_label as camera_distance


def _called_names_in_function(path: str, function_name: str) -> set[str]:
    tree = ast.parse(Path(path).read_text(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
    raise AssertionError(f"function {function_name} not found in {path}")


def test_camera_distance_uses_shared_object_scene_renderer_without_draw_helper_facade() -> None:
    assert camera_distance._RenderParams is object_scene._RenderParams
    assert camera_distance._make_object_spec is object_scene._make_object_spec
    assert camera_distance.render_object_scene_3d is object_scene.render_object_scene_3d
    assert not any(name.startswith("_draw_") for name in dir(camera_distance))


def test_object_scene_rendering_reexports_primitive_helpers() -> None:
    assert object_scene_rendering._bbox_union is object_scene_primitives._bbox_union
    assert object_scene_rendering._draw_box_object is object_scene_primitives._draw_box_object
    assert object_scene_rendering._draw_cylinder_object is object_scene_primitives._draw_cylinder_object
    assert object_scene_rendering._draw_torus_object is object_scene_primitives._draw_torus_object
    assert object_scene_rendering._sub_box_spec is object_scene_primitives._sub_box_spec


def test_box_visible_faces_use_rotated_local_axes() -> None:
    camera = CameraSpec(
        camera_position=(10.0, 2.0, 5.0),
        target=(0.0, 0.0, 0.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        forward=(-1.0, 0.0, 0.0),
        yaw_degrees=0.0,
        pitch_degrees=25.0,
        distance=10.0,
    )
    spec = {
        "world_xyz": (0.0, 0.0, 0.25),
        "base_xyz": (0.0, 0.0, 0.0),
        "dimensions_xyz": (0.5, 0.5, 0.5),
        "orientation_deg": 180.0,
    }

    assert object_scene_primitives._camera_facing_local_signs(spec, camera) == (-1, -1)


def test_object_scene_delegates_shape_dispatch_to_shared_renderer() -> None:
    source = Path("src/trace_tasks/tasks/three_d/shared/object_scene.py").read_text()

    assert "render_three_d_object(" in source
    assert "_draw_sphere_object" not in source
    assert "_draw_box_object" not in source
    assert object_rendering.render_three_d_object is not None


def test_scene_render_loops_delegate_reusable_objects_to_shared_renderer() -> None:
    checks = (
        (
            "src/trace_tasks/tasks/three_d/room/wall_mounted_rendering.py",
            "render_room_scene_3d",
            {"_draw_wall_object", "_draw_floor_object"},
        ),
        (
            "src/trace_tasks/tasks/three_d/street/shared/rendering.py",
            "render_street_intersection_scene_3d",
            {"_draw_candidate_object", "_draw_context_object"},
        ),
        (
            "src/trace_tasks/tasks/three_d/warehouse/shared/rendering.py",
            "render_warehouse_robot_scene_3d",
            {"_draw_warehouse_object"},
        ),
        (
            "src/trace_tasks/tasks/three_d/warehouse/shared/rendering.py",
            "render_warehouse_robot_nearest_scene_3d",
            {"_draw_warehouse_object", "_draw_reference_object"},
        ),
    )
    for path, function_name, forbidden_calls in checks:
        call_names = _called_names_in_function(path, function_name)
        assert "render_three_d_object" in call_names
        assert call_names.isdisjoint(forbidden_calls)

    preview_source = Path("src/trace_tasks/tasks/three_d/shared/object_inventory_preview.py").read_text()
    assert "render_three_d_object(" in preview_source
    for forbidden in (
        "_draw_room_floor_object",
        "room_scene._draw_wall_object",
        "street_scene._draw_context_object",
        "street_scene._draw_candidate_object",
        "warehouse_scene._draw_warehouse_object",
    ):
        assert forbidden not in preview_source


def test_reusable_three_d_object_renderers_are_shared_owned() -> None:
    shared_paths = (
        "src/trace_tasks/tasks/three_d/shared/room_wall_object_rendering.py",
        "src/trace_tasks/tasks/three_d/shared/room_wall_rendering_geometry.py",
        "src/trace_tasks/tasks/three_d/shared/room_floor_object_rendering.py",
        "src/trace_tasks/tasks/three_d/shared/street_object_rendering_common.py",
        "src/trace_tasks/tasks/three_d/shared/street_object_rendering.py",
        "src/trace_tasks/tasks/three_d/shared/street_vehicle_object_rendering.py",
        "src/trace_tasks/tasks/three_d/shared/street_fixture_object_rendering.py",
        "src/trace_tasks/tasks/three_d/shared/street_pedestrian_object_rendering.py",
        "src/trace_tasks/tasks/three_d/shared/street_landscape_object_rendering.py",
        "src/trace_tasks/tasks/three_d/shared/warehouse_object_rendering.py",
    )
    for path in shared_paths:
        assert Path(path).exists()

    removed_scene_paths = (
        "src/trace_tasks/tasks/three_d/room/wall_mounted_wall_objects.py",
        "src/trace_tasks/tasks/three_d/room/wall_mounted_floor_objects.py",
        "src/trace_tasks/tasks/three_d/street/intersection_object_rendering.py",
        "src/trace_tasks/tasks/three_d/street/intersection_vehicle_rendering.py",
        "src/trace_tasks/tasks/three_d/street/intersection_fixture_rendering.py",
        "src/trace_tasks/tasks/three_d/street/intersection_pedestrian_rendering.py",
        "src/trace_tasks/tasks/three_d/street/intersection_landscape_rendering.py",
        "src/trace_tasks/tasks/three_d/room/wall_mounted_scene.py",
        "src/trace_tasks/tasks/three_d/street/intersection_rendering_common.py",
        "src/trace_tasks/tasks/three_d/shared/room_wall_common.py",
        "src/trace_tasks/tasks/three_d/shared/street_rendering_common.py",
    )
    for path in removed_scene_paths:
        assert not Path(path).exists()

    dispatch_source = Path("src/trace_tasks/tasks/three_d/shared/object_rendering.py").read_text()
    for forbidden in (
        "..room.",
        "..street.",
        "..warehouse.",
        "wall_mounted_wall_objects",
        "wall_mounted_floor_objects",
        "intersection_object_rendering",
        "robot_forward_path import _draw_warehouse_object",
    ):
        assert forbidden not in dispatch_source


def test_scene_support_profiles_are_not_generic_standalone_objects() -> None:
    support_by_type = {
        str(profile.object_type): profile
        for profile in THREE_D_OBJECT_PROFILES
        if str(profile.resource_kind) == "scene_support"
    }
    assert {"building", "store", "office_building", "shelf_rack"}.issubset(support_by_type)
    assert all(str(support_by_type[object_type].resource_kind) == "scene_support" for object_type in support_by_type)


def test_three_d_object_profile_helpers_resolve_scene_role_identity() -> None:
    profile = object_profile(source_scene=OBJECT_SCENE_ID, role="spatial_small_shape", object_type="sphere")

    assert object_profile_by_id(profile.profile_id) is profile
    assert object_profile_or_none(source_scene=OBJECT_SCENE_ID, role="spatial_small_shape", object_type="sphere") is profile
    assert object_profile_or_none(source_scene=OBJECT_SCENE_ID, role="spatial_context_shape", object_type="sphere") is None
    assert profile_display_name(source_scene=OBJECT_SCENE_ID, role="spatial_small_shape", object_type="sphere") == "ball"
    assert profile_dimensions_xyz(source_scene=OBJECT_SCENE_ID, role="spatial_small_shape", object_type="sphere") == (0.48, 0.48, 0.48)
    assert profile.profile_id in scene_profile_ids(source_scene=OBJECT_SCENE_ID, role="spatial_small_shape")


def test_profile_and_placement_build_shared_object_spec() -> None:
    profile = object_profile(source_scene=OBJECT_SCENE_ID, role="spatial_small_shape", object_type="sphere")
    placement = ThreeDPlacementSpec(
        object_id="object_A",
        object_type="sphere",
        shape_type="sphere",
        role="candidate",
        world_xyz=(0.0, 0.0, 0.24),
        base_xyz=(0.0, 0.0, 0.0),
        dimensions_xyz=(0.48, 0.48, 0.48),
        screen_xy=(240.0, 180.0),
        semantic_attributes={"is_answer_candidate": True},
        visual_attributes={"dimension_scale": 1.0},
    )

    spec = object_rendering.ThreeDObjectSpec.from_profile_and_placement(
        profile,
        placement,
        role="candidate",
        source_entity_type="three_d_object_scene_object",
    )

    assert spec.object_id == "object_A"
    assert spec.object_type == "sphere"
    assert spec.public_name == "ball"
    assert spec.canonical_id == profile.canonical_id
    assert spec.renderer_id == "object_scene_shape"
    assert spec.semantic_attributes["shape_type"] == "sphere"
    assert spec.semantic_attributes["is_answer_candidate"] is True
    assert spec.visual_attributes["dimension_scale"] == 1.0


def test_three_d_scene_style_spec_serializes_trace_metadata() -> None:
    style = ThreeDSceneStyleSpec(
        scene_id="object_scene",
        scene_variant="floor_grid_room",
        camera_style_id="oblique_left",
        palette_id="cool_floor",
        surface_style_id="grid_floor",
        colors={"floor": (232, 239, 242)},
        params={"grid_step": 0.8},
    )

    payload = style.as_dict()
    assert payload["scene_id"] == "object_scene"
    assert payload["scene_variant"] == "floor_grid_room"
    assert payload["renderer_style"] == "projected_3d"
    assert payload["colors"]["floor"] == [232, 239, 242]
    assert payload["params"]["grid_step"] == 0.8


def test_object_scene_entities_include_shared_object_record() -> None:
    render_params = object_scene._RenderParams(
        canvas_width=640,
        canvas_height=480,
        scene_margin_left_px=42,
        scene_margin_right_px=42,
        scene_margin_top_px=36,
        scene_margin_bottom_px=42,
        room_extent=3.2,
        room_height=3.0,
        grid_step=0.8,
        marker_radius_px=18,
        label_font_size_px=18,
        line_width_px=2,
        floor_rgb=(232, 239, 242),
        grid_rgb=(184, 197, 207),
        edge_rgb=(93, 108, 124),
        text_rgb=(30, 34, 42),
        text_stroke_rgb=(255, 255, 255),
        full_bleed_floor=False,
        full_bleed_floor_extent_multiplier=3.0,
    )
    spec = object_scene._make_object_spec(
        object_id="object_A",
        shape_type="sphere",
        object_role="candidate",
        xy=(-0.55, 0.25),
        dimensions_xyz=(0.62, 0.62, 0.62),
        dimension_scale=1.0,
        label="A",
    )
    camera = object_scene._sample_camera(random.Random(17), yaw_band_degrees=(48.0, 82.0))
    frame = object_scene._build_projection_frame(
        camera=camera,
        render_params=render_params,
        point_worlds=object_scene._object_reference_points(spec),
    )
    screen = object_scene._project_screen(spec["world_xyz"], camera, frame)
    finalized_spec = dict(spec)
    finalized_spec.update(
        {
            "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
            "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
            "camera_distance": round(float(screen[7]), 4),
        }
    )
    dataset = {
        "scene_variant": "floor_grid_room",
        "camera": {
            "camera_position": list(camera.camera_position),
            "target": list(camera.target),
            "right": list(camera.right),
            "up": list(camera.up),
            "forward": list(camera.forward),
            "yaw_degrees": camera.yaw_degrees,
            "pitch_degrees": camera.pitch_degrees,
            "distance": camera.distance,
        },
        "projection_frame": {
            "scale": frame.scale,
            "center_x": frame.center_x,
            "center_y": frame.center_y,
            "normalized_center_u": frame.normalized_center_u,
            "normalized_center_v": frame.normalized_center_v,
        },
        "point_specs": [finalized_spec],
        "context_object_specs": [],
        "answer_label": "A",
        "answer_point_id": "object_A",
    }

    rendered = object_scene.render_object_scene_3d(
        Image.new("RGB", (render_params.canvas_width, render_params.canvas_height), (255, 255, 255)),
        dataset=dataset,
        render_params=render_params,
        draw_candidate_labels=False,
    )

    candidate = next(entity for entity in rendered.entities if entity["entity_id"] == "object_A")
    record = candidate["attrs"]["object_record"]
    assert record["object_id"] == "object_A"
    assert record["object_type"] == "sphere"
    assert record["canonical_id"] == "ball"
    assert record["bbox"]
    assert record["world_xyz"] == [round(float(value), 3) for value in finalized_spec["world_xyz"]]
    assert record["semantic_attributes"]["shape_type"] == "sphere"
    assert record["semantic_attributes"]["resource_kind"] == "standalone"
    assert record["visual_attributes"]["renderer_id"] == "object_scene_shape"
    assert record["visual_attributes"]["renderer_style"] == "projected_3d"
