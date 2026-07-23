"""Tests for the synthetic 3D object-cluster count task."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.named_colors import available_named_colors
from trace_tasks.tasks.three_d.shared.object_inventory_preview import render_three_d_object_profile_preview
from trace_tasks.tasks.three_d.shared.object_resources import OBJECT_CLUSTER_EXTRA_SHAPE_TYPES, object_profiles
from trace_tasks.tasks.three_d.shared import object_scene_glyphs_tools_devices as tools_glyphs
from trace_tasks.tasks.three_d.object_cluster.color_membership_count import TASK_ID as COLOR_MEMBERSHIP_COUNT_TASK_ID
from trace_tasks.tasks.three_d.object_cluster.object_type_count_arithmetic import (
    DIFFERENCE_QUERY_ID as OBJECT_TYPE_ARITHMETIC_DIFFERENCE_QUERY_ID,
    TASK_ID as OBJECT_TYPE_COUNT_ARITHMETIC_TASK_ID,
)
from trace_tasks.tasks.three_d.object_cluster.color_count_arithmetic import TASK_ID as COLOR_COUNT_ARITHMETIC_TASK_ID
from trace_tasks.tasks.three_d.object_cluster.counterfactual_count import TASK_ID as COUNTERFACTUAL_COUNT_TASK_ID
from trace_tasks.tasks.three_d.object_cluster.multi_attribute_and_count import TASK_ID as MULTI_ATTRIBUTE_AND_TASK_ID
from trace_tasks.tasks.three_d.object_cluster.multi_attribute_exclusion_count import (
    TASK_ID as MULTI_ATTRIBUTE_EXCLUSION_COUNT_TASK_ID,
)
from trace_tasks.tasks.three_d.object_cluster.multi_attribute_or_count import TASK_ID as MULTI_ATTRIBUTE_OR_COUNT_TASK_ID
from trace_tasks.tasks.three_d.object_cluster.multi_attribute_xor_count import TASK_ID as MULTI_ATTRIBUTE_XOR_COUNT_TASK_ID
from trace_tasks.tasks.three_d.object_cluster.shared.defaults import (
    COLOR_READOUT_CLUSTER_SHAPE_TYPES,
    COLOR_SAFE_CLUSTER_SHAPE_TYPES,
    COUNTABLE_SHAPE_TYPES,
    MAX_RENDERED_CUMULATIVE_OCCLUSION_FRACTION,
    MAX_RENDERED_PAIRWISE_OVERLAP_FRACTION,
    MAX_RENDERED_PAIRWISE_OVERLAP_PX,
    MIN_RENDERED_VISIBLE_BBOX_FRACTION,
    NAMED_CLUSTER_SHAPE_TYPES,
    OBJECT_CLUSTER_ORIENTATION_DEGREES,
    PROMPT_COLOR_RGB,
)
from trace_tasks.tasks.three_d.object_cluster.shared.objects import screen_span_requirements
from trace_tasks.tasks.three_d.object_cluster.shared.relations import semantic_color_label
from trace_tasks.tasks.three_d.object_cluster.object_type_count import TASK_ID
from trace_tasks.tasks.three_d.object_cluster.total_object_count import TASK_ID as TOTAL_OBJECT_COUNT_TASK_ID
from trace_tasks.tasks.three_d.shared.object_confusions import confusable_shape_names
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


COUNTQA_CLUSTER_ADDITIONS = {
    "mini_chair",
    "mini_table",
    "flower",
    "jar",
    "can",
    "lid",
    "tube",
    "clip",
    "socket",
    "chess_piece",
    "light_bulb",
    "egg",
    "paint_brush",
    "ticket",
    "marble",
    "bead",
    "bolt",
    "cushion",
    "stool",
    "tray",
    "coaster",
    "rose",
    "tomato",
    "hook",
    "tape_roll",
    "bag",
}
REMOVED_OBJECT_CLUSTER_SHAPE_TYPES = {"bucket", "chili", "coffee_bean", "dot", "glass", "heater", "paper_clip", "pillow"}
EXPECTED_OBJECT_CLUSTER_NAMED_SHAPES = {
    "anchor",
    "apple",
    "arrow",
    "basket",
    "bell",
    "bottle",
    "bowl",
    "button",
    "cactus",
    "calculator",
    "candle",
    "card",
    "carrot",
    "chess_piece",
    "clock",
    "compass",
    "cone",
    "crown",
    "cube",
    "cup",
    "cylinder",
    "diamond",
    "dice",
    "fish",
    "flower",
    "glove",
    "half_cylinder",
    "hat",
    "heart",
    "helmet",
    "horseshoe",
    "jar",
    "key",
    "kite",
    "lantern",
    "leaf",
    "mail_envelope",
    "mini_chair",
    "mini_table",
    "mushroom",
    "open_book",
    "pencil",
    "plate",
    "plug",
    "puzzle_piece",
    "pyramid",
    "remote_control",
    "ruler",
    "screw",
    "shield",
    "sphere",
    "star_prism",
    "stick",
    "stool",
    "sword",
    "torus",
    "tray",
    "trophy",
    "umbrella",
}
PROMOTED_COLOR_READOUT_SHAPES = {
    "bell",
    "candle",
    "crown",
    "dice",
    "hat",
    "helmet",
    "mini_chair",
    "mini_table",
    "pencil",
    "ruler",
    "trophy",
    "umbrella",
}


def _renderer_function_for_shape(shape_type: str) -> str:
    """Return the projected-object renderer dispatch target for one shape."""

    if str(shape_type) == "cube":
        return "_draw_box_object"
    lines = Path("src/trace_tasks/tasks/three_d/shared/object_rendering.py").read_text().splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        matches_direct = f'shape_type == "{shape_type}"' in stripped
        matches_group = "shape_type in {" in stripped and f'"{shape_type}"' in stripped
        if not (matches_direct or matches_group):
            continue
        block = "\n".join(lines[index : index + 12])
        if "scene_rendering." not in block:
            break
        return block.split("scene_rendering.", 1)[1].split("(", 1)[0]
    raise AssertionError(f"could not resolve renderer dispatch for {shape_type}")


def _renderer_fill_load_count(function_name: str) -> int:
    """Count runtime reads of the semantic ``fill`` argument in one renderer."""

    for path in (
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_primitives.py"),
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_glyphs_symbolic.py"),
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_glyphs_household.py"),
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_glyphs_large_furniture.py"),
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_glyphs_large_stage.py"),
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_glyphs_nature_apparel.py"),
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_glyphs_misc.py"),
        Path("src/trace_tasks/tasks/three_d/shared/object_scene_glyphs_tools_devices.py"),
    ):
        text = path.read_text()
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == str(function_name):
                return sum(
                    isinstance(item, ast.Name) and item.id == "fill" and isinstance(item.ctx, ast.Load)
                    for item in ast.walk(node)
                )
    raise AssertionError(f"could not find renderer function {function_name}")


def _max_pairwise_render_overlap(object_bboxes_px: dict[str, list[float]]) -> tuple[float, float]:
    """Return max bbox overlap as fraction of smaller box and absolute pixels."""

    def area(box: list[float]) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    def intersection(left: list[float], right: list[float]) -> float:
        x0 = max(float(left[0]), float(right[0]))
        y0 = max(float(left[1]), float(right[1]))
        x1 = min(float(left[2]), float(right[2]))
        y1 = min(float(left[3]), float(right[3]))
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    boxes = [list(box) for box in object_bboxes_px.values()]
    max_fraction = 0.0
    max_pixels = 0.0
    for index, left in enumerate(boxes):
        left_area = area(left)
        for right in boxes[index + 1 :]:
            overlap = intersection(left, right)
            max_pixels = max(max_pixels, float(overlap))
            smaller_area = max(1.0, min(left_area, area(right)))
            max_fraction = max(max_fraction, float(overlap) / smaller_area)
    return float(max_fraction), float(max_pixels)


def _min_visible_render_fraction(object_bboxes_px: dict[str, list[float]], *, width: int, height: int) -> float:
    def area(box: list[float]) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    fractions = []
    for box in object_bboxes_px.values():
        full = max(1.0, area(list(box)))
        x0 = max(0.0, float(box[0]))
        y0 = max(0.0, float(box[1]))
        x1 = min(float(width), float(box[2]))
        y1 = min(float(height), float(box[3]))
        visible = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        fractions.append(float(visible) / float(full))
    return float(min(fractions)) if fractions else 1.0


def test_object_cluster_prompt_colors_use_canonical_palette() -> None:
    canonical = {
        str(name): (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        for name, rgb in available_named_colors()
    }

    assert dict(PROMPT_COLOR_RGB) == canonical


def test_object_cluster_total_object_count_answer_and_annotation() -> None:
    task = create_task(TOTAL_OBJECT_COUNT_TASK_ID)
    output = task.generate(
        20260606,
        params={
            "query_id": "single",
            "scene_variant": "shallow_tray",
            "object_count": 14,
            "primary_shape_type": "button",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=240,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    counted_object_ids = [str(object_id) for object_id in trace["counted_object_ids"]]
    object_specs = list(trace["object_specs"])

    assert output.scene_id == "object_cluster"
    assert output.query_id == "single"
    assert trace["internal_query_id"] == "total_object_count"
    object_count_support = {int(value) for value in output.trace_payload["query_spec"]["params"]["object_count_probabilities"]}
    assert min(object_count_support) == 6
    assert max(object_count_support) == 20
    assert trace["cluster_composition_mode"] == "single_type_cluster"
    assert trace["object_count"] == 14
    assert trace["distractor_count"] == 0
    assert output.answer_gt.type == "integer"
    assert output.answer_gt.value == 14
    assert output.annotation_gt.type == "bbox_set"
    assert len(output.annotation_gt.value) == int(output.answer_gt.value)
    assert counted_object_ids == [str(spec["object_id"]) for spec in sorted(object_specs, key=lambda item: str(item["object_id"]))]
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in counted_object_ids]
    assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_bbox_set"] == output.annotation_gt.value
    assert trace["cluster_count"] in {1, 2, 3}
    assert 0.0 <= float(trace["cluster_compactness"]) <= 1.0
    assert len(trace["cluster_layout"]["centers"]) == int(trace["cluster_count"])
    composition_offset = trace["composition_offset"]
    assert str(composition_offset["offset_kind"]) in {"horizontal_edge", "vertical_edge", "corner_bias", "mild"}
    assert -0.30 <= float(composition_offset["dx_frac"]) <= 0.30
    assert -0.26 <= float(composition_offset["dy_frac"]) <= 0.26
    rendered_stats = trace["rendered_layout_stats"]
    assert int(rendered_stats["object_count"]) == 14
    assert int(rendered_stats["center_inside_canvas_count"]) == 14
    canvas_width = int(output.trace_payload["render_spec"]["final_canvas_width"])
    canvas_height = int(output.trace_payload["render_spec"]["final_canvas_height"])
    min_x_span, min_y_span = screen_span_requirements(14, width=canvas_width, height=canvas_height)
    assert float(rendered_stats["center_x_span_px"]) >= min_x_span
    assert float(rendered_stats["center_y_span_px"]) >= min_y_span
    assert _min_visible_render_fraction(render_map["object_bboxes_px"], width=canvas_width, height=canvas_height) >= float(
        MIN_RENDERED_VISIBLE_BBOX_FRACTION
    )
    max_overlap_fraction, max_overlap_pixels = _max_pairwise_render_overlap(render_map["object_bboxes_px"])
    assert max_overlap_fraction <= float(MAX_RENDERED_PAIRWISE_OVERLAP_FRACTION)
    assert max_overlap_pixels <= float(MAX_RENDERED_PAIRWISE_OVERLAP_PX)
    raw_rendered_stats = output.trace_payload["render_spec"]["raw_rendered_layout_stats"]
    assert float(raw_rendered_stats["min_visible_bbox_fraction"]) >= float(MIN_RENDERED_VISIBLE_BBOX_FRACTION)
    assert float(raw_rendered_stats["max_depth_aware_occlusion_fraction"]) <= float(
        MAX_RENDERED_CUMULATIVE_OCCLUSION_FRACTION
    )
    assert float(raw_rendered_stats["min_depth_aware_final_visible_bbox_fraction"]) >= float(
        MIN_RENDERED_VISIBLE_BBOX_FRACTION
    )
    assert all(str(spec["shape_type"]) == "button" for spec in object_specs)
    assert all(str(spec["shape_type"]) in set(NAMED_CLUSTER_SHAPE_TYPES) for spec in object_specs)
    assert all(spec["fill_rgb"] == list(PROMPT_COLOR_RGB[str(spec["color_name"])]) for spec in object_specs)
    assert trace["target_spec"]["color_role"] == "non_semantic_visual_variation"
    assert set(trace["target_spec"]["visual_color_names"]) == set(trace["color_counts"])
    assert trace["target_spec"]["visual_color_counts"] == trace["color_counts"]
    assert 2 <= len(trace["color_counts"]) <= 4
    assert sum(int(count) for count in trace["color_counts"].values()) == 14
    assert all(bool(spec.get("matches_query", False)) for spec in object_specs)
    assert all(bool(spec.get("is_countable_object", False)) for spec in object_specs)
    assert "button" not in output.prompt.lower()
    assert_three_d_canvas_contract(output)


def test_object_cluster_instance_count_answer_and_annotation() -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260601,
        params={
            "query_id": "single",
            "scene_variant": "tabletop_pile",
            "object_count": 20,
            "target_count": 6,
            "target_shape_type": "button",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=240,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    object_specs = list(trace["object_specs"])

    assert output.scene_id == "object_cluster"
    assert output.query_id == "single"
    assert trace["internal_query_id"] == "type_count"
    assert trace["cluster_composition_mode"] == "mixed_type_cluster"
    assert output.answer_gt.type == "integer"
    assert output.answer_gt.value == 6
    assert output.annotation_gt.type == "bbox_set"
    assert len(output.annotation_gt.value) == int(output.answer_gt.value)
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_object_ids]
    assert trace["shape_counts"]["button"] == 6
    assert all(str(spec["shape_type"]) == "button" for spec in object_specs if str(spec["object_id"]) in set(target_object_ids))
    assert all(not bool(spec.get("is_answer_candidate", False)) for spec in object_specs)
    assert all(bool(spec.get("is_countable_object", False)) for spec in object_specs)
    assert all(
        -float(OBJECT_CLUSTER_ORIENTATION_DEGREES) <= float(spec["orientation_deg"]) <= float(OBJECT_CLUSTER_ORIENTATION_DEGREES)
        for spec in object_specs
    )
    assert trace["solver_trace"]["cluster_object_pool_size"] == len(NAMED_CLUSTER_SHAPE_TYPES)
    assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_bbox_set"] == output.annotation_gt.value
    assert_three_d_canvas_contract(output)


def test_object_cluster_instance_count_default_composition_has_distractors() -> None:
    scene_defaults = get_scene_defaults("three_d", "object_cluster")
    gen_defaults, _render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults,
        task_id=TASK_ID,
    )

    assert dict(gen_defaults["composition_mode_weights"]) == {
        "near_homogeneous_cluster": 0.7,
        "mixed_type_cluster": 0.3,
    }

    task = create_task(TASK_ID)
    for seed in range(20260620, 20260626):
        output = task.generate(
            seed,
            params={
                "query_id": "single",
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=240,
        )
        trace = output.trace_payload["execution_trace"]
        assert trace["cluster_composition_mode"] in {"near_homogeneous_cluster", "mixed_type_cluster"}
        assert int(trace["distractor_count"]) > 0


def test_object_cluster_single_type_mode_counts_every_object() -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260602,
        params={
            "query_id": "single",
            "scene_variant": "cluster_mat",
            "composition_mode": "single_type_cluster",
            "target_count": 8,
            "target_shape_type": "stool",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=240,
    )

    trace = output.trace_payload["execution_trace"]
    object_specs = list(trace["object_specs"])

    assert trace["cluster_composition_mode"] == "single_type_cluster"
    assert trace["object_count"] == 8
    assert trace["target_count"] == 8
    assert trace["distractor_count"] == 0
    assert output.answer_gt.value == 8
    assert len(output.annotation_gt.value) == 8
    assert all(str(spec["shape_type"]) == "stool" for spec in object_specs)
    assert all(bool(spec.get("matches_query", False)) for spec in object_specs)


def test_object_cluster_task_registered_in_three_d_taxonomy() -> None:
    ensure_scene_tasks_registered("three_d", "object_cluster")
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in TASK_REGISTRY
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_cluster"
    assert not taxonomy.source_scene_id


def test_object_cluster_total_object_count_registered_in_three_d_taxonomy() -> None:
    ensure_scene_tasks_registered("three_d", "object_cluster")
    taxonomy = resolve_task_taxonomy(TOTAL_OBJECT_COUNT_TASK_ID)

    assert TOTAL_OBJECT_COUNT_TASK_ID in TASK_REGISTRY
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_cluster"
    assert not taxonomy.source_scene_id


def test_object_cluster_countqa_additions_have_profiles_and_render() -> None:
    assert COUNTQA_CLUSTER_ADDITIONS.issubset(set(OBJECT_CLUSTER_EXTRA_SHAPE_TYPES))
    assert REMOVED_OBJECT_CLUSTER_SHAPE_TYPES.isdisjoint(set(OBJECT_CLUSTER_EXTRA_SHAPE_TYPES))
    assert REMOVED_OBJECT_CLUSTER_SHAPE_TYPES.isdisjoint(set(COUNTABLE_SHAPE_TYPES))
    assert REMOVED_OBJECT_CLUSTER_SHAPE_TYPES.isdisjoint(set(COLOR_SAFE_CLUSTER_SHAPE_TYPES))
    assert set(NAMED_CLUSTER_SHAPE_TYPES) == EXPECTED_OBJECT_CLUSTER_NAMED_SHAPES
    assert set(COUNTABLE_SHAPE_TYPES) == EXPECTED_OBJECT_CLUSTER_NAMED_SHAPES
    assert set(COLOR_SAFE_CLUSTER_SHAPE_TYPES) == EXPECTED_OBJECT_CLUSTER_NAMED_SHAPES
    assert "apple" in set(NAMED_CLUSTER_SHAPE_TYPES)
    assert "stick" in set(COUNTABLE_SHAPE_TYPES)
    assert "bucket" not in set(NAMED_CLUSTER_SHAPE_TYPES)
    assert "drum" not in set(NAMED_CLUSTER_SHAPE_TYPES)
    assert "drum" not in set(COUNTABLE_SHAPE_TYPES)
    assert "pencil" in set(COUNTABLE_SHAPE_TYPES)
    assert "pen" not in set(COUNTABLE_SHAPE_TYPES)
    assert "pen" not in set(NAMED_CLUSTER_SHAPE_TYPES)
    profiles = {
        str(profile.object_type): profile
        for profile in object_profiles(source_scene="object_cluster", role="cluster_small_shape")
    }
    assert REMOVED_OBJECT_CLUSTER_SHAPE_TYPES.isdisjoint(set(profiles))

    for shape_type in OBJECT_CLUSTER_EXTRA_SHAPE_TYPES:
        profile = profiles[str(shape_type)]
        assert profile.display_name
        assert profile.dimensions_xyz is not None
        preview = render_three_d_object_profile_preview(
            profile,
            canvas_width=320,
            canvas_height=250,
            instance_seed=20260604,
            crop_to_object=True,
        )
        x0, y0, x1, y1 = preview.object_bbox_px
        assert x1 - x0 > 8.0
        assert y1 - y0 > 8.0


def test_object_cluster_orientation_sensitive_glyphs_use_local_yaw_projection() -> None:
    """Flat cluster glyph bodies must not collapse sampled yaw into axis-aligned boxes."""

    for renderer in (
        tools_glyphs._draw_tray_object,
        tools_glyphs._draw_pillow_cushion_object,
    ):
        source = inspect.getsource(renderer)
        assert "_project_local_xy_rect" in source
        assert "_bbox_from_screen_points(_project_face(list(_object_vertices(spec).values())" not in source

    flat_rect_source = inspect.getsource(tools_glyphs._draw_flat_rect_object)
    card_branch = flat_rect_source.split('if shape_type == "card":', 1)[1].split('if shape_type == "bookmark":', 1)[0]
    assert "_project_local_xy_rect" in card_branch
    assert "_draw_box_object" not in card_branch
    bookmark_branch = flat_rect_source.split('if shape_type == "bookmark":', 1)[1].split("base_fill = {", 1)[0]
    assert "_project_local_xy_rect" in bookmark_branch
    assert "_draw_box_object" not in bookmark_branch
    small_box_branch = flat_rect_source.split('elif shape_type == "small_box":', 1)[1].split('elif shape_type == "towel":', 1)[0]
    assert "(221, 194, 117)" not in small_box_branch
    ticket_source = inspect.getsource(tools_glyphs._draw_ticket_tag_object)
    ticket_branch = ticket_source.split('if shape_type == "ticket":', 1)[1].split("base_fill = _tint", 1)[0]
    assert "_project_local_xy_rect" in ticket_branch
    assert "_draw_box_object" not in ticket_branch


def test_object_cluster_multi_attribute_and_count_answer_and_annotation() -> None:
    task = create_task(MULTI_ATTRIBUTE_AND_TASK_ID)
    output = task.generate(
        20260605,
        params={
            "query_id": "single",
            "scene_variant": "tabletop_pile",
            "object_count": 18,
            "target_count": 4,
            "target_shape_type": "cube",
            "target_color_name": "red",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    object_specs = list(trace["object_specs"])
    expected_ids = [
        str(spec["object_id"])
        for spec in sorted(object_specs, key=lambda item: str(item["object_id"]))
        if str(spec["shape_type"]) == "cube" and str(spec["color_name"]) == "red"
    ]

    assert output.scene_id == "object_cluster"
    assert output.query_id == "single"
    assert trace["internal_query_id"] == "type_and_color_count"
    assert output.answer_gt.type == "integer"
    assert output.answer_gt.value == 4
    assert output.annotation_gt.type == "bbox_set"
    assert target_object_ids == expected_ids
    assert len(output.annotation_gt.value) == int(output.answer_gt.value)
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_object_ids]
    assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_bbox_set"] == output.annotation_gt.value
    assert trace["target_property_phrase"] == "red cubes"
    assert trace["target_spec"]["target_property_prompt_phrase"] == f"{semantic_color_label('red')} cubes"
    assert semantic_color_label("red") in output.prompt
    assert trace["property_counts"]["red_cube"] == 4
    assert all("color_name" in spec and "prompt_color_name" in spec and "fill_rgb" in spec for spec in object_specs)
    assert all(spec["fill_rgb"] == list(PROMPT_COLOR_RGB[str(spec["color_name"])]) for spec in object_specs)
    assert any(str(spec["count_role"]) == "same_type_wrong_color" for spec in object_specs)
    assert any(str(spec["count_role"]) == "same_color_wrong_type" for spec in object_specs)
    assert all(bool(spec["matches_query"]) == (str(spec["shape_type"]) == "cube" and str(spec["color_name"]) == "red") for spec in object_specs)
    assert_three_d_canvas_contract(output)


def test_object_cluster_type_color_distractors_avoid_visually_confusable_shapes() -> None:
    output = create_task(MULTI_ATTRIBUTE_AND_TASK_ID).generate(
        2026062623,
        params={
            "query_id": "single",
            "scene_variant": "tabletop_pile",
            "object_count": 18,
            "target_count": 4,
            "target_shape_type": "card",
            "target_color_name": "red",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )
    trace = output.trace_payload["execution_trace"]
    object_specs = list(trace["object_specs"])
    confusable_shapes = set(confusable_shape_names("card"))

    assert trace["target_shape_type"] == "card"
    assert trace["target_color_name"] == "red"
    assert all(
        str(spec["shape_type"]) not in confusable_shapes
        for spec in object_specs
        if str(spec["shape_type"]) != "card"
    )


def test_object_cluster_multi_attribute_and_count_registered_in_three_d_taxonomy() -> None:
    ensure_scene_tasks_registered("three_d", "object_cluster")
    taxonomy = resolve_task_taxonomy(MULTI_ATTRIBUTE_AND_TASK_ID)

    assert MULTI_ATTRIBUTE_AND_TASK_ID in TASK_REGISTRY
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_cluster"
    assert not taxonomy.source_scene_id
    assert len(COLOR_SAFE_CLUSTER_SHAPE_TYPES) == len(NAMED_CLUSTER_SHAPE_TYPES)
    assert len(COLOR_READOUT_CLUSTER_SHAPE_TYPES) == 39
    assert {
        "sphere",
        "cube",
        "cylinder",
        "puzzle_piece",
        "cup",
        "shield",
        "half_cylinder",
        "button",
        "card",
        "mail_envelope",
        "glove",
        "horseshoe",
        "key",
        "kite",
        "plate",
    }.issubset(set(COLOR_READOUT_CLUSTER_SHAPE_TYPES))
    assert PROMOTED_COLOR_READOUT_SHAPES.issubset(set(COLOR_READOUT_CLUSTER_SHAPE_TYPES))
    assert {
        "marble",
        "bead",
        "dot",
        "ticket",
        "coaster",
        "tape_roll",
        "open_book",
        "apple",
        "clock",
        "calculator",
        "light_bulb",
        "tomato",
        "lantern",
        "flask",
        "bucket",
        "straw",
        "sword",
    }.isdisjoint(set(COLOR_READOUT_CLUSTER_SHAPE_TYPES))
    assert {
        "apple",
        "stick",
        "sword",
        "remote_control",
        "open_book",
        "stool",
        "tray",
    }.issubset(set(COLOR_SAFE_CLUSTER_SHAPE_TYPES))
    for shape_type in COLOR_READOUT_CLUSTER_SHAPE_TYPES:
        renderer_function = _renderer_function_for_shape(str(shape_type))
        assert _renderer_fill_load_count(renderer_function) > 0, (shape_type, renderer_function)


def test_object_cluster_color_membership_count_answer_and_annotation() -> None:
    task = create_task(COLOR_MEMBERSHIP_COUNT_TASK_ID)
    output = task.generate(
        20260611,
        params={
            "query_id": "single",
            "scene_variant": "shallow_tray",
            "object_count": 18,
            "target_count": 5,
            "target_color_name": "blue",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    object_specs = list(trace["object_specs"])
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    expected_ids = [
        str(spec["object_id"])
        for spec in sorted(object_specs, key=lambda item: str(item["object_id"]))
        if str(spec["color_name"]) == "blue"
    ]

    assert output.query_id == "single"
    assert trace["internal_query_id"] == "color_count"
    assert output.answer_gt.value == 5
    assert output.annotation_gt.type == "bbox_set"
    assert target_object_ids == expected_ids
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_object_ids]
    assert all(bool(spec["matches_query"]) == (str(spec["color_name"]) == "blue") for spec in object_specs)
    assert semantic_color_label("blue") in output.prompt
    assert not any(str(spec["color_name"]) in {"cyan", "purple"} for spec in object_specs if str(spec["color_name"]) != "blue")
    assert all(str(spec["shape_type"]) in set(COLOR_READOUT_CLUSTER_SHAPE_TYPES) for spec in object_specs)
    max_overlap_fraction, max_overlap_pixels = _max_pairwise_render_overlap(render_map["object_bboxes_px"])
    assert max_overlap_fraction <= float(MAX_RENDERED_PAIRWISE_OVERLAP_FRACTION)
    assert max_overlap_pixels <= float(MAX_RENDERED_PAIRWISE_OVERLAP_PX)


def test_object_cluster_counterfactual_count_answer_and_starting_annotation() -> None:
    task = create_task(COUNTERFACTUAL_COUNT_TASK_ID)
    output = task.generate(
        20260624,
        params={
            "query_id": "single",
            "scene_variant": "cluster_mat",
            "predicate_kind": "color_object",
            "target_shape_type": "button",
            "target_color_name": "blue",
            "target_count": 4,
            "distractor_count": 8,
            "edit_step_count": 3,
            "edit_amount_max": 2,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    solver_trace = trace["solver_trace"]
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    object_specs = list(trace["object_specs"])
    blue_button_phrase = f"{semantic_color_label('blue')} buttons"

    assert output.scene_id == "object_cluster"
    assert output.query_id == "single"
    assert trace["internal_query_id"] == "attribute_count_after_edits"
    assert output.answer_gt.type == "integer"
    assert trace["target_count"] == 4
    assert solver_trace["initial_target_count"] == 4
    assert solver_trace["counterfactual_step_count"] == 3
    steps = list(solver_trace["counterfactual_steps"])
    assert len(steps) == 3
    assert any(bool(step["affects_target_property"]) for step in steps)
    assert any(not bool(step["affects_target_property"]) for step in steps)
    assert {str(step["predicate_relation_to_target"]) for step in steps} <= {"subset", "disjoint"}
    expected_final_count = int(solver_trace["initial_target_count"]) + sum(int(step["target_delta"]) for step in steps)
    assert expected_final_count == int(solver_trace["final_target_count"])
    assert output.answer_gt.value == expected_final_count
    assert solver_trace["target_delta_total"] == expected_final_count - int(solver_trace["initial_target_count"])
    assert trace["target_spec"]["mode"] == "count_after_edits"
    assert trace["target_spec"]["base_predicate_mode"] == "by_type_and_color"
    assert trace["target_spec"]["target_property_prompt_phrase"] == blue_button_phrase
    assert output.annotation_gt.type == "bbox_set"
    assert len(output.annotation_gt.value) == 4
    assert len(output.annotation_gt.value) != int(output.answer_gt.value)
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_object_ids]
    assert all(
        str(spec["shape_type"]) == "button" and str(spec["color_name"]) == "blue"
        for spec in object_specs
        if str(spec["object_id"]) in set(target_object_ids)
    )
    for step in steps:
        assert f"{int(step['step_index'])}. {step['step_text']}" in output.prompt
    assert blue_button_phrase in output.prompt
    assert_three_d_canvas_contract(output)


def test_object_cluster_multi_attribute_or_count_counts_overlap_once() -> None:
    task = create_task(MULTI_ATTRIBUTE_OR_COUNT_TASK_ID)
    output = task.generate(
        20260612,
        params={
            "query_id": "single",
            "scene_variant": "tabletop_pile",
            "object_count": 20,
            "target_count": 6,
            "target_shape_type": "cube",
            "target_color_name": "red",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )

    trace = output.trace_payload["execution_trace"]
    object_specs = list(trace["object_specs"])
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    expected_ids = [
        str(spec["object_id"])
        for spec in sorted(object_specs, key=lambda item: str(item["object_id"]))
        if str(spec["shape_type"]) == "cube" or str(spec["color_name"]) == "red"
    ]

    assert output.query_id == "single"
    assert trace["internal_query_id"] == "type_or_color_count"
    assert output.answer_gt.value == len(expected_ids) == 6
    assert target_object_ids == expected_ids
    assert len(target_object_ids) == len(set(target_object_ids))
    assert semantic_color_label("red") in output.prompt
    assert any(str(spec["shape_type"]) == "cube" and str(spec["color_name"]) == "red" for spec in object_specs)


def test_object_cluster_multi_attribute_xor_count_excludes_overlap() -> None:
    task = create_task(MULTI_ATTRIBUTE_XOR_COUNT_TASK_ID)
    output = task.generate(
        20260626,
        params={
            "query_id": "single",
            "scene_variant": "tabletop_pile",
            "object_count": 20,
            "target_count": 6,
            "target_shape_type": "cube",
            "target_color_name": "red",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    object_specs = list(trace["object_specs"])
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    expected_ids = [
        str(spec["object_id"])
        for spec in sorted(object_specs, key=lambda item: str(item["object_id"]))
        if (str(spec["shape_type"]) == "cube") ^ (str(spec["color_name"]) == "red")
    ]

    assert output.scene_id == "object_cluster"
    assert output.query_id == "single"
    assert trace["internal_query_id"] == "type_xor_color_count"
    assert output.answer_gt.value == len(expected_ids) == 6
    assert output.annotation_gt.type == "bbox_set"
    assert target_object_ids == expected_ids
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_object_ids]
    assert any(str(spec["count_role"]) == "excluded_overlap" for spec in object_specs)
    assert any(str(spec["shape_type"]) == "cube" and str(spec["color_name"]) == "red" for spec in object_specs)
    assert all(
        bool(spec["matches_query"]) == ((str(spec["shape_type"]) == "cube") ^ (str(spec["color_name"]) == "red"))
        for spec in object_specs
    )
    assert semantic_color_label("red") in output.prompt
    assert "not both" in output.prompt.lower() or "excluding objects that are both" in output.prompt.lower() or "exactly one" in output.prompt.lower()
    assert_three_d_canvas_contract(output)


def test_object_cluster_multi_attribute_exclusion_count_answer_and_annotation() -> None:
    task = create_task(MULTI_ATTRIBUTE_EXCLUSION_COUNT_TASK_ID)
    output = task.generate(
        20260613,
        params={
            "query_id": "type_and_not_color_count",
            "scene_variant": "cluster_mat",
            "object_count": 19,
            "target_count": 5,
            "target_shape_type": "cube",
            "target_color_name": "red",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )

    trace = output.trace_payload["execution_trace"]
    object_specs = list(trace["object_specs"])
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    expected_ids = [
        str(spec["object_id"])
        for spec in sorted(object_specs, key=lambda item: str(item["object_id"]))
        if str(spec["shape_type"]) == "cube" and str(spec["color_name"]) != "red"
    ]

    assert output.query_id == "type_and_not_color_count"
    assert output.answer_gt.value == 5
    assert target_object_ids == expected_ids
    assert output.annotation_gt.type == "bbox_set"
    assert semantic_color_label("red") in output.prompt
    assert any(str(spec["shape_type"]) == "cube" and str(spec["color_name"]) == "red" for spec in object_specs)


def test_object_cluster_count_arithmetic_keyed_operand_annotation() -> None:
    task = create_task(OBJECT_TYPE_COUNT_ARITHMETIC_TASK_ID)
    output = task.generate(
        20260615,
        params={
            "query_id": OBJECT_TYPE_ARITHMETIC_DIFFERENCE_QUERY_ID,
            "scene_variant": "shallow_tray",
            "left_shape_type": "button",
            "right_shape_type": "dice",
            "left_operand_count": 6,
            "right_operand_count": 2,
            "object_count": 18,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=300,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    role_object_ids = trace["role_object_ids"]
    left_ids = [str(object_id) for object_id in role_object_ids["left_operand"]]
    right_ids = [str(object_id) for object_id in role_object_ids["right_operand"]]

    assert output.query_id == OBJECT_TYPE_ARITHMETIC_DIFFERENCE_QUERY_ID
    assert trace["internal_query_id"] == "two_type_difference_count"
    assert output.answer_gt.value == 4
    assert output.annotation_gt.type == "bbox_set_map"
    assert set(output.annotation_gt.value) == {"left_operand", "right_operand"}
    assert len(output.annotation_gt.value["left_operand"]) == 6
    assert len(output.annotation_gt.value["right_operand"]) == 2
    assert output.annotation_gt.value["left_operand"] == [render_map["object_bboxes_px"][object_id] for object_id in left_ids]
    assert output.annotation_gt.value["right_operand"] == [render_map["object_bboxes_px"][object_id] for object_id in right_ids]
    assert output.trace_payload["projected_annotation"]["bbox_set_map"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_bbox_set_map"] == output.annotation_gt.value


def test_object_cluster_first_wave_tasks_registered_in_three_d_taxonomy() -> None:
    ensure_scene_tasks_registered("three_d", "object_cluster")
    for task_id in (
        COLOR_MEMBERSHIP_COUNT_TASK_ID,
        COUNTERFACTUAL_COUNT_TASK_ID,
        MULTI_ATTRIBUTE_OR_COUNT_TASK_ID,
        MULTI_ATTRIBUTE_XOR_COUNT_TASK_ID,
        MULTI_ATTRIBUTE_EXCLUSION_COUNT_TASK_ID,
        OBJECT_TYPE_COUNT_ARITHMETIC_TASK_ID,
        COLOR_COUNT_ARITHMETIC_TASK_ID,
    ):
        taxonomy = resolve_task_taxonomy(task_id)

        assert task_id in TASK_REGISTRY
        assert taxonomy.domain == "three_d"
        assert taxonomy.scene_id == "object_cluster"
        assert not taxonomy.source_scene_id
