from __future__ import annotations

from collections import Counter
from inspect import getsourcefile
from pathlib import Path

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.illustrations.pixel_village.missing_patch_label import (
    _sample_spec as _sample_missing_patch_spec,
)
from trace_tasks.tasks.illustrations.pixel_village.rotated_tile_label import (
    _sample_spec as _sample_rotated_tile_spec,
)
from trace_tasks.tasks.illustrations.pixel_village.swapped_tile_pair_label import (
    _sample_spec as _sample_swapped_tile_pair_spec,
)
from trace_tasks.tasks.illustrations.pixel_village.shared.sampling import _build_river_side_object_sample
from trace_tasks.tasks.illustrations.shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS


MISSING_PATCH_TASK_ID = "task_illustrations__pixel_village__missing_patch_label"
OBJECT_TASK_ID = "task_illustrations__pixel_village__object_type_count"
PATH_TASK_ID = "task_illustrations__pixel_village__person_path_count"
ROTATED_TILE_TASK_ID = "task_illustrations__pixel_village__rotated_tile_label"
SWAPPED_TILE_PAIR_TASK_ID = "task_illustrations__pixel_village__swapped_tile_pair_label"
TERRITORY_TASK_ID = "task_illustrations__pixel_village__territory_object_count"
RIVER_SIDE_TASK_ID = "task_illustrations__pixel_village__river_side_object_count"
TARGETS = ("building", "person", "tree", "lamp_post", "well", "pond")
RIVER_SIDE_TARGETS = ("building", "person", "tree")
RIVER_SIDES = ("left", "right", "above", "below")
RIVER_SIDE_ORIENTATION = {
    "left": "vertical",
    "right": "vertical",
    "above": "horizontal",
    "below": "horizontal",
}
TERRITORY_TARGETS = {
    "cemetery_grave_marker": ("cemetery_0", "grave marker"),
    "orchard_tree": ("orchard_0", "tree"),
}
TASK_SOURCE_STEMS = {
    MISSING_PATCH_TASK_ID: "missing_patch_label.py",
    OBJECT_TASK_ID: "object_type_count.py",
    PATH_TASK_ID: "person_path_count.py",
    ROTATED_TILE_TASK_ID: "rotated_tile_label.py",
    SWAPPED_TILE_PAIR_TASK_ID: "swapped_tile_pair_label.py",
    TERRITORY_TASK_ID: "territory_object_count.py",
    RIVER_SIDE_TASK_ID: "river_side_object_count.py",
}


def _assert_scene_packaged_task(task_id: str) -> None:
    task = create_task(task_id)
    assert not hasattr(task, "scene_id")
    assert Path(getsourcefile(task.__class__) or "").as_posix().endswith(
        f"src/trace_tasks/tasks/illustrations/pixel_village/{TASK_SOURCE_STEMS[task_id]}"
    )


def _assert_scene_prompt_metadata(trace: dict) -> None:
    prompt_variant = trace["query_spec"]["prompt_variant"]
    assert prompt_variant["prompt_bundle_id"] == "illustrations_pixel_village_v0"
    assert prompt_variant["prompt_scene_id"] == "pixel_village"
    assert trace["query_spec"]["params"]["query_id"] == "single"


def _footprint(tile_xywh: list[int]) -> set[tuple[int, int]]:
    x, y, w, h = [int(value) for value in tile_xywh]
    return {
        (xx, yy)
        for xx in range(x, x + w)
        for yy in range(y, y + h)
    }


def _expanded(tiles: set[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for x, y in tiles:
        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                out.add((x + dx, y + dy))
    return out


def _strictly_on_river_side(entity: dict, *, side: str, river_bounds: dict) -> bool:
    footprint = _footprint(entity["tile_xywh"])
    xs = [x for x, _ in footprint]
    ys = [y for _, y in footprint]
    if side == "left":
        return max(xs) < int(river_bounds["min_x"])
    if side == "right":
        return min(xs) > int(river_bounds["max_x"])
    if side == "above":
        return max(ys) < int(river_bounds["min_y"])
    if side == "below":
        return min(ys) > int(river_bounds["max_y"])
    raise AssertionError(f"unexpected side: {side}")


def _matches_target(entity: dict, target: str) -> bool:
    if target == "building":
        return entity["category"] == "building"
    if target == "person":
        return entity["category"] == "person"
    return entity["public_name"] == target.replace("_", " ")


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def _assert_annotation_inside_canvas(out) -> None:
    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas(bbox, width=width, height=height)


def _assert_keyed_annotation_inside_canvas(out) -> None:
    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        _assert_bbox_inside_canvas(bbox, width=width, height=height)


def _assert_hash_balanced_counts(counts: Counter, expected_keys: set[int]) -> None:
    assert set(counts) == set(expected_keys)
    assert max(counts.values()) - min(counts.values()) <= 15


def test_pixel_village_missing_patch_label_contract() -> None:
    _assert_scene_packaged_task(MISSING_PATCH_TASK_ID)
    out = create_task(MISSING_PATCH_TASK_ID).generate(
        hash64(2026061505, "pixel-village-missing-patch", 0),
        params={"option_count": 4, "correct_index": 2},
        max_attempts=160,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    answer_label = str(out.answer_gt.value)
    render_map = trace["render_map"]
    annotation = out.annotation_gt.value

    assert out.scene_id == "pixel_village"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox_map"
    assert answer_label == "C"
    assert set(annotation) == {"missing_region", "selected_option"}
    assert annotation["missing_region"] == render_map["missing_region_bbox_px"]
    assert annotation["selected_option"] == render_map["selected_option_bbox_px"]
    assert annotation["selected_option"] == render_map["option_bboxes_px_by_label"][answer_label]
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert trace["query_spec"]["params"]["patch_mode"] == "plain"
    assert trace["query_spec"]["params"]["option_labels"] == ["A", "B", "C", "D"]
    assert trace["query_spec"]["params"]["correct_index"] == 2
    assert trace["execution_trace"]["selected_transform"] == "none"
    assert trace["render_spec"]["style"]["patch_frame_style"]["style_id"] == "frameless_illustration"
    assert len(render_map["option_source_crop_boxes_px"]) == 4
    assert render_map["option_source_crop_boxes_px"][2] == render_map["source_crop_box_px"]
    _assert_keyed_annotation_inside_canvas(out)


def test_pixel_village_missing_patch_sampler_covers_options() -> None:
    samples = [
        _sample_missing_patch_spec(
            instance_seed=hash64(2026061505, "pixel-village-missing-patch-sampling", index),
            params={"_sample_cursor": index},
            attempt_index=0,
        )
        for index in range(100)
    ]
    option_counts = Counter(sample.option_count for sample in samples)
    answer_counts = Counter(sample.correct_index for sample in samples)

    _assert_hash_balanced_counts(option_counts, {4, 6})
    assert set(answer_counts) <= set(range(6))
    assert {0, 1, 2, 3} <= set(answer_counts)


def test_pixel_village_rotated_tile_label_contract() -> None:
    _assert_scene_packaged_task(ROTATED_TILE_TASK_ID)
    out = create_task(ROTATED_TILE_TASK_ID).generate(
        hash64(2026061505, "pixel-village-rotated-tile", 0),
        params={"rotation_degrees": 90, "canvas_profile": "landscape"},
        max_attempts=200,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    answer_label = str(out.answer_gt.value)
    tile_bboxes = render_map["tile_bboxes_px_by_label"]
    annotation = out.annotation_gt.value

    assert out.scene_id == "pixel_village"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert answer_label in {"A", "B", "C", "D", "E", "F"}
    assert sorted(tile_bboxes) == ["A", "B", "C", "D", "E", "F"]
    for bbox in tile_bboxes.values():
        assert round(float(bbox[2]) - float(bbox[0]), 3) == round(float(bbox[3]) - float(bbox[1]), 3)
    assert annotation == tile_bboxes[answer_label]
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == annotation
    assert render_map["rotated_tile_bbox_px"] == tile_bboxes[answer_label]
    assert trace["render_spec"]["canvas_size"] == [1296, 864]
    assert trace["render_spec"]["style"]["rotated_grid_style"]["style_id"] == "frameless_illustration"
    assert execution["query_id"] == "single"
    assert execution["answer_label"] == answer_label
    assert execution["rotation_degrees"] == 90
    assert execution["grid_shape"] == [2, 3]
    assert execution["tile_labels"] == ["A", "B", "C", "D", "E", "F"]
    assert trace["query_spec"]["params"]["canvas_profile"] == "landscape"
    assert execution["rotated_tile_index"] in execution["usable_tile_indices"]
    _assert_bbox_inside_canvas(annotation, width=out.image.width, height=out.image.height)


def test_pixel_village_rotated_tile_label_uses_square_3x3_grid() -> None:
    out = create_task(ROTATED_TILE_TASK_ID).generate(
        hash64(2026061505, "pixel-village-rotated-tile-square", 0),
        params={"rotation_degrees": 270, "canvas_profile": "square"},
        max_attempts=200,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]

    assert trace["render_spec"]["canvas_size"] == [1008, 1008]
    assert execution["grid_shape"] == [3, 3]
    assert execution["tile_labels"] == ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    assert sorted(render_map["tile_bboxes_px_by_label"]) == ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    assert out.answer_gt.value in render_map["tile_bboxes_px_by_label"]


def test_pixel_village_rotated_tile_sampler_covers_rotation_support() -> None:
    samples = [
        _sample_rotated_tile_spec(
            instance_seed=hash64(2026061505, "pixel-village-rotated-tile-sampling", index),
            params={},
            attempt_index=0,
        )
        for index in range(100)
    ]
    _assert_hash_balanced_counts(Counter(sample.rotation_degrees for sample in samples), {90, 270})


def test_pixel_village_swapped_tile_pair_label_contract() -> None:
    _assert_scene_packaged_task(SWAPPED_TILE_PAIR_TASK_ID)
    out = create_task(SWAPPED_TILE_PAIR_TASK_ID).generate(
        hash64(2026061604, "pixel-village-swapped-tile-pair", 0),
        params={"correct_index": 2, "canvas_profile": "landscape"},
        max_attempts=240,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    render_map = trace["render_map"]
    params = trace["query_spec"]["params"]
    answer_label = str(out.answer_gt.value)
    annotation = out.annotation_gt.value
    option_pairs = render_map["option_pairs_by_label"]
    source_width, source_height = [int(value) for value in render_map["source_size"]]
    canvas_width, canvas_height = [int(value) for value in trace["render_spec"]["canvas_size"]]

    assert out.scene_id == "pixel_village"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox_set"
    assert answer_label == "C"
    assert sorted(render_map["option_bboxes_px_by_label"]) == ["A", "B", "C", "D"]
    assert len(annotation) == 2
    assert sorted(annotation) == sorted(render_map["swapped_cell_bboxes_px"])
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert sorted(trace["projected_annotation"]["bbox_set"]) == sorted(annotation)
    assert trace["projected_annotation"]["pixel_bbox_set"] == trace["projected_annotation"]["bbox_set"]
    assert option_pairs[answer_label] == render_map["swapped_cell_numbers"]
    assert params["option_pairs_by_label"][answer_label] == render_map["swapped_cell_numbers"]
    assert params["swapped_pair_indices"] == render_map["swapped_pair_indices"]
    assert params["candidate_pair_count"] >= 4
    assert params["grid_shape"] == [3, 3]
    assert source_width % 3 == 0
    assert source_height % 3 == 0
    assert source_width % 48 == 0
    assert source_height % 48 == 0
    assert canvas_width * canvas_height <= MAX_RECONSTRUCTION_OUTPUT_PIXELS
    assert trace["render_spec"]["style"]["swapped_grid_style"]["style_id"] == "frameless_illustration"
    assert "swapped" in out.prompt.lower()
    assert "tile numbers" in out.prompt.lower() or "numbered cells" in out.prompt.lower()
    _assert_annotation_inside_canvas(out)


def test_pixel_village_swapped_tile_pair_sampler_covers_answer_labels() -> None:
    samples = [
        _sample_swapped_tile_pair_spec(
            instance_seed=hash64(2026061604, "pixel-village-swapped-tile-pair-sampling", index),
            params={"_sample_cursor": index},
            attempt_index=0,
        )
        for index in range(100)
    ]
    assert Counter(sample.correct_index for sample in samples) == Counter({0: 25, 1: 25, 2: 25, 3: 25})


def test_pixel_village_object_type_count_targets_are_metadata_grounded() -> None:
    _assert_scene_packaged_task(OBJECT_TASK_ID)
    task = create_task(OBJECT_TASK_ID)
    for index, target in enumerate(TARGETS):
        out = task.generate(
            hash64(20260609, OBJECT_TASK_ID, index),
            params={"target_object": target},
            max_attempts=80,
        )
        trace = out.trace_payload
        _assert_scene_prompt_metadata(trace)
        params = trace["query_spec"]["params"]
        counted_ids = trace["execution_trace"]["counted_entity_ids"]
        entity_bboxes = trace["render_map"]["entity_bboxes_px"]
        entities = {entity["entity_id"]: entity for entity in trace["scene_ir"]["entities"]}

        assert out.scene_id == "pixel_village"
        assert out.query_id == "single"
        assert params["target_object"] == target
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert int(out.answer_gt.value) == len(counted_ids) == len(out.annotation_gt.value)
        assert int(out.answer_gt.value) > 0
        assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_entity_bboxes_px"])
        assert len(trace["render_map"]["counted_entity_bboxes_px"]) == len(counted_ids)
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert "villager" not in out.prompt.lower()
        if target == "tree":
            assert trace["query_spec"]["params"]["render_constraints"]["cemetery_mode"] == "none"
            assert trace["query_spec"]["params"]["renderer"]["cemetery_present"] is False
            assert all(entity["public_name"] != "dead tree" for entity in entities.values())

        for entity_id in counted_ids:
            entity = entities[entity_id]
            if target == "building":
                assert entity["category"] == "building"
            elif target == "person":
                assert entity["category"] == "person"
            else:
                assert entity["public_name"] == target.replace("_", " ")


def test_pixel_village_person_path_count_uses_path_tile_intersection() -> None:
    _assert_scene_packaged_task(PATH_TASK_ID)
    out = create_task(PATH_TASK_ID).generate(
        hash64(20260609, PATH_TASK_ID, 0),
        params={"path_person_count": 4},
        max_attempts=40,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    counted_ids = trace["execution_trace"]["counted_entity_ids"]
    entity_bboxes = trace["render_map"]["entity_bboxes_px"]
    entities = {entity["entity_id"]: entity for entity in trace["scene_ir"]["entities"]}
    path_tiles = {tuple(int(value) for value in tile) for tile in trace["execution_trace"]["path_tiles"]}
    path_clearance = int(trace["execution_trace"]["background_person_path_clearance"])
    path_neighborhood = _expanded(path_tiles, path_clearance)

    assert out.scene_id == "pixel_village"
    assert out.query_id == "single"
    assert trace["query_spec"]["params"]["path_person_count"] == 4
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == 4
    assert len(counted_ids) == 4
    assert path_clearance == 1
    assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_entity_bboxes_px"])
    assert len(trace["render_map"]["counted_entity_bboxes_px"]) == len(counted_ids)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    assert "villager" not in out.prompt.lower()
    assert "path tiles" not in out.prompt.lower()

    for entity_id in counted_ids:
        entity = entities[entity_id]
        assert entity["category"] == "person"
        assert _footprint(entity["tile_xywh"]) & path_tiles

    for entity_id, entity in entities.items():
        if entity_id in set(counted_ids) or entity["category"] != "person":
            continue
        assert not (_footprint(entity["tile_xywh"]) & path_neighborhood)


def test_pixel_village_territory_object_count_targets_are_metadata_grounded() -> None:
    _assert_scene_packaged_task(TERRITORY_TASK_ID)
    task = create_task(TERRITORY_TASK_ID)
    for index, (target, (territory_id, public_name)) in enumerate(TERRITORY_TARGETS.items()):
        out = task.generate(
            hash64(20260609, TERRITORY_TASK_ID, index),
            params={"territory_object": target},
            max_attempts=120,
        )
        trace = out.trace_payload
        _assert_scene_prompt_metadata(trace)
        params = trace["query_spec"]["params"]
        counted_ids = trace["execution_trace"]["counted_entity_ids"]
        entity_bboxes = trace["render_map"]["entity_bboxes_px"]
        entities = {entity["entity_id"]: entity for entity in trace["scene_ir"]["entities"]}

        assert out.scene_id == "pixel_village"
        assert out.query_id == "single"
        assert params["territory_object"] == target
        assert params["territory_id"] == territory_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert 0 < int(out.answer_gt.value) <= 8
        assert int(out.answer_gt.value) == len(counted_ids) == len(out.annotation_gt.value)
        assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_entity_bboxes_px"])
        assert len(trace["render_map"]["counted_entity_bboxes_px"]) == len(counted_ids)
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value

        for entity_id in counted_ids:
            entity = entities[entity_id]
            assert entity["public_name"] == public_name
            assert entity["metadata"]["territory_id"] == territory_id


def test_pixel_village_river_side_object_count_uses_strict_tile_side_membership() -> None:
    _assert_scene_packaged_task(RIVER_SIDE_TASK_ID)
    task = create_task(RIVER_SIDE_TASK_ID)
    cases = [(target, side) for target in RIVER_SIDE_TARGETS for side in RIVER_SIDES]
    for index, (target, side) in enumerate(cases):
        out = task.generate(
            hash64(20260610, RIVER_SIDE_TASK_ID, index),
            params={"target_object": target, "river_side": side, "target_count": 1},
            max_attempts=200,
        )
        trace = out.trace_payload
        _assert_scene_prompt_metadata(trace)
        params = trace["query_spec"]["params"]
        counted_ids = trace["execution_trace"]["counted_entity_ids"]
        entity_bboxes = trace["render_map"]["entity_bboxes_px"]
        entities = {entity["entity_id"]: entity for entity in trace["scene_ir"]["entities"]}
        river_bounds = trace["execution_trace"]["river_bounds"]

        assert out.scene_id == "pixel_village"
        assert out.query_id == "single"
        assert params["target_object"] == target
        assert params["river_side"] == side
        assert params["river_orientation"] == RIVER_SIDE_ORIENTATION[side]
        assert params["requested_target_count"] == 1
        assert params["target_count"] == 1
        assert params["render_constraints"]["river_mode"] == "force"
        assert params["render_constraints"]["river_placement"] == "balanced"
        assert trace["query_spec"]["params"]["renderer"]["river_present"] is True
        assert trace["query_spec"]["params"]["renderer"]["river_orientation"] == RIVER_SIDE_ORIENTATION[side]
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert 0 < int(out.answer_gt.value) <= 8
        assert int(out.answer_gt.value) == len(counted_ids) == len(out.annotation_gt.value)
        assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_entity_bboxes_px"])
        assert len(trace["render_map"]["counted_entity_bboxes_px"]) == len(counted_ids)
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert "villager" not in out.prompt.lower()
        assert "tile" not in out.prompt.lower()
        assert "water" not in out.prompt.lower()

        expected_ids = sorted(
            entity_id
            for entity_id, entity in entities.items()
            if _matches_target(entity, target)
            and _strictly_on_river_side(entity, side=side, river_bounds=river_bounds)
        )
        assert counted_ids == expected_ids

        if target == "tree":
            assert params["render_constraints"]["cemetery_mode"] == "none"
            assert params["renderer"]["cemetery_present"] is False
            assert all(entity["public_name"] != "dead tree" for entity in entities.values())


def test_pixel_village_river_side_object_count_sampler_cycles_target_counts() -> None:
    samples = [
        _build_river_side_object_sample(
            instance_seed=hash64(20260610, RIVER_SIDE_TASK_ID, index),
            params={"_sample_cursor": index, "target_count_support": [1, 2, 3, 4, 5]},
            defaults={},
            namespace=RIVER_SIDE_TASK_ID,
        )
        for index in range(20)
    ]
    counts = Counter(sample.target_count for sample in samples)

    assert counts == Counter({1: 4, 2: 4, 3: 4, 4: 4, 5: 4})
    assert all(sample.target_count_support == (1, 2, 3, 4, 5) for sample in samples)
    assert all(sample.target_count_probabilities == {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2} for sample in samples)
