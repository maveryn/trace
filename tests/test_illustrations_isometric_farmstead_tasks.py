from __future__ import annotations

from trace_tasks.tasks.registry import TASK_REGISTRY, create_task
from trace_tasks.tasks.illustrations.isometric_farmstead.shared.rendering import (
    SCENE_ID,
    SUPPORTED_LEVELS,
    render_isometric_farmstead_scene,
)
from trace_tasks.tasks.illustrations.isometric_farmstead.terrain_elevation_extremum_label import (
    SUPPORTED_QUERY_IDS as ELEVATION_QUERY_IDS,
    TASK_ID as ELEVATION_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_farmstead.farmer_same_level_tile_label import (
    SUPPORTED_QUERY_IDS as FARMER_SAME_LEVEL_QUERY_IDS,
    TASK_ID as FARMER_SAME_LEVEL_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_farmstead.highest_terrain_tile_count import (
    SUPPORTED_QUERY_IDS as HIGHEST_TILE_COUNT_QUERY_IDS,
    TASK_ID as HIGHEST_TILE_COUNT_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_farmstead.terrain_level_object_count import (
    SUPPORTED_QUERY_IDS as OBJECT_COUNT_QUERY_IDS,
    TARGET_OBJECT_TYPES,
    TASK_ID as OBJECT_COUNT_TASK_ID,
)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= float(width)
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= float(height)


def _assert_no_one_tile_terrace_border_gap(trace: dict) -> None:
    cols = int(trace["grid_cols"])
    rows = int(trace["grid_rows"])
    for rects in trace["level_shapes"].values():
        for x, y, w, h in rects:
            assert int(x) != 1
            assert int(y) != 1
            assert int(cols) - (int(x) + int(w)) != 1
            assert int(rows) - (int(y) + int(h)) != 1


def _assert_no_entity_on_unsafe_tile(scene: object) -> None:
    unsafe_ids = set(str(value) for value in scene.trace["object_unsafe_low_adjacent_higher_tile_ids"])
    for entity in scene.entities:
        assert not unsafe_ids.intersection(str(tile_id) for tile_id in entity.tile_ids)


def _assert_connected_tiles(tile_records: list[dict]) -> None:
    cells = {(int(tile["col"]), int(tile["row"])) for tile in tile_records}
    assert cells
    stack = [next(iter(cells))]
    visited = {stack[0]}
    while stack:
        col, row = stack.pop()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (int(col) + dc, int(row) + dr)
            if neighbor in cells and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    assert visited == cells


def test_isometric_farmstead_renderer_is_deterministic_and_profile_safe() -> None:
    for width, height, profile, expected_grid in (
        (1200, 800, "landscape", (16, 12)),
        (960, 960, "square", (14, 14)),
    ):
        first = render_isometric_farmstead_scene(
            2026062301,
            width=width,
            height=height,
            canvas_profile=profile,
            canvas_profile_probabilities={profile: 1.0},
        )
        second = render_isometric_farmstead_scene(
            2026062301,
            width=width,
            height=height,
            canvas_profile=profile,
            canvas_profile_probabilities={profile: 1.0},
        )
        assert first.image.size == (width, height)
        assert first.image.tobytes() == second.image.tobytes()
        assert first.trace["renderer_id"] == "isometric_farmstead_v0"
        assert first.trace["projection"]["type"] == "2:1_isometric"
        assert (int(first.trace["grid_cols"]), int(first.trace["grid_rows"])) == expected_grid
        assert first.trace["supported_levels"] == list(SUPPORTED_LEVELS)
        active_levels = [int(level) for level in first.trace["levels"]]
        assert active_levels[0] == 0
        assert 1 <= int(first.trace["active_max_level"]) <= 2
        assert active_levels == list(range(0, int(first.trace["active_max_level"]) + 1))
        assert set(first.trace["level_tile_counts"]) == {str(level) for level in active_levels}
        assert all(int(first.trace["level_tile_counts"][str(level)]) > 0 for level in active_levels)
        assert first.trace["layout_family"] not in {"diagonal_ridge", "stepped_hillside"}
        _assert_no_one_tile_terrace_border_gap(first.trace)
        _assert_no_entity_on_unsafe_tile(first)
        assert first.trace["farm_patches"]
        assert first.trace["context_object_counts"]["tree"] >= 1
        assert first.trace["context_object_counts"]["domestic_animal"] >= 1
        assert first.trace["transition_tile_ids"] == []
        assert first.transitions == ()
        assert all(transition.upper_level in active_levels for transition in first.transitions)
        assert all(transition.lower_level in active_levels for transition in first.transitions)
        assert first.trace["eligible_tile_ids"]
        for tile in first.tiles:
            _assert_bbox_inside_canvas(list(tile.bbox_xyxy), width=width, height=height)
        for transition in first.transitions:
            _assert_bbox_inside_canvas(list(transition.bbox_xyxy), width=width, height=height)
        for entity in first.entities:
            _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=width, height=height)


def test_isometric_farmstead_elevation_task_contract() -> None:
    task = create_task(ELEVATION_TASK_ID)
    cases = (
        ("highest_terrain_tile", "landscape", 2026062311),
        ("lowest_terrain_tile", "square", 2026062312),
        ("highest_terrain_tile", "landscape", 2026062313),
        ("lowest_terrain_tile", "square", 2026062314),
    )
    for query_id, profile, seed in cases:
        out = task.generate(
            seed,
            params={"query_id": query_id, "canvas_profile": profile, "candidate_count": 4},
            max_attempts=30,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == query_id
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value in {"A", "B", "C", "D"}
        assert out.annotation_gt.type == "bbox"
        width, height = out.image.size
        _assert_bbox_inside_canvas(list(out.annotation_gt.value), width=width, height=height)
        assert "ground tile" in out.prompt or "terrain tile" in out.prompt

        trace = out.trace_payload
        assert trace["query_spec"]["query_id"] == query_id
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_farmstead_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == SCENE_ID
        assert trace["projected_annotation"]["type"] == "bbox"
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
        assert trace["render_map"]["selected_label"] == out.answer_gt.value
        assert trace["render_map"]["selected_tile_bbox_px"] == out.annotation_gt.value
        assert len(trace["render_map"]["candidate_tile_ids_by_label"]) == 4
        levels = trace["render_map"]["candidate_levels_by_label"]
        selected_level = int(levels[str(out.answer_gt.value)])
        if query_id == "highest_terrain_tile":
            assert selected_level == max(int(value) for value in levels.values())
            assert sum(1 for value in levels.values() if int(value) == selected_level) == 1
        else:
            assert selected_level == min(int(value) for value in levels.values())
            assert sum(1 for value in levels.values() if int(value) == selected_level) == 1


def test_isometric_farmstead_terrain_level_object_count_contract() -> None:
    task = create_task(OBJECT_COUNT_TASK_ID)
    cases = (
        ("highest_terrain_object_count", "domestic_animal", "landscape", 2026062321),
        ("lowest_terrain_object_count", "tree", "square", 2026062322),
        ("highest_terrain_object_count", "tree", "landscape", 2026062323),
        ("lowest_terrain_object_count", "domestic_animal", "square", 2026062324),
    )
    for query_id, target_object_type, profile, seed in cases:
        out = task.generate(
            seed,
            params={
                "query_id": query_id,
                "target_object_type": target_object_type,
                "canvas_profile": profile,
                "answer_count_support": [0, 1, 2, 3, 4, 5],
            },
            max_attempts=50,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert 0 <= int(out.answer_gt.value) <= 5
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        width, height = out.image.size
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(list(bbox), width=width, height=height)
            assert float(bbox[2]) - float(bbox[0]) >= 24.0
            assert float(bbox[3]) - float(bbox[1]) >= 24.0
        assert "highest" in out.prompt or "lowest" in out.prompt
        assert ("farm animals" in out.prompt) if target_object_type == "domestic_animal" else ("trees" in out.prompt)

        trace = out.trace_payload
        assert trace["query_spec"]["query_id"] == query_id
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_farmstead_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == SCENE_ID
        assert trace["render_map"]["target_object_type"] == target_object_type
        assert trace["render_map"]["answer_count"] == int(out.answer_gt.value)
        assert trace["render_map"]["counted_entity_bboxes_px"] == out.annotation_gt.value
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value

        entity_by_id = {str(entity["entity_id"]): entity for entity in trace["scene_ir"]["entities"]}
        unsafe_tile_ids = set(str(value) for value in trace["execution_trace"]["renderer"]["object_unsafe_low_adjacent_higher_tile_ids"])
        for entity in entity_by_id.values():
            assert not unsafe_tile_ids.intersection(str(tile_id) for tile_id in entity["tile_ids"])
        counted_ids = list(trace["render_map"]["counted_entity_ids"])
        assert len(counted_ids) == int(out.answer_gt.value)
        active_levels = [int(level) for level in trace["scene_ir"]["relations"].get("active_levels", trace["render_spec"]["style"]["levels"])]
        expected_level = max(active_levels) if query_id == "highest_terrain_object_count" else min(active_levels)
        assert int(trace["render_map"]["target_level"]) == int(expected_level)
        for entity_id in counted_ids:
            entity = entity_by_id[str(entity_id)]
            assert entity["object_type"] == target_object_type
            assert int(entity["level"]) == int(expected_level)


def test_isometric_farmstead_farmer_same_level_tile_contract() -> None:
    task = create_task(FARMER_SAME_LEVEL_TASK_ID)
    cases = (
        ("landscape", 2026062331),
        ("square", 2026062332),
        ("landscape", 2026062333),
        ("square", 2026062334),
    )
    for profile, seed in cases:
        out = task.generate(
            seed,
            params={"query_id": "single", "canvas_profile": profile, "candidate_count": 4},
            max_attempts=50,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == "single"
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value in {"A", "B", "C", "D"}
        assert out.annotation_gt.type == "bbox"
        width, height = out.image.size
        _assert_bbox_inside_canvas(list(out.annotation_gt.value), width=width, height=height)
        assert "farmer" in out.prompt
        assert "same" in out.prompt or "matches" in out.prompt

        trace = out.trace_payload
        assert trace["query_spec"]["query_id"] == "single"
        assert trace["query_spec"]["internal_query_id"] == "farmer_same_level_tile"
        assert trace["query_spec"]["params"]["internal_query_id"] == "farmer_same_level_tile"
        assert trace["execution_trace"]["query_id"] == "single"
        assert trace["execution_trace"]["internal_query_id"] == "farmer_same_level_tile"
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_farmstead_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == SCENE_ID
        assert trace["projected_annotation"]["type"] == "bbox"
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert trace["render_map"]["selected_label"] == out.answer_gt.value
        assert trace["render_map"]["selected_tile_bbox_px"] == out.annotation_gt.value
        assert len(trace["render_map"]["candidate_tile_ids_by_label"]) == 4

        scene_entities = {str(entity["entity_id"]): entity for entity in trace["scene_ir"]["entities"]}
        farmer = scene_entities["farmer_00"]
        assert farmer["object_type"] == "farmer"
        assert farmer["role"] == "reference"
        assert farmer["metadata"]["role"] == "reference"
        assert trace["render_map"]["reference_farmer_entity_id"] == "farmer_00"
        assert trace["render_map"]["reference_farmer_tile_id"] == farmer["tile_ids"][0]
        assert trace["render_map"]["reference_farmer_bbox_px"] == farmer["bbox"]
        unsafe_tile_ids = set(str(value) for value in trace["execution_trace"]["renderer"]["object_unsafe_low_adjacent_higher_tile_ids"])
        assert str(farmer["tile_ids"][0]) not in unsafe_tile_ids

        candidate_levels = trace["render_map"]["candidate_levels_by_label"]
        farmer_level = int(trace["render_map"]["reference_farmer_level"])
        selected_level = int(candidate_levels[str(out.answer_gt.value)])
        assert selected_level == farmer_level
        assert sum(1 for level in candidate_levels.values() if int(level) == farmer_level) == 1


def test_isometric_farmstead_highest_terrain_tile_count_contract() -> None:
    task = create_task(HIGHEST_TILE_COUNT_TASK_ID)
    cases = (
        ("landscape", 4, 2026062341),
        ("square", 5, 2026062342),
        ("landscape", 7, 2026062343),
        ("square", 8, 2026062344),
    )
    for profile, target_count, seed in cases:
        out = task.generate(
            seed,
            params={"query_id": "single", "canvas_profile": profile, "target_count": target_count},
            max_attempts=30,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == "single"
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == int(target_count)
        assert 4 <= int(out.answer_gt.value) <= 8
        assert out.annotation_gt.type == "bbox"
        width, height = out.image.size
        _assert_bbox_inside_canvas(list(out.annotation_gt.value), width=width, height=height)
        assert "highest" in out.prompt
        assert "tiles" in out.prompt

        trace = out.trace_payload
        assert trace["query_spec"]["query_id"] == "single"
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_farmstead_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == SCENE_ID
        assert trace["projected_annotation"]["type"] == "bbox"
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert trace["render_map"]["answer_count"] == int(out.answer_gt.value)
        assert trace["render_map"]["highest_level_bbox_px"] == out.annotation_gt.value

        target_level = int(trace["render_map"]["target_level"])
        tiles = trace["scene_ir"]["tiles"]
        highest_tiles = [tile for tile in tiles if int(tile["level"]) == target_level]
        assert len(highest_tiles) == int(out.answer_gt.value)
        assert target_level == max(int(tile["level"]) for tile in tiles)
        assert sorted(str(tile["tile_id"]) for tile in highest_tiles) == sorted(trace["render_map"]["counted_tile_ids"])
        _assert_connected_tiles(highest_tiles)

        highest_ids = {str(tile["tile_id"]) for tile in highest_tiles}
        renderer_trace = trace["execution_trace"]["renderer"]
        assert highest_ids == set(str(value) for value in renderer_trace["reserved_highest_level_tile_ids"])
        assert highest_ids.isdisjoint(str(value) for value in renderer_trace["farm_patch_tile_ids"])
        for entity in trace["scene_ir"]["entities"]:
            assert highest_ids.isdisjoint(str(tile_id) for tile_id in entity["tile_ids"])


def test_isometric_farmstead_tasks_registered() -> None:
    assert ELEVATION_TASK_ID in TASK_REGISTRY
    elevation_task_cls = TASK_REGISTRY[ELEVATION_TASK_ID]
    assert tuple(elevation_task_cls.supported_query_ids) == tuple(ELEVATION_QUERY_IDS)
    assert OBJECT_COUNT_TASK_ID in TASK_REGISTRY
    object_count_task_cls = TASK_REGISTRY[OBJECT_COUNT_TASK_ID]
    assert tuple(object_count_task_cls.supported_query_ids) == tuple(OBJECT_COUNT_QUERY_IDS)
    assert tuple(TARGET_OBJECT_TYPES) == ("domestic_animal", "tree")
    assert FARMER_SAME_LEVEL_TASK_ID in TASK_REGISTRY
    farmer_same_level_task_cls = TASK_REGISTRY[FARMER_SAME_LEVEL_TASK_ID]
    assert tuple(farmer_same_level_task_cls.supported_query_ids) == ("single",)
    assert tuple(FARMER_SAME_LEVEL_QUERY_IDS) == ("farmer_same_level_tile",)
    assert HIGHEST_TILE_COUNT_TASK_ID in TASK_REGISTRY
    highest_tile_count_task_cls = TASK_REGISTRY[HIGHEST_TILE_COUNT_TASK_ID]
    assert tuple(highest_tile_count_task_cls.supported_query_ids) == tuple(HIGHEST_TILE_COUNT_QUERY_IDS)
