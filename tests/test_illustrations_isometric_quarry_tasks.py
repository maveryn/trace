from __future__ import annotations

from trace_tasks.tasks.registry import TASK_REGISTRY, create_task
from trace_tasks.tasks.illustrations.isometric_quarry.shared.rendering import (
    OBJECT_COUNT_QUARRY_OBJECT_TYPES,
    SCENE_ID,
    SUPPORTED_LEVELS,
    render_isometric_quarry_scene,
)
from trace_tasks.tasks.illustrations.isometric_quarry.terrain_elevation_extremum_label import (
    SUPPORTED_QUERY_IDS as ELEVATION_QUERY_IDS,
    TASK_ID as ELEVATION_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_quarry.terrain_level_object_count import (
    SUPPORTED_QUERY_IDS as OBJECT_COUNT_QUERY_IDS,
    TARGET_OBJECT_TYPES,
    TASK_ID as OBJECT_COUNT_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_quarry.worker_same_level_tile_label import (
    SUPPORTED_QUERY_IDS as WORKER_SAME_LEVEL_QUERY_IDS,
    TASK_ID as WORKER_SAME_LEVEL_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_quarry.highest_terrain_tile_count import (
    SUPPORTED_QUERY_IDS as HIGHEST_TILE_COUNT_QUERY_IDS,
    TASK_ID as HIGHEST_TILE_COUNT_TASK_ID,
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


def test_isometric_quarry_renderer_is_deterministic_and_profile_safe() -> None:
    for width, height, profile, expected_grid in (
        (1200, 800, "landscape", (16, 12)),
        (960, 960, "square", (14, 14)),
    ):
        first = render_isometric_quarry_scene(
            2026062401,
            width=width,
            height=height,
            canvas_profile=profile,
            canvas_profile_probabilities={profile: 1.0},
        )
        second = render_isometric_quarry_scene(
            2026062401,
            width=width,
            height=height,
            canvas_profile=profile,
            canvas_profile_probabilities={profile: 1.0},
        )
        assert first.image.size == (width, height)
        assert first.image.tobytes() == second.image.tobytes()
        assert first.trace["renderer_id"] == "isometric_quarry_v0"
        assert first.trace["renderer_style"] == "isometric_pixel_quarry"
        assert first.trace["projection"]["type"] == "2:1_isometric"
        assert (int(first.trace["grid_cols"]), int(first.trace["grid_rows"])) == expected_grid
        assert first.trace["supported_levels"] == list(SUPPORTED_LEVELS)
        active_levels = [int(level) for level in first.trace["levels"]]
        assert active_levels[0] == 0
        assert 1 <= int(first.trace["active_max_level"]) <= 2
        assert active_levels == list(range(0, int(first.trace["active_max_level"]) + 1))
        assert first.trace["layout_family"] not in {"diagonal_ridge", "stepped_hillside"}
        _assert_no_one_tile_terrace_border_gap(first.trace)
        _assert_no_entity_on_unsafe_tile(first)
        assert first.trace["quarry_patches"]
        assert first.trace["context_object_counts"]["quarry_object"] >= 4
        assert first.trace["transition_tile_ids"] == []
        assert first.transitions == ()
        assert first.trace["eligible_tile_ids"]
        assert {str(tile.terrain) for tile in first.tiles}.issuperset({"rock"})
        assert all(str(entity.object_type) == "quarry_object" for entity in first.entities)
        for tile in first.tiles:
            _assert_bbox_inside_canvas(list(tile.bbox_xyxy), width=width, height=height)
        for entity in first.entities:
            _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=width, height=height)


def test_isometric_quarry_renderer_required_objects_and_worker_reference() -> None:
    base = render_isometric_quarry_scene(
        2026062402,
        width=1200,
        height=800,
        canvas_profile="landscape",
        canvas_profile_probabilities={"landscape": 1.0},
    )
    reference_tile_id = str(base.trace["eligible_tile_ids"][0])
    scene = render_isometric_quarry_scene(
        2026062402,
        width=1200,
        height=800,
        canvas_profile="landscape",
        canvas_profile_probabilities={"landscape": 1.0},
        required_entity_counts_by_level_type={"ore_vein": {0: 2}, "mine_cart": {1: 2}},
        reference_worker_tile_id=reference_tile_id,
    )
    entities = list(scene.entities)
    assert any(entity.entity_id == "worker_00" for entity in entities)
    worker = next(entity for entity in entities if entity.entity_id == "worker_00")
    assert worker.object_type == "worker"
    assert worker.role == "reference"
    assert worker.metadata["person_variant_id"] == "worker"
    assert scene.trace["reference_worker_tile_id"] == reference_tile_id
    assert scene.trace["context_object_counts"]["worker"] == 1
    ore_on_low = [
        entity
        for entity in entities
        if entity.object_type == "quarry_object"
        and entity.metadata.get("quarry_object_type") == "ore_vein"
        and int(entity.level) == 0
    ]
    carts_on_middle = [
        entity
        for entity in entities
        if entity.object_type == "quarry_object"
        and entity.metadata.get("quarry_object_type") == "mine_cart"
        and int(entity.level) == 1
    ]
    assert len(ore_on_low) == 2
    assert len(carts_on_middle) == 2


def test_isometric_quarry_elevation_task_contract() -> None:
    task = create_task(ELEVATION_TASK_ID)
    cases = (
        ("highest_terrain_tile", "landscape", 2026062411),
        ("lowest_terrain_tile", "square", 2026062412),
        ("highest_terrain_tile", "landscape", 2026062413),
        ("lowest_terrain_tile", "square", 2026062414),
    )
    for query_id, profile, seed in cases:
        out = task.generate(
            seed,
            params={"query_id": query_id, "canvas_profile": profile, "candidate_count": 4},
            max_attempts=40,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == query_id
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value in {"A", "B", "C", "D"}
        assert out.annotation_gt.type == "bbox"
        width, height = out.image.size
        _assert_bbox_inside_canvas(list(out.annotation_gt.value), width=width, height=height)
        assert "quarry" in out.prompt
        assert "ground tile" in out.prompt or "terrain tile" in out.prompt

        trace = out.trace_payload
        assert trace["query_spec"]["query_id"] == query_id
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_quarry_v0"
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


def test_isometric_quarry_terrain_level_object_count_contract() -> None:
    task = create_task(OBJECT_COUNT_TASK_ID)
    cases = (
        ("highest_terrain_object_count", "ore_vein", "landscape", 2026062421),
        ("lowest_terrain_object_count", "mine_cart", "square", 2026062422),
        ("highest_terrain_object_count", "mine_cart", "landscape", 2026062423),
        ("lowest_terrain_object_count", "ore_vein", "square", 2026062424),
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
            max_attempts=60,
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
        assert ("ore veins" in out.prompt) if target_object_type == "ore_vein" else ("mine carts" in out.prompt)

        trace = out.trace_payload
        assert trace["query_spec"]["query_id"] == query_id
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_quarry_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == SCENE_ID
        assert trace["render_map"]["target_object_type"] == target_object_type
        assert trace["render_map"]["answer_count"] == int(out.answer_gt.value)
        assert trace["render_map"]["counted_entity_bboxes_px"] == out.annotation_gt.value
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        renderer_trace = trace["execution_trace"]["renderer"]
        assert renderer_trace["quarry_patch_mode"] == "none"
        assert renderer_trace["quarry_patches"] == []
        assert tuple(renderer_trace["quarry_object_type_pool"]) == tuple(OBJECT_COUNT_QUARRY_OBJECT_TYPES)
        assert set(renderer_trace["context_object_counts"]).issuperset(set(OBJECT_COUNT_QUARRY_OBJECT_TYPES))

        entity_by_id = {str(entity["entity_id"]): entity for entity in trace["scene_ir"]["entities"]}
        counted_ids = list(trace["render_map"]["counted_entity_ids"])
        active_levels = [int(level) for level in trace["render_spec"]["style"]["levels"]]
        expected_level = max(active_levels) if query_id == "highest_terrain_object_count" else min(active_levels)
        assert int(trace["render_map"]["target_level"]) == int(expected_level)
        for entity_id in counted_ids:
            entity = entity_by_id[str(entity_id)]
            assert entity["object_type"] == "quarry_object"
            assert entity["metadata"]["quarry_object_type"] == target_object_type
            assert int(entity["level"]) == int(expected_level)


def test_isometric_quarry_worker_same_level_tile_contract() -> None:
    task = create_task(WORKER_SAME_LEVEL_TASK_ID)
    cases = (
        ("landscape", 2026062431),
        ("square", 2026062432),
        ("landscape", 2026062433),
        ("square", 2026062434),
    )
    for profile, seed in cases:
        out = task.generate(
            seed,
            params={"query_id": "single", "canvas_profile": profile, "candidate_count": 4},
            max_attempts=60,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == "single"
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value in {"A", "B", "C", "D"}
        assert out.annotation_gt.type == "bbox"
        width, height = out.image.size
        _assert_bbox_inside_canvas(list(out.annotation_gt.value), width=width, height=height)
        assert "worker" in out.prompt
        assert "elevation" in out.prompt or "same" in out.prompt or "matches" in out.prompt

        trace = out.trace_payload
        assert trace["query_spec"]["query_id"] == "single"
        assert trace["query_spec"]["prompt_query_key"] == "worker_same_level_tile"
        assert trace["query_spec"]["params"]["internal_query_id"] == "worker_same_level_tile"
        assert trace["execution_trace"]["query_id"] == "single"
        assert trace["execution_trace"]["internal_query_id"] == "worker_same_level_tile"
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_quarry_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == SCENE_ID
        assert trace["projected_annotation"]["type"] == "bbox"
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert trace["render_map"]["selected_label"] == out.answer_gt.value
        assert trace["render_map"]["selected_tile_bbox_px"] == out.annotation_gt.value
        assert len(trace["render_map"]["candidate_tile_ids_by_label"]) == 4

        scene_entities = {str(entity["entity_id"]): entity for entity in trace["scene_ir"]["entities"]}
        worker = scene_entities["worker_00"]
        assert worker["object_type"] == "worker"
        assert worker["role"] == "reference"
        assert worker["metadata"]["role"] == "reference"
        assert trace["render_map"]["reference_worker_entity_id"] == "worker_00"
        assert trace["render_map"]["reference_worker_tile_id"] == worker["tile_ids"][0]
        assert trace["render_map"]["reference_worker_bbox_px"] == worker["bbox"]

        candidate_levels = trace["render_map"]["candidate_levels_by_label"]
        worker_level = int(trace["render_map"]["reference_worker_level"])
        selected_level = int(candidate_levels[str(out.answer_gt.value)])
        assert selected_level == worker_level
        assert sum(1 for level in candidate_levels.values() if int(level) == worker_level) == 1


def test_isometric_quarry_highest_terrain_tile_count_contract() -> None:
    task = create_task(HIGHEST_TILE_COUNT_TASK_ID)
    cases = (
        ("landscape", 4, 2026062441),
        ("square", 5, 2026062442),
        ("landscape", 7, 2026062443),
        ("square", 8, 2026062444),
    )
    for profile, target_count, seed in cases:
        out = task.generate(
            seed,
            params={"query_id": "single", "canvas_profile": profile, "target_count": target_count},
            max_attempts=40,
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
        assert trace["query_spec"]["prompt_query_key"] == "highest_terrain_tile_count"
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_quarry_v0"
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
        assert highest_ids.isdisjoint(str(value) for value in renderer_trace["quarry_patch_tile_ids"])
        for entity in trace["scene_ir"]["entities"]:
            assert highest_ids.isdisjoint(str(tile_id) for tile_id in entity["tile_ids"])


def test_isometric_quarry_task_registered() -> None:
    assert ELEVATION_TASK_ID in TASK_REGISTRY
    elevation_task_cls = TASK_REGISTRY[ELEVATION_TASK_ID]
    assert tuple(elevation_task_cls.supported_query_ids) == tuple(ELEVATION_QUERY_IDS)
    assert OBJECT_COUNT_TASK_ID in TASK_REGISTRY
    object_count_task_cls = TASK_REGISTRY[OBJECT_COUNT_TASK_ID]
    assert tuple(object_count_task_cls.supported_query_ids) == tuple(OBJECT_COUNT_QUERY_IDS)
    assert tuple(TARGET_OBJECT_TYPES) == ("ore_vein", "mine_cart")
    assert WORKER_SAME_LEVEL_TASK_ID in TASK_REGISTRY
    worker_same_level_task_cls = TASK_REGISTRY[WORKER_SAME_LEVEL_TASK_ID]
    assert tuple(worker_same_level_task_cls.supported_query_ids) == tuple(WORKER_SAME_LEVEL_QUERY_IDS)
    assert HIGHEST_TILE_COUNT_TASK_ID in TASK_REGISTRY
    highest_tile_count_task_cls = TASK_REGISTRY[HIGHEST_TILE_COUNT_TASK_ID]
    assert tuple(highest_tile_count_task_cls.supported_query_ids) == tuple(HIGHEST_TILE_COUNT_QUERY_IDS)
