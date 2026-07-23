from __future__ import annotations

from trace_tasks.tasks.registry import TASK_REGISTRY, create_task
from trace_tasks.tasks.illustrations.isometric_harbor.boat_side_count import (
    QUERY_TO_SIDE,
    SUPPORTED_QUERY_IDS,
    TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_harbor.boat_mooring_status_count import (
    QUERY_TO_STATUS,
    SUPPORTED_QUERY_IDS as MOORING_SUPPORTED_QUERY_IDS,
    TASK_ID as MOORING_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_harbor.boat_heading_status_count import (
    QUERY_TO_HEADING_STATUS,
    SUPPORTED_QUERY_IDS as HEADING_SUPPORTED_QUERY_IDS,
    TASK_ID as HEADING_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_harbor.shoreline_nearest_boat_label import (
    SUPPORTED_QUERY_IDS as SHORELINE_SUPPORTED_QUERY_IDS,
    TASK_ID as SHORELINE_TASK_ID,
)
from trace_tasks.tasks.illustrations.isometric_harbor.shared.rendering import (
    BOAT_HEADING_STATUS_ORIENTATION,
    DEFAULT_BOAT_CANDIDATE_LABELS,
    RENDERER_ID,
    SCENE_ID,
    render_isometric_harbor_scene,
)
from trace_tasks.tasks.illustrations.isometric_harbor.shared.spatial_primitives import dock_is_connected


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= float(width)
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= float(height)


def _assert_bbox_min_side(bbox: list[float], *, min_side: float = 24.0) -> None:
    assert float(bbox[2]) - float(bbox[0]) >= float(min_side)
    assert float(bbox[3]) - float(bbox[1]) >= float(min_side)


def test_isometric_harbor_renderer_is_deterministic_and_profile_safe() -> None:
    for width, height, profile, expected_grid in (
        (1200, 800, "landscape", (16, 12)),
        (960, 960, "square", (14, 14)),
    ):
        first = render_isometric_harbor_scene(
            2026062501,
            width=width,
            height=height,
            canvas_profile=profile,
            canvas_profile_probabilities={profile: 1.0},
            required_boat_counts_by_side={"left": 5, "right": 0},
        )
        second = render_isometric_harbor_scene(
            2026062501,
            width=width,
            height=height,
            canvas_profile=profile,
            canvas_profile_probabilities={profile: 1.0},
            required_boat_counts_by_side={"left": 5, "right": 0},
        )
        assert first.image.size == (width, height)
        assert first.image.tobytes() == second.image.tobytes()
        assert first.trace["renderer_id"] == RENDERER_ID
        assert first.trace["renderer_style"] == "isometric_pixel_harbor"
        assert first.trace["theme_id"] == "isometric_harbor_shoreline_dock"
        assert first.trace["background_rgb"] == first.trace["background_tone_rgb"]
        assert first.trace["projection"]["type"] == "2:1_isometric"
        assert (int(first.trace["grid_cols"]), int(first.trace["grid_rows"])) == expected_grid
        assert first.trace["boat_counts_by_side"] == {"left": 5, "right": 0}
        assert first.trace["boat_counts_by_mooring_status"] == {"moored": 5, "open_water": 0}
        assert first.trace["context_object_counts"]["boat"] == 5
        assert dock_is_connected(first)
        assert {str(tile.terrain) for tile in first.tiles} == {"dock", "land", "water"}
        assert first.trace["terrain_tile_counts"]["land"] > 0
        assert first.trace["terrain_tile_counts"]["water"] > first.trace["terrain_tile_counts"]["land"]
        assert first.trace["terrain_tile_counts"]["dock"] == len(first.trace["dock_tile_ids"])
        for tile in first.tiles:
            _assert_bbox_inside_canvas(list(tile.bbox_xyxy), width=width, height=height)
        for entity in first.entities:
            _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=width, height=height)


def test_isometric_harbor_renderer_supports_open_water_boats() -> None:
    scene = render_isometric_harbor_scene(
        2026062602,
        width=1200,
        height=800,
        canvas_profile="landscape",
        required_moored_boat_count=3,
        required_open_water_boat_count=6,
    )
    boats = [entity for entity in scene.entities if entity.object_type == "boat"]
    moored = [entity for entity in boats if entity.metadata.get("mooring_status") == "moored"]
    open_water = [entity for entity in boats if entity.metadata.get("mooring_status") == "open_water"]
    assert len(moored) == 3
    assert len(open_water) == 6
    assert scene.trace["boat_counts_by_mooring_status"] == {"moored": 3, "open_water": 6}
    assert {str(entity.metadata.get("orientation")) for entity in open_water}
    for entity in open_water:
        assert "dock_side" not in entity.metadata
        assert len(entity.tile_ids) == 1
        tile = next(tile for tile in scene.tiles if tile.tile_id == entity.tile_ids[0])
        assert tile.terrain == "water"
        _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=scene.image.size[0], height=scene.image.size[1])


def test_isometric_harbor_renderer_supports_shoreline_candidate_boats() -> None:
    scene = render_isometric_harbor_scene(
        2026062801,
        width=1200,
        height=800,
        canvas_profile="landscape",
        shoreline_candidate_labels=DEFAULT_BOAT_CANDIDATE_LABELS,
        shoreline_nearest_label="D",
    )
    boats = [entity for entity in scene.entities if entity.object_type == "boat"]
    assert len(boats) == 6
    assert scene.trace["nearest_label"] == "D"
    assert set(scene.trace["candidate_boat_ids_by_label"]) == set(DEFAULT_BOAT_CANDIDATE_LABELS)
    assert scene.trace["boat_counts_by_side"] == {"left": 0, "right": 0}
    assert scene.trace["boat_counts_by_mooring_status"] == {"moored": 0, "open_water": 6}
    distances = {str(key): int(value) for key, value in scene.trace["shoreline_distance_tiles_by_label"].items()}
    assert distances["D"] == min(distances.values())
    assert list(distances.values()).count(distances["D"]) == 1
    candidate_cols = {
        int(next(tile for tile in scene.tiles if tile.tile_id == entity.tile_ids[0]).col)
        for entity in boats
    }
    assert len(candidate_cols) >= 3
    for entity in boats:
        assert entity.metadata.get("mooring_status") == "open_water"
        assert entity.metadata.get("orientation") == "shore_facing"
        assert entity.metadata.get("shoreline_candidate_label") in DEFAULT_BOAT_CANDIDATE_LABELS
        assert entity.metadata.get("label_bbox_xyxy")
        tile = next(tile for tile in scene.tiles if tile.tile_id == entity.tile_ids[0])
        assert tile.terrain == "water"
        _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=scene.image.size[0], height=scene.image.size[1])


def test_isometric_harbor_renderer_supports_heading_status_boats() -> None:
    counts = {"toward_shoreline": 2, "away_from_shoreline": 4}
    scene = render_isometric_harbor_scene(
        2026062901,
        width=1200,
        height=800,
        canvas_profile="landscape",
        required_heading_status_counts=counts,
    )
    boats = [entity for entity in scene.entities if entity.object_type == "boat"]
    assert len(boats) == 6
    assert scene.trace["boat_counts_by_heading_status"] == counts
    assert scene.trace["boat_counts_by_mooring_status"] == {"moored": 0, "open_water": 6}
    assert scene.trace["boat_counts_by_side"] == {"left": 0, "right": 0}
    heading_cells = [
        next(tile for tile in scene.tiles if tile.tile_id == entity.tile_ids[0])
        for entity in boats
    ]
    assert len({int(tile.col) for tile in heading_cells}) >= 3
    assert len({int(tile.row) for tile in heading_cells}) >= 2
    for entity in boats:
        heading_status = str(entity.metadata.get("heading_status"))
        assert heading_status in counts
        assert entity.metadata.get("orientation") == BOAT_HEADING_STATUS_ORIENTATION[heading_status]
        assert entity.metadata.get("orientation") in {"shore_facing", "shore_away"}
        assert entity.metadata.get("mooring_status") == "open_water"
        tile = next(tile for tile in scene.tiles if tile.tile_id == entity.tile_ids[0])
        assert tile.terrain == "water"
        _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=scene.image.size[0], height=scene.image.size[1])


def test_isometric_harbor_boat_side_count_contract() -> None:
    task = create_task(TASK_ID)
    cases = (
        ("left_side_boat_count", 0, "landscape", "image-left"),
        ("right_side_boat_count", 5, "square", "image-right"),
    )
    for query_id, target_count, profile, prompt_side in cases:
        out = task.generate(
            2026062511 + int(target_count),
            params={"query_id": query_id, "target_count": target_count, "canvas_profile": profile},
            max_attempts=4,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == int(target_count)
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == int(target_count)
        assert prompt_side in out.prompt
        trace = out.trace_payload
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_harbor_v1"
        assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
        assert trace["query_spec"]["params"]["target_side"] == QUERY_TO_SIDE[query_id]
        assert trace["execution_trace"]["answer"] == int(target_count)
        assert trace["render_map"]["answer_count"] == int(target_count)
        assert trace["render_map"]["target_side"] == QUERY_TO_SIDE[query_id]
        assert len(trace["render_map"]["counted_entity_ids"]) == int(target_count)
        assert len(trace["projected_annotation"]["bbox_set"]) == int(target_count)
        if int(target_count) == 0:
            assert out.annotation_gt.value == []
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(list(bbox), width=out.image.size[0], height=out.image.size[1])


def test_isometric_harbor_boat_mooring_status_count_contract() -> None:
    task = create_task(MOORING_TASK_ID)
    cases = (
        ("moored_boat_count", 1, 3, "landscape", "tied along the main dock"),
        ("open_water_boat_count", 6, 2, "square", "open water"),
    )
    for query_id, target_count, other_count, profile, prompt_text in cases:
        out = task.generate(
            2026062711 + int(target_count),
            params={
                "query_id": query_id,
                "target_count": target_count,
                "other_count": other_count,
                "canvas_profile": profile,
            },
            max_attempts=4,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == int(target_count)
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == int(target_count)
        assert prompt_text in out.prompt
        trace = out.trace_payload
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_harbor_v1"
        assert trace["query_spec"]["params"]["target_mooring_status"] == QUERY_TO_STATUS[query_id]
        assert trace["execution_trace"]["answer"] == int(target_count)
        assert trace["render_map"]["answer_count"] == int(target_count)
        assert trace["render_map"]["target_mooring_status"] == QUERY_TO_STATUS[query_id]
        assert len(trace["render_map"]["counted_entity_ids"]) == int(target_count)
        assert len(trace["projected_annotation"]["bbox_set"]) == int(target_count)
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(list(bbox), width=out.image.size[0], height=out.image.size[1])
            _assert_bbox_min_side(list(bbox))


def test_isometric_harbor_boat_heading_status_count_contract() -> None:
    task = create_task(HEADING_TASK_ID)
    cases = (
        ("toward_shoreline_boat_count", 2, "landscape", "toward"),
        ("away_from_shoreline_boat_count", 5, "square", "away"),
    )
    for query_id, target_count, profile, prompt_text in cases:
        out = task.generate(
            2026062911 + int(target_count),
            params={"query_id": query_id, "target_count": target_count, "canvas_profile": profile},
            max_attempts=4,
        )
        assert out.scene_id == SCENE_ID
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == int(target_count)
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == int(target_count)
        assert prompt_text in out.prompt
        trace = out.trace_payload
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_harbor_v1"
        assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
        assert trace["query_spec"]["params"]["target_heading_status"] == QUERY_TO_HEADING_STATUS[query_id]
        assert trace["query_spec"]["params"]["heading_status_counts"][QUERY_TO_HEADING_STATUS[query_id]] == int(target_count)
        assert sum(int(value) for value in trace["query_spec"]["params"]["heading_status_counts"].values()) == 6
        assert set(trace["query_spec"]["params"]["heading_status_counts"]) == {"toward_shoreline", "away_from_shoreline"}
        assert set(trace["query_spec"]["params"]["allowed_heading_statuses"]) == {"toward_shoreline", "away_from_shoreline"}
        assert trace["execution_trace"]["answer"] == int(target_count)
        assert trace["render_map"]["answer_count"] == int(target_count)
        assert trace["render_map"]["target_heading_status"] == QUERY_TO_HEADING_STATUS[query_id]
        assert len(trace["render_map"]["counted_entity_ids"]) == int(target_count)
        assert len(trace["projected_annotation"]["bbox_set"]) == int(target_count)
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(list(bbox), width=out.image.size[0], height=out.image.size[1])


def test_isometric_harbor_shoreline_nearest_boat_label_contract() -> None:
    task = create_task(SHORELINE_TASK_ID)
    out = task.generate(
        2026062811,
        params={"selected_label": "F", "canvas_profile": "square"},
        max_attempts=4,
    )
    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "F"
    assert out.annotation_gt.type == "bbox"
    _assert_bbox_inside_canvas(list(out.annotation_gt.value), width=out.image.size[0], height=out.image.size[1])
    assert "shore" in out.prompt
    trace = out.trace_payload
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_isometric_harbor_v1"
    assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert trace["query_spec"]["params"]["candidate_count"] == 6
    assert trace["query_spec"]["params"]["candidate_label"] == "F"
    assert trace["query_spec"]["params"]["selected_label"] == "F"
    assert trace["execution_trace"]["answer"] == "F"
    assert trace["render_map"]["selected_label"] == "F"
    assert trace["render_map"]["selected_shoreline_distance_tiles"] == min(
        int(value) for value in trace["render_map"]["shoreline_distance_tiles_by_label"].values()
    )
    assert len(trace["render_map"]["candidate_boat_ids_by_label"]) == 6
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["witness_symbolic"]["answer_label"] == "F"


def test_isometric_harbor_task_registered() -> None:
    task = create_task(TASK_ID)
    mooring_task = create_task(MOORING_TASK_ID)
    heading_task = create_task(HEADING_TASK_ID)
    shoreline_task = create_task(SHORELINE_TASK_ID)
    assert TASK_ID in TASK_REGISTRY
    assert MOORING_TASK_ID in TASK_REGISTRY
    assert HEADING_TASK_ID in TASK_REGISTRY
    assert SHORELINE_TASK_ID in TASK_REGISTRY
    assert task.domain == "illustrations"
    assert mooring_task.domain == "illustrations"
    assert heading_task.domain == "illustrations"
    assert shoreline_task.domain == "illustrations"
    assert tuple(task.supported_query_ids) == SUPPORTED_QUERY_IDS
    assert tuple(mooring_task.supported_query_ids) == MOORING_SUPPORTED_QUERY_IDS
    assert tuple(heading_task.supported_query_ids) == HEADING_SUPPORTED_QUERY_IDS
    assert tuple(shoreline_task.supported_query_ids) == SHORELINE_SUPPORTED_QUERY_IDS
