from __future__ import annotations

from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.illustrations.rpg_house.adjacent_room_count import TASK_ID as ADJACENT_COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_house.missing_patch_label import TASK_ID as MISSING_PATCH_TASK_ID
from trace_tasks.tasks.illustrations.rpg_house.reachable_room_count import TASK_ID as REACHABLE_COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_house.room_count import TASK_ID as ROOM_COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_house.swapped_tile_pair_label import TASK_ID as SWAPPED_TILE_PAIR_TASK_ID
from trace_tasks.tasks.illustrations.rpg_house.shared.relations import adjacent_room_ids, reachable_room_ids
from trace_tasks.tasks.illustrations.rpg_house.shared.rendering import (
    MAX_ROOM_COUNT,
    MIN_ROOM_COUNT,
    THEMES,
    draw_rpg_house_debug_overlay,
    render_rpg_house_scene,
)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= float(width)
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= float(height)


def _shared_wall_overlap(room_a: tuple[int, int, int, int], room_b: tuple[int, int, int, int], *, orientation: str) -> tuple[int, int]:
    ax, ay, aw, ah = room_a
    bx, by, bw, bh = room_b
    if orientation == "vertical":
        return max(ay, by), min(ay + ah, by + bh)
    return max(ax, bx), min(ax + aw, bx + bw)


def _rgb_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return sum((int(a) - int(b)) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5


def test_rpg_house_theme_doors_contrast_with_walls() -> None:
    for theme_id, theme in THEMES.items():
        wall_rgb = theme["wall_rgb"]
        door_rgb = theme["door_rgb"]
        door_outline_rgb = theme["door_outline_rgb"]
        assert _rgb_distance(wall_rgb, door_rgb) >= 45.0, theme_id
        assert _rgb_distance(wall_rgb, door_outline_rgb) >= 55.0, theme_id


def test_rpg_house_renderer_is_deterministic_and_profile_safe() -> None:
    for width, height in ((1296, 864), (1008, 1008), (864, 1296)):
        first = render_rpg_house_scene(
            12345,
            width=width,
            height=height,
            room_count=6,
        )
        second = render_rpg_house_scene(
            12345,
            width=width,
            height=height,
            room_count=6,
        )
        assert first.image.size == (width, height)
        assert list(first.image.getdata()) == list(second.image.getdata())
        assert draw_rpg_house_debug_overlay(first).size == first.image.size
        assert len(first.rooms) == 6
        assert MIN_ROOM_COUNT <= len(first.rooms) <= MAX_ROOM_COUNT
        assert len(first.doors) >= len(first.rooms) - 1
        assert len(first.entities) >= len(first.rooms)
        assert min(float(room.bbox_xyxy[0]) for room in first.rooms) == 0.0
        assert min(float(room.bbox_xyxy[1]) for room in first.rooms) == 0.0
        assert max(float(room.bbox_xyxy[2]) for room in first.rooms) == float(width)
        assert max(float(room.bbox_xyxy[3]) for room in first.rooms) == float(height)
        rooms_by_id = {room.room_id: room for room in first.rooms}
        for room in first.rooms:
            _assert_bbox_inside_canvas(list(room.bbox_xyxy), width=width, height=height)
        wide_door_seen = False
        for door in first.doors:
            _assert_bbox_inside_canvas(list(door.bbox_xyxy), width=width, height=height)
            door_width = float(door.bbox_xyxy[2]) - float(door.bbox_xyxy[0])
            door_height = float(door.bbox_xyxy[3]) - float(door.bbox_xyxy[1])
            if door.orientation == "vertical":
                assert door_width <= first.trace["tile_px"] * 1.05
            else:
                assert door_height <= first.trace["tile_px"] * 1.05
            assert min(door_width, door_height) >= first.trace["tile_px"] * 0.25
            assert max(door_width, door_height) >= first.trace["tile_px"] * 0.70
            overlap0, overlap1 = _shared_wall_overlap(
                rooms_by_id[door.room_a_id].tile_xywh,
                rooms_by_id[door.room_b_id].tile_xywh,
                orientation=door.orientation,
            )
            along_start = door.tile_xy[1] if door.orientation == "vertical" else door.tile_xy[0]
            span_tiles = int(door.metadata.get("span_tiles", 1))
            overlap_tiles = overlap1 - overlap0
            assert 2 <= span_tiles <= 3
            assert overlap_tiles >= 4
            assert span_tiles / overlap_tiles <= 0.5
            if overlap_tiles <= 6:
                assert span_tiles == 2
            assert along_start > overlap0
            assert along_start + span_tiles < overlap1
            wide_door_seen = True
            assert max(door_width, door_height) >= first.trace["tile_px"] * 1.70
        assert wide_door_seen
        for entity in first.entities:
            _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=width, height=height)

    player_probe = render_rpg_house_scene(90125, width=960, height=720, room_count=5)
    player_scene = render_rpg_house_scene(
        90125,
        width=960,
        height=720,
        room_count=5,
        player_room_id=player_probe.rooms[0].room_id,
    )
    player_entities = [entity for entity in player_scene.entities if entity.public_name == "player"]
    assert len(player_entities) == 1
    assert player_entities[0].metadata["role"] == "reference"


def test_rpg_house_renderer_samples_room_count_range() -> None:
    seen_counts = set()
    for seed in range(30):
        scene = render_rpg_house_scene(5000 + seed, width=960, height=720)
        seen_counts.add(len(scene.rooms))
        assert MIN_ROOM_COUNT <= len(scene.rooms) <= MAX_ROOM_COUNT
    assert seen_counts == set(range(MIN_ROOM_COUNT, MAX_ROOM_COUNT + 1))


def test_rpg_house_room_count_contract() -> None:
    task = create_task(ROOM_COUNT_TASK_ID)
    out = task.generate(
        2026061603,
        params={
            "canvas_profile": "square",
            "room_count": 7,
        },
        max_attempts=20,
    )
    assert out.scene_id == "rpg_house"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 7
    assert "red-outlined" not in out.prompt
    assert "lettered candidate" not in out.prompt
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == 7
    width, height = out.image.size
    for point in out.annotation_gt.value:
        assert 0 <= float(point[0]) <= float(width)
        assert 0 <= float(point[1]) <= float(height)
    trace = out.trace_payload
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_house_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_house"
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["render_map"]["counted_room_count"] == 7
    assert len(trace["render_map"]["counted_room_ids"]) == 7
    door_states = {str(door["state"]) for door in trace["execution_trace"]["renderer"]["doors"]}
    assert trace["execution_trace"]["renderer"]["door_state_policy"] == "mixed"
    assert {"open", "closed"}.issubset(door_states)


def test_rpg_house_reachable_room_count_contract() -> None:
    task = create_task(REACHABLE_COUNT_TASK_ID)
    out = task.generate(
        2026061607,
        params={
            "canvas_profile": "portrait",
            "room_count": 6,
            "reachable_room_count": 2,
        },
        max_attempts=20,
    )
    assert out.scene_id == "rpg_house"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 2
    assert out.annotation_gt.type == "point_set_map"
    assert sorted(out.annotation_gt.value) == ["player", "reachable_rooms"]
    assert len(out.annotation_gt.value["player"]) == 1
    assert len(out.annotation_gt.value["reachable_rooms"]) == 2
    width, height = out.image.size
    for points in out.annotation_gt.value.values():
        for point in points:
            assert 0 <= float(point[0]) <= float(width)
            assert 0 <= float(point[1]) <= float(height)
    trace = out.trace_payload
    assert trace["projected_annotation"]["type"] == "point_set_map"
    assert trace["projected_annotation"]["point_set_map"] == out.annotation_gt.value
    assert trace["render_map"]["reachable_count"] == 2
    assert trace["render_map"]["player_room_id"] == trace["query_spec"]["params"]["player_room_id"]
    entities = trace["scene_ir"]["entities"]
    players = [entity for entity in entities if entity["public_name"] == "player"]
    assert len(players) == 1
    assert players[0]["room_id"] == trace["render_map"]["player_room_id"]
    doors = trace["scene_ir"]["doors"]
    reachable = set(
        reachable_room_ids(
            tuple(
                type(
                    "Door",
                    (),
                    {
                        "room_a_id": door["room_a_id"],
                        "room_b_id": door["room_b_id"],
                        "door_id": door["door_id"],
                        "state": door["state"],
                    },
                )()
                for door in doors
            ),
            start_room_id=trace["render_map"]["player_room_id"],
        )
    )
    reachable.discard(trace["render_map"]["player_room_id"])
    assert sorted(reachable) == sorted(trace["render_map"]["reachable_room_ids"])

    zero = task.generate(
        2026061608,
        params={"canvas_profile": "landscape", "room_count": 5, "reachable_room_count": 0},
        max_attempts=20,
    )
    assert zero.answer_gt.value == 0
    assert zero.annotation_gt.value["reachable_rooms"] == []


def test_rpg_house_adjacent_room_count_contract() -> None:
    task = create_task(ADJACENT_COUNT_TASK_ID)
    out = task.generate(
        2026061609,
        params={
            "canvas_profile": "square",
            "room_count": 6,
            "adjacent_room_count": 2,
        },
        max_attempts=80,
    )
    assert out.scene_id == "rpg_house"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 2
    assert out.annotation_gt.type == "point_set_map"
    assert sorted(out.annotation_gt.value) == ["adjacent_rooms", "player"]
    assert len(out.annotation_gt.value["player"]) == 1
    assert len(out.annotation_gt.value["adjacent_rooms"]) == 2
    width, height = out.image.size
    for points in out.annotation_gt.value.values():
        for point in points:
            assert 0 <= float(point[0]) <= float(width)
            assert 0 <= float(point[1]) <= float(height)
    trace = out.trace_payload
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_house_v0"
    assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_house"
    assert trace["projected_annotation"]["type"] == "point_set_map"
    assert trace["projected_annotation"]["point_set_map"] == out.annotation_gt.value
    assert trace["render_map"]["adjacent_count"] == 2
    assert trace["render_map"]["player_room_id"] == trace["query_spec"]["params"]["player_room_id"]
    entities = trace["scene_ir"]["entities"]
    players = [entity for entity in entities if entity["public_name"] == "player"]
    assert len(players) == 1
    assert players[0]["room_id"] == trace["render_map"]["player_room_id"]
    doors = trace["scene_ir"]["doors"]
    adjacent = set(
        adjacent_room_ids(
            tuple(
                type(
                    "Door",
                    (),
                    {
                        "room_a_id": door["room_a_id"],
                        "room_b_id": door["room_b_id"],
                        "door_id": door["door_id"],
                        "state": door["state"],
                    },
                )()
                for door in doors
            ),
            start_room_id=trace["render_map"]["player_room_id"],
        )
    )
    assert sorted(adjacent) == sorted(trace["render_map"]["adjacent_room_ids"])


def test_rpg_house_missing_patch_label_contract() -> None:
    task = create_task(MISSING_PATCH_TASK_ID)
    out = task.generate(
        2026061711,
        params={
            "canvas_profile": "landscape",
            "source_room_count": 8,
            "option_count": 4,
            "correct_index": 2,
        },
        max_attempts=200,
    )
    assert out.scene_id == "rpg_house"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value) == ["missing_region", "selected_option"]
    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        _assert_bbox_inside_canvas(bbox, width=width, height=height)
    trace = out.trace_payload
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_house_v0"
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["render_map"]["selected_option_bbox_px"] == out.annotation_gt.value["selected_option"]
    assert trace["render_map"]["missing_region_bbox_px"] == out.annotation_gt.value["missing_region"]
    assert trace["render_map"]["candidate_crop_count"] > 0


def test_rpg_house_swapped_tile_pair_label_contract() -> None:
    task = create_task(SWAPPED_TILE_PAIR_TASK_ID)
    out = task.generate(
        2026061713,
        params={
            "canvas_profile": "landscape",
            "source_room_count": 8,
            "correct_index": 1,
        },
        max_attempts=200,
    )
    assert out.scene_id == "rpg_house"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "B"
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 2
    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas(bbox, width=width, height=height)
    trace = out.trace_payload
    assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_house_v0"
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["render_map"]["swapped_cell_bboxes_px"] == out.annotation_gt.value
    assert trace["query_spec"]["params"]["grid_shape"] == [3, 3]
    assert out.answer_gt.value in trace["render_map"]["option_bboxes_px_by_label"]
