from __future__ import annotations

from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.illustrations.rpg_dungeon.missing_patch_label import TASK_ID as MISSING_PATCH_LABEL_TASK_ID
from trace_tasks.tasks.illustrations.rpg_dungeon.monster_chamber_count import TASK_ID as MONSTER_CHAMBER_COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_dungeon.reachable_chest_count import TASK_ID as REACHABLE_CHEST_COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_dungeon.safe_reachable_chest_count import TASK_ID as SAFE_REACHABLE_CHEST_COUNT_TASK_ID
from trace_tasks.tasks.illustrations.rpg_dungeon.shared.output import safe_reachable_chest_ids
from trace_tasks.tasks.illustrations.rpg_dungeon.shared.relations import reachable_tiles
from trace_tasks.tasks.illustrations.rpg_dungeon.shared.rendering import (
    MAX_MONSTER_CHAMBER_COUNT,
    MAX_REACHABLE_CHEST_COUNT,
    MAX_TOTAL_CHEST_COUNT,
    MIN_MONSTER_CHAMBER_COUNT,
    MIN_REACHABLE_CHEST_COUNT,
    MIN_TOTAL_CHEST_COUNT,
    MONSTER_OBJECT_TYPES,
    draw_rpg_dungeon_debug_overlay,
    render_rpg_dungeon_scene,
)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= float(width)
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= float(height)


def test_rpg_dungeon_renderer_is_deterministic_and_profile_safe() -> None:
    for width, height, total_count in ((1296, 864, 6), (1008, 1008, 5), (864, 1296, 4)):
        first = render_rpg_dungeon_scene(
            2026062001,
            width=width,
            height=height,
            total_chest_count=total_count,
            reachable_chest_count=3,
        )
        second = render_rpg_dungeon_scene(
            2026062001,
            width=width,
            height=height,
            total_chest_count=total_count,
            reachable_chest_count=3,
        )
        assert first.image.size == (width, height)
        assert first.image.tobytes() == second.image.tobytes()
        assert draw_rpg_dungeon_debug_overlay(first).size == first.image.size
        assert len(first.chest_entity_ids) == total_count
        assert first.trace["total_chest_count"] == total_count
        assert len(first.reachable_chest_ids) == 3
        assert first.trace["layout_orientation"] in {"top_bottom", "left_right"}
        assert sorted(first.trace["side_counts"].values()) in ([2, 2], [2, 3], [3, 3])
        assert sum(first.trace["side_counts"].values()) == total_count
        assert first.player_entity_id == "player_00"
        assert len(first.floor_tiles) > 0
        assert len(first.corridor_tiles) > 0
        assert len(first.blocked_tiles) == len(first.blockers)
        assert len(first.blocked_tiles) == len(set(first.blocked_tiles))
        assert sum(1 for edge_id in first.trace["edge_ids"] if str(edge_id).startswith("edge_start_")) == total_count
        assert any(not str(edge_id).startswith("edge_start_") for edge_id in first.trace["edge_ids"])
        assert set(first.trace["blocked_edge_ids"]) == {
            str(blocker.metadata["edge_id"])
            for blocker in first.blockers
        }
        for chamber in first.chambers:
            _assert_bbox_inside_canvas(list(chamber.bbox_xyxy), width=width, height=height)
        for blocker in first.blockers:
            _assert_bbox_inside_canvas(list(blocker.bbox_xyxy), width=width, height=height)
            assert blocker.blocker_type == "boulder"
            assert blocker.metadata["passable"] is False
        for entity in first.entities:
            _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=width, height=height)
            assert entity.object_type in {"chest", "person"}


def test_rpg_dungeon_renderer_supports_distinct_monsters() -> None:
    first = render_rpg_dungeon_scene(
        2026062201,
        width=1296,
        height=864,
        total_chest_count=6,
        reachable_chest_count=6,
        monster_chamber_count=4,
    )
    second = render_rpg_dungeon_scene(
        2026062201,
        width=1296,
        height=864,
        total_chest_count=6,
        reachable_chest_count=6,
        monster_chamber_count=4,
    )
    assert first.image.tobytes() == second.image.tobytes()
    monsters = [entity for entity in first.entities if str(entity.object_type).startswith("monster_")]
    assert len(monsters) == 4
    assert first.trace["monster_count"] == 4
    assert set(first.trace["monster_entity_ids"]) == {str(entity.entity_id) for entity in monsters}
    assert set(first.trace["monster_chamber_ids"]) == {str(entity.chamber_id) for entity in monsters}
    assert {str(entity.object_type) for entity in monsters}.issubset(set(MONSTER_OBJECT_TYPES))
    assert len({str(entity.chamber_id) for entity in monsters}) == 4
    for entity in monsters:
        _assert_bbox_inside_canvas(list(entity.bbox_xyxy), width=1296, height=864)
        assert entity.role == "queryable"
        assert entity.metadata["visual_attributes"]["monster_type"] == entity.object_type


def test_rpg_dungeon_renderer_controls_reachable_monster_split() -> None:
    scene = render_rpg_dungeon_scene(
        2026062202,
        width=1296,
        height=864,
        total_chest_count=6,
        reachable_chest_count=4,
        monster_chamber_count=3,
        reachable_monster_chamber_count=2,
    )
    monsters = [entity for entity in scene.entities if str(entity.object_type).startswith("monster_")]
    reachable_chambers = {
        str(entity.chamber_id)
        for entity in scene.entities
        if str(entity.entity_id) in set(scene.reachable_chest_ids)
    }
    reachable_monster_chambers = {
        str(entity.chamber_id)
        for entity in monsters
        if str(entity.chamber_id) in reachable_chambers
    }
    assert len(scene.reachable_chest_ids) == 4
    assert len(monsters) == 3
    assert len(reachable_monster_chambers) == 2
    assert len(safe_reachable_chest_ids(scene)) == 2
    assert len(scene.blockers) > 0


def test_rpg_dungeon_renderer_samples_reachable_count_range() -> None:
    seen_counts = set()
    seen_totals = set()
    seen_orientations = set()
    for seed in range(120):
        scene = render_rpg_dungeon_scene(7000 + seed, width=1296, height=864)
        total = len(scene.chest_entity_ids)
        seen_counts.add(len(scene.reachable_chest_ids))
        seen_totals.add(total)
        seen_orientations.add(str(scene.trace["layout_orientation"]))
        assert MIN_TOTAL_CHEST_COUNT <= total <= MAX_TOTAL_CHEST_COUNT
        assert MIN_REACHABLE_CHEST_COUNT <= len(scene.reachable_chest_ids) <= MAX_REACHABLE_CHEST_COUNT
        assert len(scene.reachable_chest_ids) <= total
        assert MIN_MONSTER_CHAMBER_COUNT <= scene.trace["monster_count"] <= MAX_MONSTER_CHAMBER_COUNT
        assert scene.trace["monster_count"] == 0
    assert seen_counts == set(range(MIN_REACHABLE_CHEST_COUNT, MAX_REACHABLE_CHEST_COUNT + 1))
    assert seen_totals == set(range(MIN_TOTAL_CHEST_COUNT, MAX_TOTAL_CHEST_COUNT + 1))
    assert {"top_bottom", "left_right"}.issubset(seen_orientations)


def test_rpg_dungeon_reachable_chest_count_contract() -> None:
    task = create_task(REACHABLE_CHEST_COUNT_TASK_ID)
    for profile, total_count, count, seed in (
        ("landscape", 4, 0, 2026062011),
        ("square", 5, 2, 2026062012),
        ("portrait", 6, 6, 2026062013),
    ):
        out = task.generate(
            seed,
            params={
                "canvas_profile": profile,
                "total_chest_count": total_count,
                "reachable_chest_count": count,
            },
            max_attempts=20,
        )
        assert out.scene_id == "rpg_dungeon"
        assert out.query_id == "single"
        assert out.answer_gt.type == "integer"
        assert out.answer_gt.value == count
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == count
        assert "red-outlined" not in out.prompt
        assert "lettered candidate" not in out.prompt
        width, height = out.image.size
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(list(bbox), width=width, height=height)

        trace = out.trace_payload
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_dungeon_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_dungeon"
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert trace["render_map"]["reachable_chest_bboxes_px"] == {
            entity_id: bbox
            for entity_id, bbox in trace["render_map"]["chest_bboxes_px"].items()
            if entity_id in trace["render_map"]["reachable_chest_ids"]
        }
        assert trace["render_map"]["reachable_count"] == count
        assert trace["render_map"]["total_chest_count"] == total_count
        assert len(trace["render_map"]["chest_entity_ids"]) == total_count
        assert trace["query_spec"]["params"]["total_chest_count"] == total_count

        scene_ir = trace["scene_ir"]
        reached = reachable_tiles(
            (tuple(tile) for tile in scene_ir["floor_tiles"]),
            blocked_tiles=(tuple(tile) for tile in scene_ir["blocked_tiles"]),
            start_tile=tuple(trace["execution_trace"]["renderer"]["player_tile"]),
        )
        reachable_tile_set = set(reached)
        chest_tile_map = {
            str(entity_id): tuple(tile)
            for entity_id, tile in trace["execution_trace"]["renderer"]["chest_tile_map"].items()
        }
        expected_ids = sorted(
            entity_id
            for entity_id, tile in chest_tile_map.items()
            if tuple(tile) in reachable_tile_set
        )
        assert expected_ids == sorted(trace["render_map"]["reachable_chest_ids"])
        assert expected_ids == sorted(trace["scene_ir"]["relations"]["reachable_chest_ids"])


def test_rpg_dungeon_monster_chamber_count_contract() -> None:
    task = create_task(MONSTER_CHAMBER_COUNT_TASK_ID)
    for profile, total_count, count, seed in (
        ("landscape", 4, 1, 2026062211),
        ("square", 5, 2, 2026062212),
        ("portrait", 6, 4, 2026062213),
    ):
        out = task.generate(
            seed,
            params={
                "canvas_profile": profile,
                "total_chest_count": total_count,
                "monster_chamber_count": count,
            },
            max_attempts=20,
        )
        assert out.scene_id == "rpg_dungeon"
        assert out.query_id == "single"
        assert out.answer_gt.type == "integer"
        assert out.answer_gt.value == count
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == count
        assert "rooms" not in out.prompt.lower()
        assert "red-outlined" not in out.prompt
        assert "lettered candidate" not in out.prompt
        width, height = out.image.size
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(list(bbox), width=width, height=height)

        trace = out.trace_payload
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_dungeon_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_dungeon"
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert trace["render_map"]["monster_count"] == count
        assert trace["render_map"]["total_chest_count"] == total_count
        assert len(trace["render_map"]["chest_entity_ids"]) == total_count
        assert len(trace["render_map"]["monster_entity_ids"]) == count
        assert len(trace["render_map"]["monster_chamber_ids"]) == count
        assert len(set(trace["render_map"]["monster_chamber_ids"])) == count
        assert trace["query_spec"]["params"]["monster_chamber_count"] == count
        assert trace["query_spec"]["params"]["total_chest_count"] == total_count
        assert trace["scene_ir"]["relations"]["monster_chamber_ids"] == trace["render_map"]["monster_chamber_ids"]
        assert trace["scene_ir"]["relations"]["answer"] == count


def test_rpg_dungeon_safe_reachable_chest_count_contract() -> None:
    task = create_task(SAFE_REACHABLE_CHEST_COUNT_TASK_ID)
    for profile, total_count, count, seed in (
        ("landscape", 4, 0, 2026062221),
        ("square", 5, 2, 2026062222),
        ("portrait", 6, 5, 2026062223),
    ):
        out = task.generate(
            seed,
            params={
                "canvas_profile": profile,
                "total_chest_count": total_count,
                "safe_reachable_chest_count": count,
            },
            max_attempts=20,
        )
        assert out.scene_id == "rpg_dungeon"
        assert out.query_id == "single"
        assert out.answer_gt.type == "integer"
        assert out.answer_gt.value == count
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == count
        assert "rooms" not in out.prompt.lower()
        assert "boulder" in out.prompt.lower() or "blocked" in out.prompt.lower()
        assert "monster" in out.prompt.lower()
        width, height = out.image.size
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(list(bbox), width=width, height=height)

        trace = out.trace_payload
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_dungeon_v0"
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        assert trace["render_map"]["counted_chest_bboxes_px"] == {
            entity_id: bbox
            for entity_id, bbox in trace["render_map"]["chest_bboxes_px"].items()
            if entity_id in trace["render_map"]["counted_chest_ids"]
        }
        assert trace["render_map"]["counted_count"] == count
        assert trace["render_map"]["total_chest_count"] == total_count
        assert len(trace["render_map"]["counted_chest_ids"]) == count
        assert len(trace["render_map"]["monster_chamber_ids"]) >= 1
        assert len(trace["render_map"]["blocker_bboxes_px"]) >= 1
        assert trace["query_spec"]["params"]["safe_reachable_chest_count"] == count
        assert trace["query_spec"]["params"]["reachable_chest_count"] < total_count
        assert trace["query_spec"]["params"]["monster_chamber_count"] >= 1
        assert trace["scene_ir"]["relations"]["counted_chest_ids"] == trace["render_map"]["counted_chest_ids"]
        assert trace["scene_ir"]["relations"]["answer"] == count


def test_rpg_dungeon_missing_patch_label_contract() -> None:
    task = create_task(MISSING_PATCH_LABEL_TASK_ID)
    for profile, option_count, seed in (
        ("landscape", 4, 2026062261),
        ("square", 6, 2026062262),
        ("portrait", 4, 2026062263),
    ):
        out = task.generate(
            seed,
            params={
                "canvas_profile": profile,
                "source_chest_count": 6,
                "source_reachable_chest_count": 4,
                "source_monster_count": 2,
                "option_count": option_count,
            },
            max_attempts=40,
        )
        assert out.scene_id == "rpg_dungeon"
        assert out.query_id == "single"
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox_map"
        assert sorted(out.annotation_gt.value) == ["missing_region", "selected_option"]
        assert "patch" in out.prompt.lower()
        assert "room" not in out.prompt.lower()
        width, height = out.image.size
        for bbox in out.annotation_gt.value.values():
            _assert_bbox_inside_canvas(list(bbox), width=width, height=height)

        trace = out.trace_payload
        assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "illustrations_rpg_dungeon_v0"
        assert trace["query_spec"]["prompt_variant"]["prompt_scene_id"] == "rpg_dungeon"
        assert trace["query_spec"]["params"]["source_chest_count"] == 6
        assert trace["query_spec"]["params"]["source_reachable_chest_count"] == 4
        assert trace["query_spec"]["params"]["source_monster_count"] == 2
        assert trace["query_spec"]["params"]["option_count"] == option_count
        assert trace["projected_annotation"]["type"] == "bbox_map"
        assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
        assert trace["render_map"]["missing_region_bbox_px"] == out.annotation_gt.value["missing_region"]
        assert trace["render_map"]["selected_option_bbox_px"] == out.annotation_gt.value["selected_option"]
        assert len(trace["render_map"]["option_bboxes_px_by_label"]) == option_count
        assert out.answer_gt.value in trace["render_map"]["option_bboxes_px_by_label"]
        assert trace["render_map"]["candidate_crop_count"] >= 1
        assert trace["scene_ir"]["relations"]["answer_label"] == out.answer_gt.value
