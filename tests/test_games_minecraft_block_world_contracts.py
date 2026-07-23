"""Contract tests for Minecraft-like games tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.minecraft.resource_route_cost import (
    QUERY_ID as ROUTE_QUERY_ID,
    GamesMinecraftResourceRouteCostTask,
)
from trace_tasks.tasks.games.minecraft.stack_height_condition_count import (
    AT_LEAST_HEIGHT_QUERY_ID,
    EXACT_HEIGHT_QUERY_ID,
    GamesMinecraftStackHeightConditionCountTask,
)
from trace_tasks.tasks.games.minecraft.top_ore_stack_count import (
    QUERY_ID as TOP_RESOURCE_QUERY_ID,
    GamesMinecraftTopOreStackCountTask,
)
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "target_answer", "expected_query", "extra_params"),
    (
        (GamesMinecraftTopOreStackCountTask, 4, TOP_RESOURCE_QUERY_ID, {"resource_kind": "gold_ore"}),
        (GamesMinecraftResourceRouteCostTask, 5, ROUTE_QUERY_ID, {}),
        (GamesMinecraftStackHeightConditionCountTask, 4, EXACT_HEIGHT_QUERY_ID, {"query_id": EXACT_HEIGHT_QUERY_ID}),
        (GamesMinecraftStackHeightConditionCountTask, 4, AT_LEAST_HEIGHT_QUERY_ID, {"query_id": AT_LEAST_HEIGHT_QUERY_ID}),
    ),
)
def test_games_minecraft_public_tasks_emit_expected_contract(
    task_cls,
    target_answer: int,
    expected_query: str,
    extra_params: dict[str, str],
) -> None:
    out = task_cls().generate(
        2026052201,
        params={"target_answer": int(target_answer), "style_variant": "grass", **extra_params},
        max_attempts=512,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert out.query_id == expected_query
    assert out.scene_id == "minecraft"
    assert trace["query_spec"]["query_id"] == expected_query
    assert trace["query_spec"]["params"]["query_id"] == expected_query
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert int(execution["answer"]) == int(out.answer_gt.value)
    assert "panel_scene_style" in trace["render_spec"]
    assert trace["render_spec"]["text_style"]["font_family"]


@pytest.mark.parametrize("resource_kind", ("gold_ore", "diamond_ore"))
def test_games_minecraft_top_ore_stack_annotation_matches_target_kind(resource_kind: str) -> None:
    out = GamesMinecraftTopOreStackCountTask().generate(
        2026052202,
        params={"target_answer": 5, "resource_kind": str(resource_kind)},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    annotation_ids = {str(entity_id) for entity_id in execution["annotation_entity_ids"]}
    target_kind = str(execution["counted_resource_kind"])
    top_by_coord: dict[tuple[int, int], tuple[int, str]] = {}
    for block in execution["blocks"]:
        coord = (int(block["x"]), int(block["y"]))
        z = int(block["z"])
        assert str(block["kind"]) != "iron_ore"
        if coord not in top_by_coord or z > top_by_coord[coord][0]:
            top_by_coord[coord] = (z, str(block["kind"]))
    counted_ids = {
        f"stack_{x:02d}_{y:02d}"
        for (x, y), (_z, kind) in top_by_coord.items()
        if str(kind) == target_kind
    }
    target_coords = [
        (int(entity_id.split("_")[1]), int(entity_id.split("_")[2]))
        for entity_id in counted_ids
    ]
    distractor_coords = [
        coord
        for coord, (_z, kind) in top_by_coord.items()
        if str(kind) != target_kind
    ]

    assert target_kind == str(resource_kind)
    assert int(out.answer_gt.value) == 5
    assert annotation_ids == counted_ids
    assert all(entity_id.startswith("stack_") for entity_id in annotation_ids)
    assert len({tuple(point) for point in out.annotation_gt.value}) == len(out.annotation_gt.value)
    assert len(distractor_coords) <= 3
    assert max((int(top_z) + 1 for top_z, _kind in top_by_coord.values()), default=0) <= 3
    assert all(
        max(abs(ax - bx), abs(ay - by)) >= 2
        for index, (ax, ay) in enumerate(target_coords)
        for bx, by in target_coords[index + 1 :]
    )


def test_games_minecraft_top_ore_issue_seed_has_unique_visible_points() -> None:
    out = GamesMinecraftTopOreStackCountTask().generate(
        45545150419015,
        params={},
        max_attempts=512,
    )

    assert int(out.answer_gt.value) == len(out.annotation_gt.value)
    assert len({tuple(point) for point in out.annotation_gt.value}) == len(out.annotation_gt.value)


def test_games_minecraft_top_ore_stack_rejects_removed_iron_ore() -> None:
    with pytest.raises(RuntimeError):
        GamesMinecraftTopOreStackCountTask().generate(
            2026052203,
            params={"target_answer": 3, "resource_kind": "iron_ore"},
            max_attempts=3,
        )


def test_games_minecraft_resource_route_cost_matches_single_track_blocks() -> None:
    out = GamesMinecraftResourceRouteCostTask().generate(
        2026052204,
        params={
            "target_answer": 5,
            "grid_width": 11,
            "grid_depth": 10,
        },
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    annotation_ids = {str(entity_id) for entity_id in execution["annotation_entity_ids"]}
    block_by_id = {str(block["block_id"]): block for block in execution["blocks"]}
    ordered_track_cells = [tuple(int(v) for v in cell) for cell in execution["track_cells"]]
    track_cells = set(ordered_track_cells)
    endpoint_cells = {ordered_track_cells[0], ordered_track_cells[-1]}
    annotation_cells = {
        (int(block_by_id[entity_id]["x"]), int(block_by_id[entity_id]["y"]))
        for entity_id in annotation_ids
    }
    distractor_cells = {
        (int(block["x"]), int(block["y"]))
        for block_id, block in block_by_id.items()
        if str(block_id) not in annotation_ids
    }

    assert int(out.answer_gt.value) == 5
    assert int(execution["track_raised_block_count"]) == 5
    assert len(track_cells) >= 6
    assert annotation_ids < set(block_by_id)
    assert annotation_cells <= track_cells
    assert not (annotation_cells & endpoint_cells)
    assert len(distractor_cells) >= 2
    assert not (distractor_cells & track_cells)
    assert all(
        min(max(abs(dx - tx), abs(dy - ty)) for tx, ty in track_cells) >= 3
        for dx, dy in distractor_cells
    )
    assert all(str(block["kind"]) in {"stone", "dirt"} for block in block_by_id.values())
    assert "route_option_count" not in execution
    assert "route_costs" not in execution
    assert "selected_route_label" not in execution
    assert len(out.trace_payload["render_map"]["route_overlays"]) == 1


def test_games_minecraft_resource_route_cost_allows_zero_answer() -> None:
    out = GamesMinecraftResourceRouteCostTask().generate(
        2026052205,
        params={"target_answer": 0},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]

    assert int(out.answer_gt.value) == 0
    assert int(execution["track_raised_block_count"]) == 0
    assert execution["annotation_entity_ids"] == []
    assert out.annotation_gt.value == []
    assert execution["track_cells"]
    assert len(execution["blocks"]) >= 2


@pytest.mark.parametrize(
    ("query_id", "target_height"),
    (
        (EXACT_HEIGHT_QUERY_ID, 4),
        (AT_LEAST_HEIGHT_QUERY_ID, 3),
    ),
)
def test_games_minecraft_stack_height_condition_matches_trace(query_id: str, target_height: int) -> None:
    out = GamesMinecraftStackHeightConditionCountTask().generate(
        2026060801,
        params={
            "query_id": str(query_id),
            "target_answer": 5,
            "target_stack_height": int(target_height),
            "grid_width": 9,
            "grid_depth": 9,
        },
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    heights: dict[tuple[int, int], set[int]] = {}
    for block in execution["blocks"]:
        heights.setdefault((int(block["x"]), int(block["y"])), set()).add(int(block["z"]))

    if str(query_id) == EXACT_HEIGHT_QUERY_ID:
        expected = {
            f"stack_{x:02d}_{y:02d}"
            for (x, y), z_values in heights.items()
            if len(z_values) == int(target_height)
        }
        assert execution["stack_height_condition"] == "exact"
    else:
        expected = {
            f"stack_{x:02d}_{y:02d}"
            for (x, y), z_values in heights.items()
            if len(z_values) >= int(target_height)
        }
        assert execution["stack_height_condition"] == "at_least"

    assert execution["target_stack_height"] == int(target_height)
    assert int(out.answer_gt.value) == len(expected) == 5
    assert set(execution["annotation_entity_ids"]) == expected
    assert len(out.annotation_gt.value) == len(expected)
    assert all(entity_id.startswith("stack_") for entity_id in execution["annotation_entity_ids"])


def test_games_minecraft_stack_height_condition_rejects_answer_six() -> None:
    with pytest.raises(ValueError):
        GamesMinecraftStackHeightConditionCountTask().generate(
            2026060802,
            params={"target_answer": 6},
            max_attempts=3,
        )


def test_games_minecraft_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__minecraft"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__minecraft",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__minecraft__top_ore_stack_count", count=1, params={}),
            BuildTaskConfig(
                task_id="task_games__minecraft__resource_route_cost",
                count=2,
                params={},
            ),
            BuildTaskConfig(
                task_id="task_games__minecraft__stack_height_condition_count",
                count=2,
                params={},
            ),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-minecraft-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 5
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "minecraft" for row in rows)
    assert {row["scene_id"] for row in rows} == {"minecraft"}
