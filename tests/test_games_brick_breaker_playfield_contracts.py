"""Contract tests for games Brick-breaker playfield tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.brick_breaker.hit_row_remaining_count import GamesBrickBreakerHitRowRemainingCountTask
from trace_tasks.tasks.games.brick_breaker.next_hit_label import GamesBrickBreakerNextHitLabelTask
from trace_tasks.tasks.games.brick_breaker.paddle_catch_label import GamesBrickBreakerPaddleCatchLabelTask
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_internal_query", "expected_answer_type"),
    (
        (
            GamesBrickBreakerNextHitLabelTask,
            {"brick_rows": 5, "brick_cols": 6, "lane_count": 6, "style_variant": "neon"},
            "next_hit_label",
            "string",
        ),
        (
            GamesBrickBreakerPaddleCatchLabelTask,
            {"brick_rows": 5, "brick_cols": 6, "lane_count": 6, "style_variant": "paper"},
            "paddle_catch_label",
            "string",
        ),
        (
            GamesBrickBreakerHitRowRemainingCountTask,
            {"brick_rows": 5, "brick_cols": 6, "lane_count": 6, "style_variant": "classic"},
            "hit_row_remaining_count",
            "integer",
        ),
    ),
)
def test_games_brick_breaker_public_tasks_emit_expected_contract(
    task_cls: type[Any],
    params: dict[str, int | str],
    expected_internal_query: str,
    expected_answer_type: str,
) -> None:
    out = task_cls().generate(91000, params=params, max_attempts=256)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == expected_answer_type
    expected_annotation_type = "point_set" if expected_internal_query == "hit_row_remaining_count" else "point"
    assert out.annotation_gt.type == expected_annotation_type
    assert out.query_id == "single"
    assert out.scene_id == "brick_breaker"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["internal_query_id"] == expected_internal_query
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["internal_query_id"] == expected_internal_query
    assert execution["query_id"] == "single"
    if expected_annotation_type == "point":
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        assert len(execution["annotation_entity_ids"]) == 1
    else:
        assert len(out.annotation_gt.value) >= 1
        assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
        assert len(execution["annotation_entity_ids"]) == len(out.annotation_gt.value)
    assert trace["render_spec"]["canvas_width"] <= 980
    assert trace["render_spec"]["canvas_height"] <= 740
    assert trace["render_spec"]["panel_scene_style"]["treatment"]
    assert trace["render_spec"]["text_style"]["font_family"]
    assert trace["render_map"]["font_family"] == trace["render_spec"]["text_style"]["font_family"]
    assert float(trace["render_map"]["guide_color_safety"]["guide_anchor_lab_distance"]) >= 40.0
    paddle_bbox = trace["render_map"]["paddle_bbox_px"]
    lane_bboxes = trace["render_map"]["lane_bboxes_px"]
    first_lane_bbox = lane_bboxes["lane_0"]
    assert round(float(paddle_bbox[2] - paddle_bbox[0]), 3) == round(
        float(first_lane_bbox[2] - first_lane_bbox[0]),
        3,
    )
    assert float(paddle_bbox[3] - paddle_bbox[1]) <= 9.0
    entity_bboxes = trace["render_map"]["entity_bboxes_px"]
    annotation_points = [out.annotation_gt.value] if expected_annotation_type == "point" else out.annotation_gt.value
    for entity_id, point in zip(execution["annotation_entity_ids"], annotation_points):
        bbox = entity_bboxes[str(entity_id)]
        expected_point = [
            round(float(bbox[0] + bbox[2]) / 2.0, 3),
            round(float(bbox[1] + bbox[3]) / 2.0, 3),
        ]
        assert list(point) == expected_point
        x, y = point
        assert 0 <= float(x) <= float(trace["render_spec"]["canvas_width"])
        assert 0 <= float(y) <= float(trace["render_spec"]["canvas_height"])


def test_games_brick_breaker_next_hit_label_matches_target_brick() -> None:
    out = GamesBrickBreakerNextHitLabelTask().generate(
        91010,
        params={"brick_rows": 5, "brick_cols": 6, "lane_count": 6},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_brick_id"])
    target_brick = next(brick for brick in execution["bricks"] if str(brick["brick_id"]) == target_id)
    target_row = int(target_brick["row"])
    full_path = out.trace_payload["render_map"]["motion_path_px"]
    visible_path = out.trace_payload["render_map"]["visible_motion_path_px"]
    target_bbox = out.trace_payload["render_map"]["brick_bboxes_px"][target_id]
    ball_bbox = out.trace_payload["render_map"]["ball_bbox_px"]

    assert str(out.answer_gt.value) == str(target_brick["label"]) == str(execution["target_brick_label"])
    assert list(execution["annotation_entity_ids"]) == [target_id]
    assert len(execution["bricks"]) <= 26
    assert target_row == int(execution["brick_rows"]) - 1
    assert int(execution["ball_start_lane_index"]) >= 0
    assert full_path["start"][0] != full_path["end"][0]
    assert visible_path["end"] != full_path["end"]
    assert float(visible_path["fraction_of_full_path"]) <= 0.55
    assert float(visible_path["end"][1]) >= float(target_bbox[3]) + 20.0
    assert float(ball_bbox[1]) >= float(target_bbox[3]) + 40.0


def test_games_brick_breaker_paddle_catch_label_matches_target_lane() -> None:
    out = GamesBrickBreakerPaddleCatchLabelTask().generate(
        91020,
        params={"brick_rows": 5, "brick_cols": 6, "lane_count": 6},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    target_lane = int(execution["target_lane_index"])
    expected_lane_id = f"lane_{target_lane}"
    expected_label = chr(ord("A") + target_lane)
    full_path = out.trace_payload["render_map"]["motion_path_px"]
    visible_path = out.trace_payload["render_map"]["visible_motion_path_px"]

    assert str(out.answer_gt.value) == expected_label == str(execution["target_lane_label"])
    assert list(execution["annotation_entity_ids"]) == [expected_lane_id]
    assert len(execution["bricks"]) <= 26
    assert 0 <= int(execution["ball_start_lane_index"]) < int(execution["lane_count"])
    assert visible_path["end"] != full_path["end"]


def test_games_brick_breaker_hit_row_remaining_count_matches_row_survivors() -> None:
    out = GamesBrickBreakerHitRowRemainingCountTask().generate(
        91030,
        params={"brick_rows": 5, "brick_cols": 6, "lane_count": 6},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_brick_id"])
    target_brick = next(brick for brick in execution["bricks"] if str(brick["brick_id"]) == target_id)
    target_row = int(target_brick["row"])
    expected_ids = [
        str(brick["brick_id"])
        for brick in execution["bricks"]
        if int(brick["row"]) == target_row and str(brick["brick_id"]) != target_id
    ]
    target_bbox = out.trace_payload["render_map"]["brick_bboxes_px"][target_id]
    visible_path = out.trace_payload["render_map"]["visible_motion_path_px"]
    ball_bbox = out.trace_payload["render_map"]["ball_bbox_px"]

    assert int(out.answer_gt.value) == len(expected_ids) == int(execution["target_row_remaining_count"])
    assert set(execution["target_row_remaining_brick_ids"]) == set(expected_ids)
    assert set(execution["annotation_entity_ids"]) == set(expected_ids)
    assert len(execution["bricks"]) <= 26
    assert float(visible_path["fraction_of_full_path"]) <= 0.55
    assert float(visible_path["end"][1]) >= float(target_bbox[3]) + 20.0
    assert float(ball_bbox[1]) >= float(target_bbox[3]) + 40.0


def test_games_brick_breaker_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__brick_breaker"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__brick_breaker",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__brick_breaker__next_hit_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__brick_breaker__hit_row_remaining_count", count=1, params={}),
        ],
        max_attempts_per_instance=256,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-brick-breaker-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 2
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "brick_breaker" for row in rows)
