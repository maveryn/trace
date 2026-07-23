"""Contract tests for games Bowling lane tasks."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.bowling.first_pin_hit_label import (
    GamesBowlingFirstPinHitLabelTask,
)
from trace_tasks.tasks.games.bowling.path_hit_count import (
    GamesBowlingPathHitCountTask,
)
from trace_tasks.tasks.games.bowling.shared.rules import (
    PATH_NON_HIT_CLEARANCE_PX,
    first_intersected_pin_id,
    path_intersected_pin_ids,
)
from trace_tasks.tasks.games.bowling.shared.state import BowlingPin
from trace_tasks.tasks.games.bowling.spare_path_label import GamesBowlingSparePathLabelTask
from tests.helpers import read_jsonl


def test_games_bowling_scene_package_source_layout() -> None:
    expected_sources = {
        GamesBowlingFirstPinHitLabelTask: Path("src/trace_tasks/tasks/games/bowling/first_pin_hit_label.py"),
        GamesBowlingPathHitCountTask: Path("src/trace_tasks/tasks/games/bowling/path_hit_count.py"),
        GamesBowlingSparePathLabelTask: Path("src/trace_tasks/tasks/games/bowling/spare_path_label.py"),
    }

    for task_cls, relative_path in expected_sources.items():
        source_path = Path(inspect.getsourcefile(task_cls) or "").resolve()
        assert source_path == (Path.cwd() / relative_path).resolve()
        assert not getattr(task_cls, "scene_id", "")


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_internal_query"),
    (
        (
            GamesBowlingFirstPinHitLabelTask,
            {"target_pin_index": 7, "style_variant": "cosmic"},
            "first_pin_hit_label",
        ),
        (
            GamesBowlingPathHitCountTask,
            {"target_answer": 4, "style_variant": "retro"},
            "path_hit_count",
        ),
        (
            GamesBowlingSparePathLabelTask,
            {"path_option_count": 6, "target_path_index": 4, "style_variant": "paper"},
            "spare_path_label",
        ),
    ),
)
def test_games_bowling_public_tasks_emit_expected_contract(
    task_cls: type[Any],
    params: dict[str, int | str],
    expected_internal_query: str,
) -> None:
    out = task_cls().generate(95000, params=params, max_attempts=256)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    if expected_internal_query == "path_hit_count":
        assert out.answer_gt.type == "integer"
    else:
        assert out.answer_gt.type == "string"
    if expected_internal_query == "spare_path_label":
        assert out.annotation_gt.type == "segment"
        assert len(out.annotation_gt.value) == 2
        assert trace["projected_annotation"]["segment"] == out.annotation_gt.value
    elif expected_internal_query == "path_hit_count":
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    else:
        assert out.annotation_gt.type == "point"
        assert len(out.annotation_gt.value) == 2
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert out.query_id == "single"
    assert out.scene_id == "bowling"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["internal_query_id"] == expected_internal_query
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["internal_query_id"] == expected_internal_query
    assert execution["query_id"] == "single"
    if expected_internal_query == "path_hit_count":
        assert len(execution["annotation_entity_ids"]) == int(out.answer_gt.value)
    else:
        assert len(execution["annotation_entity_ids"]) == 1
    assert trace["render_spec"]["text_style"]["font_family"]
    assert trace["render_map"]["font_family"] == trace["render_spec"]["text_style"]["font_family"]
    assert float(trace["render_map"]["path_color_safety"]["min_path_anchor_lab_distance"]) >= 40.0
    assert len(trace["render_map"]["path_palette_rgb"]) >= 6
    if out.annotation_gt.type == "point":
        x, y = out.annotation_gt.value
        assert 0 <= float(x) <= float(trace["render_spec"]["canvas_width"])
        assert 0 <= float(y) <= float(trace["render_spec"]["canvas_height"])
    elif out.annotation_gt.type == "segment":
        for x, y in out.annotation_gt.value:
            assert 0 <= float(x) <= float(trace["render_spec"]["canvas_width"])
            assert 0 <= float(y) <= float(trace["render_spec"]["canvas_height"])
    elif out.annotation_gt.type == "bbox_set":
        for x0, y0, x1, y1 in out.annotation_gt.value:
            assert 0 <= float(x0) <= float(x1) <= float(trace["render_spec"]["canvas_width"])
            assert 0 <= float(y0) <= float(y1) <= float(trace["render_spec"]["canvas_height"])
    else:
        raise AssertionError(f"unexpected annotation type: {out.annotation_gt.type}")


def test_games_bowling_first_pin_hit_label_matches_target_pin() -> None:
    out = GamesBowlingFirstPinHitLabelTask().generate(
        95010,
        params={"target_pin_index": 8},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_pin_id"])
    target_pin = next(pin for pin in execution["pins"] if str(pin["pin_id"]) == target_id)
    full_path = out.trace_payload["render_map"]["motion_paths_px"]["shown_path"]
    pins = tuple(
        BowlingPin(
            pin_id=str(pin["pin_id"]),
            label=str(pin["label"]),
            rack_index=int(pin["rack_index"]),
            row=int(pin["row"]),
            col=int(pin["col"]),
            color_index=0,
            standing=bool(pin["standing"]),
            x_norm=float(pin["x_norm"]),
            y_norm=float(pin["y_norm"]),
        )
        for pin in execution["pins"]
    )
    first_hit = first_intersected_pin_id(
        pins=pins,
        ball_x_norm=float(execution["ball_x_norm"]),
        aim_x_norm=float(target_pin["x_norm"]),
        aim_y_norm=float(target_pin["y_norm"]),
    )

    assert str(out.answer_gt.value) == str(target_pin["label"]) == str(execution["target_pin_label"])
    assert str(first_hit) == target_id
    assert list(execution["annotation_entity_ids"]) == [target_id]
    assert bool(target_pin["standing"]) is True
    assert 4 <= int(execution["visible_pin_count"]) <= 9
    assert len(execution["pins"]) == int(execution["visible_pin_count"])
    assert full_path["visible_end"] != full_path["end"]


def test_games_bowling_path_hit_count_matches_recomputed_hits() -> None:
    out = GamesBowlingPathHitCountTask().generate(
        95015,
        params={"target_answer": 5},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_pin_id"])
    target_pin = next(pin for pin in execution["pins"] if str(pin["pin_id"]) == target_id)
    pins = tuple(
        BowlingPin(
            pin_id=str(pin["pin_id"]),
            label=str(pin["label"]),
            rack_index=int(pin["rack_index"]),
            row=int(pin["row"]),
            col=int(pin["col"]),
            color_index=0,
            standing=bool(pin["standing"]),
            x_norm=float(pin["x_norm"]),
            y_norm=float(pin["y_norm"]),
        )
        for pin in execution["pins"]
    )
    hit_ids = path_intersected_pin_ids(
        pins=pins,
        ball_x_norm=float(execution["ball_x_norm"]),
        aim_x_norm=float(target_pin["x_norm"]),
        aim_y_norm=float(target_pin["y_norm"]),
    )

    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == len(hit_ids) == 5
    assert tuple(execution["path_hit_pin_ids"]) == tuple(hit_ids)
    assert tuple(execution["annotation_entity_ids"]) == tuple(hit_ids)
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert str(execution["construction_mode"]) == "exact_path_hit_count_with_clearance"
    assert float(execution["path_clearance_px"]) >= float(PATH_NON_HIT_CLEARANCE_PX)
    shown_path = out.trace_payload["render_map"]["motion_paths_px"]["shown_path"]
    assert shown_path["visible_end"] != shown_path["end"]
    assert shown_path["visible_fraction"] == 0.62


def test_games_bowling_spare_path_label_matches_target_path() -> None:
    out = GamesBowlingSparePathLabelTask().generate(
        95020,
        params={"path_option_count": 6, "target_path_index": 5},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_path_id"])
    target_path = next(path for path in execution["path_options"] if str(path["path_id"]) == target_id)
    standing_pin_ids = {str(pin["pin_id"]) for pin in execution["pins"] if bool(pin["standing"])}
    paths_by_x = sorted(execution["path_options"], key=lambda path: float(path["aim_x_norm"]))
    path_labels_by_x = [str(path["label"]) for path in paths_by_x]
    target_rank = int(execution["target_path_index"])

    assert str(out.answer_gt.value) == str(target_path["label"]) == str(execution["target_path_label"])
    assert path_labels_by_x == [str(index + 1) for index in range(len(paths_by_x))]
    assert str(out.answer_gt.value) == str(target_rank + 1)
    assert str(paths_by_x[target_rank]["path_id"]) == target_id
    assert list(execution["annotation_entity_ids"]) == [target_id]
    assert set(execution["remaining_pin_ids"]) == standing_pin_ids
    assert len(standing_pin_ids) >= 1
    assert all(bool(pin["standing"]) for pin in execution["pins"])
    assert len(execution["pins"]) == len(standing_pin_ids)
    assert target_id in out.trace_payload["render_map"]["motion_paths_px"]
    assert out.annotation_gt.type == "segment"
    assert out.annotation_gt.value == out.trace_payload["render_map"]["path_point_pairs_px"][target_id]


def test_games_bowling_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__bowling"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__bowling",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__bowling__first_pin_hit_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__bowling__path_hit_count", count=1, params={}),
            BuildTaskConfig(task_id="task_games__bowling__spare_path_label", count=1, params={}),
        ],
        max_attempts_per_instance=256,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-bowling-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 3
    assert all(row["domain"] == "games" for row in rows)
    assert all("scene_id" not in row for row in rows)
    assert {row["task"] for row in rows} == {
        "task_games__bowling__first_pin_hit_label",
        "task_games__bowling__path_hit_count",
        "task_games__bowling__spare_path_label",
    }
