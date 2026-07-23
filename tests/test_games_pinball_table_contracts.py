"""Contract tests for games Pinball-table tasks."""

from __future__ import annotations

import inspect
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.query_ids import NO_BRANCH_QUERY_IDS
from trace_tasks.tasks.games.pinball_table.first_hit_object_label import GamesPinballFirstHitObjectLabelTask
from trace_tasks.tasks.games.pinball_table.scoreable_object_count import GamesPinballScoreableObjectCountTask
from trace_tasks.tasks.games.pinball_table.shared.sampling import first_hit_object_id
from trace_tasks.tasks.games.pinball_table.shared.state import (
    SUPPORTED_PINBALL_SCENE_VARIANTS,
    SUPPORTED_PINBALL_STYLE_VARIANTS,
    PinballObject,
)
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from tests.helpers import read_jsonl


def test_games_pinball_table_scene_package_source_layout() -> None:
    expected_sources = {
        GamesPinballFirstHitObjectLabelTask: Path("src/trace_tasks/tasks/games/pinball_table/first_hit_object_label.py"),
        GamesPinballScoreableObjectCountTask: Path("src/trace_tasks/tasks/games/pinball_table/scoreable_object_count.py"),
    }

    for task_cls, relative_path in expected_sources.items():
        source_path = Path(inspect.getsourcefile(task_cls) or "").resolve()
        assert source_path == (Path.cwd() / relative_path).resolve()
        assert not hasattr(task_cls, "scene_id")


def test_games_pinball_table_defaults_present() -> None:
    generation, _rendering, prompt = load_scene_generation_rendering_prompt_defaults(
        "games",
        "pinball_table",
        task_id="task_games__pinball_table__first_hit_object_label",
    )

    assert set(generation["scene_variant_weights"].keys()) == set(SUPPORTED_PINBALL_SCENE_VARIANTS)
    assert "query_id_weights" not in generation
    assert set(generation["style_variant_weights"].keys()) == set(SUPPORTED_PINBALL_STYLE_VARIANTS)
    assert len(SUPPORTED_PINBALL_STYLE_VARIANTS) >= 5
    assert str(prompt["bundle_id"]) == "games_pinball_table_v1"
    prompt_defaults = required_group_defaults(
        prompt,
        (
            "pinball_motion_rule_text",
            "object_description_scoreable_count",
            "annotation_hint_first_hit_object_label",
            "pinball_scoreable_rule_text",
            "annotation_hint_scoreable_object_count",
        ),
        context="pinball prompt asset defaults",
    )
    assert "straight path" in str(prompt_defaults["pinball_motion_rule_text"]).lower()
    assert "numeric score targets" in str(prompt_defaults["object_description_scoreable_count"]).lower()
    assert "[x, y] pixel point" in str(prompt_defaults["annotation_hint_first_hit_object_label"])
    assert "without a number" in str(prompt_defaults["pinball_scoreable_rule_text"]).lower()
    assert "numeric score label" in str(prompt_defaults["annotation_hint_scoreable_object_count"]).lower()
    assert "bounding boxes" in str(prompt_defaults["annotation_hint_scoreable_object_count"])


def test_games_pinball_first_hit_emits_expected_contract() -> None:
    out = GamesPinballFirstHitObjectLabelTask().generate(
        98100,
        params={"object_count": 8, "target_object_label": "H", "style_variant": "neon"},
        max_attempts=512,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point"
    assert len(out.annotation_gt.value) == 2
    assert out.query_id in NO_BRANCH_QUERY_IDS
    assert out.scene_id == "pinball_table"
    assert trace["query_spec"]["query_id"] in NO_BRANCH_QUERY_IDS
    assert execution["query_id"] in NO_BRANCH_QUERY_IDS
    assert trace["projected_annotation"]["type"] == "point"
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point"] == out.annotation_gt.value
    assert trace["render_map"]["playfield_projection"]["kind"] == "trapezoid_isometric"
    assert trace["render_map"]["decorative_entities"]
    assert "panel_scene_style" in trace["render_spec"]
    assert "text_style" in trace["render_spec"]
    assert len(execution["annotation_entity_ids"]) == 1
    decorative_ids = {str(entity["id"]) for entity in trace["render_map"]["decorative_entities"]}
    assert not decorative_ids.intersection(str(entity_id) for entity_id in execution["annotation_entity_ids"])


def test_games_pinball_first_hit_matches_recomputed_ray_hit() -> None:
    out = GamesPinballFirstHitObjectLabelTask().generate(
        98110,
        params={"object_count": 8, "target_object_label": "G", "style_variant": "classic"},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    target_id = str(execution["target_object_id"])
    ball = tuple(float(value) for value in execution["ball_xy_norm"])
    objects = tuple(
        PinballObject(
            object_id=str(obj["object_id"]),
            label=str(obj["label"]),
            kind=str(obj["kind"]),
            x_norm=float(obj["x_norm"]),
            y_norm=float(obj["y_norm"]),
            radius_norm=float(obj["radius_norm"]),
            width_norm=float(obj["width_norm"]),
            height_norm=float(obj["height_norm"]),
            color_index=0,
        )
        for obj in execution["objects"]
    )

    first_hit = first_hit_object_id(
        origin=ball,
        angle_rad=float(execution["cue_angle_rad"]),
        objects=objects,
    )
    assert str(first_hit) == target_id
    assert str(out.answer_gt.value) == str(execution["target_object_label"])
    assert list(execution["annotation_entity_ids"]) == [target_id]
    assert 5 <= len(execution["objects"]) <= 8
    assert out.annotation_gt.value == out.trace_payload["render_map"]["entity_points_px"][target_id]


def test_games_pinball_scoreable_object_count_emits_expected_contract() -> None:
    out = GamesPinballScoreableObjectCountTask().generate(
        98200,
        params={
            "object_count": 8,
            "scoreable_object_count": 4,
            "style_variant": "neon",
        },
        max_attempts=512,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    annotation_entity_ids = [str(entity_id) for entity_id in execution["annotation_entity_ids"]]
    score_by_id = {
        str(obj["object_id"]): obj["score_value"]
        for obj in execution["objects"]
    }
    scoreable_ids = [
        str(obj["object_id"])
        for obj in execution["objects"]
        if obj["score_value"] is not None
    ]
    non_scoreable_ids = [
        str(obj["object_id"])
        for obj in execution["objects"]
        if obj["score_value"] is None
    ]

    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 4
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id in NO_BRANCH_QUERY_IDS
    assert out.scene_id == "pinball_table"
    assert trace["query_spec"]["query_id"] in NO_BRANCH_QUERY_IDS
    assert execution["query_id"] in NO_BRANCH_QUERY_IDS
    assert execution["scoreable_object_count"] == 4
    assert sorted(annotation_entity_ids) == sorted(scoreable_ids)
    assert non_scoreable_ids
    assert all(score_by_id[entity_id] is not None for entity_id in annotation_entity_ids)
    assert all(score_by_id[entity_id] is None for entity_id in non_scoreable_ids)
    assert out.answer_gt.value == len(scoreable_ids)
    assert out.annotation_gt.value == [
        trace["render_map"]["entity_bboxes_px"][entity_id]
        for entity_id in annotation_entity_ids
    ]
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    for obj in execution["objects"]:
        if obj["score_value"] is None:
            assert obj["display_text"] is None
            assert obj["show_label"] is False
        else:
            assert str(obj["display_text"]) == str(int(obj["score_value"]))


def test_games_pinball_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__pinball_table"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__pinball_table",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__pinball_table__first_hit_object_label", count=1, params={}),
            BuildTaskConfig(task_id="task_games__pinball_table__scoreable_object_count", count=1, params={}),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-pinball-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 2
    assert {row["task"] for row in rows} == {
        "task_games__pinball_table__first_hit_object_label",
        "task_games__pinball_table__scoreable_object_count",
    }
    assert all(row["domain"] == "games" for row in rows)
    assert all(row["scene_id"] == "pinball_table" for row in rows)
    assert all(row.get("scene_id") for row in rows)
