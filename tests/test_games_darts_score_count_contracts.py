"""Contract tests for simplified darts games tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.darts.bullseye_membership_count import GamesDartsBullseyeMembershipCountTask
from trace_tasks.tasks.games.darts.dart_score_value import GamesDartsDartScoreValueTask
from trace_tasks.tasks.games.darts.highest_scoring_dart_label import GamesDartsHighestScoringDartLabelTask
from trace_tasks.tasks.games.darts.shared.rendering import dartboard_anchor_colors
from trace_tasks.tasks.shared.color_distance import color_distance
from tests.helpers import read_jsonl


def test_games_darts_score_value_uses_only_visible_dart_as_annotation() -> None:
    out = GamesDartsDartScoreValueTask().generate(
        92011,
        params={"scene_variant": "single_board", "target_score": 50},
        max_attempts=48,
    )
    execution = out.trace_payload["execution_trace"]
    marked = [spec for spec in execution["dart_specs"] if bool(spec["is_marked"])]
    visible_darts = list(execution["dart_specs"])

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 50
    assert out.annotation_gt.type == "point"
    assert len(out.annotation_gt.value) == 2
    assert len(visible_darts) == 1
    assert not marked
    assert visible_darts[0]["area_kind"] == "bullseye"
    assert int(visible_darts[0]["score"]) == int(out.answer_gt.value)
    assert execution["annotation_entity_ids"] == [str(visible_darts[0]["dart_id"])]
    assert out.trace_payload["projected_annotation"]["point"] == out.annotation_gt.value


@pytest.mark.parametrize("target_answer", (0, 1, 2, 3))
def test_games_darts_highest_scoring_dart_label_contract(target_answer: int) -> None:
    out = GamesDartsHighestScoringDartLabelTask().generate(
        92031 + int(target_answer),
        params={
            "scene_variant": "single_board",
            "target_answer": int(target_answer),
        },
        max_attempts=64,
    )
    execution = out.trace_payload["execution_trace"]
    specs = list(execution["dart_specs"])
    specs_by_label = {str(spec["label"]): spec for spec in specs}
    correct_label = "ABCD"[int(target_answer)]
    correct_spec = specs_by_label[correct_label]

    assert out.answer_gt.type == "option_letter"
    assert str(out.answer_gt.value) == correct_label
    assert out.annotation_gt.type == "point"
    assert set(specs_by_label) == {"A", "B", "C", "D"}
    assert len(specs) == 4
    assert all(str(spec["area_kind"]) == "sector" for spec in specs)
    assert execution["correct_option_label"] == correct_label
    assert execution["correct_dart_id"] == correct_spec["dart_id"]
    assert execution["annotation_entity_ids"] == [str(correct_spec["dart_id"])]
    assert execution["scores_by_label"] == {
        str(spec["label"]): int(spec["score"])
        for spec in specs
    }
    assert all(
        int(correct_spec["score"]) > int(spec["score"])
        for spec in specs
        if str(spec["label"]) != correct_label
    )
    assert out.trace_payload["projected_annotation"]["point"] == out.annotation_gt.value


@pytest.mark.parametrize(
    ("query_id", "target_answer", "expected_membership"),
    (
        ("inside_bullseye_count", 3, "inside"),
        ("outside_bullseye_count", 4, "outside"),
    ),
)
def test_games_darts_bullseye_membership_count_contract(
    query_id: str,
    target_answer: int,
    expected_membership: str,
) -> None:
    out = GamesDartsBullseyeMembershipCountTask().generate(
        92021,
        params={
            "scene_variant": "single_board",
            "query_id": query_id,
            "target_answer": target_answer,
            "dart_count": 7,
        },
        max_attempts=48,
    )
    execution = out.trace_payload["execution_trace"]
    specs_by_id = {str(spec["dart_id"]): spec for spec in execution["dart_specs"]}
    annotated_specs = [specs_by_id[str(dart_id)] for dart_id in execution["annotation_entity_ids"]]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(target_answer)
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == int(target_answer)
    assert execution["bullseye_membership"] == expected_membership
    if expected_membership == "inside":
        assert all(spec["area_kind"] == "bullseye" for spec in annotated_specs)
    else:
        assert all(spec["area_kind"] == "sector" for spec in annotated_specs)
    assert out.trace_payload["projected_annotation"]["bbox_set"] == out.annotation_gt.value


def test_games_darts_query_cycle_covers_simplified_answers() -> None:
    score_answers: set[int] = set()
    inside_answers: set[int] = set()
    outside_answers: set[int] = set()
    for sampling_index in range(80):
        score = GamesDartsDartScoreValueTask().generate(
            92101 + int(sampling_index),
            params={"_sample_cursor": int(sampling_index)},
            max_attempts=64,
        )
        inside = GamesDartsBullseyeMembershipCountTask().generate(
            92201 + int(sampling_index),
            params={"_sample_cursor": int(sampling_index), "query_id": "inside_bullseye_count"},
            max_attempts=64,
        )
        outside = GamesDartsBullseyeMembershipCountTask().generate(
            92301 + int(sampling_index),
            params={"_sample_cursor": int(sampling_index), "query_id": "outside_bullseye_count"},
            max_attempts=64,
        )
        score_answers.add(int(score.answer_gt.value))
        inside_answers.add(int(inside.answer_gt.value))
        outside_answers.add(int(outside.answer_gt.value))

    assert score_answers == set(range(1, 11)) | {50}
    assert {0, 1, 2, 3, 4, 5}.issubset(inside_answers)
    assert {0, 1, 2, 3, 4, 5}.issubset(outside_answers)

    label_answers = {
        str(
            GamesDartsHighestScoringDartLabelTask().generate(
                92401 + int(sampling_index),
                params={"_sample_cursor": int(sampling_index)},
                max_attempts=64,
            ).answer_gt.value
        )
        for sampling_index in range(80)
    }
    assert label_answers == {"A", "B", "C", "D"}


def test_games_darts_score_value_is_deterministic() -> None:
    params = {
        "scene_variant": "single_board",
        "target_score": 7,
    }
    task = GamesDartsDartScoreValueTask()
    out_a = task.generate(92041, params=params, max_attempts=48)
    out_b = task.generate(92041, params=params, max_attempts=48)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_darts_marker_color_is_lab_separated_from_board() -> None:
    out = GamesDartsBullseyeMembershipCountTask().generate(
        92051,
        params={
            "scene_variant": "single_board",
            "query_id": "outside_bullseye_count",
            "target_answer": 3,
            "dart_count": 7,
        },
        max_attempts=48,
    )
    render_spec = out.trace_payload["render_spec"]
    color = tuple(int(v) for v in render_spec["dart_fill_color"])
    anchors = dartboard_anchor_colors(str(out.trace_payload["execution_trace"]["style_variant"]))
    distances = [float(color_distance(color, anchor, distance_space="lab")) for anchor in anchors]
    assert min(distances) >= 40.0


def test_games_darts_prompt_bundle_uses_simplified_terms() -> None:
    bundle_text = Path("src/trace_tasks/resources/prompts/games/darts/games_darts_v1.json").read_text(encoding="utf-8")
    assert "double ring" not in bundle_text
    assert "triple ring" not in bundle_text
    assert "outer bull" not in bundle_text
    assert "inner bull" not in bundle_text

    bundle = json.loads(bundle_text)
    required = bundle["required_slots_by_key"]
    assert required["query:dart_score_value"] == ["scoring_rule_text"]
    assert required["query:inside_bullseye_count"] == []
    assert required["query:outside_bullseye_count"] == []
    assert required["query:highest_scoring_dart_label"] == ["scoring_rule_text"]
    assert bundle["static_slots_by_key"]["scene:visible_dartboard"]["object_description"]


def test_games_darts_score_prompt_asks_for_integer_score_not_option_letter() -> None:
    out = GamesDartsDartScoreValueTask().generate(
        92061,
        params={"scene_variant": "single_board", "target_score": 10},
        max_attempts=48,
    )

    assert "the dart" in out.prompt
    assert "marked dart" not in out.prompt
    assert "option letter" not in out.prompt
    assert "scores 50" in out.prompt


def test_games_darts_score_value_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__darts__dart_score_value"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__darts__dart_score_value",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__darts__dart_score_value",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=48,
        sampling_seed=71,
    )
    final_path = build_dataset(config, code_hash="games-darts-score-value-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record["task"] == "task_games__darts__dart_score_value" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_games__darts__dart_score_value"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
