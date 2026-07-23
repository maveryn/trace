"""Contract tests for games Rhythm-lanes tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.registry import create_task
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_id", "params", "prompt_query_key", "annotation_type"),
    (
        (
            "task_games__rhythm__lane_note_count",
            {"lane_count": 6, "row_count": 12, "beat_window": 6, "target_note_count": 3},
            "lane_note_count",
            "bbox_set",
        ),
        (
            "task_games__rhythm__lane_note_score_value",
            {
                "lane_count": 6,
                "row_count": 12,
                "beat_window": 6,
                "target_score": 12,
            },
            "lane_note_score_value",
            "bbox_set",
        ),
        (
            "task_games__rhythm__most_notes_lane_label",
            {"lane_count": 7, "row_count": 13, "beat_window": 5, "target_note_count": 4},
            "most_notes_lane_label",
            "bbox_set",
        ),
        (
            "task_games__rhythm__earliest_hit_lane_label",
            {"lane_count": 7, "row_count": 13, "beat_window": 5},
            "earliest_hit_lane_label",
            "bbox",
        ),
    ),
)
def test_games_rhythm_public_tasks_emit_expected_contract(
    task_id: str,
    params: dict[str, int | str],
    prompt_query_key: str,
    annotation_type: str,
) -> None:
    out = create_task(task_id).generate(98400, params=params, max_attempts=512)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == annotation_type
    assert out.query_id == "single"
    assert out.scene_id == "rhythm"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["prompt_query_key"] == prompt_query_key
    assert execution["query_id"] == "single"
    assert execution["prompt_query_key"] == prompt_query_key
    assert trace["projected_annotation"]["type"] == annotation_type
    if annotation_type == "bbox":
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    else:
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert len(out.annotation_gt.value) >= 1


def test_games_rhythm_lane_note_score_value_matches_trace() -> None:
    out = create_task("task_games__rhythm__lane_note_score_value").generate(
        98410,
        params={
            "lane_count": 6,
            "row_count": 12,
            "beat_window": 6,
            "target_score": 12,
        },
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    selected_lane = int(execution["selected_lane_index"])
    score_values = {str(color): int(value) for color, value in execution["score_values_by_color"].items()}
    expected = [
        note
        for note in execution["notes"]
        if int(note["lane_index"]) == selected_lane
    ]
    expected_score = sum(score_values[str(note["color_key"])] for note in expected)

    assert int(out.answer_gt.value) == expected_score == 12
    assert set(score_values.values()) == {1, 2, 3}
    assert set(execution["annotation_entity_ids"]) == {str(note["note_id"]) for note in expected}
    assert len(out.annotation_gt.value) == len(expected)
    assert len(expected) <= 4
    assert out.trace_payload["render_map"]["score_palette"]["values_by_color"] == score_values
    assert "POINTS palette" in out.prompt
    assert f"lane {execution['selected_lane_label']}" in out.prompt
    assert "tall note counts once" in out.prompt
    assert "add the values for its note colors" not in out.prompt
    assert '"answer":5' in out.prompt


def test_games_rhythm_most_notes_lane_label_matches_trace() -> None:
    out = create_task("task_games__rhythm__most_notes_lane_label").generate(
        98420,
        params={"lane_count": 8, "row_count": 14, "beat_window": 7, "target_note_count": 5},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    note_counts: list[int] = []
    for lane in range(int(execution["lane_count"])):
        note_counts.append(sum(1 for note in execution["notes"] if int(note["lane_index"]) == lane))
    expected_lane = note_counts.index(max(note_counts))
    expected_ids = {
        str(note["note_id"])
        for note in execution["notes"]
        if int(note["lane_index"]) == expected_lane
    }

    assert note_counts.count(max(note_counts)) == 1
    assert int(out.answer_gt.value) == expected_lane + 1
    assert set(execution["annotation_entity_ids"]) == expected_ids


def test_games_rhythm_earliest_hit_lane_label_uses_scalar_bbox() -> None:
    out = create_task("task_games__rhythm__earliest_hit_lane_label").generate(
        98430,
        params={"lane_count": 7, "row_count": 13, "beat_window": 5},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    hit_notes = [
        note
        for note in execution["notes"]
        if int(note["bottom_row_from_hit_line"]) <= int(execution["beat_window"])
    ]
    earliest = min(int(note["bottom_row_from_hit_line"]) for note in hit_notes)
    earliest_notes = [note for note in hit_notes if int(note["bottom_row_from_hit_line"]) == earliest]

    assert out.annotation_gt.type == "bbox"
    assert len(earliest_notes) == 1
    assert int(out.answer_gt.value) == int(earliest_notes[0]["lane_label"])
    assert execution["annotation_entity_ids"] == [str(earliest_notes[0]["note_id"])]


def test_games_rhythm_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__rhythm"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__rhythm",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__rhythm__lane_note_count", count=1, params={}),
            BuildTaskConfig(task_id="task_games__rhythm__most_notes_lane_label", count=1, params={}),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-rhythm-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 2
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "rhythm" for row in rows)
