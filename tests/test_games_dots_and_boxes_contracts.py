"""Contract tests for the games dots-and-boxes scene tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.dots_and_boxes.completable_box_label import (
    GamesDotsAndBoxesCompletableBoxLabelTask,
)
from trace_tasks.tasks.games.dots_and_boxes.owned_box_count import (
    GamesDotsAndBoxesOwnedBoxCountTask,
)
from trace_tasks.tasks.games.dots_and_boxes.three_sided_box_count import (
    GamesDotsAndBoxesThreeSidedBoxCountTask,
)
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "query_id", "target_answer", "prompt_query_key"),
    (
        (GamesDotsAndBoxesThreeSidedBoxCountTask, "single", 4, "three_sided_box_count"),
        (GamesDotsAndBoxesOwnedBoxCountTask, "single", 4, "owned_box_count"),
    ),
)
def test_games_dots_and_boxes_count_tasks_emit_expected_contract(
    task_cls: type[Any],
    query_id: str,
    target_answer: int,
    prompt_query_key: str,
) -> None:
    task = task_cls()
    out = task.generate(
        28101 + int(target_answer),
        params={
            "query_id": str(query_id),
            "target_answer": int(target_answer),
        },
        max_attempts=64,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(target_answer)
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["query_spec"]["params"]["query_id"] == out.query_id
    assert int(execution["target_answer"]) == int(target_answer)
    assert execution["query_id"] == out.query_id
    assert trace["render_spec"]["panel_scene_style"]
    assert trace["render_spec"]["text_style"]["font_asset"]["font_family"]
    assert trace["render_map"]["panel_scene_style"]
    assert trace["render_map"]["font_family"]
    assert len(out.annotation_gt.value) == int(target_answer)
    assert execution["branching_edge_ids"] == []
    assert execution["captured_box_ids"] == []

    if str(prompt_query_key) == "three_sided_box_count":
        assert out.annotation_gt.type == "bbox_set"
        assert (
            trace["query_spec"]["params"]["prompt_query_key"] == "three_sided_box_count"
        )
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert len(execution["counted_box_ids"]) == int(target_answer)
        assert all(
            execution["box_drawn_side_counts"][box_id] == 3
            for box_id in execution["counted_box_ids"]
        )
        for box_id, bbox in zip(execution["counted_box_ids"], out.annotation_gt.value):
            assert trace["render_map"]["box_bboxes_px"][str(box_id)] == pytest.approx(
                bbox
            )
    else:
        owner = str(execution["target_owner"])
        assert out.annotation_gt.type == "bbox_set"
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert len(execution["counted_box_ids"]) == int(target_answer)
        assert trace["render_map"]["box_owner_by_id"]
        assert all(
            trace["render_map"]["box_owner_by_id"][str(box_id)] == owner
            for box_id in execution["counted_box_ids"]
        )
        assert all(
            execution["box_drawn_side_counts"][box_id] == 4
            for box_id in execution["counted_box_ids"]
        )
        for box_id, bbox in zip(execution["counted_box_ids"], out.annotation_gt.value):
            assert trace["render_map"]["box_bboxes_px"][str(box_id)] == pytest.approx(
                bbox
            )


def test_games_dots_and_boxes_completable_box_label_contract() -> None:
    out = GamesDotsAndBoxesCompletableBoxLabelTask().generate(
        28121,
        params={"target_label": "C"},
        max_attempts=64,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["query_spec"]["params"]["prompt_query_key"] == "completable_box_label"
    assert execution["answer_label"] == "C"
    assert execution["answer"] == "C"
    assert execution["answer_box_id"] == execution["option_box_id_by_label"]["C"]
    assert execution["counted_box_ids"] == [execution["answer_box_id"]]
    assert out.annotation_gt.value == pytest.approx(
        render_map["box_bboxes_px"][execution["answer_box_id"]]
    )
    assert execution["option_count"] == 6
    assert execution["candidate_edge_ids"] == []
    assert execution["highlighted_edge_ids"] == []
    assert sorted(execution["option_box_id_by_label"]) == ["A", "B", "C", "D", "E", "F"]
    assert sorted(render_map["option_box_id_by_label"]) == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]
    assert set(render_map["option_label_bboxes_px"]) == {"A", "B", "C", "D", "E", "F"}
    visual_label_order = [
        str(label)
        for label, center in sorted(
            render_map["option_label_centers_px"].items(),
            key=lambda item: (float(item[1][1]), float(item[1][0])),
        )
    ]
    assert visual_label_order == ["A", "B", "C", "D", "E", "F"]
    assert execution["box_drawn_side_counts"][execution["answer_box_id"]] == 3
    for label, box_id in execution["option_box_id_by_label"].items():
        if label != "C":
            assert execution["box_drawn_side_counts"][box_id] < 3


def test_games_dots_and_boxes_completable_box_label_samples_all_answers() -> None:
    task = GamesDotsAndBoxesCompletableBoxLabelTask()
    answer_labels: Counter[str] = Counter()
    board_shapes: Counter[tuple[int, int]] = Counter()

    for index in range(48):
        out = task.generate(
            28125 + index,
            params={"_sample_cursor": index},
            max_attempts=64,
        )
        execution = out.trace_payload["execution_trace"]
        answer_labels[str(out.answer_gt.value)] += 1
        board_shapes[(int(execution["box_rows"]), int(execution["box_cols"]))] += 1

        assert str(out.answer_gt.value) in {"A", "B", "C", "D", "E", "F"}
        assert execution["option_count"] == 6
        assert sorted(execution["option_box_id_by_label"]) == [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]
        visual_label_order = [
            str(label)
            for label, center in sorted(
                out.trace_payload["render_map"]["option_label_centers_px"].items(),
                key=lambda item: (float(item[1][1]), float(item[1][0])),
            )
        ]
        assert visual_label_order == ["A", "B", "C", "D", "E", "F"]
        assert (
            execution["answer_box_id"]
            == execution["option_box_id_by_label"][str(out.answer_gt.value)]
        )
        assert execution["box_drawn_side_counts"][execution["answer_box_id"]] == 3

    assert set(answer_labels) == {"A", "B", "C", "D", "E", "F"}
    assert all(count > 0 for count in answer_labels.values())
    assert set(board_shapes.keys()) == {(3, 3), (3, 4), (4, 3), (4, 4)}


def test_games_dots_and_boxes_owned_box_task_samples_players() -> None:
    task = GamesDotsAndBoxesOwnedBoxCountTask()
    query_ids: Counter[str] = Counter()
    answers_by_owner: dict[str, set[int]] = {}

    index = 0
    for owner in ("A", "B"):
        for target_answer in range(9):
            index += 1
            out = task.generate(
                28160 + index,
                params={"target_owner": owner, "target_answer": target_answer},
                max_attempts=64,
            )
            execution = out.trace_payload["execution_trace"]
            query_ids[str(out.query_id)] += 1
            answers_by_owner.setdefault(str(execution["target_owner"]), set()).add(
                int(out.answer_gt.value)
            )

            assert str(out.query_id) == "single"
            assert execution["query_id"] == out.query_id
            assert execution["target_owner"] == owner
            assert out.annotation_gt.type == "bbox_set"
            assert out.trace_payload["projected_annotation"]["type"] == "bbox_set"
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
            assert len(execution["counted_box_ids"]) == int(out.answer_gt.value)
            assert all(
                execution["box_owner_by_id"][str(box_id)] == owner
                for box_id in execution["counted_box_ids"]
            )

    for index in range(18):
        out = task.generate(
            28220 + index,
            params={"_sample_cursor": index},
            max_attempts=64,
        )
        query_ids[str(out.query_id)] += 1

    assert set(query_ids) == {"single"}
    assert all(count > 0 for count in query_ids.values())
    assert answers_by_owner["A"] == set(range(9))
    assert answers_by_owner["B"] == set(range(9))


def test_games_dots_and_boxes_three_sided_task_uses_single_query_id() -> None:
    out = GamesDotsAndBoxesThreeSidedBoxCountTask().generate(
        28123,
        params={},
        max_attempts=64,
    )
    assert out.query_id == "single"
    assert (
        out.trace_payload["query_spec"]["params"]["prompt_query_key"]
        == "three_sided_box_count"
    )


def test_games_dots_and_boxes_tasks_are_deterministic() -> None:
    params = {
        "target_label": "E",
    }
    task = GamesDotsAndBoxesCompletableBoxLabelTask()
    out_a = task.generate(28131, params=params, max_attempts=64)
    out_b = task.generate(28131, params=params, max_attempts=64)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert (
        out_a.trace_payload["query_spec"]["prompt_variant"]
        == out_b.trace_payload["query_spec"]["prompt_variant"]
    )
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_dots_and_boxes_prompt_bundle_declares_variants() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/dots_and_boxes/games_dots_and_boxes_v1.json").read_text(
            encoding="utf-8"
        )
    )
    required = bundle["required_slots_by_key"]
    assert required["query:three_sided_box_count"] == []
    assert required["query:completable_box_label"] == []
    assert required["query:owned_box_count"] == ["target_owner"]


def test_games_dots_and_boxes_completable_box_label_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__dots_and_boxes__completable_box_label"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__dots_and_boxes__completable_box_label",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__dots_and_boxes__completable_box_label",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=64,
        sampling_seed=71,
    )
    final_path = build_dataset(
        config, code_hash="games-dots-and-boxes-completable-box-smoke"
    )
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record.get("scene_id") == "dots_and_boxes" for record in train_records)

    build_report = json.loads(
        (final_path / "build_report.json").read_text(encoding="utf-8")
    )
    assert (
        int(
            build_report["accepted_counts_by_task"][
                "task_games__dots_and_boxes__completable_box_label"
            ]
        )
        == 4
    )

    validation = json.loads(
        (final_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["total_errors"] == 0
