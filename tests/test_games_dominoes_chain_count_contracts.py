"""Contract tests for games dominoes scene tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.dominoes.double_count import GamesDominoesDoubleCountTask
from trace_tasks.tasks.games.dominoes.higher_sum_than_reference_count import GamesDominoesHigherSumThanReferenceCountTask
from trace_tasks.tasks.games.dominoes.invalid_join_label import GamesDominoesInvalidJoinLabelTask
from trace_tasks.tasks.games.dominoes.longest_chain_length_value import GamesDominoesLongestChainLengthValueTask
from trace_tasks.tasks.games.dominoes.matching_end_count import GamesDominoesMatchingEndCountTask
from trace_tasks.tasks.games.dominoes.shared.rules import can_connect, chain_open_end_after_play
from trace_tasks.tasks.games.dominoes.sum_to_target_count import GamesDominoesSumToTargetCountTask
from tests.helpers import read_jsonl


TASK_CASES = (
    (GamesDominoesDoubleCountTask, "double_count", {"target_answer": 3, "candidate_count": 8}),
    (GamesDominoesHigherSumThanReferenceCountTask, "higher_sum_than_reference_count", {"target_answer": 4, "candidate_count": 9}),
    (GamesDominoesLongestChainLengthValueTask, "longest_chain_length_value", {"target_answer": 3, "candidate_count": 7}),
    (GamesDominoesMatchingEndCountTask, "matching_end_count", {"target_answer": 2, "candidate_count": 8}),
    (GamesDominoesSumToTargetCountTask, "sum_to_target_count", {"target_answer": 2, "candidate_count": 8, "target_total": 6}),
)
ALL_TASK_CASES = (
    *TASK_CASES,
    (GamesDominoesInvalidJoinLabelTask, "invalid_join_label", {"target_label": "D"}),
)
CHAIN_QUERY_IDS = frozenset(
    {
        "longest_chain_length_value",
        "matching_end_count",
    }
)
TABLEAU_QUERY_IDS = frozenset(
    {
        "double_count",
        "higher_sum_than_reference_count",
        "sum_to_target_count",
    }
)


@pytest.mark.parametrize(("task_cls", "query_id", "params"), TASK_CASES)
def test_games_dominoes_tasks_emit_expected_count_contract(task_cls, query_id: str, params: dict[str, int]) -> None:
    out = task_cls().generate(
        26001,
        params={"scene_variant": "single_row", **params},
        max_attempts=256,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(params["target_answer"])
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == int(params["target_answer"])
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["prompt_variant"]["query_key"] == query_id
    assert int(execution["target_answer"]) == int(params["target_answer"])
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["witness_symbolic"] == {"type": "object_set", "ids": list(execution["annotation_entity_ids"])}
    assert all(str(tile_id).startswith("candidate_") for tile_id in execution["annotation_entity_ids"])
    if query_id in CHAIN_QUERY_IDS:
        assert execution["layout_kind"] == "chain_tableau"
        assert trace["render_map"]["layout_kind"] == "chain_tableau"
        assert execution["chain_tile_specs"]
        assert trace["render_map"]["chain_tile_ids"]
        assert set(trace["render_map"]["section_label_bboxes_px"].keys()) == {"chain", "candidates"}
        assert len(trace["render_map"]["section_separator_bbox_px"]) == 4
    else:
        assert query_id in TABLEAU_QUERY_IDS
        assert execution["layout_kind"] == "tableau"
        assert trace["render_map"]["layout_kind"] == "tableau"
        assert execution["chain_tile_specs"] == []
        assert trace["render_map"]["chain_tile_ids"] == []
        assert "section_label_bboxes_px" not in trace["render_map"]
        assert "section_separator_bbox_px" not in trace["render_map"]
    width, height = out.image.size
    for bbox in trace["render_map"]["domino_bboxes_px"].values():
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)


def test_games_dominoes_matching_end_marks_reference_and_open_half() -> None:
    out = GamesDominoesMatchingEndCountTask().generate(
        26011,
        params={"scene_variant": "single_row", "target_answer": 2, "candidate_count": 8},
        max_attempts=256,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    reference_id = str(execution["reference_tile_id"])
    reference_spec = next(spec for spec in execution["chain_tile_specs"] if str(spec["tile_id"]) == reference_id)
    assert bool(reference_spec["is_reference"]) is True
    assert int(execution["open_end_value"]) in range(7)
    assert str(reference_id) in trace["render_map"]["reference_tag_bboxes_px"]


def test_games_dominoes_sum_to_target_records_sampled_target_total() -> None:
    out = GamesDominoesSumToTargetCountTask().generate(
        26021,
        params={"scene_variant": "single_row", "target_answer": 2, "candidate_count": 8, "target_total": 6},
        max_attempts=256,
    )
    execution = out.trace_payload["execution_trace"]
    assert int(execution["target_total"]) == 6
    assert "6" in out.prompt


def test_games_dominoes_higher_sum_reference_lives_in_tableau() -> None:
    out = GamesDominoesHigherSumThanReferenceCountTask().generate(
        26031,
        params={"scene_variant": "two_row", "target_answer": 4, "candidate_count": 10},
        max_attempts=256,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    reference_id = str(execution["reference_tile_id"])
    reference_specs = [spec for spec in execution["candidate_tile_specs"] if str(spec["tile_id"]) == reference_id]

    assert execution["layout_kind"] == "tableau"
    assert execution["chain_tile_specs"] == []
    assert len(reference_specs) == 1
    assert bool(reference_specs[0]["is_reference"]) is True
    assert reference_specs[0]["role"] == "reference_sum"
    assert str(reference_id) in trace["render_map"]["reference_tag_bboxes_px"]
    assert "top chain" not in out.prompt
    assert "below" not in out.prompt


def test_games_dominoes_longest_chain_length_matches_unique_annotation_set() -> None:
    out = GamesDominoesLongestChainLengthValueTask().generate(
        26041,
        params={"scene_variant": "single_row", "target_answer": 4, "candidate_count": 7},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    candidate_specs = [dict(spec) for spec in execution["candidate_tile_specs"]]
    open_end = int(execution["open_end_value"])
    longest_length, longest_sets = _longest_chain_id_sets(candidate_specs, open_end)

    assert int(out.answer_gt.value) == 4
    assert 1 <= int(out.answer_gt.value) <= 5
    assert int(longest_length) == int(out.answer_gt.value)
    assert longest_sets == {frozenset(str(tile_id) for tile_id in execution["annotation_entity_ids"])}
    assert set(execution["longest_chain_tile_ids"]) == set(execution["annotation_entity_ids"])
    assert "longest" in out.prompt.lower() or "maximum" in out.prompt.lower()
    assert "do not count `ref`" in out.prompt.lower()


def test_games_dominoes_invalid_join_label_has_six_options_and_one_invalid_join() -> None:
    out = GamesDominoesInvalidJoinLabelTask().generate(
        26061,
        params={"scene_variant": "single_row", "target_label": "D"},
        max_attempts=256,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert str(out.answer_gt.value) == "D"
    assert out.annotation_gt.type == "segment"
    assert trace["projected_annotation"]["type"] == "segment"
    assert trace["projected_annotation"]["segment"] == out.annotation_gt.value
    assert out.annotation_gt.value == render_map["chain_join_endpoint_points_px"]["D"]
    assert trace["query_spec"]["prompt_variant"]["query_key"] == "invalid_join_label"
    assert execution["answer_option_label"] == "D"
    assert execution["annotation_entity_ids"] == ["join_D"]
    assert execution["invalid_join_tile_ids"] == ["chain_04", "chain_05"]
    assert execution["option_labels"] == list("ABCDEF")
    assert execution["candidate_tile_specs"] == []
    assert len(execution["chain_tile_specs"]) == 7
    assert len(render_map["chain_join_specs"]) == 6
    assert sorted(render_map["chain_join_label_bboxes_px"].keys()) == list("ABCDEF")

    invalid_specs = [spec for spec in render_map["chain_join_specs"] if not bool(spec["is_valid"])]
    assert [spec["option_label"] for spec in invalid_specs] == ["D"]
    for spec in render_map["chain_join_specs"]:
        left_tile = next(tile for tile in execution["chain_tile_specs"] if tile["tile_id"] == spec["left_tile_id"])
        right_tile = next(tile for tile in execution["chain_tile_specs"] if tile["tile_id"] == spec["right_tile_id"])
        expected_valid = int(left_tile["right_value"]) == int(right_tile["left_value"])
        assert bool(spec["is_valid"]) is expected_valid
    assert "loose dominoes" not in out.prompt.lower()
    assert "labeled joins" in out.prompt.lower() or "labeled join" in out.prompt.lower()


def test_games_dominoes_chain_tiles_are_closer_than_loose_dominoes() -> None:
    out = GamesDominoesMatchingEndCountTask().generate(
        26051,
        params={"scene_variant": "single_row", "target_answer": 2, "candidate_count": 8},
        max_attempts=256,
    )
    render_map = out.trace_payload["render_map"]
    domino_bboxes = render_map["domino_bboxes_px"]
    chain_ids = list(render_map["chain_tile_ids"])
    row_ids = list(render_map["candidate_row_ids"][0])

    chain_gaps = [
        float(domino_bboxes[right_id][0]) - float(domino_bboxes[left_id][2])
        for left_id, right_id in zip(chain_ids, chain_ids[1:])
    ]
    loose_gaps = [
        float(domino_bboxes[right_id][0]) - float(domino_bboxes[left_id][2])
        for left_id, right_id in zip(row_ids, row_ids[1:])
    ]

    assert min(chain_gaps) > 0.0
    assert max(chain_gaps) < min(loose_gaps)


def test_games_dominoes_each_public_task_varies_scene_and_style_axes() -> None:
    for task_cls, query_id, _params in ALL_TASK_CASES:
        scenes: set[str] = set()
        styles: set[str] = set()
        answers: set[str] = set()
        for sampling_index in range(64):
            out = task_cls().generate(
                26101 + int(sampling_index),
                params={},
                max_attempts=512,
            )
            execution = out.trace_payload["execution_trace"]
            scenes.add(str(execution["scene_variant"]))
            styles.add(str(execution["style_variant"]))
            answers.add(str(out.answer_gt.value))
        assert scenes == {"single_row", "two_row"}
        assert len(styles) >= 5
        assert len(answers) >= 3


def test_games_dominoes_tasks_are_deterministic() -> None:
    params = {"scene_variant": "two_row", "target_answer": 5, "candidate_count": 12}
    task = GamesDominoesHigherSumThanReferenceCountTask()
    out_a = task.generate(26031, params=params, max_attempts=256)
    out_b = task.generate(26031, params=params, max_attempts=256)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_dominoes_prompt_bundle_declares_static_rule_slots() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/games/dominoes/games_dominoes_v1.json").read_text(encoding="utf-8"))
    required = bundle["required_slots_by_key"]
    static = bundle["static_slots_by_key"]
    dynamic = bundle["dynamic_slots"]
    assert required["query:matching_end_count"] == ["connection_rule_text"]
    assert required["query:longest_chain_length_value"] == ["longest_chain_rule_text"]
    assert required["query:invalid_join_label"] == ["invalid_join_rule_text"]
    assert required["query:sum_to_target_count"] == ["pip_sum_rule_text", "target_total_text"]
    assert "connection_rule_text" in static["query:matching_end_count"]
    assert "longest_chain_rule_text" in static["query:longest_chain_length_value"]
    assert "invalid_join_rule_text" in static["query:invalid_join_label"]
    assert "object_description" not in static["scene:visible_domino_chain"]
    assert "connection_rule_text" not in dynamic
    assert "object_description" in dynamic


def test_games_dominoes_prompt_avoids_table_color_and_sentence_splice() -> None:
    out = GamesDominoesMatchingEndCountTask().generate(
        20260509,
        params={"scene_variant": "two_row", "target_answer": 3, "candidate_count": 12},
        max_attempts=256,
    )

    assert "green table" not in out.prompt
    assert "Using the `REF` tile at the end of the chain, A loose" not in out.prompt


def _longest_chain_id_sets(candidate_specs: list[dict], open_end_value: int) -> tuple[int, set[frozenset[str]]]:
    tiles = tuple(
        (
            str(spec["tile_id"]),
            (int(spec["left_value"]), int(spec["right_value"])),
        )
        for spec in candidate_specs
    )

    def search(current_open_end: int, remaining_indices: tuple[int, ...]) -> tuple[int, set[frozenset[str]]]:
        best_length = 0
        best_sets: set[frozenset[str]] = {frozenset()}
        for index in remaining_indices:
            tile_id, tile = tiles[int(index)]
            if not can_connect(tile, int(current_open_end)):
                continue
            next_open_end = chain_open_end_after_play(tile, int(current_open_end))
            next_remaining = tuple(other for other in remaining_indices if int(other) != int(index))
            child_length, child_sets = search(int(next_open_end), next_remaining)
            total_length = int(child_length) + 1
            total_sets = {frozenset({str(tile_id), *child_set}) for child_set in child_sets}
            if int(total_length) > int(best_length):
                best_length = int(total_length)
                best_sets = total_sets
            elif int(total_length) == int(best_length):
                best_sets.update(total_sets)
        return int(best_length), best_sets

    return search(int(open_end_value), tuple(range(len(tiles))))


def test_games_dominoes_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__dominoes__double_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__dominoes__double_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_games__dominoes__double_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=256,
        sampling_seed=59,
    )
    final_path = build_dataset(config, code_hash="games-dominoes-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "games" for record in train_records)
    assert all(record.get("scene_id") == "dominoes" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_games__dominoes__double_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
