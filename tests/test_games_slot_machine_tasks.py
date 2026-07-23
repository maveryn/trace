"""Contract tests for games slot-machine tasks."""

from __future__ import annotations

from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.slot_machine.paytable_score_value import GamesSlotMachinePaytableScoreValueTask
from trace_tasks.tasks.games.slot_machine.reel_completion_label import GamesSlotMachineReelCompletionLabelTask
from trace_tasks.tasks.games.slot_machine.winning_payline_count import GamesSlotMachineWinningPaylineCountTask


def test_games_slot_machine_winning_payline_count_contract() -> None:
    out = GamesSlotMachineWinningPaylineCountTask().generate(
        26062401,
        params={"target_winning_payline_count": 2},
        max_attempts=64,
    )
    execution = out.trace_payload["execution_trace"]
    winning_payline_ids = tuple(str(payline_id) for payline_id in execution["winning_payline_ids"])
    expected_segments = [
        out.trace_payload["render_map"]["payline_segments_px"][f"payline_{payline_id}"]
        for payline_id in winning_payline_ids
    ]

    assert out.scene_id == "slot_machine"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 2
    assert len(winning_payline_ids) == 2
    assert out.annotation_gt.type == "segment_set"
    assert out.annotation_gt.value == expected_segments
    assert out.trace_payload["projected_annotation"]["type"] == "segment_set"
    assert out.trace_payload["query_spec"]["params"]["target_winning_payline_count"] == 2
    assert execution["prompt_query_key"] == "winning_payline_count"


def test_games_slot_machine_winning_payline_count_support_and_taxonomy() -> None:
    task = GamesSlotMachineWinningPaylineCountTask()
    seen = set()
    for target in range(6):
        out = task.generate(
            26062410 + target,
            params={"target_winning_payline_count": target},
            max_attempts=64,
        )
        seen.add(int(out.answer_gt.value))
        assert int(out.answer_gt.value) == target
        assert len(out.annotation_gt.value) == target
        assert out.trace_payload["query_spec"]["params"]["winning_payline_count_support"] == [0, 1, 2, 3, 4, 5]

    taxonomy = resolve_task_taxonomy(
        "task_games__slot_machine__winning_payline_count",
        source_domain="games",
        source_scene_id="",
    )
    assert taxonomy.domain == "games"
    assert taxonomy.scene_id == "slot_machine"
    assert seen == {0, 1, 2, 3, 4, 5}


def test_games_slot_machine_paytable_score_value_contract() -> None:
    out = GamesSlotMachinePaytableScoreValueTask().generate(
        26062501,
        params={"target_score_winning_payline_count": 1},
        max_attempts=64,
    )
    execution = out.trace_payload["execution_trace"]
    winning_payline_ids = tuple(str(payline_id) for payline_id in execution["winning_payline_ids"])
    expected_segments = [
        out.trace_payload["render_map"]["payline_segments_px"][f"payline_{payline_id}"]
        for payline_id in winning_payline_ids
    ]
    expected_score = int(execution["winning_payline_score_details"][0]["score_value"])

    assert out.scene_id == "slot_machine"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == expected_score
    assert len(winning_payline_ids) == 1
    assert out.annotation_gt.type == "segment"
    assert out.annotation_gt.value == expected_segments[0]
    assert len(execution["paytable_scores"]) == 6
    assert out.trace_payload["query_spec"]["params"]["target_score_winning_payline_count"] == 1
    assert out.trace_payload["query_spec"]["params"]["score_task_winning_payline_count_support"] == [1]
    assert execution["prompt_query_key"] == "paytable_score_value"


def test_games_slot_machine_paytable_score_value_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(
        "task_games__slot_machine__paytable_score_value",
        source_domain="games",
        source_scene_id="",
    )
    assert taxonomy.domain == "games"
    assert taxonomy.scene_id == "slot_machine"


def test_games_slot_machine_reel_completion_label_contract() -> None:
    out = GamesSlotMachineReelCompletionLabelTask().generate(
        26062601,
        params={},
        max_attempts=64,
    )
    execution = out.trace_payload["execution_trace"]
    answer_label = str(out.answer_gt.value)
    assert [str(record["label"]) for record in execution["options"]] == ["A", "B", "C", "D"]
    option_records = {str(record["label"]): record for record in execution["options"]}

    assert out.scene_id == "slot_machine"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert answer_label in {"A", "B", "C", "D"}
    assert out.annotation_gt.type == "bbox"
    assert out.annotation_gt.value == out.trace_payload["render_map"]["option_bboxes_px"][answer_label]
    assert len(option_records[answer_label]["completed_payline_ids"]) == 1
    assert all(
        len(record["completed_payline_ids"]) == (1 if label == answer_label else 0)
        for label, record in option_records.items()
    )
    assert execution["prompt_query_key"] == "reel_completion_label"

    taxonomy = resolve_task_taxonomy(
        "task_games__slot_machine__reel_completion_label",
        source_domain="games",
        source_scene_id="",
    )
    assert taxonomy.domain == "games"
    assert taxonomy.scene_id == "slot_machine"
