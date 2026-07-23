"""Contract tests for sliding-block game tasks."""

from __future__ import annotations

from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.games.sliding_block.block_orientation_count import GamesSlidingBlockOrientationCountTask
from trace_tasks.tasks.games.sliding_block.movable_block_count import GamesSlidingBlockMovableBlockCountTask
from trace_tasks.tasks.games.sliding_block.shared.rules import block_ids_by_orientation
from trace_tasks.tasks.games.sliding_block.shared.rules import legal_moves
from trace_tasks.tasks.games.sliding_block.shared.state import BlockSpec
from trace_tasks.tasks.games.sliding_block.sliding_block_blocker_count import GamesSlidingBlockBlockerCountTask
from trace_tasks.tasks.games.sliding_block.sliding_block_move_result_label import GamesSlidingBlockMoveResultLabelTask


TASKS = (
    (
        "task_games__sliding_block__block_orientation_count",
        GamesSlidingBlockOrientationCountTask,
        "horizontal_block_count",
        "integer",
        "horizontal_block_count",
    ),
    (
        "task_games__sliding_block__block_orientation_count",
        GamesSlidingBlockOrientationCountTask,
        "vertical_block_count",
        "integer",
        "vertical_block_count",
    ),
    (
        "task_games__sliding_block__sliding_block_blocker_count",
        GamesSlidingBlockBlockerCountTask,
        "single",
        "integer",
        "blocker_count",
    ),
    (
        "task_games__sliding_block__movable_block_count",
        GamesSlidingBlockMovableBlockCountTask,
        "single",
        "integer",
        "movable_block_count",
    ),
    (
        "task_games__sliding_block__sliding_block_move_result_label",
        GamesSlidingBlockMoveResultLabelTask,
        "single",
        "option_letter",
        "move_result_label",
    ),
)


def _execution_blocks(execution) -> list[BlockSpec]:
    return [
        BlockSpec(
            str(block["block_id"]),
            str(block["label"]),
            int(block["row"]),
            int(block["col"]),
            int(block["height"]),
            int(block["width"]),
            str(block["role"]),
            tuple(int(value) for value in block["fill_rgb"]),
        )
        for block in execution["blocks"]
    ]


def _bbox_row_counts(bboxes: dict[str, list[float]], *, tolerance_px: float = 4.0) -> tuple[int, ...]:
    centers = sorted((float(bbox[1]) + float(bbox[3])) / 2.0 for bbox in bboxes.values())
    rows: list[list[float]] = []
    for center_y in centers:
        if rows and abs(float(rows[-1][0]) - center_y) <= float(tolerance_px):
            rows[-1].append(center_y)
        else:
            rows.append([center_y])
    return tuple(len(row) for row in rows)


def test_sliding_block_tasks_are_registered_and_taxonomy_mapped() -> None:
    for task_id, task_cls, _query_id, _answer_type, _prompt_query_key in TASKS:
        assert TASK_REGISTRY[task_id] is task_cls
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "sliding_block"


def test_sliding_block_tasks_emit_contracts() -> None:
    for index, (_task_id, task_cls, query_id, answer_type, prompt_query_key) in enumerate(TASKS):
        out = task_cls().generate(2026052700 + index, params={"query_id": query_id}, max_attempts=30)
        trace = out.trace_payload
        execution = trace["execution_trace"]

        assert out.scene_id == "sliding_block"
        assert out.query_id == query_id
        assert out.answer_gt.type == answer_type
        assert out.annotation_gt.type == "bbox_set"
        assert trace["query_spec"]["params"]["query_id"] == query_id
        assert trace["render_spec"]["scene_id"] == "sliding_block"
        assert execution["question_format"] == prompt_query_key
        if prompt_query_key == "move_result_label":
            assert trace["render_map"]["annotation_source"] == "block_bboxes_px+option_panel_bboxes_px"
        else:
            assert trace["render_map"]["annotation_source"] == "block_bboxes_px"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert out.image.size == (
            int(trace["render_spec"]["canvas_width"]),
            int(trace["render_spec"]["canvas_height"]),
        )
        if prompt_query_key == "blocker_count":
            assert len(out.annotation_gt.value) == len(execution["answer_block_ids"])
            assert int(out.answer_gt.value) == len(execution["blocking_block_ids"])
            assert execution["answer_block_ids"] == execution["blocking_block_ids"]
        elif prompt_query_key in {"horizontal_block_count", "vertical_block_count"}:
            blocks = _execution_blocks(execution)
            orientation = "horizontal" if prompt_query_key == "horizontal_block_count" else "vertical"
            expected_ids = block_ids_by_orientation(blocks, orientation=orientation)
            assert int(out.answer_gt.value) == len(expected_ids)
            assert execution["answer_block_ids"] == expected_ids
            assert execution["orientation_block_ids"] == expected_ids
            assert len(out.annotation_gt.value) == len(expected_ids)
        elif prompt_query_key == "movable_block_count":
            blocks = _execution_blocks(execution)
            moves = legal_moves(
                blocks,
                rows=int(execution["rows"]),
                cols=int(execution["cols"]),
                max_distance=1,
            )
            movable_ids = {
                str(move.block_id)
                for move in moves
            }
            expected_ids = [
                str(block.block_id)
                for block in blocks
                if str(block.block_id) != "target" and str(block.block_id) in movable_ids
            ]
            assert int(out.answer_gt.value) == len(expected_ids)
            assert execution["answer_block_ids"] == expected_ids
            assert execution["answer_block_ids"] == execution["movable_block_ids"]
            assert len(out.annotation_gt.value) == len(expected_ids)
        else:
            correct_options = [option for option in execution["option_boards"] if option["is_correct"]]
            assert len(correct_options) == 1
            assert str(out.answer_gt.value) == str(correct_options[0]["label"])
            assert 1 <= len(execution["move_sequence"]) <= 3
            assert execution["answer_block_ids"] == execution["moved_block_ids"]
            assert len(out.annotation_gt.value) == len(execution["moved_block_ids"]) + 1
        for bbox in out.annotation_gt.value:
            assert len(bbox) == 4
            assert bbox[0] < bbox[2]
            assert bbox[1] < bbox[3]


def test_sliding_block_move_result_lays_out_four_options_as_two_by_two() -> None:
    out = GamesSlidingBlockMoveResultLabelTask().generate(
        2026052799,
        params={"option_count": 4},
        max_attempts=128,
    )
    option_bboxes = out.trace_payload["render_map"]["option_panel_bboxes_px"]

    assert set(option_bboxes) == {"option_A", "option_B", "option_C", "option_D"}
    assert _bbox_row_counts(option_bboxes) == (2, 2)


def test_sliding_block_blocker_count_can_emit_zero() -> None:
    out = GamesSlidingBlockBlockerCountTask().generate(
        2026052788,
        params={"blocker_count_min": 0, "blocker_count_max": 0},
        max_attempts=128,
    )

    assert out.answer_gt.value == 0
    assert out.annotation_gt.type == "bbox_set"
    assert out.annotation_gt.value == []
    assert out.trace_payload["execution_trace"]["answer_block_ids"] == []
