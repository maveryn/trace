"""Contracts for arithmetic-constraint puzzle source-layout tasks."""

from __future__ import annotations

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.puzzles.arithmetic_panel.equal_sum_line_constraint_value import (
    PuzzlesArithmeticEqualSumLineConstraintValueTask,
)
from trace_tasks.tasks.puzzles.arithmetic_panel.number_wall_value import (
    ADDITION_WALL_QUERY,
    PuzzlesArithmeticNumberWallValueTask,
)
from trace_tasks.tasks.puzzles.arithmetic_panel.operation_table_cell_value import (
    PuzzlesArithmeticOperationTableCellValueTask,
)
from trace_tasks.tasks.puzzles.arithmetic_panel.row_column_total_missing_value import (
    PuzzlesArithmeticRowColumnTotalMissingValueTask,
)
from trace_tasks.tasks.puzzles.arithmetic_panel.shared.state import SCENE_ID
from trace_tasks.tasks.puzzles.arithmetic_panel.vertical_arithmetic_hidden_digit_value import (
    ADDITION_QUERY,
    SUBTRACTION_QUERY,
    PuzzlesArithmeticVerticalArithmeticHiddenDigitValueTask,
)

TASK_SPECS = (
    (
        "task_puzzles__arithmetic_panel__equal_sum_line_constraint_value",
        PuzzlesArithmeticEqualSumLineConstraintValueTask,
        ("single",),
    ),
    (
        "task_puzzles__arithmetic_panel__vertical_arithmetic_hidden_digit_value",
        PuzzlesArithmeticVerticalArithmeticHiddenDigitValueTask,
        (ADDITION_QUERY, SUBTRACTION_QUERY),
    ),
    (
        "task_puzzles__arithmetic_panel__operation_table_cell_value",
        PuzzlesArithmeticOperationTableCellValueTask,
        ("single",),
    ),
    (
        "task_puzzles__arithmetic_panel__row_column_total_missing_value",
        PuzzlesArithmeticRowColumnTotalMissingValueTask,
        ("single",),
    ),
    (
        "task_puzzles__arithmetic_panel__number_wall_value",
        PuzzlesArithmeticNumberWallValueTask,
        ("single",),
    ),
)


def test_arithmetic_panel_tasks_are_registered() -> None:
    for task_id, task_cls, _queries in TASK_SPECS:
        assert TASK_REGISTRY[str(task_id)] is task_cls


def test_arithmetic_panel_task_emits_public_contract() -> None:
    task = PuzzlesArithmeticEqualSumLineConstraintValueTask()
    out = task.generate(2026052301, params={}, max_attempts=80)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox"
    assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
    assert trace["query_spec"]["params"]["query_id"] == out.query_id
    assert trace["render_spec"]["scene_id"] == SCENE_ID
    assert trace["render_spec"]["text_style"]["font"]["font_family"]
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert execution["scene_id"] == SCENE_ID
    assert execution["target_item_id"] == "target"
    assert out.annotation_gt.value == trace["render_map"]["item_bboxes_px"]["target"]
    assert out.image.size == (
        int(trace["render_spec"]["canvas_width"]),
        int(trace["render_spec"]["canvas_height"]),
    )
    bbox = out.annotation_gt.value
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= out.image.size[0]
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= out.image.size[1]


def test_arithmetic_panel_scene_tasks_use_scalar_target_bbox_annotation() -> None:
    for task_id, task_cls, queries in TASK_SPECS:
        assert TASK_REGISTRY[str(task_id)] is task_cls
        task = task_cls()
        assert tuple(task.supported_query_ids) == tuple(queries)
        for index, branch in enumerate(queries):
            out = task.generate(
                2026052400 + (101 * len(str(task_id))) + index,
                params={"query_id": str(branch)},
                max_attempts=120,
            )
            item_bboxes = out.trace_payload["render_map"]["item_bboxes_px"]

            assert out.query_id == str(branch)
            assert out.annotation_gt.type == "bbox"
            assert out.annotation_gt.value == item_bboxes["target"]
            assert out.annotation_gt.value != item_bboxes["diagram_panel"]


def test_forced_arithmetic_panel_branches_are_valid() -> None:
    task_specs = (
        (PuzzlesArithmeticEqualSumLineConstraintValueTask(), ("single",)),
    )
    for task, branches in task_specs:
        for index, branch in enumerate(branches):
            out = task.generate(
                2026052310 + index,
                params={"query_id": branch},
                max_attempts=120,
            )
            trace = out.trace_payload["execution_trace"]
            data = trace["constraint_data"]

            assert out.query_id == branch
            assert int(out.answer_gt.value) == int(trace["answer_value"])

            if trace["layout_style"] == "polygon_side_sum":
                side_total = int(data["side_total"])
                corners = [int(value) for value in data["corner_values"]]
                mids = [
                    int(out.answer_gt.value) if value is None else int(value)
                    for value in data["middle_values"]
                ]
                for side_index in range(int(data["side_count"])):
                    total = (
                        corners[side_index]
                        + mids[side_index]
                        + corners[(side_index + 1) % int(data["side_count"])]
                    )
                    assert total == side_total


def _number_wall_levels(base: list[int], operator: str) -> list[list[int]]:
    levels = [list(base)]
    current = list(base)
    while len(current) > 1:
        if operator == "sum":
            current = [
                current[index] + current[index + 1] for index in range(len(current) - 1)
            ]
        elif operator == "difference":
            current = [
                abs(current[index] - current[index + 1])
                for index in range(len(current) - 1)
            ]
        else:
            current = [
                current[index] * current[index + 1] for index in range(len(current) - 1)
            ]
        levels.append(list(current))
    return levels


def _number_wall_candidate_count(
    *,
    levels: list[list[int]],
    target_level: int,
    target_index: int,
    answer_support: list[int],
    operator: str,
) -> int:
    if int(target_level) > 0:
        parent_value = _number_wall_levels(levels[int(target_level) - 1], operator)[1][
            int(target_index)
        ]
        return sum(1 for candidate in answer_support if int(candidate) == parent_value)

    count = 0
    for candidate in answer_support:
        base = list(levels[0])
        base[int(target_index)] = int(candidate)
        candidate_levels = _number_wall_levels(base, operator)
        matched = True
        for level_index, level in enumerate(levels):
            for brick_index, value in enumerate(level):
                if level_index == 0 and brick_index == int(target_index):
                    continue
                if candidate_levels[level_index][brick_index] != int(value):
                    matched = False
                    break
            if not matched:
                break
        if matched:
            count += 1
    return count


def test_number_wall_target_position_varies_and_is_unique() -> None:
    task = PuzzlesArithmeticNumberWallValueTask()
    seen_positions_by_branch: dict[str, set[tuple[int, int]]] = {}
    upper_target_count_by_branch: dict[str, int] = {}
    branches = ("single",)
    for branch_index, branch in enumerate(branches):
        for seed_offset in range(24):
            out = task.generate(
                2026062600 + (branch_index * 200) + seed_offset,
                params={"query_id": branch},
                max_attempts=160,
            )
            trace = out.trace_payload["execution_trace"]
            data = trace["constraint_data"]
            assert (
                out.trace_payload["query_spec"]["params"]["prompt_query_key"]
                == ADDITION_WALL_QUERY
            )
            target_level = int(data["target_level"])
            target_index = int(data["target_index"])
            seen_positions_by_branch.setdefault(branch, set()).add(
                (target_level, target_index)
            )
            if target_level > 0:
                upper_target_count_by_branch[branch] = (
                    upper_target_count_by_branch.get(branch, 0) + 1
                )
            levels = [[int(value) for value in row] for row in data["levels"]]
            assert levels[target_level][target_index] == int(out.answer_gt.value)
            assert (
                _number_wall_candidate_count(
                    levels=levels,
                    target_level=target_level,
                    target_index=target_index,
                    answer_support=[int(value) for value in trace["answer_support"]],
                    operator=str(data["operator"]),
                )
                == 1
            )
    for positions in seen_positions_by_branch.values():
        assert len(positions) > 1
        assert any(level > 0 for level, _index in positions)
    for branch in branches:
        assert upper_target_count_by_branch.get(branch, 0) >= 6


def test_arithmetic_panel_prompts_do_not_spell_out_hidden_rules() -> None:
    banned_fragments = (
        "shown in the note",
        "rule in the note",
        "window-total rule",
        "window-sum note",
        "each brick is the sum",
        "addition wall rule",
        "difference wall",
        "multiplication pyramid",
        "row and column totals",
        "visible numeric clues",
    )
    for _task_id, task_cls, queries in TASK_SPECS:
        task = task_cls()
        for index, branch in enumerate(queries):
            out = task.generate(
                2026062900 + (37 * index) + len(str(task.task_id)),
                params={"query_id": str(branch)},
                max_attempts=160,
            )
            prompt_lower = str(out.prompt).lower()
            for fragment in banned_fragments:
                assert fragment not in prompt_lower


def test_arithmetic_panel_task_is_deterministic() -> None:
    task = PuzzlesArithmeticRowColumnTotalMissingValueTask()
    params = {"scene_variant": "constraint_card"}
    out_a = task.generate(2026052399, params=params, max_attempts=120)
    out_b = task.generate(2026052399, params=params, max_attempts=120)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()
