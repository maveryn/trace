"""Contract tests for symbolic truth-table tasks."""

from __future__ import annotations

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.symbolic.truth_table.expression_from_rows_label import (
    INTERNAL_QUERY_KEY as EXPRESSION_FROM_ROWS_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.truth_table.expression_from_rows_label import (
    TASK_ID as EXPRESSION_FROM_ROWS_TASK_ID,
)
from trace_tasks.tasks.symbolic.truth_table.expression_from_rows_label import (
    SymbolicTruthTableExpressionFromRowsLabelTask,
)
from trace_tasks.tasks.symbolic.truth_table.satisfying_row_count import (
    INTERNAL_QUERY_KEY as COUNT_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.truth_table.satisfying_row_count import (
    TASK_ID as COUNT_TASK_ID,
)
from trace_tasks.tasks.symbolic.truth_table.satisfying_row_count import (
    SymbolicTruthTableSatisfyingRowCountTask,
)
from trace_tasks.tasks.symbolic.truth_table.shared.rules import expression_by_id
from trace_tasks.tasks.symbolic.truth_table.truth_pattern_label import (
    INTERNAL_QUERY_KEY as PATTERN_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.truth_table.truth_pattern_label import (
    TASK_ID as PATTERN_TASK_ID,
)
from trace_tasks.tasks.symbolic.truth_table.truth_pattern_label import (
    SymbolicTruthTableTruthPatternLabelTask,
)


def _assert_bbox_in_image(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def _cell_values_for_role(trace: dict, role: str) -> list[str]:
    return [
        str(entity.get("value", ""))
        for entity in trace["scene_ir"]["entities"]
        if entity.get("entity_type") == "truth_table_cell"
        and entity.get("role") == role
    ]


def test_truth_table_tasks_are_registered_and_taxonomized() -> None:
    assert TASK_REGISTRY[COUNT_TASK_ID] is SymbolicTruthTableSatisfyingRowCountTask
    assert TASK_REGISTRY[PATTERN_TASK_ID] is SymbolicTruthTableTruthPatternLabelTask
    assert (
        TASK_REGISTRY[EXPRESSION_FROM_ROWS_TASK_ID]
        is SymbolicTruthTableExpressionFromRowsLabelTask
    )
    for task_id in (
        COUNT_TASK_ID,
        PATTERN_TASK_ID,
        EXPRESSION_FROM_ROWS_TASK_ID,
    ):
        task = TASK_REGISTRY[task_id]()
        taxonomy = resolve_task_taxonomy(task_id)
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")
        assert tuple(task.supported_query_ids) == (SINGLE_QUERY_ID,)
        assert taxonomy.domain == "symbolic"
        assert taxonomy.scene_id == "truth_table"
        assert taxonomy.source_scene_id == ""


def test_satisfying_row_count_contract() -> None:
    out = SymbolicTruthTableSatisfyingRowCountTask().generate(
        2026070101,
        params={"expression_id": "a_and_b", "scene_variant": "clean_table"},
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["truth_table_metadata"]
    expected = expression_by_id("a_and_b")

    assert out.scene_id == "truth_table"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == COUNT_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == COUNT_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == expected.true_count == 2
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 2
    assert _cell_values_for_role(trace, "output_P") == [""] * 8
    assert metadata["truth_pattern"] == expected.pattern_string
    assert metadata["true_row_labels"] == ["7", "8"]
    assert trace["execution_trace"]["annotation_item_ids"] == ["row_7", "row_8"]
    assert trace["render_map"]["annotation_source"] == "row_bboxes_px"
    assert out.annotation_gt.value == [
        trace["render_map"]["row_bboxes_px"]["row_7"],
        trace["render_map"]["row_bboxes_px"]["row_8"],
    ]
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value

    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        _assert_bbox_in_image(bbox, width=width, height=height)


def test_truth_pattern_label_contract() -> None:
    out = SymbolicTruthTableTruthPatternLabelTask().generate(
        2026070102,
        params={
            "expression_id": "a_xor_b",
            "correct_label": "C",
            "scene_variant": "notebook_table",
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["truth_table_metadata"]
    expected = expression_by_id("a_xor_b")

    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == PATTERN_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert metadata["truth_pattern"] == expected.pattern_string
    assert tuple(metadata["option_patterns"]) == ("A", "B", "C", "D", "E", "F")
    assert metadata["option_patterns"]["C"] == expected.pattern_string
    assert tuple(trace["query_spec"]["params"]["target_answer_support"]) == (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    )
    assert trace["execution_trace"]["annotation_item_ids"] == ["pattern_option_C"]
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value

    width, height = out.image.size
    _assert_bbox_in_image(out.annotation_gt.value, width=width, height=height)


def test_expression_from_rows_label_contract() -> None:
    out = SymbolicTruthTableExpressionFromRowsLabelTask().generate(
        2026070104,
        params={
            "expression_id": "a_and_b",
            "correct_label": "Y",
            "scene_variant": "clean_table",
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["truth_table_metadata"]
    expected = expression_by_id("a_and_b")

    assert out.query_id == SINGLE_QUERY_ID
    assert (
        trace["query_spec"]["internal_query_id"]
        == EXPRESSION_FROM_ROWS_INTERNAL_QUERY_KEY
    )
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "Y"
    assert out.annotation_gt.type == "bbox"
    assert metadata["expression_id"] == "a_and_b"
    assert metadata["truth_pattern"] == expected.pattern_string
    assert metadata["correct_label"] == "Y"
    assert tuple(metadata["candidate_expressions"]) == ("W", "X", "Y", "Z")
    assert (
        metadata["candidate_expressions"]["Y"]["truth_pattern"]
        == expected.pattern_string
    )
    assert metadata["candidate_expressions"]["Y"]["matches_output_column"] is True
    assert [
        label
        for label, data in metadata["candidate_expressions"].items()
        if data["matches_output_column"]
    ] == ["Y"]
    assert _cell_values_for_role(trace, "output_P") == [
        str(value) for value in expected.pattern
    ]
    assert tuple(trace["query_spec"]["params"]["target_answer_support"]) == (
        "W",
        "X",
        "Y",
        "Z",
    )
    assert trace["execution_trace"]["annotation_item_ids"] == ["expression_option_Y"]
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert (
        out.annotation_gt.value
        == trace["render_map"]["item_bboxes_px"]["expression_option_Y"]
    )
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value

    width, height = out.image.size
    _assert_bbox_in_image(out.annotation_gt.value, width=width, height=height)


def test_truth_table_generation_is_deterministic() -> None:
    params = {"expression_id": "a_or_b", "scene_variant": "clean_table"}
    out_a = SymbolicTruthTableSatisfyingRowCountTask().generate(
        2026070199, params=params, max_attempts=12
    )
    out_b = SymbolicTruthTableSatisfyingRowCountTask().generate(
        2026070199, params=params, max_attempts=12
    )

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_truth_table_prompt_bundle_supports_tasks() -> None:
    bundle = load_prompt_bundle("symbolic", "truth_table", "symbolic_truth_table_v1")
    assert "truth_table" in bundle.scene_templates
    assert (
        len(bundle.task_templates["satisfying_row_count"]) == REQUIRED_PROMPT_VARIANTS
    )
    assert len(bundle.task_templates["truth_pattern_label"]) == REQUIRED_PROMPT_VARIANTS
    assert (
        len(bundle.task_templates["expression_from_rows_label"])
        == REQUIRED_PROMPT_VARIANTS
    )
    assert not bundle.query_templates
    assert list(bundle.required_slots_by_key["scene:truth_table"]) == [
        "object_description"
    ]
    assert list(bundle.required_slots_by_key["task:satisfying_row_count"]) == []
    assert list(bundle.required_slots_by_key["task:truth_pattern_label"]) == []
    assert list(bundle.required_slots_by_key["task:expression_from_rows_label"]) == []


def test_satisfying_row_count_sampling_covers_non_binary_answers() -> None:
    answers = set()
    for index in range(80):
        out = SymbolicTruthTableSatisfyingRowCountTask().generate(
            2026070200 + index,
            params={},
            max_attempts=20,
        )
        answers.add(int(out.answer_gt.value))
    assert answers.issuperset({1, 2, 3, 4})
    assert all(1 <= answer <= 7 for answer in answers)
