"""Contract tests for migrated counterfactual-board game tasks."""

from __future__ import annotations

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.core.taxonomy import lookup_task_taxonomy, resolve_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.games.counterfactual_board.board_dimension_count import (
    COLUMN_COUNT_QUERY,
    GamesCounterfactualBoardDimensionCountTask,
    ROW_COUNT_QUERY,
    TASK_ID as DIMENSION_TASK_ID,
)
from trace_tasks.tasks.games.counterfactual_board.board_line_count import (
    HORIZONTAL_LINE_COUNT_QUERY,
    GamesCounterfactualBoardLineCountTask,
    TASK_ID as LINE_TASK_ID,
    VERTICAL_LINE_COUNT_QUERY,
)
from trace_tasks.tasks.games.counterfactual_board.shared.state import SCENE_ID


def _assert_bbox_in_image(bbox: list[float], image_size: tuple[int, int]) -> None:
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= float(image_size[0])
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= float(image_size[1])
    assert min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) >= 24.0


def _assert_segment_in_image(segment: list[list[float]], image_size: tuple[int, int]) -> None:
    assert len(segment) == 2
    for point in segment:
        assert len(point) == 2
        assert 0 <= float(point[0]) <= float(image_size[0])
        assert 0 <= float(point[1]) <= float(image_size[1])


def test_counterfactual_board_tasks_are_registered_with_public_scene() -> None:
    for task_id in (DIMENSION_TASK_ID, LINE_TASK_ID):
        assert task_id in TASK_REGISTRY
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == SCENE_ID
        assert taxonomy.source_scene_id == ""
        static_taxonomy = lookup_task_taxonomy(task_id)
        assert static_taxonomy is not None
        assert static_taxonomy.source_scene_id == SCENE_ID


@pytest.mark.parametrize(
    ("task", "task_id", "params", "expected_answer", "annotation_type"),
    [
        (
            GamesCounterfactualBoardDimensionCountTask(),
            DIMENSION_TASK_ID,
            {
                "query_id": ROW_COUNT_QUERY,
                "board_style": "chess_checkers",
                "visible_rows": 6,
                "visible_columns": 8,
            },
            6,
            "bbox_set",
        ),
        (
            GamesCounterfactualBoardDimensionCountTask(),
            DIMENSION_TASK_ID,
            {
                "query_id": COLUMN_COUNT_QUERY,
                "board_style": "sudoku",
                "visible_rows": 9,
                "visible_columns": 10,
            },
            10,
            "bbox_set",
        ),
        (
            GamesCounterfactualBoardLineCountTask(),
            LINE_TASK_ID,
            {
                "query_id": HORIZONTAL_LINE_COUNT_QUERY,
                "board_style": "xiangqi",
                "visible_rows": 11,
                "visible_columns": 9,
            },
            11,
            "segment_set",
        ),
        (
            GamesCounterfactualBoardLineCountTask(),
            LINE_TASK_ID,
            {
                "query_id": VERTICAL_LINE_COUNT_QUERY,
                "board_style": "xiangqi",
                "visible_rows": 10,
                "visible_columns": 8,
            },
            8,
            "segment_set",
        ),
    ],
)
def test_counterfactual_board_task_query_contracts(
    task,
    task_id: str,
    params: dict[str, object],
    expected_answer: int,
    annotation_type: str,
) -> None:
    out = task.generate(
        int(hash64(20260529, f"{task_id}:{params['query_id']}")),
        params=params,
        max_attempts=20,
    )

    assert out.scene_id == SCENE_ID
    assert out.query_id == params["query_id"]
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == int(expected_answer)
    assert out.annotation_gt.type == annotation_type
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert "counterfactual" not in out.prompt.lower()

    trace = out.trace_payload
    execution = trace["execution_trace"]
    if annotation_type == "bbox_set":
        expected_annotation = [
            [float(value) for value in bbox]
            for bbox in execution["counted_element_bboxes_px"]
        ]
        assert trace["render_map"]["annotation_source"] == "counted_element_bboxes_px"
    else:
        expected_annotation = execution["counted_element_segments_px"]
        assert trace["render_map"]["annotation_source"] == "counted_element_segments_px"
    assert out.annotation_gt.value == expected_annotation
    assert execution["supporting_item_ids"] == execution["counted_element_ids"]
    assert trace["projected_annotation"]["type"] == annotation_type
    assert trace["projected_annotation"][annotation_type] == out.annotation_gt.value
    assert trace["witness_symbolic"]["value"] == out.annotation_gt.value
    assert trace["query_spec"]["template_id"] == "games_counterfactual_board_v1"

    if annotation_type == "bbox_set":
        for bbox in out.annotation_gt.value:
            _assert_bbox_in_image(bbox, out.image.size)
    else:
        for segment in out.annotation_gt.value:
            _assert_segment_in_image(segment, out.image.size)

    noise = trace["render_spec"]["post_image_noise"]
    assert noise["enabled"] is True
    assert float(noise["apply_prob"]) == pytest.approx(0.5)
    assert set(noise["edit_types"]) == {"blur", "downsample", "jpeg", "noise"}
    assert noise["edit_count_range"] == [1, 1]


def test_counterfactual_board_dimension_task_is_deterministic() -> None:
    task = GamesCounterfactualBoardDimensionCountTask()
    seed = int(hash64(20260529, f"{DIMENSION_TASK_ID}:deterministic"))
    params = {
        "board_style": "sudoku",
        "query_id": ROW_COUNT_QUERY,
        "visible_rows": 10,
        "visible_columns": 9,
    }

    left = task.generate(seed, params=params, max_attempts=20)
    right = task.generate(seed, params=params, max_attempts=20)

    assert left.prompt == right.prompt
    assert left.answer_gt == right.answer_gt
    assert left.annotation_gt == right.annotation_gt
    assert left.trace_payload["execution_trace"] == right.trace_payload["execution_trace"]
    assert left.image.tobytes() == right.image.tobytes()
