from __future__ import annotations

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.autoload import module_name_for_public_task_id
from trace_tasks.tasks.puzzles.color_gradient.color_gradient_completion_label import (
    PuzzlesColorGradientCompletionLabelTask,
)
from trace_tasks.tasks.puzzles.color_gradient.color_gradient_violation_cell_label import (
    PuzzlesColorGradientViolationCellLabelTask,
)
from trace_tasks.tasks.puzzles.color_gradient.shared.state import DOMAIN, SCENE_ID

VIOLATION_TASK_ID = "task_puzzles__color_gradient__color_gradient_violation_cell_label"
COMPLETION_TASK_ID = "task_puzzles__color_gradient__color_gradient_completion_label"


@pytest.mark.parametrize(
    "task_id,task_cls",
    [
        (VIOLATION_TASK_ID, PuzzlesColorGradientViolationCellLabelTask),
        (COMPLETION_TASK_ID, PuzzlesColorGradientCompletionLabelTask),
    ],
)
def test_color_gradient_tasks_are_registered(task_id, task_cls) -> None:
    assert task_id in TASK_REGISTRY
    assert isinstance(create_task(task_id), task_cls)
    taxonomy = resolve_task_taxonomy(task_id)
    assert taxonomy.domain == DOMAIN
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert module_name_for_public_task_id(task_id).startswith(
        "trace_tasks.tasks.puzzles.color_gradient."
    )


def test_color_gradient_violation_contract() -> None:
    task = PuzzlesColorGradientViolationCellLabelTask()
    out = task.generate(
        int(hash64(20260521, VIOLATION_TASK_ID, 0)),
        params={
            "grid_size_variant": "4x4",
            "answer_label": "K",
            "rule_variant": "column_hue_row_lightness",
        },
        max_attempts=20,
    )

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "K"
    assert out.annotation_gt.type == "bbox"

    trace = out.trace_payload["execution_trace"]
    assert trace["violation_cell_id"] == "cell_K"
    assert trace["answer_label"] == "K"
    assert trace["question_format"] == "color_gradient_violation_cell_label"
    assert len([cell for cell in trace["cells"] if cell["is_violation"]]) == 1
    assert (
        trace["cells"][trace["violation_index"]]["expected_rgb"]
        != trace["cells"][trace["violation_index"]]["observed_rgb"]
    )

    bbox = out.trace_payload["render_map"]["item_bboxes_px"]["cell_K"]
    assert out.annotation_gt.value == [float(value) for value in bbox]
    assert out.trace_payload["render_spec"]["label_style"]["font"]["source"] == (
        "global_font_pool"
    )
    assert out.trace_payload["render_spec"]["label_style"]["font"]["font_family"]
    assert out.trace_payload["render_spec"]["label_style"]["chip_fill_rgb"] == [
        255,
        255,
        255,
    ]
    assert out.trace_payload["render_spec"]["label_style"]["chip_outline_rgb"] == [
        36,
        42,
        52,
    ]
    assert out.trace_payload["render_spec"]["label_style"]["label_fill_rgb"] == [
        28,
        32,
        38,
    ]
    assert out.trace_payload["render_spec"]["post_image_noise_policy"]["reason"] == (
        "color_semantics_preserve_rgb_separability"
    )


def test_color_gradient_violation_is_deterministic() -> None:
    task = PuzzlesColorGradientViolationCellLabelTask()
    seed = int(hash64(20260521, VIOLATION_TASK_ID, 1))
    params = {
        "grid_size_variant": "3x3",
        "answer_label": "E",
        "rule_variant": "row_hue_column_lightness",
        "scene_variant": "swatch_notebook",
    }
    left = task.generate(seed, params=params, max_attempts=20)
    right = task.generate(seed, params=params, max_attempts=20)

    assert left.prompt == right.prompt
    assert left.answer_gt == right.answer_gt
    assert left.annotation_gt == right.annotation_gt
    assert (
        left.trace_payload["execution_trace"] == right.trace_payload["execution_trace"]
    )
    assert left.image.tobytes() == right.image.tobytes()


def test_color_gradient_completion_contract() -> None:
    task = PuzzlesColorGradientCompletionLabelTask()
    out = task.generate(
        int(hash64(20260521, COMPLETION_TASK_ID, 0)),
        params={
            "sequence_length_variant": "6_cell",
            "option_count_variant": "5_options",
            "answer_label": "D",
            "missing_index": 3,
            "rule_variant": "hue_gradient",
        },
        max_attempts=20,
    )

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox"

    trace = out.trace_payload["execution_trace"]
    assert trace["missing_index"] == 3
    assert trace["answer_label"] == "D"
    assert trace["correct_option_id"] == "option_D"
    assert trace["question_format"] == "linear_gradient_completion_label"
    assert len([option for option in trace["options"] if option["is_correct"]]) == 1

    item_bboxes = out.trace_payload["render_map"]["item_bboxes_px"]
    assert out.annotation_gt.value == [
        float(value) for value in item_bboxes["option_D"]
    ]
    projected = out.trace_payload["projected_annotation"]
    assert projected["type"] == "bbox"
    assert projected["bbox"] == out.annotation_gt.value
    assert projected["pixel_bbox"] == out.annotation_gt.value
    assert trace["context_item_ids"] == ["sequence_cell_3"]
    assert out.trace_payload["render_spec"]["label_style"]["font"]["source"] == (
        "global_font_pool"
    )
    assert out.trace_payload["render_spec"]["label_style"]["font"]["font_family"]
    assert out.trace_payload["render_spec"]["label_style"]["chip_fill_rgb"] == [
        255,
        255,
        255,
    ]
    assert out.trace_payload["render_spec"]["label_style"]["chip_outline_rgb"] == [
        36,
        42,
        52,
    ]
    assert out.trace_payload["render_spec"]["label_style"]["label_fill_rgb"] == [
        28,
        32,
        38,
    ]


def test_color_gradient_completion_is_deterministic() -> None:
    task = PuzzlesColorGradientCompletionLabelTask()
    seed = int(hash64(20260521, COMPLETION_TASK_ID, 1))
    params = {
        "sequence_length_variant": "7_cell",
        "option_count_variant": "6_options",
        "answer_label": "F",
        "rule_variant": "hue_lightness_gradient",
        "scene_variant": "swatch_card",
    }
    left = task.generate(seed, params=params, max_attempts=20)
    right = task.generate(seed, params=params, max_attempts=20)

    assert left.prompt == right.prompt
    assert left.answer_gt == right.answer_gt
    assert left.annotation_gt == right.annotation_gt
    assert (
        left.trace_payload["execution_trace"] == right.trace_payload["execution_trace"]
    )
    assert left.image.tobytes() == right.image.tobytes()
