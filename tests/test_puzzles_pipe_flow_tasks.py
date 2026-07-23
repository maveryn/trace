"""Contracts for pipe-flow puzzle tasks."""

from __future__ import annotations

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.puzzles.pipe_flow.pipe_flow_repair_tile_label import (
    SCENE_ID,
    PuzzlesPipeFlowRepairTileLabelTask,
    SUPPORTED_QUERY_IDS as REPAIR_SUPPORTED_QUERY_IDS,
    TASK_ID as REPAIR_TASK_ID,
)
from trace_tasks.tasks.puzzles.pipe_flow.misrotated_tile_label import (
    PuzzlesPipeFlowMisrotatedTileLabelTask,
    SUPPORTED_QUERY_IDS as MISROTATED_SUPPORTED_QUERY_IDS,
    TASK_ID as MISROTATED_TASK_ID,
)


def test_pipe_flow_task_is_registered() -> None:
    assert REPAIR_TASK_ID in TASK_REGISTRY
    taxonomy = resolve_task_taxonomy(REPAIR_TASK_ID)
    assert taxonomy.domain == "puzzles"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_domain == "puzzles"
    assert taxonomy.source_scene_id == ""
    assert REPAIR_SUPPORTED_QUERY_IDS == (SINGLE_QUERY_ID,)

    assert MISROTATED_TASK_ID in TASK_REGISTRY
    taxonomy = resolve_task_taxonomy(MISROTATED_TASK_ID)
    assert taxonomy.domain == "puzzles"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_domain == "puzzles"
    assert taxonomy.source_scene_id == ""
    assert MISROTATED_SUPPORTED_QUERY_IDS == (SINGLE_QUERY_ID,)


def test_pipe_flow_repair_tile_contract() -> None:
    task = PuzzlesPipeFlowRepairTileLabelTask()
    out = task.generate(
        int(hash64(20260521, REPAIR_TASK_ID, 0)),
        params={
            "query_id": SINGLE_QUERY_ID,
            "grid_size_variant": "6x6",
            "gap_size_variant": "2x2",
            "scene_variant": "water_pipe",
            "answer_label": "C",
        },
        max_attempts=80,
    )

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == {"selected_option", "missing_gap"}
    assert "annotation" in out.prompt_variants["answer_and_annotation"].lower()
    assert "selected_option" in out.prompt_variants["answer_and_annotation"]
    assert "missing_gap" in out.prompt_variants["answer_and_annotation"]

    trace = out.trace_payload["execution_trace"]
    assert trace["question_format"] == "pipe_flow_repair_tile_label"
    assert trace["query_id"] == SINGLE_QUERY_ID
    assert trace["internal_question_format"] == "pipe_flow_repair_tile_label"
    assert trace["answer_label"] == "C"
    assert trace["grid_size_variant"] == "6x6"
    assert trace["grid_size_variant_probabilities"] == {
        "5x5": 0.0,
        "6x6": 1.0,
        "7x7": 0.0,
    }
    assert trace["gap_size_variant"] == "2x2"
    assert trace["gap_size"] == 2
    assert trace["gap_size_variant_probabilities"] == {"2x2": 1.0}
    assert trace["query_id_probabilities"] == {SINGLE_QUERY_ID: 1.0}
    assert trace["answer_label_probabilities"] == {
        label: (1.0 if label == "C" else 0.0)
        for label in "ABCD"
    }
    correct_options = [option for option in trace["option_specs"] if option["is_correct"]]
    assert len(correct_options) == 1
    assert correct_options[0]["label"] == "C"
    assert correct_options[0]["rotation_allowed"] is False
    assert correct_options[0]["connects_in_place"] is True
    assert all(
        not option["connects_in_place"]
        for option in trace["option_specs"]
        if not option["is_correct"]
    )
    assert trace["rotation_allowed"] is False
    assert trace["branching_allowed"] is False
    assert trace["placement_rule"] == "place_option_as_drawn_no_rotation"
    assert trace["candidate_count"] == 4
    assert len(trace["missing_cells"]) == 4
    assert trace["missing_cell_count"] == 4
    assert all(len(option["local_openings"]) == 4 for option in trace["option_specs"])
    assert trace["branch_cells"] == []
    assert trace["branch_terminal_cells"] == []
    assert all(
        len(cell["openings"]) != 1
        for option in trace["option_specs"]
        for cell in option["local_openings"]
    )

    correct_bbox = out.trace_payload["render_map"]["item_bboxes_px"][
        trace["correct_option_panel_id"]
    ]
    missing_bbox = out.trace_payload["render_map"]["item_bboxes_px"][
        trace["missing_region_id"]
    ]
    assert out.annotation_gt.value == {
        "selected_option": [float(value) for value in correct_bbox],
        "missing_gap": [float(value) for value in missing_bbox],
    }
    assert out.trace_payload["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert out.trace_payload["render_spec"]["cell_size_max_px"] == 34


def test_pipe_flow_repair_tile_rejects_single_tile_gap() -> None:
    task = PuzzlesPipeFlowRepairTileLabelTask()
    with pytest.raises(ValueError, match="unsupported gap_size_variant"):
        task.generate(
            int(hash64(20260521, REPAIR_TASK_ID, 2)),
            params={
                "gap_size_variant": "1x1",
                "scene_variant": "industrial_conduit",
                "answer_label": "B",
            },
            max_attempts=2,
        )


def test_pipe_flow_repair_tile_is_deterministic() -> None:
    task = PuzzlesPipeFlowRepairTileLabelTask()
    seed = int(hash64(20260521, REPAIR_TASK_ID, 1))
    params = {
        "grid_size_variant": "6x6",
        "gap_size_variant": "2x2",
        "scene_variant": "circuit_trace",
        "answer_label": "D",
    }
    left = task.generate(seed, params=params, max_attempts=80)
    right = task.generate(seed, params=params, max_attempts=80)

    assert left.prompt == right.prompt
    assert left.answer_gt == right.answer_gt
    assert left.annotation_gt == right.annotation_gt
    assert left.trace_payload["execution_trace"] == right.trace_payload["execution_trace"]
    assert left.image.tobytes() == right.image.tobytes()


def test_pipe_flow_misrotated_tile_contract() -> None:
    task = PuzzlesPipeFlowMisrotatedTileLabelTask()
    out = task.generate(
        int(hash64(20260521, MISROTATED_TASK_ID, 0)),
        params={
            "query_id": SINGLE_QUERY_ID,
            "grid_size_variant": "5x5",
            "scene_variant": "water_pipe",
            "answer_label": "B",
        },
        max_attempts=120,
    )

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "B"
    assert out.annotation_gt.type == "bbox"
    assert isinstance(out.annotation_gt.value, list)
    assert len(out.annotation_gt.value) == 4
    assert "selected_tile" not in out.prompt_variants["answer_and_annotation"]

    trace = out.trace_payload["execution_trace"]
    assert trace["question_format"] == "pipe_flow_misrotated_tile_label"
    assert trace["query_id"] == SINGLE_QUERY_ID
    assert trace["internal_question_format"] == "pipe_flow_misrotated_tile_label"
    assert trace["answer_label"] == "B"
    assert trace["grid_size_variant"] == "5x5"
    assert trace["grid_size_variant_probabilities"] == {
        "5x5": 1.0,
        "6x6": 0.0,
        "7x7": 0.0,
    }
    assert trace["query_id_probabilities"] == {SINGLE_QUERY_ID: 1.0}
    assert trace["answer_label_probabilities"] == {
        label: (1.0 if label == "B" else 0.0)
        for label in "ABCD"
    }
    assert trace["candidate_count"] == 4
    assert trace["branch_cells"] == []
    assert trace["branch_terminal_cells"] == []
    assert trace["branching_allowed"] is False
    assert trace["rotation_rule"] == "rotate_exactly_one_labeled_tile"
    assert "option_specs" not in trace

    candidates = trace["candidate_specs"]
    assert [candidate["label"] for candidate in candidates] == list("ABCD")
    correct_candidates = [candidate for candidate in candidates if candidate["is_correct"]]
    assert len(correct_candidates) == 1
    assert correct_candidates[0]["label"] == "B"
    assert correct_candidates[0]["tile_id"] == trace["misrotated_tile_id"]
    assert correct_candidates[0]["connects_after_rotation"] is True
    assert correct_candidates[0]["repair_rotation_turns"]
    assert all(
        candidate["current_openings"] != candidate["required_openings"]
        for candidate in candidates
        if candidate["is_correct"]
    )
    assert all(
        not candidate["connects_after_rotation"]
        for candidate in candidates
        if not candidate["is_correct"]
    )

    selected_bbox = out.trace_payload["render_map"]["item_bboxes_px"][
        trace["correct_candidate_id"]
    ]
    assert out.annotation_gt.value == [float(value) for value in selected_bbox]
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert out.trace_payload["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert out.trace_payload["witness_symbolic"]["type"] == "bbox"
    assert out.trace_payload["witness_symbolic"]["value"] == out.annotation_gt.value
    assert out.trace_payload["render_spec"]["canvas_width"] == 760
    assert out.trace_payload["render_spec"]["canvas_height"] == 720
    assert out.trace_payload["render_spec"]["cell_gap_px"] == 5
    assert out.trace_payload["render_spec"]["cell_size_min_px"] == 60
    assert out.trace_payload["render_spec"]["cell_size_max_px"] == 68
    assert out.trace_payload["render_spec"]["pipe_width_px"] == 20
    assert out.trace_payload["render_spec"]["tile_label_font_size_px"] == 22
    assert selected_bbox[2] - selected_bbox[0] >= 60.0
    label_bbox = out.trace_payload["render_map"]["tile_bboxes_px"][
        f"{trace['correct_candidate_id']}_label"
    ]
    assert label_bbox[2] - label_bbox[0] >= 30.0


def test_pipe_flow_misrotated_tile_is_deterministic() -> None:
    task = PuzzlesPipeFlowMisrotatedTileLabelTask()
    seed = int(hash64(20260521, MISROTATED_TASK_ID, 1))
    params = {
        "grid_size_variant": "7x7",
        "scene_variant": "industrial_conduit",
        "answer_label": "D",
    }
    left = task.generate(seed, params=params, max_attempts=120)
    right = task.generate(seed, params=params, max_attempts=120)

    assert left.prompt == right.prompt
    assert left.answer_gt == right.answer_gt
    assert left.annotation_gt == right.annotation_gt
    assert left.trace_payload["execution_trace"] == right.trace_payload["execution_trace"]
    assert left.image.tobytes() == right.image.tobytes()
