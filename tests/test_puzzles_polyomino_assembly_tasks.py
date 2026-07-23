"""Behavior tests for the polyomino-assembly puzzle scene."""

from __future__ import annotations

from trace_tasks.tasks.puzzles.polyomino_assembly.composition_result_label import (
    PuzzlesPolyominoAssemblyCompositionResultLabelTask,
)
from trace_tasks.tasks.puzzles.polyomino_assembly.decomposition_pair_label import (
    PuzzlesPolyominoAssemblyDecompositionPairLabelTask,
)
from trace_tasks.tasks.puzzles.polyomino_assembly.hole_fill_piece_label import (
    PuzzlesPolyominoAssemblyHoleFillPieceLabelTask,
)
from trace_tasks.tasks.puzzles.polyomino_assembly.shared.rules import (
    can_two_pieces_tile_target,
    reflection_signature,
)
from tests.helpers import extract_prompt_json_example


def _cells(raw_cells) -> tuple[tuple[int, int], ...]:
    """Return a hashable cell tuple from JSON-style cells."""

    return tuple(sorted((int(cell[0]), int(cell[1])) for cell in raw_cells))


def _option_labels(execution: dict) -> list[str]:
    """Return visible option labels from the execution trace."""

    return [str(option["option_label"]) for option in execution["option_specs"]]


def _assert_scalar_option_bbox(out) -> None:
    """Check that annotation is exactly the selected option-card bbox."""

    assert out.annotation_gt.type == "bbox"
    execution = out.trace_payload["execution_trace"]
    render_map = out.trace_payload["render_map"]
    expected = [
        float(value)
        for value in render_map["item_bboxes_px"][str(execution["correct_option_choice_id"])]
    ]
    observed = [float(value) for value in out.annotation_gt.value]
    assert observed == expected


def _assert_option_cell_scale_matches(out) -> None:
    """Verify all rendered polyomino shapes use one cell scale."""

    entities = out.trace_payload["scene_ir"]["entities"]
    shape_widths = []
    for entity in entities:
        if str(entity["entity_type"]) != "polyomino_shape":
            continue
        cells = list(entity["cells"])
        bbox = [float(value) for value in entity["bbox_px"]]
        max_x = max(int(cell[0]) for cell in cells)
        min_x = min(int(cell[0]) for cell in cells)
        width_in_cells = int(max_x - min_x + 1)
        shape_widths.append(round((bbox[2] - bbox[0]) / float(width_in_cells), 3))
    assert shape_widths
    assert max(shape_widths) - min(shape_widths) < 4.0


def test_decomposition_pair_contract_has_unique_rotatable_pair() -> None:
    task = PuzzlesPolyominoAssemblyDecompositionPairLabelTask()
    out = task.generate(39100, params={}, max_attempts=50)
    execution = out.trace_payload["execution_trace"]
    payload = execution["variant_payload"]
    target = _cells(payload["target_cells"])

    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == execution["answer_option_label"]
    assert execution["question_format"] == "decomposition_pair_label"
    assert _option_labels(execution) == ["A", "B", "C", "D"]
    assert len(execution["option_specs"]) == 4
    _assert_scalar_option_bbox(out)
    _assert_option_cell_scale_matches(out)

    matches = []
    for option in execution["option_specs"]:
        piece_a = _cells(option["pieces"][0]["cells"])
        piece_b = _cells(option["pieces"][1]["cells"])
        if can_two_pieces_tile_target(piece_a, piece_b, target):
            matches.append(str(option["option_label"]))
    assert matches == [str(out.answer_gt.value)]


def test_composition_result_contract_has_unique_rotatable_shape() -> None:
    task = PuzzlesPolyominoAssemblyCompositionResultLabelTask()
    out = task.generate(39200, params={}, max_attempts=50)
    execution = out.trace_payload["execution_trace"]
    payload = execution["variant_payload"]
    source_a = _cells(payload["source_pieces"][0]["cells"])
    source_b = _cells(payload["source_pieces"][1]["cells"])

    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == execution["answer_option_label"]
    assert execution["question_format"] == "composition_result_label"
    assert _option_labels(execution) == ["A", "B", "C", "D"]
    assert len(execution["option_specs"]) == 4
    _assert_scalar_option_bbox(out)
    _assert_option_cell_scale_matches(out)

    matches = []
    for option in execution["option_specs"]:
        shape = _cells(option["cells"])
        if can_two_pieces_tile_target(source_a, source_b, shape):
            matches.append(str(option["option_label"]))
    assert matches == [str(out.answer_gt.value)]


def test_hole_fill_piece_contract_has_unique_reflection_aware_piece() -> None:
    task = PuzzlesPolyominoAssemblyHoleFillPieceLabelTask()
    out = task.generate(39250, params={}, max_attempts=50)
    execution = out.trace_payload["execution_trace"]
    payload = execution["variant_payload"]
    hole_shape = _cells(payload["hole_shape_cells"])

    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == execution["answer_option_label"]
    assert execution["question_format"] == "hole_fill_piece_label"
    assert execution["transform_policy"] == "rotation_and_reflection_allowed"
    assert _option_labels(execution) == ["A", "B", "C", "D"]
    assert len(execution["option_specs"]) == 4
    _assert_scalar_option_bbox(out)
    _assert_option_cell_scale_matches(out)

    correct_signature = reflection_signature(hole_shape)
    matches = []
    for option in execution["option_specs"]:
        shape = _cells(option["cells"])
        if reflection_signature(shape) == correct_signature:
            matches.append(str(option["option_label"]))
    assert matches == [str(out.answer_gt.value)]


def test_polyomino_assembly_prompt_examples_match_scalar_bbox_contract() -> None:
    for task in (
        PuzzlesPolyominoAssemblyCompositionResultLabelTask(),
        PuzzlesPolyominoAssemblyDecompositionPairLabelTask(),
        PuzzlesPolyominoAssemblyHoleFillPieceLabelTask(),
    ):
        out = task.generate(39300, params={}, max_attempts=50)
        answer_and_annotation = extract_prompt_json_example(
            out.prompt_variants["answer_and_annotation"]
        )
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])

        assert answer_and_annotation == {
            "annotation": [438, 332, 762, 522],
            "answer": "B",
        }
        assert answer_only == {"answer": "B"}
        if out.trace_payload["execution_trace"]["question_format"] == "hole_fill_piece_label":
            assert "flip" in out.prompt.lower()
        else:
            assert (
                "Do not flip" in out.prompt
                or "not flipping" in out.prompt
                or "rotation only" in out.prompt
            )


def test_polyomino_assembly_sampling_covers_letters_for_both_tasks() -> None:
    tasks = (
        PuzzlesPolyominoAssemblyDecompositionPairLabelTask(),
        PuzzlesPolyominoAssemblyCompositionResultLabelTask(),
        PuzzlesPolyominoAssemblyHoleFillPieceLabelTask(),
    )
    for task in tasks:
        observed = set()
        for seed_offset in range(160):
            out = task.generate(39400 + seed_offset, params={}, max_attempts=50)
            observed.add(str(out.answer_gt.value))
        assert observed >= {"A", "B", "C", "D"}


def test_polyomino_assembly_task_is_deterministic() -> None:
    task = PuzzlesPolyominoAssemblyDecompositionPairLabelTask()
    out_a = task.generate(39500, params={}, max_attempts=50)
    out_b = task.generate(39500, params={}, max_attempts=50)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload[
        "query_spec"
    ]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
