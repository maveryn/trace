"""Contract tests for symbolic radial code-wheel tasks."""

from __future__ import annotations

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.symbolic.radial_code_wheel.code_output_label import (
    INTERNAL_QUERY_KEY as CODE_OUTPUT_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.radial_code_wheel.code_output_label import (
    TASK_ID as CODE_OUTPUT_TASK_ID,
)
from trace_tasks.tasks.symbolic.radial_code_wheel.code_output_label import SymbolicRadialCodeOutputLabelTask
from trace_tasks.tasks.symbolic.radial_code_wheel.missing_code_symbol_label import (
    INTERNAL_QUERY_KEY as MISSING_CODE_SYMBOL_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.radial_code_wheel.missing_code_symbol_label import (
    TASK_ID as MISSING_CODE_SYMBOL_TASK_ID,
)
from trace_tasks.tasks.symbolic.radial_code_wheel.missing_code_symbol_label import SymbolicRadialMissingCodeSymbolLabelTask
from trace_tasks.tasks.symbolic.radial_code_wheel.output_code_match_label import (
    INTERNAL_QUERY_KEY as OUTPUT_CODE_MATCH_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.radial_code_wheel.output_code_match_label import (
    TASK_ID as OUTPUT_CODE_MATCH_TASK_ID,
)
from trace_tasks.tasks.symbolic.radial_code_wheel.output_code_match_label import SymbolicRadialOutputCodeMatchLabelTask
from trace_tasks.tasks.symbolic.radial_code_wheel.shared.rules import all_codes, code_to_index, terminal_label_pool


def _assert_point_in_image(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    assert 0 <= float(point[0]) <= width
    assert 0 <= float(point[1]) <= height


def _ordered_terminal_labels() -> tuple[str, ...]:
    return terminal_label_pool()


def test_radial_code_wheel_tasks_are_registered_and_taxonomized() -> None:
    assert TASK_REGISTRY[CODE_OUTPUT_TASK_ID] is SymbolicRadialCodeOutputLabelTask
    assert TASK_REGISTRY[MISSING_CODE_SYMBOL_TASK_ID] is SymbolicRadialMissingCodeSymbolLabelTask
    assert TASK_REGISTRY[OUTPUT_CODE_MATCH_TASK_ID] is SymbolicRadialOutputCodeMatchLabelTask
    for task_id in (CODE_OUTPUT_TASK_ID, MISSING_CODE_SYMBOL_TASK_ID, OUTPUT_CODE_MATCH_TASK_ID):
        task = TASK_REGISTRY[task_id]()
        taxonomy = resolve_task_taxonomy(task_id)
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")
        assert tuple(task.supported_query_ids) == (SINGLE_QUERY_ID,)
        assert taxonomy.domain == "symbolic"
        assert taxonomy.scene_id == "radial_code_wheel"
        assert taxonomy.source_scene_id == ""


def test_code_output_label_contract() -> None:
    out = SymbolicRadialCodeOutputLabelTask().generate(
        2026062501,
        params={
            "target_code": "BCD",
            "correct_label": "E",
            "scene_variant": "clean_wheel",
            "terminal_labels": _ordered_terminal_labels(),
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["radial_code_wheel_metadata"]
    expected_output = _ordered_terminal_labels()[code_to_index("BCD")]

    assert out.scene_id == "radial_code_wheel"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == CODE_OUTPUT_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == CODE_OUTPUT_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "E"
    assert out.annotation_gt.type == "point_map"
    assert sorted(out.annotation_gt.value.keys()) == ["inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol"]
    assert metadata["task_mode"] == "code_to_output"
    assert metadata["target_code"] == "BCD"
    assert metadata["target_output_label"] == expected_output
    assert metadata["option_values"]["E"] == expected_output
    assert len(metadata["terminal_labels_by_code"]) == 64
    assert len(set(metadata["terminal_labels_by_code"].values())) == 64
    assert trace["execution_trace"]["annotation_item_ids"] == ["inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol"]
    assert trace["render_map"]["annotation_source"] == "item_points_px"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for key, point in out.annotation_gt.value.items():
        assert point == trace["render_map"]["item_points_px"][str(key)]
        _assert_point_in_image(point, width=width, height=height)


def test_output_code_match_label_contract() -> None:
    out = SymbolicRadialOutputCodeMatchLabelTask().generate(
        2026062502,
        params={
            "target_code": "CAD",
            "correct_label": "B",
            "scene_variant": "exam_scan",
            "terminal_labels": _ordered_terminal_labels(),
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["radial_code_wheel_metadata"]
    expected_output = _ordered_terminal_labels()[code_to_index("CAD")]

    assert out.scene_id == "radial_code_wheel"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == OUTPUT_CODE_MATCH_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == OUTPUT_CODE_MATCH_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "B"
    assert out.annotation_gt.type == "point_map"
    assert sorted(out.annotation_gt.value.keys()) == ["inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol"]
    assert metadata["task_mode"] == "output_to_code"
    assert metadata["target_code"] == "CAD"
    assert metadata["target_output_label"] == expected_output
    assert metadata["option_values"]["B"] == "CAD"
    assert len(set(metadata["option_values"].values())) == 6
    assert trace["execution_trace"]["annotation_item_ids"] == ["inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol"]
    assert trace["render_map"]["annotation_source"] == "item_points_px"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for key, point in out.annotation_gt.value.items():
        assert point == trace["render_map"]["item_points_px"][str(key)]
        _assert_point_in_image(point, width=width, height=height)


def test_missing_code_symbol_label_contract() -> None:
    out = SymbolicRadialMissingCodeSymbolLabelTask().generate(
        2026062503,
        params={
            "target_code": "BCD",
            "missing_position": "middle",
            "scene_variant": "clean_wheel",
            "terminal_labels": _ordered_terminal_labels(),
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["radial_code_wheel_metadata"]
    expected_output = _ordered_terminal_labels()[code_to_index("BCD")]

    assert out.scene_id == "radial_code_wheel"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == MISSING_CODE_SYMBOL_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == MISSING_CODE_SYMBOL_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "point"
    assert metadata["task_mode"] == "missing_code_symbol"
    assert metadata["target_code"] == "BCD"
    assert metadata["partial_code"] == "B ? D"
    assert metadata["missing_position_index"] == 1
    assert metadata["missing_ring_role"] == "middle_ring_symbol"
    assert metadata["answer_symbol"] == "C"
    assert metadata["target_output_label"] == expected_output
    assert metadata["candidate_symbols"] == ["A", "B", "C", "D"]
    assert trace["execution_trace"]["annotation_item_ids"] == ["middle_ring_symbol"]
    assert trace["render_map"]["annotation_source"] == "item_points_px"
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["render_map"]["missing_symbol_point_px"] == out.annotation_gt.value
    assert "symbol_option_bboxes_px" not in trace["render_map"]
    assert not any(key.startswith("symbol_option_") for key in trace["render_map"]["item_bboxes_px"])

    width, height = out.image.size
    _assert_point_in_image(out.annotation_gt.value, width=width, height=height)


def test_radial_generation_is_deterministic() -> None:
    params = {"scene_variant": "notebook_wheel", "target_code": "DAB", "correct_label": "F"}
    out_a = SymbolicRadialCodeOutputLabelTask().generate(2026062599, params=params, max_attempts=12)
    out_b = SymbolicRadialCodeOutputLabelTask().generate(2026062599, params=params, max_attempts=12)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_radial_rule_support_has_sixty_four_codes() -> None:
    codes = all_codes()
    assert len(codes) == 64
    assert len(set(codes)) == 64
    assert codes[0] == "AAA"
    assert codes[-1] == "DDD"


def test_symbolic_radial_prompt_bundle_supports_tasks() -> None:
    bundle = load_prompt_bundle("symbolic", "radial_code_wheel", "symbolic_radial_code_wheel_v1")
    assert "radial_code_wheel" in bundle.scene_templates
    assert len(bundle.task_templates["code_output_label"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["missing_code_symbol_label"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["output_code_match_label"]) == REQUIRED_PROMPT_VARIANTS
    assert not bundle.query_templates
    assert list(bundle.required_slots_by_key["scene:radial_code_wheel"]) == ["object_description"]
    assert list(bundle.required_slots_by_key["task:code_output_label"]) == []
    assert list(bundle.required_slots_by_key["task:missing_code_symbol_label"]) == []
    assert list(bundle.required_slots_by_key["task:output_code_match_label"]) == []
