"""Contract tests for symbolic Boolean logic-gate notation tasks."""

from __future__ import annotations

from typing import Any, Sequence

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.symbolic.logic_gate_circuit.gate_type_count import (
    INTERNAL_QUERY_ID as GATE_TYPE_COUNT_INTERNAL_QUERY_ID,
    TASK_ID as GATE_TYPE_COUNT_TASK_ID,
    SymbolicLogicGateGateTypeCountTask,
)
from trace_tasks.tasks.symbolic.logic_gate_circuit.internal_output_count import (
    INTERNAL_OUTPUT_ONE_QUERY_ID,
    INTERNAL_OUTPUT_ZERO_QUERY_ID,
    TASK_ID as INTERNAL_OUTPUT_COUNT_TASK_ID,
    SymbolicLogicGateInternalOutputCountTask,
)
from trace_tasks.tasks.symbolic.logic_gate_circuit.output_value_label import (
    OUTPUT_ONE_LABEL_QUERY_ID,
    OUTPUT_VALUE_LABEL_TASK_ID,
    OUTPUT_ZERO_LABEL_QUERY_ID,
    SUPPORTED_QUERY_IDS as LABEL_QUERY_IDS,
    SymbolicLogicGateOutputValueLabelTask,
)
from trace_tasks.tasks.symbolic.logic_gate_circuit.satisfying_assignment_label import (
    ASSIGNMENT_OUTPUTS_ONE_QUERY_ID,
    ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID,
    TASK_ID as ASSIGNMENT_TASK_ID,
    SymbolicLogicGateSatisfyingAssignmentLabelTask,
)
from trace_tasks.tasks.symbolic.logic_gate_circuit.shared.rules import evaluate_logic_gate
from trace_tasks.tasks.symbolic.logic_gate_circuit.shared.state import SCENE_ID


def _assert_fanout_free(circuit: dict) -> None:
    input_use_counts: dict[str, int] = {}
    for gate in circuit["gates"]:
        for signal_id in gate["input_signal_ids"]:
            input_use_counts[str(signal_id)] = int(input_use_counts.get(str(signal_id), 0)) + 1
    assert input_use_counts
    assert all(count == 1 for count in input_use_counts.values())


def _assert_gate_arities(circuit: dict) -> None:
    for gate in circuit["gates"]:
        gate_type = str(gate["gate_type"]).upper()
        expected = 1 if gate_type == "NOT" else 2
        assert len(gate["input_signal_ids"]) == expected


def _assert_planar_expression_tree(circuit: dict) -> None:
    spans = {
        str(input_spec["item_id"]): (index, index)
        for index, input_spec in enumerate(circuit["inputs"])
    }
    for gate in circuit["gates"]:
        gate_type = str(gate["gate_type"]).upper()
        input_spans = [spans[str(signal_id)] for signal_id in gate["input_signal_ids"]]
        if gate_type == "NOT":
            assert len(input_spans) == 1
            spans[str(gate["output_signal_id"])] = input_spans[0]
            continue
        assert len(input_spans) == 2
        left_span, right_span = sorted(input_spans)
        assert left_span[1] + 1 == right_span[0]
        spans[str(gate["output_signal_id"])] = (left_span[0], right_span[1])
    assert str(circuit["output_signal_id"]) in spans


def _wire_edges(trace: dict[str, Any]) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    edges: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for wire in trace["render_map"]["wire_segments_px"]:
        points = [(float(point[0]), float(point[1])) for point in wire["points"]]
        assert len(points) >= 2
        for start, end in zip(points, points[1:]):
            assert abs(start[0] - end[0]) <= 0.5 or abs(start[1] - end[1]) <= 0.5
            if abs(start[0] - end[0]) > 0.5 or abs(start[1] - end[1]) > 0.5:
                edges.append((str(wire["wire_id"]), start, end))
    assert edges
    return edges


def _same_point(a: Sequence[float], b: Sequence[float], *, eps: float = 0.5) -> bool:
    return abs(float(a[0]) - float(b[0])) <= eps and abs(float(a[1]) - float(b[1])) <= eps


def _between(value: float, a: float, b: float, *, eps: float = 0.5) -> bool:
    return min(float(a), float(b)) - eps <= float(value) <= max(float(a), float(b)) + eps


def _bad_axis_aligned_intersection(
    first: tuple[str, tuple[float, float], tuple[float, float]],
    second: tuple[str, tuple[float, float], tuple[float, float]],
) -> bool:
    first_id, a0, a1 = first
    second_id, b0, b1 = second
    if first_id == second_id:
        return False

    first_vertical = abs(a0[0] - a1[0]) <= 0.5
    second_vertical = abs(b0[0] - b1[0]) <= 0.5
    first_horizontal = abs(a0[1] - a1[1]) <= 0.5
    second_horizontal = abs(b0[1] - b1[1]) <= 0.5

    if first_vertical and second_horizontal:
        intersection = (a0[0], b0[1])
        if _between(intersection[1], a0[1], a1[1]) and _between(intersection[0], b0[0], b1[0]):
            return not any(_same_point(intersection, endpoint) for endpoint in (a0, a1)) or not any(
                _same_point(intersection, endpoint) for endpoint in (b0, b1)
            )
        return False
    if first_horizontal and second_vertical:
        return _bad_axis_aligned_intersection(second, first)
    if first_vertical and second_vertical and abs(a0[0] - b0[0]) <= 0.5:
        overlap_low = max(min(a0[1], a1[1]), min(b0[1], b1[1]))
        overlap_high = min(max(a0[1], a1[1]), max(b0[1], b1[1]))
        return (overlap_high - overlap_low) > 0.5
    if first_horizontal and second_horizontal and abs(a0[1] - b0[1]) <= 0.5:
        overlap_low = max(min(a0[0], a1[0]), min(b0[0], b1[0]))
        overlap_high = min(max(a0[0], a1[0]), max(b0[0], b1[0]))
        return (overlap_high - overlap_low) > 0.5
    return False


def _assert_wire_routing_clear(trace: dict[str, Any]) -> None:
    edges = _wire_edges(trace)
    for index, first in enumerate(edges):
        for second in edges[index + 1 :]:
            assert not _bad_axis_aligned_intersection(first, second), (first, second)


def _assert_standard_gate_symbol_rendering(trace: dict[str, Any]) -> None:
    assert trace["render_spec"]["logic_gate_style"]["gate_rendering"] == "standard_logic_symbols_no_gate_text"


def test_logic_gate_tasks_are_registered_and_taxonomized() -> None:
    assert TASK_REGISTRY[GATE_TYPE_COUNT_TASK_ID] is SymbolicLogicGateGateTypeCountTask
    assert TASK_REGISTRY[INTERNAL_OUTPUT_COUNT_TASK_ID] is SymbolicLogicGateInternalOutputCountTask
    assert TASK_REGISTRY[OUTPUT_VALUE_LABEL_TASK_ID] is SymbolicLogicGateOutputValueLabelTask
    assert TASK_REGISTRY[ASSIGNMENT_TASK_ID] is SymbolicLogicGateSatisfyingAssignmentLabelTask
    for task_id in (
        GATE_TYPE_COUNT_TASK_ID,
        INTERNAL_OUTPUT_COUNT_TASK_ID,
        OUTPUT_VALUE_LABEL_TASK_ID,
        ASSIGNMENT_TASK_ID,
    ):
        task = TASK_REGISTRY[task_id]()
        taxonomy = resolve_task_taxonomy(task_id)
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")
        assert taxonomy.domain == "symbolic"
        assert taxonomy.scene_id == SCENE_ID
        assert taxonomy.source_scene_id == ""


def test_logic_gate_truth_tables() -> None:
    assert evaluate_logic_gate("AND", [1, 1]) == 1
    assert evaluate_logic_gate("AND", [1, 0]) == 0
    assert evaluate_logic_gate("OR", [0, 1]) == 1
    assert evaluate_logic_gate("OR", [0, 0]) == 0
    assert evaluate_logic_gate("NOT", [0]) == 1
    assert evaluate_logic_gate("NOT", [1]) == 0
    assert evaluate_logic_gate("XOR", [1, 0]) == 1
    assert evaluate_logic_gate("XOR", [1, 1]) == 0
    assert evaluate_logic_gate("NAND", [1, 1]) == 0
    assert evaluate_logic_gate("NAND", [1, 0]) == 1
    assert evaluate_logic_gate("NOR", [0, 0]) == 1
    assert evaluate_logic_gate("NOR", [0, 1]) == 0


def test_logic_gate_output_one_label_contract() -> None:
    out = SymbolicLogicGateOutputValueLabelTask().generate(
        2026060701,
        params={"query_id": OUTPUT_ONE_LABEL_QUERY_ID, "answer_label": "C", "scene_variant": "clean_worksheet"},
        max_attempts=12,
    )
    trace = out.trace_payload
    circuits = trace["execution_trace"]["circuits"]
    matching = [circuit for circuit in circuits if circuit["output_value"] == 1]

    assert out.scene_id == SCENE_ID
    assert out.query_id == OUTPUT_ONE_LABEL_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert len(circuits) == 4
    assert [circuit["label"] for circuit in circuits] == ["A", "B", "C", "D"]
    assert len(matching) == 1
    assert matching[0]["label"] == "C"
    for circuit in circuits:
        assert len(circuit["inputs"]) == 2
        _assert_gate_arities(circuit)
        _assert_fanout_free(circuit)
        _assert_planar_expression_tree(circuit)
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["render_spec"]["logic_gate_style"]["layout"] == "four_circuit_option_grid"
    assert trace["render_spec"]["logic_gate_style"]["wire_routing"] == "planar_expression_tree_orthogonal_v4"
    _assert_standard_gate_symbol_rendering(trace)
    _assert_wire_routing_clear(trace)
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value
    assert set(LABEL_QUERY_IDS) == {OUTPUT_ONE_LABEL_QUERY_ID, OUTPUT_ZERO_LABEL_QUERY_ID}

    width, height = out.image.size
    bbox = out.annotation_gt.value
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_logic_gate_output_zero_label_contract() -> None:
    out = SymbolicLogicGateOutputValueLabelTask().generate(
        2026060702,
        params={"query_id": OUTPUT_ZERO_LABEL_QUERY_ID, "answer_label": "A", "scene_variant": "exam_scan"},
        max_attempts=12,
    )
    trace = out.trace_payload
    matching = [circuit for circuit in trace["execution_trace"]["circuits"] if circuit["output_value"] == 0]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "A"
    assert out.annotation_gt.type == "bbox"
    assert len(matching) == 1
    assert matching[0]["label"] == "A"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_logic_gate_gate_type_count_contract() -> None:
    out = SymbolicLogicGateGateTypeCountTask().generate(
        2026060704,
        params={
            "target_gate_type": "XOR",
            "answer_value": 2,
            "scene_variant": "clean_worksheet",
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    circuit = trace["execution_trace"]["source_circuit"]
    matching_gates = [gate for gate in circuit["gates"] if gate["gate_type"] == "XOR"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 2
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 2
    assert len(circuit["gates"]) == 4
    assert len(matching_gates) == 2
    assert trace["execution_trace"]["internal_query_id"] == GATE_TYPE_COUNT_INTERNAL_QUERY_ID
    assert trace["execution_trace"]["target_gate_type"] == "XOR"
    assert trace["execution_trace"]["target_answer_support"] == [0, 1, 2, 3, 4]
    assert trace["execution_trace"]["logic_gate_metadata"]["gate_type_counts"]["XOR"] == 2
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["render_spec"]["logic_gate_style"]["layout"] == "single_circuit_panel"
    assert trace["render_spec"]["logic_gate_style"]["wire_routing"] == "planar_expression_tree_orthogonal_v4"
    _assert_standard_gate_symbol_rendering(trace)
    _assert_wire_routing_clear(trace)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert "XOR" in out.prompt
    _assert_gate_arities(circuit)
    _assert_fanout_free(circuit)
    _assert_planar_expression_tree(circuit)

    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        assert len(bbox) == 4
        assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
        assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_logic_gate_internal_output_count_contract() -> None:
    out = SymbolicLogicGateInternalOutputCountTask().generate(
        2026060705,
        params={
            "query_id": INTERNAL_OUTPUT_ZERO_QUERY_ID,
            "answer_value": 3,
            "scene_variant": "clean_worksheet",
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    circuit = trace["execution_trace"]["source_circuit"]
    gate_outputs = trace["execution_trace"]["gate_outputs"]
    matching_gate_ids = [gate["item_id"] for gate in circuit["gates"] if int(gate_outputs[gate["item_id"]]) == 0]

    assert out.scene_id == SCENE_ID
    assert out.query_id == INTERNAL_OUTPUT_ZERO_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 3
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 3
    assert len(circuit["gates"]) == 4
    assert len(gate_outputs) == 4
    assert len(matching_gate_ids) == 3
    assert trace["execution_trace"]["target_output_value"] == 0
    assert trace["execution_trace"]["target_answer_support"] == [0, 1, 2, 3, 4]
    assert trace["execution_trace"]["logic_gate_metadata"]["gate_output_counts"]["0"] == 3
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["render_spec"]["logic_gate_style"]["layout"] == "single_circuit_panel"
    assert trace["render_spec"]["logic_gate_style"]["wire_routing"] == "planar_expression_tree_orthogonal_v4"
    _assert_standard_gate_symbol_rendering(trace)
    _assert_wire_routing_clear(trace)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert "0" in out.prompt
    _assert_gate_arities(circuit)
    _assert_fanout_free(circuit)
    _assert_planar_expression_tree(circuit)

    width, height = out.image.size
    for bbox in out.annotation_gt.value:
        assert len(bbox) == 4
        assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
        assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_logic_gate_wire_routing_regression_samples_are_planar() -> None:
    cases = (
        (1409349431609073, "NAND"),
        (2354966825196157, "AND"),
    )
    for seed, target_gate_type in cases:
        out = SymbolicLogicGateGateTypeCountTask().generate(
            seed,
            params={
                "target_gate_type": target_gate_type,
                "answer_value": 3,
                "scene_variant": "clean_worksheet",
            },
            max_attempts=12,
        )
        circuit = out.trace_payload["execution_trace"]["source_circuit"]
        _assert_gate_arities(circuit)
        _assert_fanout_free(circuit)
        _assert_planar_expression_tree(circuit)
        _assert_standard_gate_symbol_rendering(out.trace_payload)
        _assert_wire_routing_clear(out.trace_payload)


def test_logic_gate_satisfying_assignment_contract() -> None:
    out = SymbolicLogicGateSatisfyingAssignmentLabelTask().generate(
        2026060703,
        params={
            "query_id": ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID,
            "answer_label": "D",
            "correct_values": {"x": 1, "y": 0, "z": 1},
            "scene_variant": "notebook_problem",
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    candidates = trace["execution_trace"]["candidates"]
    correct_candidates = [candidate for candidate in candidates if candidate["output_value"] == 0]

    assert out.scene_id == SCENE_ID
    assert out.query_id == ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value.keys()) == ["selected_option", "source_circuit"]
    assert len(candidates) == 4
    _assert_gate_arities(trace["execution_trace"]["source_circuit"])
    _assert_fanout_free(trace["execution_trace"]["source_circuit"])
    assert [candidate["label"] for candidate in candidates] == ["A", "B", "C", "D"]
    assert trace["execution_trace"]["target_answer_support"] == ["A", "B", "C", "D"]
    assert len(correct_candidates) == 1
    assert correct_candidates[0]["label"] == "D"
    assert correct_candidates[0]["values"] == {"x": 1, "y": 0, "z": 1}
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["render_spec"]["logic_gate_style"]["wire_routing"] == "planar_expression_tree_orthogonal_v4"
    _assert_standard_gate_symbol_rendering(trace)
    _assert_wire_routing_clear(trace)
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        assert len(bbox) == 4
        assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
        assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_logic_gate_generation_is_deterministic() -> None:
    params = {"query_id": ASSIGNMENT_OUTPUTS_ONE_QUERY_ID, "answer_label": "B", "scene_variant": "exam_scan"}
    out_a = SymbolicLogicGateSatisfyingAssignmentLabelTask().generate(2026060799, params=params, max_attempts=12)
    out_b = SymbolicLogicGateSatisfyingAssignmentLabelTask().generate(2026060799, params=params, max_attempts=12)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_symbolic_logic_gate_circuit_prompt_bundle_supports_logic_gate_queries() -> None:
    bundle = load_prompt_bundle("symbolic", "logic_gate_circuit", "symbolic_logic_gate_circuit_v1")
    assert "logic_gate_circuit" in bundle.scene_templates
    assert len(bundle.task_templates["logic_gate_gate_type_count"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["logic_gate_internal_output_count"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["logic_gate_output_value_label"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["logic_gate_satisfying_assignment_label"]) == REQUIRED_PROMPT_VARIANTS
    for query_id in (
        OUTPUT_ONE_LABEL_QUERY_ID,
        OUTPUT_ZERO_LABEL_QUERY_ID,
        ASSIGNMENT_OUTPUTS_ONE_QUERY_ID,
        ASSIGNMENT_OUTPUTS_ZERO_QUERY_ID,
        INTERNAL_OUTPUT_ONE_QUERY_ID,
        INTERNAL_OUTPUT_ZERO_QUERY_ID,
    ):
        assert len(bundle.query_templates[query_id]) == REQUIRED_PROMPT_VARIANTS
        assert list(bundle.required_slots_by_key[f"query:{query_id}"]) == []
    assert len(bundle.query_templates[GATE_TYPE_COUNT_INTERNAL_QUERY_ID]) == REQUIRED_PROMPT_VARIANTS
    assert list(bundle.required_slots_by_key[f"query:{GATE_TYPE_COUNT_INTERNAL_QUERY_ID}"]) == ["target_gate_type"]
    assert list(bundle.required_slots_by_key["scene:logic_gate_circuit"]) == ["object_description"]
