"""Behavior tests for graph automaton state-simulation relation task."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.automaton.state_after_input_label import (
    GraphRelationAutomatonStateSimulationLabelTask,
)
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _follow_path(trace: dict) -> list[str]:
    execution = trace["execution_trace"]
    transition_function = execution["transition_function"]
    state = str(execution["start_state_label"])
    path = [state]
    for symbol in str(execution["input_string"]):
        state = str(transition_function[state][str(symbol)])
        path.append(state)
    return path


def test_graph_relation_automaton_final_state_contract_matches_trace() -> None:
    task = GraphRelationAutomatonStateSimulationLabelTask()
    out = task.generate(
        31001,
        params={
            "query_id": "final_state_label",
            "state_count": 5,
            "input_length": 4,
            "target_state_index": 3,
            "distractor_edge_count": 4,
            "layout_variant": "shell",
            "edge_routing_variant": "mixed_arc",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    state_entities = [
        entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "automaton_state"
    ]

    assert "task_graph__automaton__state_after_input_label" in TASK_REGISTRY
    assert out.scene_id == "automaton"
    assert out.query_id == "final_state_label"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point_sequence"
    assert len(out.annotation_gt.value) == 5
    assert trace["scene_ir"]["scene_kind"] == "automaton_state_transition_simulation"
    assert execution["query_id"] == "final_state_label"
    assert execution["state_count"] == 5
    assert execution["input_length"] == 4
    assert execution["transition_step_count"] == 4
    assert execution["layout_variant_requested"] == "shell"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert str(execution["input_string"]) in str(out.prompt)

    full_path = _follow_path(trace)
    assert execution["full_state_path_labels"] == full_path
    assert execution["annotation_state_path_labels"] == full_path
    assert out.answer_gt.value == full_path[-1] == execution["answer_state_label"]
    assert trace["witness_symbolic"]["state_path_labels"] == full_path
    assert trace["projected_annotation"]["type"] == "point_sequence"
    assert trace["projected_annotation"]["point_sequence"] == out.annotation_gt.value
    assert len(execution["used_transition_label_bboxes"]) == 4
    assert sum(1 for state in state_entities if bool(state["is_start_state"])) == 1
    assert sum(1 for state in state_entities if bool(state["is_answer_state"])) >= 1
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_relation_automaton_step_state_contract_matches_trace() -> None:
    task = GraphRelationAutomatonStateSimulationLabelTask()
    out = task.generate(
        31002,
        params={
            "query_id": "transition_step_state_label",
            "state_count": 6,
            "input_length": 5,
            "transition_step_count": 3,
            "target_state_index": 2,
            "layout_variant": "layered",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    full_path = _follow_path(trace)

    assert out.query_id == "transition_step_state_label"
    assert execution["input_length"] == 5
    assert execution["transition_step_count"] == 3
    assert len(out.annotation_gt.value) == 4
    assert execution["full_state_path_labels"] == full_path
    assert execution["annotation_state_path_labels"] == full_path[:4]
    assert out.answer_gt.value == full_path[3] == execution["answer_state_label"]
    assert len(execution["used_transition_edges"]) == 3
    assert len(execution["used_transition_label_bboxes"]) == 3


def test_graph_relation_automaton_prompt_examples_match_contract() -> None:
    task = GraphRelationAutomatonStateSimulationLabelTask()
    out = task.generate(
        31003,
        params={"query_id": "final_state_label", "state_count": 4, "input_length": 3},
        max_attempts=100,
    )

    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": "C"}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[150, 250], [310, 190], [480, 230]]
    assert answer_and_annotation["answer"] == "C"
    assert "[x,y]" in out.prompt_variants["answer_and_annotation"]


def test_graph_relation_automaton_balanced_sampling_covers_queries_and_answers() -> None:
    task = GraphRelationAutomatonStateSimulationLabelTask()
    query_ids: Counter[str] = Counter()
    state_counts: Counter[int] = Counter()
    answers: Counter[str] = Counter()
    layout_variants: Counter[str] = Counter()
    for index in range(90):
        out = task.generate(
            hash64(31010, "graph_relation_automaton_state_simulation_label", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        query_ids[str(out.query_id)] += 1
        state_counts[int(execution["state_count"])] += 1
        answers[str(out.answer_gt.value)] += 1
        layout_variants[str(execution["layout_variant_requested"])] += 1
        assert str(out.answer_gt.value) == str(execution["answer_state_label"])
        assert len(out.annotation_gt.value) == int(execution["transition_step_count"]) + 1

    assert set(query_ids) == {"final_state_label", "transition_step_state_label"}
    assert all(30 <= count <= 60 for count in query_ids.values())
    assert set(state_counts) == {4, 5, 6}
    assert len(answers) >= 5
    assert len(layout_variants) >= 4


def test_graph_relation_automaton_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_automaton_state_simulation"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_automaton_state_simulation",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__automaton__state_after_input_label",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash="graph-relation-automaton-state-simulation-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "automaton" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__automaton__state_after_input_label"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
