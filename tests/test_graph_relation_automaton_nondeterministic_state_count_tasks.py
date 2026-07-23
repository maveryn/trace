"""Behavior tests for graph automaton nondeterministic-state count task."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.automaton.nondeterministic_state_count import (
    EPSILON_SYMBOL,
    GraphRelationAutomatonNondeterministicStateCountTask,
)
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _nondeterministic_states_from_trace(trace: dict) -> list[str]:
    execution = trace["execution_trace"]
    result: list[str] = []
    for state_label, per_symbol in execution["transition_function"].items():
        if EPSILON_SYMBOL in per_symbol and len(per_symbol[EPSILON_SYMBOL]) > 0:
            result.append(str(state_label))
            continue
        if any(len(targets) > 1 for symbol, targets in per_symbol.items() if str(symbol) != EPSILON_SYMBOL):
            result.append(str(state_label))
    label_order = {label: index for index, label in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    return sorted(result, key=lambda label: label_order.get(str(label), 999))


def test_graph_relation_automaton_nondeterministic_state_count_contract_matches_trace() -> None:
    task = GraphRelationAutomatonNondeterministicStateCountTask()
    out = task.generate(
        912101,
        params={
            "state_count": 5,
            "target_count": 2,
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

    assert "task_graph__automaton__nondeterministic_state_count" in TASK_REGISTRY
    assert out.scene_id == "automaton"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 2
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == 2
    assert trace["scene_ir"]["scene_kind"] == "automaton_nondeterministic_state_count"
    assert execution["query_id"] == "single"
    assert execution["state_count"] == 5
    assert execution["target_count"] == 2
    assert execution["layout_variant_requested"] == "shell"
    assert execution["edge_routing_variant"] == "mixed_arc"

    recomputed = _nondeterministic_states_from_trace(trace)
    assert execution["nondeterministic_state_labels"] == recomputed
    assert out.answer_gt.value == len(recomputed)
    assert trace["witness_symbolic"]["nondeterministic_state_labels"] == recomputed
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert len(execution["witness_edges_by_state"]) == 2
    assert all(execution["witness_edges_by_state"][label] for label in recomputed)
    assert sum(1 for state in state_entities if bool(state["is_nondeterministic_state"])) == 2
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_relation_automaton_nondeterministic_state_count_zero_answer() -> None:
    task = GraphRelationAutomatonNondeterministicStateCountTask()
    out = task.generate(
        912102,
        params={
            "state_count": 4,
            "target_count": 0,
            "layout_variant": "circular",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.value == 0
    assert out.annotation_gt.value == []
    assert execution["nondeterministic_state_labels"] == []
    assert execution["witness_edges_by_state"] == {}
    assert _nondeterministic_states_from_trace(trace) == []


def test_graph_relation_automaton_nondeterministic_state_count_prompt_examples_match_contract() -> None:
    task = GraphRelationAutomatonNondeterministicStateCountTask()
    out = task.generate(
        912103,
        params={"state_count": 4, "target_count": 1},
        max_attempts=100,
    )

    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[150, 250], [310, 190]]
    assert answer_and_annotation["answer"] == 2
    assert EPSILON_SYMBOL in out.prompt
    assert "Missing" not in out.prompt
    assert "[x,y]" in out.prompt_variants["answer_and_annotation"]


def test_graph_relation_automaton_nondeterministic_state_count_sampling_smoke() -> None:
    task = GraphRelationAutomatonNondeterministicStateCountTask()
    state_counts: Counter[int] = Counter()
    answers: Counter[int] = Counter()
    layout_variants: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(
            hash64(912110, "graph_relation_automaton_nondeterministic_state_count", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        state_counts[int(execution["state_count"])] += 1
        answers[int(out.answer_gt.value)] += 1
        layout_variants[str(execution["layout_variant_requested"])] += 1
        assert int(out.answer_gt.value) == len(execution["nondeterministic_state_labels"])
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert _nondeterministic_states_from_trace(out.trace_payload) == execution["nondeterministic_state_labels"]

    assert set(state_counts) == {4, 5, 6}
    assert set(answers).issubset({0, 1, 2, 3, 4, 5})
    assert len(answers) >= 5
    assert len(layout_variants) >= 4


def test_graph_relation_automaton_nondeterministic_state_count_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_automaton_nondeterministic_state_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_automaton_nondeterministic_state_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__automaton__nondeterministic_state_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash="graph-relation-automaton-nondeterministic-state-count-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "automaton" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__automaton__nondeterministic_state_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
