"""Behavior tests for graph max-flow optimization task."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.flow_network.max_flow_value import GraphFlowNetworkMaxFlowValueTask
from trace_tasks.tasks.graph.flow_network.min_cut_edge_count import GraphFlowNetworkMinCutEdgeCountTask
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _capacity_graph_from_trace(trace: dict, *, remove_edge: tuple[str, str] | None = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    for entity in trace["scene_ir"]["entities"]:
        if entity["entity_kind"] == "graph_node":
            graph.add_node(str(entity["label"]))
    for entry in trace["execution_trace"]["capacity_by_edge"]:
        left, right = [str(value) for value in entry["edge"]]
        if remove_edge is not None and (left, right) == tuple(remove_edge):
            continue
        graph.add_edge(left, right, capacity=int(entry["capacity"]))
    return graph


def _unique_min_cut(graph: nx.DiGraph, source: str = "S", sink: str = "T") -> tuple[int, tuple[tuple[str, str], ...]]:
    nodes = sorted(str(node) for node in graph.nodes())
    internal = [node for node in nodes if node not in {source, sink}]
    cuts: list[tuple[int, tuple[tuple[str, str], ...]]] = []
    for size in range(len(internal) + 1):
        for subset in combinations(internal, size):
            source_side = {source, *subset}
            cut_edges = tuple(
                sorted(
                    (str(left), str(right))
                    for left, right in graph.edges()
                    if str(left) in source_side and str(right) not in source_side
                )
            )
            value = sum(int(graph[left][right]["capacity"]) for left, right in cut_edges)
            cuts.append((int(value), cut_edges))
    min_value = min(value for value, _edges in cuts)
    min_cuts = [edges for value, edges in cuts if int(value) == int(min_value)]
    assert len(min_cuts) == 1
    return int(min_value), tuple(min_cuts[0])


def test_graph_optimization_max_flow_contract_matches_trace() -> None:
    task = GraphFlowNetworkMaxFlowValueTask()
    out = task.generate(
        32001,
        params={
            "query_id": "single",
            "target_answer": 5,
            "target_cut_edge_count": 1,
            "node_count": 5,
            "layout_variant": "layered",
            "edge_routing_variant": "straight",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert "task_graph__flow_network__max_flow_value" in TASK_REGISTRY
    assert out.scene_id == "flow_network"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "segment_set"
    assert int(out.answer_gt.value) == 5
    assert len(out.annotation_gt.value) == 1
    assert trace["scene_ir"]["scene_kind"] == "graph_capacity_flow_network"
    assert execution["graph_directionality"] == "directed"
    assert execution["source_label"] == "S"
    assert execution["sink_label"] == "T"
    assert execution["layout_variant_requested"] == "layered"
    assert execution["edge_routing_variant"] == "straight"
    assert "source S" in str(out.prompt)
    assert "sink T" in str(out.prompt)

    graph = _capacity_graph_from_trace(trace)
    min_cut_value, min_cut_edges = _unique_min_cut(graph)
    assert int(nx.maximum_flow_value(graph, "S", "T", capacity="capacity")) == int(min_cut_value) == 5
    assert tuple(tuple(edge) for edge in execution["original_min_cut_edges"]) == tuple(min_cut_edges)
    assert tuple(tuple(edge) for edge in execution["annotation_edges"]) == tuple(min_cut_edges)
    assert trace["projected_annotation"]["type"] == "segment_set"
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_optimization_max_flow_all_query_contracts() -> None:
    cases = (
        (GraphFlowNetworkMaxFlowValueTask(), "max_flow_value", 5, 1, 5),
        (GraphFlowNetworkMinCutEdgeCountTask(), "minimum_cut_edge_count", 3, 3, 6),
    )
    for offset, (task, objective, target_answer, cut_count, node_count) in enumerate(cases):
        out = task.generate(
            32010 + offset,
            params={
                "target_answer": target_answer,
                "target_cut_edge_count": cut_count,
                "node_count": node_count,
            },
            max_attempts=100,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        graph = _capacity_graph_from_trace(trace)
        min_cut_value, min_cut_edges = _unique_min_cut(graph)

        assert out.query_id == "single"
        assert execution["objective"] == objective
        assert int(nx.maximum_flow_value(graph, "S", "T", capacity="capacity")) == int(min_cut_value)
        if objective == "minimum_cut_edge_count":
            assert int(out.answer_gt.value) == len(min_cut_edges) == int(target_answer)
        else:
            assert int(out.answer_gt.value) == int(min_cut_value) == int(target_answer)
        assert tuple(tuple(edge) for edge in execution["annotation_edges"]) == tuple(min_cut_edges)
        assert len(out.annotation_gt.value) == len(min_cut_edges)


def test_graph_optimization_max_flow_prompt_examples_match_contract() -> None:
    task = GraphFlowNetworkMinCutEdgeCountTask()
    out = task.generate(32020, params={}, max_attempts=100)

    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[[170, 280], [330, 210]], [[170, 280], [330, 350]]]
    assert answer_and_annotation["answer"] == 2
    assert "node center" in out.prompt_variants["answer_and_annotation"]


def test_graph_optimization_max_flow_rejects_removed_edge_query() -> None:
    task = GraphFlowNetworkMaxFlowValueTask()
    with pytest.raises(ValueError, match="unsupported query_id"):
        task.generate(
            32025,
            params={"query_id": "max_flow_after_edge_removal_value"},
            max_attempts=20,
        )


def test_graph_optimization_max_flow_balanced_sampling_covers_queries() -> None:
    task = GraphFlowNetworkMaxFlowValueTask()
    answers: Counter[int] = Counter()
    node_counts: Counter[int] = Counter()
    layout_variants: Counter[str] = Counter()
    edge_routing: Counter[str] = Counter()
    for index in range(90):
        out = task.generate(
            hash64(32030, "graph_optimization_max_flow_value", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == "single"
        answers[int(out.answer_gt.value)] += 1
        node_counts[int(execution["node_count"])] += 1
        layout_variants[str(execution["layout_variant_requested"])] += 1
        edge_routing[str(execution["edge_routing_variant"])] += 1
        assert int(out.answer_gt.value) == int(execution["answer"])
        assert len(out.annotation_gt.value) == len(execution["annotation_edges"])

    assert set(node_counts) == {4, 5}
    assert len(answers) >= 4
    assert set(edge_routing) == {"straight"}
    assert set(layout_variants) == {"layered"}


def test_graph_optimization_max_flow_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_max_flow_value"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_max_flow_value",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__flow_network__max_flow_value",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=41,
    )
    final_path = build_dataset(config, code_hash="graph-optimization-max-flow-value-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["task"] == "task_graph__flow_network__max_flow_value" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__flow_network__max_flow_value"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
