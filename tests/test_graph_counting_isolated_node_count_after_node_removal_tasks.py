"""Behavior tests for graph isolated-node count after node removal."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.isolated_after_removal_count import (
    GraphCountingIsolatedNodeCountAfterNodeRemovalTask,
)
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def test_graph_counting_isolated_node_count_after_node_removal_directed_contract() -> None:
    task = GraphCountingIsolatedNodeCountAfterNodeRemovalTask()
    out = task.generate(
        21401,
        params={
            "graph_directionality": "directed",
            "target_count": 3,
            "node_count": 8,
            "label_variant": "named",
            "edge_routing_variant": "mixed_arc",
            "layout_variant": "shell",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    node_entities = [
        entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_node"
    ]
    edge_entities = [
        entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_edge"
    ]

    assert "task_graph__node_link__isolated_after_removal_count" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert int(out.answer_gt.value) == 3
    assert trace["scene_ir"]["scene_kind"] == "graph_isolated_after_removal_counting"
    assert execution["query_id"] == "single"
    assert execution["graph_directionality"] == "directed"
    assert execution["layout_variant_requested"] == "shell"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"

    removed_label = str(execution["removed_node_label"])
    matching_labels = [str(label) for label in execution["matching_labels"]]
    assert f'"{removed_label}"' in str(out.prompt)
    assert removed_label not in execution["post_removal_degrees_by_label"]
    assert int(out.answer_gt.value) == len(matching_labels) == len(out.annotation_gt.value)
    assert trace["witness_symbolic"]["labels"] == matching_labels
    assert trace["witness_symbolic"]["removed_node_label"] == removed_label

    for label in matching_labels:
        assert int(execution["post_removal_degrees_by_label"][label]) == 0
        assert int(execution["post_removal_in_degrees_by_label"][label]) == 0
        assert int(execution["post_removal_out_degrees_by_label"][label]) == 0
        assert set(execution["pre_removal_adjacency_by_label"][label]) == {removed_label}
    for label, degree in execution["post_removal_degrees_by_label"].items():
        if str(label) not in set(matching_labels):
            assert int(degree) > 0

    assert sum(1 for node in node_entities if bool(node["is_removed_query_node"])) == 1
    assert sum(1 for node in node_entities if bool(node["is_post_removal_isolated"])) == 3
    assert len(edge_entities) == int(execution["edge_count"])
    assert any(bool(edge["directed"]) for edge in edge_entities)
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_counting_isolated_node_count_after_node_removal_zero_answer() -> None:
    task = GraphCountingIsolatedNodeCountAfterNodeRemovalTask()
    out = task.generate(
        21402,
        params={
            "graph_directionality": "undirected",
            "target_count": 0,
            "node_count": 7,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    execution = out.trace_payload["execution_trace"]

    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []
    assert execution["target_count"] == 0
    assert execution["matching_labels"] == []
    assert all(int(degree) > 0 for degree in execution["post_removal_degrees_by_label"].values())
    assert str(execution["removed_node_label"]) not in execution["post_removal_adjacency_by_label"]


def test_graph_counting_isolated_node_count_after_node_removal_prompt_examples_match_contract() -> None:
    task = GraphCountingIsolatedNodeCountAfterNodeRemovalTask()
    out = task.generate(
        21403,
        params={"target_count": 2, "node_count": 6},
        max_attempts=100,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[180, 220], [310, 180]]
    assert answer_and_annotation["answer"] == 2


def test_graph_counting_isolated_node_count_after_node_removal_balanced_sampling_includes_zero() -> None:
    task = GraphCountingIsolatedNodeCountAfterNodeRemovalTask()
    answers: Counter[int] = Counter()
    directionality: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(72):
        out = task.generate(
            hash64(21410, "graph_counting_isolated_node_count_after_node_removal", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        answers[int(out.answer_gt.value)] += 1
        directionality[str(execution["graph_directionality"])] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        assert int(out.answer_gt.value) == int(execution["target_count"])
        assert 0 <= int(out.answer_gt.value) <= 5

    assert set(answers) == set(range(0, 6))
    assert set(directionality) == {"undirected", "directed"}
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}


def test_graph_counting_isolated_node_count_after_node_removal_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_isolated_node_count_after_node_removal"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_isolated_node_count_after_node_removal",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__node_link__isolated_after_removal_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="graph-counting-isolated-node-after-removal-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__node_link__isolated_after_removal_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
