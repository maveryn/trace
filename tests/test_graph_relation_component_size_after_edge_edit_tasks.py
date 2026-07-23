"""Behavior tests for graph component-size-after-edge-edit task."""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.component_size_after_edge_edit import GraphRelationComponentSizeAfterEdgeEditTask
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _component_from_adjacency(adjacency: dict, start: str) -> set[str]:
    seen = {str(start)}
    queue: deque[str] = deque([str(start)])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(str(node), []):
            text = str(neighbor)
            if text in seen:
                continue
            seen.add(text)
            queue.append(text)
    return seen


def test_graph_relation_component_size_after_edge_removal_contract_matches_trace() -> None:
    task = GraphRelationComponentSizeAfterEdgeEditTask()
    out = task.generate(
        20901,
        params={
            "edit_operation": "edge_removal",
            "target_component_size": 3,
            "node_count": 8,
            "label_variant": "named",
            "edge_routing_variant": "mixed_arc",
            "layout_variant": "shell",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    query_label = str(execution["query_label"])
    edit_a, edit_b = [str(label) for label in execution["edit_edge"]]
    post_component = _component_from_adjacency(execution["post_edit_adjacency_by_label"], query_label)

    assert "task_graph__node_link__component_size_after_edge_edit" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "component_size_after_edge_removal"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert int(out.answer_gt.value) == 3
    assert len(out.annotation_gt.value) == 3
    assert trace["scene_ir"]["scene_kind"] == "graph_component_size_after_edge_edit"
    assert execution["edit_operation"] == "edge_removal"
    assert execution["edit_edge_visible_in_rendered_graph"] is True
    assert execution["graph_directionality"] == "undirected"
    assert execution["label_variant"] == "named"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert query_label in post_component
    assert post_component == set(str(label) for label in execution["matching_labels"])
    assert edit_b in execution["pre_edit_adjacency_by_label"][edit_a]
    assert edit_a in execution["pre_edit_adjacency_by_label"][edit_b]
    assert edit_b not in execution["post_edit_adjacency_by_label"][edit_a]
    assert edit_a not in execution["post_edit_adjacency_by_label"][edit_b]
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["witness_symbolic"]["labels"] == execution["matching_labels"]
    assert f'node "{query_label}"' in str(out.prompt)
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_relation_component_size_after_edge_addition_contract_matches_trace() -> None:
    task = GraphRelationComponentSizeAfterEdgeEditTask()
    out = task.generate(
        20902,
        params={
            "edit_operation": "edge_addition",
            "target_component_size": 5,
            "node_count": 9,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    execution = out.trace_payload["execution_trace"]
    query_label = str(execution["query_label"])
    edit_a, edit_b = [str(label) for label in execution["edit_edge"]]
    post_component = _component_from_adjacency(execution["post_edit_adjacency_by_label"], query_label)

    assert out.query_id == "component_size_after_edge_addition"
    assert int(out.answer_gt.value) == 5
    assert len(out.annotation_gt.value) == 5
    assert execution["edit_operation"] == "edge_addition"
    assert execution["edit_edge_visible_in_rendered_graph"] is False
    assert edit_b not in execution["pre_edit_adjacency_by_label"][edit_a]
    assert edit_a not in execution["pre_edit_adjacency_by_label"][edit_b]
    assert edit_b in execution["post_edit_adjacency_by_label"][edit_a]
    assert edit_a in execution["post_edit_adjacency_by_label"][edit_b]
    assert post_component == set(str(label) for label in execution["matching_labels"])


def test_graph_relation_component_size_after_edge_edit_prompt_examples_match_contract() -> None:
    task = GraphRelationComponentSizeAfterEdgeEditTask()
    out = task.generate(
        20903,
        params={"edit_operation": "edge_removal", "target_component_size": 3},
        max_attempts=100,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 3}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[180, 220], [310, 180], [430, 260]]
    assert answer_and_annotation["answer"] == 3


def test_graph_relation_component_size_after_edge_edit_balanced_sampling() -> None:
    task = GraphRelationComponentSizeAfterEdgeEditTask()
    answers: Counter[int] = Counter()
    query_ids: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(112):
        out = task.generate(
            hash64(20910, "graph_relation_component_size_after_edge_edit", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        answers[int(out.answer_gt.value)] += 1
        query_ids[str(out.query_id)] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        assert int(out.answer_gt.value) == int(execution["target_component_size"])
        assert 1 <= int(out.answer_gt.value) <= 8

    assert set(answers) == set(range(1, 9))
    assert set(query_ids) == {"component_size_after_edge_removal", "component_size_after_edge_addition"}
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}


def test_graph_relation_component_size_after_edge_edit_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_component_size_after_edge_edit"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_component_size_after_edge_edit",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__node_link__component_size_after_edge_edit",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="graph-relation-component-size-after-edge-edit-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__node_link__component_size_after_edge_edit"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
