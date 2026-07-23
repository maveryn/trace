"""Behavior tests for graph edge-attribute label relation task."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.edge_between_nodes_label import GraphRelationEdgeBetweenNodesLabelTask
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _bbox_min_side(bbox: list[int] | tuple[int, ...]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return min(abs(x1 - x0), abs(y1 - y0))


def test_graph_relation_edge_attribute_label_directed_contract_matches_trace() -> None:
    task = GraphRelationEdgeBetweenNodesLabelTask()
    out = task.generate(
        20901,
        params={
            "query_id": "directed_edge_between_nodes_label",
            "graph_directionality": "directed",
            "target_edge_label": "feeds",
            "edge_label_support": ["feeds", "blocks", "joins", "routes", "checks", "updates"],
            "node_count": 7,
            "label_variant": "named",
            "edge_routing_variant": "mixed_arc",
            "layout_variant": "shell",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_spec = trace["render_spec"]
    edge_entities = [
        entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_edge"
    ]

    assert "task_graph__node_link__edge_between_nodes_label" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "directed_edge_between_nodes_label"
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "feeds"
    assert out.annotation_gt.type == "bbox"
    assert trace["scene_ir"]["scene_kind"] == "graph_edge_label_lookup"
    assert execution["query_id"] == "directed_edge_between_nodes_label"
    assert execution["graph_directionality"] == "directed"
    assert execution["target_edge_label"] == "feeds"
    assert execution["question_format"] == "directed_edge_between_nodes_label"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"
    assert render_spec["canvas_size"] == [960, 720]
    assert render_spec["style"]["label_font_size_px"] == 20
    assert render_spec["style"]["edge_text_label_font_size_px"] == 22
    assert render_spec["style"]["resolved_edge_text_label_font_size_px"] == 22
    assert 'node "' in str(out.prompt)

    query_edge = tuple(str(value) for value in trace["witness_symbolic"]["edge_labels"][0])
    labels_by_edge = {
        tuple(str(value) for value in entry["edge"]): str(entry["edge_label"])
        for entry in execution["edge_attribute_labels_by_label_pair"]
    }
    assert labels_by_edge[query_edge] == "feeds"
    assert trace["witness_symbolic"]["edge_labels"] == [list(query_edge)]
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    query_entity = [
        edge
        for edge in edge_entities
        if (str(edge["node_u_label"]), str(edge["node_v_label"])) == tuple(query_edge)
    ][0]
    assert query_entity["edge_text_label"] == "feeds"
    bbox = query_entity["label_bbox_xyxy"]
    assert bbox is not None
    assert out.annotation_gt.value == bbox
    assert _bbox_min_side(out.annotation_gt.value) >= 24
    assert all(edge["label_bbox_xyxy"] is not None for edge in edge_entities)
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_relation_edge_attribute_label_undirected_prompt_and_example_contract() -> None:
    task = GraphRelationEdgeBetweenNodesLabelTask()
    out = task.generate(
        20902,
        params={
            "query_id": "edge_between_nodes_label",
            "graph_directionality": "undirected",
            "target_edge_label": "blocks",
            "edge_label_support": ["feeds", "blocks", "joins", "routes", "checks", "updates"],
            "label_variant": "numbers",
        },
        max_attempts=100,
    )

    assert out.query_id == "edge_between_nodes_label"
    assert out.answer_gt.value == "blocks"
    assert "edge" in str(out.prompt)
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": "alpha"}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [180, 220, 230, 245]
    assert answer_and_annotation["answer"] == "alpha"


def test_graph_relation_edge_attribute_label_balanced_sampling_covers_label_support() -> None:
    task = GraphRelationEdgeBetweenNodesLabelTask()
    answers: Counter[str] = Counter()
    query_ids: Counter[str] = Counter()
    directionality: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(
            hash64(20910, "graph_relation_edge_attribute_label", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        answers[str(out.answer_gt.value)] += 1
        query_ids[str(out.query_id)] += 1
        directionality[str(execution["graph_directionality"])] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        query_edge = tuple(str(value) for value in out.trace_payload["witness_symbolic"]["edge_labels"][0])
        labels_by_edge = {
            tuple(str(value) for value in entry["edge"]): str(entry["edge_label"])
            for entry in execution["edge_attribute_labels_by_label_pair"]
        }
        assert labels_by_edge[query_edge] == str(out.answer_gt.value)
        assert str(out.answer_gt.value) in set(execution["edge_label_support"])
        assert len(execution["edge_label_support"]) == 16
        assert all(3 <= len(str(label)) <= 5 for label in execution["edge_label_support"])
        node_labels = {
            str(entity["label"]).strip().lower()
            for entity in out.trace_payload["scene_ir"]["entities"]
            if entity["entity_kind"] == "graph_node"
        }
        assert not (set(str(label) for label in execution["edge_label_support"]) & node_labels)
        assert int(execution["edge_count"]) <= 12
        assert execution["edge_label_source_kind"] == "shared_label_manifest"
        assert execution["edge_label_bucket"]
        assert execution["edge_label_manifest"]
        assert out.annotation_gt.type == "bbox"
        assert len(out.annotation_gt.value) == 4

    assert len(answers) > 12
    assert set(query_ids) == {
        "edge_between_nodes_label",
        "directed_edge_between_nodes_label",
    }
    assert all(count > 0 for count in query_ids.values())
    assert set(directionality) == {"undirected", "directed"}
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}


def test_graph_relation_edge_attribute_label_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_edge_attribute_label"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_edge_attribute_label",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__node_link__edge_between_nodes_label",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=39,
    )
    final_path = build_dataset(config, code_hash="graph-relation-edge-attribute-label-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__node_link__edge_between_nodes_label"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
