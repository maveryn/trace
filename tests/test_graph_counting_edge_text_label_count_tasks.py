"""Behavior tests for graph edge-text-label counting task."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.edge_text_count import (
    GraphCountingEdgeTextLabelCountTask,
)
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _bbox_min_side(bbox: list[int] | tuple[int, ...]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return min(abs(x1 - x0), abs(y1 - y0))


def test_graph_counting_edge_text_label_count_contract_matches_trace() -> None:
    task = GraphCountingEdgeTextLabelCountTask()
    out = task.generate(
        21001,
        params={
            "target_edge_label": "feeds",
            "edge_label_support": [
                "feeds",
                "blocks",
                "joins",
                "routes",
                "checks",
                "updates",
            ],
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
    render_spec = trace["render_spec"]
    edge_entities = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if entity["entity_kind"] == "graph_edge"
    ]

    assert "task_graph__node_link__edge_text_count" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == 3
    assert len(out.annotation_gt.value) == 3
    assert trace["scene_ir"]["scene_kind"] == "graph_edge_text_counting"
    assert execution["query_id"] == "single"
    assert execution["graph_directionality"] == "undirected"
    assert execution["target_edge_label"] == "feeds"
    assert execution["question_format"] == "edge_text_label_count"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"
    assert render_spec["canvas_size"] == [960, 720]
    assert render_spec["style"]["label_font_size_px"] == 20
    assert render_spec["style"]["edge_text_label_font_size_px"] == 22
    assert render_spec["style"]["resolved_edge_text_label_font_size_px"] == 22
    assert '"feeds"' in str(out.prompt)

    matching_edges = [
        tuple(str(value) for value in edge) for edge in execution["matching_edges"]
    ]
    labels_by_edge = {
        tuple(str(value) for value in entry["edge"]): str(entry["edge_label"])
        for entry in execution["edge_attribute_labels_by_label_pair"]
    }
    assert int(out.answer_gt.value) == len(matching_edges) == len(out.annotation_gt.value)
    assert trace["witness_symbolic"]["edge_labels"] == [list(edge) for edge in matching_edges]
    assert all(labels_by_edge[tuple(edge)] == "feeds" for edge in matching_edges)
    assert sum(1 for label in labels_by_edge.values() if str(label) == "feeds") == 3
    assert all(
        str(edge["edge_text_label"]) == "feeds"
        for edge in edge_entities
        if (str(edge["node_u_label"]), str(edge["node_v_label"])) in set(matching_edges)
    )
    assert all(edge["label_bbox_xyxy"] is not None for edge in edge_entities)
    assert len(edge_entities) == int(execution["edge_count"])
    assert not any(bool(edge["directed"]) for edge in edge_entities)
    boxes_by_edge = {}
    for edge in edge_entities:
        bbox = edge["label_bbox_xyxy"]
        if bbox is None:
            continue
        endpoint_pair = (str(edge["node_u_label"]), str(edge["node_v_label"]))
        boxes_by_edge[endpoint_pair] = list(bbox)
        boxes_by_edge[(endpoint_pair[1], endpoint_pair[0])] = list(bbox)
    assert sorted(out.annotation_gt.value) == sorted(
        boxes_by_edge[tuple(edge)] for edge in matching_edges
    )
    assert all(_bbox_min_side(bbox) >= 24 for bbox in out.annotation_gt.value)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_counting_edge_text_label_prompt_examples_match_contract() -> None:
    task = GraphCountingEdgeTextLabelCountTask()
    out = task.generate(
        21002,
        params={
            "target_edge_label": "routes",
            "edge_label_support": [
                "feeds",
                "blocks",
                "joins",
                "routes",
                "checks",
                "updates",
            ],
            "target_count": 2,
        },
        max_attempts=100,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(
        out.prompt_variants["answer_and_annotation"]
    )
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [
        [248, 190, 300, 214],
        [420, 238, 472, 262],
    ]
    assert answer_and_annotation["answer"] == 2


def test_graph_counting_edge_text_label_sampling_covers_target_counts() -> None:
    task = GraphCountingEdgeTextLabelCountTask()
    answers: Counter[int] = Counter()
    directionality: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(80):
        out = task.generate(
            hash64(21010, "graph_counting_edge_text_label_count", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        answers[int(out.answer_gt.value)] += 1
        directionality[str(execution["graph_directionality"])] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        assert int(out.answer_gt.value) == int(execution["target_count"])
        assert 1 <= int(out.answer_gt.value) <= 5
        assert execution["edge_label_counts_by_value"][
            execution["target_edge_label"]
        ] == int(out.answer_gt.value)
        assert len(execution["edge_label_support"]) == 16
        assert all(3 <= len(str(label)) <= 5 for label in execution["edge_label_support"])
        node_labels = {
            str(entity["label"]).strip().lower()
            for entity in out.trace_payload["scene_ir"]["entities"]
            if entity["entity_kind"] == "graph_node"
        }
        assert not (set(str(label) for label in execution["edge_label_support"]) & node_labels)
        assert int(execution["edge_count"]) <= 12

    assert set(answers) == set(range(1, 6))
    assert set(directionality) == {"undirected"}
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}


def test_graph_counting_edge_text_label_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_edge_text_label_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_edge_text_label_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__node_link__edge_text_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=41,
    )
    final_path = build_dataset(
        config, code_hash="graph-counting-edge-text-label-count-smoke"
    )
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads(
        (final_path / "build_report.json").read_text(encoding="utf-8")
    )
    assert (
        int(
            build_report["accepted_counts_by_task"][
                "task_graph__node_link__edge_text_count"
            ]
        )
        == 4
    )

    validation = json.loads(
        (final_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["total_errors"] == 0
