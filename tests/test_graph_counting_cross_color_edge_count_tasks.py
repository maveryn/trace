"""Behavior tests for graph cross-color edge counting task."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.cross_color_edge_count import GraphCountingCrossColorEdgeCountTask
from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.shared.named_colors import named_color
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def test_graph_counting_cross_color_edge_count_contract_matches_trace() -> None:
    task = GraphCountingCrossColorEdgeCountTask()
    out = task.generate(
        20901,
        params={
            "graph_directionality": "directed",
            "source_color_name": "green",
            "target_color_name": "red",
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
    edge_entities = [
        entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_edge"
    ]
    source_label = format_named_color_with_hex("green", named_color("green"))
    target_label = format_named_color_with_hex("red", named_color("red"))

    assert "task_graph__node_link__cross_color_edge_count" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "directed_cross_color_edge_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "segment_set"
    assert int(out.answer_gt.value) == 3
    assert trace["scene_ir"]["scene_kind"] == "graph_cross_color_edge_counting"
    assert execution["query_id"] == "directed_cross_color_edge_count"
    assert execution["graph_directionality"] == "directed"
    assert execution["source_color_name"] == "green"
    assert execution["target_color_name"] == "red"
    assert execution["source_color_label"] == source_label
    assert execution["target_color_label"] == target_label
    assert execution["layout_variant_requested"] == "shell"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"
    assert source_label in str(out.prompt)
    assert target_label in str(out.prompt)

    matching_edges = [tuple(str(value) for value in edge) for edge in execution["matching_edges"]]
    node_colors = {
        str(label): str(color_name)
        for label, color_name in execution["node_color_names_by_label"].items()
    }
    assert int(out.answer_gt.value) == len(matching_edges) == len(out.annotation_gt.value)
    assert trace["witness_symbolic"]["edges"] == [list(edge) for edge in matching_edges]
    assert trace["witness_symbolic"]["source_color_name"] == "green"
    assert trace["witness_symbolic"]["target_color_name"] == "red"
    assert all(node_colors[left] == "green" and node_colors[right] == "red" for left, right in matching_edges)
    assert sum(1 for edge in edge_entities if bool(edge["is_target_cross_color_edge"])) == 3
    assert len(edge_entities) == int(execution["edge_count"])
    assert any(bool(edge["directed"]) for edge in edge_entities)
    assert trace["projected_annotation"]["type"] == "segment_set"
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    assert trace["render_spec"]["style"]["semantic_node_color_names_by_label"] == node_colors
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_counting_cross_color_edge_count_undirected_and_zero_answer() -> None:
    task = GraphCountingCrossColorEdgeCountTask()
    out = task.generate(
        20902,
        params={
            "graph_directionality": "undirected",
            "source_color_name": "cyan",
            "target_color_name": "orange",
            "target_count": 0,
            "node_count": 7,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    execution = out.trace_payload["execution_trace"]
    node_colors = {
        str(label): str(color_name)
        for label, color_name in execution["node_color_names_by_label"].items()
    }

    assert out.query_id == "cross_color_edge_count"
    assert int(out.answer_gt.value) == 0
    assert out.annotation_gt.value == []
    assert execution["target_count"] == 0
    assert execution["matching_edges"] == []
    assert format_named_color_with_hex("cyan", named_color("cyan")) in str(out.prompt)
    assert format_named_color_with_hex("orange", named_color("orange")) in str(out.prompt)
    assert "cyan" in set(node_colors.values())
    assert "orange" in set(node_colors.values())


def test_graph_counting_cross_color_prompt_examples_match_contract() -> None:
    task = GraphCountingCrossColorEdgeCountTask()
    out = task.generate(
        20903,
        params={"source_color_name": "orange", "target_color_name": "purple", "target_count": 2},
        max_attempts=100,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[[180, 220], [310, 180]], [[180, 220], [430, 260]]]
    assert answer_and_annotation["answer"] == 2


def test_graph_counting_cross_color_balanced_sampling_includes_zero() -> None:
    task = GraphCountingCrossColorEdgeCountTask()
    answers: Counter[int] = Counter()
    directionality: Counter[str] = Counter()
    query_ids: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(70):
        out = task.generate(
            hash64(20910, "graph_counting_cross_color_edge_count", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        answers[int(out.answer_gt.value)] += 1
        directionality[str(execution["graph_directionality"])] += 1
        query_ids[str(execution["query_id"])] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        assert int(out.answer_gt.value) == int(execution["target_count"])
        assert 0 <= int(out.answer_gt.value) <= 6
        assert execution["source_color_name"] != execution["target_color_name"]

    assert set(answers) == set(range(0, 7))
    assert all(count > 0 for count in answers.values())
    assert set(directionality) == {"undirected", "directed"}
    assert set(query_ids) == {"cross_color_edge_count", "directed_cross_color_edge_count"}
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}


def test_graph_counting_cross_color_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_cross_color_edge_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_cross_color_edge_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__node_link__cross_color_edge_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="graph-counting-cross-color-edge-count-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__node_link__cross_color_edge_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
