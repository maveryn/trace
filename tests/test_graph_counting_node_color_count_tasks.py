"""Behavior tests for graph node-color counting task."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.node_color_count import GraphCountingNodeColorCountTask
from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.shared.named_colors import named_color
from tests.helpers import read_jsonl


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def test_graph_counting_node_color_count_contract_matches_trace() -> None:
    task = GraphCountingNodeColorCountTask()
    out = task.generate(
        20701,
        params={
            "graph_directionality": "directed",
            "target_color_name": "green",
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
    target_label = format_named_color_with_hex("green", named_color("green"))

    assert "task_graph__node_link__node_color_count" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert int(out.answer_gt.value) == 3
    assert trace["scene_ir"]["scene_kind"] == "graph_node_color_counting"
    assert execution["query_id"] == "single"
    assert execution["graph_directionality"] == "directed"
    assert execution["target_color_name"] == "green"
    assert execution["target_color_label"] == target_label
    assert execution["question_format"] == "node_color_count"
    assert execution["layout_variant_requested"] == "shell"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"
    assert target_label in str(out.prompt)

    matching_labels = [str(label) for label in execution["matching_labels"]]
    node_colors = {str(key): str(value) for key, value in execution["node_color_names_by_label"].items()}
    assert int(out.answer_gt.value) == len(matching_labels) == len(out.annotation_gt.value)
    assert trace["witness_symbolic"]["labels"] == matching_labels
    assert trace["witness_symbolic"]["target_color_name"] == "green"
    assert all(node_colors[str(label)] == "green" for label in matching_labels)
    assert sum(1 for color_name in node_colors.values() if str(color_name) == "green") == 3
    assert sum(1 for node in node_entities if bool(node["is_target_color_node"])) == 3
    assert all(
        str(node["color_name"]) == "green"
        for node in node_entities
        if bool(node["is_target_color_node"])
    )
    assert len(edge_entities) == int(execution["edge_count"])
    assert any(bool(edge["directed"]) for edge in edge_entities)
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert trace["render_spec"]["style"]["semantic_node_color_names_by_label"] == node_colors
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_counting_node_color_count_min_answer_is_supported() -> None:
    task = GraphCountingNodeColorCountTask()
    out = task.generate(
        20702,
        params={
            "graph_directionality": "undirected",
            "target_color_name": "cyan",
            "target_count": 3,
            "node_count": 8,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    execution = out.trace_payload["execution_trace"]

    assert int(out.answer_gt.value) == 3
    assert len(out.annotation_gt.value) == 3
    assert execution["target_count"] == 3
    assert len(execution["matching_labels"]) == 3
    assert sum(1 for color_name in execution["node_color_names_by_label"].values() if str(color_name) == "cyan") == 3
    assert "cyan [#34C4E0]" in str(out.prompt)


def test_graph_counting_node_color_prompt_examples_match_contract() -> None:
    task = GraphCountingNodeColorCountTask()
    out = task.generate(
        20703,
        params={"target_color_name": "orange", "target_count": 3},
        max_attempts=100,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 3}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[180, 220], [310, 180], [430, 260]]
    assert answer_and_annotation["answer"] == 3


def test_graph_counting_node_color_balanced_sampling_uses_calibrated_count_range() -> None:
    task = GraphCountingNodeColorCountTask()
    answers: Counter[int] = Counter()
    directionality: Counter[str] = Counter()
    target_colors: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    node_counts: Counter[int] = Counter()
    for index in range(100):
        out = task.generate(
            hash64(20710, "graph_counting_node_color_count", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        answers[int(out.answer_gt.value)] += 1
        directionality[str(execution["graph_directionality"])] += 1
        target_colors[str(execution["target_color_name"])] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        node_counts[int(execution["node_count"])] += 1
        assert int(out.answer_gt.value) == int(execution["target_count"])
        assert 3 <= int(out.answer_gt.value) <= 7
        assert 8 <= int(execution["node_count"]) <= 12

    assert set(answers) == set(range(3, 8))
    assert all(count > 0 for count in answers.values())
    assert set(node_counts).issubset(set(range(8, 13)))
    assert set(directionality) == {"undirected", "directed"}
    assert set(target_colors) == {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "brown",
        "cyan",
        "magenta",
        "maroon",
    }
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}


def test_graph_counting_node_color_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_node_color_count"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_node_color_count",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__node_link__node_color_count",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=100,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="graph-counting-node-color-count-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__node_link__node_color_count"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
