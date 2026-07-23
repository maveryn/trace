"""Behavior tests for graph degree-filter remaining-node counting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.degree_after_removal_filter_count import (
    GraphCountingDegreeAfterRemovalFilterCountTask,
)
from tests.helpers import read_jsonl


TASK_ID = "task_graph__node_link__degree_after_removal_filter_count"
SUPPORTED_QUERY_IDS = (
    "undirected_degree_one_filter_remaining_count",
    "directed_in_degree_one_filter_remaining_count",
    "directed_out_degree_one_filter_remaining_count",
)


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def test_graph_counting_degree_after_removal_filter_contract_matches_trace() -> None:
    task = GraphCountingDegreeAfterRemovalFilterCountTask()
    out = task.generate(
        22001,
        params={
            "query_id": "directed_in_degree_one_filter_remaining_count",
            "target_count": 3,
            "label_variant": "named",
            "edge_routing_variant": "mixed_arc",
            "layout_variant": "shell",
        },
        max_attempts=300,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert TASK_ID in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "directed_in_degree_one_filter_remaining_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert int(out.answer_gt.value) == 3
    assert trace["scene_ir"]["scene_kind"] == "graph_degree_after_removal_filter_counting"
    assert execution["query_id"] == out.query_id
    assert execution["graph_directionality"] == "directed"
    assert execution["degree_mode"] == "in_degree"
    assert execution["query_degree"] == 1
    assert execution["layout_variant_used"] == "shell"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"

    matching_labels = [str(label) for label in execution["matching_labels"]]
    assert matching_labels == [str(label) for label in execution["target_labels"]]
    assert len(matching_labels) == int(out.answer_gt.value) == len(out.annotation_gt.value)
    assert trace["witness_symbolic"]["labels"] == matching_labels
    assert set(trace["projected_annotation"]["pixel_point_map"]) == set(matching_labels)
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_counting_degree_after_removal_filter_supported_queries() -> None:
    task = GraphCountingDegreeAfterRemovalFilterCountTask()
    for index, query_id in enumerate(SUPPORTED_QUERY_IDS):
        out = task.generate(
            22020 + index,
            params={"query_id": query_id, "target_count": 2},
            max_attempts=300,
        )
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == query_id
        assert execution["query_id"] == query_id
        assert int(out.answer_gt.value) == 2
        assert len(out.annotation_gt.value) == 2


def test_graph_counting_degree_after_removal_filter_prompt_examples_match_contract() -> None:
    task = GraphCountingDegreeAfterRemovalFilterCountTask()
    out = task.generate(
        22003,
        params={"query_id": "directed_out_degree_one_filter_remaining_count", "target_count": 3},
        max_attempts=300,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[180, 220], [310, 180]]
    assert answer_and_annotation["answer"] == 2


def test_graph_counting_degree_after_removal_filter_balanced_sampling_covers_support() -> None:
    task = GraphCountingDegreeAfterRemovalFilterCountTask()
    answers: Counter[int] = Counter()
    directionality: Counter[str] = Counter()
    query_ids: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(70):
        out = task.generate(
            hash64(22010, "graph_counting_degree_after_removal_filter", index),
            params={},
            max_attempts=300,
        )
        execution = out.trace_payload["execution_trace"]
        answers[int(out.answer_gt.value)] += 1
        directionality[str(execution["graph_directionality"])] += 1
        query_ids[str(execution["query_id"])] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        assert int(out.answer_gt.value) == len(execution["matching_labels"])
        assert 1 <= int(out.answer_gt.value) <= 5

    assert set(answers) == set(range(1, 6))
    assert set(directionality) == {"undirected", "directed"}
    assert set(query_ids) == set(SUPPORTED_QUERY_IDS)
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}


def test_graph_counting_degree_after_removal_filter_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_degree_after_removal_filter"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_degree_after_removal_filter",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id=TASK_ID,
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=300,
        sampling_seed=37,
    )
    final_path = build_dataset(config, code_hash="graph-degree-after-removal-filter-smoke")
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"][TASK_ID]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
