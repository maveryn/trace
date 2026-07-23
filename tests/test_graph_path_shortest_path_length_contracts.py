"""Contract tests for graph shortest-path-length task."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.graph.node_link.shortest_path_length import GraphPathShortestPathLengthTask
from tests.helpers import read_jsonl


def test_graph_path_shortest_path_length_deterministic() -> None:
    task = GraphPathShortestPathLengthTask()
    out_a = task.generate(19620, params={}, max_attempts=80)
    out_b = task.generate(19620, params={}, max_attempts=80)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
    assert sorted(out_a.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert out_a.prompt == out_a.prompt_variants["answer_and_annotation"]
    assert out_a.answer_gt.type == "integer"
    assert out_a.annotation_gt.type == "point_sequence"


def test_graph_path_shortest_path_length_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "graph_shortest_path_length"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_graph_shortest_path_length",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_graph__node_link__shortest_path_length",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=80,
        sampling_seed=31,
    )
    final_path = build_dataset(config, code_hash="graph-path-shortest-path-length-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "graph" for record in train_records)
    assert all(record["scene_id"] == "node_link" for record in train_records)

    build_report = json.loads((final_path / "build_report.json").read_text(encoding="utf-8"))
    assert int(build_report["accepted_counts_by_task"]["task_graph__node_link__shortest_path_length"]) == 4

    validation = json.loads((final_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["total_errors"] == 0
