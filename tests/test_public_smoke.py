from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
from pathlib import Path

from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.builder import build_dataset
from trace_tasks.core.reward_contracts import resolve_reward_contract
from trace_tasks.core.reward_scoring import score_trace_response
from trace_tasks.core.rlvr_export import export_trace_dataset_to_rlvr
from trace_tasks.core.source_layout_policy import parse_public_task_id
from trace_tasks.tasks import create_task, list_default_task_ids


EXPECTED_DOMAIN_COUNTS = {
    "charts": 180,
    "games": 170,
    "geometry": 170,
    "graph": 60,
    "icons": 50,
    "illustrations": 60,
    "pages": 80,
    "physics": 50,
    "puzzles": 60,
    "symbolic": 60,
    "three_d": 60,
}


def _png_bytes(output) -> bytes:
    buffer = BytesIO()
    output.image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_public_registry_inventory() -> None:
    task_ids = list_default_task_ids()
    assert len(task_ids) == 1000
    assert len(set(task_ids)) == len(task_ids)
    assert task_ids == sorted(task_ids)
    counts = Counter(parse_public_task_id(task_id).domain for task_id in task_ids)
    assert counts == EXPECTED_DOMAIN_COUNTS


def test_direct_generation_is_deterministic() -> None:
    task = create_task("task_geometry__graph_paper__polygon_area_value")
    params = {"scene_variant": "triangle"}
    first = task.generate(42, params=params, max_attempts=40)
    second = task.generate(42, params=params, max_attempts=40)

    assert first.prompt == second.prompt
    assert first.answer_gt.to_dict() == second.answer_gt.to_dict()
    assert first.annotation_gt.to_dict() == second.annotation_gt.to_dict()
    assert first.trace_payload == second.trace_payload
    assert _png_bytes(first) == _png_bytes(second)


def test_build_and_jsonl_export(tmp_path: Path) -> None:
    config = BuildConfig(
        output_root=str(tmp_path / "out"),
        dataset_name="public-smoke",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_geometry__graph_paper__polygon_area_value",
                count=1,
                params={"scene_variant": "triangle"},
            )
        ],
        max_attempts_per_instance=40,
        sampling_seed=7,
        workers=1,
    )
    dataset_root = build_dataset(config, code_hash="public-smoke")
    records = [
        json.loads(line)
        for line in (dataset_root / "train_instances.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["trace_ref"]["shard_id"] == "trace_shard_0001.jsonl.zst"

    result = export_trace_dataset_to_rlvr(
        dataset_root,
        tmp_path / "trace-train.jsonl",
        prompt_variant="answer",
    )
    assert result.row_count == 1
    exported = json.loads(result.output_path.read_text(encoding="utf-8").strip())
    assert exported["task"] == records[0]["task"]
    assert exported["trace_ref"] == records[0]["trace_ref"]


def test_answer_reward_scoring() -> None:
    contract = resolve_reward_contract(
        answer_type="integer",
        annotation_type="point",
    )
    scores = score_trace_response(
        response='Reasoning. {"answer": 7}',
        answer_gt={"type": "integer", "value": 7},
        annotation_gt={"type": "point", "value": [20, 30]},
        reward_contract=contract.to_dict(),
        trace_reward_mode="answer",
    )
    assert scores["answer_reward"] == 1.0
    assert scores["task_reward_effective"] == 1.0
