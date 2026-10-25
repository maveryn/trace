"""Leakage tests of DESIGN 4: public prompts and files never carry oracle data.

1. Template invariance: every public prompt is the adapter+frame template with only the
   question line and ``W x H`` substituted.
2. No oracle tokens in any public prompt, system prompt or public parquet cell.
3. RL parquet columns are whitelisted and ``answer_gt`` is the extension-id sentinel.
4. The SFT user text is byte-identical to the eval public prompt of the same instance.
5. The trace-tool request (``build_initial_messages``, as ``--dry-run`` renders it) holds only
   the system text, the user text and the image (needs ``TRACE_TOOL_ROOT``; skipped otherwise).
6. No API path takes private data as a generation input.

Uses the tiny data root built by ``test_certificate_export.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pyarrow.parquet as pq
import pytest

from tests.test_certificate_export import TINY, accepted, available_adapters, build_tiny_root, store_rows
from trace_tasks.experimental.certificates import export, prompts, registry
from trace_tasks.experimental.certificates.types import PublicContract, TraceRecord, read_trace_records

CERT_CONDITIONS = {"cert_px": "px_source", "cert_rel1000": "rel1000"}
ID_PATTERNS = [
    re.compile(p)
    for p in (
        r"lever_weight_",
        r"cell_r\d",
        r"\bpiece_\d",
        r"_c0",
        r"\br\d+_c\d+\b",
        r"tile_id",
        r"entity_id",
        r"shortest_path",
        r"selected_row",
        r"values_by_row",
        r"witness",
        r"blake3:",
    )
]
VALUE_FIELDS = re.compile(r'"(answer|total|sum|mean|count|product|weight|distance|cost)"\s*:\s*-?\d')
EVAL = tuple(s for s in export.EVAL_SPLITS if s in TINY)


def gate_skip(reason: str) -> None:
    """Skip, except under the M1 gate (``CERT_GATE=1``), where a required test must run (review CONF-12)."""
    if os.environ.get("CERT_GATE") == "1":
        pytest.fail(f"CERT_GATE=1: required test cannot run: {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def root() -> Path:
    return build_tiny_root()


def traces(root: Path, split: str) -> Dict[str, TraceRecord]:
    out: Dict[str, TraceRecord] = {}
    for name in available_adapters():
        for record in read_trace_records(export.build_root(root) / split / name, read_image_size=False):
            out[record.instance_id] = record
    return out


def public_rows(root: Path, split: str, cond: str) -> List[Dict[str, Any]]:
    return pq.read_table(root / "public" / f"{split}.{cond}.parquet").to_pylist()


def string_cells(row: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    for key, value in row.items():
        if isinstance(value, str):
            yield key, value


def without_example(prompt: str) -> str:
    """The prompt minus the fixed example line that follows ``Example JSON:``."""
    lines = prompt.split("\n")
    drop = {i + 1 for i, line in enumerate(lines) if line.strip() == "Example JSON:"}
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def assert_no_value_leak(prompt: str, answer: Any) -> None:
    """The only instance-dependent text is the question line (test 1): it carries no field
    values, and the true answer never follows ``"answer":`` outside the fixed example."""
    question = prompt.split("\n", 1)[0]
    assert not VALUE_FIELDS.search(question), question
    assert not re.search(rf'"answer"\s*:\s*{re.escape(json.dumps(answer))}(?![0-9])', without_example(prompt))


def oracle_serializations(record: TraceRecord) -> List[str]:
    """Private values that must never appear verbatim in public text."""
    out: List[str] = []
    candidates = [
        record.annotation_gt.get("value"),
        record.execution_trace.get("values_by_row"),
        record.execution_trace.get("selected_row_indices"),
        (record.trace.get("query_spec") or {}).get("params", {}).get("selected_row_labels"),
        (record.trace.get("query_spec") or {}).get("params", {}).get("filter_cell_ids"),
        record.render_map.get("shortest_path_tile_ids"),
        record.render_map.get("relevant_weight_ids"),
        record.render_map.get("witness_entity_ids"),
        record.execution_trace.get("weight_specs"),
    ]
    for value in candidates:
        if value in (None, [], {}):
            continue
        out.append(json.dumps(value))
        out.append(json.dumps(value, separators=(",", ":")))
    return out


# 1 ====================================================================================


def test_public_prompt_is_template_plus_question_and_size(root: Path) -> None:
    for split in EVAL:
        records = traces(root, split)
        for cond, frame in CERT_CONDITIONS.items():
            rows = public_rows(root, split, cond)
            assert rows
            for row in rows:
                record = records[row["instance_id"]]
                adapter = registry.resolve(record.task, record.query_id, record.instance_version)
                width, height = row["image_sizes"][0]["width"], row["image_sizes"][0]["height"]
                question = record.train["prompt_variants"]["answer_only"].split("\n", 1)[0]
                template = prompts.prompt_template(record.task, adapter.public_contract((width, height), frame))
                expected = template.replace(prompts.QUESTION_SLOT, question).replace("{W}", str(width)).replace("{H}", str(height))
                assert row["prompt_answer"] == expected
                assert row["prompt_answer"].split("\n", 1)[0] == "<image>" + question


# 2 ====================================================================================


def test_no_oracle_tokens_in_public_prompts_and_parquets(root: Path) -> None:
    system = prompts.system_prompt()
    assert not any(p.search(system) for p in ID_PATTERNS) and not VALUE_FIELDS.search(system)
    assert not re.search(r"\d", system)
    for split in EVAL:
        records = traces(root, split)
        for cond in export.PUBLIC_CONDITIONS:
            for row in public_rows(root, split, cond):
                record = records[row["instance_id"]]
                assert json.loads(row["answer_gt"]) == {"type": "integer", "value": None}
                assert json.loads(row["annotation_gt"]) == {"type": record.annotation_gt["type"], "value": []}
                assert row["reward_contract"] == "{}"
                private = oracle_serializations(record)
                for key, text in string_cells(row):
                    if key in ("instance_id", "cert_extension_id"):
                        continue  # public identities, not oracle values
                    for pattern in ID_PATTERNS:
                        assert not pattern.search(text), (split, cond, key, pattern.pattern)
                    for value in private:
                        assert value not in text, (split, cond, key, value[:80])
                assert_no_value_leak(row["prompt_answer"], record.answer_gt["value"])


# 3 ====================================================================================


def test_rl_parquet_columns_are_whitelisted_and_answer_is_a_sentinel(root: Path) -> None:
    whitelist = set(export.RL_PARQUET_COLUMNS)
    assert whitelist == {
        "images", "image_sizes", "prompt_cert", "instance_id", "domain", "task", "scene_id",
        "query_id", "scene_variant", "cert_extension_id", "answer_gt",
    }
    for split in export.RL_SPLITS:
        if split not in TINY:
            continue
        table = pq.read_table(root / "rl" / split / export.rl_parquet_name(split))
        assert set(table.column_names) <= whitelist
        records = traces(root, split)
        for row in table.to_pylist():
            assert json.loads(row["answer_gt"]) == {"cert_extension_id": row["cert_extension_id"]}
            private = oracle_serializations(records[row["instance_id"]])
            for key, text in string_cells(row):
                if key in ("instance_id", "cert_extension_id", "answer_gt"):
                    continue
                assert not any(p.search(text) for p in ID_PATTERNS), (key, text[:80])
                assert not any(value in text for value in private), key


# 4 ====================================================================================


def test_sft_user_text_equals_public_prompt(root: Path) -> None:
    """SFT is exported for training splits only (review LEAK-01); its user text is the cert_px
    public prompt the exporter rendered for the row, whose sha256 the extension froze."""
    split = "sft"
    export.export_sft(split, root)
    store = {row["base"]["instance_id"]: row for row in export.read_store(root / "private" / f"{split}.{export.STORE_NAME}")}
    public = {}
    for name in export.ADAPTER_ORDER:
        path = export.build_root(root) / split / name / "accepted.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                public[row["instance_id"]] = row["prompts"]["cert_px"]
                assert export.prompt_sha256(row["prompts"]["cert_px"]) == store[row["instance_id"]]["public"]["prompt_sha256"]["px_source"]
    records = [json.loads(line) for line in (root / "sft" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(public)
    for record in records:
        system, user, assistant = record["messages"]
        assert user["content"] == public[record["instance_id"]]
        assert user["content"].encode("utf-8") == public[record["instance_id"]].encode("utf-8")
        assert system["content"] == prompts.system_prompt()
        assert assistant["content"] == record["target"]
        assert set(record) == {
            "instance_id", "cert_extension_id", "adapter_id", "split", "task", "query_id", "cell",
            "coord_frame", "images", "image_size", "messages", "target", "prompt_sha256", "system_sha256",
        }


# 5 ====================================================================================


def _trace_tool() -> Any:
    root = os.environ.get("TRACE_TOOL_ROOT")
    if root and str(Path(root) / "src") not in sys.path:
        sys.path.insert(0, str(Path(root) / "src"))
    try:
        return importlib.import_module("trace_tool.agent")
    except ImportError:
        gate_skip("trace_tool is not importable (set TRACE_TOOL_ROOT to the trace-tool checkout)")


def test_trace_tool_request_holds_only_system_user_text_and_image(root: Path) -> None:
    agent = _trace_tool()
    from trace_tool.conditions import build_episode_config
    from trace_tool.data import rows_from_parquet
    from trace_tool.models import detect_family
    from trace_tool.prompting import ImageURLCache, to_openai_messages, to_transcript

    spec = {
        "system_prompt_files": [str(prompts.system_prompt_path())],
        "tools": [],
        "decoding": "trace",
        "max_turns": 1,
        "max_tokens_per_turn": 8192,
        "max_total_tokens": 8192,
        "max_calls_per_turn": 2,
        "min_pixels": 262144,
        "max_pixels": 4194304,
        "oracle_crop": False,
    }
    for model, cond in (("Qwen/Qwen2.5-VL-3B-Instruct", "cert_px"), ("Qwen/Qwen3.5-4B", "cert_rel1000")):
        cfg = build_episode_config(spec, seed=42, family=detect_family(model), root=Path("/"))
        records = traces(root, "e1_dev")
        for row in rows_from_parquet(root / "public" / f"e1_dev.{cond}.parquet"):
            messages, _, shown = agent.build_initial_messages(row, cfg)
            transcript = to_transcript(messages)
            assert [m["role"] for m in transcript] == ["system", "user"]
            assert transcript[0]["content"] == prompts.system_prompt()
            parts = transcript[1]["content"]
            assert [p["type"] for p in parts] == ["image", "text"]
            assert parts[1]["text"] == row.prompt[len("<image>"):]
            payload = to_openai_messages(messages, ImageURLCache())
            assert [sorted(m) for m in payload] == [["content", "role"], ["content", "role"]]
            texts = json.dumps([m["content"] if isinstance(m["content"], str) else [p for p in m["content"] if p["type"] == "text"] for m in payload])
            for value in oracle_serializations(records[row.instance_id]):
                assert value not in texts
            assert_no_value_leak(row.prompt, records[row.instance_id].answer_gt["value"])
            assert row.answer_gt["value"] is None and row.annotation_gt["value"] == [] and row.reward_contract == {}


# 6 ====================================================================================


def test_no_api_path_takes_private_data_as_generation_input(root: Path) -> None:
    assert list(inspect.signature(prompts.render_public_prompt).parameters) == ["base", "public_contract"]
    assert list(inspect.signature(export.render_prompts).parameters) == ["base", "adapter", "image_size"]
    assert {f.name for f in dataclasses.fields(PublicContract)} == {
        "kind", "schema_id", "coord_frame", "image_size", "annotation_type", "list_key", "list_cap", "max_object_bytes",
    }
    for split in EVAL:
        for name in available_adapters():
            adapter = registry.load_adapter(name)
            for record in read_trace_records(export.build_root(root) / split / name):
                public = export.public_base(record.train)
                assert set(public) <= set(export.PUBLIC_BASE_KEYS)
                assert not {"answer_gt", "annotation_gt", "reward_contract", "trace_ref", "instance_seed", "versions"} & set(public)
                original = export.render_prompts(record.train, adapter, record.source_size())
                mutated = copy.deepcopy(dict(record.train))
                mutated["answer_gt"] = {"type": "integer", "value": 987654}
                mutated["annotation_gt"] = {"type": mutated["annotation_gt"]["type"], "value": [[1, 2, 3, 4]]}
                mutated["reward_contract"] = {"leak": True}
                mutated["trace_ref"] = {"shard_id": "x", "line_index": 7, "trace_record_hash": "blake3:leak"}
                mutated["instance_seed"] = 1
                assert export.render_prompts(mutated, adapter, record.source_size()) == original
    for split in EVAL:
        for name in available_adapters():
            for row in accepted(root, split, name):
                assert not {"labels", "oracle", "sft_target", "answer_gt", "annotation_gt"} & set(row)
    # the stored public block is exactly what the prompt hashes say (no hidden inputs)
    for split in EVAL:
        prompts_by_id = {row["instance_id"]: row["prompt_answer"] for row in public_rows(root, split, "cert_rel1000")}
        for ext in store_rows(root, split):
            assert set(ext["public"]) == {"image_size", "prompt_sha256"}
            assert export.prompt_sha256(prompts_by_id[ext["base"]["instance_id"]]) == ext["public"]["prompt_sha256"]["rel1000"]
