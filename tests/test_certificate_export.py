"""Split export, private store, SFT targets and offline evaluation (DESIGN 1.1, 2.5, 2.6, 5, 6, 9).

A tiny data root (a few rows per adapter and split, in-process) is built once per session and
shared with ``test_certificate_isolation.py``. Adapters still being written are skipped.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pyarrow.parquet as pq
import pytest

from trace_tasks.experimental.certificates import evaluate, export, registry
from trace_tasks.experimental.certificates.__main__ import build_coverage, main as cli_main
from trace_tasks.experimental.certificates.types import (
    CertificateExtension,
    CertificateVersionError,
    ExtensionMismatchError,
    QuarantineError,
    read_trace_records,
)

TINY = {"sft": 2, "rl": 2, "e1_dev": 2, "e2_test": 3, "fixtures": 2}
# Stand-in for trace-tool's prompts/trace_system_prompt.txt: the ans rows below check against
# a patched expectation (see ``ans_system``), so the real TRACE text is not needed here.
TRACE_SYSTEM_TEXT = "TRACE answer-only system prompt"
CHUNK = 4


def available_adapters() -> List[str]:
    names = []
    for name in export.ADAPTER_ORDER:
        try:
            registry.load_adapter(name)
        except NotImplementedError:
            continue
        names.append(name)
    return names


@lru_cache(maxsize=None)
def build_tiny_root() -> Path:
    """Export every split (train splits first, for the test-split dedupe) into a temp root."""
    names = available_adapters()
    if not names:
        pytest.skip("no certificate adapter is implemented yet")
    root = Path(tempfile.mkdtemp(prefix="cert_export_"))
    atexit.register(shutil.rmtree, root, True)
    for split in ("sft", "rl", "e1_dev", "e2_test", "fixtures"):
        export.export_split(split, root, adapters=names, count=TINY[split], workers=1, chunk_size=CHUNK)
    export.freeze(root)
    return root


@pytest.fixture(scope="module")
def root() -> Path:
    return build_tiny_root()


def receipt(root: Path, split: str) -> Dict[str, Any]:
    return json.loads((root / "receipts" / f"{split}.json").read_text(encoding="utf-8"))


def store_rows(root: Path, split: str) -> List[Dict[str, Any]]:
    return export.read_store(root / "private" / f"{split}.{export.STORE_NAME}")


def accepted(root: Path, split: str, name: str) -> List[Dict[str, Any]]:
    path = export.build_root(root) / split / name / "accepted.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_settings(cond: str) -> Dict[str, Any]:
    """The settings block trace-tool writes for a DESIGN 7.1 condition (the parts the scorer checks)."""
    from trace_tasks.experimental.certificates import prompts

    return {
        "system_prompt_sha256": prompts.expected_system_prompt_sha256(export.PUBLIC_CONDITIONS[cond]),
        "oracle_crop": False,
        "tools": [],
        "decoding": "trace",
        "decoding_params": {"temperature": 0.6, "top_p": 0.95, "top_k": -1},
        "max_turns": 1,
    }


def transcript(cond: str, prompt: str) -> List[Dict[str, Any]]:
    """A trace-tool transcript: system text, user parts (text around the image), assistant."""
    from trace_tasks.experimental.certificates import prompts

    system = prompts.system_prompt() if cond != "ans" else "TRACE answer-only system prompt (not checked here)"
    before, after = prompt.split("<image>", 1)
    parts = ([{"type": "text", "text": before}] if before else []) + [{"type": "image", "index": 0, "size": [1, 1]}]
    parts += [{"type": "text", "text": after}] if after else []
    return [{"role": "system", "content": system}, {"role": "user", "content": parts}]


def write_run(
    root: Path, split: str, cond: str, run_dir: Path, responses: Mapping[str, str], *, model: str,
    settings: Mapping[str, Any] | None = None, with_transcript: bool = True,
) -> Path:
    """A trace-tool run directory (run_config.json + responses.jsonl) for ``responses``."""
    parquet = root / "public" / f"{split}.{cond}.parquet"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model.split("/")[-1].lower(),
        "model_id": model,
        "condition": f"{cond}_v1",
        "seed": 42,
        "settings": dict(run_settings(cond) if settings is None else settings),
        "dataset": {"split": f"cert_{split}", "path": str(parquet), "sha256": export.file_sha256(parquet)},
        "trace_tool_version": "0.1.0",
        "git_commit": "0" * 40,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = pq.read_table(parquet, columns=["instance_id", "task", "domain", "prompt_answer"]).to_pylist()
    with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row["instance_id"] not in responses:
                continue
            record = {
                "row_index": index,
                "instance_id": row["instance_id"],
                "domain": row["domain"],
                "task": row["task"],
                "answer_type": "integer",
                "final_response": responses[row["instance_id"]],
                "stop_reason": "answer",
                "completion_tokens": 100 + index,
                "condition": config["condition"],
                "seed": 42,
                "model": config["model"],
            }
            if with_transcript:
                record["transcript"] = transcript(cond, row["prompt_answer"])
                if cond == "ans":  # trace-tool records the system text it sent; ans uses TRACE's
                    record["transcript"][0]["content"] = TRACE_SYSTEM_TEXT
            handle.write(json.dumps(record) + "\n")
    return run_dir / "responses.jsonl"


# ======================================================================================
# split builder


def test_split_counts_cells_and_strata(root: Path) -> None:
    for split, count in TINY.items():
        rec = receipt(root, split)
        assert rec["sampling_seed"] == export.SPLIT_SEEDS[split]
        assert rec["totals"]["roundtrip_failed"] == 0
        for adapter_id, counts in rec["counts"].items():
            assert counts["accepted"] == count, (split, adapter_id)
            assert all(s["accepted"] == s["requested"] for s in counts["strata"].values())
        for row in store_rows(root, split):
            base = row["base"]
            assert base["split"] == split and base["sampling_seed"] == export.SPLIT_SEEDS[split]
            assert base["code_hash"] == export.CODE_HASH and base["instance_version"] == "v0"
            if split == "e2_test":
                assert base["holdout"] in export.HOLDOUT_TYPES, base["cell"]
            elif split != "fixtures":
                assert base["holdout"] == "", base["cell"]
    if "filtered_mean" in available_adapters():
        sft = receipt(root, "sft")["counts"]["filtered_mean.table_filtered_column_mean"]["strata"]
        assert {name: s["accepted"] for name, s in sft.items()} == {"above": 1, "below": 1}
        e2 = receipt(root, "e2_test")["counts"]["filtered_mean.table_filtered_column_mean"]["strata"]
        assert {name: s["accepted"] for name, s in e2.items()} == {"render": 1, "program": 1, "both": 1}


def test_quota_plan_uses_design_counts() -> None:
    for name in available_adapters():
        adapter = registry.load_adapter(name)
        for split, count in export.SPLIT_COUNTS.items():
            plan = export.quota_plan(adapter, split)
            assert sum(s.count for s in plan) == count
        if adapter.kind == "filtered_mean":
            assert [(s.name, s.count) for s in export.quota_plan(adapter, "e2_test")] == [
                ("render", 100), ("program", 100), ("both", 100)]
            assert [(s.name, s.count) for s in export.quota_plan(adapter, "sft")] == [("above", 750), ("below", 750)]
        if adapter.kind == "selected_set":
            assert {s.task for s in export.quota_plan(adapter, "rl")} == {"task_games__chess__colored_piece_kind_count"}
            assert {s.task for s in export.quota_plan(adapter, "e2_test")} == {export.CHESS_KIND_TASK}


def test_reward_train_splits_match_the_exporter() -> None:
    from trace_tasks.experimental.certificates import reward

    assert set(reward.TRAIN_SPLITS) == set(export.TRAIN_SPLITS)  # reward does not import export


def test_holdout_rules_follow_design_table() -> None:
    assert export.holdout_type("filtered_mean", "above|zebra") == ""
    assert export.holdout_type("filtered_mean", "below|card_table") == ""
    assert export.holdout_type("filtered_mean", "above|ledger") == "render"
    assert export.holdout_type("filtered_mean", "interval|spreadsheet") == "program"
    assert export.holdout_type("filtered_mean", "interval|ledger") == "both"
    assert export.holdout_type("weighted_sum", "textured_beam") == "render"
    assert export.holdout_type("weighted_sum", "offset_fulcrum") == ""
    assert export.holdout_type("path", "square") == "render"
    assert export.holdout_type("path", "portrait") == ""
    assert export.holdout_type("selected_set", "piece_kind_count") == "program"
    assert export.holdout_type("selected_set", "colored_piece_kind_count") == ""
    for name in export.ADAPTER_ORDER:  # every adapter declares its DESIGN 6 cells
        adapter = registry.load_adapter(name)
        assert adapter.train_cells and adapter.heldout_cells
        assert {adapter.holdout_type(cell) for cell in adapter.train_cells} == {""}
    for name, cell in (("filtered_mean", "zebra"), ("weighted_sum", "unknown_variant"), ("path", "x")):
        with pytest.raises(QuarantineError) as raised:  # an undeclared cell is a data problem
            export.holdout_type(name, cell)
        assert raised.value.reason == "unknown_cell"


def test_build_matches_trace_builder(root: Path, tmp_path: Path) -> None:
    """Candidates come from TRACE's per-instance pipeline: same ids as ``build_dataset``."""
    from trace_tasks.core.builder import build_dataset
    from trace_tasks.core.config import BuildConfig, BuildTaskConfig

    name = available_adapters()[0]
    rows = accepted(root, "fixtures", name)
    task = rows[0]["task"]
    first = [row for row in rows if row["task"] == task][0]
    config = BuildConfig(
        output_root=str(tmp_path),
        dataset_name="cert_builder_pin",
        instance_version="v0",
        image_format="png",
        tasks=[BuildTaskConfig(task_id=task, count=first["seed_index"] + 1)],
        sampling_seed=export.SPLIT_SEEDS["fixtures"],
        workers=1,
    )
    built = read_trace_records(build_dataset(config, code_hash=export.CODE_HASH), read_image_size=False)
    ours = {r.instance_id: r for r in read_trace_records(export.build_root(root) / "fixtures" / name, read_image_size=False)}
    match = [r for r in built if r.instance_id == first["instance_id"]]
    assert len(match) == 1
    assert match[0].trace_record_hash == ours[first["instance_id"]].trace_record_hash
    assert match[0].instance_seed == first["instance_seed"]


def test_export_reload_reproduces_extension_ids_and_prompt_shas(root: Path) -> None:
    for split in TINY:
        store = {row["certificate_extension_id"]: row for row in store_rows(root, split)}
        for name in available_adapters():
            adapter = registry.load_adapter(name)
            rows = {row["instance_id"]: row for row in accepted(root, split, name)}
            records = read_trace_records(export.build_root(root) / split / name)
            assert [r.instance_id for r in records] == list(rows)
            for record in records:
                row = rows[record.instance_id]
                prompts = export.render_prompts(record.train, adapter, record.source_size())
                assert prompts == row["prompts"]
                rebuilt = export.build_extension(
                    record,
                    None,
                    adapter,
                    split=split,
                    sampling_seed=export.SPLIT_SEEDS[split],
                    seed_index=row["seed_index"],
                    cell=row["cell"],
                    holdout=row["holdout"],
                    prompt_sha256=export.prompt_shas(prompts),
                )
                stored = store[row["cert_extension_id"]]
                assert rebuilt.certificate_extension_id == stored["certificate_extension_id"]
                assert rebuilt.expected_id() == stored["certificate_extension_id"]
                for block in ("base", "adapter", "labels", "public", "oracle", "sft_target"):
                    assert getattr(rebuilt, block) == stored[block], block
                assert rebuilt.provenance["payload_sha256"] == stored["provenance"]["payload_sha256"]


def test_export_is_deterministic_across_chunking_and_reruns(root: Path, tmp_path: Path) -> None:
    names = available_adapters()
    first = receipt(root, "fixtures")["outputs"]
    before = receipt(root, "fixtures")
    rerun = export.export_split("fixtures", root, adapters=names, count=TINY["fixtures"], workers=1, chunk_size=CHUNK)
    assert {k: v["sha256"] for k, v in rerun["outputs"].items()} == {k: v["sha256"] for k, v in first.items()}
    assert rerun["supersedes"] == before["supersedes"]  # an identical rerun replaces nothing
    other = export.export_split("fixtures", tmp_path, adapters=names, count=TINY["fixtures"], workers=2, chunk_size=3)
    assert other["workers"] == 2  # a spawn pool: accepted rows do not depend on workers or chunking
    assert {k: v["sha256"] for k, v in other["outputs"].items()} == {k: v["sha256"] for k, v in first.items()}


# ======================================================================================
# SFT target


def test_round_half_up_of_boxes_and_points() -> None:
    assert export.round_annotation([[677.0, 278.5, 735.0, 330.5]], "bbox_set") == [[677, 279, 735, 331]]
    assert export.round_annotation({"a": [[0.5, 1.49, 2.5, 3.5]]}, "bbox_set_map") == {"a": [[1, 1, 3, 4]]}
    assert export.round_annotation({"p": [10.5, 20.25, 30.0, 40.75]}, "bbox_map") == {"p": [11, 20, 30, 41]}
    assert export.round_annotation([[599.5, 200.4]], "point_sequence") == [[600, 200]]


def test_sft_target_is_compact_ordered_and_scores(root: Path) -> None:
    from trace_tasks.experimental.certificates.scoring import score_certificate_response

    for row in store_rows(root, "fixtures"):
        ext = CertificateExtension.from_dict(row)
        target = export.make_sft_target({"instance_id": ext.instance_id}, ext)
        assert ", " not in target and ": " not in target and "\n" not in target
        obj = json.loads(target)
        assert list(obj) == ["annotation", "certificate", "answer"]
        assert next(iter(obj["certificate"])) == "kind" == next(iter(ext.sft_target["certificate"]))
        assert obj["answer"] == ext.labels["answer_gt"]["value"]
        assert json.dumps(obj, separators=(",", ":"), ensure_ascii=False) == target
        leaves = json.dumps(obj["annotation"])
        assert "." not in leaves, leaves  # every box coordinate is an integer
        result = score_certificate_response(target, {"instance_id": ext.instance_id}, ext, coord_frame="px_source")
        assert (result.answer, result.semantics, result.format) == (1.0, 1.0, 1.0)
        with pytest.raises(ExtensionMismatchError):
            export.make_sft_target({"instance_id": "blake3:other"}, ext)


# ======================================================================================
# quarantine


def _quarantining(name: str) -> Any:
    base = type(registry.load_adapter(name))

    class Quarantining(base):  # type: ignore[misc, valid-type]
        _allow_foreign_module = True

        def build_oracle(self, record):
            if record.instance_seed % 2 == 1:
                raise QuarantineError("test_quarantine", record.instance_id)
            return super().build_oracle(record)

    return Quarantining()


def test_quarantine_is_counted_not_zeroed(tmp_path: Path) -> None:
    name = available_adapters()[-1]
    adapter = _quarantining(name)
    rec = export.export_split("fixtures", tmp_path, adapters=[adapter], count=4, workers=1, chunk_size=CHUNK)
    counts = rec["counts"][adapter.adapter_id]
    assert counts["accepted"] == 4  # quotas are filled with valid rows, never with zeros
    assert counts["quarantined"] >= 1 and counts["quarantine_reasons"] == {"test_quarantine": counts["quarantined"]}
    quarantined = [json.loads(line) for line in (export.build_root(tmp_path) / "fixtures" / name / "quarantine.jsonl").read_text().splitlines()]
    assert len(quarantined) == counts["quarantined"]
    assert all(q["reason"] == "test_quarantine" and q["instance_seed"] % 2 == 1 for q in quarantined)
    stored = {row["base"]["instance_id"] for row in store_rows(tmp_path, "fixtures")}
    assert stored.isdisjoint(q["instance_id"] for q in quarantined)
    assert all(row["base"]["instance_seed"] % 2 == 0 for row in store_rows(tmp_path, "fixtures"))
    # the summary reports the quarantined denominator next to the scored rows
    responses = {row["base"]["instance_id"]: export.make_sft_target(None, row) for row in store_rows(tmp_path, "fixtures")}
    run = write_run(tmp_path, "fixtures", "cert_px", tmp_path / "run", responses, model="Qwen/Qwen2.5-VL-3B-Instruct")
    summary = evaluate.score_run(run, tmp_path / "private" / f"fixtures.{export.STORE_NAME}", tmp_path / "receipts" / "fixtures.json", tmp_path / "scores")
    denominators = summary["denominators"][adapter.adapter_id]
    assert denominators["quarantined"] == counts["quarantined"] and denominators["supported"] == 4
    assert denominators["scored"] == 4 and denominators["missing"] == 0


# ======================================================================================
# outputs


def test_public_parquet_follows_trace_tool_schema(root: Path) -> None:
    import io

    from PIL import Image

    trace_tool_columns = {
        "images", "image_sizes", "prompt_answer", "answer_gt", "annotation_gt", "reward_contract",
        "instance_id", "domain", "task", "scene_id", "query_id",
    }
    for split in (s for s in export.EVAL_SPLITS if s in TINY):
        ids = None
        for cond in export.PUBLIC_CONDITIONS:
            table = pq.read_table(root / "public" / f"{split}.{cond}.parquet")
            assert trace_tool_columns <= set(table.column_names) and "cert_extension_id" in table.column_names
            rows = table.to_pylist()
            ids = ids or [r["instance_id"] for r in rows]
            assert [r["instance_id"] for r in rows] == ids  # one row order for every condition
            for row in rows:
                assert row["prompt_answer"].startswith("<image>") and row["prompt_answer"].count("<image>") == 1
                with Image.open(io.BytesIO(row["images"][0]["bytes"])) as image:
                    assert [image.size[0], image.size[1]] == [row["image_sizes"][0]["width"], row["image_sizes"][0]["height"]]
                assert json.loads(row["answer_gt"]) == {"type": "integer", "value": None}
                assert json.loads(row["annotation_gt"])["value"] == [] and row["reward_contract"] == "{}"


def test_rl_parquet_is_easyr1_ready(root: Path) -> None:
    store = {row["certificate_extension_id"]: row for split in export.RL_SPLITS for row in store_rows(root, split)}
    for split in export.RL_SPLITS:
        directory = root / "rl" / split
        assert [p.name for p in directory.iterdir()] == [export.rl_parquet_name(split)]  # parquet-only dir
        rows = pq.read_table(directory / export.rl_parquet_name(split)).to_pylist()
        assert len(rows) == TINY[split] * len(available_adapters())
        cert_px = {r["instance_id"]: r for r in pq.read_table(root / "public" / f"{split}.cert_px.parquet").to_pylist()} if split in export.EVAL_SPLITS else {}
        for row in rows:
            ext = store[row["cert_extension_id"]]
            assert ext["base"]["instance_id"] == row["instance_id"]
            assert json.loads(row["answer_gt"]) == {"cert_extension_id": row["cert_extension_id"]}
            assert export.prompt_sha256(row["prompt_cert"]) == ext["public"]["prompt_sha256"]["px_source"]
            if cert_px:
                assert row["prompt_cert"] == cert_px[row["instance_id"]]["prompt_answer"]


def test_rl_parquet_files_load_as_the_train_split_like_easyr1(root: Path, tmp_path: Path) -> None:
    """Review DATA-3: EasyR1 (``verl/utils/dataset.py``) calls ``load_dataset(type, data_files=path,
    split="train")`` for a file and ``data_dir=path`` for a directory. The file path of every RL
    parquet loads as ``train``; the ``rl/e1_dev`` *directory* is inferred as a ``validation``
    split, which is why the runner and the docs pass file paths."""
    datasets = pytest.importorskip("datasets")
    from datasets.data_files import get_data_patterns

    for split in export.RL_SPLITS:
        path = root / "rl" / split / export.rl_parquet_name(split)
        loaded = datasets.load_dataset("parquet", data_files=str(path), split="train", cache_dir=str(tmp_path / "hf"))
        assert len(loaded) == TINY[split] * len(available_adapters())
    assert list(get_data_patterns(str(root / "rl" / "rl"))) == ["train"]  # the train dir also works as a dir
    assert list(get_data_patterns(str(root / "rl" / "e1_dev"))) == ["validation"]  # never pass this dir


def test_freeze_receipt_names_every_store_and_the_rl_reward_store(root: Path) -> None:
    """Review CONF-10 / DATA-2 / LEAK-01: per-split receipts never name the shared merged store;
    ``freeze`` writes it, the RL reward store (rl + e1_dev only) and one receipt."""
    export.freeze(root)  # other tests may re-export a split after the session's first freeze
    frozen = json.loads((root / "receipts" / export.FREEZE_RECEIPT).read_text())
    assert set(frozen["splits"]) == set(TINY)
    for split, entry in frozen["splits"].items():
        assert "merged_store" not in receipt(root, split)
        assert entry["receipt"]["sha256"] == export.file_sha256(root / "receipts" / f"{split}.json")
    for key in ("merged_store", "rl_reward_store"):
        assert export.file_sha256(root / frozen[key]["path"]) == frozen[key]["sha256"]
    rl_rows = export.read_store(root / frozen["rl_reward_store"]["path"])
    assert {row["base"]["split"] for row in rl_rows} == set(export.RL_SPLITS)
    assert frozen["rl_reward_store"]["splits"] == list(export.RL_SPLITS)


def test_private_store_merges_every_split(root: Path) -> None:
    merged = export.read_store(root / "private" / export.STORE_NAME)
    per_split = [row for split in export.SPLIT_SEEDS if (root / "private" / f"{split}.{export.STORE_NAME}").exists() for row in store_rows(root, split)]
    assert merged == per_split
    ids = [row["base"]["instance_id"] for row in merged]
    assert len(ids) == len(set(ids))
    for row in merged:
        CertificateExtension.from_dict(row).verify()
        assert set(row["provenance"]) >= {
            "trace_tasks_commit", "cert_commit", "env_lock_sha256", "payload_sha256", "trace_core_sha256", "core_git_dirty",
        }


def test_receipts_hash_every_output(root: Path) -> None:
    for split in TINY:
        rec = receipt(root, split)
        for rel, entry in rec["outputs"].items():
            if rel.startswith("private/") or rel.startswith("public/") or rel.startswith("rl/"):
                assert export.file_sha256(root / rel) == entry["sha256"], rel
        prov = rec["provenance"]
        assert set(prov) >= {
            "git_head", "git_dirty", "cert_code_sha256", "env_lock_sha256", "resource_sha256", "trace_core_sha256",
            "core_git_dirty",
        }
        lock = root / rec["env_lock_file"]
        assert export.file_sha256(lock) == prov["env_lock_sha256"]  # the lock text is persisted (DATA-4)
        assert "trace_tasks" not in lock.read_text().lower().replace("-", "_")
        assert rec["dedupe"]["enabled"] == rec["dedupe"]["within_split"] == (split in export.TEST_SPLITS)
        if split in export.TEST_SPLITS:
            assert set(rec["dedupe"]["train_receipts"]) == {registry.load_adapter(n).adapter_id for n in available_adapters()}
        manifest = (export.build_root(root) / split / available_adapters()[0] / "images.sha256").read_text().splitlines()
        assert len(manifest) == TINY[split]
    assert not (root / "build").exists()  # every oracle-bearing file lives under private/ (LEAK-06)
    assert (root / "private").stat().st_mode & 0o077 == 0


def test_test_splits_require_train_splits_for_dedupe(tmp_path: Path) -> None:
    with pytest.raises(export.ExportError, match="dedupe"):
        export.export_split("e1_test", tmp_path, adapters=available_adapters()[:1], count=1, workers=1)


# ======================================================================================
# evaluation


def test_score_run_oracle_and_legacy_conditions(root: Path, tmp_path: Path) -> None:
    store = {row["base"]["instance_id"]: row for row in store_rows(root, "e1_dev")}
    targets = {iid: export.make_sft_target(None, row) for iid, row in store.items()}
    ids = sorted(targets)
    responses = {iid: "Reasoning first.\n" + targets[iid] for iid in ids[1:]}  # ids[0] is missing
    run = write_run(root, "e1_dev", "cert_px", tmp_path / "cert", responses, model="Qwen/Qwen2.5-VL-3B-Instruct")
    summary = evaluate.score_run(run, root / "private" / export.STORE_NAME, root / "receipts" / "e1_dev.json", tmp_path / "cert_out")
    rows = [json.loads(line) for line in (tmp_path / "cert_out" / "cert_scores.jsonl").read_text().splitlines()]
    assert len(rows) == len(ids) - 1
    for row in rows:
        assert (row["A"], row["M"], row["F"], row["joint"]) == (1.0, 1.0, 1.0, 1.0)
        assert row["error_class"] == "correct" and row["valid_schema"] and row["within_2048"]
        assert row["G_shown"] is not None and row["M_shown"] is not None  # Qwen2.5-VL shown-frame diagnostic
        assert row["completion_tokens"] >= 100 and row["stop_reason"] == "answer"
        assert row["split"] == "e1_dev" and row["cond"] == "cert_px" and row["coord_frame"] == "px_source"
        assert set(row["overall"]) == {"A", "G", "P", "P_gated", "G_gated"}
    assert summary["run"]["family"] == "qwen2.5-vl" and summary["run"]["split"] == "e1_dev"
    total = summary["all"][0]
    assert total["A_x_M1"]["A1_M1"] == len(rows) and total["valid_schema_rate"] == 1.0
    assert sum(d["missing"] for k, d in summary["denominators"].items() if k != "unsupported_adapters") == 1
    assert {g["adapter_id"] for g in summary["by_adapter_split_cell"]} == {store[i]["adapter"]["id"] for i in responses}
    score_receipt = json.loads((tmp_path / "cert_out" / "score_receipt.json").read_text())
    assert score_receipt["store"]["match"] == "merged_store_split_rows"

    legacy = {iid: json.dumps({"answer": json.loads(t)["answer"]}) for iid, t in targets.items()}
    ans_settings = dict(run_settings("ans"), system_prompt_sha256=hashlib.sha256(TRACE_SYSTEM_TEXT.encode()).hexdigest())
    run = write_run(root, "e1_dev", "ans", tmp_path / "ans", legacy, model="Qwen/Qwen3.5-4B", settings=ans_settings)
    with pytest.raises(CertificateVersionError, match="settings"):  # not TRACE's system prompt sha
        evaluate.score_run(run, root / "private" / "e1_dev.extensions_v1.jsonl.zst", root / "receipts" / "e1_dev.json", tmp_path / "ans_bad")
    summary = evaluate.score_run(run, root / "private" / "e1_dev.extensions_v1.jsonl.zst", root / "receipts" / "e1_dev.json", tmp_path / "ans_out", allow_settings_mismatch=True)
    assert summary["run"]["checks"]["allow_settings_mismatch"] and summary["run"]["checks"]["settings_problems"]
    assert summary["run"]["cond"] == "ans" and summary["all"][0]["A"] == 1.0
    assert "M" not in summary["all"][0] and "F" not in summary["all"][0]

    ann = {iid: json.dumps({"annotation": json.loads(t)["annotation"], "answer": json.loads(t)["answer"]}) for iid, t in targets.items()}
    run = write_run(root, "e1_dev", "ann", tmp_path / "ann", ann, model="Qwen/Qwen2.5-VL-7B-Instruct")
    evaluate.score_run(run, root / "private" / export.STORE_NAME, root / "receipts" / "e1_dev.json", tmp_path / "ann_out")
    ann_rows = [json.loads(line) for line in (tmp_path / "ann_out" / "cert_scores.jsonl").read_text().splitlines()]
    assert all(r["G_frame"] == "px_shown" and r["G_px"] is not None and r["M"] is None and r["F"] is None for r in ann_rows)
    assert all(r["A"] == 1.0 for r in ann_rows)


def test_read_responses_keeps_last_non_error_and_counts_errors(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    lines = [
        {"row_index": 0, "instance_id": "a", "stop_reason": "error"},
        {"row_index": 0, "instance_id": "a", "stop_reason": "answer", "final_response": "x"},
        {"row_index": 0, "instance_id": "a", "stop_reason": "error"},
        {"row_index": 1, "instance_id": "b", "stop_reason": "error"},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines) + '{"torn', encoding="utf-8")
    records, errors, n_lines = evaluate.read_responses(path)
    assert [r["instance_id"] for r in records] == ["a"] and records[0]["final_response"] == "x"
    assert [r["instance_id"] for r in errors] == ["b"] and n_lines == 4


def test_version_store_and_extension_mismatches_raise(root: Path, tmp_path: Path) -> None:
    rows = store_rows(root, "e1_dev")
    targets = {row["base"]["instance_id"]: export.make_sft_target(None, row) for row in rows}
    run = write_run(root, "e1_dev", "cert_px", tmp_path / "run", targets, model="Qwen/Qwen2.5-VL-3B-Instruct")
    rec_path = root / "receipts" / "e1_dev.json"
    # a store whose e1_dev rows differ from the frozen ones
    tampered = [dict(row) for row in rows]
    tampered[0] = {**tampered[0], "oracle": {**tampered[0]["oracle"], "tampered": True}}
    export.write_store(tmp_path / "tampered.jsonl.zst", tampered)
    with pytest.raises(CertificateVersionError):
        evaluate.score_run(run, tmp_path / "tampered.jsonl.zst", rec_path, tmp_path / "out1")
    # an adapter version the loaded adapter does not have (id recomputed, so only the version differs)
    bumped = [dict(row) for row in rows]
    first = CertificateExtension.from_dict(bumped[0])
    first.adapter = {**first.adapter, "version": "9.9.9"}
    first.certificate_extension_id = first.expected_id()
    bumped[0] = first.to_dict()
    export.write_store(tmp_path / "bumped.jsonl.zst", bumped)
    sha = export.file_sha256(tmp_path / "bumped.jsonl.zst")
    with pytest.raises(CertificateVersionError):
        evaluate.score_run(run, tmp_path / "bumped.jsonl.zst", rec_path, tmp_path / "out2", store_sha256=sha)
    # a stored id that does not hash from the base identity
    broken = [dict(row) for row in rows]
    broken[0] = {**broken[0], "certificate_extension_id": "blake3:" + "0" * 64}
    export.write_store(tmp_path / "broken.jsonl.zst", broken)
    with pytest.raises(CertificateVersionError):
        evaluate.load_store(tmp_path / "broken.jsonl.zst")
    # a response row whose instance has no extension
    other = write_run(root, "e1_dev", "cert_px", tmp_path / "other", targets, model="Qwen/Qwen2.5-VL-3B-Instruct")
    with other.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"row_index": 99, "instance_id": "blake3:unknown", "final_response": "{}", "stop_reason": "answer"}) + "\n")
    with pytest.raises(ExtensionMismatchError):
        evaluate.score_run(other, root / "private" / export.STORE_NAME, rec_path, tmp_path / "out3")
    # a run whose dataset is not a public parquet of the receipt
    config = json.loads((tmp_path / "run" / "run_config.json").read_text())
    config["dataset"]["sha256"] = "0" * 64
    (tmp_path / "run" / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(CertificateVersionError):
        evaluate.score_run(run, root / "private" / export.STORE_NAME, rec_path, tmp_path / "out4")
    # a receipt that is not an export receipt
    (tmp_path / "not_export.json").write_text(json.dumps({"command": "score"}))
    with pytest.raises(CertificateVersionError):
        evaluate.score_run(run, root / "private" / export.STORE_NAME, tmp_path / "not_export.json", tmp_path / "out5")


def test_cli_export_sft_score_summarize_audit(root: Path, tmp_path: Path) -> None:
    assert cli_main(["export-sft", "--split", "sft", "--out", str(root)]) == 0
    sft = [json.loads(line) for line in (root / "sft" / "sft.jsonl").read_text().splitlines()]
    assert len(sft) == TINY["sft"] * len(available_adapters())
    for record in sft:
        assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
        assert record["messages"][2]["content"] == record["target"]
        assert record["images"][0].startswith("sft/images/") and (root / record["images"][0]).is_file()
    for split in ("fixtures", "e1_dev", "e2_test"):  # a test oracle never becomes a target (LEAK-01)
        with pytest.raises(SystemExit):
            cli_main(["export-sft", "--split", split, "--out", str(root)])
        with pytest.raises(export.ExportError, match="training targets only"):
            export.export_sft(split, root)
    store = {row["base"]["instance_id"]: row for row in store_rows(root, "fixtures")}
    responses = {iid: export.make_sft_target(None, row) for iid, row in store.items()}
    run = write_run(root, "fixtures", "cert_rel1000", tmp_path / "rel", responses, model="Qwen/Qwen3.5-9B")
    assert cli_main([
        "score", "--responses", str(run), "--store", str(root / "private" / export.STORE_NAME),
        "--receipt", str(root / "receipts" / "fixtures.json"), "--out", str(tmp_path / "rel_out"),
    ]) == 0
    rows = [json.loads(line) for line in (tmp_path / "rel_out" / "cert_scores.jsonl").read_text().splitlines()]
    assert rows and all(r["coord_frame"] == "rel1000" and r["family"] == "qwen3.5" and r["G_shown"] is None for r in rows)
    assert all(r["A"] == 1.0 for r in rows)  # px targets read as rel1000: answer intact, geometry off
    out = tmp_path / "pooled.json"
    assert cli_main(["summarize", "--scores", str(tmp_path / "rel_out" / "cert_scores.jsonl"), "--out", str(out)]) == 0
    pooled = json.loads(out.read_text())
    assert pooled["all"][0]["model"] == "qwen3.5-9b" and pooled["all"][0]["n"] == len(rows)
    assert (tmp_path / "pooled.receipt.json").exists()
    manifest = tmp_path / "manifest.json"
    tasks = sorted({t for n in available_adapters() for t in registry.load_adapter(n).supported}) + [
        "task_physics__lever__missing_weight_balance_value"]
    manifest.write_text(json.dumps({"tasks": tasks}))
    assert cli_main(["audit", "--out", str(tmp_path / "coverage.v1.json"), "--manifest", str(manifest), "--receipts", str(root / "receipts")]) == 0
    coverage = json.loads((tmp_path / "coverage.v1.json").read_text())
    by_task = {entry["task"]: entry for entry in coverage["tasks"]}
    assert by_task["task_physics__lever__missing_weight_balance_value"]["status"] == "unsupported"
    for task in tasks[:-1]:
        entry = by_task[task]
        assert entry["status"] == "supported" and entry["richer_than_annotation"]["value"] in ("yes", "partly")
        assert entry["required_fields"] and isinstance(entry["quarantined"], dict)
    assert len(coverage["deviations"]) == 4 and (tmp_path / "coverage.v1.receipt.json").exists()


def test_build_coverage_marks_unknown_tasks_unsupported() -> None:
    coverage = build_coverage(tasks=["task_charts__table__threshold_count"])
    entry = coverage["tasks"][0]
    assert entry["status"] == "unsupported" and "selected-set backup" in entry["reason"]
    assert coverage["counts"]["unsupported"] == 1


def test_stratum_matching_requires_task_and_holdout() -> None:
    s = export.Stratum("above", "t", 3, ("",), "above")
    assert s.matches("filtered_mean", "t", "above|zebra", "")
    assert not s.matches("filtered_mean", "t", "below|zebra", "")
    assert not s.matches("filtered_mean", "u", "above|zebra", "")
    assert not s.matches("filtered_mean", "t", "above|ledger", "render")
    assert dataclasses.asdict(s)["count"] == 3
