"""The stop-rule generation path without a GPU: ``sft.py gen`` with a stand-in ``vllm`` module
(records what the engine would receive and answers each row with the oracle SFT target), then
``merge-gen`` and the real certificate scorer (trace_cert env). Pins: the engine gets EasyR1's
rollout inputs (``raw_prompt_ids`` of RLHFDataset and the process_image'd picture), greedy 2048-token
sampling with the image token suppressed, TP=1; shards partition the rows; the merged run is scored
as condition cert_px on split e1_dev with the Qwen2.5-VL family."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import sft_helpers as h

CALLS = []


class _Completion:
    def __init__(self, text, ids, reason):
        self.text, self.token_ids, self.finish_reason = text, ids, reason


class _Output:
    def __init__(self, completion):
        self.outputs = [completion]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    answers = {}

    def __init__(self, **kwargs):
        CALLS.append(("init", kwargs))

    def generate(self, inputs, sampling, use_tqdm=True):
        CALLS.append(("generate", inputs, sampling.kwargs))
        out = []
        for inp in inputs:
            text = FakeLLM.answers[tuple(inp["prompt_token_ids"])]
            out.append(_Output(_Completion(text, list(range(len(text) // 4)), "stop")))
        return out


@pytest.fixture()
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM, mod.SamplingParams = FakeLLM, FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    CALLS.clear()
    return mod


@pytest.fixture()
def fake_ckpt(tiny_model, tmp_path):
    ck = tmp_path / "ckpt" / "step_000002"
    shutil.copytree(tiny_model, ck)
    files = {p.name: {"sha256": "x" + p.name, "bytes": 1} for p in ck.iterdir()}
    (ck / "sft_checkpoint.json").write_text(json.dumps({"step": 2, "epoch": 0.5, "files": files}))
    return ck


def test_gen_merge_and_score(sft, cfg3b, data_root, dev_store, encoder, easyr1, fake_vllm, fake_ckpt, tmp_path):
    from trace_tasks.experimental.certificates.export import make_sft_target

    rows = sft.dev_rows(data_root / cfg3b["data"]["dev_parquet"], 2)
    assert [r["task"] for r in rows].count(rows[0]["task"]) == 2 and len(rows) == 8
    for r in rows:
        FakeLLM.answers[tuple(encoder.prompt_token_ids(r["user"]))] = make_sft_target({"instance_id": r["instance_id"]}, dev_store[r["instance_id"]])
    run = tmp_path / "dev" / "step_000002"
    for shard in range(2):
        out = run / "shards" / f"shard_{shard:02d}of02.jsonl"
        args = ["gen", "--config", str(h.CONFIG_3B), "--data", str(data_root), "--ckpt", str(fake_ckpt), "--shard", str(shard),
                "--num-shards", "2", "--rows-per-task", "2", "--out", str(out)]
        assert sft.main(args) == 0
        assert sft.main(args) == 0  # complete shard: skipped, no second engine
    inits = [c for c in CALLS if c[0] == "init"]
    gens = [c for c in CALLS if c[0] == "generate"]
    assert len(inits) == len(gens) == 2
    init = inits[0][1]
    assert init["tensor_parallel_size"] == 1 and init["dtype"] == "bfloat16" and init["limit_mm_per_prompt"] == {"image": 1}
    assert init["model"] == str(fake_ckpt.resolve()) and init["max_model_len"] == 6144
    sp = gens[0][2]
    assert sp["temperature"] == 0.0 and sp["max_tokens"] == 2048 and sp["n"] == 1 and sp["skip_special_tokens"] is True
    assert sp["logit_bias"] == {encoder.image_token_id: -100.0}
    # engine inputs == EasyR1 rollout inputs of the same rows (raw_prompt_ids + processed image)
    seen = [inp for g in gens for inp in g[1]]
    assert len(seen) == 8 and sorted(map(tuple, (i["prompt_token_ids"] for i in seen))) == sorted(FakeLLM.answers)
    ds = easyr1["dataset"]
    for r, inp in zip(rows[0::2] + rows[1::2], seen):
        assert inp["prompt_token_ids"] == encoder.prompt_token_ids(r["user"])
        img = inp["multi_modal_data"]["image"][0]
        ref = ds.process_image({"bytes": r["image"]}, 262144, 4194304)
        assert img.size == ref.size and img.mode == "RGB" and img.tobytes() == ref.tobytes()

    assert sft.main(["merge-gen", "--config", str(h.CONFIG_3B), "--data", str(data_root), "--ckpt", str(fake_ckpt), "--run", str(run),
                     "--num-shards", "2", "--rows-per-task", "2", "--label", "sft-test"]) == 0
    recs = h.read_jsonl(run / "responses.jsonl")
    assert [r["row_index"] for r in recs] == sorted(r["row_index"] for r in rows)
    assert all(r["model"] == "sft-test@step_000002" and r["stop_reason"] == "answer" for r in recs)
    rc = json.loads((run / "run_config.json").read_text())
    assert rc["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct" and rc["checkpoint"]["step"] == 2
    assert rc["dataset"]["sha256"] == sft.file_sha256(data_root / "public" / "e1_dev.cert_px.parquet")

    py = os.environ.get("TRACE_CERT_PYTHON")
    if not py:
        pytest.skip("TRACE_CERT_PYTHON not set: scorer step skipped")
    res = subprocess.run([py, "-m", "trace_tasks.experimental.certificates", "score", "--responses", str(run / "responses.jsonl"),
                          "--store", str(data_root / "private" / "e1_dev.extensions_v1.jsonl.zst"), "--receipt", str(data_root / "receipts" / "e1_dev.json"),
                          "--out", str(run / "cert"), "--allow-settings-mismatch"], cwd=str(h.TRACE_ROOT), capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    summary = json.loads((run / "cert" / "summary.json").read_text())
    agg = summary["all"][0]
    assert agg["n"] == 8 and agg["A"] == agg["M"] == agg["F"] == 1.0
    assert summary["run"]["cond"] == "cert_px" and summary["run"]["split"] == "e1_dev" and summary["run"]["family"] == "qwen2.5-vl"
    checks = summary["run"].get("checks")
    if checks is not None:  # scorer with integrity checks: only the (recorded) greedy decoding differs from DESIGN 7.1
        assert checks["dataset_frozen"] and checks["prompt_check"] and checks["prompts_checked"] == 8
        assert [p.split(" ", 1)[0] for p in checks["settings_problems"]] == ["decoding"]
    assert rc["settings"]["decoding"] == "greedy" and rc["settings"]["tools"] == [] and rc["settings"]["oracle_crop"] is False


def test_merge_refuses_incomplete_or_foreign_shards(sft, data_root, fake_vllm, fake_ckpt, tmp_path, encoder, dev_store):
    from trace_tasks.experimental.certificates.export import make_sft_target

    rows = sft.dev_rows(data_root / "public" / "e1_dev.cert_px.parquet", 1, with_images=False)
    for r in rows:
        FakeLLM.answers[tuple(encoder.prompt_token_ids(r["user"]))] = make_sft_target({"instance_id": r["instance_id"]}, dev_store[r["instance_id"]])
    run = tmp_path / "dev" / "x"
    base = ["--config", str(h.CONFIG_3B), "--data", str(data_root)]
    assert sft.main(["gen", *base, "--ckpt", str(fake_ckpt), "--shard", "0", "--num-shards", "2", "--rows-per-task", "1",
                     "--out", str(run / "shards" / "shard_00of02.jsonl")]) == 0
    merge = ["merge-gen", *base, "--ckpt", str(fake_ckpt), "--run", str(run), "--num-shards", "2", "--rows-per-task", "1", "--label", "t"]
    assert sft.main(merge) == 4                                   # shard 1 missing
    other = tmp_path / "other_ckpt"
    shutil.copytree(fake_ckpt, other)
    assert sft.main(["gen", *base, "--ckpt", str(other), "--shard", "1", "--num-shards", "2", "--rows-per-task", "1",
                     "--out", str(run / "shards" / "shard_01of02.jsonl")]) == 0
    assert sft.main(merge) == 4                                   # shard 1 is from another checkpoint
    assert not (run / "responses.jsonl").exists()
