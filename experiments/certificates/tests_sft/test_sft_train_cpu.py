"""FSDP training on CPU (gloo) with a tiny random Qwen2.5-VL and real SFT rows.

What this pins: torchrun with 1 or 2 ranks runs the exact GPU code path except the device and
the attention kernel (sdpa instead of FlashAttention-2); the saved dir is a plain HF checkpoint
with the base checkpoint's tensor names, dtypes and processor files; the ViT is frozen and the
merger and LLM train; a run stopped after a step resumes from the last save and ends bitwise
identical to an uninterrupted run; N=1 and N=2 (and a resume on another N) follow the same loss
curve; a finished root refuses another training config. The pytest process never imports torch
(the login node allows 8 GiB per user; two ranks need about 3 GiB).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import pytest

import sft_helpers as h

STEPS = 8  # 16 rows, global batch 4 -> 4 steps/epoch, 2 epochs; saves at 2, 4, 6, 8


def _st_tensors(path: Path):
    """{name: (dtype, shape, raw bytes)} of a safetensors file, without torch."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        out[name] = (meta["dtype"], tuple(meta["shape"]), data[a:b])
    return out


def _model_tensors(ckpt: Path):
    index = ckpt / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out = {}
    for name in files:
        out.update(_st_tensors(ckpt / name))
    return out


def _losses(root: Path):
    return [r["loss"] for r in h.read_jsonl(root / "train_log.jsonl")]


@pytest.fixture(scope="module")
def runs(tiny_model, mini_data, cpu_env, tmp_path_factory):
    base = tmp_path_factory.mktemp("sft_cpu")
    cfg = h.tiny_config(h.CONFIG_3B, base / "tiny.yaml", tiny_model, train__global_batch=4, train__epochs=2, train__per_device_batch=1,
                        train__lr=1.0e-3, train__warmup_ratio=0.25)
    common = ["train", "--config", str(cfg), "--data", str(mini_data)]
    out = {"cfg": cfg, "data": mini_data, "model": tiny_model}

    def go(nproc, root, *extra):
        res = h.torchrun(nproc, [*common, "--out", str(root), *extra], cpu_env, timeout=1200)
        assert res.returncode == 0, (res.stdout[-3000:], res.stderr[-6000:])
        return res

    def scenario(name, *launches):
        """Run launches [(nproc, extra args)] into a fresh root; a scenario killed by the login node's
        per-user memory cap (SIGKILL from other users' load) is restarted from scratch."""
        root = base / name
        for attempt in range(3):
            shutil.rmtree(root, ignore_errors=True)
            try:
                for nproc, extra in launches:
                    go(nproc, root, *extra)
                    if extra:
                        out[f"{name}_state_after_stop"] = sorted(p.name for p in (root / "state").iterdir())
                return root
            except AssertionError as exc:
                if "Signal 9" not in str(exc) or attempt == 2:
                    raise
                time.sleep(60)
        return root

    out["a"] = scenario("a", (2, ()))                                         # uninterrupted, N=2
    out["b"] = scenario("b", (2, ("--stop-after-step", "3")), (2, ()))        # killed after step 3 (last save 2), resumed on N=2
    out["c"] = scenario("c", (1, ()))                                         # N=1
    out["d"] = scenario("d", (2, ("--stop-after-step", "3")), (1, ()))        # resumed on another world size
    return out


def test_uninterrupted_run_writes_every_half_epoch_save(runs, tiny_model):
    a = runs["a"]
    done = json.loads((a / "train_done.json").read_text())
    assert done["plan"]["save_steps"] == [2, 4, 6, 8] and done["plan"]["total_steps"] == STEPS
    assert sorted(done["checkpoints"]) == ["2", "4", "6", "8"]
    assert not (a / "state").exists()  # the resumable state is dropped once training is done
    log = h.read_jsonl(a / "train_log.jsonl")
    assert [r["step"] for r in log] == list(range(1, STEPS + 1))
    assert all(r["rows"] == 4 for r in log) and [r["lr"] for r in log[:3]] == pytest.approx([0.0, 5e-4, 1e-3])  # warmup ceil(0.25 * 8) = 2
    # every row is seen once per epoch: label tokens of an epoch = sum over rows of (target + <|im_end|>)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tiny_model / "tokenizer.json"))
    data_rows = h.read_jsonl(runs["data"] / "sft" / "sft.jsonl")
    assert len(data_rows) == 16
    per_epoch = sum(len(tok.encode(r["target"], add_special_tokens=False).ids) + 1 for r in data_rows)
    assert sum(r["label_tokens"] for r in log[:4]) == per_epoch == sum(r["label_tokens"] for r in log[4:])
    assert log[-1]["loss"] < log[0]["loss"]  # it learns (lr 1e-3 on a tiny model)
    launches = h.read_jsonl(a / "runs.jsonl")
    assert len(launches) == 1 and launches[0]["world_size"] == 2 and launches[0]["layout"]["grad_accum"] == 2


def test_saved_dir_is_a_plain_hf_checkpoint(runs, tiny_model, cpu_env):
    ck = runs["a"] / "ckpt" / "step_000008"
    base = _model_tensors(tiny_model)
    mine = _model_tensors(ck)
    assert set(mine) == set(base)                                   # the base checkpoint's own tensor names
    assert {k: v[0] for k, v in mine.items()} == {k: v[0] for k, v in base.items()}  # and dtypes (bf16)
    assert {k: v[1] for k, v in mine.items()} == {k: v[1] for k, v in base.items()}
    frozen = [k for k in base if k.startswith(("visual.blocks.", "visual.patch_embed."))]
    trained = [k for k in base if k.startswith(("visual.merger.", "model.layers.", "model.embed_tokens", "model.norm"))]
    assert frozen and trained
    assert all(mine[k][2] == base[k][2] for k in frozen), "the ViT must stay frozen"
    assert all(mine[k][2] != base[k][2] for k in trained), "merger and LLM must train"
    for name in ("chat_template.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "config.json", "generation_config.json"):
        assert (ck / name).read_bytes() == (tiny_model / name).read_bytes(), name
        assert not (ck / name).is_symlink()
    info = json.loads((ck / "sft_checkpoint.json").read_text())
    assert info["step"] == 8 and info["epoch"] == 2.0 and info["world_size"] == 2
    for name, meta in info["files"].items():
        assert hashlib.sha256((ck / name).read_bytes()).hexdigest() == meta["sha256"], name
    # transformers loads it with no missing / unexpected keys, and so does the processor
    code = (
        "import sys, torch; from transformers import AutoModelForImageTextToText, AutoProcessor\n"
        "m, info = AutoModelForImageTextToText.from_pretrained(sys.argv[1], output_loading_info=True, torch_dtype=torch.bfloat16)\n"
        "assert not any(info[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')), info\n"
        "p = AutoProcessor.from_pretrained(sys.argv[1], use_fast=True)\n"
        "assert type(p).__name__ == 'Qwen2_5_VLProcessor'\n"
        "print('LOAD_OK', type(m).__name__)\n"
    )
    res = h.run([sys.executable, "-c", code, str(ck)], env=cpu_env, timeout=600)
    assert res.returncode == 0 and "LOAD_OK Qwen2_5_VLForConditionalGeneration" in res.stdout, res.stderr[-3000:]


def test_resume_is_bitwise_identical(runs):
    assert runs["b_state_after_stop"] == ["LATEST", "step_000002"]   # only the last save's state is kept
    a, b = runs["a"], runs["b"]
    launches = h.read_jsonl(b / "runs.jsonl")
    assert [x["resume_step"] for x in launches] == [0, 2]
    assert _losses(a) == _losses(b)
    for step in ("step_000002", "step_000004", "step_000006", "step_000008"):
        assert (a / "ckpt" / step / "model.safetensors").read_bytes() == (b / "ckpt" / step / "model.safetensors").read_bytes(), step


def test_world_size_does_not_change_the_math(runs):
    a, c, d = _losses(runs["a"]), _losses(runs["c"]), _losses(runs["d"])
    assert len(a) == len(c) == len(d) == STEPS
    assert c == pytest.approx(a, rel=2e-4)
    assert d == pytest.approx(a, rel=2e-4)
    assert [x["world_size"] for x in h.read_jsonl(runs["d"] / "runs.jsonl")] == [2, 1]


def test_a_finished_root_refuses_another_config(runs, cpu_env, tmp_path):
    other = h.tiny_config(runs["cfg"], tmp_path / "other.yaml", runs["model"], train__lr=2.0e-3)
    res = h.run([sys.executable, str(h.SFT_PY), "train", "--config", str(other), "--data", str(runs["data"]), "--out", str(runs["a"])], env=cpu_env, timeout=600)
    assert res.returncode == 4 and "another training config" in res.stderr
    same = h.run([sys.executable, str(h.SFT_PY), "train", "--config", str(runs["cfg"]), "--data", str(runs["data"]), "--out", str(runs["a"])], env=cpu_env, timeout=600)
    assert same.returncode == 0 and "already finished" in same.stdout
