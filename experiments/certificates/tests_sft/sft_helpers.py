"""Shared helpers of the M2 SFT tests (imported by conftest.py and the test modules).

Nothing here touches a GPU. The tiny model is a randomly initialised Qwen2.5-VL with the real
tokenizer/processor files of the pinned base snapshot, so tokenization, image token expansion,
FSDP wrapping, the frozen-ViT split, checkpoint names and resume run exactly as on GPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CERT_DIR = Path(__file__).resolve().parents[1]
TRACE_ROOT = CERT_DIR.parents[1]
SFT_PY = CERT_DIR / "sft.py"
SELECT_PY = CERT_DIR / "jobs" / "sft_select.py"
SFT_SH = CERT_DIR / "jobs" / "sft.sh"
CONFIG_3B = CERT_DIR / "configs" / "sft_3b.yaml"
CONFIG_7B = CERT_DIR / "configs" / "sft_7b.yaml"


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sft():
    return load_module("cert_sft", SFT_PY)


def load_select():
    return load_module("cert_sft_select", SELECT_PY)


def build_tiny_model(base_dir: Path, out: Path, *, tied: bool = True, seed: int = 0) -> Path:
    """A 2-layer Qwen2.5-VL (hidden 32, ViT depth 2) saved like the base snapshot (bf16
    safetensors via save_pretrained) plus the base snapshot's tokenizer/processor files."""
    import torch
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    base_cfg = json.loads((base_dir / "config.json").read_text())
    torch.manual_seed(seed)
    cfg = Qwen2_5_VLConfig(
        text_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            vocab_size=base_cfg["vocab_size"], rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
            tie_word_embeddings=tied, max_position_embeddings=32768, rope_theta=1000000.0,
        ),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32,
                           fullatt_block_indexes=[1], window_size=112, patch_size=14, spatial_merge_size=2, temporal_patch_size=2),
        tie_word_embeddings=tied,
        image_token_id=base_cfg["image_token_id"], video_token_id=base_cfg["video_token_id"],
        vision_start_token_id=base_cfg["vision_start_token_id"], vision_end_token_id=base_cfg["vision_end_token_id"],
        bos_token_id=base_cfg["bos_token_id"], eos_token_id=base_cfg["eos_token_id"],
    )
    model = Qwen2_5_VLForConditionalGeneration._from_config(cfg, attn_implementation="sdpa", torch_dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():  # larger random weights than the default init: visible updates
            p.normal_(0.0, 0.05)
    model.tie_weights()
    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    for name in ("chat_template.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "generation_config.json"):
        if (base_dir / name).exists():
            shutil.copyfile(base_dir / name, out / name)
    return out


def tiny_config(src: Path, out: Path, model_dir: Path, **overrides: Any) -> Path:
    """A copy of a real config pointing at the tiny model (``model.local_path``) with overrides
    given as ``block__key=value``."""
    import yaml

    cfg = yaml.safe_load(src.read_text())
    cfg["model"]["local_path"] = str(model_dir)
    cfg["model"]["attn_implementation"] = "sdpa"
    for key, value in overrides.items():
        block, name = key.split("__", 1)
        cfg[block][name] = value
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def mini_data_root(data_root: Path, out: Path, rows_per_adapter: int) -> Path:
    """A data root with the first ``rows_per_adapter`` SFT rows of each adapter (images linked)."""
    out.mkdir(parents=True, exist_ok=True)
    taken: Dict[str, int] = {}
    lines: List[str] = []
    with open(data_root / "sft" / "sft.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            kind = row["adapter_id"].split(".", 1)[0]
            if taken.get(kind, 0) >= rows_per_adapter:
                continue
            taken[kind] = taken.get(kind, 0) + 1
            lines.append(line if line.endswith("\n") else line + "\n")
            img = out / row["images"][0]
            img.parent.mkdir(parents=True, exist_ok=True)
            if not img.exists():
                os.symlink(data_root / row["images"][0], img)
    (out / "sft").mkdir(exist_ok=True)
    (out / "sft" / "sft.jsonl").write_text("".join(lines), encoding="utf-8")
    return out


FAKE_SMI = r'''#!/usr/bin/env python3
"""Stand-in for nvidia-smi (common.sh TRC_NVIDIA_SMI): GPUs from $FAKE_GPUS ("n:mib")."""
import os, sys
n, mib = (os.environ.get("FAKE_GPUS") or "2:81920").split(":")
gpus = [dict(index=str(i), uuid=f"GPU-fake{i:04d}-0000-0000-0000-000000000000", name="Fake A100 80GB", **{
    "memory.total": mib, "memory.used": "0", "utilization.gpu": "50"}) for i in range(int(n))]
args = sys.argv[1:]
if "-L" in args:
    for g in gpus: print(f"GPU {g['index']}: {g['name']} (UUID: {g['uuid']})")
    sys.exit(0)
q = next((a.split("=", 1) for a in args if a.startswith("--query-")), None)
if q is None: sys.exit(0)
if q[0] == "--query-compute-apps": sys.exit(0)
fields = q[1].split(",")
for g in gpus:
    row = []
    for f in fields:
        v = g.get(f, "0")
        if f == "memory.used" and os.environ.get("FAKE_GPU_USED"): v = os.environ["FAKE_GPU_USED"]
        row.append(v)
    print(", ".join(row))
'''


def write_fake_smi(path: Path) -> Path:
    path.write_text(FAKE_SMI)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


ORACLE_GEN = r'''#!/usr/bin/env python
"""Stand-in for `sft.py gen` (jobs/sft.sh TRC_SFT_GEN_CMD): writes the oracle SFT target of each
E1-dev row of the shard as the response, in the same format and meta as sft.py gen."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))  # experiments/certificates
import sft
from trace_tasks.experimental.certificates.export import make_sft_target, read_store
from trace_tasks.experimental.certificates.prompts import system_prompt
from trace_tasks.experimental.certificates.types import CertificateExtension
ap = argparse.ArgumentParser()
for a in ("--config", "--data", "--ckpt", "--out"): ap.add_argument(a, required=True)
ap.add_argument("--shard", type=int, required=True); ap.add_argument("--num-shards", type=int, required=True)
ap.add_argument("--rows-per-task", type=int)
args = ap.parse_args()
cfg = sft.load_config(args.config)
data = Path(args.data)
rows = sft.dev_rows(data / cfg["data"]["dev_parquet"], args.rows_per_task, with_images=False)[args.shard::args.num_shards]
store = {}
for p in read_store(data / cfg["data"]["dev_store"]):
    e = CertificateExtension.from_dict(p); store[e.instance_id] = e
ckpt = Path(args.ckpt).resolve()
out = Path(args.out)
lines = []
for r in rows:
    text = make_sft_target({"instance_id": r["instance_id"]}, store[r["instance_id"]])
    lines.append(json.dumps({"instance_id": r["instance_id"], "row_index": r["row_index"], "task": r["task"],
        "cert_extension_id": r["cert_extension_id"], "final_response": text, "completion_tokens": 100,
        "final_prompt_tokens": 1000, "stop_reason": "answer", "finish_reason": "stop", "seed": 42, "condition": "cert_v1",
        "transcript": [*sft.build_messages(system_prompt(), r["user"]), {"role": "assistant", "content": text}]}))
out.write_text("".join(l + "\n" for l in lines))
sft.write_json(out.with_name(out.name + ".meta.json"), sft.gen_meta(cfg, ckpt, rows, args.shard, args.num_shards))
'''


def write_oracle_gen(path: Path) -> Path:
    path.write_text(ORACLE_GEN)
    return path


def run(cmd: Sequence[str], env: Optional[Dict[str, str]] = None, timeout: int = 1800, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), env=env, timeout=timeout, cwd=str(cwd or TRACE_ROOT), capture_output=True, text=True)


def user_memory_headroom_gib() -> Optional[float]:
    """Free memory under this user's cgroup v1 memory limit (login nodes cap a user at 8 GiB),
    counting only resident anonymous memory as used; None when there is no such limit."""
    base = Path(f"/sys/fs/cgroup/memory/user.slice/user-{os.getuid()}.slice")
    try:
        stat = dict(line.split() for line in (base / "memory.stat").read_text().splitlines())
        limit = int(stat.get("hierarchical_memory_limit", 0))
        rss = int(stat.get("total_rss", 0)) + int(stat.get("total_shmem", 0))
    except (OSError, ValueError):
        return None
    if limit <= 0 or limit >= 1 << 50:
        return None
    return (limit - rss) / 2**30


def wait_for_memory(gib: float, timeout_s: int = 1800) -> None:
    """Block until the user cgroup has ``gib`` GiB of headroom (other users' load on a shared
    login node comes and goes); gives up silently after ``timeout_s``."""
    import time

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        free = user_memory_headroom_gib()
        if free is None or free >= gib:
            return
        time.sleep(15)


def torchrun(nproc: int, args: Sequence[str], env: Dict[str, str], timeout: int = 1800) -> subprocess.CompletedProcess:
    wait_for_memory(1.0 + 1.3 * nproc)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={nproc}", str(SFT_PY), *args]
    return run(cmd, env=env, timeout=timeout)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
