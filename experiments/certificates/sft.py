#!/usr/bin/env python
"""M2 structured SFT for TRACE certificates (DESIGN 7.2): one shared warm start for the RL arms.

Subcommands (all read one config, ``configs/sft_<size>.yaml``; data paths are relative to the
frozen data root ``--data``, normally ``$TRACE_CERT_RUNS/data/v2``):

``plan``       print the step plan (steps per epoch, total steps, warmup, save steps, batch layout
               for ``--world-size``) as JSON; ``jobs/sft.sh`` and ``jobs/sft_select.py`` read it.
``check``      CPU preflight: data and receipt sha256s, base snapshot, system prompt, every SFT user
               text against the eval public prompt hash of its instance, every target against the
               library's ``make_sft_target``, exact token lengths (<= max_length), and the token-id
               identity of SFT prompts with EasyR1's ``RLHFDataset`` (needs ``--rlvr-root``).
``train``      FSDP training, launched with ``torchrun --nproc_per_node=N``; resumes from the last
               save; saves a plain HF dir every half epoch.
``gen``        greedy E1-dev generation of one checkpoint with offline vLLM (TP=1, one process per
               GPU, rows sharded ``--shard i --num-shards N``), mirroring EasyR1's rollout inputs.
``merge-gen``  merge the shards into a trace-tool-shaped run (``responses.jsonl`` + ``run_config.json``)
               that ``python -m trace_tasks.experimental.certificates score`` reads.

Training (DESIGN 7.2): full fine-tune of the LLM and the vision merger, ViT frozen; fp32 master
weights with bf16 compute (FSDP mixed precision, like EasyR1's actor) and FlashAttention-2;
assistant-only loss: the prompt (system, user text and image tokens) is masked and the target is
the compact JSON of export-sft plus ``<|im_end|>``; the loss is the token mean over the whole global
batch (32 rows), so it does not depend on N, the micro-batch or padding. The prompt tokens are
built exactly as EasyR1's ``RLHFDataset`` builds them (same message layout, ``processor
.apply_chat_template``, ``process_image`` with min/max pixels 262144/4194304, use_fast processor):
``tests_sft/`` asserts the ids are identical for the same row. Data order: torch's
``DistributedSampler(seed=train.seed)`` with ``set_epoch``; step k of an epoch consumes the k-th
block of 32 of the epoch permutation whatever N is, so a run resumed on another N continues the
same order. Saves: ``<out>/ckpt/step_XXXXXX/`` (bf16 safetensors under the base checkpoint's own
tensor names + the processor/tokenizer/chat-template/config files copied from the base snapshot)
and ``<out>/state/step_XXXXXX/`` (torch.distributed.checkpoint of the FSDP model + AdamW, the
last save only), written in that order so a crash never leaves a resumable state without its HF dir.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import hashlib
import io
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
TRACE_ROOT = HERE.parents[1]
if str(TRACE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT / "src"))

SFT_VERSION = "cert-sft-v1"
CONFIG_SCHEMA = "trace-cert-sft-config-v1"
IGNORE_INDEX = -100
IMAGE_MARKER = "<image>"
EOS_TOKEN = "<|im_end|>"
ADAPTERS = ("selected_set", "filtered_mean", "weighted_sum", "path")
# processor / tokenizer / config files copied from the base snapshot into every saved HF dir
AUX_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)
# config keys that may change between launches of one run without changing the training math
FREE_KNOBS = {("train", "per_device_batch"), ("train", "dataloader_workers"), ("train", "keep_train_states"), ("fsdp", None)}


class ConfigError(RuntimeError):
    """A configuration or data mismatch (jobs/sft.sh maps it to exit 4)."""


# ======================================================================================
# small utilities


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(Path(path), json.dumps(payload, indent=1, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def step_name(step: int) -> str:
    return f"step_{int(step):06d}"


# ======================================================================================
# config and plan


def load_config(path: str | Path) -> Dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != CONFIG_SCHEMA:
        raise ConfigError(f"{path}: not a {CONFIG_SCHEMA} config")
    for block in ("model", "data", "train", "fsdp", "eval", "select", "smoke", "resources"):
        if not isinstance(cfg.get(block), dict):
            raise ConfigError(f"{path}: missing block {block!r}")
    if cfg["train"]["scheduler"] != "cosine":
        raise ConfigError("only the cosine schedule is implemented")
    if cfg["train"]["save"] not in ("half_epoch", "final") or cfg["smoke"]["save"] not in ("half_epoch", "final"):
        raise ConfigError("save must be half_epoch or final")
    return cfg


def math_config(cfg: Mapping[str, Any], smoke: bool) -> Dict[str, Any]:
    """The part of the config that defines the training math (resume refuses any change)."""
    out: Dict[str, Any] = {"smoke_mode": bool(smoke)}
    for block in ("model", "data", "train", "smoke"):
        values = dict(cfg[block])
        for b, key in FREE_KNOBS:
            if b == block and key is not None:
                values.pop(key, None)
        out[block] = values
    return out


def math_sha256(cfg: Mapping[str, Any], smoke: bool) -> str:
    return text_sha256(json.dumps(math_config(cfg, smoke), sort_keys=True, separators=(",", ":")))


def save_steps(mode: str, steps_per_epoch: int, total: int) -> List[int]:
    """Half-epoch saves: ceil(k * steps_per_epoch / 2) for k = 1, 2, ... up to total; always total."""
    out = set()
    if mode == "half_epoch":
        k = 1
        while (k * steps_per_epoch + 1) // 2 <= total:
            out.add((k * steps_per_epoch + 1) // 2)
            k += 1
    elif mode != "final":
        raise ConfigError(f"unknown save mode {mode!r}")
    out.add(total)
    return sorted(out)


def batch_layout(global_batch: int, world_size: int, per_device_batch: int) -> Dict[str, int]:
    """global_batch = world_size x per_device_batch x grad_accum; per_device_batch is lowered to a
    divisor of global_batch / world_size when needed."""
    if world_size < 1 or global_batch % world_size:
        raise ConfigError(f"N={world_size} GPUs does not divide the global batch {global_batch}")
    per_rank = global_batch // world_size
    b = max(1, min(int(per_device_batch), per_rank))
    while per_rank % b:
        b -= 1
    return {"world_size": world_size, "per_rank": per_rank, "per_device_batch": b, "grad_accum": per_rank // b}


def make_plan(cfg: Mapping[str, Any], n_rows: int, *, smoke: bool, world_size: Optional[int] = None) -> Dict[str, Any]:
    t = cfg["train"]
    gb = int(t["global_batch"])
    if n_rows < gb:
        raise ConfigError(f"{n_rows} training rows < global batch {gb}")
    spe = math.ceil(n_rows / gb)
    if smoke:
        total = int(cfg["smoke"]["max_steps"])
        mode = cfg["smoke"]["save"]
    else:
        total = int(t["epochs"]) * spe
        mode = t["save"]
    saves = save_steps(mode, spe, total)
    plan: Dict[str, Any] = {
        "sft_version": SFT_VERSION,
        "config_id": cfg["config_id"],
        "model": {k: cfg["model"][k] for k in ("repo_id", "revision", "tag", "architecture")},
        "smoke": bool(smoke),
        "n_train": n_rows,
        "global_batch": gb,
        "steps_per_epoch": spe,
        "total_steps": total,
        "epochs": total / spe,
        "warmup_steps": math.ceil(float(t["warmup_ratio"]) * total),
        "save_steps": saves,
        "save_epochs": [round(s / spe, 6) for s in saves],
        "lr": float(t["lr"]),
        "seed": int(t["seed"]),
        "dev_rows_per_task": int(cfg["smoke"]["dev_rows_per_task"]) if smoke else None,
        "dev": {"split": cfg["data"]["dev_split"], "parquet": cfg["data"]["dev_parquet"], "store": cfg["data"]["dev_store"],
                "receipt": cfg["data"]["dev_receipt"]},
        "select": dict(cfg["select"]),
        "eval": dict(cfg["eval"]),
        "resources": dict(cfg["resources"]),
        "math_sha256": math_sha256(cfg, smoke),
    }
    if world_size is not None:
        plan["layout"] = batch_layout(gb, world_size, int(t["per_device_batch"]))
    return plan


def lr_factor(step: int, warmup: int, total: int) -> float:
    """transformers.get_cosine_schedule_with_warmup (num_cycles 0.5) at optimizer step ``step``
    (0-based: the multiplier used by the step-th update)."""
    if step < warmup:
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * progress)))


def epoch_rank_chunks(n: int, world_size: int, rank: int, seed: int, epoch: int, global_batch: int) -> List[List[int]]:
    """Per-step dataset indices of one rank in one epoch: DistributedSampler(shuffle, seed) with
    set_epoch(epoch), cut into blocks of global_batch / world_size. Step j of the epoch then
    covers exactly positions [j*G, (j+1)*G) of the epoch permutation for every world size."""
    from torch.utils.data import DistributedSampler

    sampler = DistributedSampler(range(n), num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False)
    sampler.set_epoch(epoch)
    idx = list(sampler)
    q = global_batch // world_size
    return [idx[j * q : (j + 1) * q] for j in range((len(idx) + q - 1) // q)]


def epoch_steps(n: int, world_size: int, seed: int, epoch: int, global_batch: int) -> List[List[List[int]]]:
    """[step][rank] -> dataset indices of one epoch (every rank's share of every step)."""
    per_rank = [epoch_rank_chunks(n, world_size, r, seed, epoch, global_batch) for r in range(world_size)]
    if len({len(c) for c in per_rank}) != 1:
        raise RuntimeError("ranks disagree on the number of steps per epoch")
    return [[per_rank[r][j] for r in range(world_size)] for j in range(len(per_rank[0]))]


def step_members(n: int, world_size: int, seed: int, epoch: int, global_batch: int, j: int) -> List[int]:
    """All dataset indices of step j of an epoch (every rank's share)."""
    return [i for share in epoch_steps(n, world_size, seed, epoch, global_batch)[j] for i in share]


def micro_batches(chunk: Sequence[int], per_device_batch: int, lengths: Sequence[int]) -> List[List[int]]:
    """Split one rank's step share into micro-batches of similar length (less padding). The order
    inside a step never changes the update: the loss is a sum over the step's tokens."""
    ordered = sorted(chunk, key=lambda i: (-int(lengths[i]), i))
    return [ordered[k : k + per_device_batch] for k in range(0, len(ordered), per_device_batch)]


# ======================================================================================
# data


def base_snapshot(cfg: Mapping[str, Any]) -> Path:
    """The pinned base snapshot dir (HF cache, offline). ``model.local_path`` is for tests."""
    local = cfg["model"].get("local_path")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(cfg["model"]["repo_id"], revision=cfg["model"]["revision"], local_files_only=True))
    if path.name != cfg["model"]["revision"]:
        raise ConfigError(f"snapshot {path} is not revision {cfg['model']['revision']}")
    return path


def load_sft_records(data_root: Path, cfg: Mapping[str, Any], *, smoke: bool) -> List[Dict[str, Any]]:
    """Rows of sft.jsonl (export-sft) as flat records; the smoke subset is the first
    ``smoke.train_rows / 4`` rows of each adapter in file order."""
    path = Path(data_root) / cfg["data"]["sft_jsonl"]
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row["messages"]
            if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
                raise ConfigError(f"{path}:{line_no}: messages must be system, user, assistant")
            if msgs[2]["content"] != row["target"]:
                raise ConfigError(f"{path}:{line_no}: assistant content != target")
            if len(row["images"]) != 1 or msgs[1]["content"].count(IMAGE_MARKER) != 1:
                raise ConfigError(f"{path}:{line_no}: expected exactly one image")
            records.append(
                {
                    "line": line_no,
                    "instance_id": row["instance_id"],
                    "cert_extension_id": row["cert_extension_id"],
                    "adapter_id": row["adapter_id"],
                    "task": row["task"],
                    "image": row["images"][0],
                    "image_size": [int(v) for v in row["image_size"]],
                    "system": msgs[0]["content"],
                    "user": msgs[1]["content"],
                    "target": row["target"],
                    "prompt_sha256": row.get("prompt_sha256"),
                    "system_sha256": row.get("system_sha256"),
                }
            )
    if smoke:
        k_total = int(cfg["smoke"]["train_rows"])
        if k_total % len(ADAPTERS):
            raise ConfigError("smoke.train_rows must be a multiple of 4")
        per = k_total // len(ADAPTERS)
        taken: Dict[str, int] = {}
        subset = []
        for rec in records:
            kind = rec["adapter_id"].split(".", 1)[0]
            if taken.get(kind, 0) < per:
                taken[kind] = taken.get(kind, 0) + 1
                subset.append(rec)
        if len(subset) != k_total:
            raise ConfigError(f"smoke subset has {len(subset)} rows, want {k_total} ({taken})")
        records = subset
    return records


def process_image(image: Any, min_pixels: Optional[int], max_pixels: Optional[int]):
    """EasyR1 ``verl.utils.dataset.process_image`` (same arithmetic, same order)."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    image.load()
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        f = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if min_pixels is not None and (image.width * image.height) < min_pixels:
        f = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def processed_size(width: int, height: int, min_pixels: Optional[int], max_pixels: Optional[int]) -> Tuple[int, int]:
    """(width, height) after ``process_image`` without loading pixels."""
    w, h = int(width), int(height)
    if max_pixels is not None and w * h > max_pixels:
        f = math.sqrt(max_pixels / (w * h))
        w, h = int(w * f), int(h * f)
    if min_pixels is not None and w * h < min_pixels:
        f = math.sqrt(min_pixels / (w * h))
        w, h = int(w * f), int(h * f)
    return w, h


def load_processor(path: str | Path):
    """AutoProcessor exactly as EasyR1's get_processor (use_fast=True)."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(path), use_fast=True)
    if "Processor" not in type(proc).__name__:
        raise ConfigError(f"{path}: no multimodal processor ({type(proc).__name__})")
    return proc


def build_messages(system_prompt: Optional[str], user_text: str) -> List[Dict[str, Any]]:
    """EasyR1 ``RLHFDataset._build_messages`` (image branch): the user text split at ``<image>``
    into image and text parts, the system prompt (stripped file text) as a plain string."""
    content: List[Dict[str, Any]] = []
    for i, part in enumerate(user_text.split(IMAGE_MARKER)):
        if i != 0:
            content.append({"type": "image"})
        if part:
            content.append({"type": "text", "text": part})
    msgs: List[Dict[str, Any]] = [{"role": "user", "content": content}]
    if system_prompt:
        return [{"role": "system", "content": system_prompt}, *msgs]
    return msgs


class Encoder:
    """Prompt and target tokenization, identical to EasyR1's RLHFDataset for the prompt."""

    def __init__(self, processor: Any, system_prompt: Optional[str], min_pixels: Optional[int], max_pixels: Optional[int]):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt or None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.eos_id = self.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if self.tokenizer.pad_token_id is None:
            raise ConfigError("tokenizer has no pad token")
        self.pad_id = int(self.tokenizer.pad_token_id)

    def messages(self, user_text: str) -> List[Dict[str, Any]]:
        return build_messages(self.system_prompt, user_text)

    def prompt_text(self, user_text: str) -> str:
        return self.processor.apply_chat_template(self.messages(user_text), add_generation_prompt=True, tokenize=False)

    def prompt_token_ids(self, user_text: str) -> List[int]:
        """``raw_prompt_ids`` of RLHFDataset: what EasyR1's vLLM rollout receives."""
        return self.tokenizer.encode(self.prompt_text(user_text), add_special_tokens=False)

    def encode_prompt(self, user_text: str, image: Any) -> Dict[str, Any]:
        out = self.processor(images=[image], text=[self.prompt_text(user_text)], add_special_tokens=False, return_tensors="pt")
        return {"input_ids": out["input_ids"][0], "pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}

    def target_ids(self, target: str) -> List[int]:
        return list(self.tokenizer.encode(target, add_special_tokens=False)) + [self.eos_id]

    def encode_example(self, user_text: str, image: Any, target: str) -> Dict[str, Any]:
        import torch

        prompt = self.encode_prompt(user_text, image)
        tgt = torch.tensor(self.target_ids(target), dtype=torch.long)
        input_ids = torch.cat([prompt["input_ids"].long(), tgt])
        labels = torch.cat([torch.full_like(prompt["input_ids"], IGNORE_INDEX).long(), tgt])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": prompt["pixel_values"],
            "image_grid_thw": prompt["image_grid_thw"],
            "n_prompt": int(prompt["input_ids"].numel()),
        }

    # --- exact lengths without running the image processor
    def image_tokens(self, width: int, height: int) -> int:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        ip = self.processor.image_processor
        size = getattr(ip, "size", None)

        def pick(attr: str, key: str) -> int:
            value = getattr(ip, attr, None)
            if value is None and size is not None:
                value = size[key] if isinstance(size, Mapping) else getattr(size, key, None)
            if value is None:
                raise ConfigError(f"image processor has no {attr}/{key}")
            return int(value)

        lo, hi = pick("min_pixels", "shortest_edge"), pick("max_pixels", "longest_edge")
        patch, merge = int(ip.patch_size), int(ip.merge_size)
        w, h = processed_size(width, height, self.min_pixels, self.max_pixels)
        rh, rw = smart_resize(h, w, factor=patch * merge, min_pixels=lo, max_pixels=hi)
        return (rh // patch) * (rw // patch) // (merge * merge)

    def prompt_lengths(self, users: Sequence[str], sizes: Sequence[Sequence[int]]) -> List[int]:
        texts = [self.prompt_text(u) for u in users]
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        out = []
        for row, text_ids, size in zip(users, ids, sizes):
            n_img = sum(1 for t in text_ids if t == self.image_token_id)
            if n_img != 1:
                raise ConfigError(f"prompt has {n_img} image tokens before expansion")
            out.append(len(text_ids) - 1 + self.image_tokens(int(size[0]), int(size[1])))
        return out

    def target_lengths(self, targets: Sequence[str]) -> List[int]:
        ids = self.tokenizer(list(targets), add_special_tokens=False)["input_ids"]
        return [len(x) + 1 for x in ids]


def collate(examples: Sequence[Mapping[str, Any]], pad_id: int) -> Dict[str, Any]:
    import torch

    width = max(int(ex["input_ids"].numel()) for ex in examples)
    n = len(examples)
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    labels = torch.full((n, width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    for i, ex in enumerate(examples):
        L = int(ex["input_ids"].numel())
        input_ids[i, :L] = ex["input_ids"]
        labels[i, :L] = ex["labels"]
        attention_mask[i, :L] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": torch.cat([ex["pixel_values"] for ex in examples], dim=0),
        "image_grid_thw": torch.cat([ex["image_grid_thw"] for ex in examples], dim=0),
        "index": [int(ex["index"]) for ex in examples],
    }


class SFTDataset:
    """Map-style dataset over the flat records; the image is read from the data root."""

    def __init__(self, records: Sequence[Mapping[str, Any]], data_root: Path, encoder: Encoder, max_length: int):
        self.records = list(records)
        self.data_root = Path(data_root)
        self.encoder = encoder
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        image = process_image(self.data_root / rec["image"], self.encoder.min_pixels, self.encoder.max_pixels)
        ex = self.encoder.encode_example(rec["user"], image, rec["target"])
        if int(ex["input_ids"].numel()) > self.max_length:
            raise RuntimeError(f"row {rec['instance_id']} has {ex['input_ids'].numel()} tokens > max_length {self.max_length}")
        ex["index"] = index
        return ex


# ======================================================================================
# checkpoint files


def _safetensors_header(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def base_tensor_dtypes(base_dir: Path) -> Dict[str, str]:
    """{tensor name: safetensors dtype} of the base checkpoint (its own key names)."""
    index = base_dir / "model.safetensors.index.json"
    files = sorted(set(read_json(index)["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out: Dict[str, str] = {}
    for name in files:
        for key, meta in _safetensors_header(base_dir / name).items():
            if key != "__metadata__":
                out[key] = meta["dtype"]
    return out


def checkpoint_key(fqn: str) -> str:
    """FQN of the training module -> the key the base checkpoint stores (the format transformers'
    save_pretrained writes for Qwen2.5-VL and every loader, vLLM included, reads)."""
    key = fqn[3:] if fqn.startswith("hf.") else fqn
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    if key.startswith("model.visual."):
        return "visual." + key[len("model.visual.") :]
    return key


_ST_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def write_hf_dir(state: Dict[str, Any], base_dir: Path, final_dir: Path, meta: Mapping[str, Any], *, tied: bool) -> Dict[str, Any]:
    """Write a plain HF checkpoint: the base checkpoint's tensor names and dtypes, sharded
    safetensors + index, and the base snapshot's config/processor/tokenizer files."""
    import torch
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    dtypes = base_tensor_dtypes(base_dir)
    mapped: Dict[str, Any] = {}
    lm_head = None
    for fqn in list(state):
        tensor = state.pop(fqn)
        key = checkpoint_key(fqn)
        if key not in dtypes:
            if key == "lm_head.weight" and tied:
                lm_head = tensor
                continue
            raise ConfigError(f"trained tensor {fqn} -> {key} is not in the base checkpoint")
        target = getattr(torch, _ST_DTYPES[dtypes[key]])
        mapped[key] = tensor.detach().to(device="cpu", dtype=target, copy=True).contiguous()
    if lm_head is not None and "model.embed_tokens.weight" in dtypes and "model.embed_tokens.weight" not in mapped:
        # a tied state dict may name the shared tensor only once
        mapped["model.embed_tokens.weight"] = lm_head.detach().to(device="cpu", dtype=getattr(torch, _ST_DTYPES[dtypes["model.embed_tokens.weight"]]), copy=True).contiguous()
    missing = sorted(set(dtypes) - set(mapped))
    if missing:
        raise ConfigError(f"base tensors not produced by training: {missing[:5]} (+{max(0, len(missing) - 5)})")
    if lm_head is not None and "model.embed_tokens.weight" in mapped:
        if not torch.equal(lm_head.to(mapped["model.embed_tokens.weight"].dtype).cpu(), mapped["model.embed_tokens.weight"]):
            raise ConfigError("tied lm_head and embed_tokens diverged")
    tmp = final_dir.parent / f".{final_dir.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    split = split_torch_state_dict_into_shards(mapped, filename_pattern="model{suffix}.safetensors", max_shard_size="5GB")
    for filename, names in split.filename_to_tensors.items():
        save_file({k: mapped[k] for k in names}, str(tmp / filename), metadata={"format": "pt"})
        os.chmod(tmp / filename, 0o644)  # safetensors writes 0600; evaluation jobs may run as group members
    if split.is_sharded:
        index = {"metadata": {"total_size": sum(int(t.numel()) * t.element_size() for t in mapped.values())}, "weight_map": dict(split.tensor_to_filename)}
        (tmp / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied = []
    for name in AUX_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copyfile(src, tmp / name)  # follows the HF cache symlinks: real files
            copied.append(name)
    for required in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        if required not in copied:
            raise ConfigError(f"base snapshot {base_dir} lacks {required}")
    files = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in sorted(tmp.iterdir()) if p.is_file()}
    info = {**dict(meta), "sft_version": SFT_VERSION, "written": now_iso(), "files": files, "n_tensors": len(mapped)}
    (tmp / "sft_checkpoint.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp, final_dir)
    return info


def complete_checkpoints(root: Path) -> Dict[int, Path]:
    """{step: dir} of finished HF saves under <root>/ckpt (an unfinished save is a dot-tmp dir)."""
    out: Dict[int, Path] = {}
    ckpt = Path(root) / "ckpt"
    if ckpt.is_dir():
        for p in ckpt.iterdir():
            if p.is_dir() and p.name.startswith("step_") and (p / "sft_checkpoint.json").exists():
                out[int(p.name[5:])] = p
    return out


def latest_state(root: Path) -> Optional[Tuple[int, Path]]:
    marker = Path(root) / "state" / "LATEST"
    if not marker.exists():
        return None
    name = marker.read_text(encoding="utf-8").strip()
    path = Path(root) / "state" / name
    if not (path / "trainer_state.json").exists():
        raise ConfigError(f"state/LATEST names {name}, which is incomplete")
    return int(read_json(path / "trainer_state.json")["step"]), path


# ======================================================================================
# training


def _setup_distributed():
    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and os.environ.get("TRC_SFT_DEVICE", "") != "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if "MASTER_ADDR" not in os.environ:  # plain `python sft.py train` = one process
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            os.environ.setdefault("MASTER_PORT", str(s.getsockname()[1]))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
    kwargs: Dict[str, Any] = {"backend": backend, "timeout": datetime.timedelta(minutes=60)}
    if use_cuda:
        kwargs["device_id"] = device
    dist.init_process_group(**kwargs)
    return dist, rank, world, local_rank, device


def _sft_module_class():
    import torch
    import torch.nn.functional as F

    from torch.utils.checkpoint import checkpoint

    class _SFTModule(torch.nn.Module):
        """Root module: the HF model under ``hf`` plus an assistant-only summed CE. The lm_head runs on
        label positions only (no full-vocabulary logits for the prompt or the image), in chunks of
        ``ce_chunk`` tokens whose logits are recomputed in backward (a 152k-word vocabulary costs
        about 16 bytes per token and word between logits, log-softmax and their gradients)."""

        def __init__(self, hf, ce_chunk: int = 1024):
            super().__init__()
            self.hf = hf
            self.ce_chunk = max(1, int(ce_chunk))

        def _chunk_ce(self, hidden, target):
            logits = self.hf.lm_head(hidden).float()
            loss = F.cross_entropy(logits, target, reduction="sum")
            correct = (logits.detach().argmax(dim=-1) == target).sum()
            return loss, correct

        def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels):
            out = self.hf.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                return_dict=True,
            )
            shift = labels[:, 1:]
            sel = shift != IGNORE_INDEX
            hidden = out.last_hidden_state[:, :-1, :][sel]
            target = shift[sel]
            loss_sum = hidden.new_zeros((), dtype=torch.float32)
            correct = torch.zeros((), dtype=torch.long, device=hidden.device)
            for i in range(0, int(target.numel()), self.ce_chunk):
                h, t = hidden[i : i + self.ce_chunk], target[i : i + self.ce_chunk]
                if torch.is_grad_enabled() and h.requires_grad:
                    loss, corr = checkpoint(self._chunk_ce, h, t, use_reentrant=False)
                else:
                    loss, corr = self._chunk_ce(h, t)
                loss_sum = loss_sum + loss
                correct = correct + corr
            return loss_sum, correct

    return _SFTModule


def _init_fn(model, device):
    """EasyR1 fsdp_utils.get_init_fn: materialize meta params on ``device`` (tied params once)."""
    import torch
    from collections import defaultdict

    counts: Dict[Any, int] = defaultdict(int)
    for _, p in model.named_parameters(remove_duplicate=False):
        counts[p] += 1
    shared = {p for p, c in counts.items() if c > 1}
    made: Dict[Any, Any] = {}

    def init_fn(module):
        for name, p in module.named_parameters(recurse=False):
            if p in shared:
                module._parameters[name] = made.setdefault(p, torch.nn.Parameter(torch.empty_like(p.data, device=device), requires_grad=p.requires_grad))
            else:
                module._parameters[name] = torch.nn.Parameter(torch.empty_like(p.data, device=device), requires_grad=p.requires_grad)

    return init_fn


def build_fsdp_model(cfg: Mapping[str, Any], base_dir: Path, device, rank: int, world: int, log) -> Tuple[Any, Dict[str, Any]]:
    import torch
    from accelerate import init_empty_weights
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoConfig, AutoModelForImageTextToText

    attn = os.environ.get("TRC_SFT_ATTN") or cfg["model"]["attn_implementation"]
    hf_config = AutoConfig.from_pretrained(str(base_dir))
    rank0_init = device.type == "cuda" and world > 1
    t0 = time.time()
    if not rank0_init or rank == 0:
        hf = AutoModelForImageTextToText.from_pretrained(
            str(base_dir), config=hf_config, torch_dtype=torch.float32, attn_implementation=attn, low_cpu_mem_usage=True, device_map="cpu"
        )
    else:
        with init_empty_weights():
            hf = AutoModelForImageTextToText.from_config(hf_config, torch_dtype=torch.float32, attn_implementation=attn)
    if type(hf).__name__ != cfg["model"]["architecture"]:
        raise ConfigError(f"model class {type(hf).__name__} != {cfg['model']['architecture']}")
    # from_config ignores torch_dtype/dtype for composite sub-models: Qwen2.5-VL's language model is
    # built in text_config.dtype (bf16) while from_pretrained gives fp32. With rank-0 init the bucket
    # layout of sync_module_states then differs between ranks and the first decoder-layer broadcast
    # hangs until the NCCL timeout. Cast every rank to fp32 master weights (a no-op on rank 0; meta
    # tensors keep their Parameter identity, so tying survives) and compare the layouts below.
    hf.to(torch.float32)
    hf.tie_weights()
    hf.config.use_cache = False
    tied = bool(getattr(hf.config, "tie_word_embeddings", False) or getattr(getattr(hf.config, "text_config", None), "tie_word_embeddings", False))
    if cfg["model"]["trainable"] != "llm_and_merger":
        raise ConfigError(f"unknown model.trainable {cfg['model']['trainable']!r}")
    hf.model.visual.requires_grad_(False)
    hf.model.visual.merger.requires_grad_(True)
    n_total = sum(p.numel() for p in hf.parameters())
    n_train = sum(p.numel() for p in hf.parameters() if p.requires_grad)
    if cfg["fsdp"].get("gradient_checkpointing", True):
        hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    module = _sft_module_class()(hf, ce_chunk=int(os.environ.get("TRC_SFT_CE_CHUNK") or cfg["fsdp"].get("ce_chunk_tokens", 1024)))
    layer_names = set(getattr(hf, "_no_split_modules", None) or [])
    layer_cls = {type(m) for m in hf.modules() if type(m).__name__ in layer_names}
    if not layer_cls:
        raise ConfigError(f"no FSDP layer classes found for {sorted(layer_names)}")
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
    mp = MixedPrecision(
        param_dtype=dt[cfg["fsdp"].get("param_dtype", "bf16")],
        reduce_dtype=dt[cfg["fsdp"].get("reduce_dtype", "fp32")],
        buffer_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    strategy = getattr(ShardingStrategy, cfg["fsdp"].get("sharding_strategy", "FULL_SHARD"))
    layout = [(n, tuple(p.shape), str(p.dtype), bool(p.requires_grad)) for n, p in module.named_parameters(remove_duplicate=False)]
    layout += [(n, tuple(b.shape), str(b.dtype)) for n, b in module.named_buffers()]
    bad = sorted({str(p.dtype) for p in module.parameters() if p.dtype != torch.float32})
    if bad:
        raise ConfigError(f"model tensors must be fp32 master weights before FSDP, found {bad}")
    if world > 1:
        import torch.distributed as tdist

        digest = hashlib.sha256(json.dumps(layout).encode()).hexdigest()
        gathered: List[Any] = [None] * world
        tdist.all_gather_object(gathered, digest)
        if len(set(gathered)) != 1:
            raise ConfigError(f"parameter/buffer layout differs between ranks before FSDP sync: {gathered}")
    fsdp = FSDP(
        module,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls),
        mixed_precision=mp,
        sharding_strategy=strategy,
        device_id=device if device.type == "cpu" else torch.cuda.current_device(),
        use_orig_params=True,
        sync_module_states=rank0_init,
        param_init_fn=_init_fn(module, device) if (rank0_init and rank != 0) else None,
        limit_all_gathers=True,
        forward_prefetch=False,
    )
    info = {
        "attn_implementation": attn,
        "params_total": int(n_total),
        "params_trainable": int(n_train),
        "tied_embeddings": tied,
        "layer_classes": sorted(c.__name__ for c in layer_cls),
        "sharding_strategy": strategy.name,
        "load_s": round(time.time() - t0, 1),
    }
    log(f"model: {type(hf).__name__} params {n_total / 1e9:.3f}B trainable {n_train / 1e9:.3f}B (ViT frozen, merger trainable), "
        f"attn {attn}, FSDP {strategy.name}, wrap {info['layer_classes']}, tied {tied}, loaded in {info['load_s']}s")
    return fsdp, info


def save_hf_checkpoint(fsdp, cfg, base_dir: Path, root: Path, step: int, meta: Mapping[str, Any], *, rank: int, device, tied: bool, dist) -> None:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    state = get_model_state_dict(fsdp, options=StateDictOptions(full_state_dict=True, cpu_offload=device.type == "cuda"))
    if rank == 0:
        write_hf_dir(state, base_dir, root / "ckpt" / step_name(step), meta, tied=tied)
    del state
    dist.barrier()


def save_train_state(fsdp, optimizer, root: Path, step: int, meta: Mapping[str, Any], keep: int, *, rank: int, dist) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    state_root = root / "state"
    tmp = state_root / f".{step_name(step)}.tmp"
    final = state_root / step_name(step)
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
    dist.barrier()
    msd, osd = get_state_dict(fsdp, optimizer)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp))
    dist.barrier()
    if rank == 0:
        (tmp / "trainer_state.json").write_text(json.dumps({**dict(meta), "step": step, "written": now_iso()}, indent=1, sort_keys=True) + "\n")
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        atomic_write_text(state_root / "LATEST", final.name + "\n")
        older = sorted(p for p in state_root.iterdir() if p.is_dir() and p.name.startswith("step_") and p != final)
        for p in older[: max(0, len(older) - max(0, keep - 1))]:
            shutil.rmtree(p, ignore_errors=True)
    dist.barrier()


def load_train_state(fsdp, optimizer, path: Path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    msd, osd = get_state_dict(fsdp, optimizer)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(fsdp, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])


def _versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    out = {}
    for name in ("torch", "transformers", "accelerate", "flash_attn", "vllm", "safetensors", "huggingface-hub", "pillow", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _provenance() -> Dict[str, Any]:
    try:
        from trace_tasks.experimental.certificates.export import code_provenance

        return dict(code_provenance())
    except Exception as exc:  # recorded, never fatal
        return {"error": f"{type(exc).__name__}: {exc}"}


def cmd_train(args: argparse.Namespace) -> int:
    import torch

    cfg = load_config(args.config)
    if args.per_device_batch:  # free knob: never changes the math (global batch and token-mean loss fixed)
        cfg["train"]["per_device_batch"] = int(args.per_device_batch)
    smoke = bool(args.smoke)
    data_root = Path(args.data).resolve()
    root = Path(args.out).resolve()
    dist, rank, world, local_rank, device = _setup_distributed()

    def log(msg: str) -> None:
        if rank == 0:
            print(f"{datetime.datetime.now():%F %T} [sft] {msg}", flush=True)

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        records = load_sft_records(data_root, cfg, smoke=smoke)
        plan = make_plan(cfg, len(records), smoke=smoke, world_size=world)
        if (root / "train_done.json").exists():
            done = read_json(root / "train_done.json")
            if done["plan"]["math_sha256"] != plan["math_sha256"]:
                raise ConfigError(f"{root} holds a finished run of another training config; use a new output root")
            log(f"{root}/train_done.json exists: training already finished")
            return 0
        for s, path in complete_checkpoints(root).items():
            if read_json(path / "sft_checkpoint.json").get("math_sha256") != plan["math_sha256"]:
                raise ConfigError(f"{path} was written by another training config; use a new output root")
        layout = plan["layout"]
        seed, gb, spe, total = plan["seed"], plan["global_batch"], plan["steps_per_epoch"], plan["total_steps"]
        saves = set(plan["save_steps"])
        torch.manual_seed(seed)
        base_dir = base_snapshot(cfg)
        system_prompt = records[0]["system"]
        if any(r["system"] != system_prompt for r in records):
            raise ConfigError("SFT rows carry different system prompts")
        processor = load_processor(base_dir)
        encoder = Encoder(processor, system_prompt, cfg["data"]["min_pixels"], cfg["data"]["max_pixels"])
        n_label = encoder.target_lengths([r["target"] for r in records])
        lengths = [p + t for p, t in zip(encoder.prompt_lengths([r["user"] for r in records], [r["image_size"] for r in records]), n_label)]
        if max(lengths) > int(cfg["data"]["max_length"]):
            raise ConfigError(f"longest row has {max(lengths)} tokens > max_length {cfg['data']['max_length']}")
        log(f"{len(records)} rows (smoke={smoke}), N={world} ({device.type}), per-device batch {layout['per_device_batch']} x "
            f"accum {layout['grad_accum']} x N = {gb}; {spe} steps/epoch, {total} steps, warmup {plan['warmup_steps']}, "
            f"saves {plan['save_steps']}; tokens/row max {max(lengths)} mean {sum(lengths) / len(lengths):.0f}")

        fsdp, model_info = build_fsdp_model(cfg, base_dir, device, rank, world, log)
        params = [p for p in fsdp.parameters() if p.requires_grad]
        t = cfg["train"]
        optimizer = torch.optim.AdamW(
            params, lr=float(t["lr"]), betas=tuple(t["adam_betas"]), eps=float(t["adam_eps"]),
            weight_decay=float(t["weight_decay"]), fused=device.type == "cuda",
        )
        step = 0
        resumed = latest_state(root)
        if resumed is not None:
            r_step, r_path = resumed
            st = read_json(r_path / "trainer_state.json")
            if st.get("math_sha256") != plan["math_sha256"]:
                raise ConfigError(f"{r_path} was written by another training config ({st.get('math_sha256')} != {plan['math_sha256']})")
            load_train_state(fsdp, optimizer, r_path)
            step = r_step
            log(f"resumed from {r_path} at step {step} (saved with N={st.get('world_size')})")
        done_ckpts = complete_checkpoints(root)
        missing = [s for s in plan["save_steps"] if s <= step and s not in done_ckpts]
        if missing:
            raise ConfigError(f"train state is at step {step} but HF checkpoints {missing} are missing")
        if rank == 0:
            (root / "logs").mkdir(parents=True, exist_ok=True)
            log_path = root / "train_log.jsonl"
            if log_path.exists():  # drop entries past the resume point
                kept = [ln for ln in log_path.read_text().splitlines() if ln.strip() and json.loads(ln)["step"] <= step]
                atomic_write_text(log_path, "".join(k + "\n" for k in kept))
            launch = {
                "launch": now_iso(), "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "world_size": world, "device": device.type,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if device.type == "cuda" else [],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "resume_step": step, "layout": layout,
                "model": model_info, "versions": _versions(), "provenance": _provenance(), "math_sha256": plan["math_sha256"],
                "base_dir": str(base_dir), "config": os.path.abspath(args.config),
            }
            with open(root / "runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(launch, sort_keys=True) + "\n")
            write_json(root / "plan.train.json", plan)

        dataset = SFTDataset(records, data_root, encoder, int(cfg["data"]["max_length"]))
        from torch.utils.data import DataLoader

        workers = int(os.environ.get("TRC_SFT_WORKERS", t.get("dataloader_workers", 4)))
        # no_sync() keeps unsharded fp32 grads between micro-batches (less communication). FSDP turns
        # FULL_SHARD into NO_SHARD at N=1, where no_sync + bf16 mixed precision + use_orig_params
        # fails (bf16 grad assigned to the fp32 param), so N=1 accumulates the ordinary way.
        accumulate_unsharded = bool(cfg["fsdp"].get("accumulate_unsharded_grads", False)) and world > 1
        max_norm = float(t["max_grad_norm"])
        base_lr = float(t["lr"])
        stop_after = int(args.stop_after_step) if args.stop_after_step else None
        tied = bool(model_info["tied_embeddings"])
        state_meta = {"math_sha256": plan["math_sha256"], "world_size": world, "global_batch": gb, "n_train": len(records), "seed": seed}
        fsdp.train()
        t_last = time.time()
        while step < total:
            epoch, j0 = divmod(step, spe)
            steps_of_epoch = epoch_steps(len(records), world, seed, epoch, gb)
            if len(steps_of_epoch) != spe:
                raise RuntimeError(f"epoch {epoch}: {len(steps_of_epoch)} steps, want {spe}")
            j_end = min(spe, j0 + (total - step))
            per_step = [micro_batches(steps_of_epoch[j][rank], layout["per_device_batch"], lengths) for j in range(j0, j_end)]
            flat = [mb for mbs in per_step for mb in mbs]
            loader = DataLoader(
                dataset, batch_sampler=flat, collate_fn=functools.partial(collate, pad_id=encoder.pad_id), num_workers=workers,
                pin_memory=device.type == "cuda", prefetch_factor=4 if workers > 0 else None, persistent_workers=False,
            )
            it = iter(loader)
            for jj, mbs in enumerate(per_step):
                j = j0 + jj
                members = [i for share in steps_of_epoch[j] for i in share]
                tokens_global = sum(n_label[i] for i in members)
                lr = base_lr * lr_factor(step, plan["warmup_steps"], total)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss_acc = torch.zeros(2, dtype=torch.float64, device=device)  # [loss sum, correct]
                seq_tokens = 0
                for k, mb in enumerate(mbs):
                    batch = next(it)
                    if batch.pop("index") != mb:
                        raise RuntimeError("data loader order does not match the plan")
                    seq_tokens += int(batch["attention_mask"].sum())
                    batch = {key: v.to(device, non_blocking=True) for key, v in batch.items()}
                    sync = not accumulate_unsharded or k == len(mbs) - 1
                    ctx = contextlib.nullcontext() if sync else fsdp.no_sync()
                    with ctx:
                        loss_sum, correct = fsdp(**batch)
                        # FSDP averages gradients over ranks: x world turns that into the global sum,
                        # / tokens_global into the token mean over the whole global batch
                        (loss_sum * (world / tokens_global)).backward()
                    loss_acc[0] += loss_sum.detach().double()
                    loss_acc[1] += correct.detach().double()
                grad_norm = fsdp.clip_grad_norm_(max_norm)
                grad_norm = float(grad_norm)
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"step {step}: non-finite gradient norm {grad_norm}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                dist.all_reduce(loss_acc)
                seq = torch.tensor([seq_tokens], dtype=torch.float64, device=device)
                dist.all_reduce(seq)
                step += 1
                now = time.time()
                rec = {
                    "step": step, "epoch": round(step / spe, 6), "lr": lr, "loss": float(loss_acc[0]) / tokens_global,
                    "token_acc": float(loss_acc[1]) / tokens_global, "grad_norm": grad_norm, "label_tokens": tokens_global,
                    "seq_tokens": int(seq.item()), "rows": len(members), "time_s": round(now - t_last, 3),
                    "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None,
                }
                t_last = now
                if rank == 0:
                    with open(root / "train_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, sort_keys=True) + "\n")
                    log(f"step {step}/{total} ep {rec['epoch']:.3f} loss {rec['loss']:.4f} acc {rec['token_acc']:.3f} "
                        f"gnorm {grad_norm:.3f} lr {lr:.3e} tok {rec['seq_tokens']} {rec['time_s']:.1f}s mem {rec['mem_gb']}G")
                if step in saves:
                    t_save = time.time()
                    meta = {
                        "step": step, "epoch": round(step / spe, 6), "base": dict(plan["model"]), "base_dir": str(base_dir),
                        "math_sha256": plan["math_sha256"], "config_id": cfg["config_id"], "smoke": smoke,
                        "n_train": len(records), "global_batch": gb, "world_size": world,
                    }
                    save_hf_checkpoint(fsdp, cfg, base_dir, root, step, meta, rank=rank, device=device, tied=tied, dist=dist)
                    save_train_state(fsdp, optimizer, root, step, state_meta, int(t.get("keep_train_states", 1)), rank=rank, dist=dist)
                    log(f"saved {step_name(step)} (HF dir + train state) in {time.time() - t_save:.0f}s")
                    t_last = time.time()
                if stop_after is not None and step >= stop_after:
                    log(f"--stop-after-step {stop_after}: stopping")
                    return 0
            del it, loader
        if rank == 0:
            ckpts = complete_checkpoints(root)
            missing = [s for s in plan["save_steps"] if s not in ckpts]
            if missing:
                raise RuntimeError(f"training ended without checkpoints {missing}")
            write_json(root / "train_done.json", {
                "sft_version": SFT_VERSION, "finished": now_iso(), "plan": plan,
                "checkpoints": {str(s): str(ckpts[s].relative_to(root)) for s in plan["save_steps"]},
                "train_log_sha256": file_sha256(root / "train_log.jsonl"), "model": model_info,
                "versions": _versions(), "provenance": _provenance(), "train_state_removed": True,
            })
            # the resumable FSDP state (fp32 params + AdamW, ~12 bytes/param) is useless once done
            shutil.rmtree(root / "state", ignore_errors=True)
            log(f"training finished: {total} steps, checkpoints {plan['save_steps']}")
        dist.barrier()
        return 0
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


# ======================================================================================
# CPU preflight


def _sha_of_output(receipt: Mapping[str, Any], key: str) -> Optional[str]:
    return ((receipt.get("outputs") or {}).get(key) or {}).get("sha256")


def check_data(cfg: Mapping[str, Any], data_root: Path) -> Dict[str, Any]:
    """Data and receipt sha256s (raises ConfigError on any mismatch)."""
    d = cfg["data"]
    out: Dict[str, Any] = {}
    sft_receipt = read_json(data_root / d["sft_receipt"])
    sft_sha = file_sha256(data_root / d["sft_jsonl"])
    if _sha_of_output(sft_receipt, d["sft_jsonl"]) != sft_sha:
        raise ConfigError(f"{d['sft_jsonl']} sha256 {sft_sha} != {d['sft_receipt']}")
    export_sha = file_sha256(data_root / d["export_receipt"])
    if (sft_receipt.get("export_receipt") or {}).get("sha256") != export_sha:
        raise ConfigError(f"{d['export_receipt']} sha256 {export_sha} != the one {d['sft_receipt']} recorded")
    export_receipt = read_json(data_root / d["export_receipt"])
    store_sha = file_sha256(data_root / d["store"])
    if _sha_of_output(export_receipt, d["store"]) != store_sha:
        raise ConfigError(f"{d['store']} sha256 {store_sha} != {d['export_receipt']}")
    dev_receipt = read_json(data_root / d["dev_receipt"])
    dev_sha = file_sha256(data_root / d["dev_parquet"])
    if _sha_of_output(dev_receipt, d["dev_parquet"]) != dev_sha:
        raise ConfigError(f"{d['dev_parquet']} sha256 {dev_sha} != {d['dev_receipt']}")
    dev_store_sha = file_sha256(data_root / d["dev_store"])
    if _sha_of_output(dev_receipt, d["dev_store"]) != dev_store_sha:
        raise ConfigError(f"{d['dev_store']} sha256 {dev_store_sha} != {d['dev_receipt']}")
    rl_dev_sha = file_sha256(data_root / d["rl_dev_parquet"])
    if _sha_of_output(dev_receipt, d["rl_dev_parquet"]) != rl_dev_sha:
        raise ConfigError(f"{d['rl_dev_parquet']} sha256 {rl_dev_sha} != {d['dev_receipt']}")
    out["files"] = {
        d["sft_jsonl"]: sft_sha, d["sft_receipt"]: file_sha256(data_root / d["sft_receipt"]), d["export_receipt"]: export_sha,
        d["store"]: store_sha, d["dev_parquet"]: dev_sha, d["dev_receipt"]: file_sha256(data_root / d["dev_receipt"]),
        d["dev_store"]: dev_store_sha, d["rl_dev_parquet"]: rl_dev_sha,
    }
    out["sft_rows"] = int(sft_receipt["rows"])
    out["export_provenance"] = sft_receipt.get("provenance")
    out["system_prompt_sha256"] = sft_receipt.get("system_prompt_sha256")
    out["dev_rows"] = int((dev_receipt.get("totals") or {}).get("accepted", 0)) or None
    return out


def check_records(cfg: Mapping[str, Any], data_root: Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every SFT row: system prompt == cert_system_v1, user text hash == the eval public prompt hash
    of its instance (extension public.prompt_sha256.px_source), target == make_sft_target."""
    from trace_tasks.experimental.certificates import prompts
    from trace_tasks.experimental.certificates.export import RL_FRAME, make_sft_target, read_store
    from trace_tasks.experimental.certificates.types import CertificateExtension

    system = prompts.system_prompt()
    store = {}
    for payload in read_store(data_root / cfg["data"]["store"]):
        ext = CertificateExtension.from_dict(payload)
        store[ext.certificate_extension_id] = ext
    bad: List[str] = []
    for rec in records:
        ext = store.get(rec["cert_extension_id"])
        if ext is None or ext.instance_id != rec["instance_id"]:
            bad.append(f"{rec['instance_id']}: no matching extension")
            continue
        if rec["system"] != system or rec["system_sha256"] != text_sha256(system):
            bad.append(f"{rec['instance_id']}: system prompt differs from cert_system_v1")
        want = ext.public["prompt_sha256"].get(RL_FRAME)
        if text_sha256(rec["user"]) != want or rec["prompt_sha256"] != want:
            bad.append(f"{rec['instance_id']}: user text is not the {RL_FRAME} public prompt")
        if rec["target"] != make_sft_target({"instance_id": rec["instance_id"]}, ext):
            bad.append(f"{rec['instance_id']}: target != make_sft_target")
        if len(bad) > 20:
            break
    if bad:
        raise ConfigError("SFT rows fail the record checks: " + "; ".join(bad[:5]))
    return {"rows_checked": len(records), "system_prompt_sha256": text_sha256(system), "system_prompt_file_sha256": file_sha256(prompts.system_prompt_path())}


def rl_equivalence(encoder: Encoder, processor_dir: Path, rlvr_root: Path, rows: Sequence[Mapping[str, Any]], system_prompt_file: Path,
                   min_pixels: int, max_pixels: int, max_prompt_length: int = 3072) -> Dict[str, Any]:
    """Prompt token ids of ``rows`` (dicts with user, image bytes or path) through EasyR1's
    RLHFDataset (a temporary RL-schema parquet) and through ``encoder``; also raw_prompt_ids and the
    actor's pixel_values (processor.image_processor on process_image). Returns per-row results."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    backend = Path(rlvr_root) / "rlvr" / "easyr1_backend"
    tmp = Path(tempfile.mkdtemp(prefix="sft_rlcheck_"))
    os.environ["HF_DATASETS_CACHE"] = str(tmp / "hf_datasets")
    sys.dont_write_bytecode = True  # never write into the read-only trace-rlvr checkout
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    try:
        from verl.utils.dataset import RLHFDataset, process_image as easyr1_process_image
        from verl.utils.tokenizer import get_processor, get_tokenizer

        tokenizer = get_tokenizer(str(processor_dir), use_fast=True)
        processor = get_processor(str(processor_dir), use_fast=True)
        table = []
        for i, row in enumerate(rows):
            image = row["image"]
            data = image if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
            table.append({"images": [{"bytes": bytes(data), "path": None}], "prompt_cert": row["user"], "instance_id": row["instance_id"],
                          "answer_gt": json.dumps({"cert_extension_id": row.get("cert_extension_id", "")})})
        path = tmp / "rows.parquet"
        pq.write_table(pa.Table.from_pylist(table), path)
        ds = RLHFDataset(str(path), tokenizer, processor, prompt_key="prompt_cert", answer_key="answer_gt", max_prompt_length=max_prompt_length,
                         truncation="right", system_prompt_file=str(system_prompt_file), min_pixels=min_pixels, max_pixels=max_pixels,
                         filter_overlong_prompts=False)
        results = []
        for i, row in enumerate(rows):
            ex = ds[i]
            mask = ex["attention_mask"].bool()
            rl_ids = ex["input_ids"][mask].tolist()
            image = row["image"]
            img = process_image(image if isinstance(image, (bytes, bytearray)) else Path(image), min_pixels, max_pixels)
            mine = encoder.encode_prompt(row["user"], img)
            rl_img = easyr1_process_image(ex["multi_modal_data"]["images"][0], min_pixels, max_pixels)
            actor = dict(processor.image_processor(images=[rl_img], return_tensors="pt"))
            results.append({
                "instance_id": row["instance_id"],
                "prompt_ids_equal": rl_ids == mine["input_ids"].tolist(),
                "raw_prompt_ids_equal": list(ex["raw_prompt_ids"]) == encoder.prompt_token_ids(row["user"]),
                "pixel_values_equal": bool(torch.equal(actor["pixel_values"], mine["pixel_values"])),
                "image_grid_thw_equal": bool(torch.equal(actor["image_grid_thw"], mine["image_grid_thw"])),
                "prompt_tokens": len(rl_ids),
                "truncated": bool(mask.all()) and len(rl_ids) >= max_prompt_length,
            })
        return {"rows": results, "all_equal": all(r["prompt_ids_equal"] and r["raw_prompt_ids_equal"] and r["pixel_values_equal"]
                                                  and r["image_grid_thw_equal"] and not r["truncated"] for r in results)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def dev_rows(parquet: Path, rows_per_task: Optional[int] = None, *, with_images: bool = True) -> List[Dict[str, Any]]:
    """E1-dev rows of the public cert_px parquet in file order (``rows_per_task``: first k per task)."""
    import pyarrow.parquet as pq

    cols = ["instance_id", "prompt_answer", "task", "cert_extension_id"] + (["images"] if with_images else [])
    table = pq.read_table(parquet, columns=cols).to_pylist()
    out, seen = [], {}
    for i, row in enumerate(table):
        if rows_per_task is not None:
            if seen.get(row["task"], 0) >= rows_per_task:
                continue
            seen[row["task"]] = seen.get(row["task"], 0) + 1
        item = {"row_index": i, "instance_id": row["instance_id"], "task": row["task"], "cert_extension_id": row["cert_extension_id"], "user": row["prompt_answer"]}
        if with_images:
            if len(row["images"]) != 1:
                raise ConfigError(f"{parquet} row {i}: expected one image")
            item["image"] = row["images"][0]["bytes"]
        out.append(item)
    return out


def cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    smoke = bool(args.smoke)
    data_root = Path(args.data).resolve()
    report: Dict[str, Any] = {"sft_version": SFT_VERSION, "checked": now_iso(), "config": os.path.abspath(args.config),
                              "config_sha256": file_sha256(args.config), "smoke": smoke, "ok": False}
    try:
        report["data"] = check_data(cfg, data_root)
        base_dir = base_snapshot(cfg)
        report["base"] = {"dir": str(base_dir), "repo_id": cfg["model"]["repo_id"], "revision": cfg["model"]["revision"],
                          "files": {n: file_sha256(base_dir / n) for n in AUX_FILES if (base_dir / n).exists()}}
        index = base_dir / "model.safetensors.index.json"
        if index.exists():
            report["base"]["files"]["model.safetensors.index.json"] = file_sha256(index)
        all_records = load_sft_records(data_root, cfg, smoke=False)
        if len(all_records) != report["data"]["sft_rows"]:
            raise ConfigError(f"sft.jsonl has {len(all_records)} rows, receipt says {report['data']['sft_rows']}")
        report["records"] = check_records(cfg, data_root, all_records)
        records = load_sft_records(data_root, cfg, smoke=smoke)
        processor = load_processor(base_dir)
        encoder = Encoder(processor, records[0]["system"], cfg["data"]["min_pixels"], cfg["data"]["max_pixels"])
        prompt_len = encoder.prompt_lengths([r["user"] for r in records], [r["image_size"] for r in records])
        target_len = encoder.target_lengths([r["target"] for r in records])
        total = [p + t for p, t in zip(prompt_len, target_len)]
        report["lengths"] = {"rows": len(records), "max_total": max(total), "max_prompt": max(prompt_len), "max_target": max(target_len),
                             "mean_total": sum(total) / len(total), "mean_target": sum(target_len) / len(target_len),
                             "label_tokens": sum(target_len), "max_length": int(cfg["data"]["max_length"])}
        if max(total) > int(cfg["data"]["max_length"]):
            raise ConfigError(f"{sum(1 for x in total if x > cfg['data']['max_length'])} rows exceed max_length")
        # exact length formula vs the processor on a few rows
        for idx in sorted({0, len(records) // 3, (2 * len(records)) // 3, len(records) - 1}):
            rec = records[idx]
            img = process_image(data_root / rec["image"], encoder.min_pixels, encoder.max_pixels)
            n = int(encoder.encode_prompt(rec["user"], img)["input_ids"].numel())
            if n != prompt_len[idx]:
                raise ConfigError(f"prompt length formula {prompt_len[idx]} != processor {n} for {rec['instance_id']}")
        plan = make_plan(cfg, len(records), smoke=smoke)
        report["plan"] = plan
        from trace_tasks.core import canonical

        if getattr(canonical, "rfc8785", None) is None:
            raise ConfigError("rfc8785 is not importable: trace_tasks canonical JSON would silently fall back")
        report["canonical_rfc8785"] = True
        if args.rlvr_root:
            from trace_tasks.experimental.certificates import prompts

            by_kind: Dict[str, List[Mapping[str, Any]]] = {}
            for rec in records:
                by_kind.setdefault(rec["adapter_id"].split(".", 1)[0], []).append(rec)
            picks = [dict(rec, image=data_root / rec["image"]) for kind in ADAPTERS for rec in by_kind.get(kind, [])[:2]]
            sft_eq = rl_equivalence(encoder, base_dir, Path(args.rlvr_root), picks, prompts.system_prompt_path(),
                                    cfg["data"]["min_pixels"], cfg["data"]["max_pixels"])
            dev = dev_rows(data_root / cfg["data"]["dev_parquet"], 2)
            dev_eq = rl_equivalence(encoder, base_dir, Path(args.rlvr_root), dev, prompts.system_prompt_path(),
                                    cfg["data"]["min_pixels"], cfg["data"]["max_pixels"])
            report["easyr1_equivalence"] = {"sft_rows": sft_eq, "e1_dev_rows": dev_eq, "rlvr_head": _git_head(Path(args.rlvr_root))}
            if not (sft_eq["all_equal"] and dev_eq["all_equal"]):
                raise ConfigError("SFT prompt tokens differ from EasyR1 RLHFDataset (see check.json easyr1_equivalence)")
        else:
            report["easyr1_equivalence"] = None
        report["versions"] = _versions()
        report["provenance"] = _provenance()
        report["ok"] = True
        return 0
    except ConfigError as exc:
        report["error"] = str(exc)
        print(f"check failed: {exc}", file=sys.stderr)
        return 4
    finally:
        if args.out:
            write_json(Path(args.out) / "check.json", report)


def _git_head(path: Path) -> Optional[str]:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=60).stdout.strip()
    except Exception:
        return None


# ======================================================================================
# dev generation (offline vLLM) and merge


def gen_meta(cfg: Mapping[str, Any], ckpt: Path, rows: Sequence[Mapping[str, Any]], shard: int, num_shards: int) -> Dict[str, Any]:
    info = read_json(ckpt / "sft_checkpoint.json") if (ckpt / "sft_checkpoint.json").exists() else {}
    e = cfg["eval"]
    return {
        "ckpt": str(ckpt), "ckpt_step": info.get("step"), "ckpt_files": {k: v.get("sha256") for k, v in (info.get("files") or {}).items()},
        "shard": shard, "num_shards": num_shards, "rows": [r["instance_id"] for r in rows],
        "decoding": {"temperature": float(e["temperature"]), "max_new_tokens": int(e["max_new_tokens"]), "seed": int(e["seed"]),
                     "suppress_image_token": bool(e.get("suppress_image_token", True)), "max_model_len": int(e["max_model_len"])},
    }


STOP_RULE_DECODING = "greedy"  # DESIGN 7.2: not trace-tool's "trace" preset (temperature 0.6), on purpose


def run_settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``settings`` block of run_config.json, in trace-tool's vocabulary, that the certificate
    scorer compares with DESIGN 7.1. Everything matches the cert_v1 condition except the
    decoding (DESIGN 7.2's greedy 2048-token stop-rule generation), so jobs/sft.sh scores these
    runs with --allow-settings-mismatch and sft_select.py accepts that single recorded difference."""
    from trace_tasks.experimental.certificates import prompts

    e = cfg["eval"]
    return {
        "system_prompt_sha256": text_sha256(prompts.system_prompt()),
        "oracle_crop": False,
        "tools": [],
        "decoding": STOP_RULE_DECODING,
        "decoding_params": {"temperature": float(e["temperature"]), "top_p": 1.0, "top_k": -1, "seed": int(e["seed"]),
                            "max_tokens": int(e["max_new_tokens"]), "suppress_image_token": bool(e.get("suppress_image_token", True))},
        "max_turns": 1,
        "max_tokens_per_turn": int(e["max_new_tokens"]),
        "max_total_tokens": int(e["max_new_tokens"]),
        "min_pixels": int(cfg["data"]["min_pixels"]),
        "max_pixels": int(cfg["data"]["max_pixels"]),
    }


def cmd_gen(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data_root = Path(args.data).resolve()
    ckpt = Path(args.ckpt).resolve()
    out = Path(args.out)
    per_task = args.rows_per_task
    rows_all = dev_rows(data_root / cfg["data"]["dev_parquet"], per_task)
    rows = rows_all[args.shard :: args.num_shards]
    meta = gen_meta(cfg, ckpt, rows, args.shard, args.num_shards)
    meta_path = out.with_name(out.name + ".meta.json")
    if out.exists() and meta_path.exists():
        old = read_json(meta_path)
        n = sum(1 for ln in out.read_text().splitlines() if ln.strip())
        if old.get("rows") == meta["rows"] and old.get("ckpt_files") == meta["ckpt_files"] and old.get("decoding") == meta["decoding"] and n == len(rows):
            print(f"{out}: complete ({n} rows), skipping")
            return 0
    from trace_tasks.experimental.certificates import prompts
    from vllm import LLM, SamplingParams

    processor = load_processor(ckpt)
    encoder = Encoder(processor, prompts.system_prompt(), cfg["data"]["min_pixels"], cfg["data"]["max_pixels"])
    e = cfg["eval"]
    inputs = [{"prompt_token_ids": encoder.prompt_token_ids(r["user"]),
               "multi_modal_data": {"image": [process_image(r["image"], encoder.min_pixels, encoder.max_pixels)]}} for r in rows]
    t0 = time.time()
    llm = LLM(model=str(ckpt), tokenizer=str(ckpt), dtype="bfloat16", seed=int(e["seed"]), max_model_len=int(e["max_model_len"]),
              gpu_memory_utilization=float(e["gpu_memory_utilization"]), limit_mm_per_prompt={"image": 1}, mm_processor_cache_gb=0,
              tensor_parallel_size=1, trust_remote_code=False)
    sampling = SamplingParams(n=1, temperature=float(e["temperature"]), top_p=1.0, top_k=-1, max_tokens=int(e["max_new_tokens"]),
                              seed=int(e["seed"]), skip_special_tokens=True,
                              logit_bias={encoder.image_token_id: -100.0} if e.get("suppress_image_token", True) else None)
    t1 = time.time()
    outputs = llm.generate(inputs, sampling, use_tqdm=False)
    t2 = time.time()
    lines = []
    for r, o, inp in zip(rows, outputs, inputs):
        c = o.outputs[0]
        lines.append(json.dumps({
            "instance_id": r["instance_id"], "row_index": r["row_index"], "task": r["task"], "cert_extension_id": r["cert_extension_id"],
            "final_response": c.text, "completion_tokens": len(c.token_ids), "final_prompt_tokens": len(inp["prompt_token_ids"]),
            "stop_reason": "answer" if c.finish_reason == "stop" else ("length" if c.finish_reason == "length" else str(c.finish_reason)),
            "finish_reason": c.finish_reason, "seed": int(e["seed"]), "condition": cfg["eval"]["condition"],
            # trace-tool-shaped transcript: the scorer re-checks the prompt against the frozen hash
            "transcript": [*encoder.messages(r["user"]), {"role": "assistant", "content": c.text}],
        }, sort_keys=True))
    atomic_write_text(out, "".join(ln + "\n" for ln in lines))
    write_json(meta_path, {**meta, "timing_s": {"load": round(t1 - t0, 1), "generate": round(t2 - t1, 1)}, "versions": _versions(), "written": now_iso()})
    print(f"{out}: {len(lines)} rows, load {t1 - t0:.0f}s, generate {t2 - t1:.0f}s")
    return 0


def cmd_merge_gen(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data_root = Path(args.data).resolve()
    ckpt = Path(args.ckpt).resolve()
    run = Path(args.run)
    rows = dev_rows(data_root / cfg["data"]["dev_parquet"], args.rows_per_task, with_images=False)
    want = {r["instance_id"]: r for r in rows}
    got: Dict[str, Dict[str, Any]] = {}
    shards = sorted((run / "shards").glob(f"shard_*of{args.num_shards:02d}.jsonl"))
    if len(shards) != args.num_shards:
        raise ConfigError(f"{run}/shards has {len(shards)} of {args.num_shards} shard files")
    files = {}
    for path in shards:
        meta = read_json(path.with_name(path.name + ".meta.json"))
        if meta.get("ckpt") != str(ckpt):
            raise ConfigError(f"{path} was generated from {meta.get('ckpt')}, not {ckpt}")
        files[path.name] = file_sha256(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["instance_id"] in got:
                    raise ConfigError(f"{rec['instance_id']} generated twice")
                got[rec["instance_id"]] = rec
    if set(got) != set(want):
        raise ConfigError(f"shards hold {len(got)} rows, want {len(want)} (missing {len(set(want) - set(got))}, extra {len(set(got) - set(want))})")
    info = read_json(ckpt / "sft_checkpoint.json")
    label = f"{args.label}@{step_name(info['step'])}"
    records = sorted(got.values(), key=lambda r: r["row_index"])
    for rec in records:
        rec["model"] = label
    atomic_write_text(run / "responses.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    parquet = data_root / cfg["data"]["dev_parquet"]
    write_json(run / "run_config.json", {
        "model": label, "model_id": cfg["model"]["repo_id"], "model_revision": f"sft:{step_name(info['step'])}",
        "checkpoint": {"path": str(ckpt), "step": info["step"], "epoch": info.get("epoch"), "files": info.get("files")},
        "condition": cfg["eval"]["condition"], "seed": int(cfg["eval"]["seed"]), "split": f"cert_{cfg['data']['dev_split']}",
        "dataset": {"path": str(parquet), "sha256": file_sha256(parquet), "rows_selected": len(rows), "rows_per_task": args.rows_per_task},
        "decoding": gen_meta(cfg, ckpt, [], 0, 1)["decoding"], "engine": "vllm-offline", "shards": files, "written": now_iso(),
        "settings": run_settings(cfg),
    })
    print(f"{run}/responses.jsonl: {len(records)} rows from {len(shards)} shards")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data_root = Path(args.data).resolve()
    n = len(load_sft_records(data_root, cfg, smoke=bool(args.smoke)))
    plan = make_plan(cfg, n, smoke=bool(args.smoke), world_size=args.world_size)
    text = json.dumps(plan, indent=1, sort_keys=True)
    if args.out:
        atomic_write_text(Path(args.out), text + "\n")
    print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, data=True):
        p.add_argument("--config", required=True, help="configs/sft_<size>.yaml")
        if data:
            p.add_argument("--data", required=True, help="frozen data root ($TRACE_CERT_RUNS/data/v2)")

    p = sub.add_parser("plan", help="print the step plan as JSON")
    common(p)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--world-size", type=int)
    p.add_argument("--out")
    p = sub.add_parser("check", help="CPU preflight (writes <out>/check.json)")
    common(p)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out")
    p.add_argument("--rlvr-root", help="trace-rlvr checkout: compare prompt token ids with EasyR1 RLHFDataset")
    p = sub.add_parser("train", help="FSDP training (torchrun --nproc_per_node=N sft.py train ...)")
    common(p)
    p.add_argument("--out", required=True, help="run root, e.g. $TRACE_CERT_RUNS/sft/qwen2.5-vl-3b")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--per-device-batch", type=int, help="override train.per_device_batch (math unchanged)")
    p.add_argument("--stop-after-step", type=int, help=argparse.SUPPRESS)  # tests: stop (as if killed) after a step
    p = sub.add_parser("gen", help="greedy E1-dev generation of one checkpoint shard (offline vLLM, one GPU)")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--rows-per-task", type=int)
    p.add_argument("--out", required=True)
    p = sub.add_parser("merge-gen", help="merge generation shards into responses.jsonl + run_config.json")
    common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", required=True, help="dev/step_XXXXXX directory (holds shards/)")
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--rows-per-task", type=int)
    p.add_argument("--label", required=True, help="model label, e.g. sft-qwen2.5-vl-3b")
    args = ap.parse_args(argv)
    handlers = {"plan": cmd_plan, "check": cmd_check, "train": cmd_train, "gen": cmd_gen, "merge-gen": cmd_merge_gen}
    try:
        return handlers[args.cmd](args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
