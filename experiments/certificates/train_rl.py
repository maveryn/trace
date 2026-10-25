#!/usr/bin/env python3
"""M3 RL runner for TRACE certificates (DESIGN 7.3): check | prepare | smoke | run | merge | status | report.

Wraps TRACE's vendored EasyR1 (GRPO) from the read-only trace-rlvr checkout exactly as
``rlvr/train.py`` launches it (``python -m verl.trainer.main config=<resolved.yaml>`` with
``cwd = rlvr/easyr1_backend``), but with the experimental certificate reward and the DESIGN 7.3
key changes from ``configs/rl_<model>.yaml``. Arms A / G / P / P_gated / G_gated differ only in
``worker.reward.reward_function_kwargs`` (``arm``, ``gated``).

Run in the RL env (torch 2.8.0+cu128, vLLM 0.10.2, transformers 4.57.6, flash-attn 2.8.3), from
any cwd. Machine locations come from the environment (``$TRACE_CERT_RUNS/host/env.sh``):

  TRACE_CERT_RUNS   artifact root; runs go to rl/<model>/<arm>/s<seed>/, smokes to rl_smoke/ (TRC_RL_ROOT,
                    TRC_RL_SMOKE_ROOT override those two; the check caches stay in rl/.cache)
  TRACE_RLVR_ROOT   read-only trace-rlvr checkout (release config, EasyR1 backend, model merger)
  HF_HOME           cache with the pinned base snapshots (smoke model, processor identity)
  TRC_DATA          optional, default $TRACE_CERT_RUNS/data/v2 (frozen; never written; common.sh default)
  TRC_NVIDIA_SMI    optional nvidia-smi stand-in (tests)

Exit codes: 0 ok / complete; 1 incomplete (rerun resumes); 2 usage; 4 configuration (check failed,
identity mismatch with an existing run, wrong N or hardware).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

CERT_DIR = Path(__file__).resolve().parent
TRACE_ROOT = CERT_DIR.parents[1]
CONFIG_DIR = CERT_DIR / "configs"
REWARD_ENTRY = CERT_DIR / "reward_entry.py"
SYSTEM_PROMPT = TRACE_ROOT / "src" / "trace_tasks" / "resources" / "prompts" / "experimental" / "certificates" / "cert_system_v1.txt"
MODELS = ("3b", "7b")
ARMS = ("A", "G", "P", "P_gated", "G_gated")
SPEC_SCHEMA = "trace-cert-rl-config-v1"
RECEIPT_SCHEMA = "trace-cert-rl-run-receipt-v1"
MERGE_SCHEMA = "trace-cert-rl-merge-receipt-v1"
RUN_RECEIPT = "run_receipt.json"
MERGE_RECEIPT = "merge_receipt.json"
TRACKER = "checkpoint_tracker.json"
CHECK_VERSION = 3  # bump when a cached check would no longer be valid
# trace_tasks runtime deps that rlvr/requirements-cu128.txt does not pin (DESIGN 1)
TRACE_TASKS_DEPS = ("scipy", "networkx", "rfc8785", "blake3", "zstandard", "cairosvg", "PyYAML", "tqdm", "numpy", "pillow")
# packages whose version mismatch against rlvr/requirements-cu128.txt is fatal
CORE_PINS = ("torch", "vllm", "transformers", "flash-attn", "ray", "accelerate", "peft", "liger-kernel",
             "qwen-vl-utils", "tensordict", "torchdata", "omegaconf", "datasets", "pyarrow", "safetensors", "numpy")
# non-weight files a served checkpoint needs; copied from the model dir into every merged dir
PROCESSOR_FILES = ("chat_template.json", "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
                   "processor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "generation_config.json")
MODEL_REQUIRED = ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")


class ConfigError(RuntimeError):
    """A configuration, input or identity problem (exit 4)."""


class Incomplete(RuntimeError):
    """The run stopped before it finished; a rerun resumes (exit 1)."""


# ===================================================================================== logging

TAG = os.environ.get("TRC_RL_TAG", "train_rl")


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"{_ts()} [{TAG}] {message}", flush=True)


def flag(kind: str, detail: str) -> None:
    """``FLAG <kind> <detail>``: machine-read by the host watcher (same shape as common.sh trc_flag)."""
    log(f"FLAG {kind} {detail}")


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ===================================================================================== small helpers


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def canonical_sha(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def get_dotted(tree: Mapping[str, Any], dotted: str) -> Any:
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def set_dotted(tree: Dict[str, Any], dotted: str, value: Any, *, must_exist: bool = True) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ConfigError(f"release easyr1 block has no section {dotted!r}")
        node = node[part]
    if must_exist and parts[-1] not in node:
        raise ConfigError(f"release easyr1 block has no key {dotted!r}; refusing to invent EasyR1 keys")
    node[parts[-1]] = value


def flatten(tree: Any, prefix: str = "") -> Dict[str, Any]:
    """Dotted leaves; dicts are expanded, everything else (lists included) is a leaf."""
    out: Dict[str, Any] = {}
    if isinstance(tree, Mapping) and tree:
        for key, value in tree.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out[prefix] = tree
    return out


def config_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """``{dotted key: (a value, b value)}`` for every leaf that differs (or exists on one side)."""
    fa, fb = flatten(a), flatten(b)
    missing = object()
    out = {}
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key, missing), fb.get(key, missing)
        # an empty mapping on one side that is filled on the other shows up as its leaves only
        if (va == {} and vb is missing and any(k.startswith(key + ".") for k in fb)) or \
                (vb == {} and va is missing and any(k.startswith(key + ".") for k in fa)):
            continue
        if va is missing or vb is missing or va != vb or type(va) is not type(vb):
            out[key] = (None if va is missing else va, None if vb is missing else vb)
    return out


def git_info(path: Path) -> Dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=120, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("status", "--porcelain")
    return {
        "head": head.strip() if head else None,
        "dirty": bool(status.strip()) if status is not None else None,
        "untracked_or_dirty": bool(untracked.strip()) if untracked is not None else None,
    }


def short_host() -> str:
    return socket.gethostname().split(".")[0]


def env_required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ConfigError(f"environment variable {name} is not set (source $TRACE_CERT_RUNS/host/env.sh)")
    return value


class ShaCache:
    """sha256 of large files keyed by (realpath, size, mtime_ns, inode); shared JSON on NFS."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        if path and path.is_file():
            with contextlib.suppress(Exception):
                self.data = read_json(path)

    def sha(self, file: Path) -> str:
        file = Path(file)
        real = os.path.realpath(file)
        st = os.stat(real)
        key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino}
        hit = self.data.get(real)
        if isinstance(hit, dict) and all(hit.get(k) == v for k, v in key.items()) and hit.get("sha256"):
            return hit["sha256"]
        value = sha256_file(Path(real))
        self.data[real] = {**key, "sha256": value}
        if self.path:
            with contextlib.suppress(Exception):
                fresh = read_json(self.path) if self.path.is_file() else {}
                fresh[real] = self.data[real]
                write_json(self.path, fresh)
        return value


# ===================================================================================== spec + paths


def load_spec(model: str) -> Dict[str, Any]:
    if model not in MODELS:
        raise ConfigError(f"--model must be one of {MODELS}, got {model!r}")
    path = CONFIG_DIR / f"rl_{model}.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != SPEC_SCHEMA or spec.get("model") != model:
        raise ConfigError(f"{path} is not a {SPEC_SCHEMA} spec for {model}")
    spec["_path"] = str(path)
    spec["_sha256"] = sha256_file(path)
    return spec


class Paths:
    """Machine locations, all from the environment (no machine paths in this checkout)."""

    def __init__(self, spec: Mapping[str, Any], env: Optional[Mapping[str, str]] = None) -> None:
        env = os.environ if env is None else env

        def need(name: str) -> str:
            value = env.get(name, "")
            if not value:
                raise ConfigError(f"environment variable {name} is not set (source $TRACE_CERT_RUNS/host/env.sh)")
            return value

        self.trace_root = TRACE_ROOT
        self.runs_root = Path(need("TRACE_CERT_RUNS"))
        self.rl_root = Path(env.get("TRC_RL_ROOT") or self.runs_root / "rl")
        self.smoke_root = Path(env.get("TRC_RL_SMOKE_ROOT") or self.runs_root / "rl_smoke")
        self.rlvr_root = Path(need("TRACE_RLVR_ROOT"))
        self.data = Path(env.get("TRC_DATA") or self.runs_root / "data" / "v2")
        self.hf_home = Path(env["HF_HOME"]) if env.get("HF_HOME") else None
        self.release_config = self.rlvr_root / spec["release"]["config"]
        self.backend = self.rlvr_root / spec["release"]["backend"]
        self.merger = self.rlvr_root / spec["release"]["merger"]
        self.release_train_py = self.rlvr_root / "rlvr" / "train.py"
        self.requirements = self.rlvr_root / "rlvr" / "requirements-cu128.txt"
        self.cache = self.runs_root / "rl" / ".cache"
        self.train_parquet = self.data / spec["data"]["train"]["path"]
        self.train_dir = self.data / spec["data"]["train"]["dir"]
        self.val_parquet = self.data / spec["data"]["val"]["path"]
        self.store = self.data / spec["data"]["store"]
        self.receipts = self.data / "receipts"

    def base_snapshot(self, spec: Mapping[str, Any]) -> Path:
        if self.hf_home is None:
            raise ConfigError("HF_HOME is not set; it locates the pinned base snapshot")
        repo = spec["base_model"]["repo_id"].replace("/", "--")
        path = self.hf_home / "hub" / f"models--{repo}" / "snapshots" / spec["base_model"]["revision"]
        if not (path / "config.json").is_file():
            raise ConfigError(f"base snapshot {spec['base_model']['repo_id']}@{spec['base_model']['revision']} is not cached at {path}")
        return path

    def run_dir(self, spec: Mapping[str, Any], arm: str, seed: int) -> Path:
        return self.rl_root / spec["model"] / arm / f"s{seed}"


def release_easyr1(spec: Mapping[str, Any], paths: Paths) -> Dict[str, Any]:
    """The release config's ``easyr1:`` block, after checking the file's pinned sha256."""
    path = paths.release_config
    if not path.is_file():
        raise ConfigError(f"release config {path} does not exist")
    found = sha256_file(path)
    if found != spec["release"]["sha256"]:
        raise ConfigError(f"release config {path} sha256 {found} != pinned {spec['release']['sha256']}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("config_id") != spec["release"]["config_id"]:
        raise ConfigError(f"{path} is config {payload.get('config_id')!r}, expected {spec['release']['config_id']!r}")
    return copy.deepcopy(payload["easyr1"])


# ===================================================================================== config resolution

PLACEHOLDER = re.compile(r"\$\{([A-Z_]+)\}")


def substitute(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        whole = PLACEHOLDER.fullmatch(value)
        if whole:
            name = whole.group(1)
            if name not in variables:
                raise ConfigError(f"unknown placeholder ${{{name}}}")
            return copy.deepcopy(variables[name])

        def repl(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in variables:
                raise ConfigError(f"unknown placeholder ${{{name}}}")
            return str(variables[name])

        return PLACEHOLDER.sub(repl, value)
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, variables) for key, item in value.items()}
    return value


def reward_kwargs(spec: Mapping[str, Any], arm: str, *, store_path: str, store_sha256: str, details_dir: Optional[str]) -> Dict[str, Any]:
    if arm not in spec["reward"]["arms"]:
        raise ConfigError(f"--arm must be one of {tuple(spec['reward']['arms'])}, got {arm!r}")
    arm_spec = spec["reward"]["arms"][arm]
    kwargs: Dict[str, Any] = {
        "store_path": str(store_path),
        "store_sha256": str(store_sha256),
        "arm": str(arm_spec["arm"]),
        "alpha": float(spec["reward"]["alpha"]),
        "format_weight": float(spec["reward"]["format_weight"]),
        "gated": bool(arm_spec["gated"]),
        "coord_frame": str(spec["reward"]["coord_frame"]),
        # which extension splits the reward may score, and as which phase (held-out oracles never)
        "split_phases": {spec["data"]["train"]["split"]: "train", spec["data"]["val"]["split"]: "val"},
    }
    if details_dir:
        kwargs["details_dir"] = str(details_dir)
    return kwargs


def experiment_name(spec: Mapping[str, Any], arm: str, seed: int, smoke: bool = False) -> str:
    return f"cert-rl-{spec['model']}-{arm}-s{seed}" + ("-smoke" if smoke else "")


def resolve_config(
    spec: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    n_gpus: int,
    model_path: str,
    run_dir: str,
    data_root: str,
    store_sha256: str,
    save_freq: Optional[int] = None,
    smoke: bool = False,
    max_steps: Optional[int] = None,
    val_files: Optional[str] = None,
    details: bool = True,
) -> Dict[str, Any]:
    """The EasyR1 config of one run: the release block + DESIGN 7.3 overrides + plumbing (+ smoke)."""
    if n_gpus not in spec["hardware"]["n_gpus"]:
        raise ConfigError(f"N={n_gpus} GPUs is not allowed (DESIGN 7.3: N in {spec['hardware']['n_gpus']})")
    if not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"seed must be a non-negative integer, got {seed!r}")
    variables = {
        "TRACE_ROOT": str(TRACE_ROOT),
        "TRC_DATA": str(data_root),
        "N": int(n_gpus),
        "MODEL_PATH": str(model_path),
        "SEED": int(seed),
        "SAVE_FREQ": int(save_freq if save_freq is not None else spec["schedule"]["save_freq"]),
        "RUN_DIR": str(run_dir),
        "EXPERIMENT": experiment_name(spec, arm, seed, smoke),
        "REWARD_KWARGS": reward_kwargs(
            spec,
            arm,
            store_path=str(Path(data_root) / spec["data"]["store"]),
            store_sha256=store_sha256,
            details_dir=str(Path(run_dir) / "reward_details") if details else None,
        ),
    }
    config = copy.deepcopy(dict(release))
    for section in ("overrides", "plumbing"):
        for dotted, value in spec[section].items():
            set_dotted(config, dotted, substitute(value, variables))
    if smoke:
        for dotted, value in spec["smoke"]["overrides"].items():
            set_dotted(config, dotted, substitute(value, variables))
        config["trainer"]["max_steps"] = int(max_steps if max_steps is not None else spec["smoke"]["steps"])
    elif max_steps is not None:
        raise ConfigError("max_steps may only be overridden for a smoke run")
    if val_files is not None:
        if not smoke:
            raise ConfigError("val_files may only be overridden for a smoke run")
        config["data"]["val_files"] = str(val_files)
    validate_config(config, n_gpus)
    return config


def validate_config(config: Mapping[str, Any], n_gpus: int) -> None:
    """The divisibility and size rules EasyR1 enforces at worker start (fsdp_workers, ray_trainer, protocol)."""
    data, worker, trainer = config["data"], config["worker"], config["trainer"]
    actor, rollout = worker["actor"], worker["rollout"]
    rb, gb, n = int(data["rollout_batch_size"]), int(actor["global_batch_size"]), int(rollout["n"])
    errors = []
    if trainer["n_gpus_per_node"] != n_gpus or trainer["nnodes"] != 1:
        errors.append("n_gpus_per_node/nnodes")
    if rb != gb:
        errors.append(f"rollout_batch_size {rb} != global_batch_size {gb} (one optimizer step per rollout)")
    if rb % gb:
        errors.append("rollout_batch_size % global_batch_size")
    if rb % n_gpus or (rb * n) % n_gpus:
        errors.append(f"rollout batch {rb} (x{n}) not divisible by N={n_gpus}")
    per_device = gb * n // n_gpus
    if per_device == 0 or per_device % int(actor["micro_batch_size_per_device_for_update"]):
        errors.append("global batch per device vs micro_batch_size_per_device_for_update")
    if (rb * n) % int(actor["micro_batch_size_per_device_for_experience"]):
        errors.append("rollout batch * n vs micro_batch_size_per_device_for_experience")
    tp = int(rollout["tensor_parallel_size"])
    if tp > n_gpus or n_gpus % tp:
        errors.append(f"tensor_parallel_size {tp} vs N={n_gpus}")
    if int(rollout["max_num_batched_tokens"]) < int(data["max_prompt_length"]) + int(data["max_response_length"]):
        errors.append("max_num_batched_tokens < max_prompt_length + max_response_length")
    if n < 2:
        errors.append("GRPO needs rollout.n > 1")
    sf = int(trainer["save_freq"])
    if sf <= 0:
        errors.append("save_freq must be positive (resume and keep/ depend on it)")
    if not trainer.get("find_last_checkpoint"):
        errors.append("find_last_checkpoint must be true (resume)")
    if errors:
        raise ConfigError("invalid EasyR1 config: " + "; ".join(errors))


def key_allowed(key: str, allowed: Iterable[str]) -> bool:
    """``key`` is one of ``allowed`` or a leaf below one (e.g. reward_function_kwargs.arm)."""
    return any(key == a or key.startswith(a + ".") for a in allowed)


def allowed_release_diff(spec: Mapping[str, Any], smoke: bool = False) -> set:
    keys = set(spec["overrides"]) | set(spec["plumbing"])
    if smoke:
        keys |= set(spec["smoke"]["overrides"]) | {"trainer.max_steps", "data.val_files"}
    return keys


def arm_identity_keys() -> set:
    """Keys that differ between the arms only through the run directory / experiment name."""
    return {
        "trainer.save_checkpoint_path",
        "trainer.experiment_name",
        "worker.reward.reward_function_kwargs.details_dir",
    }


def arm_diff(configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Pairwise diff of resolved arm configs with the run identity (dir, name) normalised away."""
    normalised = {}
    for arm, config in configs.items():
        flat = flatten(config)
        for key in arm_identity_keys():
            if key in flat:
                flat[key] = "<run identity>"
        normalised[arm] = flat
    arms = list(configs)
    keys: set = set()
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            keys |= set(config_diff(normalised[a], normalised[b]))
    return {"differing_keys": sorted(keys), "reward_kwargs": {arm: {k: flatten(c)[f"worker.reward.reward_function_kwargs.{k}"] for k in ("arm", "gated")} for arm, c in configs.items()}}


def easyr1_structured(config: Mapping[str, Any], backend: Path) -> None:
    """Load the resolved dict into EasyR1's own PPOConfig dataclasses (catches unknown keys / types)."""
    with backend_on_path(backend):
        from omegaconf import OmegaConf  # noqa: WPS433

        from verl.trainer.config import PPOConfig  # type: ignore  # noqa: WPS433

        merged = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.create(json.loads(json.dumps(config))))
        OmegaConf.to_object(merged)


@contextlib.contextmanager
def backend_on_path(backend: Path):
    """Import from the vendored EasyR1 without writing bytecode into the read-only checkout."""
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    added = str(backend) not in sys.path
    if added:
        sys.path.insert(0, str(backend))
    try:
        yield
    finally:
        sys.dont_write_bytecode = old_flag


# ===================================================================================== release / vendor checks


def load_release_module(paths: Paths):
    """rlvr/train.py as a module (its _validate_vendor_manifest / check_release, reused verbatim)."""
    if not paths.release_train_py.is_file():
        raise ConfigError(f"{paths.release_train_py} does not exist")
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("trace_rlvr_release_train", paths.release_train_py)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_flag
    return module


def check_release(spec: Mapping[str, Any], paths: Paths) -> Dict[str, Any]:
    module = load_release_module(paths)
    try:
        module._validate_vendor_manifest()  # the 64-file EasyR1 unit, byte for byte
        result = module.check_release(spec["release"]["config_id"])  # profiles, inventory, prompt, receipt
    except module.ReleaseError as exc:
        raise ConfigError(f"trace-rlvr release check failed: {exc}") from exc
    merger_sha = sha256_file(paths.merger)
    if merger_sha != spec["release"]["merger_sha256"]:
        raise ConfigError(f"model_merger.py sha256 {merger_sha} != pinned {spec['release']['merger_sha256']}")
    return {"vendor_manifest": "ok", "release_check": result, "rlvr_git": git_info(paths.rlvr_root), "merger_sha256": merger_sha}


# ===================================================================================== environment checks


def _dist_version(name: str) -> Optional[str]:
    from importlib import metadata

    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def requirement_pins(path: Path) -> Dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[re.sub(r"\[.*\]", "", name).strip()] = version.strip()
    return pins


def check_environment(paths: Paths) -> Dict[str, Any]:
    """Imports, version pins, flash-attn, canonical rfc8785, Hungarian assignment, trace_tasks origin."""
    errors: List[str] = []
    warnings: List[str] = []
    versions: Dict[str, Optional[str]] = {}
    pins = requirement_pins(paths.requirements)
    for name, want in pins.items():
        have = _dist_version(name)
        versions[name] = have
        # PEP 440: "==2.8.0" matches the local version "2.8.0+cu128"
        if have != want and not (have and "+" not in want and have.split("+", 1)[0] == want):
            (errors if name in CORE_PINS else warnings).append(f"{name}=={have} (rlvr pins {want})")
    for name in TRACE_TASKS_DEPS:
        versions[name] = _dist_version(name)
        if versions[name] is None:
            errors.append(f"{name} is not installed (trace_tasks runtime dependency)")
    if sys.version_info[:2] not in ((3, 10), (3, 12)):
        errors.append(f"python {sys.version.split()[0]}: flash-attn wheels exist for cp310/cp312 only")

    # trace_tasks must be THIS checkout's src (the env's editable install points at trace-rlvr/src,
    # which has no experimental package): the runner puts our src first on PYTHONPATH.
    src = str(TRACE_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import trace_tasks  # noqa: WPS433
    from trace_tasks.core import canonical  # noqa: WPS433

    origin = Path(trace_tasks.__file__).resolve()
    if not str(origin).startswith(str((TRACE_ROOT / "src").resolve())):
        errors.append(f"trace_tasks imports from {origin}, not {src}")
    if canonical.rfc8785 is None:
        errors.append("trace_tasks.core.canonical falls back to json.dumps: rfc8785 is not importable")
    else:
        import rfc8785  # noqa: WPS433

        probe = {"b": [1, 2.5, "x"], "a": {"z": None, "y": True}}
        if canonical.canonical_json_bytes(probe) != rfc8785.dumps(probe):
            errors.append("canonical_json_bytes differs from rfc8785.dumps")
    from scipy.optimize import linear_sum_assignment  # noqa: WPS433

    rows, cols = linear_sum_assignment([[4, 1], [2, 3]])
    if list(rows) != [0, 1] or list(cols) != [1, 0]:
        errors.append("scipy linear_sum_assignment returned an unexpected assignment")
    from trace_tasks.experimental.certificates import reward as reward_module  # noqa: WPS433
    from trace_tasks.experimental.certificates.types import require_rfc8785  # noqa: WPS433

    require_rfc8785()
    reward_module.require_scipy_assignment()

    # the reward entry exactly as EasyR1 loads it (verl/workers/reward/function.py): by file path
    entry_spec = importlib.util.spec_from_file_location("custom_reward_fn", REWARD_ENTRY)
    entry = importlib.util.module_from_spec(entry_spec)
    assert entry_spec.loader is not None
    entry_spec.loader.exec_module(entry)
    if getattr(entry, "REWARD_TYPE", None) != "batch" or not callable(getattr(entry, "compute_score", None)):
        errors.append("reward_entry.py must export compute_score with REWARD_TYPE 'batch'")
    if entry.compute_score is not reward_module.compute_score:
        errors.append("reward_entry.compute_score is not trace_tasks.experimental.certificates.reward.compute_score")

    # flash-attn: EasyR1 hard-codes flash_attention_2 and silently falls back to a torch
    # cross-entropy when the triton op is missing (utils/torch_functional.py), so require all of it.
    fa = {}
    try:
        import flash_attn  # noqa: WPS433
        import flash_attn_2_cuda  # noqa: F401,WPS433
        from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401,WPS433
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401,WPS433
        from flash_attn.ops.triton.cross_entropy import cross_entropy_loss  # noqa: F401,WPS433

        fa = {"version": flash_attn.__version__, "file": flash_attn.__file__}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"flash-attn import failed: {type(exc).__name__}: {exc}")
    try:
        import torch  # noqa: WPS433

        torch_info = {"version": torch.__version__, "cuda": torch.version.cuda, "arch_list": torch.cuda.get_arch_list()}
    except Exception as exc:  # noqa: BLE001
        torch_info = {}
        errors.append(f"torch import failed: {exc}")
    with backend_on_path(paths.backend):
        try:
            import verl  # noqa: F401,WPS433
            from verl.trainer.config import PPOConfig  # noqa: F401,WPS433

            verl_file = str(Path(verl.__file__).resolve())
            if not verl_file.startswith(str(paths.backend.resolve())):
                errors.append(f"verl imports from {verl_file}, not the vendored backend")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"vendored verl import failed: {type(exc).__name__}: {exc}")
    return {
        "errors": errors,
        "warnings": warnings,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": versions,
        "trace_tasks": str(origin),
        "rfc8785_active": canonical.rfc8785 is not None,
        "flash_attn": fa,
        "torch": torch_info,
    }


def env_lock() -> Dict[str, str]:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all", "--disable-pip-version-check"],
                             capture_output=True, text=True, timeout=300, check=True).stdout
        method = "pip_freeze"
    except (OSError, subprocess.SubprocessError):
        from importlib import metadata

        out = "\n".join(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
        method = "importlib_metadata"
    text = "\n".join(sorted(line.strip() for line in out.splitlines() if line.strip())) + "\n"
    return {"sha256": sha256_bytes(text.encode("utf-8")), "method": method, "python": sys.executable}


# ===================================================================================== data checks


def load_receipts(paths: Paths) -> Dict[str, Any]:
    receipts = {}
    for path in sorted(paths.receipts.glob("*.json")):
        with contextlib.suppress(Exception):
            payload = read_json(path)
            if isinstance(payload, dict) and "outputs" in payload:
                receipts[path.stem] = payload
    if not receipts:
        raise ConfigError(f"no export receipts under {paths.receipts}")
    return receipts


def check_data(spec: Mapping[str, Any], paths: Paths, cache: ShaCache) -> Dict[str, Any]:
    """Frozen-data sha256s against the export receipts; train dir is parquet-only; store sha."""
    receipts = load_receipts(paths)
    out: Dict[str, Any] = {}
    for role in ("train", "val"):
        entry = spec["data"][role]
        rel, split = entry["path"], entry["split"]
        path = paths.data / rel
        if not path.is_file():
            raise ConfigError(
                f"{path} does not exist; build it with `python -m trace_tasks.experimental.certificates export "
                f"--split {split} --out $TRC_DATA` (DESIGN 6) before RL"
            )
        receipt = receipts.get(split)
        if receipt is None or rel not in receipt["outputs"]:
            raise ConfigError(f"receipts/{split}.json has no output entry for {rel}")
        want = receipt["outputs"][rel]["sha256"]
        got = cache.sha(path)
        if got != want:
            raise ConfigError(f"{rel} sha256 {got} != receipts/{split}.json {want} (frozen data changed)")
        out[role] = {"path": str(path), "split": split, "sha256": got, "rows": receipt["outputs"][rel].get("rows"),
                     "bytes": path.stat().st_size}
    listing = sorted(os.listdir(paths.train_dir))
    if listing != [Path(spec["data"]["train"]["path"]).name]:
        raise ConfigError(f"train dir {paths.train_dir} must hold only the RL parquet (EasyR1 loads the whole dir): {listing}")
    # the merged store: on-disk sha must equal the merged_store sha of the most recent export receipt
    store_rel = spec["data"]["store"]
    store_sha = cache.sha(paths.store)
    owners = [name for name, r in receipts.items() if r.get("merged_store", {}).get("path") == store_rel
              and r["merged_store"].get("sha256") == store_sha]
    if not owners:
        raise ConfigError(f"{store_rel} sha256 {store_sha} matches no export receipt's merged_store")
    for role in ("train", "val"):
        split = out[role]["split"]
        if split not in receipts[owners[0]]["merged_store"].get("splits", []):
            raise ConfigError(f"merged store does not contain split {split}")
    out["store"] = {"path": str(paths.store), "sha256": store_sha, "receipts": owners,
                    "rows": receipts[owners[0]]["merged_store"].get("rows")}
    # system prompt against the export provenance (SFT and RL records were built with this file)
    name = spec["data"]["system_prompt"]
    if not SYSTEM_PROMPT.is_file():
        raise ConfigError(f"{SYSTEM_PROMPT} does not exist")
    sys_sha = sha256_file(SYSTEM_PROMPT)
    for role in ("train", "val"):
        want = receipts[out[role]["split"]]["provenance"]["resource_sha256"].get(name)
        if want != sys_sha:
            raise ConfigError(f"{name} sha256 {sys_sha} != receipts/{out[role]['split']}.json provenance {want}")
    out["system_prompt"] = {"path": str(SYSTEM_PROMPT), "sha256": sys_sha}
    out["cert_code_sha256_at_export"] = {r: receipts[out[r]["split"]]["provenance"]["cert_code_sha256"] for r in ("train", "val")}
    return out


def check_store_join(spec: Mapping[str, Any], paths: Paths, data: Mapping[str, Any]) -> Dict[str, Any]:
    """Every RL / val row's sentinel resolves in the store with the same instance_id (as the reward joins)."""
    import pyarrow.parquet as pq  # noqa: WPS433

    key = canonical_sha({"v": CHECK_VERSION, "train": data["train"]["sha256"], "val": data["val"]["sha256"], "store": data["store"]["sha256"]})
    cache_file = paths.cache / f"join_{key[:24]}.json"
    if cache_file.is_file():
        with contextlib.suppress(Exception):
            cached = read_json(cache_file)
            if cached.get("ok"):
                return {**cached, "cached": True}
    if str(TRACE_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(TRACE_ROOT / "src"))
    from trace_tasks.experimental.certificates.reward import ExtensionStore, sentinel_id  # noqa: WPS433

    store = ExtensionStore.load(paths.store, data["store"]["sha256"])  # verifies the sha over the bytes it parses
    ids: Dict[str, str] = {}
    for ext_id, line in store._lines.items():  # instance ids without building every extension object
        ids[ext_id] = str(json.loads(line)["base"]["instance_id"])
    result: Dict[str, Any] = {"ok": True, "store_rows": len(ids)}
    for role in ("train", "val"):
        table = pq.read_table(data[role]["path"], columns=["instance_id", "cert_extension_id", "answer_gt", "task"])
        inst = table.column("instance_id").to_pylist()
        cols = table.column("cert_extension_id").to_pylist()
        gts = table.column("answer_gt").to_pylist()
        bad = []
        for i, (iid, col, gt) in enumerate(zip(inst, cols, gts)):
            try:
                ext = sentinel_id(gt)
            except ValueError as exc:
                bad.append(f"row {i}: {exc}")
                continue
            if ext != col:
                bad.append(f"row {i}: sentinel {ext} != column {col}")
            elif ext not in ids:
                bad.append(f"row {i}: {ext} not in store")
            elif ids[ext] != iid:
                bad.append(f"row {i}: instance_id {iid} != store {ids[ext]}")
            if len(bad) > 5:
                break
        if bad:
            raise ConfigError(f"{role} parquet does not join the store: {bad[:5]}")
        tasks: Dict[str, int] = {}
        for task in table.column("task").to_pylist():
            tasks[task] = tasks.get(task, 0) + 1
        result[role] = {"rows": len(inst), "unique_extensions": len(set(cols)), "tasks": tasks}
        if len(set(cols)) != len(cols):
            raise ConfigError(f"{role} parquet repeats cert_extension_id values")
    with contextlib.suppress(Exception):
        write_json(cache_file, result)
    return result


def check_reward(spec: Mapping[str, Any], release: Mapping[str, Any], paths: Paths, data: Mapping[str, Any],
                 per_task: int = 2) -> Dict[str, Any]:
    """Score the oracle SFT target of a few train and val rows per task with the reward exactly as EasyR1
    calls it (entry file by path + the resolved kwargs of every arm). Fails when the current certificate
    code cannot score the frozen store (e.g. adapter verifier versions changed after the export)."""
    import functools  # noqa: WPS433

    import pyarrow.parquet as pq  # noqa: WPS433

    from trace_tasks.experimental.certificates.export import cert_code_sha256, make_sft_target  # noqa: WPS433
    from trace_tasks.experimental.certificates.reward import load_store  # noqa: WPS433

    store = load_store(paths.store, data["store"]["sha256"])
    entry_spec = importlib.util.spec_from_file_location("custom_reward_fn", REWARD_ENTRY)
    entry = importlib.util.module_from_spec(entry_spec)
    assert entry_spec.loader is not None
    entry_spec.loader.exec_module(entry)
    import inspect  # noqa: WPS433

    params = inspect.signature(entry.compute_score).parameters
    out: Dict[str, Any] = {"cert_code_sha256": cert_code_sha256(), "export_cert_code_sha256": data["cert_code_sha256_at_export"]}
    for role in ("train", "val"):
        table = pq.read_table(data[role]["path"], columns=["prompt_cert", "instance_id", "cert_extension_id", "answer_gt", "task"])
        rows, counts = [], {}
        for row in table.to_pylist():
            if counts.get(row["task"], 0) < per_task:
                counts[row["task"]] = counts.get(row["task"], 0) + 1
                rows.append(row)
        inputs = []
        for row in rows:
            response = make_sft_target(None, store.get(row["cert_extension_id"]))
            reward_input = {"response": response, "response_length": len(response), "ground_truth": row.pop("answer_gt")}
            reward_input.update(row)
            inputs.append(reward_input)
        for arm in ARMS:
            config = resolve_config(spec, release, arm=arm, seed=1, n_gpus=spec["hardware"]["n_gpus"][0], model_path="-",
                                    run_dir="-", data_root=str(paths.data), store_sha256=data["store"]["sha256"], details=False)
            kwargs = config["worker"]["reward"]["reward_function_kwargs"]
            unknown = set(kwargs) - set(params)
            if unknown:
                raise ConfigError(f"reward_function_kwargs {sorted(unknown)} are not parameters of compute_score")
            try:
                scores = functools.partial(entry.compute_score, **kwargs)([dict(r) for r in inputs])
            except Exception as exc:  # noqa: BLE001
                raise ConfigError(
                    f"the certificate reward cannot score the frozen {role} rows with the current code ({type(exc).__name__}: {exc}); "
                    f"data exported with cert_code_sha256 {data['cert_code_sha256_at_export'][role][:12]}, code now "
                    f"{out['cert_code_sha256'][:12]}: rebuild the data (DESIGN 6) or restore the scorer") from exc
            bad = [(r["task"], s) for r, s in zip(rows, scores)
                   if s["accuracy"] != 1.0 or s["semantic"] != 1.0 or s["cert_format"] != 1.0]
            if bad:
                raise ConfigError(f"oracle SFT targets do not score A = M = F = 1 under arm {arm}: {bad[:3]}")
            out.setdefault(role, {})[arm] = round(sum(s["overall"] for s in scores) / len(scores), 4)
        out[role]["rows"] = len(rows)
    return out


# ===================================================================================== model dir checks


def processor_digest(model_path: Path) -> str:
    digest = hashlib.sha256()
    for name in PROCESSOR_FILES:
        path = model_path / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def safetensors_summary(model_path: Path) -> Dict[str, Any]:
    """Parameter count and dtypes from the safetensors headers (no tensor data is read)."""
    import struct  # noqa: WPS433

    numel, dtypes, files = 0, {}, []
    tensors: Dict[str, List[int]] = {}
    for path in sorted(model_path.glob("*.safetensors")):
        with path.open("rb") as handle:
            (size,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(size))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for dim in meta["shape"]:
                count *= int(dim)
            numel += count
            dtypes[meta["dtype"]] = dtypes.get(meta["dtype"], 0) + count
            tensors[name] = list(meta["shape"])
        files.append({"name": path.name, "bytes": path.stat().st_size})
    return {"numel": numel, "dtypes": dtypes, "files": files, "tensors": len(tensors)}


def check_model_dir(spec: Mapping[str, Any], model_path: Path, *, kind: str) -> Dict[str, Any]:
    """An SFT dir (or the base snapshot): HF files present, same architecture and size as the spec."""
    if not model_path.is_dir():
        raise ConfigError(f"model dir {model_path} does not exist")
    missing = [name for name in MODEL_REQUIRED if not (model_path / name).is_file()]
    if not list(model_path.glob("*.safetensors")):
        missing.append("*.safetensors")
    if not any((model_path / name).is_file() for name in ("chat_template.json", "chat_template.jinja")) and \
            "chat_template" not in (model_path / "tokenizer_config.json").read_text(encoding="utf-8", errors="replace"):
        missing.append("chat_template.{json,jinja}")
    if missing:
        raise ConfigError(f"model dir {model_path} lacks {missing}")
    config = read_json(model_path / "config.json")
    base = spec["base_model"]
    arch = (config.get("architectures") or [None])[0]
    text = config.get("text_config") or {}
    hidden = config.get("hidden_size", text.get("hidden_size"))
    layers = config.get("num_hidden_layers", text.get("num_hidden_layers"))
    if arch != base["architecture"] or hidden != base["hidden_size"] or layers != base["num_hidden_layers"]:
        raise ConfigError(f"model dir {model_path} is {arch} hidden {hidden} layers {layers}; the {spec['model']} spec needs "
                          f"{base['architecture']} hidden {base['hidden_size']} layers {base['num_hidden_layers']}")
    info = {
        "path": str(model_path),
        "kind": kind,
        "config_sha256": sha256_file(model_path / "config.json"),
        "processor_digest": processor_digest(model_path),
        "weights": safetensors_summary(model_path),
    }
    for name in ("sft_receipt.json", "receipt.json", "train_receipt.json"):
        if (model_path / name).is_file():
            info["sft_receipt"] = {"name": name, "sha256": sha256_file(model_path / name)}
            break
    return info


# ===================================================================================== prompt clip check


def easyr1_resized(width: int, height: int, min_pixels: Optional[int], max_pixels: Optional[int]) -> Tuple[int, int]:
    """The size ``verl.utils.dataset.process_image`` produces (same arithmetic, same order)."""
    if max_pixels is not None and width * height > max_pixels:
        factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * factor), int(height * factor)
    if min_pixels is not None and width * height < min_pixels:
        factor = math.sqrt(min_pixels / (width * height))
        width, height = int(width * factor), int(height * factor)
    return width, height


class PromptMeter:
    """Exact EasyR1 prompt length of RL rows: chat template + system prompt + image tokens.

    Uses EasyR1's own message builder and processor loaders. ``fast`` counts image tokens from the
    processor's patch arithmetic; ``exact`` runs the full processor call of RLHFDataset.__getitem__.
    """

    def __init__(self, model_path: Path, backend: Path, *, prompt_key: str, system_prompt_file: Path,
                 min_pixels: Optional[int], max_pixels: Optional[int], max_prompt_length: int) -> None:
        with backend_on_path(backend):
            from verl.utils.dataset import RLHFDataset, process_image  # type: ignore  # noqa: WPS433
            from verl.utils.tokenizer import get_processor, get_tokenizer  # type: ignore  # noqa: WPS433
        self.process_image = process_image
        self.tokenizer = get_tokenizer(str(model_path), override_chat_template=None, trust_remote_code=False, use_fast=True)
        self.processor = get_processor(str(model_path), override_chat_template=None, trust_remote_code=False, use_fast=True)
        if self.processor is None:
            raise ConfigError(f"{model_path} has no multimodal processor")
        ds = RLHFDataset.__new__(RLHFDataset)
        ds.tokenizer, ds.processor = self.tokenizer, self.processor
        ds.prompt_key, ds.answer_key, ds.image_key, ds.video_key = prompt_key, "answer_gt", "images", "videos"
        ds.image_dir, ds.video_fps, ds.format_prompt = None, 2.0, None
        ds.max_prompt_length, ds.truncation = max_prompt_length, "right"
        ds.min_pixels, ds.max_pixels = min_pixels, max_pixels
        with open(system_prompt_file, encoding="utf-8") as handle:
            ds.system_prompt = handle.read().strip() or None
        self.ds = ds
        self.prompt_key = prompt_key
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.max_prompt_length = max_prompt_length
        ip = self.processor.image_processor
        self.merge = int(ip.merge_size) ** 2
        self.image_pad = getattr(self.processor, "image_token", "<|image_pad|>")
        self._text: Dict[str, Tuple[int, int]] = {}

    def text_len(self, prompt: str) -> Tuple[int, int]:
        """(token count of the templated text, number of image pads in it)."""
        hit = self._text.get(prompt)
        if hit is None:
            messages = self.ds._build_messages({self.prompt_key: prompt, "images": None})
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            hit = (len(ids), text.count(self.image_pad))
            self._text[prompt] = hit
        return hit

    def fast(self, prompt: str, sizes: Sequence[Tuple[int, int]]) -> int:
        n_text, n_pads = self.text_len(prompt)
        if n_pads != len(sizes):
            raise ConfigError(f"prompt has {n_pads} image pads for {len(sizes)} images")
        total = n_text - n_pads
        for width, height in sizes:
            w, h = easyr1_resized(width, height, self.min_pixels, self.max_pixels)
            total += self.processor.image_processor.get_number_of_image_patches(h, w, {}) // self.merge
        return total

    def exact(self, row: Mapping[str, Any]) -> Tuple[int, int]:
        """(untruncated length, attention-mask length after RLHFDataset.__getitem__)."""
        messages = self.ds._build_messages(row)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = [self.process_image(image, self.min_pixels, self.max_pixels) for image in row["images"]]
        full = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")["input_ids"].size(-1)
        self.ds.dataset = [dict(row)]
        item = self.ds.__getitem__(0)
        return int(full), int(item["attention_mask"].sum())


def image_sizes_from_bytes(images: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    from PIL import Image  # noqa: WPS433

    sizes = []
    for image in images:
        with Image.open(io.BytesIO(image["bytes"])) as handle:
            sizes.append(tuple(handle.size))
    return sizes


def check_prompt_clip(spec: Mapping[str, Any], release: Mapping[str, Any], paths: Paths, data: Mapping[str, Any],
                      model_path: Path, *, sample_top: int = 24, sample_random: int = 6) -> Dict[str, Any]:
    """Longest tokenized prompt over every RL and val row must be <= max_prompt_length (clip ratio 0).

    Every row's image size is read from its PNG header; lengths use the processor's patch arithmetic,
    validated against the full EasyR1 processor path (RLHFDataset.__getitem__) on the longest rows,
    random rows per task and the up-scaled small images.
    """
    import random  # noqa: WPS433

    import pyarrow.parquet as pq  # noqa: WPS433

    max_len = int(spec["overrides"]["data.max_prompt_length"])
    prompt_key = spec["overrides"]["data.prompt_key"]
    min_px, max_px = release["data"]["min_pixels"], release["data"]["max_pixels"]
    key = canonical_sha({
        "v": CHECK_VERSION, "train": data["train"]["sha256"], "val": data["val"]["sha256"], "proc": processor_digest(model_path),
        "sys": data["system_prompt"]["sha256"], "max": max_len, "min_px": min_px, "max_px": max_px, "key": prompt_key,
    })
    cache_file = paths.cache / f"clip_{key[:24]}.json"
    if cache_file.is_file():
        with contextlib.suppress(Exception):
            cached = read_json(cache_file)
            if cached.get("ok"):
                return {**cached, "cached": True}
    t0 = time.time()
    meter = PromptMeter(model_path, paths.backend, prompt_key=prompt_key, system_prompt_file=SYSTEM_PROMPT,
                        min_pixels=min_px, max_pixels=max_px, max_prompt_length=max_len)
    result: Dict[str, Any] = {"max_prompt_length": max_len, "model_path": str(model_path), "processor_digest": processor_digest(model_path)}
    rng = random.Random(0)
    worst_overall = 0
    for role in ("train", "val"):
        pf = pq.ParquetFile(data[role]["path"])
        meta = pf.read(columns=[prompt_key, "task", "image_sizes"])
        prompts = meta.column(prompt_key).to_pylist()
        tasks = meta.column("task").to_pylist()
        declared = [[(int(s["width"]), int(s["height"])) for s in sizes] for sizes in meta.column("image_sizes").to_pylist()]
        sizes: List[List[Tuple[int, int]]] = []
        for group in range(pf.num_row_groups):
            for images in pf.read_row_group(group, columns=["images"]).column("images").to_pylist():
                sizes.append(image_sizes_from_bytes(images))
        if len(sizes) != len(prompts):
            raise ConfigError(f"{role}: {len(sizes)} image rows for {len(prompts)} prompts")
        mismatched = sum(1 for a, b in zip(sizes, declared) if list(a) != list(b))
        lengths = [meter.fast(p, s) for p, s in zip(prompts, sizes)]
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        chosen = set(order[:sample_top])
        by_task: Dict[str, List[int]] = {}
        for i, task in enumerate(tasks):
            by_task.setdefault(task, []).append(i)
        for rows in by_task.values():
            chosen.update(rng.sample(rows, min(sample_random, len(rows))))
        chosen.update(sorted(range(len(sizes)), key=lambda i: sizes[i][0][0] * sizes[i][0][1])[:4])
        # read the chosen rows in full (row-group by row-group) and run the exact EasyR1 path
        offsets, start = [], 0
        for group in range(pf.num_row_groups):
            offsets.append((start, start + pf.metadata.row_group(group).num_rows))
            start = offsets[-1][1]
        exact_bad, exact_n, truncated = [], 0, 0
        for group, (lo, hi) in enumerate(offsets):
            wanted = sorted(i for i in chosen if lo <= i < hi)
            if not wanted:
                continue
            table = pf.read_row_group(group).to_pylist()
            for i in wanted:
                full, kept = meter.exact(table[i - lo])
                exact_n += 1
                if full != lengths[i]:
                    exact_bad.append({"row": i, "fast": lengths[i], "exact": full})
                if kept != min(full, max_len):
                    exact_bad.append({"row": i, "exact": full, "attention": kept})
                if full > max_len:
                    truncated += 1
        if exact_bad:
            raise ConfigError(f"{role}: prompt-length arithmetic disagrees with the EasyR1 processor path: {exact_bad[:5]}")
        per_task = {}
        for task, rows in by_task.items():
            vals = sorted(lengths[i] for i in rows)
            per_task[task] = {"rows": len(rows), "max": vals[-1], "p50": vals[len(vals) // 2], "min": vals[0]}
        worst = lengths[order[0]]
        over = sum(1 for value in lengths if value > max_len)
        at_cap = sum(1 for value in lengths if value == max_len)
        worst_overall = max(worst_overall, worst)
        result[role] = {
            "rows": len(lengths), "max": worst, "argmax_row": order[0], "over_limit": over, "at_limit": at_cap,
            "clip_ratio": over / len(lengths), "per_task": per_task, "image_size_mismatch_vs_image_sizes": mismatched,
            "exact_checked_rows": exact_n, "exact_truncated": truncated,
        }
        if over or at_cap:
            raise ConfigError(f"{role}: {over} prompts exceed and {at_cap} reach max_prompt_length {max_len} (longest {worst}); "
                              "EasyR1 would right-truncate them silently")
    result.update({"ok": True, "max": worst_overall, "margin": max_len - worst_overall, "seconds": round(time.time() - t0, 1)})
    with contextlib.suppress(Exception):
        write_json(cache_file, result)
    return result


# ===================================================================================== GPU inventory


def gpu_inventory() -> List[Dict[str, str]]:
    smi = os.environ.get("TRC_NVIDIA_SMI", "nvidia-smi")
    try:
        out = subprocess.run([smi, "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigError(f"nvidia-smi failed: {exc}") from exc
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5 and parts[1]:
            rows.append(dict(zip(("index", "uuid", "name", "memory_mib", "driver"), parts[:5])))
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if visible:
        rows = [row for row in rows if row["index"] in visible or row["uuid"] in visible]
        if len(rows) != len(visible):
            raise ConfigError(f"CUDA_VISIBLE_DEVICES={','.join(visible)} does not match nvidia-smi")
    return rows


def check_hardware(spec: Mapping[str, Any], gpus: Sequence[Mapping[str, str]], n_gpus: int) -> Dict[str, Any]:
    if len(gpus) != n_gpus:
        raise ConfigError(f"--n-gpus {n_gpus} but {len(gpus)} GPUs are visible")
    if n_gpus not in spec["hardware"]["n_gpus"]:
        raise ConfigError(f"N={n_gpus} GPUs; DESIGN 7.3 allows only {spec['hardware']['n_gpus']}")
    need_gb = spec["hardware"]["min_vram_gb"][n_gpus]
    small = [g for g in gpus if float(g["memory_mib"]) < need_gb * 1000]
    if small:
        raise ConfigError(f"{spec['model']} RL on {n_gpus} GPUs needs >= {need_gb} GB per card; "
                          f"{[(g['index'], g['name'], g['memory_mib']) for g in small]}")
    names = sorted({g["name"] for g in gpus})
    if len(names) != 1:
        raise ConfigError(f"mixed GPU types {names}: arms are matched per hardware class")
    return {"n": n_gpus, "name": names[0], "memory_mib": [g["memory_mib"] for g in gpus], "uuids": [g["uuid"] for g in gpus],
            "driver": gpus[0]["driver"]}


GPU_PROBE = r"""
import json, torch
from flash_attn import flash_attn_varlen_func
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
out = []
for i in range(torch.cuda.device_count()):
    dev = torch.device("cuda", i)
    cap = torch.cuda.get_device_capability(i)
    q = torch.randn(64, 4, 64, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, 32, 64], device=dev, dtype=torch.int32)
    o = flash_attn_varlen_func(q, q, q, cu, cu, 32, 32, causal=True)
    logits = torch.randn(8, 1000, device=dev, dtype=torch.float32)
    labels = torch.randint(0, 1000, (8,), device=dev)
    loss = cross_entropy_loss(logits, labels)[0]
    ref = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    out.append({"index": i, "name": torch.cuda.get_device_name(i), "capability": f"sm_{cap[0]}{cap[1]}",
                "flash_attn_ok": bool(torch.isfinite(o).all()), "triton_ce_max_err": float((loss - ref).abs().max()),
                "arch_in_torch": f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()})
print("GPU_PROBE " + json.dumps(out))
"""


def check_gpu_kernels(n_gpus: int) -> Dict[str, Any]:
    """flash-attn varlen + triton cross-entropy on every visible GPU (the sm_120 risk, DESIGN 10). The probe
    compiles into its own node-local Triton cache (see compile_cache_dirs)."""
    local = "/tmp" if os.access("/tmp", os.W_OK) else None
    with tempfile.TemporaryDirectory(prefix="trc_triton_probe_", dir=local) as cache:
        proc = subprocess.run([sys.executable, "-c", GPU_PROBE], capture_output=True, text=True, timeout=900,
                              env=dict(os.environ, TRITON_CACHE_DIR=cache))
    line = next((l for l in proc.stdout.splitlines() if l.startswith("GPU_PROBE ")), None)
    if proc.returncode != 0 or line is None:
        raise ConfigError(f"GPU kernel probe failed (rc {proc.returncode}): {proc.stderr[-2000:]}")
    rows = json.loads(line[len("GPU_PROBE "):])
    bad = [r for r in rows if not r["flash_attn_ok"] or r["triton_ce_max_err"] > 1e-3 or not r["arch_in_torch"]]
    if len(rows) != n_gpus or bad:
        raise ConfigError(f"GPU kernel probe: {rows}")
    return {"devices": rows}


# ===================================================================================== check (CPU, cached)


def run_check(spec: Mapping[str, Any], paths: Paths, *, model_path: Path, model_kind: str, n_gpus: Optional[int],
              arm: Optional[str] = None, seed: int = 1, gpu: bool = False, skip_clip: bool = False) -> Dict[str, Any]:
    """Everything that can fail before a GPU is touched (DESIGN 7.2 `train_rl.py check`)."""
    t0 = time.time()
    cache = ShaCache(paths.cache / "sha256.json")
    report: Dict[str, Any] = {"model": spec["model"], "config_id": spec["config_id"], "spec_sha256": spec["_sha256"], "when": now_iso()}
    env = check_environment(paths)
    report["environment"] = env
    if env["errors"]:
        raise ConfigError("environment: " + "; ".join(env["errors"]))
    for warning in env["warnings"]:
        log(f"warning: {warning}")
    log(f"environment ok: python {env['python']}, torch {env['torch'].get('version')}, vllm {env['versions'].get('vllm')}, "
        f"transformers {env['versions'].get('transformers')}, flash-attn {env['flash_attn'].get('version')}, "
        f"rfc8785 canonical active, trace_tasks from {env['trace_tasks']}")
    report["release"] = check_release(spec, paths)
    log("trace-rlvr release check ok (vendor manifest: 64 EasyR1 files byte-identical; release inventory, prompt, receipts)")
    release = release_easyr1(spec, paths)
    report["data"] = check_data(spec, paths, cache)
    log(f"data ok: train {report['data']['train']['sha256'][:12]} ({report['data']['train']['rows']} rows), "
        f"val {report['data']['val']['sha256'][:12]} ({report['data']['val']['rows']} rows), store {report['data']['store']['sha256'][:12]}")
    report["join"] = check_store_join(spec, paths, report["data"])
    log(f"store join ok: {report['join']['train']['rows']} train + {report['join']['val']['rows']} val rows resolve with matching instance_id")
    report["reward"] = check_reward(spec, release, paths, report["data"])
    log(f"reward ok: oracle SFT targets of {report['reward']['train']['rows']} train + {report['reward']['val']['rows']} val rows "
        f"score A = M = F = 1 in every arm through reward_entry.py (mean overall train {report['reward']['train']})")
    report["model_dir"] = check_model_dir(spec, model_path, kind=model_kind)
    log(f"model dir ok ({model_kind}): {model_path} ({report['model_dir']['weights']['numel'] / 1e9:.2f}B params)")
    if not skip_clip:
        report["prompt_clip"] = check_prompt_clip(spec, release, paths, report["data"], model_path)
        clip = report["prompt_clip"]
        log(f"prompt clip ok: longest prompt {clip['max']} tokens <= {clip['max_prompt_length']} (margin {clip['margin']}; "
            f"train max {clip['train']['max']}, val max {clip['val']['max']}){' [cached]' if clip.get('cached') else ''}")
    # every arm resolves, loads into EasyR1's dataclasses, and the arms differ only in reward kwargs
    ns = [n_gpus] if n_gpus else list(spec["hardware"]["n_gpus"])
    configs = {}
    for n in ns:
        for each in ARMS:
            configs[(n, each)] = resolve_config(spec, release, arm=each, seed=seed, n_gpus=n, model_path=str(model_path),
                                                run_dir=str(paths.run_dir(spec, each, seed)), data_root=str(paths.data),
                                                store_sha256=report["data"]["store"]["sha256"])
        diff = arm_diff({a: configs[(n, a)] for a in ARMS})
        allowed = {"worker.reward.reward_function_kwargs.arm", "worker.reward.reward_function_kwargs.gated"}
        if not set(diff["differing_keys"]) <= allowed:
            raise ConfigError(f"arms differ outside the reward kwargs: {diff['differing_keys']}")
        report.setdefault("arm_diff", {})[str(n)] = diff
        release_diff = set(config_diff(release, configs[(n, "P")]))
        extra = {key for key in release_diff if not key_allowed(key, allowed_release_diff(spec))}
        if extra:
            raise ConfigError(f"resolved config changes release keys outside DESIGN 7.3: {sorted(extra)}")
        report.setdefault("release_diff", {})[str(n)] = sorted(release_diff)
    easyr1_structured(configs[(ns[0], arm or "P")], paths.backend)
    log(f"configs ok: {len(configs)} resolved (arms {', '.join(ARMS)} x N {ns}); arms differ only in reward kwargs "
        f"{sorted(report['arm_diff'][str(ns[0])]['differing_keys'])}; EasyR1 PPOConfig accepts them")
    if gpu:
        if not n_gpus:
            raise ConfigError("--gpu needs --n-gpus")
        report["hardware"] = check_hardware(spec, gpu_inventory(), n_gpus)
        report["gpu_kernels"] = check_gpu_kernels(n_gpus)
        log(f"GPU ok: {n_gpus} x {report['hardware']['name']}; flash-attn varlen + triton CE run on every device "
            f"({', '.join(d['capability'] for d in report['gpu_kernels']['devices'])})")
    report["seconds"] = round(time.time() - t0, 1)
    report["ok"] = True
    return report


# ===================================================================================== run directory bookkeeping


def identity_of(spec: Mapping[str, Any], *, arm: str, seed: int, n_gpus: int, model_info: Mapping[str, Any],
                data: Mapping[str, Any], config: Mapping[str, Any], smoke: bool) -> Dict[str, Any]:
    """What must not change between attempts of one run (a resume with any change is refused)."""
    stable = copy.deepcopy(dict(config))
    stable["trainer"].pop("save_freq", None)  # may follow the preemptible save rule per attempt
    if smoke:
        stable["trainer"].pop("max_steps", None)
    return {
        "config_id": spec["config_id"],
        "arm": arm,
        "seed": seed,
        "n_gpus": n_gpus,
        "smoke": smoke,
        "model_config_sha256": model_info["config_sha256"],
        "model_weights": {"numel": model_info["weights"]["numel"], "files": model_info["weights"]["files"]},
        "train_sha256": data["train"]["sha256"],
        "val_sha256": data["val"]["sha256"],
        "store_sha256": data["store"]["sha256"],
        "resolved_config_sha256_wo_save_freq": canonical_sha(stable),
    }


def tracker_step(ckpt_dir: Path) -> int:
    path = ckpt_dir / TRACKER
    if not path.is_file():
        return 0
    try:
        return int(read_json(path).get("last_global_step", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def step_complete(step_dir: Path, n_gpus: int) -> bool:
    actor = step_dir / "actor"
    shards = [actor / f"model_world_size_{n_gpus}_rank_{rank}.pt" for rank in range(n_gpus)]
    return all(p.is_file() and p.stat().st_size > 0 for p in shards) and (actor / "huggingface" / "config.json").is_file()


def link_keep(run_dir: Path, steps: Iterable[int], n_gpus: int) -> List[int]:
    """Hard-link ckpt/global_step_<k> into keep/ once its save is complete (DESIGN 7.3 retention bug).

    A save is complete when the tracker reports a step >= k (it is written after every rank saved)
    and every model shard plus the HF config exist. ``cp -al`` gives new directory entries for the
    same inodes, so EasyR1's later ``rmtree`` of the original leaves keep/ intact.
    """
    ckpt, keep = run_dir / "ckpt", run_dir / "keep"
    last = tracker_step(ckpt)
    linked = []
    for step in steps:
        dest = keep / f"global_step_{step}"
        src = ckpt / f"global_step_{step}"
        if dest.is_dir() or last < step or not src.is_dir() or not step_complete(src, n_gpus):
            continue
        keep.mkdir(parents=True, exist_ok=True)
        for stale in keep.glob(f".global_step_{step}.tmp.*"):  # left by a killed runner
            shutil.rmtree(stale, ignore_errors=True)
        tmp = keep / f".global_step_{step}.tmp.{os.getpid()}"
        subprocess.run(["cp", "-al", str(src), str(tmp)], check=True)
        if not step_complete(tmp, n_gpus):
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        os.replace(tmp, dest)
        linked.append(step)
        log(f"keep: hard-linked global_step_{step} -> {dest}")
    return linked


def save_freq_for(spec: Mapping[str, Any], host: str, step_seconds: Optional[float]) -> Tuple[int, str]:
    """25, or on preemptible nodes the largest divisor of 100 with a save every <= 45 min (DESIGN 7.3)."""
    rule = spec["schedule"]["preemptible"]
    base = int(spec["schedule"]["save_freq"])
    if not re.search(rule["nodes"], host):
        return base, "default"
    seconds = step_seconds or float(rule["default_step_seconds"])
    budget = float(rule["max_minutes_between_saves"]) * 60.0
    for choice in sorted(rule["save_freq_choices"], reverse=True):
        if choice <= base and choice * seconds <= budget:
            return int(choice), f"preemptible {host}: step {seconds:.0f}s ({'measured' if step_seconds else 'default'})"
    return 1, f"preemptible {host}: step {seconds:.0f}s exceeds the save budget"


# ===================================================================================== experiment logs


def experiment_records(run_dir: Path) -> List[Dict[str, Any]]:
    """Every logged step across attempts (logs/experiment_log.attempt*.jsonl, then ckpt/experiment_log.jsonl).
    EasyR1 writes a validation step as a second record with the same step number, so the records of one
    step within one file are merged (train metrics + val_*); a later attempt's step replaces an earlier one."""
    files = sorted((run_dir / "logs").glob("experiment_log.attempt*.jsonl"),
                   key=lambda p: int(re.search(r"attempt(\d+)", p.name).group(1)))
    current = run_dir / "ckpt" / "experiment_log.jsonl"
    if current.is_file():
        files.append(current)
    steps: Dict[int, Dict[str, Any]] = {}
    for path in files:
        in_file: Dict[int, Dict[str, Any]] = {}
        with contextlib.suppress(OSError):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                with contextlib.suppress(json.JSONDecodeError, KeyError, TypeError, ValueError):
                    record = json.loads(line)
                    in_file.setdefault(int(record["step"]), {}).update(record)
        steps.update(in_file)
    return [steps[k] for k in sorted(steps)]


def flat_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return flatten(record)


def measured_step_seconds(run_dir: Path) -> Optional[float]:
    values = [flat_record(r).get("timing_s.step") for r in experiment_records(run_dir)]
    values = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    return statistics.median(values) if values else None


STEP_FIELDS = (
    ("timing_s.step", "step_s"), ("timing_s.gen", "gen_s"), ("timing_s.reward", "reward_s"), ("timing_s.old", "old_s"),
    ("timing_s.update_actor", "update_s"), ("timing_s.save_checkpoint", "save_s"), ("timing_s.validation", "val_s"),
    ("reward.overall", "R"), ("reward.accuracy", "A"), ("reward.grounding", "G"), ("reward.semantic", "M"),
    ("reward.cert_format", "F"), ("reward.joint", "joint"), ("reward.b_term", "B"), ("reward.verifier_ms", "verifier_ms"),
    ("response_length.mean", "resp_len"), ("response_length.clip_ratio", "resp_clip"), ("prompt_length.max", "prompt_max"),
    ("prompt_length.clip_ratio", "prompt_clip"), ("perf.max_memory_allocated_gb", "gpu_alloc_gb"),
    ("perf.max_memory_reserved_gb", "gpu_reserved_gb"), ("perf.cpu_memory_used_gb", "host_gb"), ("actor.pg_loss", "pg_loss"),
    ("actor.grad_norm", "grad_norm"), ("rlvr_stats.zero_solve_rate", "zero_solve"),
)


def step_summary(record: Mapping[str, Any]) -> Dict[str, Any]:
    flat = flat_record(record)
    out: Dict[str, Any] = {"step": record.get("step")}
    for key, name in STEP_FIELDS:
        if key in flat and isinstance(flat[key], (int, float)):
            out[name] = flat[key]
    val = {k[len("val."):]: v for k, v in flat.items() if k.startswith("val.") and isinstance(v, (int, float))}
    if val:
        out["val"] = {k: v for k, v in val.items() if k in ("reward_score", "reward.overall", "reward.accuracy", "reward.grounding",
                                                            "reward.semantic", "reward.cert_format", "reward.joint")}
    return out


def format_step(summary: Mapping[str, Any], max_steps: int) -> str:
    def f(name: str, fmt: str = "{:.3f}") -> str:
        value = summary.get(name)
        return fmt.format(value) if isinstance(value, (int, float)) else "-"

    text = (f"step {summary.get('step')}/{max_steps} {f('step_s', '{:.0f}')}s (gen {f('gen_s', '{:.0f}')} upd {f('update_s', '{:.0f}')}) "
            f"R {f('R')} A {f('A')} G {f('G')} M {f('M')} F {f('F')} resp {f('resp_len', '{:.0f}')} clip {f('resp_clip')} "
            f"prompt_clip {f('prompt_clip')} mem {f('gpu_alloc_gb', '{:.1f}')}/{f('gpu_reserved_gb', '{:.1f}')}GB")
    if summary.get("val"):
        text += " val " + " ".join(f"{k}={v:.3f}" for k, v in summary["val"].items())
    return text


def details_summary(run_dir: Path, phase: Optional[str] = None, limit_files: Optional[int] = None) -> Dict[str, Any]:
    """Per-adapter A/G/M/F and the overall every arm WOULD give the same rollouts (from reward_details)."""
    root = run_dir / "reward_details"
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else []
    if phase:
        files = [p for p in files if p.name.startswith(phase + "_")]
    if limit_files:
        files = files[-limit_files:]
    agg: Dict[str, Dict[str, List[float]]] = {}
    errors: Dict[str, int] = {}
    n = 0
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            with contextlib.suppress(Exception):
                rec = json.loads(line)
                n += 1
                for group in ("all", str(rec.get("adapter_id"))):
                    bucket = agg.setdefault(group, {})
                    for key in ("A", "G", "M", "F", "joint"):
                        bucket.setdefault(key, []).append(float(rec[key]))
                    for arm, value in (rec.get("overall") or {}).items():
                        bucket.setdefault(f"overall[{arm}]", []).append(float(value))
                    bucket.setdefault("verifier_ms", []).append(float((rec.get("diagnostics") or {}).get("verifier_ms", 0.0)))
                errors[str(rec.get("error_class"))] = errors.get(str(rec.get("error_class")), 0) + 1
    return {
        "rows": n,
        "files": len(files),
        "by_adapter": {g: {k: round(sum(v) / len(v), 4) for k, v in b.items()} for g, b in agg.items()},
        "error_class": errors,
    }


# ===================================================================================== host memory (cgroup)
# Slurm enforces each allocation's host-RAM limit through the job's memory cgroup, and EasyR1 with the
# release offload settings (actor params + optimizer offloaded, ref model CPU-offloaded) keeps most of the
# model state in host RAM: a 120 GB 2 x A100 allocation was OOM-killed during the first smoke update.

_UNLIMITED = 1 << 60


def _memory_cgroup() -> Optional[Tuple[Path, str]]:
    """(directory, "v1" | "v2") of this process's memory cgroup, or None."""
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    v2 = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if "memory" in parts[1].split(","):
            path = Path("/sys/fs/cgroup/memory") / parts[2].lstrip("/")
            if path.is_dir():
                return path, "v1"
        elif parts[0] == "0" and parts[1] == "":
            v2 = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
    return (v2, "v2") if v2 is not None and v2.is_dir() else None


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None  # missing, unreadable or "max"


def host_memory() -> Dict[str, Any]:
    """Host RAM of this job step in GB: the tightest cgroup limit on the path to the root, the
    unreclaimable use (rss/anon + shmem: what remains once the page cache is dropped, i.e. what the
    OOM killer counts) and the cgroup's own peak (which includes page cache). None where unknown."""
    out: Dict[str, Any] = {"cgroup": None, "limit_gb": None, "used_gb": None, "peak_gb": None}
    found = _memory_cgroup()
    if not found:
        return out
    base, version = found
    out["cgroup"] = f"{version}:{base}"
    root = Path("/sys/fs/cgroup/memory") if version == "v1" else Path("/sys/fs/cgroup")
    limits, node = [], base
    while True:
        value = _read_int(node / ("memory.limit_in_bytes" if version == "v1" else "memory.max"))
        if value is not None and value < _UNLIMITED:
            limits.append(value)
        if node == root or node.parent == node or root not in node.parents:
            break
        node = node.parent
    gb = float(1024 ** 3)
    if limits:
        out["limit_gb"] = round(min(limits) / gb, 1)
    stat: Dict[str, int] = {}
    with contextlib.suppress(OSError):
        for line in (base / "memory.stat").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            with contextlib.suppress(ValueError):
                stat[key] = int(value)
    if stat:
        used = (stat.get("total_rss", 0) + stat.get("total_shmem", 0)) if version == "v1" else (stat.get("anon", 0) + stat.get("shmem", 0))
        out["used_gb"] = round(used / gb, 1)
    peak = _read_int(base / ("memory.max_usage_in_bytes" if version == "v1" else "memory.peak"))
    if peak:
        out["peak_gb"] = round(peak / gb, 1)
    return out


def host_ram_share(env: Optional[Mapping[str, str]] = None) -> int:
    """How many runs share this allocation's memory cgroup: 1, or 2 under jobs/rl_pair.sh
    (TRC_RL_HOST_RAM_SHARE). Every run of a pair gets an equal share of the limit."""
    env = os.environ if env is None else env
    raw = (env.get("TRC_RL_HOST_RAM_SHARE") or "1").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= 8:
        raise ConfigError(f"TRC_RL_HOST_RAM_SHARE={raw!r} must be an integer in 1..8")
    return int(raw)


def check_host_ram(spec: Mapping[str, Any], n_gpus: int) -> Dict[str, Any]:
    """Refuse an allocation whose host-RAM limit, divided by the number of runs sharing it
    (TRC_RL_HOST_RAM_SHARE, pair mode: 2), is below the spec's hardware.min_host_ram_gb[N]
    (TRC_RL_MIN_HOST_RAM_GB overrides it; 0 disables the gate)."""
    info = host_memory()
    need = os.environ.get("TRC_RL_MIN_HOST_RAM_GB")
    need_gb = float(need) if need not in (None, "") else float((spec["hardware"].get("min_host_ram_gb") or {}).get(n_gpus, 0))
    share = host_ram_share()
    info["min_gb"] = need_gb
    info["share"] = share
    info["share_gb"] = round(info["limit_gb"] / share, 1) if info["limit_gb"] is not None else None
    if need_gb > 0 and info["share_gb"] is not None and info["share_gb"] < need_gb:
        shared = f", shared by {share} concurrent runs: {info['share_gb']} GB each" if share > 1 else ""
        raise ConfigError(f"{spec['model']} RL on {n_gpus} GPUs needs >= {need_gb:.0f} GB host RAM (EasyR1 release offload "
                          f"settings); this allocation's memory cgroup allows {info['limit_gb']} GB{shared} ({info['cgroup']})")
    return info


# ===================================================================================== GPU memory sampler


class GpuSampler(threading.Thread):
    """nvidia-smi memory/utilisation every ``period`` s into a TSV; tracks the per-GPU peak."""

    def __init__(self, path: Path, period: float = 30.0, host_path: Optional[Path] = None) -> None:
        super().__init__(daemon=True)
        self.path, self.period, self.host_path = path, period, host_path
        self.peak: Dict[str, int] = {}
        self.host_peak: Dict[str, Any] = {}
        self.stop_event = threading.Event()
        self.smi = os.environ.get("TRC_NVIDIA_SMI", "nvidia-smi")
        self.visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]

    def sample(self) -> List[Tuple[str, int, int]]:
        try:
            out = subprocess.run([self.smi, "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            if self.visible and parts[0] not in self.visible and parts[1] not in self.visible:
                continue
            with contextlib.suppress(ValueError):
                rows.append((parts[0], int(float(parts[2])), int(float(parts[3]))))
        return rows

    def run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                stamp = int(time.time())
                for index, used, util in self.sample():
                    self.peak[index] = max(self.peak.get(index, 0), used)
                    handle.write(f"{stamp}\tgpu{index}\t{used}\t{util}\n")
                handle.flush()
                if self.host_path is not None:
                    host = host_memory()
                    for key in ("used_gb", "peak_gb"):
                        if host.get(key) is not None:
                            self.host_peak[key] = max(self.host_peak.get(key) or 0.0, host[key])
                    self.host_peak["limit_gb"] = host.get("limit_gb")
                    with self.host_path.open("a", encoding="utf-8") as host_handle:
                        host_handle.write(f"{stamp}\t{host.get('used_gb')}\t{host.get('peak_gb')}\t{host.get('limit_gb')}\n")
                self.stop_event.wait(self.period)

    def stop(self) -> None:
        self.stop_event.set()


# ===================================================================================== launching EasyR1


def pair_slot(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The pair slot ("a" / "b") jobs/rl_pair.sh gives each of its two runs (TRC_RL_SLOT); None for a single run."""
    env = os.environ if env is None else env
    slot = (env.get("TRC_RL_SLOT") or "").strip()
    if not slot:
        return None
    if not re.fullmatch(r"[a-z]", slot):
        raise ConfigError(f"TRC_RL_SLOT={slot!r} must be one lowercase letter")
    return slot


def ray_tmp_dir(job: str, pid: int, slot: Optional[str] = None, base: Path = Path("/tmp")) -> Path:
    """This launch's Ray temp dir: short (AF_UNIX socket paths are limited to 107 bytes) and unique per
    launch (job, trainer pid, pair slot). Ray only auto-connects to the cluster recorded under its own
    temp dir, so two launches on one node always start two separate Ray instances."""
    job = re.sub(r"[^0-9A-Za-z]", "", job)[:12] or "local"
    return base / (f"trc_ray_{job}_{pid}" + (f"_{slot}" if slot else ""))


def placement(hardware: Mapping[str, Any]) -> Dict[str, Any]:
    """Where an attempt runs: hardware class (GPU type x N), single or pair mode, the narrowed device mask."""
    slot = pair_slot()
    return {"gpu_class": f"{hardware['name']} x{hardware['n']}", "n_gpus": hardware["n"],
            "mode": "pair" if slot else "single", "slot": slot, "pair": os.environ.get("TRC_RL_PAIR") or None,
            "host_ram_share": host_ram_share(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def launch_env(paths: Paths, run_dir: Path, ray_tmp: Path, n_gpus: int, reward_cpus: int = 4) -> Dict[str, str]:
    env = dict(os.environ)
    python_paths = [str(paths.backend), str(TRACE_ROOT / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"      # the backend checkout stays byte-identical
    env["TOKENIZERS_PARALLELISM"] = "true"   # as rlvr/train.py
    env["RAY_TMPDIR"] = str(ray_tmp)         # short: AF_UNIX socket paths are limited to 107 bytes
    env["WANDB_MODE"] = "disabled"
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
    env["HF_DATASETS_CACHE"] = str(run_dir / "hf_datasets_cache")  # per run: no NFS lock races between arms
    env["TQDM_MININTERVAL"] = env.get("TQDM_MININTERVAL", "30")
    for key in ("RAY_ADDRESS", "WANDB_NAME", "WANDB_PROJECT", "WANDB_RESUME", "WANDB_RUN_ID", "USE_MODELSCOPE_HUB"):
        env.pop(key, None)
    # Per-launch, node-local compile caches. The defaults (~/.triton/cache, ~/.cache/vllm) are one shared NFS
    # directory for every node: two launches compiling the same kernel at once (a pair on one node, or runs on
    # two nodes) race on its renames. The first sm_120 smokes died on it (triton metadata read: OSError 116,
    # stale file handle), as sft.sh's generation shards did on the vLLM cache. Training math is unaffected.
    for key, path in compile_cache_dirs(ray_tmp).items():
        env[key] = str(path)
    needed = 1 + 2 * reward_cpus + n_gpus  # Runner + train/val reward actors + one CPU per GPU bundle
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    if available < needed:
        # Ray counts the step's cpuset; its CPU resources are logical (the scorer is single-threaded),
        # so let Ray see the node's CPU count instead of leaving actors unschedulable.
        env["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1"
        flag("cpu_oversubscribed", f"{available} CPUs in this step's cpuset < {needed} logical CPUs EasyR1 asks Ray for")
    return env


def vllm_cache_dir(ray_tmp: Path) -> Path:
    """Node-local VLLM_CACHE_ROOT of one launch (next to its Ray temp dir; removed after it)."""
    return ray_tmp.parent / ray_tmp.name.replace("trc_ray_", "trc_vllm_", 1)


def compile_cache_dirs(ray_tmp: Path) -> Dict[str, Path]:
    """VLLM_CACHE_ROOT and TRITON_CACHE_DIR of one launch: node-local, unique, removed after the attempt."""
    return {"VLLM_CACHE_ROOT": vllm_cache_dir(ray_tmp),
            "TRITON_CACHE_DIR": ray_tmp.parent / ray_tmp.name.replace("trc_ray_", "trc_triton_", 1)}


def ray_session_pids(ray_tmp: Path, proc: Path = Path("/proc")) -> List[int]:
    """Our processes whose command line names this launch's Ray temp dir as a whole path component
    (the dir itself or something under it). /tmp/trc_ray_<job>_12 never matches /tmp/trc_ray_<job>_123,
    so a second Ray instance on the same node (pair mode) is never touched."""
    target = str(ray_tmp).rstrip("/")
    me, uid, found = os.getpid(), os.getuid(), []
    try:
        entries = list(os.scandir(proc))
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            raw = (Path(entry.path) / "cmdline").read_bytes()
        except OSError:
            continue
        for arg in raw.decode("utf-8", errors="replace").split("\0"):
            start = arg.find(target)
            while start != -1:
                end = start + len(target)
                if end == len(arg) or arg[end] == "/":
                    found.append(int(entry.name))
                    break
                start = arg.find(target, start + 1)
            else:
                continue
            break
    return found


def ray_listen_ports(ray_tmp: Path, extra_pids: Iterable[int] = ()) -> Dict[str, List[int]]:
    """TCP ports the processes of this launch's Ray session (plus ``extra_pids``, e.g. the driver) listen on,
    by process name, from ``ss -ltnpH``. Logged once per attempt: in pair mode the two runs' sets must be
    disjoint (every Ray port is OS-assigned except the dashboard's, which retries upward)."""
    pids = set(ray_session_pids(ray_tmp)) | set(extra_pids)
    if not pids:
        return {}
    try:
        out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    ports: Dict[str, set] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        port = parts[3].rsplit(":", 1)[-1]
        for name, pid in re.findall(r'\("([^"]+)",pid=(\d+)', line):
            if int(pid) in pids and port.isdigit():
                ports.setdefault(name, set()).add(int(port))
    return {name: sorted(values) for name, values in sorted(ports.items())}


def kill_ray(ray_tmp: Path) -> None:
    """Stop anything still running from this launch's Ray session (raylet, gcs, workers): TERM, then KILL."""
    for sig, pause in ((signal.SIGTERM, 5), (signal.SIGKILL, 1)):
        pids = ray_session_pids(ray_tmp)
        if not pids:
            break
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        time.sleep(pause)


class Trainer:
    """One EasyR1 launch with keep-linking, progress lines, GPU sampling and signal forwarding."""

    def __init__(self, paths: Paths, run_dir: Path, config_path: Path, attempt: int, *, keep_steps: Sequence[int],
                 n_gpus: int, max_steps: int, stall_minutes: float, poll_seconds: float = 30.0) -> None:
        self.paths, self.run_dir, self.config_path, self.attempt = paths, run_dir, config_path, attempt
        self.poll_seconds = poll_seconds
        self.keep_steps, self.n_gpus, self.max_steps = list(keep_steps), n_gpus, max_steps
        self.stall_seconds = stall_minutes * 60
        self.proc: Optional[subprocess.Popen] = None
        self.signalled: Optional[int] = None
        self.signalled_at = 0.0
        self.gpu_peak: Dict[str, int] = {}
        self.host_peak: Dict[str, Any] = {}
        self.ray_ports: Dict[str, List[int]] = {}
        self.ray_tmp = ray_tmp_dir(os.environ.get("SLURM_JOB_ID", "local"), os.getpid(), pair_slot())

    def _forward(self, signum: int, _frame: Any) -> None:
        self.signalled = signum
        self.signalled_at = time.time()
        log(f"signal {signal.Signals(signum).name}: stopping the trainer")
        if self.proc and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)

    def run(self) -> int:
        logs = self.run_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        # FileLogger truncates ckpt/experiment_log.jsonl at every start: keep the previous attempt's
        current = self.run_dir / "ckpt" / "experiment_log.jsonl"
        if current.is_file() and current.stat().st_size > 0:
            os.replace(current, logs / f"experiment_log.attempt{self.attempt - 1}.jsonl")
        gens = self.run_dir / "ckpt" / "generations.log"
        if gens.is_file() and gens.stat().st_size > 0:
            os.replace(gens, logs / f"generations.attempt{self.attempt - 1}.log")
        self.ray_tmp.mkdir(parents=True, exist_ok=True)
        env = launch_env(self.paths, self.run_dir, self.ray_tmp, self.n_gpus)
        cmd = [sys.executable, "-m", "verl.trainer.main", f"config={self.config_path}"]
        out_log = logs / f"trainer.attempt{self.attempt}.log"
        log(f"launch attempt {self.attempt}: {' '.join(cmd)} (cwd {self.paths.backend}, RAY_TMPDIR {self.ray_tmp}, log {out_log})")
        sampler = GpuSampler(logs / f"gpu_mem.attempt{self.attempt}.tsv", host_path=logs / f"host_mem.attempt{self.attempt}.tsv")
        sampler.start()
        old_term = signal.signal(signal.SIGTERM, self._forward)
        old_int = signal.signal(signal.SIGINT, self._forward)
        old_hup = signal.signal(signal.SIGHUP, self._forward)
        reported = 0  # lines of this attempt's ckpt/experiment_log.jsonl already reported
        last_step: Optional[int] = None
        last_progress = time.time()
        started = time.time()
        util_checked = False
        ports_logged = False
        rc = None
        try:
            with out_log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n==== {now_iso()} attempt {self.attempt} {' '.join(cmd)}\n")
                handle.flush()
                self.proc = subprocess.Popen(cmd, cwd=str(self.paths.backend), env=env, stdout=handle, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                while True:
                    rc = self.proc.poll()
                    link_keep(self.run_dir, self.keep_steps + [self.max_steps], self.n_gpus)
                    text = current.read_text(encoding="utf-8", errors="replace") if current.is_file() else ""
                    lines = text.splitlines()
                    if lines and not text.endswith("\n"):
                        lines = lines[:-1]  # a line still being written
                    if len(lines) < reported:
                        reported = 0  # truncated by a (re)started FileLogger
                    for line in lines[reported:]:
                        with contextlib.suppress(json.JSONDecodeError, KeyError, TypeError, ValueError):
                            record = json.loads(line)
                            summary = step_summary(record)
                            last_step = int(record["step"])
                            log(format_step(summary, self.max_steps))
                            self._flags(summary)
                            last_progress = time.time()
                    reported = len(lines)
                    write_json(self.run_dir / "progress.json", {
                        "updated": now_iso(), "attempt": self.attempt, "tracker_step": tracker_step(self.run_dir / "ckpt"),
                        "last_logged_step": last_step, "max_steps": self.max_steps, "pid": self.proc.pid,
                        "host": short_host(), "gpu_peak_mib": sampler.peak, "host_ram_gb": sampler.host_peak,
                        "placement": {"slot": pair_slot(), "pair": os.environ.get("TRC_RL_PAIR") or None,
                                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                        "ray_tmp": str(self.ray_tmp), "ray_ports": self.ray_ports,
                    })
                    if rc is not None:
                        break
                    if self.signalled and time.time() - self.signalled_at > 90:
                        log("trainer did not stop within 90 s of the signal; killing it")
                        break
                    if not ports_logged and time.time() - started > 180:
                        ports_logged = True
                        with contextlib.suppress(Exception):
                            self.ray_ports = ray_listen_ports(self.ray_tmp, [self.proc.pid])
                            log(f"ray ports ({pair_slot() or 'single'}, RAY_TMPDIR {self.ray_tmp}): {json.dumps(self.ray_ports)}")
                    if not util_checked and time.time() - started > 1200:
                        util_checked = True
                        if sampler.peak and max(sampler.peak.values()) < 10_000:
                            flag("util_low", f"after 20 min the GPUs peak at {sampler.peak} MiB")
                    if time.time() - last_progress > self.stall_seconds:
                        flag("stall", f"no new step for {int((time.time() - last_progress) / 60)} min (attempt {self.attempt}, log {out_log})")
                        last_progress = time.time()
                    deadline = time.time() + self.poll_seconds
                    while time.time() < deadline and self.proc.poll() is None and not self.signalled:
                        time.sleep(min(1.0, self.poll_seconds))
        finally:
            sampler.stop()
            if self.proc and self.proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=90)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                rc = self.proc.wait()
            kill_ray(self.ray_tmp)
            link_keep(self.run_dir, self.keep_steps + [self.max_steps], self.n_gpus)
            if rc not in (0, None):
                session_logs = self.ray_tmp / "session_latest" / "logs"
                if session_logs.is_dir():
                    with contextlib.suppress(Exception):
                        shutil.copytree(session_logs, logs / f"ray.attempt{self.attempt}", dirs_exist_ok=True)
            shutil.rmtree(self.ray_tmp, ignore_errors=True)
            for cache in compile_cache_dirs(self.ray_tmp).values():
                shutil.rmtree(cache, ignore_errors=True)
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGHUP, old_hup)
        self.gpu_peak = dict(sampler.peak)
        self.host_peak = dict(sampler.host_peak)
        log(f"trainer attempt {self.attempt} exited rc={rc}; tracker step {tracker_step(self.run_dir / 'ckpt')}; GPU peak MiB {self.gpu_peak}; "
            f"host RAM GB (cgroup, 30 s samples) {self.host_peak}")
        if self.signalled:
            return 128 + self.signalled
        return int(rc or 0)

    def _flags(self, summary: Mapping[str, Any]) -> None:
        for key in ("pg_loss", "grad_norm", "R"):
            value = summary.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                flag("nan", f"{key}={value} at step {summary.get('step')}")
        if isinstance(summary.get("prompt_clip"), (int, float)) and summary["prompt_clip"] > 0:
            flag("prompt_clip", f"prompt_length/clip_ratio={summary['prompt_clip']} at step {summary.get('step')}")


# ===================================================================================== merge


def merge_step(spec: Mapping[str, Any], paths: Paths, run_dir: Path, step: int, *, n_gpus: int, model_path: Path) -> Dict[str, Any]:
    """FSDP shards of one kept step -> bf16 HF dir hf/global_step_<k> (vendored model_merger.py, CPU).

    Processor / tokenizer / chat-template files come from the run's model dir (the SFT dir, which
    carries the base snapshot's), so a merged dir serves with exactly the base chat template.
    """
    out = run_dir / "hf" / f"global_step_{step}"
    receipt_path = out / MERGE_RECEIPT
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("status") == "ok":
            return receipt
    src = None
    for candidate in (run_dir / "keep" / f"global_step_{step}", run_dir / "ckpt" / f"global_step_{step}"):
        if candidate.is_dir() and step_complete(candidate, n_gpus):
            src = candidate
            break
    if src is None:
        raise Incomplete(f"no complete checkpoint for step {step} in keep/ or ckpt/")
    actor = src / "actor"
    worlds = {m.group(1) for p in actor.glob("model_world_size_*_rank_0.pt") if (m := re.match(r"model_world_size_(\d+)_rank_0\.pt", p.name))}
    if worlds != {str(n_gpus)}:
        raise ConfigError(f"{actor}: shard world sizes {sorted(worlds)} != N={n_gpus}")
    tmp = run_dir / "hf" / f".global_step_{step}.tmp.{os.getpid()}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    for stale in tmp.parent.glob(f".global_step_{step}.tmp.*"):  # left by a killed merge
        shutil.rmtree(stale, ignore_errors=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(paths.backend), str(TRACE_ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["CUDA_VISIBLE_DEVICES"] = ""          # CPU merge; the GPUs stay free
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.time()
    log(f"merge step {step}: {actor} -> {out}")
    with (run_dir / "logs" / f"merge.global_step_{step}.log").open("a", encoding="utf-8") as handle:
        proc = subprocess.run([sys.executable, str(paths.merger), "--local_dir", str(actor), "--output_dir", str(tmp)],
                              cwd=str(paths.backend), env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise Incomplete(f"model_merger.py failed for step {step} (rc {proc.returncode}); see logs/merge.global_step_{step}.log")
    # serve-identical processor files: drop what the trainer's processor.save_pretrained wrote, copy the model dir's
    for name in PROCESSOR_FILES:
        with contextlib.suppress(FileNotFoundError):
            (tmp / name).unlink()
    copied = []
    for name in PROCESSOR_FILES:
        if (model_path / name).is_file():
            shutil.copy2(model_path / name, tmp / name)
            copied.append(name)
    config = read_json(tmp / "config.json")
    patched = bf16_config(config)
    if patched:
        (tmp / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    merged = safetensors_summary(tmp)
    base = safetensors_summary(model_path)
    non_bf16 = {k: v for k, v in merged["dtypes"].items() if k != "BF16"}
    vocab_x_hidden = int(config.get("vocab_size", (config.get("text_config") or {}).get("vocab_size", 0))) * int(spec["base_model"]["hidden_size"])
    numel_ok = merged["numel"] in (base["numel"], base["numel"] + vocab_x_hidden, base["numel"] - vocab_x_hidden)
    problems = []
    if non_bf16:
        problems.append(f"non-bf16 tensors {non_bf16}")
    if not numel_ok:
        problems.append(f"parameter count {merged['numel']} vs model dir {base['numel']}")
    if processor_digest(tmp) != processor_digest(model_path):
        problems.append("processor files differ from the model dir")
    template_check = chat_template_equal(tmp, model_path)
    if not template_check["equal"]:
        problems.append(f"chat template renders differently: {template_check}")
    if problems:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ConfigError(f"merged step {step} failed validation: {problems}")
    receipt = {
        "schema_version": MERGE_SCHEMA,
        "status": "ok",
        "step": step,
        "source": str(src),
        "world_size": n_gpus,
        "merger_sha256": sha256_file(paths.merger),
        "processor_files_from": str(model_path),
        "processor_files": copied,
        "config_patched": patched,
        "weights": merged,
        "model_dir_weights_numel": base["numel"],
        "safetensors_sha256": {p.name: sha256_file(p) for p in sorted(tmp.glob("*.safetensors"))},
        "chat_template_check": template_check,
        "run_receipt_sha256": sha256_file(run_dir / RUN_RECEIPT) if (run_dir / RUN_RECEIPT).is_file() else None,
        "seconds": round(time.time() - t0, 1),
        "when": now_iso(),
    }
    write_json(tmp / MERGE_RECEIPT, receipt)
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    log(f"merged step {step} in {receipt['seconds']:.0f}s: {merged['numel'] / 1e9:.3f}B bf16 params, processor files from the model dir"
        f"{'; ' + '; '.join(patched) if patched else ''}")
    return receipt


def bf16_config(config: Dict[str, Any]) -> List[str]:
    """The actor trains in fp32, so its saved config says float32; the merged weights are bf16 and
    vLLM (dtype auto) must serve them as bf16, like the base model. Patches nested sub-configs too."""
    patched: List[str] = []

    def walk(node: Dict[str, Any], where: str) -> None:
        for key in ("torch_dtype", "dtype"):
            if key in node and node[key] != "bfloat16":
                patched.append(f"{where}{key}: {node[key]} -> bfloat16")
                node[key] = "bfloat16"
        for key, value in node.items():
            if isinstance(value, dict) and key.endswith("_config"):
                walk(value, f"{where}{key}.")

    walk(config, "")
    if "torch_dtype" not in config and "dtype" not in config:
        config["torch_dtype"] = "bfloat16"
        patched.append("torch_dtype: <absent> -> bfloat16")
    return patched


def chat_template_equal(a: Path, b: Path) -> Dict[str, Any]:
    from transformers import AutoProcessor  # noqa: WPS433

    messages = [{"role": "system", "content": "S"},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}]
    rendered = []
    for path in (a, b):
        processor = AutoProcessor.from_pretrained(str(path), use_fast=True)
        rendered.append(processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    return {"equal": rendered[0] == rendered[1], "sha256": sha256_bytes(rendered[1].encode())}


VLLM_PROBE = r"""
import base64, json, sys
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
model, parquet, system_file = sys.argv[1], sys.argv[2], sys.argv[3]
row = pq.ParquetFile(parquet).read_row_group(0, columns=["images", "prompt_cert", "instance_id"]).slice(0, 1).to_pylist()[0]
question = row["prompt_cert"].replace("<image>", "", 1)
url = "data:image/png;base64," + base64.b64encode(row["images"][0]["bytes"]).decode()
system = open(system_file, encoding="utf-8").read().strip()
llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.5, limit_mm_per_prompt={"image": 1, "video": 0},
          mm_processor_kwargs={"min_pixels": 262144, "max_pixels": 4194304}, generation_config="vllm", seed=0, enforce_eager=True)
messages = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}]}]
out = llm.chat(messages, SamplingParams(temperature=0.0, max_tokens=512))
text = out[0].outputs[0].text
print("VLLM_PROBE " + json.dumps({"ok": bool(text.strip()), "instance_id": row["instance_id"], "tokens": len(out[0].outputs[0].token_ids),
                                   "finish_reason": out[0].outputs[0].finish_reason, "text": text[:600]}))
"""


def _root_cause(stderr: str) -> str:
    """The distinct exception lines of a vLLM traceback (the engine core's root cause comes first)."""
    lines = [l.split(") ", 1)[-1] if l.startswith("(EngineCore") else l for l in stderr.splitlines()]
    errors = list(dict.fromkeys(l.strip() for l in lines if re.match(r"^\s*[A-Za-z_.]*(Error|Exception)\b.*:", l)))
    return " | ".join(errors[-4:]) if errors else stderr[-800:]


def vllm_check(merged_dir: Path, vllm_python: str, val_parquet: Path, gpu: str) -> Dict[str, Any]:
    """Load a merged dir with vLLM (the serving env) and decode one E1-dev row greedily."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.pop("PYTHONPATH", None)
    env["HF_HUB_OFFLINE"] = "1"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    # the serving venv's bin/ goes first on PATH (as `activate` does for serve_vllm.sh): vLLM's
    # FlashInfer / torch.compile JIT spawns `ninja`, which lives there
    venv_bin = str(Path(vllm_python).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(Path(venv_bin).parent)
    t0 = time.time()
    local = "/tmp" if os.access("/tmp", os.W_OK) else None
    with tempfile.TemporaryDirectory(prefix="trc_vllmcheck_", dir=local) as cache:
        # own node-local compile caches (see compile_cache_dirs: the shared NFS defaults race across launches)
        env["VLLM_CACHE_ROOT"] = os.path.join(cache, "vllm")
        env["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
        proc = subprocess.run([vllm_python, "-c", VLLM_PROBE, str(merged_dir), str(val_parquet), str(SYSTEM_PROMPT)],
                              capture_output=True, text=True, timeout=1800, env=env, cwd=str(merged_dir.parent))
    line = next((l for l in proc.stdout.splitlines() if l.startswith("VLLM_PROBE ")), None)
    result = json.loads(line[len("VLLM_PROBE "):]) if line else {"ok": False}
    result.update({"rc": proc.returncode, "seconds": round(time.time() - t0, 1), "python": vllm_python})
    if not result.get("ok"):
        result["stderr_tail"] = proc.stderr[-8000:]
    return result


# ===================================================================================== run / smoke


def receipt_base(spec: Mapping[str, Any], paths: Paths, check: Mapping[str, Any], *, arm: str, seed: int, mode: str,
                 hardware: Mapping[str, Any], identity: Mapping[str, Any]) -> Dict[str, Any]:
    from trace_tasks.experimental.certificates.export import cert_code_sha256  # noqa: WPS433

    return {
        "schema_version": RECEIPT_SCHEMA,
        "mode": mode,
        "model": spec["model"],
        "config_id": spec["config_id"],
        "spec": {"path": str(Path(spec["_path"]).relative_to(TRACE_ROOT)), "sha256": spec["_sha256"]},
        "arm": arm,
        "reward_kwargs": None,
        "seed": seed,
        "n_gpus": hardware["n"],
        "gpu": hardware,
        "identity": identity,
        "git": {
            "trace": {**git_info(TRACE_ROOT), "cert_code_sha256": cert_code_sha256(), "train_rl_sha256": sha256_file(Path(__file__))},
            "trace_rlvr": git_info(paths.rlvr_root),
        },
        "env": {"lock": env_lock(), "versions": check["environment"]["versions"], "python": check["environment"]["python"],
                "flash_attn": check["environment"]["flash_attn"], "torch": check["environment"]["torch"]},
        "release": {"config": str(paths.release_config), "sha256": spec["release"]["sha256"],
                    "vendor_manifest": check["release"]["vendor_manifest"], "merger_sha256": check["release"]["merger_sha256"]},
        "data": check["data"],
        "join": {k: v for k, v in check["join"].items() if k != "cached"},
        "model_dir": check["model_dir"],
        "prompt_clip": {k: check.get("prompt_clip", {}).get(k) for k in ("max", "margin", "max_prompt_length")},
        "created": now_iso(),
        "attempts": [],
        "keep": {},
        "merged": {},
        "status": "created",
    }


def prepare_run_dir(spec: Mapping[str, Any], paths: Paths, *, run_dir: Path, arm: str, seed: int, n_gpus: int, model_path: Path,
                    model_kind: str, check: Mapping[str, Any], hardware: Mapping[str, Any], mode: str, save_freq: Optional[int],
                    smoke: bool, max_steps: Optional[int] = None, val_files: Optional[str] = None,
                    allow_hw_change: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    """Resolve the config, write resolved_easyr1.yaml and create / validate run_receipt.json."""
    release = release_easyr1(spec, paths)
    config = resolve_config(spec, release, arm=arm, seed=seed, n_gpus=n_gpus, model_path=str(model_path), run_dir=str(run_dir),
                            data_root=str(paths.data), store_sha256=check["data"]["store"]["sha256"], save_freq=save_freq,
                            smoke=smoke, max_steps=max_steps, val_files=val_files)
    identity = identity_of(spec, arm=arm, seed=seed, n_gpus=n_gpus, model_info=check["model_dir"], data=check["data"],
                           config=config, smoke=smoke)
    receipt_path = run_dir / RUN_RECEIPT
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("identity") != identity:
            diff = config_diff(receipt.get("identity", {}), identity)
            raise ConfigError(f"{run_dir} belongs to a different run (identity differs in {sorted(diff)}); refusing to resume into it")
        first = receipt.get("gpu", {}).get("name")
        if first and first != hardware["name"] and not allow_hw_change:
            raise ConfigError(f"{run_dir} started on {first}, this allocation has {hardware['name']}; arms are matched per GPU type "
                              "(set TRC_RL_ALLOW_HW_CHANGE=1 to resume anyway)")
    else:
        receipt = receipt_base(spec, paths, check, arm=arm, seed=seed, mode=mode, hardware=hardware, identity=identity)
        receipt["reward_kwargs"] = config["worker"]["reward"]["reward_function_kwargs"]
    # the class all arms of a seed must share (DESIGN 7.3 Matching): GPU type and N, fixed at the first attempt
    receipt.setdefault("hardware_class", {"name": hardware["name"], "n_gpus": hardware["n"]})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    config_path = run_dir / "resolved_easyr1.yaml"
    write_yaml(config_path, config)
    receipt["resolved_config_sha256"] = sha256_file(config_path)
    receipt["release_diff"] = {k: list(v) for k, v in config_diff(release, config).items()}
    write_json(receipt_path, receipt)
    return config, receipt, config_path


def start_attempt(receipt: Dict[str, Any], run_dir: Path, *, hardware: Mapping[str, Any], save_freq: int, save_reason: str,
                  max_steps: int) -> Dict[str, Any]:
    attempt = {
        "attempt": len(receipt["attempts"]) + 1,
        "started": now_iso(),
        "host": short_host(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_step": os.environ.get("SLURM_STEP_ID"),
        "gpu": {"name": hardware["name"], "uuids": hardware["uuids"]},
        "placement": placement(hardware),
        "save_freq": save_freq,
        "save_freq_reason": save_reason,
        "max_steps": max_steps,
        "resumed_from_step": tracker_step(run_dir / "ckpt"),
        "status": "running",
    }
    receipt["attempts"].append(attempt)
    receipt["status"] = "running"
    write_json(run_dir / RUN_RECEIPT, receipt)
    return attempt


def finish_attempt(receipt: Dict[str, Any], run_dir: Path, attempt: Dict[str, Any], rc: int, gpu_peak: Mapping[str, int]) -> None:
    attempt.update({"ended": now_iso(), "rc": rc, "last_step": tracker_step(run_dir / "ckpt"), "gpu_peak_mib": dict(gpu_peak),
                    "status": "ok" if rc == 0 else ("signalled" if rc > 128 else "failed")})
    write_json(run_dir / RUN_RECEIPT, receipt)


def update_keep(receipt: Dict[str, Any], run_dir: Path, steps: Iterable[int]) -> None:
    for step in steps:
        path = run_dir / "keep" / f"global_step_{step}"
        if path.is_dir():
            receipt["keep"][str(step)] = str(path)


def check_disk(spec: Mapping[str, Any], run_dir: Path) -> None:
    """Refuse to start when the run filesystem is nearly full (a save interrupted by ENOSPC is lost)."""
    probe = run_dir
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    need = float(spec["schedule"]["min_free_disk_gb"])
    if free_gb < need:
        raise ConfigError(f"only {free_gb:.0f} GB free under {probe}; a {spec['model']} RL run needs >= {need:.0f} GB")


def do_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.model)
    paths = Paths(spec)
    model_path = Path(args.sft_dir).resolve()
    hardware = check_hardware(spec, gpu_inventory(), args.n_gpus)
    hardware["host_ram"] = check_host_ram(spec, args.n_gpus)
    log(f"host RAM: {hardware['host_ram']}")
    check = run_check(spec, paths, model_path=model_path, model_kind="sft", n_gpus=args.n_gpus, arm=args.arm, seed=args.seed,
                      gpu=True)
    run_dir = Path(args.run_dir).resolve() if args.run_dir else paths.run_dir(spec, args.arm, args.seed)
    max_steps = int(spec["overrides"]["trainer.max_steps"])
    keep_steps = [int(s) for s in spec["schedule"]["keep_steps"]]
    allow_hw = os.environ.get("TRC_RL_ALLOW_HW_CHANGE") == "1"
    if args.save_freq == "auto":
        save_freq, reason = save_freq_for(spec, short_host(), measured_step_seconds(run_dir) if run_dir.is_dir() else None)
    else:
        save_freq, reason = int(args.save_freq), "--save-freq"
        if 100 % save_freq or save_freq > int(spec["schedule"]["save_freq"]):
            raise ConfigError(f"--save-freq {save_freq} must divide 100 and be <= {spec['schedule']['save_freq']} (keep steps)")
    check_disk(spec, run_dir)
    config, receipt, config_path = prepare_run_dir(spec, paths, run_dir=run_dir, arm=args.arm, seed=args.seed, n_gpus=args.n_gpus,
                                                   model_path=model_path, model_kind="sft", check=check, hardware=hardware,
                                                   mode="run", save_freq=save_freq, smoke=False, allow_hw_change=allow_hw)
    log(f"run {run_dir}: {spec['config_id']} arm {args.arm} seed {args.seed} N={args.n_gpus} x {hardware['name']}, "
        f"save_freq {save_freq} ({reason}), reward kwargs {json.dumps({k: v for k, v in config['worker']['reward']['reward_function_kwargs'].items() if k in ('arm', 'gated', 'alpha', 'format_weight')})}")
    return train_merge_verify(spec, paths, run_dir, receipt, config_path, hardware=hardware, save_freq=save_freq, reason=reason,
                              max_steps=max_steps, keep_steps=keep_steps, model_path=model_path, n_gpus=args.n_gpus,
                              vllm_python=args.vllm_python, stall_minutes=args.stall_minutes)


def train_merge_verify(spec: Mapping[str, Any], paths: Paths, run_dir: Path, receipt: Dict[str, Any], config_path: Path, *,
                       hardware: Mapping[str, Any], save_freq: int, reason: str, max_steps: int, keep_steps: Sequence[int],
                       model_path: Path, n_gpus: int, vllm_python: Optional[str], stall_minutes: float,
                       merge_steps: Optional[Sequence[int]] = None, poll_seconds: float = 30.0) -> int:
    ckpt = run_dir / "ckpt"
    link_keep(run_dir, list(keep_steps) + [max_steps], n_gpus)
    if tracker_step(ckpt) >= max_steps and step_complete(ckpt / f"global_step_{max_steps}", n_gpus):
        log(f"training already finished (tracker step {tracker_step(ckpt)}); skipping the trainer")
    else:
        attempt = start_attempt(receipt, run_dir, hardware=hardware, save_freq=save_freq, save_reason=reason, max_steps=max_steps)
        trainer = Trainer(paths, run_dir, config_path, attempt["attempt"], keep_steps=keep_steps, n_gpus=n_gpus,
                          max_steps=max_steps, stall_minutes=stall_minutes, poll_seconds=poll_seconds)
        rc = trainer.run()
        receipt = read_json(run_dir / RUN_RECEIPT)
        attempt = receipt["attempts"][-1]
        finish_attempt(receipt, run_dir, attempt, rc, getattr(trainer, "gpu_peak", {}))
        update_keep(receipt, run_dir, keep_steps)
        write_json(run_dir / RUN_RECEIPT, receipt)
        if trainer.signalled:
            raise Incomplete(f"signalled at tracker step {tracker_step(ckpt)}; rerun resumes")
        if rc != 0 or tracker_step(ckpt) < max_steps:
            raise Incomplete(f"trainer rc={rc} at tracker step {tracker_step(ckpt)} of {max_steps}; rerun resumes from the last save")
    receipt = read_json(run_dir / RUN_RECEIPT)
    update_keep(receipt, run_dir, keep_steps)
    missing_keep = [s for s in keep_steps if not (run_dir / "keep" / f"global_step_{s}").is_dir()
                    and not step_complete(ckpt / f"global_step_{s}", n_gpus)]
    if missing_keep:
        flag("keep_lost", f"{run_dir}: steps {missing_keep} were deleted before they could be hard-linked")
    steps = list(merge_steps) if merge_steps is not None else [s for s in keep_steps if s not in missing_keep] + [max_steps]
    for step in steps:
        receipt["merged"][str(step)] = merge_step(spec, paths, run_dir, step, n_gpus=n_gpus, model_path=model_path)
        write_json(run_dir / RUN_RECEIPT, receipt)
    for step in steps if vllm_python else []:
        # every merged dir must serve: load it with the serving env's vLLM and decode one E1-dev row
        merged_dir = run_dir / "hf" / f"global_step_{step}"
        merge_receipt = read_json(merged_dir / MERGE_RECEIPT)
        if (merge_receipt.get("vllm_check") or {}).get("ok"):
            continue
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        result = vllm_check(merged_dir, vllm_python, paths.val_parquet, gpu)
        merge_receipt["vllm_check"] = result
        write_json(merged_dir / MERGE_RECEIPT, merge_receipt)
        receipt["merged"][str(step)] = merge_receipt
        write_json(run_dir / RUN_RECEIPT, receipt)
        log(f"vLLM load check of {merged_dir}: ok={result.get('ok')} ({result.get('seconds')}s, {result.get('tokens')} tokens, "
            f"finish {result.get('finish_reason')}): {str(result.get('text', ''))[:160]!r}")
        if not result.get("ok"):
            raise Incomplete(f"vLLM could not load / decode {merged_dir}: {_root_cause(result.get('stderr_tail', ''))}")
    if missing_keep:
        receipt["status"] = "incomplete_keep_lost"
        write_json(run_dir / RUN_RECEIPT, receipt)
        raise Incomplete(f"steps {missing_keep} are lost (not in keep/); the run cannot complete as pre-registered")
    receipt["status"] = "completed"
    receipt["completed"] = now_iso()
    receipt["summary"] = run_summary(run_dir, max_steps)
    write_json(run_dir / RUN_RECEIPT, receipt)
    shutil.rmtree(run_dir / "hf_datasets_cache", ignore_errors=True)
    log(f"COMPLETE {run_dir}: steps {max_steps}, merged {sorted(int(s) for s in receipt['merged'])}")
    return 0


def run_summary(run_dir: Path, max_steps: int) -> Dict[str, Any]:
    records = [step_summary(r) for r in experiment_records(run_dir)]
    train = [r for r in records if "step_s" in r]
    def mean(key: str, rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(values) / len(values), 4) if values else None
    out = {
        "logged_steps": len(train),
        "step_seconds_median": round(statistics.median([r["step_s"] for r in train]), 1) if train else None,
        "projected_hours_for_max_steps": round(statistics.median([r["step_s"] for r in train]) * max_steps / 3600, 1) if train else None,
        "gpu_alloc_gb_max": max((r.get("gpu_alloc_gb", 0) for r in train), default=None),
        "gpu_reserved_gb_max": max((r.get("gpu_reserved_gb", 0) for r in train), default=None),
        "host_gb_max": max((r.get("host_gb", 0) for r in train), default=None),
        "reward_mean": {k: mean(k, train) for k in ("R", "A", "G", "M", "F", "joint", "B", "verifier_ms", "resp_len", "resp_clip", "prompt_clip")},
        "last": train[-1] if train else None,
        "val": [dict(step=r["step"], **r["val"]) for r in records if r.get("val")],
    }
    gpu_files = sorted((run_dir / "logs").glob("gpu_mem.attempt*.tsv"))
    peak: Dict[str, int] = {}
    for path in gpu_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                with contextlib.suppress(ValueError):
                    peak[parts[1]] = max(peak.get(parts[1], 0), int(parts[2]))
    out["nvidia_smi_peak_mib"] = peak
    host: Dict[str, Any] = {}
    for path in sorted((run_dir / "logs").glob("host_mem.attempt*.tsv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            for key, value in zip(("used_gb_max", "peak_gb_max", "limit_gb"), parts[1:]):
                with contextlib.suppress(ValueError):
                    host[key] = float(value) if key == "limit_gb" else max(host.get(key, 0.0), float(value))
    out["host_ram_cgroup"] = host
    return out


def smoke_val_parquet(paths: Paths, spec: Mapping[str, Any], out: Path) -> Path:
    """A few E1-dev RL rows (per task) for the validation pass EasyR1 runs after the last step."""
    import pyarrow as pa  # noqa: WPS433
    import pyarrow.parquet as pq  # noqa: WPS433

    if out.is_file():
        return out
    per_task = int(spec["smoke"]["val_rows_per_task"])
    table = pq.read_table(paths.val_parquet)
    tasks = table.column("task").to_pylist()
    take, counts = [], {}
    for i, task in enumerate(tasks):
        if counts.get(task, 0) < per_task:
            counts[task] = counts.get(task, 0) + 1
            take.append(i)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    pq.write_table(table.take(pa.array(take)), tmp)
    os.replace(tmp, out)
    return out


def do_smoke(args: argparse.Namespace) -> int:
    """GPU smoke (DESIGN 7.2 W2a): BASE model, rollout 8 x n 4, 1 step, relaunch (resume) to step 2,
    merge, vLLM load check; prints step time, memory and reward stats per component and per arm."""
    spec = load_spec(args.model)
    paths = Paths(spec)
    model_path = paths.base_snapshot(spec)
    hardware = check_hardware(spec, gpu_inventory(), args.n_gpus)
    hardware["host_ram"] = check_host_ram(spec, args.n_gpus)
    log(f"host RAM: {hardware['host_ram']}")
    check = run_check(spec, paths, model_path=model_path, model_kind="base", n_gpus=args.n_gpus, arm=args.arm, seed=args.seed,
                      gpu=True)
    stamp = time.strftime("%m%d_%H%M%S")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (
        paths.smoke_root / f"{spec['model']}_{args.arm}_s{args.seed}" / f"{stamp}_{short_host()}")
    steps = int(args.steps or spec["smoke"]["steps"])
    check_disk(spec, run_dir)
    val = smoke_val_parquet(paths, spec, run_dir / "data" / "val_smoke.parquet")
    smoke_spec = copy.deepcopy(spec)
    if args.batch:
        smoke_spec["smoke"]["overrides"]["data.rollout_batch_size"] = args.batch
        smoke_spec["smoke"]["overrides"]["worker.actor.global_batch_size"] = args.batch
    if args.n:
        smoke_spec["smoke"]["overrides"]["worker.rollout.n"] = args.n
    t_start = time.time()
    phases = [1, steps] if steps > 1 and not args.no_resume_test else [steps]
    receipt = None
    config_path = None
    for phase_steps in phases:
        config, receipt, config_path = prepare_run_dir(smoke_spec, paths, run_dir=run_dir, arm=args.arm, seed=args.seed,
                                                       n_gpus=args.n_gpus, model_path=model_path, model_kind="base", check=check,
                                                       hardware=hardware, mode="smoke", save_freq=None, smoke=True,
                                                       max_steps=phase_steps, val_files=str(val))
        log(f"smoke phase: train to step {phase_steps} (rollout {config['data']['rollout_batch_size']} x n {config['worker']['rollout']['n']}, "
            f"N={args.n_gpus}, val {val.name}) in {run_dir}")
        attempt = start_attempt(receipt, run_dir, hardware=hardware, save_freq=int(config["trainer"]["save_freq"]),
                                save_reason="smoke", max_steps=phase_steps)
        trainer = Trainer(paths, run_dir, config_path, attempt["attempt"], keep_steps=[1], n_gpus=args.n_gpus,
                          max_steps=phase_steps, stall_minutes=args.stall_minutes)
        rc = trainer.run()
        receipt = read_json(run_dir / RUN_RECEIPT)
        finish_attempt(receipt, run_dir, receipt["attempts"][-1], rc, getattr(trainer, "gpu_peak", {}))
        if rc != 0 or tracker_step(run_dir / "ckpt") < phase_steps:
            raise Incomplete(f"smoke trainer rc={rc} at tracker step {tracker_step(run_dir / 'ckpt')} (phase to {phase_steps}); "
                             f"see {run_dir}/logs/trainer.attempt{attempt['attempt']}.log")
        if phase_steps > 1 and len(phases) > 1:
            text = (run_dir / "logs" / f"trainer.attempt{attempt['attempt']}.log").read_text(encoding="utf-8", errors="replace")
            resumed = "global_step_1" in text and "Load from checkpoint" in text
            receipt["resume_test"] = {"ok": resumed, "detail": "trainer log shows 'Load from checkpoint: .../global_step_1'"}
            write_json(run_dir / RUN_RECEIPT, receipt)
            log(f"resume test: {'ok' if resumed else 'FAILED'} (relaunch loaded global_step_1)")
            if not resumed:
                raise Incomplete("the relaunch did not resume from global_step_1")
    link_keep(run_dir, [1, steps], args.n_gpus)
    rc = train_merge_verify(smoke_spec, paths, run_dir, read_json(run_dir / RUN_RECEIPT), config_path, hardware=hardware,
                            save_freq=1, reason="smoke", max_steps=steps, keep_steps=[], model_path=model_path,
                            n_gpus=args.n_gpus, vllm_python=args.vllm_python, stall_minutes=args.stall_minutes,
                            merge_steps=[steps])
    clipped = []
    for record in experiment_records(run_dir):
        flat = flat_record(record)
        for key in ("prompt_length.clip_ratio", "val_prompt_length.clip_ratio"):
            if isinstance(flat.get(key), (int, float)) and flat[key] > 0:
                clipped.append((record["step"], key, flat[key]))
    if clipped:
        raise Incomplete(f"smoke: prompts were truncated during training: {clipped}")
    report = smoke_report(run_dir, steps, hardware, time.time() - t_start)
    receipt = read_json(run_dir / RUN_RECEIPT)
    receipt["smoke_report"] = report
    write_json(run_dir / RUN_RECEIPT, receipt)
    if not args.keep_ckpt:
        for sub in ("ckpt", "keep"):
            for step_dir in (run_dir / sub).glob("global_step_*"):
                shutil.rmtree(step_dir, ignore_errors=True)
        log("smoke: deleted the FSDP checkpoints (ckpt/, keep/); hf/ and logs stay")
    return rc


def smoke_report(run_dir: Path, steps: int, hardware: Mapping[str, Any], seconds: float) -> Dict[str, Any]:
    summary = run_summary(run_dir, steps)
    details_train = details_summary(run_dir, "train")
    details_val = details_summary(run_dir, "val")
    log(f"SMOKE {run_dir.name}: {hardware['n']} x {hardware['name']}, wall {seconds / 60:.1f} min")
    for record in experiment_records(run_dir):
        log("  " + format_step(step_summary(record), steps))
    log(f"  step time median {summary['step_seconds_median']}s (smoke batch, not the 64 x 8 production step); "
        f"GPU max_allocated {summary['gpu_alloc_gb_max']} GB, reserved {summary['gpu_reserved_gb_max']} GB, host {summary['host_gb_max']} GB, "
        f"nvidia-smi peak MiB {summary['nvidia_smi_peak_mib']}; host RAM cgroup {summary.get('host_ram_cgroup')}")
    for name, det in (("train", details_train), ("val", details_val)):
        log(f"  reward components ({name}, {det['rows']} rollouts): error classes {det['error_class']}")
        for group, values in sorted(det["by_adapter"].items()):
            arms = " ".join(f"{k[8:-1]}={v:.3f}" for k, v in values.items() if k.startswith("overall["))
            log(f"    {group:<40} A {values.get('A', 0):.3f} G {values.get('G', 0):.3f} M {values.get('M', 0):.3f} "
                f"F {values.get('F', 0):.3f} joint {values.get('joint', 0):.3f} verifier {values.get('verifier_ms', 0):.2f}ms | overall per arm: {arms}")
    return {"wall_seconds": round(seconds, 1), "summary": summary, "reward_details": {"train": details_train, "val": details_val}}


def do_merge(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    receipt = read_json(run_dir / RUN_RECEIPT)
    spec = load_spec(receipt["model"])
    paths = Paths(spec)
    model_path = Path(receipt["model_dir"]["path"])
    steps = [int(s) for s in args.steps.split(",")] if args.steps else (
        [int(s) for s in spec["schedule"]["keep_steps"]] + [int(spec["overrides"]["trainer.max_steps"])])
    link_keep(run_dir, steps, receipt["n_gpus"])
    for step in steps:
        receipt["merged"][str(step)] = merge_step(spec, paths, run_dir, step, n_gpus=receipt["n_gpus"], model_path=model_path)
        write_json(run_dir / RUN_RECEIPT, receipt)
    if args.vllm_python:
        for step in steps:
            out = run_dir / "hf" / f"global_step_{step}"
            result = vllm_check(out, args.vllm_python, paths.val_parquet, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
            merge_receipt = read_json(out / MERGE_RECEIPT)
            merge_receipt["vllm_check"] = result
            write_json(out / MERGE_RECEIPT, merge_receipt)
            log(f"vLLM load check step {step}: ok={result.get('ok')} {str(result.get('text', ''))[:120]!r}")
            if not result.get("ok"):
                raise Incomplete(f"vLLM check failed for {out}")
    return 0


def do_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    receipt_path = run_dir / RUN_RECEIPT
    status: Dict[str, Any] = {"run_dir": str(run_dir), "exists": receipt_path.is_file()}
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        status.update({"status": receipt.get("status"), "attempts": len(receipt.get("attempts", [])),
                       "tracker_step": tracker_step(run_dir / "ckpt"), "keep": sorted(receipt.get("keep", {}), key=int),
                       "merged": sorted(receipt.get("merged", {}), key=int)})
    status["complete"] = status.get("status") == "completed"
    print(json.dumps(status, sort_keys=True))
    return 0


def do_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    receipt = read_json(run_dir / RUN_RECEIPT)
    max_steps = max((a.get("max_steps", 0) for a in receipt.get("attempts", [])), default=0)
    out = {"summary": run_summary(run_dir, max_steps), "reward_details": {"train": details_summary(run_dir, "train", args.last_files),
                                                                            "val": details_summary(run_dir, "val")}}
    for record in experiment_records(run_dir)[-args.tail:]:
        log(format_step(step_summary(record), max_steps))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def do_check(args: argparse.Namespace) -> int:
    spec = load_spec(args.model)
    paths = Paths(spec)
    if args.sft_dir:
        model_path, kind = Path(args.sft_dir).resolve(), "sft"
    else:
        model_path, kind = paths.base_snapshot(spec), "base"
    report = run_check(spec, paths, model_path=model_path, model_kind=kind, n_gpus=args.n_gpus, gpu=args.gpu, skip_clip=args.skip_clip)
    if args.json:
        write_json(Path(args.json), report)
    log(f"check ok in {report['seconds']}s ({kind} model {model_path})")
    return 0


def do_prepare(args: argparse.Namespace) -> int:
    """Resolve and write the EasyR1 config(s) without GPUs; print the diff against the release block."""
    spec = load_spec(args.model)
    paths = Paths(spec)
    model_path = Path(args.sft_dir).resolve() if args.sft_dir else paths.base_snapshot(spec)
    cache = ShaCache(paths.cache / "sha256.json")
    data = check_data(spec, paths, cache)
    release = release_easyr1(spec, paths)
    out = Path(args.out).resolve()
    arms = ARMS if args.all_arms else (args.arm,)
    configs = {}
    for arm in arms:
        run_dir = out / arm / f"s{args.seed}"
        configs[arm] = resolve_config(spec, release, arm=arm, seed=args.seed, n_gpus=args.n_gpus, model_path=str(model_path),
                                      run_dir=str(run_dir), data_root=str(paths.data), store_sha256=data["store"]["sha256"],
                                      smoke=args.smoke, val_files=None)
        write_yaml(run_dir / "resolved_easyr1.yaml", configs[arm])
    diff = config_diff(release, configs[arms[0]])
    report = {"release_diff": {k: [a, b] for k, (a, b) in diff.items()}, "out": str(out)}
    if len(arms) > 1:
        report["arm_diff"] = arm_diff(configs)
    write_json(out / "prepare.json", report)
    for key, (old, new) in diff.items():
        log(f"  {key}: {json.dumps(old, default=str)} -> {json.dumps(new, default=str)}")
    if "arm_diff" in report:
        log(f"arms differ in {report['arm_diff']['differing_keys']}: {json.dumps(report['arm_diff']['reward_kwargs'])}")
    log(f"wrote {len(arms)} resolved config(s) under {out}")
    return 0


# ===================================================================================== CLI


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, *, arm: bool = True) -> None:
        sp.add_argument("--model", required=True, choices=MODELS)
        if arm:
            sp.add_argument("--arm", required=True, choices=ARMS)
            sp.add_argument("--seed", type=int, required=True)

    sp = sub.add_parser("check", help="imports, vendor manifest, data shas, store join, model dir, prompt clip, arm configs")
    common(sp, arm=False)
    sp.add_argument("--sft-dir", help="the M2 SFT dir (default: the pinned base snapshot)")
    sp.add_argument("--n-gpus", type=int, choices=(2, 4))
    sp.add_argument("--gpu", action="store_true", help="also check the visible GPUs (count, VRAM, flash-attn kernels)")
    sp.add_argument("--skip-clip", action="store_true", help="skip the prompt-length pass (debug only)")
    sp.add_argument("--json", help="write the full report here")

    sp = sub.add_parser("prepare", help="resolve EasyR1 config(s) into --out and show the diff against the release")
    common(sp)
    sp.add_argument("--sft-dir")
    sp.add_argument("--n-gpus", type=int, default=2, choices=(2, 4))
    sp.add_argument("--out", required=True)
    sp.add_argument("--all-arms", action="store_true")
    sp.add_argument("--smoke", action="store_true")

    for name in ("run", "smoke"):
        sp = sub.add_parser(name, help="full 300-step run" if name == "run" else "GPU smoke on the base model")
        common(sp)
        sp.add_argument("--n-gpus", type=int, required=True, choices=(2, 4))
        sp.add_argument("--run-dir", help="override the run directory")
        sp.add_argument("--vllm-python", help="python of the serving env; the final merged dir is loaded and decoded with its vLLM")
        sp.add_argument("--stall-minutes", type=float, default=45.0)
        if name == "run":
            sp.add_argument("--sft-dir", required=True)
            sp.add_argument("--save-freq", default="auto", help="auto (25; preemptible nodes: <= 45 min between saves) or a divisor of 100")
        else:
            sp.add_argument("--steps", type=int, help="default: smoke.steps (2)")
            sp.add_argument("--batch", type=int, help="rollout_batch_size = global_batch_size (default 8)")
            sp.add_argument("--n", type=int, help="rollouts per prompt (default 4)")
            sp.add_argument("--keep-ckpt", action="store_true", help="keep the FSDP checkpoints after the smoke")
            sp.add_argument("--no-resume-test", action="store_true", help="train all steps in one launch")

    sp = sub.add_parser("merge", help="merge kept steps of a run to bf16 HF dirs (CPU)")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--steps", help="comma list (default: keep steps + max_steps)")
    sp.add_argument("--vllm-python")

    sp = sub.add_parser("status", help="one JSON line: run status, tracker step, kept and merged steps")
    sp.add_argument("--run-dir", required=True)

    sp = sub.add_parser("report", help="step lines + summary of a run or smoke")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--tail", type=int, default=10)
    sp.add_argument("--last-files", type=int, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    handlers = {"check": do_check, "prepare": do_prepare, "run": do_run, "smoke": do_smoke, "merge": do_merge,
                "status": do_status, "report": do_report}
    try:
        return handlers[args.command](args)
    except ConfigError as exc:
        log(f"FATAL: config error: {exc}")
        return 4
    except Incomplete as exc:
        log(f"INCOMPLETE: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
