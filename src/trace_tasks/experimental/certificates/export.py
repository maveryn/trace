"""Split builder and exporter of the certificate extension (DESIGN 1.1, 2.5, 2.6, 4, 6).

One ``export_split`` call builds one split for every requested adapter:

1. Candidates are generated in ``seed_index`` order at the split's ``sampling_seed`` with
   TRACE's own per-instance pipeline (``hash64`` seed, ``task.generate``, the builder's
   finalizer and candidate validation), so every row equals the row ``build_dataset`` would
   write for the same ``(task, sampling_seed, seed_index)``. Work is cached per chunk of seed
   indices under ``private/build/``; a chunk is reused only when its header (certificate
   code, TRACE generator code, env lock, adapter, strata) matches, so reruns resume from the
   cache and give identical outputs.
2. Each candidate is assigned to a quota stratum from ``adapter.split_cell`` (train/E1 cells
   vs held-out E2 cells, DESIGN 6 table, D13). Rows of other cells are excluded, invariant
   failures (including an oracle round-trip failure) are quarantined with a reason (never
   model zeros), and test rows whose semantic key appears in SFT or RL, or earlier in the same
   test split, are dropped and counted.
3. Accepted rows are written as: a TRACE dataset dir per adapter (``train_instances.jsonl``,
   ``traces/``, images), the private extension store, the trace-tool public parquets
   (value-free labels), the EasyR1 RL parquet (``answer_gt`` sentinel) and a receipt with
   seeds, counts, sha256 of every output and code/env provenance.

Layout under ``--out`` (paths in receipts are relative to it)::

    private/  (mode 0700: every oracle-bearing file lives here)
      build/<split>/<adapter>/{train_instances.jsonl, traces/, images/, accepted.jsonl,
                               quarantine.jsonl, images.sha256, chunks/}   # TRACE dataset layout + cache
      <split>.extensions_v1.jsonl.zst                                    # per-split private store
      extensions_v1.jsonl.zst, rl_reward.extensions_v1.jsonl.zst          # written by ``freeze`` only
    public/<split>.{cert_px,cert_rel1000,ans,ann}.parquet              # trace-tool --dataset-path
    rl/<split>/<split>.parquet                                         # EasyR1 train_files / val_files (file paths)
    sft/<split>.jsonl, sft/images/<adapter>/...                        # export_sft (its own image root)
    receipts/<split>.json, receipts/<split>.sft.json, receipts/freeze.json, receipts/env_lock.<sha>.txt

An SFT or RL job needs only ``sft/`` or ``rl/`` (images are copied out or embedded), never
``private/`` (review LEAK-06). Give EasyR1 the RL parquet *file* paths: HF ``datasets`` infers
the split of a directory from its file names, so the directory ``rl/e1_dev`` (file
``e1_dev.parquet``) loads as a ``validation`` split and EasyR1's ``split="train"`` fails; the
file path always loads as ``train`` (review DATA-3; ``rl/rl`` also works as a directory). Per-split receipts
never record the shared merged store: ``freeze`` writes the merged store, the RL reward store
(rl + e1_dev rows only) and one ``receipts/freeze.json`` naming every split's outputs
(review CONF-10 / DATA-2).

Oracle data never reaches a public prompt: prompts are rendered from :func:`public_base`,
which whitelists the public TrainInstance fields.
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import registry
from .adapters.base import HOLDOUT_TYPES, Adapter
from .geometry import BOX_ANNOTATION_TYPES, POINT_ANNOTATION_TYPES, round_half_up
from .types import (
    COORD_FRAMES,
    CertificateError,
    CertificateExtension,
    CertificateVersionError,
    ExtensionMismatchError,
    QuarantineError,
    TraceRecord,
    canonical_digest,
    certificate_extension_id,
    scene_group,
)

EXPORT_VERSION = "cert-export-v2"  # v2: private/build layout, freeze receipt, env lock file, prompt shas of every condition
CODE_HASH = "cert-v1"
INSTANCE_VERSION = "v0"
MAX_ATTEMPTS_PER_INSTANCE = 100
STORE_NAME = "extensions_v1.jsonl.zst"
RL_REWARD_STORE = "rl_reward." + STORE_NAME  # rl + e1_dev rows only: the store the RL reward loads
TRACE_SHARD = "trace_shard_0001.jsonl.zst"
BUILD_DIR = ("private", "build")
FREEZE_RECEIPT = "freeze.json"

SPLIT_SEEDS: Dict[str, int] = {
    "fixtures": 7001,
    "sft": 7101,
    "rl": 7201,
    "e1_dev": 7301,
    "e1_test": 7401,
    "e2_test": 7501,
}
SPLIT_COUNTS: Dict[str, int] = {"fixtures": 16, "sft": 1500, "rl": 5000, "e1_dev": 128, "e1_test": 500, "e2_test": 300}
SPLIT_GROUPS: Dict[str, str] = {
    "fixtures": "any",
    "sft": "train",
    "rl": "train",
    "e1_dev": "train",
    "e1_test": "train",
    "e2_test": "heldout",
}
TRAIN_SPLITS = ("sft", "rl")
SFT_SPLITS = TRAIN_SPLITS  # export_sft never writes an oracle of a test split as a target (review LEAK-01)
TEST_SPLITS = ("e1_dev", "e1_test", "e2_test")
EVAL_SPLITS = ("fixtures", "e1_dev", "e1_test", "e2_test")  # trace-tool public parquets
RL_SPLITS = ("rl", "e1_dev")  # EasyR1 train_files / val_files
ADAPTER_ORDER = ("selected_set", "filtered_mean", "weighted_sum", "path")

# trace-tool condition -> coordinate frame (cert) or legacy prompt (ans/ann).
PUBLIC_CONDITIONS: Dict[str, str] = {
    "cert_px": "px_source",
    "cert_rel1000": "rel1000",
    "ans": "ans_v1",
    "ann": "ann_v1",
}
FRAME_CONDITION = {"px_source": "cert_px", "rel1000": "cert_rel1000"}
# public.prompt_sha256 keys of an extension: both frames and the two legacy prompts, so every
# trace-tool condition can be checked against the prompt the scorer joins (review LEAK-02).
PROMPT_KEYS: Dict[str, str] = {"px_source": "cert_px", "rel1000": "cert_rel1000", "ans_v1": "ans", "ann_v1": "ann"}
RL_FRAME = "px_source"  # every trained model and every SFT/RL target uses px_source (D8)

# TrainInstance fields a public prompt may be rendered from (never labels, trace or seeds).
PUBLIC_BASE_KEYS = ("instance_id", "instance_version", "domain", "task", "scene_id", "query_id", "prompt_variants")
PUBLIC_PROMPT_VARIANTS = ("answer_only", "answer_and_annotation")
PUBLIC_PARQUET_COLUMNS = (
    "images",
    "image_sizes",
    "prompt_answer",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "instance_id",
    "domain",
    "task",
    "scene_id",
    "query_id",
    "scene_variant",
    "cert_extension_id",
)
RL_PARQUET_COLUMNS = (
    "images",
    "image_sizes",
    "prompt_cert",
    "instance_id",
    "domain",
    "task",
    "scene_id",
    "query_id",
    "scene_variant",
    "cert_extension_id",
    "answer_gt",
)

CHESS_KIND_TASK = "task_games__chess__piece_kind_count"


class ExportError(CertificateError):
    """The split cannot be built as requested (configuration, quota or cache problem)."""


def rl_parquet_name(split: str) -> str:
    """Basename of a split's RL parquet. Pass ``rl/<split>/<split>.parquet`` to EasyR1 as a
    file path: as a directory, ``rl/e1_dev`` is read by HF as a ``validation`` split."""
    return f"{split}.parquet"


def build_root(out: Path) -> Path:
    """Root of the TRACE dataset dirs and chunk caches (under ``private/``)."""
    return Path(out).joinpath(*BUILD_DIR)


def _private_dir(out: Path) -> Path:
    """``out/private`` with mode 0700 (owner only): oracle-bearing files are not world-readable."""
    path = Path(out) / "private"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:  # pragma: no cover - a filesystem without chmod
        pass
    return path


# ======================================================================================
# provenance


@lru_cache(maxsize=None)
def _package_dirs() -> Tuple[Path, Path]:
    package = Path(__file__).resolve().parent
    resources = package.parents[1] / "resources" / "prompts" / "experimental" / "certificates"
    return package, resources


def _files_sha256(roots: Sequence[Path], suffixes: Tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes):
            if "__pycache__" in path.parts:
                continue
            digest.update(f"{root.name}/{path.relative_to(root).as_posix()}\0".encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def cert_code_sha256() -> str:
    """sha256 of the certificate package and its prompt/schema resources."""
    return _files_sha256(_package_dirs(), (".py", ".md", ".json", ".txt"))


def _git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(_package_dirs()[0]), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


_EXPERIMENTAL_PARTS = (("experimental",), ("resources", "prompts", "experimental"))


def _core_root() -> Path:
    """The importable ``trace_tasks`` package dir (not symlink-resolved, so a symlinked copy
    with one edited file is hashed as it is imported)."""
    import trace_tasks

    return Path(trace_tasks.__file__).parent


@lru_cache(maxsize=None)
def trace_core_sha256() -> str:
    """sha256 of TRACE's generator code and resources: every file of the ``trace_tasks``
    package outside the certificate package and its prompt resources (review DATA-1)."""
    root = _core_root()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if any(rel.parts[: len(prefix)] == prefix for prefix in _EXPERIMENTAL_PARTS):
            continue
        digest.update(rel.as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _core_git_dirty() -> Optional[bool]:
    """Uncommitted changes in TRACE's own code (the certificate package excluded)."""
    status = _git(
        "status",
        "--porcelain",
        "--",
        ":/src/trace_tasks",
        ":!/src/trace_tasks/experimental",
        ":!/src/trace_tasks/resources/prompts/experimental",
    )
    return bool(status.strip()) if status is not None else None


_LOCAL_PACKAGES = re.compile(r"^(-e\s.*[#&]egg=trace[-_]tasks\b.*|trace[-_]tasks(==|\s@\s).*)$", re.IGNORECASE)


def env_lock_text() -> Tuple[str, str]:
    """``(text, method)``: the sorted package list of this interpreter (``pip freeze --all``).

    Independent of ``PYTHONPATH`` (removed for the subprocess) and of the trace checkout's
    git HEAD: the local ``trace_tasks`` line is dropped, since the commit and code hashes are
    recorded separately (review DATA-4).
    """
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all", "--disable-pip-version-check"],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
            env=env,
        )
        lines, method = result.stdout.splitlines(), "pip_freeze"
    except (OSError, subprocess.SubprocessError):
        from importlib import metadata

        lines = [f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()]
        method = "importlib_metadata"
    kept = sorted(line.strip() for line in lines if line.strip() and not _LOCAL_PACKAGES.match(line.strip()))
    return "\n".join(kept) + "\n", method


@lru_cache(maxsize=None)
def env_lock() -> Tuple[str, str]:
    """``(sha256, method)`` of :func:`env_lock_text` (the text is persisted by the exporter)."""
    text, method = env_lock_text()
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), method


def write_env_lock(out: Path) -> str:
    """Persist the env lock text as ``receipts/env_lock.<sha256>.txt``; returns its path."""
    text, _ = env_lock_text()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if sha != env_lock()[0]:
        raise ExportError("the environment changed while exporting (env lock differs)")
    path = Path(out) / "receipts" / f"env_lock.{sha}.txt"
    if not path.exists():
        _atomic_write(path, text.encode("utf-8"))
    return path.relative_to(out).as_posix()


@lru_cache(maxsize=None)
def code_provenance() -> Dict[str, Any]:
    """Git HEAD + dirty flags, certificate and TRACE generator code shas, env lock, versions."""
    import trace_tasks

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    lock_sha, lock_method = env_lock()
    try:
        from .prompts import resource_digests

        resources = resource_digests()
    except Exception as exc:  # resources are optional for provenance; record why
        resources = {"error": type(exc).__name__}
    return {
        "git_head": head.strip() if head else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "core_git_dirty": _core_git_dirty(),
        "cert_code_sha256": cert_code_sha256(),
        "trace_core_sha256": trace_core_sha256(),
        "env_lock_sha256": lock_sha,
        "env_lock_method": lock_method,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "trace_tasks_version": str(getattr(trace_tasks, "__version__", "")),
        "resource_sha256": resources,
    }


def extension_provenance() -> Dict[str, Any]:
    """The ``provenance`` block of an extension (``payload_sha256`` is added per row)."""
    prov = code_provenance()
    return {
        "trace_tasks_commit": prov["git_head"],
        "cert_commit": prov["git_head"],
        "git_dirty": prov["git_dirty"],
        "core_git_dirty": prov["core_git_dirty"],
        "cert_code_sha256": prov["cert_code_sha256"],
        "trace_core_sha256": prov["trace_core_sha256"],
        "env_lock_sha256": prov["env_lock_sha256"],
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ======================================================================================
# public view, extension, SFT target (plan API)


def public_base(base: Any) -> Dict[str, Any]:
    """The whitelisted public fields of a TrainInstance dict (or ``TraceRecord``).

    This is the only input of every public prompt. Labels (``answer_gt``, ``annotation_gt``,
    ``reward_contract``), ``trace_ref``, seeds, versions and the trace sidecar are dropped.
    """
    row = getattr(base, "train", base)
    if not isinstance(row, Mapping):
        raise TypeError(f"base must be a TrainInstance mapping or TraceRecord, got {type(base).__name__}")
    out: Dict[str, Any] = {key: row[key] for key in PUBLIC_BASE_KEYS if key in row and key != "prompt_variants"}
    variants = row.get("prompt_variants")
    if isinstance(variants, Mapping):
        out["prompt_variants"] = {k: variants[k] for k in PUBLIC_PROMPT_VARIANTS if isinstance(variants.get(k), str)}
    return out


def render_prompts(base: Any, adapter: Adapter, image_size: Sequence[int]) -> Dict[str, str]:
    """Public prompts of one row for every trace-tool condition (``cert_px``, ``cert_rel1000``,
    ``ans``, ``ann``). Raises QuarantineError when a row cannot be rendered safely."""
    from . import prompts

    public = public_base(base)
    out: Dict[str, str] = {}
    for condition, target in PUBLIC_CONDITIONS.items():
        if target in COORD_FRAMES:
            out[condition] = prompts.render_public_prompt(public, adapter.public_contract(image_size, target))
        else:
            out[condition] = prompts.legacy_prompt(public, target)
    for condition, prompt in out.items():
        if prompt.count("<image>") != 1 or not prompt.startswith("<image>"):
            raise QuarantineError("image_marker", f"{condition} prompt of {public.get('instance_id')}")
    return out


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prompt_shas(prompts: Mapping[str, str]) -> Dict[str, str]:
    """``public.prompt_sha256`` of an extension from :func:`render_prompts` output, keyed by
    frame (``px_source``, ``rel1000``) and legacy condition target (``ans_v1``, ``ann_v1``)."""
    return {key: prompt_sha256(prompts[condition]) for key, condition in PROMPT_KEYS.items()}


def scene_variant_of(trace: Mapping[str, Any]) -> str:
    """Render-grammar variant as TRACE's RLVR export reads it (``render_spec.scene_variant``)."""
    for block in ("render_spec", "execution_trace"):
        value = (trace.get(block) or {}).get("scene_variant")
        if isinstance(value, str) and value:
            return value
    return ""


def _record_of(base: Any, trace: Optional[Mapping[str, Any]], image_size: Optional[Sequence[int]]) -> TraceRecord:
    if isinstance(base, TraceRecord):
        return base
    if not isinstance(base, Mapping) or not isinstance(trace, Mapping):
        raise TypeError("build_extension needs a TrainInstance mapping and its trace mapping (or a TraceRecord)")
    size = (int(image_size[0]), int(image_size[1])) if image_size is not None else None
    return TraceRecord(train=base, trace=trace, image_size=size)


def build_extension(
    base: Any,
    trace: Optional[Mapping[str, Any]],
    adapter: Adapter,
    *,
    image_size: Optional[Sequence[int]] = None,
    split: str = "",
    sampling_seed: Optional[int] = None,
    seed_index: Optional[int] = None,
    cell: Optional[str] = None,
    holdout: str = "",
    prompt_sha256: Optional[Mapping[str, str]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> CertificateExtension:
    """Private extension record of one generated instance (plan API, DESIGN 2.6).

    ``base`` is the TrainInstance dict (or a ``TraceRecord``, then ``trace`` is ignored) and
    ``trace`` its TraceInstance sidecar dict. Raises :class:`QuarantineError` on any adapter
    invariant failure; the extension id hashes the base identity, adapter version and schema.
    The plan's three-argument call is complete: ``prompt_sha256`` defaults to the shas of the
    row's rendered public prompts (both frames and the legacy ``ans_v1``/``ann_v1``) and
    ``provenance`` to this checkout's code provenance (review CONF-09).
    """
    record = _record_of(base, trace, image_size)
    instance_version = record.instance_version or INSTANCE_VERSION
    if not adapter.supports(record.task, record.query_id, instance_version):
        raise QuarantineError("unsupported", f"{adapter.adapter_id} does not support {record.task}/{record.query_id}")
    labels = {
        "answer_gt": record.answer_gt,
        "annotation_gt": record.annotation_gt,
        "reward_contract": record.reward_contract,
    }
    if labels["annotation_gt"].get("type") != adapter.annotation_type:
        raise QuarantineError("annotation_type", f"{labels['annotation_gt'].get('type')} != {adapter.annotation_type}")
    oracle = adapter.build_oracle(record)
    annotation, certificate, answer = adapter.sft_certificate(oracle, record.instance_seed)
    if not isinstance(certificate, Mapping) or certificate.get("kind") != adapter.kind:
        raise QuarantineError("sft_certificate_kind", f"{adapter.adapter_id}: {type(certificate).__name__}")
    if type(answer) is not int or answer != labels["answer_gt"].get("value"):
        raise QuarantineError("sft_answer_mismatch", f"{answer!r} != answer_gt {labels['answer_gt'].get('value')!r}")
    identity = adapter.identity()
    ext_id = certificate_extension_id(
        instance_id=record.instance_id,
        trace_record_hash=record.trace_record_hash,
        adapter_id=identity["id"],
        adapter_version=identity["version"],
        schema_id=identity["schema_id"],
    )
    width, height = record.source_size()
    if prompt_sha256 is None:
        # A stripped record (no prompt variants, e.g. a unit fixture) has no public prompt to hash.
        has_prompts = bool(public_base(record.train).get("prompt_variants"))
        prompt_sha256 = prompt_shas(render_prompts(record.train, adapter, (width, height))) if has_prompts else {}
    versions = record.train.get("versions") or {}
    base_block = {
        "instance_id": record.instance_id,
        "trace_record_hash": record.trace_record_hash,
        "task": record.task,
        "query_id": record.query_id,
        "instance_version": instance_version,
        "instance_seed": record.instance_seed,
        "sampling_seed": sampling_seed,
        "seed_index": seed_index,
        "scene_group": scene_group(record.task, record.instance_seed),
        "split": split,
        "cell": cell if cell is not None else adapter.split_cell(record),
        "holdout": holdout,
        "domain": str(record.train.get("domain", "")),
        "scene_id": str(record.train.get("scene_id", "")),
        "scene_variant": scene_variant_of(record.trace),
        "code_hash": str(versions.get("code_hash", "")),
    }
    sft_target = {"annotation": annotation, "certificate": _kind_first(certificate), "answer": answer}
    payload = {"labels": labels, "oracle": oracle, "sft_target": sft_target}
    prov = dict(extension_provenance() if provenance is None else provenance)
    prov["payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return CertificateExtension(
        certificate_extension_id=ext_id,
        base=base_block,
        adapter=identity,
        labels=labels,
        public={"image_size": [width, height], "prompt_sha256": dict(prompt_sha256 or {})},
        oracle=dict(oracle),
        sft_target=sft_target,
        provenance=prov,
    )


def _canonical_bytes(payload: Any) -> bytes:
    from trace_tasks.core.canonical import canonical_json_bytes

    canonical_digest({})  # asserts rfc8785 is importable
    return canonical_json_bytes(payload)


def _kind_first(certificate: Mapping[str, Any]) -> Dict[str, Any]:
    out = {"kind": certificate["kind"]}
    out.update((key, value) for key, value in certificate.items() if key != "kind")
    return out


def _round_floats(value: Any) -> Any:
    """Certificate numbers are integers (DESIGN 2.1): floats are rounded half-up."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round_half_up(value)
    if isinstance(value, Mapping):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    raise TypeError(f"unexpected certificate value {type(value).__name__}")


def round_annotation(value: Any, annotation_type: str) -> Any:
    """Round every box/point of a legacy annotation half-up to integers (278.5 -> 279)."""
    if annotation_type in BOX_ANNOTATION_TYPES:
        leaf = 4
    elif annotation_type in POINT_ANNOTATION_TYPES:
        leaf = 2
    else:
        raise ValueError(f"cannot round annotation type {annotation_type!r}")

    def walk(node: Any) -> Any:
        if isinstance(node, (list, tuple)):
            if len(node) == leaf and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node):
                return [round_half_up(float(x)) for x in node]
            return [walk(item) for item in node]
        if isinstance(node, Mapping):
            return {key: walk(item) for key, item in node.items()}
        return node

    return walk(value)


def sft_target_object(private_extension: CertificateExtension | Mapping[str, Any]) -> Dict[str, Any]:
    """The serialized SFT object: annotation (rounded) -> certificate (kind first) -> answer."""
    ext = _as_extension(private_extension)
    target = ext.sft_target
    if not target or any(key not in target for key in ("annotation", "certificate", "answer")):
        raise CertificateVersionError(f"extension {ext.certificate_extension_id} has no sft_target")
    annotation_type = str(ext.labels["annotation_gt"].get("type"))
    answer = target["answer"]
    if isinstance(answer, float) and float(answer).is_integer():
        answer = int(answer)
    return {
        "annotation": round_annotation(target["annotation"], annotation_type),
        "certificate": _round_floats(_kind_first(target["certificate"])),
        "answer": answer,
    }


def make_sft_target(base: Any, private_extension: CertificateExtension | Mapping[str, Any]) -> str:
    """SFT target string (plan API, D9, DESIGN 7.2): compact JSON, keys annotation ->
    certificate -> answer, ``kind`` first, boxes rounded half-up. ``base`` (the public row,
    or None) must belong to the extension."""
    ext = _as_extension(private_extension)
    if base is not None:
        row = getattr(base, "train", base)
        if isinstance(row, Mapping) and "instance_id" in row and str(row["instance_id"]) != ext.instance_id:
            raise ExtensionMismatchError(f"row {row['instance_id']} != extension instance {ext.instance_id}")
    return json.dumps(sft_target_object(ext), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _as_extension(value: CertificateExtension | Mapping[str, Any]) -> CertificateExtension:
    return value if isinstance(value, CertificateExtension) else CertificateExtension.from_dict(value)


# ======================================================================================
# held-out cells (DESIGN 6 table) and quotas


def _tokens(cell: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9_]+", str(cell).lower()) if token]


def _table_query(tokens: Sequence[str], cell: str) -> str:
    found = sorted({q for q in ("above", "below", "interval") for t in tokens if t.startswith(q) or t == f"in_{q}"})
    if len(found) != 1:
        raise ExportError(f"cannot read the table predicate from split_cell {cell!r}")
    return found[0]


def holdout_type(adapter: Adapter | str, cell: str) -> str:
    """``""`` for a train/E1 cell, else the E2 type ``render``, ``program`` or ``both``.

    Read from the adapter's declared ``train_cells`` / ``heldout_cells`` (DESIGN 6 table, D13):
    chess holds out the kind-only task (program); table holds out the ledger render, the
    interval predicate (program) and both; lever holds out textured_beam (render); rpg holds
    out the square canvas profile (render). An undeclared cell raises QuarantineError.
    """
    if isinstance(adapter, str):
        adapter = registry.load_adapter(adapter)
    return adapter.holdout_type(cell)


@dataclass(frozen=True)
class Stratum:
    """One quota cell of a split: rows of ``task`` whose holdout type is in ``holdout``
    (and, for the table SFT balance, whose predicate is ``query``)."""

    name: str
    task: str
    count: int
    holdout: Tuple[str, ...]
    query: Optional[str] = None

    def matches(self, kind: str, task: str, cell: str, holdout: str) -> bool:
        if task != self.task or holdout not in self.holdout:
            return False
        return self.query is None or _table_query(_tokens(cell), cell) == self.query

    def signature(self) -> List[Any]:
        return [self.name, self.task, list(self.holdout), self.query]


def _split_evenly(total: int, parts: int) -> List[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def quota_plan(adapter: Adapter, split: str, count: Optional[int] = None) -> Tuple[Stratum, ...]:
    """Quota strata of one adapter in one split (DESIGN 6 counts and cells)."""
    if split not in SPLIT_GROUPS:
        raise ExportError(f"unknown split {split!r}; known: {sorted(SPLIT_GROUPS)}")
    total = SPLIT_COUNTS[split] if count is None else int(count)
    if total < 1:
        raise ExportError(f"count must be >= 1, got {total}")
    group = SPLIT_GROUPS[split]
    tasks = sorted(adapter.supported)
    kind = adapter.kind
    if group == "any":
        return tuple(
            Stratum(f"any:{task}", task, n, ("",) + HOLDOUT_TYPES)
            for task, n in zip(tasks, _split_evenly(total, len(tasks)))
            if n > 0
        )
    if kind == "selected_set":
        task = CHESS_KIND_TASK if group == "heldout" else next(t for t in tasks if t != CHESS_KIND_TASK)
        return (Stratum(group, task, total, ("program",) if group == "heldout" else ("",)),)
    if len(tasks) != 1:
        raise ExportError(f"{adapter.adapter_id}: no task assignment rule for {len(tasks)} tasks")
    task = tasks[0]
    if group == "train":
        if kind == "filtered_mean" and split == "sft":
            above, below = _split_evenly(total, 2)
            return (Stratum("above", task, above, ("",), "above"), Stratum("below", task, below, ("",), "below"))
        return (Stratum("train", task, total, ("",)),)
    if kind == "filtered_mean":
        return tuple(Stratum(t, task, n, (t,)) for t, n in zip(HOLDOUT_TYPES, _split_evenly(total, 3)) if n > 0)
    return (Stratum("heldout", task, total, ("render",) if kind != "selected_set" else ("program",)),)


# ======================================================================================
# candidate generation (worker side)


class _MemoryTraceWriter:
    """Stand-in for TRACE's ``TraceShardWriter``: keeps the finalized trace record in memory.

    ``trace_ref.line_index`` is rewritten when the accepted rows are written as a shard; the
    ``trace_record_hash`` (blake3 of the JCS record) does not depend on the position.
    """

    shard_id = TRACE_SHARD

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def preview_ref(self, record: Dict[str, Any]) -> Any:
        from trace_tasks.core.canonical import canonical_json_bytes
        from trace_tasks.core.hash_utils import blake3_hex
        from trace_tasks.core.types import TraceRef

        return TraceRef(shard_id=self.shard_id, line_index=0, trace_record_hash=blake3_hex(canonical_json_bytes(record)))

    def append(self, record: Dict[str, Any]) -> Any:
        ref = self.preview_ref(record)
        self.records.append(record)
        return ref


@dataclass(frozen=True)
class CandidateJob:
    """Deterministic request for one candidate ``(task, sampling_seed, seed_index)``."""

    adapter_id: str
    task: str
    split: str
    sampling_seed: int
    seed_index: int
    stage_root: str
    strata: Tuple[Stratum, ...]
    provenance: Mapping[str, Any]


_TASK_CACHE: Dict[str, Any] = {}
_INPROCESS: Dict[str, Adapter] = {}  # adapters of the running in-process export, by adapter_id
_TYPE_REGISTRY: List[Any] = []


def _type_registry() -> Any:
    if not _TYPE_REGISTRY:
        from trace_tasks.core.type_registry import DEFAULT_REGISTRY_PATH, load_type_registry

        _TYPE_REGISTRY.append(load_type_registry(DEFAULT_REGISTRY_PATH))
    return _TYPE_REGISTRY[0]


def _generate(job: CandidateJob) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], int]:
    """TRACE's per-instance pipeline: ``(train, trace, rejection_reason, instance_seed)``."""
    from trace_tasks.core.builder import CandidateValidationError, _candidate_validation_rejection_reason
    from trace_tasks.core.builder import _finalize_generated_output
    from trace_tasks.core.config import BuildConfig
    from trace_tasks.core.seed import hash64
    from trace_tasks.tasks import create_task

    instance_seed = int(hash64(job.sampling_seed, f"{job.task}:instance_seed", job.seed_index))
    task = _TASK_CACHE.get(job.task)
    if task is None:
        task = _TASK_CACHE[job.task] = create_task(job.task)
    try:
        generated = task.generate(instance_seed, params={}, max_attempts=MAX_ATTEMPTS_PER_INSTANCE)
    except Exception as exc:  # the builder counts these as rejections, seed_index consumed
        return None, None, f"generate:{type(exc).__name__}", instance_seed
    config = BuildConfig(
        output_root=job.stage_root,
        dataset_name=f"cert_{job.split}",
        instance_version=INSTANCE_VERSION,
        image_format="png",
        tasks=[],
        sampling_seed=job.sampling_seed,
    )
    writer = _MemoryTraceWriter()
    try:
        train, _, _ = _finalize_generated_output(
            task=task,
            generated=generated,
            accepted_index=job.seed_index,
            instance_seed=instance_seed,
            config=config,
            stage_root=Path(job.stage_root),
            code_hash=CODE_HASH,
            type_registry=_type_registry(),
            trace_writer=writer,
        )
    except CandidateValidationError as exc:
        return None, None, _candidate_validation_rejection_reason(exc.validation_report), instance_seed
    return train, writer.records[0], None, instance_seed


def run_candidate(job: CandidateJob) -> Dict[str, Any]:
    """Generate, classify, verify and render one candidate (runs in a worker process)."""
    head = {"seed_index": job.seed_index, "task": job.task}
    train, trace, rejected, instance_seed = _generate(job)
    head["instance_seed"] = instance_seed
    if rejected is not None:
        return {**head, "status": "rejected", "reason": rejected}
    assert train is not None and trace is not None
    image_rel = str(train["images"][0]["path"])
    image_abs = Path(job.stage_root) / image_rel
    from PIL import Image

    with Image.open(image_abs) as image:
        size = [int(image.size[0]), int(image.size[1])]
    head["instance_id"] = train["instance_id"]
    adapter = _INPROCESS.get(job.adapter_id) or registry.adapter_by_id(job.adapter_id)
    record = TraceRecord(train=train, trace=trace, image_size=tuple(size), image_path=str(image_abs))
    try:
        cell = adapter.split_cell(record)
        holdout = adapter.holdout_type(cell)
        stratum = next((s.name for s in job.strata if s.matches(adapter.kind, record.task, cell, holdout)), None)
        head.update(cell=cell, holdout=holdout)
        if stratum is None:
            image_abs.unlink(missing_ok=True)
            return {**head, "status": "excluded"}
        prompts = render_prompts(train, adapter, size)
        ext = build_extension(
            train,
            trace,
            adapter,
            image_size=size,
            split=job.split,
            sampling_seed=job.sampling_seed,
            seed_index=job.seed_index,
            cell=cell,
            holdout=holdout,
            prompt_sha256=prompt_shas(prompts),
            provenance=job.provenance,
        )
        key = adapter.semantic_key(record)
        roundtrip = oracle_roundtrip(ext, adapter)
        if not roundtrip["ok"]:
            # DESIGN 2.5: a registry invariant failure is quarantined, never exported (review CONF-11)
            raise QuarantineError("roundtrip_failed", f"oracle {roundtrip['oracle']} sft {roundtrip['sft']}")
    except QuarantineError as exc:
        image_abs.unlink(missing_ok=True)
        return {**head, "status": "quarantined", "reason": exc.reason, "message": str(exc)}
    return {
        **head,
        "status": "ok",
        "stratum": stratum,
        "semantic_key": key,
        "image": {"path": image_rel, "size": size, "sha256": file_sha256(image_abs)},
        "prompts": prompts,
        "roundtrip": roundtrip,
        "extension": ext.to_dict(),
        "train": train,
        "trace": trace,
    }


def oracle_roundtrip(ext: CertificateExtension, adapter: Adapter) -> Dict[str, Any]:
    """A/G/M/F of the unrounded oracle object and of the serialized SFT target (DESIGN 9)."""
    from .scoring import score_certificate_response

    base = {"instance_id": ext.instance_id}
    target = ext.sft_target
    raw = json.dumps({key: target[key] for key in ("annotation", "certificate", "answer")}, allow_nan=False)
    out: Dict[str, Any] = {}
    for name, response in (("oracle", raw), ("sft", make_sft_target(None, ext))):
        result = score_certificate_response(response, base, ext, coord_frame="px_source", adapter=adapter)
        out[name] = [result.answer, result.grounding, result.semantics, result.format]
    out["ok"] = out["oracle"] == [1.0, 1.0, 1.0, 1.0] and [out["sft"][i] for i in (0, 2, 3)] == [1.0, 1.0, 1.0]
    return out


# ======================================================================================
# chunk cache and the quota fill (parent side)


def json_line(row: Any) -> bytes:
    """One compact JSON line (the store, chunk, train-instance and trace-shard row format)."""
    return (json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def jsonl_zst_bytes(rows: Iterable[Mapping[str, Any] | bytes]) -> bytes:
    """Deterministic compact-JSON lines, zstd level 9 (the store and chunk format). Items may
    be dicts or lines already serialized by :func:`json_line` (same bytes either way)."""
    import zstandard as zstd

    buffer = io.BytesIO()
    for row in rows:
        buffer.write(row if isinstance(row, bytes) else json_line(row))
    return zstd.ZstdCompressor(level=9).compress(buffer.getvalue())


def _zstd_lines_write(path: Path, rows: Iterable[Mapping[str, Any] | bytes]) -> None:
    _atomic_write(path, jsonl_zst_bytes(rows))


def read_jsonl_zst(path: str | Path) -> List[Dict[str, Any]]:
    import zstandard as zstd

    with Path(path).open("rb") as handle, zstd.ZstdDecompressor().stream_reader(handle) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        return [json.loads(line) for line in text if line.strip()]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    _atomic_write(path, text.encode("utf-8"))


def default_workers() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count() or 1
    return max(1, min(16, cpus))


class _Runner:
    """Runs candidate jobs in-process (workers=1) or in a lazily created process pool."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, int(workers))
        self._pool: Any = None

    def map(self, jobs: Sequence[CandidateJob]) -> List[Dict[str, Any]]:
        if self.workers == 1 or len(jobs) == 1:
            return [run_candidate(job) for job in jobs]
        if self._pool is None:
            # spawn: safe with any parent state and, unlike forkserver, no AF_UNIX socket dir
            # under $TMPDIR (whose cleanup fails noisily on NFS)
            self._pool = mp.get_context("spawn").Pool(self.workers)
        return list(self._pool.imap(run_candidate, jobs, chunksize=1))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


class _BuildWriter:
    """Streams accepted rows into the TRACE dataset layout while the quota fills.

    The parent keeps only a light outcome per row (no train/trace dicts; the extension as its
    serialized store line), so memory stays flat on 5,000-row splits. ``train_instances.jsonl``
    and the zstd trace shard are written to temporary names and moved into place by ``close``.
    """

    def __init__(self, stage_root: Path) -> None:
        import zstandard as zstd

        self.stage_root = stage_root
        (stage_root / "traces").mkdir(parents=True, exist_ok=True)
        self._train_path = stage_root / f".train_instances.jsonl.tmp{os.getpid()}"
        self._trace_path = stage_root / "traces" / f".{TRACE_SHARD}.tmp{os.getpid()}"
        self._train = self._train_path.open("wb")
        self._trace = zstd.ZstdCompressor(level=9).stream_writer(self._trace_path.open("wb"))
        self.rows = 0

    def add(self, outcome: Mapping[str, Any]) -> Dict[str, Any]:
        """Write one accepted outcome; returns its light form (kept for the manifests)."""
        train = dict(outcome["train"])
        train["trace_ref"] = {**train["trace_ref"], "shard_id": TRACE_SHARD, "line_index": self.rows}
        self._train.write(json_line(train))
        self._trace.write(json_line(outcome["trace"]))
        self.rows += 1
        extension = outcome["extension"]
        light = {key: value for key, value in outcome.items() if key not in ("train", "trace", "extension")}
        light.update(
            query_id=train["query_id"],
            domain=train["domain"],
            scene_id=train["scene_id"],
            scene_variant=extension["base"]["scene_variant"],
            cert_extension_id=extension["certificate_extension_id"],
            extension_line=json_line(extension),
        )
        return light

    def close(self, keep: bool = True) -> None:
        self._train.close()
        self._trace.close()  # ends the zstd frame and closes the file
        if keep:
            os.replace(self._train_path, self.stage_root / "train_instances.jsonl")
            os.replace(self._trace_path, self.stage_root / "traces" / TRACE_SHARD)
        else:
            self._train_path.unlink(missing_ok=True)
            self._trace_path.unlink(missing_ok=True)


@dataclass
class _AdapterFill:
    adapter: Adapter
    name: str
    strata: Tuple[Stratum, ...]
    writer: Optional[_BuildWriter] = None
    accepted: List[Dict[str, Any]] = field(default_factory=list)  # light outcomes (see _BuildWriter)
    quarantine: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)


def _chunk_header(fill: _AdapterFill, split: str, task: str, seed: int, start: int, end: int) -> Dict[str, Any]:
    """Cache key of one chunk: any change to the certificate code, TRACE's generator code and
    resources (review DATA-1), the env lock, the adapter or the strata regenerates it."""
    prov = code_provenance()
    return {
        "format": EXPORT_VERSION,
        "split": split,
        "task": task,
        "sampling_seed": seed,
        "start": start,
        "end": end,
        "adapter": fill.adapter.identity(),
        "strata": [s.signature() for s in fill.strata],
        "cert_code_sha256": prov["cert_code_sha256"],
        "trace_core_sha256": prov["trace_core_sha256"],
        "env_lock_sha256": prov["env_lock_sha256"],
    }


def _load_or_run_chunk(
    fill: _AdapterFill,
    split: str,
    task: str,
    seed: int,
    start: int,
    end: int,
    stage_root: Path,
    runner: _Runner,
) -> List[Dict[str, Any]]:
    path = stage_root / "chunks" / task / f"{start:07d}-{end:07d}.jsonl.zst"
    header = _chunk_header(fill, split, task, seed, start, end)
    if path.exists():
        rows = read_jsonl_zst(path)
        if rows and rows[0].get("header") == header and len(rows) == end - start + 1:
            return rows[1:]
        print(f"[cert-export] stale chunk {path.name} ({task}); regenerating", file=sys.stderr)
    provenance = extension_provenance()
    jobs = [
        CandidateJob(fill.adapter.adapter_id, task, split, seed, index, str(stage_root), fill.strata, provenance)
        for index in range(start, end)
    ]
    outcomes = runner.map(jobs)
    _zstd_lines_write(path, [{"header": header}, *outcomes])
    return outcomes


def _fill_adapter(
    fill: _AdapterFill,
    split: str,
    seed: int,
    stage_root: Path,
    runner: _Runner,
    chunk_size: int,
    train_keys: Optional[set],
    dedupe_within: bool = False,
) -> None:
    """Fill the quotas in ``seed_index`` order. ``train_keys`` drops test rows seen in SFT/RL;
    ``dedupe_within`` also drops a test row whose semantic key an earlier accepted row of the
    same split already has (review DATA-5)."""
    accepted_keys: set = set()
    counts: Dict[str, Any] = {
        "strata": {s.name: {"task": s.task, "requested": s.count, "accepted": 0} for s in fill.strata},
        "by_task": {},
        "rejected_reasons": {},
        "quarantine_reasons": {},
        "cells": {},
        "roundtrip_failed": [],
    }
    for task in dict.fromkeys(s.task for s in fill.strata):
        strata = [s for s in fill.strata if s.task == task]
        remaining = {s.name: s.count for s in strata}
        tally: Dict[str, Any] = {
            k: 0
            for k in (
                "candidates", "rejected", "excluded", "quarantined", "dedupe_dropped", "dedupe_within_dropped",
                "overflow", "accepted",
            )
        }
        tally["quarantine_reasons"] = {}
        limit = 20 * sum(remaining.values()) + 2 * chunk_size
        start = 0
        last_seed = -1
        while any(remaining.values()):
            if start >= limit:
                raise ExportError(
                    f"{fill.adapter.adapter_id} {split} {task}: quotas {remaining} not filled after {start} candidates"
                )
            outcomes = _load_or_run_chunk(fill, split, task, seed, start, start + chunk_size, stage_root, runner)
            for outcome in outcomes:
                if not any(remaining.values()):
                    break
                last_seed = int(outcome["seed_index"])
                tally["candidates"] += 1
                status = outcome["status"]
                if status == "rejected":
                    tally["rejected"] += 1
                    reason = str(outcome["reason"])
                    counts["rejected_reasons"][reason] = counts["rejected_reasons"].get(reason, 0) + 1
                    continue
                if status == "excluded":
                    tally["excluded"] += 1
                    continue
                if status == "quarantined":
                    tally["quarantined"] += 1
                    reason = str(outcome["reason"])
                    if reason == "roundtrip_failed":
                        counts["roundtrip_failed"].append(outcome.get("instance_id"))
                    counts["quarantine_reasons"][reason] = counts["quarantine_reasons"].get(reason, 0) + 1
                    tally["quarantine_reasons"][reason] = tally["quarantine_reasons"].get(reason, 0) + 1
                    fill.quarantine.append({"split": split, "adapter_id": fill.adapter.adapter_id, **outcome})
                    continue
                if train_keys is not None and outcome["semantic_key"] in train_keys:
                    tally["dedupe_dropped"] += 1
                    continue
                if dedupe_within and outcome["semantic_key"] in accepted_keys:
                    tally["dedupe_within_dropped"] += 1
                    continue
                stratum = outcome["stratum"]
                if remaining.get(stratum, 0) <= 0:
                    tally["overflow"] += 1
                    continue
                remaining[stratum] -= 1
                tally["accepted"] += 1
                accepted_keys.add(outcome["semantic_key"])
                counts["strata"][stratum]["accepted"] += 1
                counts["cells"][outcome["cell"]] = counts["cells"].get(outcome["cell"], 0) + 1
                if not outcome["roundtrip"]["ok"]:  # run_candidate quarantines these; kept as a guard
                    raise ExportError(f"row {outcome['instance_id']} failed the oracle round trip but was not quarantined")
                fill.accepted.append(fill.writer.add(outcome) if fill.writer is not None else outcome)
            start += chunk_size
        tally["last_seed_index"] = last_seed
        counts["by_task"][task] = tally
    keys = [row["semantic_key"] for row in fill.accepted]
    counts["accepted"] = len(fill.accepted)
    counts["quarantined"] = len(fill.quarantine)
    counts["duplicate_semantic_keys_within_split"] = len(keys) - len(set(keys))
    for name in ("rejected", "excluded", "dedupe_dropped", "dedupe_within_dropped", "overflow", "candidates"):
        counts[name] = sum(t[name] for t in counts["by_task"].values())
    fill.counts = counts


# ======================================================================================
# writers


def _rows_by_adapter(out: Path, split: str, adapter_name: str) -> List[Dict[str, Any]]:
    path = build_root(out) / split / adapter_name / "accepted.jsonl"
    if not path.exists():
        raise ExportError(f"split {split!r} has no build for adapter {adapter_name!r} ({path})")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_adapter_build(stage_root: Path, fill: _AdapterFill) -> Dict[str, Dict[str, Any]]:
    """Manifests of the accepted rows (``train_instances.jsonl`` and the trace shard were
    streamed by the fill's :class:`_BuildWriter`): accepted, quarantine and image manifests."""
    accepted = [
        {
            "instance_id": o["instance_id"],
            "cert_extension_id": o["cert_extension_id"],
            "adapter_id": fill.adapter.adapter_id,
            "task": o["task"],
            "query_id": o["query_id"],
            "domain": o["domain"],
            "scene_id": o["scene_id"],
            "scene_variant": o["scene_variant"],
            "seed_index": o["seed_index"],
            "instance_seed": o["instance_seed"],
            "cell": o["cell"],
            "holdout": o["holdout"],
            "stratum": o["stratum"],
            "semantic_key": o["semantic_key"],
            "image": o["image"],
            "prompts": o["prompts"],
            "roundtrip": o["roundtrip"],
        }
        for o in fill.accepted
    ]
    _write_jsonl(stage_root / "accepted.jsonl", accepted)
    _write_jsonl(stage_root / "quarantine.jsonl", fill.quarantine)
    manifest = "".join(f"{row['image']['sha256']}  {row['image']['path']}\n" for row in accepted)
    _atomic_write(stage_root / "images.sha256", manifest.encode("utf-8"))
    return {name: {"rows": n} for name, n in (
        ("train_instances.jsonl", len(accepted)),
        (f"traces/{TRACE_SHARD}", len(accepted)),
        ("accepted.jsonl", len(accepted)),
        ("quarantine.jsonl", len(fill.quarantine)),
        ("images.sha256", len(accepted)),
    )}


def write_store(path: Path, extensions: Iterable[Mapping[str, Any] | bytes]) -> int:
    """Write a private store from extension dicts or their :func:`json_line` lines."""
    lines = [item if isinstance(item, bytes) else json_line(item) for item in extensions]
    ids = [json.loads(line)["certificate_extension_id"] for line in lines]
    if len(ids) != len(set(ids)):
        raise ExportError(f"duplicate certificate_extension_id in {path}")
    _zstd_lines_write(path, lines)
    return len(lines)


def _store_lines(path: Path) -> List[bytes]:
    import zstandard as zstd

    with path.open("rb") as handle, zstd.ZstdDecompressor().stream_reader(handle) as reader:
        return [line for line in reader.read().splitlines(keepends=True) if line.strip()]


def read_store(path: str | Path) -> List[Dict[str, Any]]:
    """All extension records of a private store (``*.jsonl.zst``)."""
    return read_jsonl_zst(path)


def merge_stores(
    out: Path, *, splits: Optional[Sequence[str]] = None, name: str = STORE_NAME
) -> Tuple[Path, List[str], int]:
    """Concatenate the built per-split stores (DESIGN split order; ``splits`` restricts them)
    into ``private/<name>``. Called by :func:`freeze` only."""
    lines: List[bytes] = []
    found: List[str] = []
    for split in SPLIT_SEEDS:
        if splits is not None and split not in splits:
            continue
        path = out / "private" / f"{split}.{STORE_NAME}"
        if path.exists():
            lines.extend(_store_lines(path))
            found.append(split)
    merged = out / "private" / name
    n = write_store(merged, lines)
    return merged, found, n


def _image_column(out: Path, split: str, row: Mapping[str, Any], adapter_name: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
    path = build_root(out) / split / adapter_name / row["image"]["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["image"]["sha256"]:
        raise ExportError(f"image {path} changed since it was built")
    width, height = row["image"]["size"]
    return [{"bytes": data, "path": None}], [{"width": int(width), "height": int(height)}]


def _parquet_schema(columns: Sequence[str]) -> Any:
    import pyarrow as pa

    types = {
        "images": pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "image_sizes": pa.list_(pa.struct([("width", pa.int64()), ("height", pa.int64())])),
    }
    return pa.schema([(name, types.get(name, pa.string())) for name in columns])


def _write_parquet(path: Path, columns: Sequence[str], rows: Iterator[Dict[str, Any]], batch: int = 128) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _parquet_schema(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    n = 0
    with pq.ParquetWriter(str(tmp), schema) as writer:
        pending: List[Dict[str, Any]] = []
        for row in rows:
            pending.append(row)
            if len(pending) >= batch:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                n += len(pending)
                pending = []
        if pending or n == 0:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
            n += len(pending)
    os.replace(tmp, path)
    return n


def placeholder_labels(annotation_type: str, answer_type: str = "integer") -> Dict[str, str]:
    """Value-free label columns of a public parquet (DESIGN 4)."""
    return {
        "answer_gt": json.dumps({"type": answer_type, "value": None}),
        "annotation_gt": json.dumps({"type": annotation_type, "value": []}),
        "reward_contract": "{}",
    }


def _public_rows(out: Path, split: str, condition: str, groups: Sequence[Tuple[str, Adapter, List[Dict[str, Any]]]]) -> Iterator[Dict[str, Any]]:
    for name, adapter, rows in groups:
        for row in rows:
            images, sizes = _image_column(out, split, row, name)
            yield {
                "images": images,
                "image_sizes": sizes,
                "prompt_answer": row["prompts"][condition],
                **placeholder_labels(adapter.annotation_type),
                "instance_id": row["instance_id"],
                "domain": row["domain"],
                "task": row["task"],
                "scene_id": row["scene_id"],
                "query_id": row["query_id"],
                "scene_variant": row["scene_variant"],
                "cert_extension_id": row["cert_extension_id"],
            }


def _rl_rows(out: Path, split: str, groups: Sequence[Tuple[str, Adapter, List[Dict[str, Any]]]]) -> Iterator[Dict[str, Any]]:
    for name, _, rows in groups:
        for row in rows:
            images, sizes = _image_column(out, split, row, name)
            yield {
                "images": images,
                "image_sizes": sizes,
                "prompt_cert": row["prompts"][FRAME_CONDITION[RL_FRAME]],
                "instance_id": row["instance_id"],
                "domain": row["domain"],
                "task": row["task"],
                "scene_id": row["scene_id"],
                "query_id": row["query_id"],
                "scene_variant": row["scene_variant"],
                "cert_extension_id": row["cert_extension_id"],
                "answer_gt": json.dumps({"cert_extension_id": row["cert_extension_id"]}),
            }


def _output_entry(out: Path, path: Path, **extra: Any) -> Tuple[str, Dict[str, Any]]:
    return path.relative_to(out).as_posix(), {"sha256": file_sha256(path), "bytes": path.stat().st_size, **extra}


# ======================================================================================
# split export (the M1 entry point)


def _resolve_adapters(adapters: Optional[Sequence[Any]]) -> List[Tuple[str, Adapter]]:
    """``[(build dir name, adapter)]`` in DESIGN order; names may be module names or instances."""
    chosen: List[Tuple[str, Adapter]] = []
    if adapters is None:
        for name in ADAPTER_ORDER:
            try:
                chosen.append((name, registry.load_adapter(name)))
            except NotImplementedError:
                continue
        if not chosen:
            raise ExportError("no certificate adapter is implemented yet")
        return chosen
    for item in adapters:
        if isinstance(item, Adapter):
            chosen.append((item.adapter_id.split(".", 1)[0], item))
        else:
            chosen.append((str(item), registry.load_adapter(str(item))))
    names = [name for name, _ in chosen]
    if len(names) != len(set(names)):
        raise ExportError(f"duplicate adapters {names}")
    return sorted(chosen, key=lambda pair: ADAPTER_ORDER.index(pair[0]) if pair[0] in ADAPTER_ORDER else 99)


def _train_semantic_keys(out: Path, adapter_name: str) -> Tuple[set, Dict[str, str]]:
    keys: set = set()
    receipts: Dict[str, str] = {}
    for split in TRAIN_SPLITS:
        receipt = out / "receipts" / f"{split}.json"
        if not receipt.exists():
            raise ExportError(
                f"test splits are deduped against SFT and RL: build split {split!r} first "
                f"(missing {receipt}) or pass dedupe=False / --no-dedupe"
            )
        receipts[split] = file_sha256(receipt)
        keys.update(row["semantic_key"] for row in _rows_by_adapter(out, split, adapter_name))
    return keys, receipts


def export_split(
    split: str,
    out: str | Path,
    *,
    adapters: Optional[Sequence[Any]] = None,
    count: Optional[int] = None,
    workers: Optional[int] = None,
    chunk_size: int = 64,
    dedupe: bool = True,
) -> Dict[str, Any]:
    """Build one split for every adapter and write all its outputs under ``out``.

    Returns the receipt (also written to ``receipts/<split>.json``). ``count`` overrides the
    per-adapter DESIGN 6 count (tests use tiny splits). Resumable and deterministic: candidate
    chunks are cached under ``private/build/<split>/<adapter>/chunks`` and acceptance runs in
    ``seed_index`` order, so worker count and chunk size never change the accepted rows. The
    receipt records only this split's outputs (never the shared merged store, which ``freeze``
    writes) and, when it replaces an earlier build of the split, that build's shas.
    ``workers > 1`` uses a spawn pool: a calling script needs the usual
    ``if __name__ == "__main__":`` guard (the CLI has it).
    """
    if split not in SPLIT_SEEDS:
        raise ExportError(f"unknown split {split!r}; known: {list(SPLIT_SEEDS)}")
    out = Path(out).resolve()
    seed = SPLIT_SEEDS[split]
    chosen = _resolve_adapters(adapters)
    _private_dir(out)
    previous = _previous_build(out, split)
    n_workers = default_workers() if workers is None else max(1, int(workers))
    if any(adapter is not _module_adapter(name) for name, adapter in chosen):
        n_workers = 1  # adapters passed as instances (tests) are not importable by workers
    runner = _Runner(n_workers)
    _INPROCESS.update({adapter.adapter_id: adapter for _, adapter in chosen})
    fills: List[_AdapterFill] = []
    dedupe_receipts: Dict[str, Dict[str, str]] = {}
    outputs: Dict[str, Dict[str, Any]] = {}
    groups: List[Tuple[str, Adapter, List[Dict[str, Any]]]] = []
    extensions: List[bytes] = []
    try:
        for name, adapter in chosen:
            fill = _AdapterFill(adapter=adapter, name=name, strata=quota_plan(adapter, split, count))
            train_keys = None
            if dedupe and split in TEST_SPLITS:
                train_keys, dedupe_receipts[adapter.adapter_id] = _train_semantic_keys(out, name)
            stage_root = build_root(out) / split / name
            fill.writer = _BuildWriter(stage_root)
            try:
                _fill_adapter(
                    fill, split, seed, stage_root, runner, int(chunk_size), train_keys,
                    dedupe_within=bool(dedupe and split in TEST_SPLITS),
                )
            except BaseException:
                fill.writer.close(keep=False)
                raise
            fill.writer.close()
            for rel, meta in _write_adapter_build(stage_root, fill).items():
                key, entry = _output_entry(out, stage_root / rel, **meta)
                outputs[key] = entry
            groups.append((name, adapter, _rows_by_adapter(out, split, name)))
            extensions.extend(o["extension_line"] for o in fill.accepted)
            fill.accepted, fill.writer = [], None
            fills.append(fill)
    finally:
        runner.close()
        _INPROCESS.clear()

    store = out / "private" / f"{split}.{STORE_NAME}"
    write_store(store, extensions)
    key, entry = _output_entry(out, store, rows=len(extensions), kind="private_store")
    outputs[key] = entry
    if split in EVAL_SPLITS:
        for condition, target in PUBLIC_CONDITIONS.items():
            path = out / "public" / f"{split}.{condition}.parquet"
            n = _write_parquet(path, PUBLIC_PARQUET_COLUMNS, _public_rows(out, split, condition, groups))
            frame = target if target in COORD_FRAMES else None
            key, entry = _output_entry(out, path, rows=n, kind="public_parquet", condition=condition, coord_frame=frame, split=split)
            outputs[key] = entry
    if split in RL_SPLITS:
        path = out / "rl" / split / rl_parquet_name(split)
        n = _write_parquet(path, RL_PARQUET_COLUMNS, _rl_rows(out, split, groups))
        key, entry = _output_entry(out, path, rows=n, kind="rl_parquet", coord_frame=RL_FRAME, split=split)
        outputs[key] = entry
    env_lock_file = write_env_lock(out)

    receipt = {
        "receipt_version": EXPORT_VERSION,
        "command": "export",
        "split": split,
        "sampling_seed": seed,
        "code_hash": CODE_HASH,
        "instance_version": INSTANCE_VERSION,
        "group": SPLIT_GROUPS[split],
        "count_per_adapter": count if count is not None else SPLIT_COUNTS[split],
        "adapters": {fill.adapter.adapter_id: fill.adapter.identity() for fill in fills},
        "unsupported_adapters": _unsupported_adapters(),
        "strata": {fill.adapter.adapter_id: [asdict(s) for s in fill.strata] for fill in fills},
        "counts": {fill.adapter.adapter_id: fill.counts for fill in fills},
        "totals": {
            "accepted": sum(f.counts["accepted"] for f in fills),
            "quarantined": sum(f.counts["quarantined"] for f in fills),
            "rejected": sum(f.counts["rejected"] for f in fills),
            "excluded": sum(f.counts["excluded"] for f in fills),
            "dedupe_dropped": sum(f.counts["dedupe_dropped"] for f in fills),
            "dedupe_within_dropped": sum(f.counts["dedupe_within_dropped"] for f in fills),
            "roundtrip_failed": sum(len(f.counts["roundtrip_failed"]) for f in fills),
        },
        "dedupe": {
            "enabled": bool(dedupe and split in TEST_SPLITS),
            "against": list(TRAIN_SPLITS),
            "within_split": bool(dedupe and split in TEST_SPLITS),
            "train_receipts": dedupe_receipts,
        },
        "public_conditions": dict(PUBLIC_CONDITIONS),
        "outputs": outputs,
        "supersedes": _superseded(previous, outputs),
        "env_lock_file": env_lock_file,
        "workers": n_workers,
        "chunk_size": int(chunk_size),
        "provenance": code_provenance(),
    }
    _write_json(out / "receipts" / f"{split}.json", receipt)
    return receipt


def _superseded(previous: Optional[Dict[str, Any]], outputs: Mapping[str, Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The build this export replaces, or (for an identical rerun) the one its predecessor replaced."""
    if previous is None:
        return None
    if previous["outputs"] == {name: entry.get("sha256") for name, entry in sorted(outputs.items())}:
        return previous.get("supersedes")
    return previous


def _previous_build(out: Path, split: str) -> Optional[Dict[str, Any]]:
    """The shas of an earlier build of ``split`` this export replaces (its receipt and every
    output it recorded), so a rebuilt frozen split is never silent (review DATA-2)."""
    path = out / "receipts" / f"{split}.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text(encoding="utf-8"))
    return {
        "receipt_sha256": file_sha256(path),
        "receipt_version": old.get("receipt_version"),
        "outputs": {name: entry.get("sha256") for name, entry in sorted((old.get("outputs") or {}).items())},
        "supersedes": old.get("supersedes"),
    }


def freeze(out: str | Path) -> Dict[str, Any]:
    """Write the shared stores and one receipt naming every frozen output (review CONF-10).

    Checks every per-split receipt against the files on disk, then writes
    ``private/extensions_v1.jsonl.zst`` (all built splits, DESIGN split order) and
    ``private/rl_reward.extensions_v1.jsonl.zst`` (rl and e1_dev rows only: the store the RL
    reward loads, so no E1-test/E2-test oracle reaches a training job; review LEAK-01), and
    ``receipts/freeze.json`` with each split's receipt, store and parquet shas plus both
    merged-store shas. Run it once after the last ``export``/``export-sft``.
    """
    out = Path(out).resolve()
    _private_dir(out)
    splits: Dict[str, Any] = {}
    for split in SPLIT_SEEDS:
        receipt_path = out / "receipts" / f"{split}.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("receipt_version") != EXPORT_VERSION or receipt.get("command") != "export":
            raise ExportError(f"{receipt_path} is not a {EXPORT_VERSION} export receipt")
        for name, entry in (receipt.get("outputs") or {}).items():
            if "sha256" not in entry:
                continue
            if not (out / name).exists() or file_sha256(out / name) != entry["sha256"]:
                raise ExportError(f"{name} does not match {receipt_path}; rebuild the split before freezing")
        sft_receipt = out / "receipts" / f"{split}.sft.json"
        splits[split] = {
            "receipt": {"path": receipt_path.relative_to(out).as_posix(), "sha256": file_sha256(receipt_path)},
            "outputs": {name: entry["sha256"] for name, entry in sorted(receipt["outputs"].items()) if "sha256" in entry},
            "sft_receipt": (
                {"path": sft_receipt.relative_to(out).as_posix(), "sha256": file_sha256(sft_receipt)}
                if sft_receipt.exists()
                else None
            ),
        }
    if not splits:
        raise ExportError(f"no export receipts under {out / 'receipts'}")
    merged, merged_splits, merged_rows = merge_stores(out)
    rl_store, rl_splits, rl_rows = merge_stores(out, splits=RL_SPLITS, name=RL_REWARD_STORE)
    outputs = dict(
        [
            _output_entry(out, merged, rows=merged_rows, kind="private_store", splits=merged_splits),
            _output_entry(out, rl_store, rows=rl_rows, kind="private_store", splits=rl_splits),
        ]
    )
    receipt = {
        "receipt_version": EXPORT_VERSION,
        "command": "freeze",
        "outputs": outputs,
        "splits": splits,
        "merged_store": {
            "path": merged.relative_to(out).as_posix(), "sha256": file_sha256(merged),
            "splits": merged_splits, "rows": merged_rows,
        },
        "rl_reward_store": {
            "path": rl_store.relative_to(out).as_posix(), "sha256": file_sha256(rl_store),
            "splits": rl_splits, "rows": rl_rows,
            "split_phases": {"rl": "train", "e1_dev": "val"},
        },
        "provenance": code_provenance(),
    }
    _write_json(out / "receipts" / FREEZE_RECEIPT, receipt)
    return receipt


def _module_adapter(name: str) -> Optional[Adapter]:
    if name not in registry.ADAPTER_MODULES:
        return None
    try:
        return registry.load_adapter(name)
    except NotImplementedError:
        return None


def _unsupported_adapters() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ADAPTER_ORDER:
        try:
            registry.load_adapter(name)
        except NotImplementedError as exc:
            out[name] = f"adapter_not_implemented: {exc}"
    return out


# ======================================================================================
# SFT export


def export_sft(
    split: str,
    out: str | Path,
    *,
    system_prompt: Optional[str] = None,
    system_prompt_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Write ``sft/<split>.jsonl`` (messages + image path + target) from a built training split.

    Only training splits (``sft``, ``rl``) can be exported: a test split's oracle never becomes
    a supervised target (review LEAK-01). The user text is byte-identical to the ``cert_px``
    public prompt of the same instance and the system text is ``cert_system_v1`` unless another
    file is given. Images are copied (hard-linked when possible) to ``sft/images/`` and
    referenced by path relative to ``out``, so an SFT job never needs ``private/`` (LEAK-06).
    """
    from . import prompts

    if split not in SFT_SPLITS:
        raise ExportError(f"export-sft writes training targets only: split must be one of {SFT_SPLITS}, got {split!r}")
    out = Path(out).resolve()
    receipt_path = out / "receipts" / f"{split}.json"
    if not receipt_path.exists():
        raise ExportError(f"split {split!r} is not built (missing {receipt_path})")
    export_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if system_prompt is None:
        if system_prompt_file is not None:
            system_prompt = Path(system_prompt_file).read_text(encoding="utf-8").strip()
        else:
            system_prompt = prompts.system_prompt()
    store_path = out / "private" / f"{split}.{STORE_NAME}"
    store_key = store_path.relative_to(out).as_posix()
    if file_sha256(store_path) != export_receipt["outputs"][store_key]["sha256"]:
        raise CertificateVersionError(f"{store_path} does not match receipt {receipt_path}")
    store = {row["certificate_extension_id"]: CertificateExtension.from_dict(row) for row in read_store(store_path)}
    records: List[Dict[str, Any]] = []
    image_manifest: List[str] = []
    for name in ADAPTER_ORDER:
        if not (build_root(out) / split / name / "accepted.jsonl").exists():
            continue
        for row in _rows_by_adapter(out, split, name):
            ext = store.get(row["cert_extension_id"])
            if ext is None or ext.instance_id != row["instance_id"]:
                raise ExtensionMismatchError(f"no extension {row['cert_extension_id']} for {row['instance_id']}")
            user = row["prompts"][FRAME_CONDITION[RL_FRAME]]
            if prompt_sha256(user) != ext.public["prompt_sha256"].get(RL_FRAME):
                raise CertificateVersionError(f"prompt sha mismatch for {row['instance_id']}")
            target = make_sft_target({"instance_id": row["instance_id"]}, ext)
            image_rel = _sft_image(out, split, name, row)
            image_manifest.append(f"{row['image']['sha256']}  {image_rel}\n")
            records.append(
                {
                    "instance_id": row["instance_id"],
                    "cert_extension_id": row["cert_extension_id"],
                    "adapter_id": row["adapter_id"],
                    "split": split,
                    "task": row["task"],
                    "query_id": row["query_id"],
                    "cell": row["cell"],
                    "coord_frame": RL_FRAME,
                    "images": [image_rel],
                    "image_size": row["image"]["size"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": target},
                    ],
                    "target": target,
                    "prompt_sha256": prompt_sha256(user),
                    "system_sha256": prompt_sha256(system_prompt),
                }
            )
    path = out / "sft" / f"{split}.jsonl"
    _write_jsonl(path, records)
    key, entry = _output_entry(out, path, rows=len(records), kind="sft_jsonl", coord_frame=RL_FRAME)
    manifest = out / "sft" / f"{split}.images.sha256"
    _atomic_write(manifest, "".join(image_manifest).encode("utf-8"))
    manifest_key, manifest_entry = _output_entry(out, manifest, rows=len(image_manifest), kind="sft_images")
    receipt = {
        "receipt_version": EXPORT_VERSION,
        "command": "export-sft",
        "split": split,
        "export_receipt": {"path": receipt_path.relative_to(out).as_posix(), "sha256": file_sha256(receipt_path)},
        "system_prompt_sha256": prompt_sha256(system_prompt),
        "target_format": "compact JSON: annotation -> certificate (kind first) -> answer; boxes rounded half-up",
        "rows": len(records),
        "outputs": {key: entry, manifest_key: manifest_entry},
        "image_root": "sft/images",
        "provenance": code_provenance(),
    }
    _write_json(out / "receipts" / f"{split}.sft.json", receipt)
    return receipt


def _sft_image(out: Path, split: str, adapter_name: str, row: Mapping[str, Any]) -> str:
    """Copy (hard-link when possible) one SFT image out of ``private/build`` into
    ``sft/images/<split>/<adapter>/``; returns its path relative to ``out``."""
    source = build_root(out) / split / adapter_name / row["image"]["path"]
    target = out / "sft" / "images" / split / adapter_name / row["image"]["path"]
    if not target.exists() or file_sha256(target) != row["image"]["sha256"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
        tmp.unlink(missing_ok=True)
        try:
            os.link(source, tmp)
        except OSError:
            shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    if file_sha256(target) != row["image"]["sha256"]:
        raise ExportError(f"SFT image {target} does not match its build sha256")
    return target.relative_to(out).as_posix()


__all__ = [
    "ADAPTER_ORDER",
    "BUILD_DIR",
    "FREEZE_RECEIPT",
    "PROMPT_KEYS",
    "RL_REWARD_STORE",
    "SFT_SPLITS",
    "build_root",
    "extension_provenance",
    "freeze",
    "prompt_shas",
    "rl_parquet_name",
    "trace_core_sha256",
    "CODE_HASH",
    "EVAL_SPLITS",
    "ExportError",
    "PUBLIC_CONDITIONS",
    "PUBLIC_PARQUET_COLUMNS",
    "RL_PARQUET_COLUMNS",
    "RL_SPLITS",
    "SPLIT_COUNTS",
    "SPLIT_SEEDS",
    "STORE_NAME",
    "Stratum",
    "TEST_SPLITS",
    "TRAIN_SPLITS",
    "build_extension",
    "code_provenance",
    "export_sft",
    "export_split",
    "file_sha256",
    "holdout_type",
    "make_sft_target",
    "merge_stores",
    "oracle_roundtrip",
    "placeholder_labels",
    "public_base",
    "quota_plan",
    "json_line",
    "jsonl_zst_bytes",
    "read_jsonl_zst",
    "read_store",
    "render_prompts",
    "round_annotation",
    "run_candidate",
    "sft_target_object",
    "write_store",
]
