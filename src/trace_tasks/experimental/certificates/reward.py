"""EasyR1 batch reward for certificate RL (DESIGN 5 "Batch API", 7.3).

EasyR1 loads ``experiments/certificates/reward_entry.py`` by file path, which re-exports
:func:`compute_score`; ``worker.reward.reward_function_kwargs`` supplies ``store_path``,
``store_sha256``, ``arm`` and the weights through ``functools.partial``.

Contract (``verl/workers/reward/function.py``): ``reward_inputs[i]`` carries ``response``
(decoded with ``skip_special_tokens``), ``response_length``, ``ground_truth`` (the RL parquet's
``answer_gt`` sentinel ``{"cert_extension_id": ...}``) and every other non-tensor column of
the row (``instance_id``, ``cert_extension_id``, ``prompt_cert``, ...). The function returns
one dict of floats per input, in input order; EasyR1 puts ``overall`` on the last response
token and logs every key as ``reward/<key>``.

- The private extension store is read once per process and verified against ``store_sha256``;
  rows join it by the sentinel's ``cert_extension_id``, and the row's ``instance_id`` must
  equal the extension's. A missing extension, an id mismatch or a version mismatch raises:
  it is never a model zero (DESIGN 2.5).
- Every arm logs A, G, M and F, so arms differ only in ``overall`` (and ``b_term``).
- Scores never depend on wall-clock time; ``verifier_ms`` is logged only. Per-check records
  go to ``details_dir/<train|val>_<pid>_<call:06d>.jsonl`` when ``details_dir`` is set. The
  reward actor is never told the step: ``train``/``val`` comes from the extensions' split
  through ``split_phases`` (default ``{"rl": "train", "e1_dev": "val"}``); an extension of any
  other split (SFT, the frozen E1-test/E2-test oracles, fixtures) raises (review LEAK-01).
- The prompt column (``prompt_cert``) never changes a score. When a row carries it, its
  sha256 must equal the extension's ``public.prompt_sha256[coord_frame]``: a reward scoring
  in a frame the prompt did not declare, or a drifted prompt, raises (review EX-6, LEAK-02).
- Every model output scores: a model-producible exception of the legacy helpers is a scored
  zero inside ``score_certificate_response``; anything that still raises here is an
  implementation error and aborts the batch (never a model zero).
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .scoring import (
    ALPHA,
    FORMAT_WEIGHT,
    check_prompt,
    require_scipy_assignment,
    reward_arms,
    score_certificate_response,
)
from .types import (
    COORD_FRAMES,
    CertificateError,
    CertificateExtension,
    CertificateVersionError,
    ExtensionMismatchError,
    ScoreResult,
    require_rfc8785,
)

REWARD_NAME = "trace_certificate_v1"
REWARD_TYPE = "batch"
BASE_ARMS = ("A", "G", "P")
GATED_ARMS = {"G": "G_gated", "P": "P_gated"}
METRIC_KEYS = (
    "overall",
    "score",
    "accuracy",
    "answer_reward",
    "grounding",
    "semantic",
    "cert_format",
    "joint",
    "b_term",
    "verifier_ms",
)
UNIT_KEYS = METRIC_KEYS[:-1]  # verifier_ms is a wall-clock diagnostic (ms), never a score
TRAIN_SPLITS = ("rl", "sft")  # export.TRAIN_SPLITS
# RL reward splits -> phase. SFT rows never reach RL, and the frozen E1-test/E2-test oracles
# never reach any training-time reward.
DEFAULT_SPLIT_PHASES: Dict[str, str] = {"rl": "train", "e1_dev": "val"}
PHASES = ("train", "val")
PARSED_CACHE_SIZE = 4096
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class ExtensionStore:
    """The private extension store (``*.jsonl.zst`` or plain ``*.jsonl``) indexed by id.

    Lines are kept as text and parsed on demand (LRU of ``PARSED_CACHE_SIZE``), so a 30k-row
    store costs about its JSON size in memory.
    """

    def __init__(self, path: str, sha256: str, lines: Dict[str, str]) -> None:
        self.path = path
        self.sha256 = sha256
        self._lines = lines
        self._parsed: "OrderedDict[str, CertificateExtension]" = OrderedDict()

    @classmethod
    def load(cls, path: str | Path, sha256: str) -> "ExtensionStore":
        """Read ``path``, verify its sha256 over the exact bytes parsed, and index every line."""
        expected = str(sha256 or "").strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise CertificateVersionError(f"store_sha256 must be 64 hex characters, got {sha256!r}")
        raw = Path(path).read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != expected:
            raise CertificateVersionError(f"extension store {path} has sha256 {found}, expected {expected}")
        if raw[:4] == _ZSTD_MAGIC:
            import zstandard as zstd

            with zstd.ZstdDecompressor().stream_reader(io.BytesIO(raw), read_across_frames=True) as reader:
                raw = reader.read()
        lines: Dict[str, str] = {}
        for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            ext_id = record.get("certificate_extension_id") if isinstance(record, dict) else None
            if not isinstance(ext_id, str) or not ext_id:
                raise CertificateVersionError(f"{path}:{number}: record has no certificate_extension_id")
            if ext_id in lines:
                raise CertificateVersionError(f"{path}:{number}: duplicate certificate_extension_id {ext_id}")
            lines[ext_id] = line
        return cls(str(path), found, lines)

    def __len__(self) -> int:
        return len(self._lines)

    def __contains__(self, ext_id: object) -> bool:
        return ext_id in self._lines

    def get(self, ext_id: str) -> CertificateExtension:
        """The parsed extension; raises :class:`CertificateVersionError` when it is missing."""
        cached = self._parsed.get(ext_id)
        if cached is not None:
            self._parsed.move_to_end(ext_id)
            return cached
        line = self._lines.get(ext_id)
        if line is None:
            raise CertificateVersionError(f"certificate extension {ext_id} is not in store {self.path}")
        extension = CertificateExtension.from_dict(json.loads(line))
        if extension.certificate_extension_id != ext_id:
            raise ExtensionMismatchError(f"store line for {ext_id} carries {extension.certificate_extension_id}")
        extension.verify()
        self._parsed[ext_id] = extension
        if len(self._parsed) > PARSED_CACHE_SIZE:
            self._parsed.popitem(last=False)
        return extension


_STORES: Dict[Tuple[str, str], ExtensionStore] = {}
_CALLS = itertools.count()  # per process; numbers the details files


def load_store(store_path: str | Path, store_sha256: str) -> ExtensionStore:
    """The verified store for ``(path, sha256)``, loaded once per process (Ray actor)."""
    key = (os.path.realpath(os.fspath(store_path)), str(store_sha256 or "").strip().lower())
    store = _STORES.get(key)
    if store is None:
        require_rfc8785()
        require_scipy_assignment()
        store = ExtensionStore.load(store_path, store_sha256)
        _STORES[key] = store
    return store


def resolve_arm(arm: str, gated: bool = False) -> str:
    """The ``scoring.reward_arms`` key: ``A``, ``G``, ``P``, or ``G_gated``/``P_gated``.

    ``gated`` must be a real bool: a quoted YAML ``'false'`` is a non-empty string and would
    otherwise silently select the gated arm (review CONTRACT-4).
    """
    if not isinstance(gated, bool):
        raise ValueError(f"gated must be a bool (true/false), got {gated!r} ({type(gated).__name__})")
    if not isinstance(arm, str):
        raise ValueError(f"arm must be a string, got {arm!r}")
    name = str(arm)
    if name in GATED_ARMS.values():
        return name
    if name not in BASE_ARMS:
        raise ValueError(f"arm must be one of {BASE_ARMS} (or {tuple(GATED_ARMS.values())}), got {arm!r}")
    if not gated:
        return name
    if name not in GATED_ARMS:
        raise ValueError("arm 'A' has no gated form (DESIGN 5 defines P_gated and G_gated)")
    return GATED_ARMS[name]


def b_term(result: ScoreResult, arm_key: str) -> float:
    """The arm's B: A for arm A, G for G/G_gated, (G + M) / 2 for P/P_gated."""
    if arm_key == "A":
        return result.answer
    if arm_key in ("G", "G_gated"):
        return result.grounding
    if arm_key in ("P", "P_gated"):
        return (result.grounding + result.semantics) / 2.0
    raise ValueError(f"unknown arm {arm_key!r}")


def row_metrics(
    result: ScoreResult,
    arm_key: str,
    *,
    alpha: float = ALPHA,
    format_weight: float = FORMAT_WEIGHT,
) -> Dict[str, float]:
    """The EasyR1 metric dict of one scored row (always the keys of ``METRIC_KEYS``)."""
    overall = reward_arms(
        result.answer, result.grounding, result.semantics, result.format, alpha=alpha, format_weight=format_weight
    )[arm_key]
    metrics = {
        "overall": overall,
        "score": overall,
        "accuracy": result.answer,
        "answer_reward": result.answer,
        "grounding": result.grounding,
        "semantic": result.semantics,
        "cert_format": result.format,
        "joint": result.joint,
        "b_term": b_term(result, arm_key),
        "verifier_ms": result.diagnostics.get("verifier_ms", 0.0),
    }
    for key in METRIC_KEYS:
        value = float(metrics[key])
        if not math.isfinite(value) or value < 0.0 or (key in UNIT_KEYS and value > 1.0):
            raise ValueError(f"{REWARD_NAME}: metric {key} = {value!r} is out of range")
        metrics[key] = value
    return metrics


def sentinel_id(ground_truth: Any) -> str:
    """``cert_extension_id`` of the RL ``answer_gt`` sentinel (a JSON string or a mapping)."""
    value = ground_truth
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("ground_truth must be the JSON sentinel {\"cert_extension_id\": ...}") from exc
    if not isinstance(value, Mapping) or set(value) != {"cert_extension_id"}:
        raise ValueError(
            "ground_truth must be exactly {\"cert_extension_id\": ...}; a legacy answer_gt means the "
            "RL parquet was not exported for certificates"
        )
    ext_id = value["cert_extension_id"]
    if not isinstance(ext_id, str) or not ext_id:
        raise ValueError(f"ground_truth cert_extension_id must be a non-empty string, got {ext_id!r}")
    return ext_id


def _join(
    reward_input: Mapping[str, Any], store: ExtensionStore, index: int, coord_frame: str
) -> Tuple[str, CertificateExtension, bool]:
    ext_id = sentinel_id(reward_input.get("ground_truth"))
    column = reward_input.get("cert_extension_id")
    if column is not None and str(column) != ext_id:
        raise ExtensionMismatchError(f"row {index}: cert_extension_id column {column} != sentinel {ext_id}")
    extension = store.get(ext_id)
    instance_id = reward_input.get("instance_id")
    if instance_id is None:
        raise ExtensionMismatchError(f"row {index}: reward input has no instance_id")
    if str(instance_id) != extension.instance_id:
        raise ExtensionMismatchError(
            f"row {index}: instance_id {instance_id} != extension {ext_id} instance_id {extension.instance_id}"
        )
    checked = check_prompt(reward_input.get("prompt_cert"), extension, coord_frame, f"row {index}")
    return ext_id, extension, checked


def _phase(extensions: Sequence[CertificateExtension], split_phases: Mapping[str, str]) -> str:
    """The phase of a batch from its extensions' splits; any split outside ``split_phases``
    (a test oracle, SFT, fixtures) or a batch mixing phases raises."""
    phases = set()
    for ext in extensions:
        split = str(ext.base.get("split", ""))
        if split not in split_phases:
            raise CertificateVersionError(
                f"{REWARD_NAME}: extension {ext.certificate_extension_id} is from split {split!r}, which this "
                f"reward does not score (split_phases={dict(split_phases)}); held-out oracles never reach RL"
            )
        phases.add(split_phases[split])
    if len(phases) > 1:
        raise CertificateVersionError(f"{REWARD_NAME}: one reward call mixes phases {sorted(phases)}")
    return phases.pop() if phases else "val"


def _split_phases(split_phases: Optional[Mapping[str, str]]) -> Dict[str, str]:
    phases = dict(DEFAULT_SPLIT_PHASES if split_phases is None else split_phases)
    if not isinstance(split_phases, (Mapping, type(None))) or not phases:
        raise ValueError(f"split_phases must be a non-empty mapping split -> train|val, got {split_phases!r}")
    for split, phase in phases.items():
        if phase not in PHASES:
            raise ValueError(f"split_phases[{split!r}] must be one of {PHASES}, got {phase!r}")
    return phases


def _write_details(details_dir: str | Path, phase: str, call: int, records: Sequence[Mapping[str, Any]]) -> Path:
    """Write one call's per-row records; never overwrites an existing file (resumed actors)."""
    root = Path(details_dir)
    root.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str) + "\n" for record in records
    )
    suffix = 0
    while True:
        name = f"{phase}_{os.getpid()}_{call:06d}" + (f".{suffix}" if suffix else "") + ".jsonl"
        try:
            with (root / name).open("x", encoding="utf-8") as handle:
                handle.write(text)
            return root / name
        except FileExistsError:
            suffix += 1


def _details_record(
    index: int,
    reward_input: Mapping[str, Any],
    response: str,
    extension: CertificateExtension,
    result: ScoreResult,
    metrics: Mapping[str, float],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    base = extension.base
    return {
        **context,
        "row": index,
        "instance_id": extension.instance_id,
        "cert_extension_id": extension.certificate_extension_id,
        "adapter_id": extension.adapter_id,
        "split": base.get("split"),
        "cell": base.get("cell"),
        "scene_group": base.get("scene_group"),
        "response_length": reward_input.get("response_length"),
        "response_sha256": hashlib.sha256(response.encode("utf-8", "surrogatepass")).hexdigest(),
        "metrics": dict(metrics),
        **result.to_dict(),
    }


def compute_score(
    reward_inputs: Sequence[Mapping[str, Any]],
    *,
    store_path: str,
    store_sha256: str,
    arm: str,
    alpha: float = ALPHA,
    format_weight: float = FORMAT_WEIGHT,
    gated: bool = False,
    coord_frame: str = "px_source",
    details_dir: Optional[str] = None,
    split_phases: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, float]]:
    """Score an EasyR1 batch in input order: one dict of ``METRIC_KEYS`` floats per row.

    ``R = (1-f)[aA + (1-a)B] + fF`` with B per arm (A: A, G: G, P: (G + M) / 2); ``gated=True``
    selects ``(1-f)A[a + (1-a)B] + fF`` for G and P. Raises on any store, sentinel, id, prompt,
    split or version problem and on mistyped kwargs; never returns a model zero for one.
    ``split_phases`` maps the extension splits this reward may score to ``train``/``val``.
    """
    arm_key = resolve_arm(arm, gated)
    reward_arms(0.0, 0.0, 0.0, 0.0, alpha=alpha, format_weight=format_weight)  # validates the weights
    if coord_frame not in COORD_FRAMES:
        raise ValueError(f"coord_frame must be one of {COORD_FRAMES}, got {coord_frame!r}")
    phases = _split_phases(split_phases)
    store = load_store(store_path, store_sha256)
    require_scipy_assignment()
    joined = [_join(reward_input, store, index, coord_frame) for index, reward_input in enumerate(reward_inputs)]
    shown = bool(details_dir) and coord_frame == "px_source"  # G_shown/M_shown are logged only
    outputs: List[Dict[str, float]] = []
    records: List[Dict[str, Any]] = []
    phase = _phase([extension for _, extension, _ in joined], phases)
    call = next(_CALLS)
    context = {
        "reward_name": REWARD_NAME,
        "phase": phase,
        "call": call,
        "arm": arm_key,
        "alpha": float(alpha),
        "format_weight": float(format_weight),
        "coord_frame": coord_frame,
        "store_sha256": store.sha256,
        "split_phases": phases,
    }
    for index, (reward_input, (ext_id, extension, prompt_checked)) in enumerate(zip(reward_inputs, joined)):
        response = reward_input.get("response")
        response = response if isinstance(response, str) else str(response or "")
        try:
            result = score_certificate_response(
                response,
                {"instance_id": extension.instance_id, "cert_extension_id": ext_id},
                extension,
                coord_frame=coord_frame,
                shown_frame_diagnostics=shown,
                alpha=alpha,
                format_weight=format_weight,
            )
        except CertificateError:
            raise
        except Exception as exc:  # an adapter bug is an implementation error, never a model zero
            message = f"{REWARD_NAME}: scoring row {index} ({ext_id}) raised {type(exc).__name__}: {exc}"
            raise RuntimeError(message) from exc
        metrics = row_metrics(result, arm_key, alpha=alpha, format_weight=format_weight)
        outputs.append(metrics)
        if details_dir:
            record = _details_record(index, reward_input, response, extension, result, metrics, context)
            record["prompt_checked"] = prompt_checked
            records.append(record)
    if details_dir and records:
        _write_details(details_dir, phase, call, records)
    return outputs


__all__ = [
    "DEFAULT_SPLIT_PHASES",
    "METRIC_KEYS",
    "REWARD_NAME",
    "REWARD_TYPE",
    "TRAIN_SPLITS",
    "UNIT_KEYS",
    "ExtensionStore",
    "b_term",
    "check_prompt",
    "compute_score",
    "load_store",
    "resolve_arm",
    "row_metrics",
    "sentinel_id",
]
