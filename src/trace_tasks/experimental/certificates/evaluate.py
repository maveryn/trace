"""Offline certificate evaluation of trace-tool runs (DESIGN 5, 7.1).

``score_run`` reads one trace-tool ``responses.jsonl`` (one condition, one decoding seed),
joins every row to its private extension by ``instance_id`` inside the scorer, and writes:

- ``cert_scores.jsonl``: one row per instance (identity, A/G/M/F, checks, error class and
  generation facts such as ``completion_tokens``, ``stop_reason`` and ``within_2048``);
- ``summary.json``: per adapter / split / cell aggregates with supported, missing and
  quarantined denominators;
- ``score_receipt.json``: input/output sha256, store and export receipt, code provenance.

The run's public condition (``cert_px``, ``cert_rel1000``, ``ans``, ``ann``) and split come
from the dataset sha256 in ``run_config.json``, matched against the export receipt; a run on
any other parquet is refused unless ``allow_unfrozen`` (recorded; review CONF-07). The run's
system prompt, ``oracle_crop``, tools and decoding preset must be DESIGN 7.1's for the
condition (``allow_settings_mismatch`` records an override; review LEAK-05 / CONF-14), and
every row's transcript prompt must hash to the prompt the extension froze for that condition
and frame (``prompt_check``; review LEAK-02). Missing extensions, an edited store row
(``payload_sha256``), a store that is not in the receipt and version mismatches raise; they
are never model zeros. Endpoint errors and server-rejected requests (no tokens generated) are
infrastructure exclusions with their own denominators, never model zeros (review CONF-04).
Legacy ``ans``/``ann`` rows get A (and G for ``ann``); M and F are N/A.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import prompts, registry
from .adapters.base import Adapter, PublicContext
from .export import (
    EXPORT_VERSION,
    PUBLIC_CONDITIONS,
    STORE_NAME,
    _write_json,
    _write_jsonl,
    code_provenance,
    file_sha256,
    jsonl_zst_bytes,
    read_store,
)
from .geometry import annotation_view, shown_frame, source_frame
from .reward import check_prompt
from .scoring import legacy_prediction, score_answer, score_certificate_response, score_grounding
from .types import SCORER_VERSION, CertificateExtension, CertificateVersionError, ExtensionMismatchError

SUMMARY_VERSION = "cert-summary-v1"
TOKEN_CAP = 2048
LEGACY_CONDITIONS = ("ans", "ann")
POSITION_ONLY_REASONS = frozenset({"not_final"})


# ======================================================================================
# inputs


@dataclass
class Store:
    """The private extension store, indexed by extension id and by instance id."""

    path: Path
    sha256: str
    rows: List[Dict[str, Any]]
    by_id: Dict[str, CertificateExtension]
    by_instance: Dict[str, CertificateExtension]

    def for_instance(self, instance_id: str) -> CertificateExtension:
        ext = self.by_instance.get(str(instance_id))
        if ext is None:
            raise ExtensionMismatchError(f"no certificate extension for instance {instance_id} in {self.path}")
        return ext

    def split_sha256(self, split: str) -> str:
        """sha256 the split's own store file would have (rows of ``split`` in store order)."""
        rows = [row for row in self.rows if (row.get("base") or {}).get("split") == split]
        return hashlib.sha256(jsonl_zst_bytes(rows)).hexdigest()


def load_store(path: str | Path) -> Store:
    """Load a private store and check every extension id."""
    path = Path(path)
    rows = read_store(path)
    by_id: Dict[str, CertificateExtension] = {}
    by_instance: Dict[str, CertificateExtension] = {}
    for payload in rows:
        ext = CertificateExtension.from_dict(payload)
        ext.verify()  # id and payload_sha256 (labels, oracle, SFT target; review LEAK-04)
        if ext.certificate_extension_id in by_id:
            raise CertificateVersionError(f"duplicate extension {ext.certificate_extension_id} in {path}")
        if ext.instance_id in by_instance:
            raise CertificateVersionError(f"instance {ext.instance_id} has two extensions in {path}")
        by_id[ext.certificate_extension_id] = ext
        by_instance[ext.instance_id] = ext
    return Store(path=path, sha256=file_sha256(path), rows=rows, by_id=by_id, by_instance=by_instance)


def verify_store(store: Store, export_receipt: Mapping[str, Any], *, store_sha256: Optional[str] = None) -> str:
    """Raise unless the store is the one the export receipt froze; returns how it matched.

    Accepted: the split's own store listed in the receipt, or a merged store (``freeze``)
    whose rows of this split re-serialize to exactly that split store. An explicit
    ``store_sha256`` is an *additional* requirement, never a substitute for the receipt
    (review LEAK-04).
    """
    prefix = ""
    if store_sha256 is not None:
        if store.sha256 != str(store_sha256).strip().lower():
            raise CertificateVersionError(f"store {store.path} sha256 {store.sha256} != expected {store_sha256}")
        prefix = "explicit_sha256+"
    split = str(export_receipt.get("split"))
    split_store = (export_receipt.get("outputs") or {}).get(f"private/{split}.{STORE_NAME}") or {}
    if store.sha256 == split_store.get("sha256"):
        return prefix + "split_store"
    if split_store.get("sha256") and store.split_sha256(split) == split_store["sha256"]:
        return prefix + "merged_store_split_rows"
    raise CertificateVersionError(
        f"store {store.path} (sha256 {store.sha256}) does not contain the {split} rows frozen by the export receipt"
    )


def infrastructure_failure(record: Mapping[str, Any]) -> Optional[str]:
    """``endpoint_error`` / ``rejected`` for a record that is an infrastructure failure, else
    None. trace-tool writes ``stop_reason="rejected"`` when the server refuses the request
    (HTTP 4xx, e.g. context length) before generating anything (review CONF-04)."""
    stop = record.get("stop_reason")
    if stop == "error":
        return "endpoint_error"
    if stop == "rejected" and not record.get("turns") and not record.get("completion_tokens"):
        return "rejected"
    return None


def read_responses(path: str | Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """trace-tool ``responses.jsonl`` -> ``(records, infrastructure_failure_records, n_lines)``.

    Same rule as trace-tool's ``load_responses``: one record per ``instance_id``, the last
    record that is not an infrastructure failure wins. Rows whose only records are endpoint
    errors or server rejections (:func:`infrastructure_failure`) are returned separately
    (excluded from scoring, counted). A crash-torn last line is ignored.
    """
    chosen: Dict[str, Dict[str, Any]] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n_lines = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        n_lines += 1
        if not isinstance(record, dict) or "instance_id" not in record:
            continue
        previous = chosen.get(record["instance_id"])
        if previous is None or infrastructure_failure(record) is None or infrastructure_failure(previous) is not None:
            chosen[record["instance_id"]] = record
    records = sorted(chosen.values(), key=lambda r: (r.get("row_index", 0), str(r["instance_id"])))
    failures = [r for r in records if infrastructure_failure(r) is not None]
    return [r for r in records if infrastructure_failure(r) is None], failures, n_lines


def model_family(model_id: Optional[str]) -> str:
    """trace-tool's family rule: ``qwen2.5-vl``, ``qwen3-vl``, ``qwen3.5`` or ``other``."""
    key = "".join(ch for ch in str(model_id or "").lower() if ch.isalnum())
    if "qwen25vl" in key:
        return "qwen2.5-vl"
    if "qwen3vl" in key:
        return "qwen3-vl"
    if "qwen35" in key:
        return "qwen3.5"
    return "other"


@dataclass(frozen=True)
class RunInfo:
    """What a trace-tool run scored: public condition, split, frame and model family, plus the
    integrity checks that admitted it (recorded in the score receipt)."""

    cond: str
    split: Optional[str]
    coord_frame: Optional[str]
    family: str
    run_config: Mapping[str, Any]
    dataset_frozen: bool = True
    allow_unfrozen: bool = False
    settings_problems: Tuple[str, ...] = ()
    allow_settings_mismatch: bool = False
    run_config_sha256: Optional[str] = None

    @property
    def prompt_key(self) -> str:
        """Key of ``public.prompt_sha256`` for this run's prompts (frame or legacy target)."""
        return str(self.coord_frame) if self.coord_frame is not None else PUBLIC_CONDITIONS[self.cond]

    def checks(self) -> Dict[str, Any]:
        return {
            "dataset_frozen": self.dataset_frozen,
            "allow_unfrozen": self.allow_unfrozen,
            "settings_problems": list(self.settings_problems),
            "allow_settings_mismatch": self.allow_settings_mismatch,
        }


SETTINGS_RECORDED = ("system_prompt_sha256", "oracle_crop", "tools", "decoding", "decoding_params",
                     "max_turns", "max_tokens_per_turn", "max_total_tokens", "min_pixels", "max_pixels")


def settings_problems(cond: str, run_config: Mapping[str, Any]) -> List[str]:
    """Differences between a run's generation settings and DESIGN 7.1 for ``cond``: the
    condition's system prompt (cert_system_v1 for cert/ann, TRACE's for ans), no oracle crop,
    no tools and the TRACE decoding preset."""
    settings = run_config.get("settings")
    if not isinstance(settings, Mapping):
        return ["run_config.json has no settings block"]
    problems: List[str] = []
    want = prompts.expected_system_prompt_sha256(PUBLIC_CONDITIONS[cond])
    found = settings.get("system_prompt_sha256")
    if found != want:
        problems.append(f"system_prompt_sha256 {str(found)[:12]} != {want[:12]} expected for {cond}")
    if settings.get("oracle_crop") is not False:
        problems.append(f"oracle_crop is {settings.get('oracle_crop')!r}, expected false")
    if list(settings.get("tools") or []) or "tools" not in settings:
        problems.append(f"tools {settings.get('tools')!r}, expected []")
    if settings.get("decoding") != "trace":
        problems.append(f"decoding {settings.get('decoding')!r}, expected the 'trace' preset")
    return problems


def resolve_run(
    responses_path: Path,
    export_receipt: Mapping[str, Any],
    *,
    cond: Optional[str] = None,
    family: Optional[str] = None,
    allow_unfrozen: bool = False,
    allow_settings_mismatch: bool = False,
) -> RunInfo:
    """Condition and split of a run from ``run_config.json`` (dataset sha256 -> receipt).

    ``cond`` only confirms the condition of a frozen parquet; a run whose dataset sha256 is
    absent or not a public parquet of the receipt is refused unless ``allow_unfrozen`` (then
    ``cond`` names the condition and the mismatch is recorded). The run's settings must match
    DESIGN 7.1 unless ``allow_settings_mismatch`` (recorded).
    """
    config_path = responses_path.parent / "run_config.json"
    run_config: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config_sha = file_sha256(config_path) if config_path.exists() else None
    split = None
    frozen = False
    dataset_sha = (run_config.get("dataset") or {}).get("sha256")
    for name, entry in (export_receipt.get("outputs") or {}).items():
        if entry.get("kind") == "public_parquet" and dataset_sha is not None and entry.get("sha256") == dataset_sha:
            if cond is not None and cond != entry.get("condition"):
                raise CertificateVersionError(f"--cond {cond} but the run's dataset is {name} ({entry.get('condition')})")
            cond, split = str(entry["condition"]), str(entry.get("split") or export_receipt.get("split"))
            frozen = True
            break
    if not frozen:
        where = "has no run_config.json" if not config_path.exists() else f"has dataset sha256 {dataset_sha}"
        if not allow_unfrozen:
            raise CertificateVersionError(
                f"{responses_path} {where}, which is not a frozen public parquet of the export receipt; "
                "pass the right --receipt (or, to score it anyway, --allow-unfrozen with --cond; recorded)"
            )
        if cond is None:
            raise CertificateVersionError(f"{responses_path} {where}: --allow-unfrozen needs --cond")
    if cond not in PUBLIC_CONDITIONS:
        raise ValueError(f"cond must be one of {sorted(PUBLIC_CONDITIONS)}, got {cond!r}")
    problems = settings_problems(cond, run_config)
    if problems and not allow_settings_mismatch:
        raise CertificateVersionError(
            f"{responses_path}: generation settings differ from DESIGN 7.1 for {cond}: {problems} "
            "(--allow-settings-mismatch scores it anyway; recorded)"
        )
    frame = PUBLIC_CONDITIONS[cond] if cond not in LEGACY_CONDITIONS else None
    model = run_config.get("model_id") or run_config.get("model")
    return RunInfo(
        cond=cond,
        split=split or export_receipt.get("split"),
        coord_frame=frame,
        family=family or model_family(model),
        run_config=run_config,
        dataset_frozen=frozen,
        allow_unfrozen=bool(allow_unfrozen),
        settings_problems=tuple(problems),
        allow_settings_mismatch=bool(allow_settings_mismatch),
        run_config_sha256=config_sha,
    )


def transcript_prompt(record: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """``(system text, user prompt)`` of a trace-tool record's transcript; the user prompt is
    rebuilt with ``<image>`` at each image part, i.e. the public parquet's prompt text."""
    transcript = record.get("transcript")
    if not isinstance(transcript, list):
        return None, None

    def text_of(content: Any) -> Optional[str]:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return None
        parts = []
        for part in content:
            if not isinstance(part, Mapping):
                return None
            if part.get("type") == "image":
                parts.append("<image>")
            elif part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                return None
        return "".join(parts)

    system = next((m for m in transcript if isinstance(m, Mapping) and m.get("role") == "system"), None)
    user = next((m for m in transcript if isinstance(m, Mapping) and m.get("role") == "user"), None)
    return (
        text_of(system.get("content")) if system is not None else None,
        text_of(user.get("content")) if user is not None else None,
    )


def check_record_prompt(record: Mapping[str, Any], ext: CertificateExtension, info: RunInfo) -> None:
    """Integrity of one row (never a score input): the transcript's prompt must be the prompt
    the extension froze for the run's condition and frame, and its system text the condition's
    system prompt (review LEAK-02, LEAK-05)."""
    where = f"row {record.get('row_index')} ({record.get('instance_id')})"
    system, user = transcript_prompt(record)
    if user is None:
        raise CertificateVersionError(
            f"{where}: the record carries no transcript user prompt to check against the frozen prompt "
            "(--no-prompt-check skips this; recorded)"
        )
    check_prompt(user, ext, info.prompt_key, where)
    if not info.allow_settings_mismatch and system is not None:
        want = prompts.expected_system_prompt_sha256(PUBLIC_CONDITIONS[info.cond])
        if hashlib.sha256(system.encode("utf-8")).hexdigest() != want:
            raise CertificateVersionError(f"{where}: transcript system prompt is not the {info.cond} system prompt")


# ======================================================================================
# per-row scoring


def _identity_fields(record: Mapping[str, Any], ext: CertificateExtension, info: RunInfo) -> Dict[str, Any]:
    base = ext.base
    return {
        "instance_id": ext.instance_id,
        "cert_extension_id": ext.certificate_extension_id,
        "adapter_id": ext.adapter_id,
        "task": base.get("task"),
        "query_id": base.get("query_id"),
        "split": base.get("split"),
        "cell": base.get("cell"),
        "holdout": base.get("holdout", ""),
        "scene_group": base.get("scene_group"),
        "condition": record.get("condition") or info.run_config.get("condition"),
        "cond": info.cond,
        "coord_frame": info.coord_frame,
        "model": record.get("model") or info.run_config.get("model"),
        "family": info.family,
        "seed": record.get("seed", info.run_config.get("seed")),
        "row_index": record.get("row_index"),
    }


def _generation_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    tokens = record.get("completion_tokens")
    stop = record.get("stop_reason")
    return {
        "completion_tokens": tokens,
        "final_prompt_tokens": record.get("final_prompt_tokens"),
        "stop_reason": stop,
        "truncated": stop == "length",
        "within_2048": bool(stop == "answer" and isinstance(tokens, int) and tokens <= TOKEN_CAP),
    }


def _legacy_frame(family: str, image_size: Tuple[int, int]) -> Tuple[str, Any]:
    """``ann`` states no frame: G is primary in the family-native frame (DESIGN 5)."""
    if family == "qwen2.5-vl":
        return "px_shown", shown_frame(image_size)
    if family in ("qwen3.5", "qwen3-vl"):
        return "rel1000", source_frame("rel1000", image_size)
    return "px_source", source_frame("px_source", image_size)


def score_record(record: Mapping[str, Any], ext: CertificateExtension, adapter: Adapter, info: RunInfo) -> Dict[str, Any]:
    """One ``cert_scores.jsonl`` row."""
    response = record.get("final_response")
    response = response if isinstance(response, str) else ("" if response is None else str(response))
    row = _identity_fields(record, ext, info)
    if info.cond not in LEGACY_CONDITIONS:
        result = score_certificate_response(
            response,
            {"instance_id": record["instance_id"], "cert_extension_id": ext.certificate_extension_id},
            ext,
            coord_frame=str(info.coord_frame),
            adapter=adapter,
            shown_frame_diagnostics=info.family == "qwen2.5-vl" and info.coord_frame == "px_source",
        )
        diag = result.diagnostics
        reasons = list(diag.get("format_reasons") or [])
        row.update(
            A=result.answer,
            A_legacy=diag.get("A_legacy"),
            G=result.grounding,
            G_id=diag.get("G_id"),
            G_shown=diag.get("G_shown"),
            M=result.semantics,
            M_shown=diag.get("M_shown"),
            F=result.format,
            joint=result.joint,
            overall=dict(result.overall),
            error_class=result.error_class,
            answer_guard=diag.get("answer_guard"),
            envelope_error=diag.get("envelope_error"),
            format_reasons=reasons,
            valid_schema=diag.get("envelope_error") is None and set(reasons) <= POSITION_ONLY_REASONS,
            n_claimed=diag.get("n_claimed"),
            checks=[check.to_dict() for check in result.checks],
            verifier_ms=diag.get("verifier_ms"),
            slow_row=bool(diag.get("slow_row")),
            coverage_status=result.coverage_status,
            scorer_version=result.scorer_version,
        )
    else:
        started = time.perf_counter()
        registry.check_adapter_identity(adapter, ext)
        answer, answer_legacy, guard = score_answer(response, ext.labels["answer_gt"], use_scored_object=False)
        row.update(A=answer, A_legacy=answer_legacy, answer_guard=guard, M=None, F=None, joint=None, overall=None)
        if info.cond == "ann":
            image_size = ext.image_size
            frame_name, frame = _legacy_frame(info.family, image_size)
            _, annotation, _ = legacy_prediction(response)
            grounding, _, _ = score_grounding(annotation, ext.labels, adapter.annotation_type, frame, image_size)
            g_px, _, _ = score_grounding(
                annotation, ext.labels, adapter.annotation_type, source_frame("px_source", image_size), image_size
            )
            ctx = PublicContext(
                task=str(ext.base["task"]),
                query_id=str(ext.base["query_id"]),
                image_size=image_size,
                coord_frame=frame_name,
                frame=frame,
            )
            _, g_id = adapter.identity_report(annotation_view(annotation, adapter.annotation_type, frame), ext.oracle, ctx)
            row.update(G=grounding, G_frame=frame_name, G_px=g_px, G_id=g_id)
        else:
            row.update(G=None, G_id=None)
        elapsed = (time.perf_counter() - started) * 1000.0
        row.update(
            error_class=None,
            checks=[],
            verifier_ms=elapsed,
            slow_row=False,
            coverage_status="supported",
            scorer_version=SCORER_VERSION,
        )
    row.update(_generation_fields(record))
    return row


# ======================================================================================
# aggregation


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _numbers(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)]) if q > 0 else float(ordered[0])


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Metrics of one group of ``cert_scores`` rows (DESIGN 7.1 report list)."""
    out: Dict[str, Any] = {"n": len(rows)}
    for key in ("A", "A_legacy", "G", "G_id", "G_shown", "G_px", "M", "M_shown", "F", "joint"):
        values = _numbers(rows, key)
        if values:
            out[key] = _mean(values)
    cert_rows = [row for row in rows if row.get("M") is not None]
    if cert_rows:
        out["valid_schema_rate"] = _mean([float(bool(row.get("valid_schema"))) for row in cert_rows])
        out["envelope_failure_rate"] = _mean([float(row.get("envelope_error") is not None) for row in cert_rows])
        a_by_m = {"A1_M1": 0, "A1_Mlt1": 0, "A0_M1": 0, "A0_Mlt1": 0}
        classes: Dict[str, int] = {}
        for row in cert_rows:
            a_ok, m_ok = row.get("A") == 1.0, row.get("M") == 1.0
            a_by_m[f"A{int(a_ok)}_M{'1' if m_ok else 'lt1'}"] += 1
            classes[str(row.get("error_class"))] = classes.get(str(row.get("error_class")), 0) + 1
        out["A_x_M1"] = a_by_m
        out["error_classes"] = dict(sorted(classes.items()))
        for arm in ("A", "G", "P", "P_gated", "G_gated"):
            values = [float(row["overall"][arm]) for row in cert_rows if isinstance(row.get("overall"), Mapping) and arm in row["overall"]]
            if values:
                out.setdefault("overall", {})[arm] = _mean(values)
    out["completion_rate"] = _mean([float(row.get("stop_reason") == "answer") for row in rows]) if rows else None
    out["truncation_rate"] = _mean([float(bool(row.get("truncated"))) for row in rows]) if rows else None
    out["within_2048_rate"] = _mean([float(bool(row.get("within_2048"))) for row in rows]) if rows else None
    tokens = _numbers(rows, "completion_tokens")
    out["tokens"] = {
        "total": int(sum(tokens)),
        "mean": _mean(tokens),
        "median": float(statistics.median(tokens)) if tokens else None,
    }
    ms = _numbers(rows, "verifier_ms")
    out["verifier_ms"] = {
        "mean": _mean(ms),
        "median": float(statistics.median(ms)) if ms else None,
        "p99": _quantile(ms, 0.99),
        "max": max(ms) if ms else None,
        "slow_rows": sum(1 for row in rows if row.get("slow_row")),
    }
    return out


def _group(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    return [
        {**dict(zip(keys, values)), **aggregate(members)}
        for values, members in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_keys: Sequence[str] = (),
    denominators: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregates per adapter, adapter x split and adapter x split x cell (plus ``run_keys``)."""
    keys = tuple(run_keys)
    return {
        "summary_version": SUMMARY_VERSION,
        "scorer_version": SCORER_VERSION,
        "denominators": dict(denominators or {}),
        "all": _group(rows, keys) if keys else [aggregate(rows)],
        "by_adapter": _group(rows, keys + ("adapter_id",)),
        "by_adapter_split": _group(rows, keys + ("adapter_id", "split")),
        "by_adapter_split_cell": _group(rows, keys + ("adapter_id", "split", "cell")),
        "by_adapter_split_holdout": _group(rows, keys + ("adapter_id", "split", "holdout")),
    }


DENOMINATOR_KEYS = ("supported", "scored", "missing", "endpoint_errors", "rejected", "quarantined")


def _denominators(
    store: Store,
    split: Optional[str],
    scored: Sequence[Mapping[str, Any]],
    error_ids: Sequence[str],
    export_receipt: Mapping[str, Any],
    rejected_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """Supported (expected), scored, missing, endpoint-error, rejected (server refused the
    request) and quarantined rows per adapter."""
    expected: Dict[str, set] = {}
    for ext in store.by_id.values():
        if split is None or ext.base.get("split") == split:
            expected.setdefault(ext.adapter_id, set()).add(ext.instance_id)
    scored_ids: Dict[str, set] = {}
    for row in scored:
        scored_ids.setdefault(str(row["adapter_id"]), set()).add(str(row["instance_id"]))
    errors = set(error_ids)
    rejected = set(rejected_ids)
    counts = export_receipt.get("counts") or {}
    out: Dict[str, Any] = {}
    for adapter_id in sorted(set(expected) | set(scored_ids) | set(counts)):
        want = expected.get(adapter_id, set())
        got = scored_ids.get(adapter_id, set())
        out[adapter_id] = {
            "supported": len(want),
            "scored": len(got),
            "missing": len(want - got - errors - rejected),
            "endpoint_errors": len(want & errors),
            "rejected": len(want & rejected),
            "quarantined": int((counts.get(adapter_id) or {}).get("quarantined", 0)),
            "quarantine_reasons": dict((counts.get(adapter_id) or {}).get("quarantine_reasons", {})),
        }
    out["unsupported_adapters"] = dict(export_receipt.get("unsupported_adapters") or {})
    return out


# ======================================================================================
# entry points


def score_run(
    responses: str | Path,
    store: str | Path,
    receipt: str | Path,
    out: str | Path,
    *,
    cond: Optional[str] = None,
    family: Optional[str] = None,
    store_sha256: Optional[str] = None,
    allow_unfrozen: bool = False,
    allow_settings_mismatch: bool = False,
    prompt_check: bool = True,
) -> Dict[str, Any]:
    """Score one trace-tool run directory's ``responses.jsonl``; returns the summary.

    The overrides (``allow_unfrozen``, ``allow_settings_mismatch``, ``prompt_check=False``)
    are recorded in ``score_receipt.json`` and ``summary.json`` (``run.checks``).
    """
    responses = Path(responses)
    receipt = Path(receipt)
    out = Path(out)
    export_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    if export_receipt.get("receipt_version") != EXPORT_VERSION or export_receipt.get("command") != "export":
        raise CertificateVersionError(f"{receipt} is not a {EXPORT_VERSION} export receipt")
    loaded = load_store(store)
    store_match = verify_store(loaded, export_receipt, store_sha256=store_sha256)
    info = resolve_run(
        responses,
        export_receipt,
        cond=cond,
        family=family,
        allow_unfrozen=allow_unfrozen,
        allow_settings_mismatch=allow_settings_mismatch,
    )
    records, failures, n_lines = read_responses(responses)
    error_ids = [r["instance_id"] for r in failures if infrastructure_failure(r) == "endpoint_error"]
    rejected_ids = [r["instance_id"] for r in failures if infrastructure_failure(r) == "rejected"]

    rows: List[Dict[str, Any]] = []
    adapters: Dict[str, Adapter] = {}
    for record in records:
        ext = loaded.for_instance(record["instance_id"])
        if info.split is not None and ext.base.get("split") != info.split:
            raise ExtensionMismatchError(f"instance {ext.instance_id} is in split {ext.base.get('split')}, not {info.split}")
        if prompt_check:
            check_record_prompt(record, ext, info)
        adapter = adapters.get(ext.adapter_id)
        if adapter is None:
            adapter = adapters[ext.adapter_id] = registry.adapter_for_extension(ext)
        rows.append(score_record(record, ext, adapter, info))

    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "cert_scores.jsonl"
    _write_jsonl(scores_path, rows)
    denominators = _denominators(loaded, info.split, rows, error_ids, export_receipt, rejected_ids)
    summary = summarize_rows(rows, denominators=denominators)
    settings = info.run_config.get("settings") if isinstance(info.run_config.get("settings"), Mapping) else {}
    checks = {**info.checks(), "prompt_check": bool(prompt_check), "prompts_checked": len(rows) if prompt_check else 0}
    summary["run"] = {
        "model": info.run_config.get("model"),
        "model_id": info.run_config.get("model_id"),
        "condition": info.run_config.get("condition"),
        "seed": info.run_config.get("seed"),
        "cond": info.cond,
        "split": info.split,
        "coord_frame": info.coord_frame,
        "family": info.family,
        "settings": {key: settings.get(key) for key in SETTINGS_RECORDED},
        "checks": checks,
    }
    summary_path = out / "summary.json"
    _write_json(summary_path, summary)
    run_config_keys = ("model", "model_id", "model_revision", "condition", "seed", "dataset", "settings",
                       "trace_tool_version", "git_commit", "git_dirty", "code_sha256", "python", "started_at")
    score_receipt = {
        "receipt_version": EXPORT_VERSION,
        "command": "score",
        "scorer_version": SCORER_VERSION,
        "responses": {
            "path": str(responses), "sha256": file_sha256(responses), "lines": n_lines, "scored": len(rows),
            "endpoint_errors": len(error_ids), "rejected": len(rejected_ids),
        },
        "run_config": {key: info.run_config.get(key) for key in run_config_keys},
        "run_config_sha256": info.run_config_sha256,
        "checks": checks,
        "store": {"path": str(Path(store)), "sha256": loaded.sha256, "rows": len(loaded.by_id), "match": store_match},
        "export_receipt": {"path": str(receipt), "sha256": file_sha256(receipt), "split": export_receipt.get("split")},
        "run": summary["run"],
        "adapters": {aid: adapter.identity() for aid, adapter in sorted(adapters.items())},
        "outputs": {
            "cert_scores.jsonl": {"sha256": file_sha256(scores_path), "rows": len(rows)},
            "summary.json": {"sha256": file_sha256(summary_path)},
        },
        "provenance": code_provenance(),
    }
    _write_json(out / "score_receipt.json", score_receipt)
    return summary


def read_scores(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def pooled_denominators(paths: Sequence[Path]) -> Dict[str, Any]:
    """Sum each input's run denominators (its sibling ``summary.json``) per model, condition
    and split; an input without one is refused (review CONF-06)."""
    pooled: Dict[str, Any] = {}
    for path in paths:
        sibling = path.with_name("summary.json")
        if not sibling.exists():
            raise CertificateVersionError(f"{path} has no sibling summary.json with its denominators; pool runs scored by `score`")
        run_summary = json.loads(sibling.read_text(encoding="utf-8"))
        run = run_summary.get("run") or {}
        key = f"{run.get('model')}|{run.get('cond')}|{run.get('split')}"
        group = pooled.setdefault(key, {"model": run.get("model"), "cond": run.get("cond"), "split": run.get("split"),
                                        "runs": 0, "by_adapter": {}})
        group["runs"] += 1
        for adapter_id, counts in (run_summary.get("denominators") or {}).items():
            if not isinstance(counts, Mapping) or "supported" not in counts:
                continue
            total = group["by_adapter"].setdefault(adapter_id, {name: 0 for name in DENOMINATOR_KEYS})
            for name in DENOMINATOR_KEYS:
                total[name] += int(counts.get(name, 0) or 0)
    return pooled


def summarize_files(paths: Sequence[str | Path], out: str | Path) -> Dict[str, Any]:
    """Pool several ``cert_scores.jsonl`` files (e.g. decoding seeds) per model and condition.

    The pooled denominators (supported, scored, missing, endpoint errors, rejected,
    quarantined) are summed from each input's ``summary.json``."""
    paths = [Path(p) for p in paths]
    rows = read_scores(paths)
    denominators = pooled_denominators(paths)
    summary = summarize_rows(rows, run_keys=("model", "cond"), denominators=denominators)
    summary["by_seed"] = _group(rows, ("model", "cond", "seed", "adapter_id", "split"))
    out = Path(out)
    _write_json(out, summary)
    receipt = {
        "receipt_version": EXPORT_VERSION,
        "command": "summarize",
        "scorer_version": SCORER_VERSION,
        "inputs": {str(p): file_sha256(p) for p in paths},
        "input_summaries": {str(p.with_name("summary.json")): file_sha256(p.with_name("summary.json")) for p in paths},
        "rows": len(rows),
        "denominators": denominators,
        "outputs": {out.name: {"sha256": file_sha256(out)}},
        "provenance": code_provenance(),
    }
    _write_json(out.with_name(out.stem + ".receipt.json"), receipt)
    return summary


__all__ = [
    "DENOMINATOR_KEYS",
    "LEGACY_CONDITIONS",
    "RunInfo",
    "SUMMARY_VERSION",
    "Store",
    "aggregate",
    "check_record_prompt",
    "infrastructure_failure",
    "load_store",
    "model_family",
    "pooled_denominators",
    "read_responses",
    "read_scores",
    "resolve_run",
    "score_record",
    "score_run",
    "settings_problems",
    "summarize_files",
    "transcript_prompt",
    "summarize_rows",
    "verify_store",
]
