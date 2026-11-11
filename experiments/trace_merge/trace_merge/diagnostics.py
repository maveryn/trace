"""Protocol comparison, environment drift, and length/format/judge diagnostics (handoff 4.4, 5.3; DESIGN.md 3).

Why: reusing archived outputs is valid only where content, prompt, generation, preprocessing, stopping and scoring
contracts match the comparison being made. These checks make the match (or the documented difference) explicit:

* **Protocol comparison** (``audit/protocol_comparison.json``): per run x seed x benchmark slice, the dataset revision,
  normalized generation settings, scorer ids, prompt/completion token distributions and finish reasons; across the
  archived models, agreement of ``prompt_sha256`` and of a *model-neutral* request/prompt-contract digest. The public
  ``request_sha256`` (and the ``prompt_contract_sha256`` that binds it) include ``model_id`` and ``seed`` in the
  generation settings, so they differ across models by construction; the neutral digests recompute them with the
  exporter's helpers after dropping only those two documented identity fields.
* **Prompt conformance of local runs**: every registered local run is compared row by row with the archived reference
  of its protocol (P1: archived ``B``; P2: archived ``V@P0``; a P0 rerun: its own archived label), keyed by
  (source, run_id, label, seed): ``prompt_sha256``, the neutral request digest and ``prompt_tokens`` must be equal on
  every row. Any non-conforming run makes the command exit non-zero.
* **Scorer identity**: ingestion forces every ``scorer_id`` to ``<benchmark>-primary-v1``, so equal ids say nothing
  about scorer code; the report compares the archive parts' producer code and score-contract digests per model and
  lists each local run's code hash and commits next to them.
* **Environment drift**: an archived slice against a local rerun of the same model and seed (same pinned code, data and
  decoding; different hardware), with every pair of archived seeds of that model as the resampling reference.
* **Length/format/judge audit** (``tables/length_format_judge_audit.csv`` with a column dictionary next to it): token
  distributions, recorded length stops (never inferred from token counts) cross-tabulated with scores and extraction
  outcomes, protocol extraction status, judge use (extraction flag, scorer route, benchmark scoring contract) and
  deterministic final-answer markers, with missing values kept missing.

Everything is computed slice by slice (one archived seed, or one local run, at a time) to bound memory.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import StudyConfig, sha256_bytes, write_json
from .ingest import (CONFORMANCE_COLUMNS, EXPORTER, IDENTITY_SETTING_FIELDS, MEDIA_CONTRACT, IngestError,
                     aligned_index, conformance_reference_label, load_aggregates, load_aligned, prompt_conformance,
                     protocol_module)
from .ingest import normalized_settings_json as _normalized_settings_json

Log = Callable[[str], None]
SLICE_KEYS = ["source", "run_id", "protocol_id", "model_label", "seed", "benchmark_id"]
PROTOCOL_COLUMNS = ["source", "run_id", "protocol_id", "model_label", "seed", "benchmark_id", "content_key",
                    "dataset_revision", "dataset_sha256", "runtime_revision", "prompt_sha256", "request_sha256",
                    "prompt_contract_sha256", "generation_settings_json", "scorer_id", "prompt_tokens",
                    "completion_tokens", "finish_reason", "ordered_media_sha256", "source_row_sha256",
                    "source_record_sha256", "media_set_sha256", "source_ordinal", "source_index",
                    "archived_source_row_sha256", "archived_source_record_sha256", "row_identity_status"]
DRIFT_COLUMNS = ["benchmark_id", "content_key", "archived_row_score", "score_available", "score_kind",
                 "response_sha256", "completion_tokens", "prompt_tokens", "finish_reason", "request_sha256",
                 "prompt_sha256", "extraction_value_json", "extraction_status"]
AUDIT_COLUMNS = ["source", "run_id", "protocol_id", "model_label", "seed", "benchmark_id", "response_text",
                 "finish_reason", "completion_tokens", "extraction_status", "extraction_method",
                 "extraction_value_json", "used_judge", "archived_row_score", "score_available", "row_scorer"]
AGREEMENT_FIELDS = ("prompt_sha256", "neutral_request_sha256", "neutral_prompt_contract_sha256",
                    "prompt_contract_sha256")
KNOWN_DIFFERENCES = [
    "prompt_contract_sha256 and request_sha256 bind generation settings that include model_id and seed, so raw "
    "agreement across models (or seeds) is 0 by construction; compare the neutral digests instead.",
    "T's served view lacks <|endoftext|> (151643) as an extra stop id; B and V list it (DESIGN.md section 3).",
    "V's native chat template replaces the system message with a ~490-token reasoning prompt and pre-fills "
    "'<think>\\n'; its prompt tokens are longer although prompt_sha256 (the pre-template message payload) matches.",
    "The archive's dataset_sha256 (producer dataset manifest) differs from the prepared clone's whole-manifest digest; "
    "row identity is checked per row against the clone manifest instead (see 'dataset'). The neutral prompt-contract "
    "digest binds dataset_sha256, so it differs between archived and local rows by construction.",
    "The existing local reruns were served from the protocol clone's prepared commit snapshots (each local run "
    "report's served_model.served_view records the file set and its content revision). For T this is the full "
    "18-file snapshot, while the archived T used the content-bound 10-file runtime view recorded in "
    "manifests/model_roles.json (sha256set:8396b0be...). Prompt tokens agree on every row, but the T drift comparison "
    "conflates hardware with this view difference.",
    "Archived parts record the producer's code identity (producer_code_revision / producer_code_sha256); local runs "
    "record the spool's trace_git_commit and final25_code_hash. Both are listed under 'scorer_code', not assumed "
    "equal.",
    "Every local rerun shares its sampling seed with the archived slice it is compared with, so its row-score "
    "agreement and identical-text rate mix seed coupling with environment drift; the archived seed pairs use "
    "different seeds and bound resampling noise only.",
]

# deterministic final-answer markers (presence only; no claim about correctness)
MARKERS: dict[str, re.Pattern[str]] = {
    "answer_tag": re.compile(r"<answer>.*?</answer>", re.S | re.I),
    "boxed": re.compile(r"\\boxed\s*\{"),
    "answer_is": re.compile(r"\bthe\s+(?:final\s+|correct\s+)?answer\s+is\b", re.I),
    "answer_colon_line": re.compile(r"(?im)^\s*[*#_ ]*(?:final\s+)?answer\s*[*_]*\s*[:：]"),
    "think_close": re.compile(r"</think>", re.I),
}
# the extraction judge's own statement that it found no answer (it then often guesses one); deterministic sentinel
JUDGE_INCOMPLETE = re.compile(r"^\s*the model(?:'s)? response (?:is incomplete|does not)", re.I)
JUDGE_SCORER = re.compile(r"judge|qwen3", re.I)
JUDGE_CONTRACT = re.compile(r"judge|qwen3-32b", re.I)
EXTRACTION_METHODS = ("judge", "deterministic_parser", "benchmark_adapter")

AUDIT_COLUMN_DOC = {
    "n_rows": "rows in the slice",
    "completion_tokens_*": "engine-reported completion tokens over rows where known (n_known next to it)",
    "n_tokens_at_or_above_cap": "rows whose completion tokens reach max_tokens; NOT a length stop by itself",
    "finish_reason_*": "recorded finish reasons; length_stop_rate = length / rows with a known finish reason",
    "length_stop_score_available / _score_zero": "length-stopped rows with an official row score / scored 0",
    "length_stop_value_is_response": "length-stopped rows whose extracted value is the raw response",
    "length_stop_judge_incomplete_sentinel": "length-stopped rows whose extracted value is the judge's 'incomplete' "
                                             "statement",
    "length_stop_marker_any_final": "length-stopped rows that still contain a final-answer marker",
    "extraction_<status>": "protocol extraction status counts (resolved, ambiguous, invalid, excluded)",
    "extraction_status_not_resolved_rate": "share of rows whose protocol status is not 'resolved'. This is the "
                                           "exporter's status, not an answer-extraction failure measure: truncated "
                                           "or unparseable responses are usually 'resolved' with the raw text as "
                                           "value",
    "extraction_method_*": "rows per extraction method",
    "extraction_value_is_response_rate": "share of rows whose extracted value equals the (stripped) response: the "
                                         "official scorer parses the response itself. Normal on the benchmark_adapter "
                                         "route, so not a failure measure",
    "judge_incomplete_sentinel_rows": "rows whose extracted value starts with the judge's 'The model response is "
                                      "incomplete / does not ...' statement (deterministic sentinel of no answer)",
    "extraction_failure_rate": "not reported: no deterministic extraction-failure measure applies across the suite "
                               "(on several routes the value is the raw response by design); use the sentinel, the "
                               "markers and the length_stop_* cross-tab",
    "used_judge_rate": "export's extraction used_judge flag over rows where known",
    "row_scorer_judge_rows / row_scorer_known": "local runs: rows scored by a judge-backed scorer (scorer name matches "
                                                "judge/qwen3) / rows with a recorded scorer; null for the archive, which "
                                                "does not record per-row scorers",
    "scoring_route / score_contract_names_judge": "suite scoring route, and whether the suite score contract or "
                                                  "benchmark provenance names a judge (judge-backed official scoring "
                                                  "the used_judge flag may not show)",
    "marker_*_rate": "deterministic final-answer markers over rows with known text (presence only)",
}


# --------------------------------------------------------------------------------------------- helpers
def normalized_settings_json(settings_json: str) -> str:
    """Generation settings with only the documented identity fields (model_id, seed) removed, canonical JSON."""

    return _normalized_settings_json(settings_json)


def _media_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(json.loads(value))
    return tuple(str(v) for v in list(value))


def add_neutral_digests(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute request and prompt-contract digests with model_id/seed removed (exporter helpers, cached)."""

    exp = protocol_module(EXPORTER)
    out = df.copy()
    norm_cache: dict[str, str] = {}
    out["normalized_settings_json"] = [norm_cache.setdefault(s, normalized_settings_json(s))
                                       for s in out["generation_settings_json"]]
    req_cache: dict[tuple[Any, ...], str] = {}
    con_cache: dict[tuple[Any, ...], str] = {}
    requests, contracts = [], []
    for prompt, media, settings, dsha, srow, srec, mset in zip(
            out["prompt_sha256"], out["ordered_media_sha256"], out["normalized_settings_json"],
            out["dataset_sha256"], out["source_row_sha256"], out["source_record_sha256"], out["media_set_sha256"]):
        media_t = _media_tuple(media)
        rkey = (prompt, media_t, settings)
        req = req_cache.get(rkey)
        if req is None:
            req = exp.public_request_sha256_from_hashes(prompt_digest=prompt, ordered_media_sha256=list(media_t),
                                                        generation_settings=json.loads(settings))
            req_cache[rkey] = req
        ckey = (dsha, srow, srec, mset, prompt, req)
        con = con_cache.get(ckey)
        if con is None:
            con = exp.prompt_contract_sha256(dataset_sha256=dsha, source_row_sha256=srow, source_record_digest=srec,
                                             media_digest=mset, prompt_digest=prompt, request_digest=req)
            con_cache[ckey] = con
        requests.append(req)
        contracts.append(con)
    out["neutral_request_sha256"] = requests
    out["neutral_prompt_contract_sha256"] = contracts
    return out


def _stats(values: pd.Series) -> dict[str, Any]:
    known = pd.to_numeric(values, errors="coerce").dropna()
    if known.empty:
        return {"n_known": 0, "n_null": int(len(values)), "mean": None, "median": None, "p95": None, "min": None,
                "max": None}
    return {"n_known": int(len(known)), "n_null": int(len(values) - len(known)), "mean": float(known.mean()),
            "median": float(known.median()), "p95": float(known.quantile(0.95)), "min": float(known.min()),
            "max": float(known.max())}


def _counts(values: pd.Series) -> dict[str, int]:
    return {("null" if pd.isna(k) else str(k)): int(v) for k, v in values.value_counts(dropna=False).items()}


def counter_median(counts: Mapping[Any, int]) -> float | None:
    """Exact median of a multiset given as value -> count (numpy convention for an even count)."""

    n = int(sum(counts.values()))
    if n == 0:
        return None
    items = sorted((float(k), int(c)) for k, c in counts.items())
    lo_rank, hi_rank = (n - 1) // 2, n // 2
    out, seen = [], 0
    for value, c in items:
        for rank in (lo_rank, hi_rank):
            if seen <= rank < seen + c:
                out.append(value)
        seen += c
    return float(sum(out) / len(out))


def run_key(source: str, run_id: str, label: str) -> str:
    """Readable key of one run of one label: archive rows come from a single run, local runs are named by run id."""

    return f"{source}/{label}" if source == "archive" else f"{source}/{run_id}/{label}"


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


# --------------------------------------------------------------------------------------------- protocol comparison
def slice_summaries(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for key, g in df.groupby(SLICE_KEYS, sort=True, dropna=False):
        entry = dict(zip(SLICE_KEYS, [_scalar(k) for k in key]))
        entry.update({
            "rows": int(len(g)),
            "dataset_revisions": sorted(set(g["dataset_revision"])),
            "runtime_revisions": sorted(set(g["runtime_revision"])),
            "normalized_settings_sha256": sorted({sha256_bytes(s.encode("utf-8"))
                                                  for s in g["normalized_settings_json"]}),
            "scorer_ids": sorted(set(g["scorer_id"])),
            "prompt_tokens": _stats(g["prompt_tokens"]),
            "completion_tokens": _stats(g["completion_tokens"]),
            "finish_reasons": _counts(g["finish_reason"]),
        })
        out.append(entry)
    return out


def cross_label_agreement(df: pd.DataFrame, *, by: Sequence[str] = ("seed", "benchmark_id")) -> pd.DataFrame:
    """Row-level agreement between every pair of model labels on shared content keys, per ``by`` group.

    Each label must come from one run (source, run_id), otherwise the pairing would be ambiguous. Rows carry counts
    (``*_equal_n``, ``prompt_tokens_known_both``) and the multiset of paired prompt-token differences so that groups
    can be pooled exactly.
    """

    runs = df.groupby("model_label")[["source", "run_id"]].nunique() if "source" in df.columns else None
    if runs is not None and (runs > 1).any(axis=None):
        raise IngestError("cross-label agreement needs one run per label; compare local runs with prompt_conformance")
    labels = sorted(df["model_label"].unique())
    cols = list(by) + ["content_key", *AGREEMENT_FIELDS, "prompt_tokens"]
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            left = df[df["model_label"] == a][cols]
            right = df[df["model_label"] == b][cols]
            m = left.merge(right, on=list(by) + ["content_key"], suffixes=("_a", "_b"), validate="one_to_one")
            for key, g in m.groupby(list(by), sort=True):
                key = key if isinstance(key, tuple) else (key,)
                row = dict(zip(by, [_scalar(k) for k in key]))
                row.update({"label_a": a, "label_b": b, "shared_rows": int(len(g))})
                for f in AGREEMENT_FIELDS:
                    eq = int((g[f"{f}_a"] == g[f"{f}_b"]).sum())
                    row[f"{f}_equal_n"] = eq
                    row[f"{f}_equal_rate"] = eq / len(g) if len(g) else None
                pa_, pb = pd.to_numeric(g["prompt_tokens_a"]), pd.to_numeric(g["prompt_tokens_b"])
                both = pa_.notna() & pb.notna()
                diff = (pb[both] - pa_[both])
                row["prompt_tokens_known_both"] = int(both.sum())
                row["prompt_tokens_equal_n"] = int((diff == 0).sum())
                row["prompt_tokens_equal_rate"] = float((diff == 0).mean()) if both.any() else None
                row["prompt_tokens_median_a"] = float(pa_.median()) if pa_.notna().any() else None
                row["prompt_tokens_median_b"] = float(pb.median()) if pb.notna().any() else None
                row["prompt_tokens_median_paired_diff_b_minus_a"] = float(diff.median()) if both.any() else None
                row["_diff_counts"] = dict(Counter(int(v) for v in diff))
                rows.append(row)
    return pd.DataFrame(rows)


def protocol_parts(df: pd.DataFrame) -> dict[str, Any]:
    """Mergeable pieces of the protocol comparison for one frame (one archived seed, or one local run)."""

    work = add_neutral_digests(df)
    settings: dict[str, list[str]] = {}
    for key, g in work.groupby(SLICE_KEYS, sort=True):
        for s in sorted(set(g["normalized_settings_json"])):
            settings.setdefault(s, []).append("/".join(str(k) for k in key))
    archive = work[work["source"] == "archive"]
    agreement = cross_label_agreement(archive) if len(archive) else pd.DataFrame()
    scorers: dict[str, dict[str, set[str]]] = {}
    finish: dict[str, Counter] = {}
    prompt_values: dict[str, Counter] = {}
    for (src, rid, lab, bench), g in work.groupby(["source", "run_id", "model_label", "benchmark_id"], sort=True):
        rk = run_key(src, rid, lab)
        scorers.setdefault(str(bench), {}).setdefault(rk, set()).update(g["scorer_id"])
        finish.setdefault(rk, Counter()).update(_counts(g["finish_reason"]))
        prompt_values.setdefault(rk, Counter()).update(
            Counter(int(v) for v in pd.to_numeric(g["prompt_tokens"]).dropna()))
    return {"settings": settings, "slices": slice_summaries(work), "agreement": agreement, "scorers": scorers,
            "finish": finish, "prompt_values": prompt_values,
            "dataset_revisions": {src: set(g["dataset_revision"]) for src, g in work.groupby("source")}}


def summarize_protocol_parts(parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool protocol parts exactly (sums of counts, medians from pooled multisets)."""

    parts = list(parts)
    settings: dict[str, list[str]] = {}
    slices: list[dict[str, Any]] = []
    scorers: dict[str, dict[str, set[str]]] = {}
    finish: dict[str, Counter] = {}
    prompt_values: dict[str, Counter] = {}
    revisions: dict[str, set[str]] = {}
    for p in parts:
        for s, v in p["settings"].items():
            settings.setdefault(s, []).extend(v)
        slices.extend(p["slices"])
        for bench, per in p["scorers"].items():
            for rk, ids in per.items():
                scorers.setdefault(bench, {}).setdefault(rk, set()).update(ids)
        for rk, c in p["finish"].items():
            finish.setdefault(rk, Counter()).update(c)
        for rk, c in p["prompt_values"].items():
            prompt_values.setdefault(rk, Counter()).update(c)
        for src, v in p["dataset_revisions"].items():
            revisions.setdefault(src, set()).update(v)
    frames = [p["agreement"] for p in parts if len(p["agreement"])]
    agreement = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    pair_summary: dict[str, Any] = {}
    if len(agreement):
        for (a, b), g in agreement.groupby(["label_a", "label_b"], sort=True):
            shared = int(g["shared_rows"].sum())
            known = int(g["prompt_tokens_known_both"].sum())
            diffs: Counter = Counter()
            for d in g["_diff_counts"]:
                diffs.update(d)
            pair_summary[f"{a}|{b}"] = {
                "shared_rows": shared,
                **{f"{f}_equal_rate": (float(g[f"{f}_equal_n"].sum()) / shared if shared else None)
                   for f in AGREEMENT_FIELDS},
                "prompt_tokens_known_both": known,
                "prompt_tokens_equal_rate": float(g["prompt_tokens_equal_n"].sum()) / known if known else None,
                "prompt_tokens_median_paired_diff_b_minus_a": counter_median(diffs),
            }
        agreement = agreement.drop(columns=["_diff_counts"])
    scorer_ids = {bench: {"by_run": {rk: sorted(v) for rk, v in per.items()},
                          "identical": len({tuple(sorted(v)) for v in per.values()}) == 1}
                  for bench, per in sorted(scorers.items())}
    return {
        "normalized_settings": {"distinct": len(settings), "dropped_identity_fields": list(IDENTITY_SETTING_FIELDS),
                                "values": {s: {"slices": len(v), "examples": v[:4]} for s, v in settings.items()}},
        "dataset_revisions": {src: sorted(v) for src, v in sorted(revisions.items())},
        "cross_model_pairs": pair_summary,
        "cross_model_by_seed_benchmark": agreement.to_dict("records"),
        "scorers": {"all_identical_across_labels": all(v["identical"] for v in scorer_ids.values()),
                    "note": "scorer ids are forced to '<benchmark>-primary-v1' by ingestion; scorer code identity is "
                            "under 'scorer_code'",
                    "by_benchmark": scorer_ids},
        "finish_reasons": {rk: dict(c) for rk, c in sorted(finish.items())},
        "prompt_tokens_median": {rk: counter_median(c) for rk, c in sorted(prompt_values.items())},
        "slices": slices,
    }


def compare_protocols(df: pd.DataFrame) -> dict[str, Any]:
    """Pure protocol comparison over rows with PROTOCOL_COLUMNS (one frame; see audit_protocols for pooling)."""

    return summarize_protocol_parts([protocol_parts(df)])


def dataset_identity(cfg: StudyConfig, df: pd.DataFrame) -> dict[str, Any]:
    """Per-benchmark row snapshot digest of archived rows vs the prepared manifest's per-dataset receipt.

    Uses the identity exactly as archived (``archived_source_*`` where a producer alias rewrote it), so a benchmark
    whose rows were matched through an alias is reported as a mismatch with its alias status, not hidden.
    """

    mc = protocol_module(MEDIA_CONTRACT)
    manifest = json.loads(cfg.dataset_manifest.read_bytes())
    per = {}
    for (bench, label, seed), g in df[df["source"] == "archive"].groupby(["benchmark_id", "model_label", "seed"]):
        if bench in per:
            continue
        g = g.sort_values("source_ordinal")
        row_hash = g["archived_source_row_sha256"].where(g["archived_source_row_sha256"].notna(),
                                                         g["source_row_sha256"])
        rec_hash = g["archived_source_record_sha256"].where(g["archived_source_record_sha256"].notna(),
                                                            g["source_record_sha256"])
        rows = [{"ordinal": int(o), "index": str(i), "source_row_hash": a, "media_set_sha256": m,
                 "source_record_sha256": r}
                for o, i, a, m, r in zip(g["source_ordinal"], g["source_index"], row_hash, g["media_set_sha256"],
                                         rec_hash)]
        receipt = (manifest.get("datasets") or {}).get(bench) or {}
        digest = mc.dataset_snapshot_sha256(rows)
        aliased = int(g["archived_source_row_sha256"].notna().sum())
        per[bench] = {"archive_rows": len(rows), "archive_row_snapshot_sha256": digest,
                      "manifest_dataset_snapshot_sha256": receipt.get("dataset_snapshot_sha256"),
                      "match": digest == receipt.get("dataset_snapshot_sha256"), "rows_identity_aliased": aliased,
                      "identity_status": sorted(set(g["row_identity_status"])), "from_slice": f"{label}/{seed}"}
    archive_shas = sorted(set(df.loc[df["source"] == "archive", "dataset_sha256"]))
    return {"per_benchmark": per, "all_benchmarks_match": all(v["match"] for v in per.values()),
            "benchmarks_matched_via_alias": sorted(b for b, v in per.items() if not v["match"]
                                                   and v["rows_identity_aliased"] == v["archive_rows"]),
            "archive_dataset_sha256": archive_shas,
            "clone_manifest_dataset_snapshot_sha256": manifest.get("dataset_snapshot_sha256"),
            "whole_snapshot_equal": archive_shas == [manifest.get("dataset_snapshot_sha256")],
            "note": "the whole-snapshot digest binds the producer's manifest (schema, suite sha, dataset set); "
                    "per-benchmark row snapshots are the content check"}


def archive_part_provenance(cfg: StudyConfig) -> dict[str, Any]:
    """Distinct producer provenance values per config and model from the archive's part manifests."""

    root = cfg.archive_run_root / "metadata" / "parts"
    fields = ("producer_code_revision", "producer_code_sha256", "evaluation_harness_revision",
              "campaign_config_sha256", "dataset_sha256", "source_contract_sha256")
    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for path in sorted(root.rglob("*.manifest.json")):
        part = json.loads(path.read_text(encoding="utf-8"))
        cfg_name = str(part.get("config_name"))
        model = str((part.get("identity") or {}).get("model_id"))
        prov = part.get("provenance") or {}
        bucket = out.setdefault(cfg_name, {}).setdefault(model, {f: [] for f in fields})
        for f in fields:
            v = prov.get(f)
            if v is not None and str(v) not in bucket[f]:
                bucket[f].append(str(v))
    return out


def scorer_code_identity(part_provenance: Mapping[str, Any],
                         local_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Scorer code identity per archived model (score parts) and per local run (spool provenance)."""

    scores = part_provenance.get("scores") or {}
    keys = ("producer_code_revision", "producer_code_sha256", "evaluation_harness_revision", "source_contract_sha256")
    per_model = {m: {k: sorted(v.get(k) or []) for k in keys} for m, v in sorted(scores.items())}
    local = {}
    for name, rep in sorted(local_reports.items()):
        prov = rep.get("spool_provenance") or {}
        local[name] = {k: prov.get(k) for k in ("trace_git_commit", "vlmevalkit_git_commit", "final25_code_hash",
                                                 "campaign_config_hash")}
        local[name]["score_contract_versions"] = (prov.get("contract_versions") or {}).get("score")
    return {"archive_score_parts": per_model,
            "archive_identical_across_models": len({json.dumps(v, sort_keys=True) for v in per_model.values()}) <= 1,
            "local_runs": local,
            "local_code_hashes": sorted({v["final25_code_hash"] for v in local.values() if v["final25_code_hash"]})}


def local_run_conformance(cfg: StudyConfig, frame: pd.DataFrame) -> dict[str, Any]:
    """Prompt conformance of one local run's rows against its protocol's archived reference, per seed."""

    out: dict[str, Any] = {}
    for (src, rid, label, protocol_id, seed), g in frame.groupby(["source", "run_id", "model_label", "protocol_id",
                                                                   "seed"], sort=True):
        ref_label = conformance_reference_label(str(label), str(protocol_id))
        reference = load_aligned(cfg, columns=CONFORMANCE_COLUMNS, labels=[ref_label], seeds=[int(seed)],
                                 benchmarks=sorted(set(g["benchmark_id"])))
        res = prompt_conformance(g, reference)
        out[run_key(src, rid, label) + f"/seed={int(seed)}"] = {
            "reference": f"archive/{ref_label}/seed={int(seed)}", "conforms": res["conforms"], "total": res["total"],
            "nonconforming_benchmarks": res["nonconforming_benchmarks"]}
    return out


# --------------------------------------------------------------------------------------------- environment drift
def compare_runs(reference: pd.DataFrame, other: pd.DataFrame, reference_aggs: pd.DataFrame | None = None,
                 other_aggs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Row-level and official comparison of two runs of the same model (reference = archived slice).

    Rows join one-to-one on (benchmark_id, content_key). Row-score agreement uses only rows whose official row score
    is available in both runs (binary/grouped row benchmarks); aggregate-only benchmarks contribute official deltas.
    Pooled summary rates are computed over the rows on which each quantity is known in both runs.
    """

    m = reference[DRIFT_COLUMNS].merge(other[DRIFT_COLUMNS], on=["benchmark_id", "content_key"], how="outer",
                                       suffixes=("_ref", "_oth"), indicator=True, validate="one_to_one")
    ref_aggs = {} if reference_aggs is None else dict(zip(reference_aggs["benchmark_id"],
                                                          reference_aggs["official_value"]))
    oth_aggs = {} if other_aggs is None else dict(zip(other_aggs["benchmark_id"], other_aggs["official_value"]))
    rows = []
    for bench, g in m.groupby("benchmark_id", sort=False):
        both = g[g["_merge"] == "both"]
        row: dict[str, Any] = {
            "benchmark_id": bench, "rows_reference": int((g["_merge"] != "right_only").sum()),
            "rows_other": int((g["_merge"] != "left_only").sum()), "rows_matched": int(len(both)),
            "score_kind": both["score_kind_ref"].iloc[0] if len(both) else None}
        for f in ("request_sha256", "prompt_sha256", "response_sha256", "extraction_value_json"):
            row[f"{f}_equal_rate"] = float((both[f"{f}_ref"] == both[f"{f}_oth"]).mean()) if len(both) else None
        pr = pd.to_numeric(both["prompt_tokens_ref"]), pd.to_numeric(both["prompt_tokens_oth"])
        known = pr[0].notna() & pr[1].notna()
        row["prompt_tokens_known_both"] = int(known.sum())
        row["prompt_tokens_equal_rate"] = float((pr[0][known] == pr[1][known]).mean()) if known.any() else None
        cr, co = pd.to_numeric(both["completion_tokens_ref"]), pd.to_numeric(both["completion_tokens_oth"])
        known = cr.notna() & co.notna()
        row.update({
            "completion_tokens_mean_reference": float(cr.mean()) if cr.notna().any() else None,
            "completion_tokens_mean_other": float(co.mean()) if co.notna().any() else None,
            "completion_tokens_known_both": int(known.sum()),
            "completion_tokens_mean_paired_shift": float((co[known] - cr[known]).mean()) if known.any() else None,
            "completion_tokens_median_reference": float(cr.median()) if cr.notna().any() else None,
            "completion_tokens_median_other": float(co.median()) if co.notna().any() else None,
            "length_stops_reference": int((both["finish_reason_ref"] == "length").sum()),
            "length_stops_other": int((both["finish_reason_oth"] == "length").sum()),
        })
        scored = both[both["score_available_ref"].astype(bool) & both["score_available_oth"].astype(bool)]
        if len(scored):
            sr = scored["archived_row_score_ref"].astype(float)
            so = scored["archived_row_score_oth"].astype(float)
            row.update({
                "rows_scored_both": int(len(scored)), "row_score_agreement_rate": float((sr == so).mean()),
                "both_correct": int(((sr == 1) & (so == 1)).sum()), "both_wrong": int(((sr == 0) & (so == 0)).sum()),
                "reference_only_correct": int(((sr == 1) & (so == 0)).sum()),
                "other_only_correct": int(((sr == 0) & (so == 1)).sum()),
                "row_mean_reference_pct": float(100 * sr.mean()), "row_mean_other_pct": float(100 * so.mean())})
        else:
            row.update({"rows_scored_both": 0, "row_score_agreement_rate": None})
        ra, oa = ref_aggs.get(bench), oth_aggs.get(bench)
        row.update({"official_reference": ra, "official_other": oa,
                    "official_delta_other_minus_reference": (oa - ra) if ra is not None and oa is not None else None})
        rows.append(row)
    table = pd.DataFrame(rows)
    deltas = table["official_delta_other_minus_reference"].dropna()
    complete = len(deltas) == len(table) and len(table) > 0
    both = m[m["_merge"] == "both"]
    pr, po = pd.to_numeric(both["prompt_tokens_ref"]), pd.to_numeric(both["prompt_tokens_oth"])
    pk = pr.notna() & po.notna()
    cr, co = pd.to_numeric(both["completion_tokens_ref"]), pd.to_numeric(both["completion_tokens_oth"])
    ck = cr.notna() & co.notna()
    scored = both[both["score_available_ref"].astype(bool) & both["score_available_oth"].astype(bool)]
    summary = {
        "benchmarks": int(len(table)), "rows_matched": int(len(both)),
        "rows_unmatched": int((m["_merge"] != "both").sum()),
        "official_macro_reference": float(table["official_reference"].mean()) if complete else None,
        "official_macro_other": float(table["official_other"].mean()) if complete else None,
        "official_macro_delta": float(deltas.mean()) if complete else None,
        "official_delta_mean_abs": float(deltas.abs().mean()) if len(deltas) else None,
        "official_delta_max_abs": float(deltas.abs().max()) if len(deltas) else None,
        "row_score_agreement_rate_pooled": (float((scored["archived_row_score_ref"].astype(float)
                                                   == scored["archived_row_score_oth"].astype(float)).mean())
                                            if len(scored) else None),
        "rows_scored_both": int(len(scored)),
        "response_text_equal_rate": float((both["response_sha256_ref"] == both["response_sha256_oth"]).mean())
        if len(both) else None,
        "request_sha256_equal_rate": float((both["request_sha256_ref"] == both["request_sha256_oth"]).mean())
        if len(both) else None,
        "prompt_tokens_known_both": int(pk.sum()),
        "prompt_tokens_equal_rate": float((pr[pk] == po[pk]).mean()) if pk.any() else None,
        "completion_tokens_known_both": int(ck.sum()),
        "completion_tokens_mean_paired_shift": float((co[ck] - cr[ck]).mean()) if ck.any() else None,
        "length_stops_reference": int((both["finish_reason_ref"] == "length").sum()),
        "length_stops_other": int((both["finish_reason_oth"] == "length").sum()),
    }
    return table, summary


DRIFT_RANGE_METRICS = ("official_macro_delta", "official_delta_mean_abs", "official_delta_max_abs",
                       "row_score_agreement_rate_pooled", "response_text_equal_rate",
                       "completion_tokens_mean_paired_shift")


def environment_drift(cfg: StudyConfig, label: str, seed: int, local_name: str, *,
                      reference_seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Archived ``label``/``seed`` slice vs a local rerun, with every archived seed pair of ``label`` as reference.

    One seed pair is a single draw of resampling noise; all pairs of the archived seeds give its observed range. The
    length-stop count of every archived seed is listed next to the rerun's so a systematic shift is visible.
    """

    seeds = [int(s) for s in (reference_seeds or cfg.raw["archive"]["seeds"])]
    if int(seed) not in seeds:
        seeds = [int(seed), *seeds]
    archived = {s: load_aligned(cfg, columns=DRIFT_COLUMNS, labels=[label], seeds=[s]) for s in seeds}
    loc = load_aligned(cfg, columns=DRIFT_COLUMNS, include_archive=False, local_runs=[local_name])
    aggs = load_aggregates(cfg, local_runs=[local_name])
    arch_aggs = {s: aggs[(aggs["source"] == "archive") & (aggs["model_label"] == label) & (aggs["seed"] == s)]
                 for s in seeds}
    l_aggs = aggs[(aggs["source"] != "archive") & (aggs["model_label"] == label) & (aggs["seed"] == seed)]
    table, summary = compare_runs(archived[int(seed)], loc, arch_aggs[int(seed)], l_aggs)
    out: dict[str, Any] = {"label": label, "seed": seed, "local_run": local_name, "summary": summary,
                           "per_benchmark": table.to_dict("records")}
    pairs: dict[str, Any] = {}
    pair_tables: dict[str, list[dict[str, Any]]] = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            t, s = compare_runs(archived[a], archived[b], arch_aggs[a], arch_aggs[b])
            pairs[f"{a}-{b}"] = s
            pair_tables[f"{a}-{b}"] = t.to_dict("records")
    ranges = {}
    for metric in DRIFT_RANGE_METRICS:
        values = [v[metric] for v in pairs.values() if v.get(metric) is not None]
        rerun = summary.get(metric)
        position = None
        if values and rerun is not None:
            position = "below" if rerun < min(values) else ("above" if rerun > max(values) else "within")
        ranges[metric] = {"rerun": rerun, "seed_pairs_min": min(values) if values else None,
                          "seed_pairs_max": max(values) if values else None, "rerun_vs_seed_pair_range": position}
    if ranges["official_macro_delta"]["rerun"] is not None and pairs:
        abs_pairs = [abs(v["official_macro_delta"]) for v in pairs.values() if v["official_macro_delta"] is not None]
        ranges["official_macro_delta"]["rerun_abs_within_seed_pairs_max_abs"] = (
            abs(ranges["official_macro_delta"]["rerun"]) <= max(abs_pairs)) if abs_pairs else None
    stops = {str(s): int((archived[s]["finish_reason"] == "length").sum()) for s in seeds}
    rerun_stops = int((loc["finish_reason"] == "length").sum())
    out["seed_reference"] = {
        "comparison": f"archive {label}: every pair of seeds {seeds}", "pairs": pairs, "per_benchmark": pair_tables,
        "ranges": ranges,
        "length_stops": {"archived_by_seed": stops, "rerun": rerun_stops,
                         "rerun_outside_archived_range": not (min(stops.values()) <= rerun_stops
                                                              <= max(stops.values()))},
        "note": "the rerun shares seed "
                f"{seed} with the archived slice it is compared with (seed coupling); seed pairs differ in seed",
    }
    return out


# --------------------------------------------------------------------------------------------- length/format/judge
def _extracted_text(value_json: Any) -> str | None:
    if value_json is None or (isinstance(value_json, float) and np.isnan(value_json)):
        return None
    try:
        value = json.loads(value_json)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, str) else None


def length_format_judge_audit(df: pd.DataFrame, *, max_tokens: int | None = None) -> pd.DataFrame:
    """Per run x model_label x seed x benchmark response diagnostics, null-preserving (columns: AUDIT_COLUMN_DOC).

    Length stops are counted only from the recorded ``finish_reason``; ``n_tokens_at_or_above_cap`` is reported
    separately (when ``max_tokens`` is given) and is not a length stop. Rates use the rows where the underlying field is
    known as their denominator; the number of unknown rows is reported next to each.
    """

    keys = [k for k in ("source", "run_id", "protocol_id", "model_label", "seed", "benchmark_id") if k in df.columns]
    work = df.copy()
    text = work["response_text"]
    known_text = text.notna()
    for name, pattern in MARKERS.items():
        work[f"_m_{name}"] = text.where(known_text).map(lambda t, p=pattern: None if t is None or (
            isinstance(t, float) and np.isnan(t)) else bool(p.search(t)))
    final = work[[f"_m_{n}" for n in ("answer_tag", "boxed", "answer_is", "answer_colon_line")]]
    work["_m_any_final"] = pd.Series(np.where(final.notna().any(axis=1), final.eq(True).any(axis=1), None),
                                     index=work.index, dtype=object)
    has_value = "extraction_value_json" in work.columns
    if has_value:
        values = [_extracted_text(v) for v in work["extraction_value_json"]]
        work["_value_is_response"] = [None if v is None or not isinstance(t, str) else v.strip() == t.strip()
                                      for v, t in zip(values, text)]
        work["_sentinel"] = [v is not None and bool(JUDGE_INCOMPLETE.match(v)) for v in values]
    rows = []
    for key, g in work.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, [_scalar(k) for k in key]))
        n = len(g)
        tok = pd.to_numeric(g["completion_tokens"], errors="coerce")
        row.update({
            "n_rows": n, "completion_tokens_n_known": int(tok.notna().sum()),
            "completion_tokens_mean": float(tok.mean()) if tok.notna().any() else None,
            "completion_tokens_median": float(tok.median()) if tok.notna().any() else None,
            "completion_tokens_p95": float(tok.quantile(0.95)) if tok.notna().any() else None,
            "completion_tokens_max": float(tok.max()) if tok.notna().any() else None,
        })
        if max_tokens is not None:
            row["n_tokens_at_or_above_cap"] = int((tok >= max_tokens).sum())
        fr = g["finish_reason"]
        fr_known = fr.notna()
        length = fr == "length"
        row.update({
            "finish_reason_n_known": int(fr_known.sum()), "finish_reason_stop": int((fr == "stop").sum()),
            "finish_reason_length": int(length.sum()),
            "finish_reason_other": int((fr_known & ~fr.isin(["stop", "length"])).sum()),
            "finish_reason_null": int((~fr_known).sum()),
            "length_stop_rate": float(length.sum() / fr_known.sum()) if fr_known.any() else None,
        })
        if "score_available" in g.columns:
            avail = g["score_available"].fillna(False).astype(bool)
            score = pd.to_numeric(g["archived_row_score"], errors="coerce")
            row["length_stop_score_available"] = int((length & avail).sum())
            row["length_stop_score_zero"] = int((length & avail & (score == 0)).sum())
        if has_value:
            row["length_stop_value_is_response"] = int((length & g["_value_is_response"].eq(True)).sum())
            row["length_stop_judge_incomplete_sentinel"] = int((length & g["_sentinel"]).sum())
        row["length_stop_marker_any_final"] = int((length & g["_m_any_final"].eq(True)).sum())
        st = g["extraction_status"]
        st_known = st.notna()
        for s in ("resolved", "ambiguous", "invalid", "excluded"):
            row[f"extraction_{s}"] = int((st == s).sum())
        row["extraction_status_null"] = int((~st_known).sum())
        row["extraction_status_not_resolved_rate"] = (float((st_known & (st != "resolved")).sum() / st_known.sum())
                                                      if st_known.any() else None)
        if "extraction_method" in g.columns:
            meth = g["extraction_method"]
            for m_ in EXTRACTION_METHODS:
                row[f"extraction_method_{m_}"] = int((meth == m_).sum())
            row["extraction_method_other"] = int((meth.notna() & ~meth.isin(EXTRACTION_METHODS)).sum())
        if has_value:
            vir = g["_value_is_response"]
            row["extraction_value_is_response_rate"] = float(vir[vir.notna()].astype(bool).mean()) \
                if vir.notna().any() else None
            row["judge_incomplete_sentinel_rows"] = int(g["_sentinel"].sum())
        uj = g["used_judge"]
        uj_known = uj.notna()
        row["used_judge_n_known"] = int(uj_known.sum())
        row["used_judge_rate"] = float(uj[uj_known].astype(bool).mean()) if uj_known.any() else None
        if "row_scorer" in g.columns:
            rs = g["row_scorer"]
            rs_known = rs.notna()
            row["row_scorer_known"] = int(rs_known.sum())
            row["row_scorer_judge_rows"] = int(rs[rs_known].astype(str).str.contains(JUDGE_SCORER).sum()) \
                if rs_known.any() else None
        tk = g["response_text"].notna()
        row["response_text_n_known"] = int(tk.sum())
        for name in list(MARKERS) + ["any_final"]:
            col = g[f"_m_{name}"]
            k = col.notna()
            row[f"marker_{name}_rate"] = float(col[k].astype(bool).mean()) if k.any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_scoring_routes(cfg: StudyConfig) -> pd.DataFrame:
    """Per benchmark: suite scoring route and whether its official scoring contract names a judge."""

    suite = {b["key"]: b for b in cfg.suite()["benchmarks"]}
    prov_path = cfg.repo_path(cfg.raw["benchmark_provenance"])
    prov = {b["benchmark_id"]: b for b in json.loads(prov_path.read_text(encoding="utf-8")).get("benchmarks") or []}
    rows = []
    for key, b in suite.items():
        texts = [str(b.get("score_contract") or ""), str((prov.get(key) or {}).get("official_scorer") or "")]
        rows.append({"benchmark_id": key, "scoring_route": b.get("route"),
                     "score_contract_names_judge": any(JUDGE_CONTRACT.search(t) for t in texts)})
    return pd.DataFrame(rows)


def write_length_format_judge_audit(cfg: StudyConfig, *, local_runs: Sequence[str] | None = None,
                                    log: Log = print) -> Path:
    """Compute the audit slice by slice (bounded memory) and write tables/length_format_judge_audit.csv (+ columns)."""

    max_tokens = int(cfg.raw["protocol"]["generation"]["max_tokens"])
    frames = []
    arch = load_aligned(cfg, columns=["model_label", "seed"])
    pairs = sorted(set(zip(arch["model_label"], arch["seed"])))
    del arch
    for label, seed in pairs:
        part = load_aligned(cfg, columns=AUDIT_COLUMNS, labels=[label], seeds=[int(seed)])
        frames.append(length_format_judge_audit(part, max_tokens=max_tokens))
        log(f"[diagnostics] length/format/judge audit: {label} seed {seed}")
        del part
    for name in local_runs or []:
        part = load_aligned(cfg, columns=AUDIT_COLUMNS, include_archive=False, local_runs=[name])
        frames.append(length_format_judge_audit(part, max_tokens=max_tokens))
        log(f"[diagnostics] length/format/judge audit: {name}")
        del part
    columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    # all-null columns (e.g. row_scorer_* on archive slices) are left to reindex instead of typing the concat
    table = pd.concat([f.dropna(axis=1, how="all") for f in frames], ignore_index=True).reindex(columns=columns)
    table = table.merge(benchmark_scoring_routes(cfg), on="benchmark_id", how="left", validate="many_to_one")
    path = cfg.work("tables", "length_format_judge_audit.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    write_json(path.with_name("length_format_judge_audit.columns.json"),
               {"table": path.name, "max_tokens": max_tokens, "columns": AUDIT_COLUMN_DOC})
    return path


# --------------------------------------------------------------------------------------------- orchestration
def _local_reruns(cfg: StudyConfig) -> dict[str, dict[str, Any]]:
    try:
        return {k: v for k, v in aligned_index(cfg).get("local_runs", {}).items()}
    except IngestError:
        return {}


def audit_protocols(cfg: StudyConfig, *, local_runs: Sequence[str] | None = None, drift: bool = True,
                    log: Log = print) -> dict[str, Any]:
    """Write audit/protocol_comparison.json (and environment-drift tables for existing local reruns).

    Processes one archived seed, then one local run, at a time; pools the parts exactly.
    """

    registered = _local_reruns(cfg)
    names = list(local_runs) if local_runs is not None else sorted(registered)
    seeds = sorted({int(s) for s in load_aligned(cfg, columns=["seed"])["seed"]})
    report: dict[str, Any] = {"schema": "trace_merge_protocol_comparison_v2", "study_id": cfg.study_id,
                              "config_sha256": cfg.sha256, "local_runs": names,
                              "known_differences": KNOWN_DIFFERENCES}
    parts = []
    dataset: dict[str, Any] | None = None
    for seed in seeds:
        df = load_aligned(cfg, columns=PROTOCOL_COLUMNS, seeds=[seed])
        log(f"[diagnostics] protocol comparison: archive seed {seed} ({len(df)} rows)")
        parts.append(protocol_parts(df))
        if dataset is None:
            dataset = dataset_identity(cfg, df)
        del df
    conformance: dict[str, Any] = {}
    local_reports: dict[str, dict[str, Any]] = {}
    for name in names:
        df = load_aligned(cfg, columns=PROTOCOL_COLUMNS, include_archive=False, local_runs=[name])
        log(f"[diagnostics] protocol comparison: local run {name} ({len(df)} rows)")
        parts.append(protocol_parts(df))
        conformance[name] = local_run_conformance(cfg, df)
        del df
        rpath = cfg.work("aligned", registered[name]["report"])
        if rpath.is_file():
            local_reports[name] = json.loads(rpath.read_text(encoding="utf-8"))
    report.update(summarize_protocol_parts(parts))
    del parts
    report["dataset"] = dataset
    report["prompt_conformance"] = conformance
    report["all_local_runs_conform"] = all(v["conforms"] for per in conformance.values() for v in per.values())
    provenance = archive_part_provenance(cfg)
    report["archive_part_provenance"] = provenance
    report["scorer_code"] = scorer_code_identity(provenance, local_reports)
    report["local_run_provenance"] = {
        name: {k: rep.get(k) for k in ("spool_identity", "spool_provenance", "served_model", "public_model_id",
                                       "protocol_id", "source", "run_id", "warnings")}
        for name, rep in local_reports.items()}
    if drift:
        report["environment_drift"] = {}
        tables = []
        for name in names:
            entry = registered[name]
            if entry.get("source") != "existing_local_rerun":
                continue
            d = environment_drift(cfg, entry["model_label"], int(entry["seed"]), name)
            write_json(cfg.work("audit", f"environment_drift_{name}.json"), d)
            ref = d["seed_reference"]
            report["environment_drift"][name] = {"summary": d["summary"], "seed_pairs": ref["pairs"],
                                                 "ranges": ref["ranges"], "length_stops": ref["length_stops"]}
            t = pd.DataFrame(d["per_benchmark"])
            t.insert(0, "comparison", f"archive {entry['model_label']} seed {entry['seed']} vs {name}")
            tables.append(t)
            for pair, recs in ref["per_benchmark"].items():
                r = pd.DataFrame(recs)
                r.insert(0, "comparison", f"archive {entry['model_label']} seeds {pair}")
                tables.append(r)
            log(f"[diagnostics] environment drift {name}: {json.dumps(d['summary'])}")
        if tables:
            path = cfg.work("tables", "environment_drift.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(tables, ignore_index=True).to_csv(path, index=False)
    write_json(cfg.work("audit", "protocol_comparison.json"), report)
    return report


# --------------------------------------------------------------------------------------- answer-format sensitivity
# A diagnostic, never a score: how many officially wrong multiple-choice rows end with an option letter that maps to
# the correct answer when the letter is read leniently (bare, \( X \), $X$, \boxed{X}). It separates "the model chose
# the right option but wrote it in a form the pinned scorer does not read" from "the model chose a wrong option".

FINAL_LETTER = re.compile(r"(?:\\\(\s*\\?(?:text\{)?([A-J])\}?\s*\\\)|\$\s*([A-J])\s*\$|\\boxed\{\s*([A-J])\s*\}|"
                          r"\b([A-J])\b)[\s.\)\}]*$")
LATEX_LETTER_END = re.compile(r"\\\(\s*[A-J]\s*\\\)\s*\.?$")
# benchmark -> how its licensed answer relates to an option letter
LETTER_ANSWER_BENCHMARKS = {"puzzlevqa": "option_text", "visualpuzzles": "letter"}


def final_option_letter(text: str) -> str | None:
    m = FINAL_LETTER.search((text or "").strip()[-80:])
    return next((g for g in (m.groups() if m else ()) if g), None)


def answer_format_sensitivity(rows: pd.DataFrame, scores: pd.Series, meta: pd.DataFrame, benchmark: str) -> dict[str, Any]:
    """Rates over one (label, seed, benchmark) slice: official accuracy, responses ending in a LaTeX-wrapped letter,
    and officially-0 rows whose leniently read final letter is the correct option (``meta`` from
    ``metrics.load_benchmark_meta``, indexed by source ordinal)."""

    import ast

    kind = LETTER_ANSWER_BENCHMARKS[benchmark]
    m = meta.set_index("source_ordinal").loc[rows["source_ordinal"].astype(int)]
    texts = rows["response_text"].fillna("").astype(str).to_numpy()
    answers = m["answer"].astype(str).str.strip().to_numpy()
    options = m["options"].to_numpy() if "options" in m.columns else [None] * len(rows)
    lenient_hits = latex_end = 0
    for s, text, ans, opts in zip(scores.to_numpy(), texts, answers, options):
        latex_end += bool(LATEX_LETTER_END.search(text.strip()[-60:]))
        if s != 0:
            continue
        letter = final_option_letter(text)
        if letter is None:
            continue
        if kind == "letter":
            lenient_hits += letter == ans
        else:
            try:
                opt = ast.literal_eval(opts) if isinstance(opts, str) else list(opts or [])
            except (ValueError, SyntaxError):
                opt = []
            k = ord(letter) - ord("A")
            lenient_hits += 0 <= k < len(opt) and str(opt[k]).strip().lower() == ans.lower()
    n = len(rows)
    return {"benchmark_id": benchmark, "n_rows": n, "official_accuracy_pct": 100.0 * float(np.mean(scores)) if n else None,
            "latex_letter_ending_pct": 100.0 * latex_end / n if n else None,
            "official_zero_but_final_letter_correct_pct": 100.0 * lenient_hits / n if n else None,
            "definition": "diagnostic only; final option letter read leniently (bare, \\( X \\), $X$, \\boxed{X}); "
                          "PuzzleVQA letters mapped to option text, VisualPuzzles answers are letters"}


# --------------------------------------------------------------------------------------------- judge health
# DESIGN.md A8: one run's judge returned outputs that did not belong to their prompts. MathVerse is judge-extracted for
# every model and nearly every response ends in \boxed{}, so the share of rows whose extracted answer equals that boxed
# answer is a model-independent check that the judge read the response it was given.
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
JUDGE_AGREEMENT_FLOOR_PCT = 30.0
JUDGED_TABLE = "scoring/seed_*/runs/mathverse/*/*/*_judged_*.xlsx"
JUDGE_AGREEMENT_COLUMNS = ["view_slug", "subset", "seed", "quarantined", "rows", "rows_with_boxed",
                           "boxed_equals_extracted_pct", "below_floor"]


def last_boxed(text: Any) -> str | None:
    found = BOXED.findall(str(text or ""))
    return found[-1].strip() if found else None


def boxed_extract_agreement(judged: pd.DataFrame) -> dict[str, Any]:
    """Rows of one judged MathVerse table whose judge-extracted answer equals the response's last \\boxed{} answer
    (rows without a boxed answer are left out of the share)."""

    boxed = judged["prediction"].map(last_boxed)
    has = boxed.notna()
    equal = boxed[has] == judged.loc[has, "extract"].astype(str).str.strip()
    pct = 100.0 * float(equal.mean()) if has.any() else None
    return {"rows": len(judged), "rows_with_boxed": int(has.sum()), "boxed_equals_extracted_pct": pct,
            "below_floor": pct is not None and pct < JUDGE_AGREEMENT_FLOOR_PCT}


def write_judge_agreement(cfg: StudyConfig, *, log: Log = print) -> pd.DataFrame:
    """tables/judge_extraction_agreement.csv: the check above for every local run's MathVerse score slice, including
    slices moved to a run's ``quarantine_*`` directory (marked)."""

    runs = cfg.work("runs")
    found = [(p, False) for p in runs.glob(f"*/*/{JUDGED_TABLE}")]
    found += [(p, True) for p in runs.glob(f"*/*/quarantine_*/{JUDGED_TABLE}")]
    out = []
    for path, quarantined in sorted(found):
        parts = path.relative_to(runs).parts
        seed_dir = next(p for p in parts if p.startswith("seed_"))
        res = boxed_extract_agreement(pd.read_excel(path, usecols=["prediction", "extract"]))
        out.append({"view_slug": parts[0], "subset": parts[1], "seed": int(seed_dir.split("_", 1)[1]),
                    "quarantined": quarantined, **res})
        if res["below_floor"] and not quarantined:
            log(f"[diagnostics] JUDGE FAULT? {parts[0]} {parts[1]} {seed_dir}: {res['boxed_equals_extracted_pct']:.1f}% "
                f"of boxed answers extracted as written (floor {JUDGE_AGREEMENT_FLOOR_PCT:.0f}%)")
    table = pd.DataFrame(out, columns=JUDGE_AGREEMENT_COLUMNS)
    path = cfg.work("tables", "judge_extraction_agreement.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    log(f"[diagnostics] judge extraction agreement: {len(table)} MathVerse slices -> {path}")
    return table


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .config import ensure_work_tree, load_config

    p = argparse.ArgumentParser(prog="trace_merge.diagnostics", description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["protocols", "length-audit", "judge-agreement", "all"])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--protocol-clone", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config, work_root=args.work_root, protocol_clone=args.protocol_clone)
    ensure_work_tree(cfg)
    status = 0
    if args.command in ("protocols", "all"):
        rep = audit_protocols(cfg)
        drift = {name: {"summary": v["summary"], "ranges": v["ranges"], "length_stops": v["length_stops"]}
                 for name, v in (rep.get("environment_drift") or {}).items()}
        print(json.dumps({"cross_model_pairs": rep["cross_model_pairs"],
                          "normalized_settings_distinct": rep["normalized_settings"]["distinct"],
                          "scorer_code_archive_identical": rep["scorer_code"]["archive_identical_across_models"],
                          "local_code_hashes": rep["scorer_code"]["local_code_hashes"],
                          "dataset_per_benchmark_match": rep["dataset"]["all_benchmarks_match"],
                          "prompt_conformance": {n: {k: (v["conforms"], v["total"]) for k, v in per.items()}
                                                 for n, per in rep["prompt_conformance"].items()},
                          "environment_drift": drift}, indent=1))
        if not rep["all_local_runs_conform"]:
            print("FATAL: a local run's rendered inputs do not conform to its archived reference")
            status = 4
    if args.command in ("length-audit", "all"):
        path = write_length_format_judge_audit(cfg, local_runs=sorted(_local_reruns(cfg)))
        print(f"wrote {path.relative_to(cfg.require_work_root())}")
    if args.command in ("judge-agreement", "all"):
        table = write_judge_agreement(cfg)
        flagged = table[table["below_floor"].astype(bool) & ~table["quarantined"].astype(bool)]
        if len(flagged):
            print(f"FATAL: {len(flagged)} MathVerse score slice(s) below the judge-agreement floor (DESIGN.md A8)")
            status = 4
    return status


if __name__ == "__main__":
    import sys

    sys.exit(main())
