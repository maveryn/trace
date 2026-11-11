"""Analysis stage of the follow-up (design/followup_v1.md §4-§6): ingestion, F0 scoring, component effects.

``ingest_runs`` (CLI ``ingest-runs``) ingests every complete follow-up run with the reviewed study's
``trace_merge.ingest.ingest_local_run``, unchanged, through a small shim:

* ``FollowupIngestConfig`` is the reviewed ``StudyConfig`` re-rooted on the follow-up work root. Four read-only inputs
  of the reviewed work root keep their reviewed location (the archive table and aggregates for the prompt-conformance
  reference, the archive ingest report for the decoding contract, the metric registry); every other path, and every
  write, resolves under the follow-up work root. The view registry the reviewed check reads (``manifests/views.json``)
  is derived from ``views/*/view_manifest.json`` and the hybrid registry (read where the reviewed code reads
  ``manifests/merges.json``) from ``hybrids/<ID>/manifest.json``; both are rewritten on every ingest.
* The follow-up labels (``TF@P1``, ...) are added to ``trace_merge.ingest.MODEL_ROLES`` only inside
  ``followup_roles`` (restored on exit) and to the shim's in-memory evaluation matrix. Nothing is patched on disk.

So every reviewed validation runs as written: spool and descriptor digests, identity joins to the prepared dataset
manifest, plan coverage and the frozen split's row list, provenance (pinned protocol and VLMEvalKit commits),
decoding settings, served model = registered view marker (= the parent pin or the hybrid's content revision), prompt
conformance with archived ``B`` (P1) or ``V@P0`` (P2), spool aggregate = the campaign summary. The shim adds three
checks: the frozen split files equal the reviewed ones byte for byte, a hybrid view's served weight files hash to the
hybrid manifest's output hashes (and the build verification passed), and the judge-agreement health check (reviewed
DESIGN.md A8; MathVerse boxed vs judge-extracted answers, floor 30%).

After ingestion each run is scored once with the frozen F0 rule (``rescore.normalized_row_scores``: official and
``format-normalized-v2`` row scores) into ``aligned/scored/<run>.parquet``, and the acceptance rule of the reviewed
study is checked per benchmark: the adapter's recomputed campaign value (``campaign_value``; WeMath's fixed-525 value,
reviewed DESIGN.md A2) must reproduce the run's own score-campaign aggregate to 1e-6 pp. A benchmark failing on any
run of an analysis leaves that analysis's macro, which is then renamed (``statistics.macro_name``).

Incomplete runs (no ``validation.seed<S>.json`` with ``complete: true``) are skipped with a status; a run whose state
changed since its ingestion (``analysis.run_fingerprint``) is ingested and scored again. Every step is idempotent.

``analyze`` (CLI ``analyze``) computes design §6 on the confirmation partition (P1, seeds 42-44): the seven-cell
performance table, g_F, g_L, I, full gain and reversion effects per parent with the paired cluster bootstrap of
``trace_merge.statistics`` (10,000 replicates, seed 314159, all models and seeds attached to each bootstrap unit,
native metrics recomputed per replicate) on the official scale and, with the same draws, on format-normalized-v2;
Holm over the primary family on the official macro. Plus the P2 template-sensitivity matrix, the screen with the
technical-stop checks (design §4), behaviour on the validated binary benchmarks (reviewed DESIGN.md A3) and run costs.
p-values are two-sided and computed from the centered bootstrap distribution (design §6):
``p = (1 + #{|d* - d| >= |d|}) / (R + 1)``; the reviewed study's percentile-inversion p-value is reported beside it.

With ``provisional=True`` the analysis runs on the complete runs only (primary matrix on the seeds that every cell
has) and writes to ``tables/provisional/`` with every row marked ``PROVISIONAL``; the final call needs every run.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from . import rescore as rs
from .common import FollowupConfig, FollowupError, canonical_json, sha256_bytes, sha256_file, write_json
from .evaluate import tier_runs
from .views import view_specs

from trace_merge import analysis as an  # noqa: E402
from trace_merge import behavior as bh  # noqa: E402
from trace_merge import diagnostics as dg  # noqa: E402
from trace_merge import ingest as ig  # noqa: E402
from trace_merge import metrics as mt  # noqa: E402
from trace_merge import statistics as st  # noqa: E402
from trace_merge.config import StudyConfig, load_config  # noqa: E402

Log = Callable[[str], None]

F1_CELLS = ("B", "T", "V", "TF", "TL", "VF", "VL")
PARENTS = ("T", "V")
PRIMARY_INTERFACE = "P1"
TEMPLATE_INTERFACE = "P2"
OFFICIAL = "official"
NORMALIZED = rs.METRIC_NAME
SCALES = {OFFICIAL: "official_row_score", NORMALIZED: "normalized_row_score"}
PRIMARY_FAMILY = ("gF_T", "gL_T", "I_T", "gF_V", "gL_V", "I_V")
EFFECTS = ("gF", "gL", "I", "full", "revF", "revL")
EFFECT_MEANING = {
    "gF": "S(F_P, L_B) - S(B): parent frontend transplanted into B",
    "gL": "S(F_B, L_P) - S(B): parent language component with B's frontend",
    "I": "S(P) - S(F_P, L_B) - S(F_B, L_P) + S(B): interaction (scale-dependent)",
    "full": "S(P) - S(B) = gF + gL + I",
    "revF": "S(P) - S(F_B, L_P) = gF + I: loss when P's frontend is reverted to B's",
    "revL": "S(P) - S(F_P, L_B) = gL + I: loss when P's language component is reverted to B's",
}
IDENTITIES = (("full", ("gF", "gL", "I")), ("revF", ("gF", "I")), ("revL", ("gL", "I")))
REPRODUCTION_TOLERANCE_PP = 1e-6
IDENTITY_TOLERANCE_PP = 1e-9
EMPTY_STOP_PCT = 50.0          # design §4: more than 50% of responses empty
TRUNCATION_STOP_PCT = 80.0     # design §4: more than 80% truncated at the 4,096-token limit
JUDGE_HEALTH_MIN_BOXED = 50    # below this many boxed MathVerse rows the agreement share is recorded, not gating
STATUS_PROVISIONAL = "PROVISIONAL"
STATUS_FINAL = "FINAL"

# reviewed work-root inputs read in place (never written); everything else resolves under the follow-up work root
REVIEWED_READ_ONLY = frozenset({("aligned", "predictions.parquet"), ("aligned", "official_aggregates.parquet"),
                                ("audit", "archive_ingest.json"), ("manifests", "metric_registry.json")})
REGISTRY_ROUTES = {("manifests", "views.json"): ("manifests", "views.json"),
                   ("manifests", "merges.json"): ("manifests", "hybrids.json")}
SPLIT_FILES = ("split_manifest.json", "group_assignments.parquet", "bootstrap_units.json", "bootstrap_units.parquet")
SCORED_COLUMNS = (*rs.LEAN_COLUMNS, "official_excluded", "finish_reason", "completion_tokens", "used_judge",
                  "response_empty")
SCORER_FILES = ("rescore.py", "extract.py")


class FollowupAnalysisError(FollowupError):
    pass


# ======================================================================================================== ingest shim
@dataclass(frozen=True)
class FollowupIngestConfig(StudyConfig):
    """The reviewed ``StudyConfig`` with ``work_root`` = the follow-up work root (see module docstring)."""

    reviewed_root: Path | None = None

    def work(self, *parts: str) -> Path:
        key = tuple(parts)
        if key in REVIEWED_READ_ONLY:
            if self.reviewed_root is None:
                raise FollowupAnalysisError("the reviewed work root is required (TRACE_MERGE_WORK_ROOT)")
            return self.reviewed_root.joinpath(*parts)
        if key in REGISTRY_ROUTES:
            return super().work(*REGISTRY_ROUTES[key])
        return super().work(*parts)


def reviewed_root_of(fcfg: FollowupConfig, reviewed_root: Path | str | None = None) -> Path:
    given = reviewed_root or os.environ.get("TRACE_MERGE_WORK_ROOT")
    if not given:
        raise FollowupAnalysisError("the reviewed study's work root is required (--reviewed-root or "
                                    "TRACE_MERGE_WORK_ROOT); it is read, never written")
    root = Path(given).resolve()
    if root == fcfg.require_work_root().resolve():
        raise FollowupAnalysisError("the reviewed work root and the follow-up work root must differ")
    if not (root / "aligned" / "predictions.parquet").is_file():
        raise FollowupAnalysisError(f"{root} does not hold the reviewed archive table (aligned/predictions.parquet)")
    return root


def all_labels(fcfg: FollowupConfig) -> list[str]:
    """Every view label of the follow-up (F1 cells and any configured extension cells, both interfaces)."""

    return list(view_specs(fcfg))


def all_cells(fcfg: FollowupConfig) -> list[str]:
    return list(dict.fromkeys([*fcfg.cells, *(fcfg.raw.get("extension_cells") or {})]))


def shim_config(fcfg: FollowupConfig, reviewed_root: Path) -> FollowupIngestConfig:
    """The reviewed config for ingesting follow-up runs; its evaluation matrix declares the follow-up labels."""

    raw = copy.deepcopy(fcfg.base.raw)
    raw.setdefault("evaluation_matrix", {})["followup"] = {"models": all_labels(fcfg)}
    base = fcfg.base
    return FollowupIngestConfig(path=base.path, raw=raw, sha256=base.sha256, work_root=fcfg.require_work_root(),
                                protocol_clone=base.protocol_clone, reviewed_root=reviewed_root)


def reviewed_config(fcfg: FollowupConfig, reviewed_root: Path) -> StudyConfig:
    """The reviewed study's own config on its work root (metric registry, binary-benchmark list); read only."""

    return load_config(fcfg.base.path, work_root=reviewed_root, protocol_clone=fcfg.base.protocol_clone)


@contextlib.contextmanager
def followup_roles(fcfg: FollowupConfig) -> Iterator[None]:
    """``trace_merge.ingest.MODEL_ROLES`` extended by the follow-up cells, and ``spool_provenance`` taught the
    rescored-run split of design amendments A1/A3 (``overlay_tolerant_provenance``), for the duration of the block."""

    saved, saved_prov = ig.MODEL_ROLES, ig.spool_provenance
    ig.MODEL_ROLES = tuple(dict.fromkeys([*saved, *all_cells(fcfg)]))
    ig.spool_provenance = overlay_tolerant_provenance(fcfg, saved_prov)
    try:
        yield
    finally:
        ig.MODEL_ROLES, ig.spool_provenance = saved, saved_prov


OVERLAY_SPLIT_FIELDS = ("final25_code_hash", "campaign_config_hash")


def overlay_tolerant_provenance(fcfg: FollowupConfig, original: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """The reviewed provenance check, plus one exception (A1/A3): a run listed in
    ``dispatch/protocol_overlay/approved_code_hashes.json`` whose scoring half was redone under the private overlay
    carries the pinned shared clone's code hash on every generation descriptor and exactly one approved overlay code
    hash on every extraction/score descriptor (and one campaign-config hash per half). Every other provenance field
    must still be unique and pinned; that is checked by the reviewed function on a copy with the split unified."""

    def registry() -> dict[str, Any] | None:   # read on first use (the role patch alone needs no work root)
        reg_path = fcfg.work("dispatch", "protocol_overlay", "approved_code_hashes.json")
        return json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.is_file() else None

    def check(cfg: Any, spool: Mapping[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]]) -> dict[str, Any]:
        reg = registry()
        if reg is None:
            return original(cfg, spool)
        labels = {slug: label for label, (slug, _src, _proto) in view_specs(fcfg).items()}
        halves: dict[str, dict[str, set[str]]] = {"generation": {}, "scoring": {}}
        idents = set()
        for (_bench, stage), (desc, _records) in spool.items():
            prov = desc.get("provenance") or {}
            half = halves["generation" if stage == "generation" else "scoring"]
            for k in OVERLAY_SPLIT_FIELDS:
                half.setdefault(k, set()).add(str(prov.get(k) or ""))
            ident = desc.get("identity") or {}
            idents.add((str(ident.get("model_slug")), str(ident.get("run_id")), str(ident.get("seed"))))
        gen, sco = halves["generation"].get("final25_code_hash"), halves["scoring"].get("final25_code_hash")
        if not (len(idents) == 1 and gen == {reg["pinned_shared_clone"]} and sco and len(sco) == 1
                and next(iter(sco)) in reg["overlay"]
                and all(len(v) == 1 for h in halves.values() for v in h.values())):
            return original(cfg, spool)
        slug, run_id, seed = next(iter(idents))
        subset = run_id.removeprefix("trace_eval_v1_").rsplit("_temp06_seed", 1)[0]
        name = an.local_run_name(labels.get(slug, slug), subset, int(seed))
        if name not in reg["rescored_runs"]:
            raise FollowupAnalysisError(f"{name}: overlay-scored descriptors but not an approved rescored run")
        unified = {key: ({**desc, "provenance": {**(desc.get("provenance") or {}),
                                                 **{k: next(iter(halves["scoring"][k])) for k in OVERLAY_SPLIT_FIELDS}}},
                         records) for key, (desc, records) in spool.items()}
        out = original(cfg, unified)
        out["overlay_rescore"] = {"amendment": reg["rescored_runs"][name],
                                  "generation": {k: next(iter(halves["generation"][k])) for k in OVERLAY_SPLIT_FIELDS},
                                  "scoring": {k: next(iter(halves["scoring"][k])) for k in OVERLAY_SPLIT_FIELDS},
                                  "overlay": reg["overlay"][next(iter(sco))]}
        return out

    return check


def check_frozen_splits(fcfg: FollowupConfig, reviewed_root: Path) -> dict[str, str]:
    """The follow-up's split files are byte copies of the reviewed study's (design §4: copied, hash-checked)."""

    out = {}
    for name in SPLIT_FILES:
        mine, theirs = fcfg.work("splits", name), reviewed_root / "splits" / name
        if not mine.is_file() or not theirs.is_file():
            raise FollowupAnalysisError(f"splits/{name} is missing in the follow-up or the reviewed work root")
        digest = sha256_file(mine)
        if digest != sha256_file(theirs):
            raise FollowupAnalysisError(f"splits/{name} differs from the reviewed study's frozen copy")
        out[name] = digest
    return out


def hybrid_registry(fcfg: FollowupConfig) -> dict[str, dict[str, Any]]:
    """Built hybrids: content revision, manifest digest, output file hashes and the build verification result."""

    verification = {}
    vpath = fcfg.work("audit", "build_verification.json")
    if vpath.is_file():
        verification = json.loads(vpath.read_text(encoding="utf-8")).get("hybrids") or {}
    for extra in ("build_verification_F2c.json", "build_verification_F3.json"):   # decision_F2.md, decision_F3.md
        x_path = fcfg.work("audit", extra)
        if x_path.is_file():
            verification.update(json.loads(x_path.read_text(encoding="utf-8")))
    out = {}
    for hid in [*sorted(fcfg.hybrids), *sorted(fcfg.scaled)]:
        path = fcfg.constructed(hid)[1]
        if not path.is_file():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        out[hid] = {"content_revision": m["content_revision"], "manifest_sha256": sha256_file(path),
                    "output_file_hashes": m["output_file_hashes"], "recipe_sha256": m["recipe_sha256"],
                    "build_verified": bool((verification.get(hid) or {}).get("donor_equal")
                                           or (verification.get(hid) or {}).get("verified"))
                    and (verification.get(hid) or {}).get("content_revision") == m["content_revision"]}
    return out


def write_registries(fcfg: FollowupConfig) -> dict[str, Any]:
    """``manifests/views.json`` (from the view manifests) and ``manifests/hybrids.json`` (from the hybrid manifests),
    the registries the reviewed served-model check reads. Rewritten from the sources on every call."""

    views = {}
    for label, (slug, _source, _protocol) in view_specs(fcfg).items():
        path = fcfg.work("views", slug, "view_manifest.json")
        if path.is_file():
            m = json.loads(path.read_text(encoding="utf-8"))
            if m.get("label") != label or m.get("slug") != slug:
                raise FollowupAnalysisError(f"{path}: label/slug {m.get('label')}/{m.get('slug')} != {label}/{slug}")
            views[label] = m
    hybrids = hybrid_registry(fcfg)
    write_json(fcfg.work("manifests", "views.json"), views)
    write_json(fcfg.work("manifests", "hybrids.json"), hybrids)
    return {"views": views, "hybrids": hybrids}


def check_hybrid_view(fcfg: FollowupConfig, label: str, slug: str, hybrids: Mapping[str, Any]) -> dict[str, Any]:
    """A hybrid view serves the built hybrid: marker weight digests = hybrid output hashes, build verified."""

    source = view_specs(fcfg)[label][1]
    if fcfg.constructed(source) is None:
        return {"kind": "parent", "weight_source": source}
    entry = hybrids.get(source)
    if entry is None:
        raise FollowupAnalysisError(f"{label}: hybrid {source} has no manifest")
    marker = json.loads(fcfg.work("views", slug, ig.MODEL_MARKER).read_text(encoding="utf-8"))
    files = marker.get("file_sha256") or {}
    bad = sorted(n for n, h in entry["output_file_hashes"].items() if files.get(n) != h)
    if bad:
        raise FollowupAnalysisError(f"{label}: served weight files differ from hybrid {source}'s manifest: {bad[:3]}")
    if not entry["build_verified"]:
        raise FollowupAnalysisError(f"{label}: hybrid {source} has no passing audit/build_verification.json entry")
    return {"kind": "hybrid", "weight_source": source, "content_revision": entry["content_revision"],
            "weight_files_checked": len(entry["output_file_hashes"]), "build_verified": True}


def judge_health(run_dir: Path, seed: int) -> dict[str, Any]:
    """Reviewed DESIGN.md A8 health check on this seed's MathVerse judge outputs (``diagnostics``' own measure)."""

    tables = sorted(run_dir.glob(f"scoring/seed_{seed}/runs/mathverse/*/*/*_judged_*.xlsx"))
    if not tables:
        return {"status": "no_judged_table", "gating": False}
    res = [dg.boxed_extract_agreement(pd.read_excel(p, usecols=["prediction", "extract"])) for p in tables]
    boxed = sum(r["rows_with_boxed"] for r in res)
    pct = [r["boxed_equals_extracted_pct"] for r in res if r["boxed_equals_extracted_pct"] is not None]
    below = any(r["below_floor"] for r in res)
    gating = boxed >= JUDGE_HEALTH_MIN_BOXED
    status = "ok" if not below else ("below_floor" if gating else "below_floor_small_sample")
    return {"status": status, "gating": gating, "rows_with_boxed": boxed, "tables": len(tables),
            "boxed_equals_extracted_pct": min(pct) if pct else None, "floor_pct": dg.JUDGE_AGREEMENT_FLOOR_PCT}


# ============================================================================================= ingestion and scoring
@dataclass(frozen=True)
class RunKey:
    tier: str
    label: str
    slug: str
    subset: str
    seed: int

    @property
    def name(self) -> str:
        return an.local_run_name(self.label, self.subset, self.seed)

    @property
    def cell(self) -> str:
        return self.label.split("@", 1)[0]

    @property
    def interface(self) -> str:
        return self.label.split("@", 1)[1]


def planned_runs(fcfg: FollowupConfig, tiers: Sequence[str] | None = None,
                 cells: Sequence[str] | None = None) -> list[RunKey]:
    """Every planned run of the tiers (default all, F1 cells), one entry per (label, subset, seed)."""

    out: dict[tuple[str, str, int], RunKey] = {}
    for tier in tiers or list(fcfg.raw["evaluation"]):
        for label, slug, subset, seed in tier_runs(fcfg, tier, list(cells) if cells else list(fcfg.cells)):
            out.setdefault((label, subset, int(seed)), RunKey(tier, label, slug, subset, int(seed)))
    return list(out.values())


def scorer_code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    out = {f"trace_followup/{n}": sha256_file(here / n) for n in SCORER_FILES}
    out["trace_merge/metrics.py"] = sha256_file(Path(mt.__file__))
    return out


def scored_paths(fcfg: FollowupConfig, name: str) -> tuple[Path, Path]:
    d = fcfg.work("aligned", "scored")
    return d / f"{name}.parquet", d / f"{name}.check.json"


def campaign_reproduction(scored: pd.DataFrame, metrics: Mapping[str, Any],
                          aggregates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per benchmark: the adapter's campaign value on the run's rows vs the run's own score-campaign aggregate."""

    aggs = aggregates.set_index("benchmark_id")
    out = {}
    for bench, sub in scored.groupby("benchmark_id", sort=True):
        m = metrics[bench]
        totals = m.row_contributions(sub, sub["official_row_score"].astype(float)).sum(axis=0)
        adapter = float(m.finalize(totals))
        campaign = float(m.campaign_value(totals))
        agg = aggs.loc[bench]
        spool = float(agg["official_value"])
        err = abs(campaign - spool)
        status = str(agg["campaign_summary_status"])
        out[str(bench)] = {
            "rows": int(len(sub)), "adapter_value": adapter, "campaign_value": campaign, "campaign_aggregate": spool,
            "campaign_summary_value": None if pd.isna(agg["campaign_summary_value"]) else
            float(agg["campaign_summary_value"]), "campaign_summary_status": status, "abs_error_pp": err,
            "reproduced": bool(err <= REPRODUCTION_TOLERANCE_PP and status in ("equal", "summary_rescaled")),
            "units": "WeMath: fixed-525 campaign value (reviewed DESIGN.md A2)" if bench == "wemath" else "percent"}
    return out


def score_run(fcfg: FollowupConfig, shim: FollowupIngestConfig, key: RunKey, metrics: Mapping[str, Any], *,
              force: bool = False, log: Log = print) -> dict[str, Any]:
    """Official + format-normalized-v2 row scores of one ingested run (cached) and its campaign reproduction."""

    entry = an.registered_local_runs(shim)[key.name]
    pq_path, check_path = scored_paths(fcfg, key.name)
    code = scorer_code_hashes()
    if not force and pq_path.is_file() and check_path.is_file():
        check = json.loads(check_path.read_text(encoding="utf-8"))
        if (check.get("ingested_sha256") == entry["sha256"] and check.get("scorer_code_sha256") == code
                and check.get("scored_sha256") == sha256_file(pq_path)):
            return check
    cols = list(dict.fromkeys([*rs.REQUIRED_COLUMNS, "finish_reason", "completion_tokens", "used_judge",
                               "response_text"]))
    rows = ig.load_aligned(shim, columns=cols, include_archive=False, local_runs=[key.name])
    rows["response_empty"] = rows["response_text"].fillna("").astype(str).str.strip().eq("")
    rows = rows.drop(columns=["response_text"])
    with rs._quiet():
        scored = rs.normalized_row_scores(rows, metrics)
    aggs = pd.read_parquet(shim.work("aligned", entry["aggregates"]))
    repro = campaign_reproduction(scored, metrics, aggs)
    out = scored[list(SCORED_COLUMNS)].copy()
    out["completion_tokens"] = pd.to_numeric(out["completion_tokens"], errors="coerce").astype("Int64")
    for col in ("fn_rule", "fn_extracted", "fn_detail", "finish_reason"):
        out[col] = out[col].astype(object).where(out[col].notna(), None)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq_path.with_name(f".{pq_path.name}.tmp.{os.getpid()}")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, pq_path)
    check = {"name": key.name, "label": key.label, "subset": key.subset, "seed": key.seed,
             "ingested_sha256": entry["sha256"], "scorer": rs.METRIC_NAME, "scorer_code_sha256": code,
             "scored_sha256": sha256_file(pq_path), "rows": int(len(out)),
             "fn_status_counts": {str(k): int(v) for k, v in out["fn_status"].value_counts().sort_index().items()},
             "reproduction": repro, "all_reproduced": all(v["reproduced"] for v in repro.values()),
             "not_reproduced": sorted(b for b, v in repro.items() if not v["reproduced"]),
             "max_abs_error_pp": max(v["abs_error_pp"] for v in repro.values())}
    write_json(check_path, check)
    log(f"[score] {key.name}: {len(out)} rows, {check['fn_status_counts'].get(rs.REPAIRED, 0)} repaired, "
        f"campaign reproduction max |error| {check['max_abs_error_pp']:.2e} pp"
        + ("" if check["all_reproduced"] else f", NOT reproduced: {check['not_reproduced']}"))
    return check


def ingest_runs(fcfg: FollowupConfig, *, tiers: Sequence[str] | None = None, cells: Sequence[str] | None = None,
                force: bool = False, reviewed_root: Path | str | None = None, log: Log = print) -> dict[str, Any]:
    """Ingest and score every complete planned run; returns (and writes ``aligned/ingest_status.json``) a status per
    run. Re-running only touches runs that are new, changed (fingerprint) or whose scorer code changed."""

    root = reviewed_root_of(fcfg, reviewed_root)
    shim = shim_config(fcfg, root)
    splits = check_frozen_splits(fcfg, root)
    registries = write_registries(fcfg)
    metrics = rs.load_metrics(reviewed_config(fcfg, root))
    dmi = None
    status: dict[str, dict[str, Any]] = {}
    with followup_roles(fcfg):
        for key in planned_runs(fcfg, tiers, cells):
            rd = fcfg.work("runs", key.slug, key.subset)
            rec: dict[str, Any] = {"tier": key.tier, "label": key.label, "subset": key.subset, "seed": key.seed}
            index = an.registered_local_runs(shim)
            val = rd / f"validation.seed{key.seed}.json"
            if not val.is_file() or not json.loads(val.read_text(encoding="utf-8")).get("complete"):
                rec["status"] = "not_complete" + (" (an earlier ingest is stale)" if key.name in index else "")
                status[key.name] = rec
                continue
            rec["judge_health"] = health = judge_health(rd, key.seed)
            if health["status"] == "below_floor":
                rec["status"] = "judge_health_failed (rescore and quarantine the failed scoring, design §3)"
                status[key.name] = rec
                log(f"[ingest] {key.name}: judge agreement {health['boxed_equals_extracted_pct']:.1f}% below the floor")
                continue
            fp = an.run_fingerprint(rd, key.seed)
            current = key.name in index and an.ingested_fingerprint(shim, key.name) == fp
            try:
                hybrid_check = check_hybrid_view(fcfg, key.label, key.slug, registries["hybrids"])
                if current and not force:
                    rec["ingest"] = "already_ingested"
                else:
                    if dmi is None:
                        dmi = ig.load_dataset_manifest_index(shim.dataset_manifest, shim.benchmark_keys())
                    df = ig.ingest_local_run(shim, rd, key.label, None, key.seed, "local_run", manifest_index=dmi,
                                             log=lambda _m: None)
                    df.attrs["ingest_report"]["source_fingerprint"] = fp
                    df.attrs["ingest_report"]["followup_checks"] = {"weights": hybrid_check, "judge_health": health,
                                                                    "frozen_splits_sha256": splits}
                    ig.write_local_run(shim, df, key.name)
                    rec["ingest"] = ("re-ingested" if key.name in index else "ingested") + f" ({len(df)} rows)"
                    log(f"[ingest] {key.name}: {rec['ingest']}")
                    del df
                check = score_run(fcfg, shim, key, metrics, force=force, log=log)
            except (ig.IngestError, mt.MetricError, rs.RescoreError, FollowupError) as error:
                rec["status"] = f"failed: {error}"
                status[key.name] = rec
                log(f"[ingest] {key.name}: FAILED {error}")
                continue
            rec.update(status="ingested", rows=check["rows"], all_reproduced=check["all_reproduced"],
                       not_reproduced=check["not_reproduced"], max_abs_error_pp=check["max_abs_error_pp"],
                       repaired_rows=check["fn_status_counts"].get(rs.REPAIRED, 0))
            status[key.name] = rec
    out = {"runs": dict(sorted(status.items())), "frozen_splits_sha256": splits,
           "registries": {"views": sorted(registries["views"]), "hybrids": sorted(registries["hybrids"])},
           "counts": {s: sum(1 for r in status.values() if r["status"].split(" ")[0] == s)
                      for s in sorted({r["status"].split(" ")[0] for r in status.values()})}}
    write_json(fcfg.work("aligned", "ingest_status.json"), out)
    return out


# ================================================================================================ inputs for analysis
@dataclass
class Context:
    fcfg: FollowupConfig
    shim: FollowupIngestConfig
    reviewed: StudyConfig
    metrics: dict[str, Any]
    ga: pd.DataFrame
    categories: dict[str, list[str]]
    suite: list[str]
    n_replicates: int
    seed: int
    confidence: float
    behavior_replicates: int
    provisional: bool
    log: Log


def make_context(fcfg: FollowupConfig, *, reviewed_root: Path | str | None = None, provisional: bool = False,
                 n_replicates: int | None = None, behavior_replicates: int | None = None,
                 log: Log = print) -> Context:
    root = reviewed_root_of(fcfg, reviewed_root)
    shim = shim_config(fcfg, root)
    check_frozen_splits(fcfg, root)
    reviewed = reviewed_config(fcfg, root)
    stats = fcfg.raw["statistics"]
    if stats.get("multiplicity") != "holm" or list(stats.get("primary_family") or []) != list(PRIMARY_FAMILY):
        raise FollowupAnalysisError(f"configured family {stats.get('primary_family')} / multiplicity differ from the "
                                    f"design's {list(PRIMARY_FAMILY)} / holm")
    n_rep = int(n_replicates or stats["bootstrap_replicates"])
    return Context(fcfg=fcfg, shim=shim, reviewed=reviewed, metrics=rs.load_metrics(reviewed), ga=an.assignments(shim),
                   categories=fcfg.base.categories(), suite=fcfg.base.benchmark_keys(), n_replicates=n_rep,
                   seed=int(stats["seed"]), confidence=float(stats["confidence"]),
                   behavior_replicates=int(behavior_replicates if behavior_replicates is not None else n_rep),
                   provisional=provisional, log=log)


def run_table(ctx: Context, tiers: Sequence[str] | None = None, cells: Sequence[str] | None = None) -> pd.DataFrame:
    """Every planned run of the cells (default the F1 cells): complete / ingested / current / scored / reproduced,
    with the reason when not usable."""

    index = an.registered_local_runs(ctx.shim)
    rows = []
    for key in planned_runs(ctx.fcfg, tiers, cells):
        rd = ctx.fcfg.work("runs", key.slug, key.subset)
        val = rd / f"validation.seed{key.seed}.json"
        complete = val.is_file() and bool(json.loads(val.read_text(encoding="utf-8")).get("complete"))
        rec = {"tier": key.tier, "label": key.label, "cell": key.cell, "interface": key.interface, "slug": key.slug,
               "subset": key.subset, "seed": key.seed, "name": key.name, "complete": complete,
               "ingested": key.name in index, "current": False, "scored": False, "all_reproduced": None,
               "not_reproduced": "", "usable": False, "problem": ""}
        if not complete:
            rec["problem"] = "run not complete"
        elif key.name not in index:
            rec["problem"] = "complete but not ingested (run ingest-runs)"
        else:
            rec["current"] = an.ingested_fingerprint(ctx.shim, key.name) == an.run_fingerprint(rd, key.seed)
            pq_path, check_path = scored_paths(ctx.fcfg, key.name)
            if not rec["current"]:
                rec["problem"] = "ingested from an earlier state of the run (run ingest-runs)"
            elif check_path.is_file() and pq_path.is_file():
                check = json.loads(check_path.read_text(encoding="utf-8"))
                rec["scored"] = (check.get("ingested_sha256") == index[key.name]["sha256"]
                                 and check.get("scorer_code_sha256") == scorer_code_hashes()
                                 and check.get("scored_sha256") == sha256_file(pq_path))
                rec["all_reproduced"] = bool(check.get("all_reproduced"))
                rec["not_reproduced"] = ",".join(check.get("not_reproduced") or [])
                if not rec["scored"]:
                    rec["problem"] = "scored rows are stale (run ingest-runs)"
            else:
                rec["problem"] = "not scored (run ingest-runs)"
        rec["usable"] = bool(rec["complete"] and rec["current"] and rec["scored"])
        rows.append(rec)
    return pd.DataFrame(rows)


def load_scored(ctx: Context, runs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name in runs["name"]:
        pq_path, _ = scored_paths(ctx.fcfg, name)
        frames.append(pd.read_parquet(pq_path))
    rows = pd.concat(frames, ignore_index=True)
    rows["seed"] = pd.to_numeric(rows["seed"]).astype(int)
    if rows.duplicated(["model_label", "seed", "content_key"]).any():
        raise FollowupAnalysisError("duplicate (label, seed, content) rows across the selected runs")
    return rows


def restrict_partition(rows: pd.DataFrame, ga: pd.DataFrame, partition: str) -> pd.DataFrame:
    """``analysis.restrict`` that must keep every row (the runs' row lists are the frozen partition's)."""

    kept = an.restrict(rows, ga, partition)
    if len(kept) != len(rows):
        raise FollowupAnalysisError(f"{len(rows) - len(kept)} rows fall outside the frozen {partition} partition")
    return kept


def macro_benchmarks(ctx: Context, runs: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Accepted adapters (reviewed registry) whose campaign value reproduced on every run used; excluded -> runs."""

    accepted = [b for b in ctx.suite if ctx.metrics[b].kind != "unsupported"]
    failed: dict[str, list[str]] = {}
    for rec in runs.itertuples():
        for b in [b for b in str(rec.not_reproduced).split(",") if b]:
            failed.setdefault(b, []).append(rec.name)
    for b in ctx.suite:
        if b not in accepted:
            failed.setdefault(b, []).append("adapter not accepted in the reviewed metric registry")
    return [b for b in accepted if b not in failed], failed


# ======================================================================================================= statistics
def centered_p_value(replicates: np.ndarray, estimate: float) -> float:
    """Two-sided p-value from the bootstrap distribution centered at the estimate (design §6).

    Under H0 the centered replicates ``d* - d`` approximate the null distribution of the estimate, so
    ``p = (1 + #{|d* - d| >= |d|}) / (R + 1)`` over the R finite replicates (add-one Monte Carlo correction).
    """

    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0 or not math.isfinite(estimate):
        return float("nan")
    hits = int(np.count_nonzero(np.abs(r - estimate) >= abs(estimate)))
    return float(min(1.0, (hits + 1) / (r.size + 1)))


def combine(*terms: tuple[float, Mapping[str, float]]) -> dict[str, float]:
    """Linear combination of label -> coefficient contrasts (zero coefficients dropped)."""

    out: dict[str, float] = {}
    for coef, w in terms:
        for lab, c in w.items():
            out[lab] = out.get(lab, 0.0) + coef * float(c)
    return {k: v for k, v in out.items() if v != 0.0}


def decomposition_contrasts(interface: str, parents: Sequence[str] = PARENTS) -> dict[str, dict[str, float]]:
    """g_F, g_L, I, full gain and both reversion effects per parent (design §6), as label -> coefficient."""

    out: dict[str, dict[str, float]] = {}
    for p in parents:
        b, full, pf, pl = (f"{c}@{interface}" for c in ("B", p, f"{p}F", f"{p}L"))
        out[f"gF_{p}"] = {pf: 1.0, b: -1.0}
        out[f"gL_{p}"] = {pl: 1.0, b: -1.0}
        out[f"I_{p}"] = {full: 1.0, pf: -1.0, pl: -1.0, b: 1.0}
        out[f"full_{p}"] = {full: 1.0, b: -1.0}
        out[f"revF_{p}"] = {full: 1.0, pl: -1.0}
        out[f"revL_{p}"] = {full: 1.0, pf: -1.0}
    return out


def cross_parent_contrasts(effects: Mapping[str, Mapping[str, float]], a: str = "T",
                           b: str = "V") -> dict[str, dict[str, float]]:
    """Exploratory differences between parents, e.g. ``gF_T_minus_gF_V``."""

    return {f"{e}_{a}_minus_{e}_{b}": combine((1.0, effects[f"{e}_{a}"]), (-1.0, effects[f"{e}_{b}"]))
            for e in EFFECTS}


def scope_vectors(res: st.BootstrapResult) -> list[tuple[str, str, int, np.ndarray]]:
    """(level, scope, benchmark_count, weights over res.benchmarks): overall macro, categories, benchmarks."""

    idx = {b: i for i, b in enumerate(res.benchmarks)}
    n = len(res.benchmarks)
    out = [("overall", st.macro_name(n), n, np.full(n, 1.0 / n))]
    for cat, members in res.categories.items():
        v = np.zeros(n)
        v[[idx[m] for m in members]] = 1.0 / len(members)
        out.append(("category", cat, len(members), v))
    for b in res.benchmarks:
        v = np.zeros(n)
        v[idx[b]] = 1.0
        out.append(("benchmark", b, 1, v))
    return out


def _label_weights(res: st.BootstrapResult, weights: Mapping[str, float]) -> np.ndarray | None:
    pos = {lab: i for i, lab in enumerate(res.labels)}
    if any(lab not in pos for lab in weights):
        return None
    w = np.zeros(len(res.labels))
    for lab, c in weights.items():
        w[pos[lab]] = c
    return w


def contrast_arrays(res: st.BootstrapResult, weights: Mapping[str, float]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(point [B], replicates [R, B], per-seed point [S, B]) of one contrast at benchmark level; None if a label of
    the contrast was not evaluated."""

    w = _label_weights(res, weights)
    if w is None:
        return None
    point = w @ res.point_benchmark_scores
    reps = np.tensordot(res.replicate_benchmark_scores, w, axes=([1], [0]))
    seeds = np.tensordot(res.point_seed_scores, w, axes=([0], [0]))
    return point, reps, seeds


def contrast_rows(res: st.BootstrapResult, contrasts: Mapping[str, Mapping[str, float]], *, scale: str,
                  per_seed: bool = True) -> pd.DataFrame:
    """One row per contrast x scope (seed mean, with interval and p-values) plus per-seed point estimates."""

    rows = []
    scopes = scope_vectors(res)
    for name, weights in contrasts.items():
        arrays = contrast_arrays(res, weights)
        base = {"effect": name, "scale": scale, "weights": canonical_json(dict(sorted(weights.items())))}
        if arrays is None:
            missing = sorted(lab for lab in weights if lab not in res.labels)
            rows.append({**base, "level": "overall", "scope": st.macro_name(len(res.benchmarks)),
                         "seed_scope": "mean_over_seeds", "status": f"not computable: {','.join(missing)} missing"})
            continue
        point, reps, seed_pts = arrays
        for level, scope, count, v in scopes:
            d = reps @ v
            est = float(point @ v)
            lo, hi = st.percentile_interval(d, res.confidence)
            finite = d[np.isfinite(d)]
            rows.append({**base, "level": level, "scope": scope, "benchmark_count": count,
                         "seed_scope": "mean_over_seeds", "seed": pd.NA, "n_seeds": len(res.seeds), "estimate": est,
                         "ci_low": lo, "ci_high": hi,
                         "bootstrap_se": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                         "p_value": centered_p_value(d, est), "p_value_percentile_inversion": st.bootstrap_p_value(d),
                         "n_valid_replicates": int(finite.size), "status": "ok"})
            if per_seed and len(res.seeds) > 1:
                for si, s in enumerate(res.seeds):
                    rows.append({**base, "level": level, "scope": scope, "benchmark_count": count,
                                 "seed_scope": "single_seed", "seed": int(s), "n_seeds": 1,
                                 "estimate": float(seed_pts[si] @ v), "status": "ok"})
    out = pd.DataFrame(rows)
    out["seed"] = out["seed"].astype("Int64") if "seed" in out.columns else pd.NA
    return out


def identity_check(res: st.BootstrapResult, interface: str, parents: Sequence[str] = PARENTS) -> dict[str, Any]:
    """full = gF + gL + I, revF = gF + I, revL = gL + I on point estimates and in every replicate, every scope."""

    eff = decomposition_contrasts(interface, parents)
    out: dict[str, Any] = {}
    for p in parents:
        arrays = {e: contrast_arrays(res, eff[f"{e}_{p}"]) for e in EFFECTS}
        if any(a is None for a in arrays.values()):
            out[p] = {"status": "not computable (cells missing)"}
            continue
        point_dev = rep_dev = 0.0
        for level, scope, count, v in scope_vectors(res):
            for lhs, rhs in IDENTITIES:
                lp = float(arrays[lhs][0] @ v)
                rp = sum(float(arrays[r][0] @ v) for r in rhs)
                point_dev = max(point_dev, abs(lp - rp))
                lr = arrays[lhs][1] @ v
                rr = sum(arrays[r][1] @ v for r in rhs)
                rep_dev = max(rep_dev, float(np.nanmax(np.abs(lr - rr))))
        out[p] = {"identities": [f"{l} = {' + '.join(r)}" for l, r in IDENTITIES], "max_abs_dev_point_pp": point_dev,
                  "max_abs_dev_replicates_pp": rep_dev, "tolerance_pp": IDENTITY_TOLERANCE_PP,
                  "replicates": int(res.n_replicates),
                  "holds": bool(point_dev <= IDENTITY_TOLERANCE_PP and rep_dev <= IDENTITY_TOLERANCE_PP)}
    return out


def apply_holm(table: pd.DataFrame, family: Sequence[str]) -> pd.DataFrame:
    """Holm over ``family`` on the overall macro's seed-mean rows, separately within each scale."""

    table = table.copy()
    table["family_member"] = False
    table["p_holm"] = np.nan
    for scale in table["scale"].unique():
        sel = ((table["scale"] == scale) & (table["level"] == "overall") & (table["seed_scope"] == "mean_over_seeds")
               & table["effect"].isin(list(family)) & (table["status"] == "ok"))
        if sel.sum() != len(family):
            continue   # a family member is not computable: no adjusted values (never a smaller family)
        pv = dict(zip(table.loc[sel, "effect"], table.loc[sel, "p_value"].astype(float)))
        adj = st.holm_adjust({f: pv[f] for f in family})
        table.loc[sel, "family_member"] = True
        table.loc[sel, "p_holm"] = table.loc[sel, "effect"].map(adj).to_numpy()
    return table


def scale_bootstrap(rows: pd.DataFrame, metrics: Mapping[str, Any], ga: pd.DataFrame, *, labels: Sequence[str],
                    seeds: Sequence[int], categories: Mapping[str, Sequence[str]] | None, n_replicates: int,
                    seed: int, confidence: float) -> dict[str, st.BootstrapResult]:
    """F0's ``paired_contrasts`` (reviewed bootstrap, official and normalized with the same draws), checked.

    The replicate stream depends only on (seed, replicates, number of units) and the units are the sorted ids of the
    rows' bootstrap units, so identical unit ids on both scales mean identical draws; this is verified."""

    res = rs.paired_contrasts(rows, metrics, ga, labels=list(labels), seeds=list(seeds), comparisons=[],
                              categories=categories, n_replicates=n_replicates, seed=seed, confidence=confidence)
    o, n = res["official"], res["normalized"]
    same = (np.array_equal(o.component_ids, n.component_ids) and o.n_replicates == n.n_replicates
            and o.seed == n.seed and o.labels == n.labels and o.seeds == n.seeds and o.benchmarks == n.benchmarks)
    if not same:
        raise FollowupAnalysisError("official and normalized bootstraps did not use the same units and draws")
    return {OFFICIAL: o, NORMALIZED: n}


def draws_record(results: Mapping[str, st.BootstrapResult]) -> dict[str, Any]:
    o = results[OFFICIAL]
    return {"identical_draws_official_and_normalized": True, "n_units": int(o.n_components),
            "n_replicates": int(o.n_replicates), "seed": int(o.seed), "rng_block": st.RNG_BLOCK,
            "unit_ids_sha256": sha256_bytes("\n".join(map(str, o.component_ids)).encode("utf-8")),
            "basis": "statistics.iter_weight_blocks(n_units, replicates, seed) over the same sorted unit ids"}


def units_report(rows: pd.DataFrame, ga: pd.DataFrame, metrics: Mapping[str, Any], label: str,
                 seed: int) -> dict[str, Any]:
    """Resampling units and rows per benchmark, and WeMath's complete problem groups (one label/seed slice)."""

    comp = ga.set_index("content_key")["component_id"]
    sub = rows[(rows["model_label"] == label) & (rows["seed"] == seed)]
    per = {}
    for bench, g in sub.groupby("benchmark_id", sort=True):
        per[str(bench)] = {"rows": int(len(g)), "units": int(g["content_key"].map(comp).nunique())}
        if bench == "wemath":
            totals = metrics[bench].row_contributions(g, g["official_row_score"].astype(float)).sum(axis=0)
            per[str(bench)]["complete_problem_groups"] = int(totals[0])
    return {"content_rows": int(len(sub)), "units": int(sub["content_key"].map(comp).nunique()), "per_benchmark": per}


# ========================================================================================================== tables
def performance_table(results: Mapping[str, st.BootstrapResult]) -> pd.DataFrame:
    """Per label x scope: seed-mean value with interval and per-seed values, official and normalized side by side."""

    o, n = results[OFFICIAL], results[NORMALIZED]
    rows = []
    for li, lab in enumerate(o.labels):
        for level, scope, count, v in scope_vectors(o):
            rec = {"model_label": lab, "cell": lab.split("@")[0], "interface": lab.split("@")[1], "level": level,
                   "scope": scope, "benchmark_count": count, "seed_scope": "mean_over_seeds", "seed": pd.NA,
                   "n_seeds": len(o.seeds)}
            for tag, res in (("official", o), ("normalized", n)):
                d = res.replicate_benchmark_scores[:, li, :] @ v
                lo, hi = st.percentile_interval(d, res.confidence)
                rec.update({f"value_{tag}": float(res.point_benchmark_scores[li] @ v), f"ci_low_{tag}": lo,
                            f"ci_high_{tag}": hi})
            rows.append(rec)
            if len(o.seeds) > 1:
                for si, s in enumerate(o.seeds):
                    rows.append({**{k: rec[k] for k in ("model_label", "cell", "interface", "level", "scope",
                                                        "benchmark_count")},
                                 "seed_scope": "single_seed", "seed": int(s), "n_seeds": 1,
                                 "value_official": float(o.point_seed_scores[li, si] @ v),
                                 "value_normalized": float(n.point_seed_scores[li, si] @ v)})
    out = pd.DataFrame(rows)
    out["seed"] = out["seed"].astype("Int64")
    out["normalized_minus_official"] = out["value_normalized"] - out["value_official"]
    return out


def effects_table(results: Mapping[str, st.BootstrapResult], contrasts: Mapping[str, Mapping[str, float]],
                  roles: Mapping[str, str], family: Sequence[str] = ()) -> pd.DataFrame:
    frames = [contrast_rows(results[s], contrasts, scale=s) for s in (OFFICIAL, NORMALIZED)]
    table = pd.concat(frames, ignore_index=True)
    table.insert(1, "role", table["effect"].map(roles))
    table.insert(2, "analysis", np.where(table["scale"] == OFFICIAL, "primary_scale", "sensitivity_analysis"))
    return apply_holm(table, family) if family else table.assign(family_member=False, p_holm=np.nan)


def effect_roles(contrasts: Mapping[str, Any], family: Sequence[str]) -> dict[str, str]:
    out = {}
    for name in contrasts:
        if name in family:
            out[name] = "primary_family"
        elif "_minus_" in name:
            out[name] = "cross_parent_exploratory"
        elif name.startswith(("full_", "revF_", "revL_")):
            out[name] = "decomposition_reported"
        else:
            out[name] = "exploratory"
    return out


def format_rates(rows: pd.DataFrame, max_tokens: int) -> pd.DataFrame:
    """Per run (label, seed) overall and per benchmark: empty responses, truncation at the token limit, response
    length, and the format-normalized-v2 repair share (diagnostics, never scores)."""

    df = rows.assign(truncated=rows["finish_reason"].astype(str).eq("length"),
                     at_limit=pd.to_numeric(rows["completion_tokens"], errors="coerce") >= max_tokens,
                     repaired=rows["fn_status"].eq(rs.REPAIRED),
                     tokens=pd.to_numeric(rows["completion_tokens"], errors="coerce"))
    out = []
    for (label, seed), g in df.groupby(["model_label", "seed"], sort=True):
        parts = [("overall", "all benchmarks", g)] + [("benchmark", b, gb) for b, gb in g.groupby("benchmark_id")]
        for level, scope, part in parts:
            n = len(part)
            out.append({"model_label": label, "seed": int(seed), "level": level, "scope": scope, "rows": n,
                        "empty_rows": int(part["response_empty"].sum()),
                        "empty_pct": 100.0 * float(part["response_empty"].mean()),
                        "truncated_rows": int(part["truncated"].sum()),
                        "truncated_pct": 100.0 * float(part["truncated"].mean()),
                        "at_token_limit_rows": int(part["at_limit"].sum()),
                        "finish_reason_null_rows": int(part["finish_reason"].isna().sum()),
                        "mean_completion_tokens": float(part["tokens"].mean()),
                        "fn_repaired_rows": int(part["repaired"].sum()),
                        "fn_repaired_pct": 100.0 * float(part["repaired"].mean())})
    return pd.DataFrame(out)


def _write_csv(df: pd.DataFrame, path: Path, status: str) -> Path:
    out = df.copy()
    out.insert(0, "analysis_status", status)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def _overall(table: pd.DataFrame, **match: Any) -> pd.DataFrame:
    sel = table["level"].eq("overall")
    for k, v in match.items():
        sel &= table[k].eq(v)
    return table[sel]


# ===================================================================================================== the analyses
def primary_analysis(ctx: Context, runs: pd.DataFrame) -> dict[str, Any]:
    """Design §6 on the confirmation partition, P1, seeds 42-44 (or, provisionally, the seeds every cell has)."""

    seeds_cfg = [int(s) for s in ctx.fcfg.raw["evaluation"]["analysis"]["seeds"]]
    ar = runs[(runs["tier"] == "analysis") & runs["cell"].isin(F1_CELLS)]
    usable = ar[ar["usable"]]
    missing = ar[~ar["usable"]]
    seeds = [s for s in seeds_cfg if set(usable.loc[usable["seed"] == s, "cell"]) >= set(F1_CELLS)]
    if len(missing) and not ctx.provisional:
        raise FollowupAnalysisError("main-matrix runs not usable: " + "; ".join(
            f"{r.name}: {r.problem}" for r in missing.itertuples()) + " (pass --provisional for a provisional run)")
    if not seeds:
        raise FollowupAnalysisError("no seed has all seven cells usable")
    use = usable[usable["seed"].isin(seeds)]
    labels = [f"{c}@{PRIMARY_INTERFACE}" for c in F1_CELLS]
    benches, excluded = macro_benchmarks(ctx, use)
    rows = restrict_partition(load_scored(ctx, use), ctx.ga, "confirmation")
    rows = rows[rows["benchmark_id"].isin(benches)].reset_index(drop=True)
    ctx.log(f"[analyze] primary: {len(rows)} rows, labels {labels}, seeds {seeds}, {len(benches)} benchmarks")
    results = scale_bootstrap(rows, ctx.metrics, ctx.ga, labels=labels, seeds=seeds, categories=ctx.categories,
                              n_replicates=ctx.n_replicates, seed=ctx.seed, confidence=ctx.confidence)
    eff = decomposition_contrasts(PRIMARY_INTERFACE)
    contrasts = {**eff, **cross_parent_contrasts(eff)}
    effects = effects_table(results, contrasts, effect_roles(contrasts, PRIMARY_FAMILY), PRIMARY_FAMILY)
    identity = {s: identity_check(r, PRIMARY_INTERFACE) for s, r in results.items()}
    if not all(v.get("holds") for d in identity.values() for v in d.values()):
        raise FollowupAnalysisError(f"decomposition identity failed: {identity}")
    return {"seeds": seeds, "labels": labels, "benchmarks": benches, "excluded_benchmarks": excluded,
            "macro_name": st.macro_name(len(benches)), "missing_runs": missing["name"].tolist(),
            "results": results, "effects": effects, "performance": performance_table(results), "identity": identity,
            "draws": draws_record(results), "rows": rows,
            "units": units_report(rows, ctx.ga, ctx.metrics, labels[0], seeds[0])}


def template_analysis(ctx: Context, runs: pd.DataFrame) -> dict[str, Any]:
    """Development screen, seed 42: the cells under P1 (screen) and P2 (template), one bootstrap for both."""

    sr = runs[runs["tier"].isin(["screen", "template"]) & runs["cell"].isin(F1_CELLS)]
    use = sr[sr["usable"]]
    missing = sr[~sr["usable"]]
    if len(missing) and not ctx.provisional:
        raise FollowupAnalysisError("screen/template runs not usable: " + "; ".join(
            f"{r.name}: {r.problem}" for r in missing.itertuples()))
    seeds = sorted(set(use["seed"]))
    order = {f"{c}@{i}": k for k, (i, c) in enumerate((i, c) for i in (PRIMARY_INTERFACE, TEMPLATE_INTERFACE)
                                                     for c in F1_CELLS)}
    labels = sorted(use["label"].unique(), key=order.__getitem__)
    benches, excluded = macro_benchmarks(ctx, use)
    rows = restrict_partition(load_scored(ctx, use), ctx.ga, "development_screen")
    rows = rows[rows["benchmark_id"].isin(benches)].reset_index(drop=True)
    ctx.log(f"[analyze] screen/template: {len(rows)} rows, labels {labels}, seeds {seeds}")
    results = scale_bootstrap(rows, ctx.metrics, ctx.ga, labels=labels, seeds=seeds, categories=ctx.categories,
                              n_replicates=ctx.n_replicates, seed=ctx.seed, confidence=ctx.confidence)
    contrasts: dict[str, dict[str, float]] = {}
    roles: dict[str, str] = {}
    for c in F1_CELLS:
        contrasts[f"P2_minus_P1:{c}"] = {f"{c}@{TEMPLATE_INTERFACE}": 1.0, f"{c}@{PRIMARY_INTERFACE}": -1.0}
        roles[f"P2_minus_P1:{c}"] = "template_dependence_per_cell"
    d1, d2 = decomposition_contrasts(PRIMARY_INTERFACE), decomposition_contrasts(TEMPLATE_INTERFACE)
    for name in d1:
        contrasts[f"{name}@P1"], roles[f"{name}@P1"] = d1[name], "decomposition_P1_screen"
        contrasts[f"{name}@P2"], roles[f"{name}@P2"] = d2[name], "decomposition_P2_exploratory"
        contrasts[f"{name}:P2_minus_P1"] = combine((1.0, d2[name]), (-1.0, d1[name]))
        roles[f"{name}:P2_minus_P1"] = "decomposition_template_dependence"
    effects = effects_table(results, contrasts, roles)
    identity = {s: {i: identity_check(r, i) for i in (PRIMARY_INTERFACE, TEMPLATE_INTERFACE)}
                for s, r in results.items()}
    bad = [(s, i, p) for s, d in identity.items() for i, dd in d.items() for p, v in dd.items()
           if "holds" in v and not v["holds"]]
    if bad:
        raise FollowupAnalysisError(f"decomposition identity failed on the screen/template matrix: {bad}")
    return {"seeds": seeds, "labels": labels, "benchmarks": benches, "excluded_benchmarks": excluded,
            "macro_name": st.macro_name(len(benches)), "missing_runs": missing["name"].tolist(), "results": results,
            "effects": effects, "performance": performance_table(results), "identity": identity,
            "draws": draws_record(results), "rows": rows}


def technical_checks(ctx: Context, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Format rates of every usable run, and the design §4 technical-stop rule per (cell, interface) on its smoke and
    screen runs (empty > 50% or truncated at the token limit > 80%; build verification)."""

    max_tokens = int(ctx.fcfg.base.raw["protocol"]["generation"]["max_tokens"])
    use = runs[runs["usable"]]
    frames = []
    for subset, part in use.groupby("subset", sort=True):
        rates = format_rates(load_scored(ctx, part), max_tokens)
        rates.insert(0, "subset", subset)
        frames.append(rates)
    rates = pd.concat(frames, ignore_index=True)
    rates.insert(1, "cell", rates["model_label"].str.split("@").str[0])
    rates.insert(2, "interface", rates["model_label"].str.split("@").str[1])
    verification = {}
    vpath = ctx.fcfg.work("audit", "build_verification.json")
    if vpath.is_file():
        verification = json.loads(vpath.read_text(encoding="utf-8")).get("hybrids") or {}
    checks = []
    for cell in F1_CELLS:
        for iface in (PRIMARY_INTERFACE, TEMPLATE_INTERFACE):
            label = f"{cell}@{iface}"
            subsets = ["smoke", "development_screen"] if iface == PRIMARY_INTERFACE else ["smoke"]
            rec: dict[str, Any] = {"cell": cell, "interface": iface, "model_label": label}
            stop_reasons = []
            for subset in subsets:
                r = _overall(rates, model_label=label, subset=subset)
                tag = "smoke" if subset == "smoke" else "screen"
                if r.empty:
                    rec[f"{tag}_status"] = "not available"
                    continue
                r = r.iloc[0]
                rec.update({f"{tag}_status": "available", f"{tag}_rows": int(r["rows"]),
                            f"{tag}_empty_pct": r["empty_pct"], f"{tag}_truncated_pct": r["truncated_pct"]})
                if r["empty_pct"] > EMPTY_STOP_PCT:
                    stop_reasons.append(f"{tag}: {r['empty_pct']:.1f}% empty")
                if r["truncated_pct"] > TRUNCATION_STOP_PCT:
                    stop_reasons.append(f"{tag}: {r['truncated_pct']:.1f}% truncated")
            if cell in ctx.fcfg.hybrids:
                ok = bool((verification.get(cell) or {}).get("donor_equal"))
                rec["build_verified"] = ok
                if not ok:
                    stop_reasons.append("build verification missing or failed")
            evaluated = all(rec.get(f"{'smoke' if s == 'smoke' else 'screen'}_status") == "available" for s in subsets)
            rec["technical_stop"] = bool(stop_reasons)
            rec["technical_stop_reasons"] = "; ".join(stop_reasons)
            rec["technical_stop_evaluated"] = evaluated
            rec["not_checked_here"] = "view loading / missing-key warnings and non-finite logits (runner logs)"
            checks.append(rec)
    return rates, pd.DataFrame(checks)


RETENTION_RENAMES = (("merge_label", "retaining_label"), ("trace_label", "parent_label"),
                     ("vero_label", "base_label"), ("acc_merge_pct", "acc_retaining_pct"),
                     ("acc_trace_pct", "acc_parent_pct"), ("acc_vero_pct", "acc_base_pct"),
                     ("parent_seeds", "definition_seed"), ("merge_seed", "measurement_seed"))


def _retention_columns(t: pd.DataFrame) -> pd.DataFrame:
    """capture_damage's columns in success-set terms; the filler label's columns (base slot) are dropped."""

    t = t.drop(columns=[c for c in t.columns if c == "base_label" or c.startswith("acc_base_pct")])
    names = {}
    for c in t.columns:
        for old, new in RETENTION_RENAMES:
            if c == old or c.startswith(old + "_ci_"):
                names[c] = new + c[len(old):]
    return t.rename(columns=names)


def behavior_analysis(ctx: Context, runs: pd.DataFrame, primary_seeds: Sequence[int]) -> pd.DataFrame:
    """Exploratory behaviour on the reviewed study's validated binary benchmarks (DESIGN.md A3), both scales.

    * corrections/regressions of every cell against B (``behavior.corrections_regressions``) on the primary seeds;
    * per parent P, its success set (P = 1, B = 0 on the definition seed) and the share of it each hybrid re-solves on
      the measurement seed (``behavior.capture_damage``'s independent-repeat condition with parents (P, B) and the
      hybrid in the merge role; the same table gives the hybrid's damage on rows both solve). Prespecified: defined
      on 42, measured on 43; robustness: the reverse. P itself and B on the measurement seed are the reference rows
      (repeat of P's own success; floor). A provisional run without seed 43 adds 44 in place of 43, labelled so.
    """

    binary = an.binary_benchmarks(ctx.reviewed)
    ar = runs[(runs["tier"] == "analysis") & runs["cell"].isin(F1_CELLS) & runs["usable"]]
    rows = restrict_partition(load_scored(ctx, ar), ctx.ga, "confirmation")
    rows = rows[rows["benchmark_id"].isin(binary)].reset_index(drop=True)
    have = {(r.label, int(r.seed)) for r in ar.itertuples()}
    spec = bh.BootstrapSpec.from_config(ctx.fcfg.raw["statistics"], n_replicates=ctx.behavior_replicates) \
        if ctx.behavior_replicates >= 2 else None
    comps = an.components_of(rows, ctx.ga)
    base = f"B@{PRIMARY_INTERFACE}"
    labels_p1 = [f"{c}@{PRIMARY_INTERFACE}" for c in F1_CELLS]
    frames = []
    pairs = [(42, 43, "prespecified"), (43, 42, "reverse")]
    if ctx.provisional:
        pairs += [(42, 44, "provisional_substitute"), (44, 42, "provisional_substitute_reverse")]
    for scale, col in SCALES.items():
        data = rows.assign(binary_correct=rows[col].astype(float))
        # reference copies so P and B can sit in the "merge" role next to themselves (as analysis.with_base_reference)
        refs = []
        for cell in ("B", *PARENTS):
            r = data[data["model_label"] == f"{cell}@{PRIMARY_INTERFACE}"].copy()
            r["model_label"] = f"{cell}@{PRIMARY_INTERFACE} (repeat)"
            refs.append(r)
        matrix = bh.build_outcome_matrix(pd.concat([data, *refs], ignore_index=True), expected_benchmarks=binary)
        models = [f"{c}@{PRIMARY_INTERFACE}" for c in F1_CELLS if c != "B"]
        cr = bh.corrections_regressions(matrix, base=base, models=models, seeds=list(primary_seeds),
                                        categories=ctx.categories, components=comps, uncertainty=spec)
        cr.insert(0, "section", "corrections_regressions_vs_B")
        cr.insert(1, "scale", scale)
        frames.append(cr)
        for p in PARENTS:
            parent = f"{p}@{PRIMARY_INTERFACE}"
            keepers = [f"{p}F@{PRIMARY_INTERFACE}", f"{p}L@{PRIMARY_INTERFACE}", f"{parent} (repeat)",
                       f"{base} (repeat)"]
            for d, e, kind in pairs:
                for keeper in keepers:
                    k_lab = keeper.replace(" (repeat)", "")
                    if not all(x in have for x in [(parent, d), (base, d), (k_lab, e)]):
                        continue
                    # capture_damage needs a fourth distinct label with outcomes on the definition seed; it only
                    # feeds acc_base_pct, which is dropped below
                    filler = next((lab for lab in labels_p1 if lab not in (k_lab, parent, base) and (lab, d) in have),
                                  None)
                    if filler is None:
                        continue
                    t = bh.capture_damage(matrix, merge=keeper, trace=parent, vero=base, base=filler,
                                          merge_seeds=[e], parent_seeds=[d], variants=("independent_repeat",),
                                          categories=ctx.categories, components=comps, uncertainty=spec)
                    t = _retention_columns(t)
                    t["group"] = t["group"].map({"trace_only": "parent_success_set (P=1, B=0)",
                                                 "vero_only": "base_only (P=0, B=1)",
                                                 "shared_success": "shared_success (P=1, B=1)",
                                                 "joint_failure": "joint_failure (P=0, B=0)"}).fillna(t["group"])
                    t.insert(0, "section", "success_set_retention")
                    t.insert(1, "scale", scale)
                    t.insert(2, "seed_pair", kind)
                    t.insert(3, "reference_row", "(repeat)" in keeper)
                    frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    out.insert(2, "binary_benchmarks", len(binary))
    return out


def evaluation_costs(ctx: Context, runs: pd.DataFrame) -> pd.DataFrame:
    """GPU-hours (attempt wall time x cards), wall hours, attempts, generated tokens, judge settings and storage per
    run, from the runner's ``timing.seed<S>.json`` and ``summary.seed<S>.json`` (failed attempts count)."""

    out = []
    for key in planned_runs(ctx.fcfg):
        rd = ctx.fcfg.work("runs", key.slug, key.subset)
        timing = rd / f"timing.seed{key.seed}.json"
        rec: dict[str, Any] = {"tier": key.tier, "label": key.label, "cell": key.cell, "interface": key.interface,
                               "slug": key.slug, "subset": key.subset, "seed": key.seed}
        attempts = ok = 0
        gpu_h = wall_h = 0.0
        gpus, judge_tp = set(), set()
        if timing.is_file():
            for a in json.loads(timing.read_text(encoding="utf-8")).get("attempts", []):
                ep = a.get("epoch") or {}
                if "start" not in ep or len(ep) < 2:
                    continue
                m = re.match(r"(\d+)x\s*(.*)", str(a.get("gpus") or ""))
                n = int(m.group(1)) if m else 1
                secs = max(ep.values()) - ep["start"]
                attempts += 1
                ok += int(a.get("rc") == 0)
                wall_h += secs / 3600
                gpu_h += n * secs / 3600
                if m:
                    gpus.add(m.group(2).strip())
                if (a.get("serving") or {}).get("judge_tp"):
                    judge_tp.add(str(a["serving"]["judge_tp"]))
        summary = rd / f"summary.seed{key.seed}.json"
        tokens_est = None
        if summary.is_file():
            s = json.loads(summary.read_text(encoding="utf-8"))
            tokens_est = float(sum(float(b.get("mean_output_tokens") or 0) * int(b.get("rows") or 0)
                                   for b in s.get("benchmarks") or []))
        exact = None
        r = runs[runs["name"] == key.name]
        if len(r) and bool(r["usable"].iloc[0]):
            pq_path, _ = scored_paths(ctx.fcfg, key.name)
            exact = int(pd.to_numeric(pd.read_parquet(pq_path, columns=["completion_tokens"])["completion_tokens"],
                                      errors="coerce").fillna(0).sum())
        rec.update({"complete": bool(len(r) and r["complete"].iloc[0]), "attempts": attempts,
                    "successful_attempts": ok, "wall_hours": round(wall_h, 4), "gpu_hours": round(gpu_h, 4),
                    "gpu_types": "; ".join(sorted(gpus)), "judge_tp": ",".join(sorted(judge_tp)),
                    "generated_tokens": exact, "generated_tokens_from_summary": tokens_est,
                    "quarantined_dirs": len(list(rd.glob("quarantine_*"))) if rd.is_dir() else 0})
        out.append(rec)
    table = pd.DataFrame(out)
    sizes = {}
    for slug, subset in sorted(set(zip(table["slug"], table["subset"]))):
        rd = ctx.fcfg.work("runs", slug, subset)
        sizes[(slug, subset)] = round(sum(p.stat().st_size for p in rd.rglob("*") if p.is_file()) / 2**30, 3) \
            if rd.is_dir() else 0.0
    first = ~table.duplicated(["slug", "subset"])
    table["run_dir_gib_shared_by_seeds"] = [sizes[(s, b)] if f else np.nan
                                            for s, b, f in zip(table["slug"], table["subset"], first)]
    return table


# ======================================================================================================== the driver
def _round_ci(r: pd.Series) -> list[float | None]:
    return [None if pd.isna(r.get(k)) else round(float(r[k]), 4) for k in ("estimate", "ci_low", "ci_high")]


def _family_summary(effects: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scale in (OFFICIAL, NORMALIZED):
        sel = _overall(effects, scale=scale, seed_scope="mean_over_seeds")
        out[scale] = {r["effect"]: {"estimate_ci": _round_ci(r), "p_value": r.get("p_value"),
                                    "p_holm": None if pd.isna(r.get("p_holm")) else float(r["p_holm"]),
                                    "p_value_percentile_inversion": r.get("p_value_percentile_inversion"),
                                    "role": r.get("role"), "status": r.get("status")}
                      for _, r in sel.iterrows()}
    return out


def analyze(fcfg: FollowupConfig, *, provisional: bool = False, reviewed_root: Path | str | None = None,
            n_replicates: int | None = None, behavior_replicates: int | None = None, out_dir: Path | None = None,
            log: Log = print) -> dict[str, Any]:
    """Run every analysis of design §6 and write ``tables/`` (``tables/provisional/`` when provisional)."""

    ctx = make_context(fcfg, reviewed_root=reviewed_root, provisional=provisional, n_replicates=n_replicates,
                       behavior_replicates=behavior_replicates, log=log)
    status = STATUS_PROVISIONAL if provisional else STATUS_FINAL
    out = Path(out_dir) if out_dir else fcfg.work("tables", *(["provisional"] if provisional else []))
    runs = run_table(ctx)
    with followup_roles(fcfg):
        stale = an.ingested_run_problems(ctx.shim, [(r.label, r.slug, r.subset, int(r.seed))
                                                    for r in runs[runs["complete"]].itertuples()])
    t0 = time.monotonic()
    primary = primary_analysis(ctx, runs)
    log(f"[analyze] primary done ({time.monotonic() - t0:.0f} s)")
    template = template_analysis(ctx, runs)
    log(f"[analyze] screen/template done ({time.monotonic() - t0:.0f} s)")
    rates, checks = technical_checks(ctx, runs)
    smoke_fail = checks[(checks["interface"] == PRIMARY_INTERFACE) & checks["technical_stop"]]
    if len(smoke_fail) and not provisional:
        raise FollowupAnalysisError(f"technical stops: {smoke_fail[['cell', 'technical_stop_reasons']].to_dict('records')}")
    behavior = behavior_analysis(ctx, runs, primary["seeds"])
    log(f"[analyze] behaviour done ({time.monotonic() - t0:.0f} s)")
    costs = evaluation_costs(ctx, runs)

    ctx_cols = {"partition": "confirmation", "interface": PRIMARY_INTERFACE,
                "seeds": "+".join(map(str, primary["seeds"]))}
    effects = primary["effects"].copy()
    for k, v in reversed(list(ctx_cols.items())):
        effects.insert(0, k, v)
    _write_csv(effects, out / "component_effects.csv", status)
    perf = primary["performance"].copy()
    for k, v in reversed(list(ctx_cols.items())):
        if k not in perf.columns:
            perf.insert(0, k, v)
    _write_csv(perf, out / "performance_7cell.csv", status)
    tpl = pd.concat([template["performance"].assign(section="scores"),
                     template["effects"].assign(section="contrasts")], ignore_index=True)
    tpl.insert(0, "section", tpl.pop("section"))
    tpl.insert(1, "partition", "development_screen")
    tpl.insert(2, "seeds", "+".join(map(str, template["seeds"])))
    _write_csv(tpl, out / "template_sensitivity.csv", status)
    screen_scores = template["performance"][template["performance"]["interface"] == PRIMARY_INTERFACE]
    screen = pd.concat([screen_scores.assign(section="scores_P1_development_screen"),
                        checks.assign(section="technical_stop")], ignore_index=True)
    screen.insert(0, "section", screen.pop("section"))
    _write_csv(screen, out / "screen.csv", status)
    fmt = rates.assign(section="format_rates")
    fmt.insert(0, "section", fmt.pop("section"))
    _write_csv(pd.concat([behavior, fmt], ignore_index=True), out / "behavior_and_format.csv", status)
    _write_csv(costs, out / "evaluation_costs.csv", status)

    perf_overall = _overall(primary["performance"], seed_scope="mean_over_seeds")
    tpl_perf = _overall(template["performance"], seed_scope="mean_over_seeds")
    summary = {
        "status": status,
        "design": "design/followup_v1.md §6",
        "config_sha256": fcfg.sha256,
        "scales": {"primary": OFFICIAL, "sensitivity": NORMALIZED},
        "p_value_method": "two-sided, centered bootstrap distribution: (1 + #{|d*-d| >= |d|}) / (R + 1); the "
                          "reviewed study's percentile-inversion p is reported beside it",
        "ci_method": st.CI_METHOD,
        "runs": {"planned": int(len(runs)), "usable": int(runs["usable"].sum()),
                 "not_usable": {r.name: r.problem for r in runs[~runs["usable"]].itertuples()},
                 "stale_ingests": stale},
        "primary": {
            "seeds": primary["seeds"], "labels": primary["labels"], "macro_name": primary["macro_name"],
            "benchmarks": primary["benchmarks"], "excluded_benchmarks": primary["excluded_benchmarks"],
            "missing_runs": primary["missing_runs"], "draws": primary["draws"], "units": primary["units"],
            "identity": primary["identity"], "family": list(PRIMARY_FAMILY),
            "effects_overall": _family_summary(primary["effects"]),
            "seven_cell_macro": {r.model_label: {"official": [round(r.value_official, 4), round(r.ci_low_official, 4),
                                                              round(r.ci_high_official, 4)],
                                                 "normalized": [round(r.value_normalized, 4),
                                                                round(r.ci_low_normalized, 4),
                                                                round(r.ci_high_normalized, 4)]}
                                 for r in perf_overall.itertuples()},
            "per_seed_macro": {f"{r.model_label}/s{r.seed}": [round(r.value_official, 4), round(r.value_normalized, 4)]
                               for r in _overall(primary["performance"], seed_scope="single_seed").itertuples()},
        },
        "template": {
            "seeds": template["seeds"], "labels": template["labels"], "macro_name": template["macro_name"],
            "excluded_benchmarks": template["excluded_benchmarks"], "missing_runs": template["missing_runs"],
            "draws": template["draws"], "identity": template["identity"],
            "macro_official_normalized": {r.model_label: [round(r.value_official, 4), round(r.value_normalized, 4)]
                                          for r in tpl_perf.itertuples()},
            "effects_overall": _family_summary(template["effects"]),
        },
        "technical_stop": checks[["model_label", "technical_stop", "technical_stop_reasons",
                                  "technical_stop_evaluated"]].to_dict("records"),
        "costs": {"gpu_hours": round(float(costs["gpu_hours"].sum()), 2),
                  "wall_hours": round(float(costs["wall_hours"].sum()), 2), "attempts": int(costs["attempts"].sum()),
                  "failed_attempts": int((costs["attempts"] - costs["successful_attempts"]).sum()),
                  "generated_tokens_ingested_runs": int(pd.to_numeric(costs["generated_tokens"]).fillna(0).sum()),
                  "run_dirs_gib": round(float(costs["run_dir_gib_shared_by_seeds"].sum()), 2),
                  "hybrid_weights_gib": round(sum(p.stat().st_size for h in F1_CELLS if h in fcfg.hybrids
                                                  for p in fcfg.work("hybrids", h).glob("*.safetensors")) / 2**30, 2)},
        "settings": {"replicates": ctx.n_replicates, "seed": ctx.seed, "confidence": ctx.confidence,
                     "behavior_replicates": ctx.behavior_replicates, "reproduction_tolerance_pp":
                     REPRODUCTION_TOLERANCE_PP, "identity_tolerance_pp": IDENTITY_TOLERANCE_PP},
        "code_sha256": {**scorer_code_hashes(), "trace_followup/analyze.py": sha256_file(Path(__file__)),
                        "trace_merge/statistics.py": sha256_file(Path(st.__file__)),
                        "trace_merge/behavior.py": sha256_file(Path(bh.__file__)),
                        "trace_merge/ingest.py": sha256_file(Path(ig.__file__))},
        "outputs": sorted(p.name for p in out.glob("*.csv")),
        "notes": ["Exploratory follow-up (design §0); the confirmation partition was examined in the reviewed study "
                  "and is not unseen (design §4).",
                  "Intervals describe evaluation uncertainty for fixed checkpoints, not training variation; "
                  "non-significance is not equivalence; no noninferiority margin is set (design §6).",
                  "The interaction I is scale-dependent, not an intrinsic measure of synergy (design §6)."],
    }
    if provisional:
        summary["notes"].insert(0, "PROVISIONAL: computed on the runs complete at the time; missing runs listed in "
                                   "runs.not_usable; the primary matrix uses only the seeds every cell has.")
    an._write_json(out / "analysis_summary.json", summary)
    log(f"[analyze] {status}: wrote {len(summary['outputs'])} tables and analysis_summary.json to {out}")
    return summary
