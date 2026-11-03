"""Stage orchestration over the archive and the new runs (handoff stages B, C and E; DESIGN.md sections 3-8).

* ``stage_b``: archived parents (B, T, V@P0; seeds 42-44; full split and each partition): official scores with
  cluster-bootstrap intervals, corrections/regressions, eight-cell tables, overlap, persistence.
* ``screen``: the frozen development screen at seed 42: P1 merges and controls against the archived parents and V@P1,
  the P2 diagnostic, capture/damage. Descriptive; nothing is selected from it.
* ``confirm``: the confirmation partition at seeds 42-44, run only after ``lock-selection`` and only while the lock
  still holds (``verify_lock``: every frozen hash recomputed, labels and seeds taken from the lock, served revisions,
  fresh ingests and smoke/development completion checked): the prespecified primary family with Holm-adjusted p-values, categories,
  benchmarks, capture/damage, and an environment sensitivity table that substitutes the existing local seed-42 reruns
  of B and T.

Every function reads the aligned tables written by ``ingest`` and the accepted metric adapters (``metrics``), and
resamples the frozen components (``grouping``). Tables go to ``<work>/tables/<stage>/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import behavior as bh
from . import evaluation as ev
from . import ingest as ig
from . import metrics as mt
from . import statistics as st
from .config import PROJECT_ROOT, StudyConfig, canonical_json, sha256_bytes, sha256_file, write_json
from .evaluation import LOCK_NAME, TIER_SUBSET, run_dir, tier_runs

ARCHIVE_LABELS = ("B", "T", "V@P0")
ROW_COLUMNS = ["model_label", "model_role", "protocol_id", "source", "run_id", "seed", "benchmark_id", "content_key",
               "source_ordinal", "source_index", "source_row_sha256", "source_record_sha256", "archived_row_score",
               "official_excluded", "extraction_status", "extraction_value_json", "judge_output"]
LOCAL_RERUN_NAMES = {"B": "B_seed42_existing_local_rerun", "T": "T_seed42_existing_local_rerun"}
REPRODUCTION_TOLERANCE = 1e-6   # DESIGN.md section 5: percent units, full split, every archived slice
Log = Callable[[str], None]


class AnalysisError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ inputs


def metric_registry(cfg: StudyConfig) -> dict[str, Any]:
    return json.loads(cfg.work("manifests", "metric_registry.json").read_text(encoding="utf-8"))


def binary_benchmarks(cfg: StudyConfig) -> list[str]:
    """Benchmarks whose official row score is 0/1 and whose official value is the pooled row mean (amendment A3)."""

    reg = metric_registry(cfg)["benchmarks"]
    keys = cfg.benchmark_keys()
    return [k for k in keys if (reg[k].get("row_diagnostics") or {}).get("supported")]


def macro_benchmarks(cfg: StudyConfig, metrics: dict[str, Any]) -> list[str]:
    """Suite benchmarks with an accepted adapter (all 24 when every reproduction passed)."""

    return [k for k in cfg.benchmark_keys() if metrics[k].kind != "unsupported"]


def assignments(cfg: StudyConfig) -> pd.DataFrame:
    """The frozen group assignments, refused unless they hash to the split manifest's value."""

    path = cfg.work("splits", "group_assignments.parquet")
    manifest = cfg.work("splits", "split_manifest.json")
    if not manifest.is_file():
        raise AnalysisError("splits/split_manifest.json is missing (freeze-splits)")
    want = json.loads(manifest.read_text(encoding="utf-8")).get("group_assignments_sha256")
    if not path.is_file() or sha256_file(path) != want:
        raise AnalysisError("splits/group_assignments.parquet does not hash to split_manifest.group_assignments_sha256")
    ga = pd.read_parquet(path)
    # amendment A7: resample bootstrap units (components joined by verified near-duplicate image twins); the
    # partition's own component ids stay available as partition_component_id
    units_manifest = cfg.work("splits", "bootstrap_units.json")
    if not units_manifest.is_file():
        raise AnalysisError("splits/bootstrap_units.json is missing (DESIGN amendment A7)")
    um = json.loads(units_manifest.read_text(encoding="utf-8"))
    units_path = cfg.work("splits", "bootstrap_units.parquet")
    if um.get("group_assignments_sha256") != want or sha256_file(units_path) != um.get("bootstrap_units_sha256"):
        raise AnalysisError("splits/bootstrap_units.* do not match the frozen group assignments")
    units = pd.read_parquet(units_path, columns=["content_key", "bootstrap_unit", "in_dev_twin"])
    ga = ga.merge(units, on="content_key", how="left", validate="one_to_one")
    if ga["bootstrap_unit"].isna().any():
        raise AnalysisError("bootstrap units do not cover every frozen row")
    ga["partition_component_id"] = ga["component_id"]
    ga["component_id"] = ga["bootstrap_unit"]
    return ga


def local_run_name(label: str, subset: str, seed: int) -> str:
    return f"{label}.{subset}.s{seed}"


def registered_local_runs(cfg: StudyConfig) -> dict[str, Any]:
    path = cfg.work("aligned", "index.json")
    return json.loads(path.read_text(encoding="utf-8")).get("local_runs", {}) if path.is_file() else {}


def run_fingerprint(rd: Path, seed: int) -> str | None:
    """Digest of one seed's scored state in a run directory: its validation, plan and campaign summary files and its
    own spool descriptors (each binds its payload's sha256; the seeds of a tier share one spool). None while the seed
    has no validation file."""

    val = rd / f"validation.seed{seed}.json"
    if not val.is_file():
        return None
    files = {n: sha256_file(rd / n) for n in (val.name, f"plan.seed{seed}.json", f"summary.seed{seed}.json")
             if (rd / n).is_file()}
    spool = []
    for p in sorted((rd / "hf_archive" / "ready").glob("*.ready.json")):
        if int(json.loads(p.read_text(encoding="utf-8"))["identity"]["seed"]) == int(seed):
            spool.append([p.name, sha256_file(p)])
    return sha256_bytes(canonical_json({"files": files, "spool": spool}).encode("utf-8"))


def ingested_fingerprint(cfg: StudyConfig, name: str) -> str | None:
    """The run fingerprint recorded in an ingested local run's report (None for runs ingested without one)."""

    entry = registered_local_runs(cfg).get(name) or {}
    report = cfg.work("aligned", entry["report"]) if entry.get("report") else None
    if report is None or not report.is_file():
        return None
    return json.loads(report.read_text(encoding="utf-8")).get("source_fingerprint")


def ingested_run_problems(cfg: StudyConfig, runs: Sequence[tuple[str, str, str, int]]) -> list[str]:
    """Runs whose ingested rows are missing or no longer match the run directory (re-run, resumed or re-scored)."""

    index = registered_local_runs(cfg)
    problems = []
    for label, slug, subset, seed in runs:
        name = local_run_name(label, subset, seed)
        if name not in index:
            problems.append(f"{name} is not ingested")
            continue
        fp, recorded = run_fingerprint(run_dir(cfg, slug, subset), seed), ingested_fingerprint(cfg, name)
        if fp is None:
            problems.append(f"{name} is ingested but its run has no validation file now")
        elif recorded is None:
            problems.append(f"{name} was ingested without a run fingerprint (ingest it again with ingest-runs)")
        elif recorded != fp:
            problems.append(f"{name} was ingested from a different state of its run (re-run ingest-runs)")
    return problems


def ingest_tier(cfg: StudyConfig, tier: str, *, force: bool = False, log: Log = print) -> dict[str, str]:
    """Ingest every completed run of a tier (validated generation and scores) that is not ingested yet.

    A run ingested earlier is ingested again when its fingerprint (``run_fingerprint``) changed since, e.g. after a
    discarded run was repeated (DESIGN.md A4) or an incomplete one resumed (A5); ``force`` re-ingests regardless.
    """

    done = registered_local_runs(cfg)
    status: dict[str, str] = {}
    for label, slug, subset, seed in tier_runs(cfg, tier):
        name = local_run_name(label, subset, seed)
        rd = run_dir(cfg, slug, subset)
        val = rd / f"validation.seed{seed}.json"
        if not val.is_file() or not json.loads(val.read_text(encoding="utf-8")).get("complete"):
            status[name] = "not_complete" + (" (an earlier ingest is stale)" if name in done else "")
            continue
        fp = run_fingerprint(rd, seed)
        if name in done and not force:
            if ingested_fingerprint(cfg, name) == fp:
                status[name] = "already_ingested"
                continue
            log(f"[analysis] {name}: the run changed since it was ingested; ingesting it again")
        df = ig.ingest_local_run(cfg, rd, label, None, seed, "local_run", log=log)
        df.attrs.setdefault("ingest_report", {})["source_fingerprint"] = fp
        ig.write_local_run(cfg, df, name)
        status[name] = f"{'re-ingested' if name in done else 'ingested'} ({len(df)} rows)"
        log(f"[analysis] ingested {name}: {len(df)} rows")
    return status


def load_rows(cfg: StudyConfig, *, labels: Sequence[str], seeds: Sequence[int], benchmarks: Sequence[str],
              local: dict[str, Sequence[int]] | None = None, subset: str | None = None,
              columns: Sequence[str] = ROW_COLUMNS) -> pd.DataFrame:
    """Archive rows of ``labels`` plus the named local runs; ``local[label]`` lists that label's local seeds.

    A local seed overrides the archive slice of the same (label, seed) only (the existing seed-42 B/T reruns); the
    label's other seeds still come from the archive, so one (model_label, seed, content_key) never appears twice.
    """

    local = {l: [int(s) for s in v] for l, v in (local or {}).items()}
    frames = []
    archive_seeds: dict[tuple[int, ...], list[str]] = {}
    for label in labels:
        if label in ARCHIVE_LABELS:
            aseeds = tuple(int(s) for s in seeds if int(s) not in local.get(label, ()))
            if aseeds:
                archive_seeds.setdefault(aseeds, []).append(label)
    for aseeds, alabels in archive_seeds.items():
        frames.append(ig.load_aligned(cfg, columns=list(columns), labels=alabels, seeds=list(aseeds),
                                      benchmarks=list(benchmarks), include_archive=True))
    index = registered_local_runs(cfg)
    for label, lseeds in local.items():
        for s in lseeds:
            if label in LOCAL_RERUN_NAMES and s == 42:
                name = LOCAL_RERUN_NAMES[label]
            elif subset is None:
                raise AnalysisError(f"local run of {label} seed {s} needs a subset")
            else:
                name = local_run_name(label, subset, s)
            if name not in index:
                raise AnalysisError(f"local run {name} is not ingested")
            frames.append(ig.load_aligned(cfg, columns=list(columns), benchmarks=list(benchmarks),
                                          include_archive=False, local_runs=[name]))
    if not frames:
        raise AnalysisError(f"no rows requested (labels {list(labels)}, seeds {list(seeds)}, local {local})")
    rows = pd.concat(frames, ignore_index=True)
    rows["seed"] = pd.to_numeric(rows["seed"]).astype(int)
    rows = rows[rows["seed"].isin([int(s) for s in seeds])]
    dup = rows.duplicated(["model_label", "seed", "content_key"])
    if dup.any():
        raise AnalysisError(f"{int(dup.sum())} duplicate (label, seed, content) rows")
    if "official_excluded" in rows.columns and rows["official_excluded"].dtype != bool:
        rows["official_excluded"] = bh._as_bool(rows["official_excluded"], "official_excluded")
    return rows.reset_index(drop=True)


def restrict(rows: pd.DataFrame, ga: pd.DataFrame, partition: str) -> pd.DataFrame:
    """'all' | 'development' | 'confirmation' | 'development_screen' | 'smoke'."""

    if partition == "all":
        keep = set(ga["content_key"])
    elif partition in ("development", "confirmation"):
        keep = set(ga.loc[ga["split_role"] == partition, "content_key"])
    elif partition == "development_screen":
        keep = set(ga.loc[ga["in_dev_screen"], "content_key"])
    elif partition == "smoke":
        keep = set(ga.loc[ga["in_smoke"], "content_key"])
    else:
        raise AnalysisError(f"unknown partition {partition}")
    missing = set(rows["content_key"]) - set(ga["content_key"])
    if missing:
        raise AnalysisError(f"{len(missing)} content keys are absent from the frozen split")
    return rows[rows["content_key"].isin(keep)].reset_index(drop=True)


def with_binary(rows: pd.DataFrame, metrics: dict[str, Any], binary: Sequence[str]) -> pd.DataFrame:
    """Official row scores of the binary-eligible benchmarks as ``binary_correct`` (null elsewhere)."""

    rows = rows.copy()
    rows["binary_correct"] = np.nan
    for (label, seed, bench), idx in rows.groupby(["model_label", "seed", "benchmark_id"], sort=False).indices.items():
        if bench not in binary:
            continue
        sub = rows.iloc[idx]
        scores = metrics[bench].row_scores(sub)
        vals = pd.to_numeric(scores, errors="raise").to_numpy(dtype=float)
        if not np.isin(vals, (0.0, 1.0)).all():
            raise AnalysisError(f"{bench}/{label}/{seed}: binary diagnostic needs 0/1 row scores")
        rows.loc[sub.index, "binary_correct"] = vals
    return rows


def components_of(rows: pd.DataFrame, ga: pd.DataFrame) -> pd.Series:
    return ga.set_index("content_key")["component_id"].reindex(rows["content_key"].unique())


def run_bootstrap(cfg: StudyConfig, rows: pd.DataFrame, metrics: dict[str, Any], ga: pd.DataFrame, *,
                  labels: Sequence[str], seeds: Sequence[int], benchmarks: Sequence[str],
                  comparisons: Sequence[str] = (), family: Sequence[str] = (), allow_partial: bool = False,
                  n_replicates: int | None = None) -> st.BootstrapResult:
    """Paired cluster bootstrap with the configured settings; when an adapter failed (``benchmarks`` short of the
    suite) its categories are reported as renamed partial categories (DESIGN.md section 5) instead of failing."""

    sc = cfg.raw["statistics"]
    partial = allow_partial or bool(set(cfg.benchmark_keys()) - set(benchmarks))
    return st.bootstrap_from_rows(
        rows, metrics, ga, labels=list(labels), seeds=list(seeds), benchmarks=list(benchmarks),
        categories=cfg.categories(), allow_partial_categories=partial, comparisons=list(comparisons),
        family=list(family), n_replicates=int(n_replicates or sc["bootstrap_replicates"]),
        confidence=float(sc["confidence"]), seed=int(sc["seed"]))


def _json_safe(value: Any) -> Any:
    """NaN/inf -> null and numpy scalars -> Python, recursively, so summaries are strict JSON."""

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    clean = _json_safe(value)
    json.dumps(clean, allow_nan=False)   # strict JSON or fail before writing
    write_json(path, clean)


def _write(df: pd.DataFrame, path: Path, **context: Any) -> Path:
    out = df.copy()
    for k, v in context.items():
        out.insert(0, k, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def seed_table(res: st.BootstrapResult) -> pd.DataFrame:
    rows = []
    for li, lab in enumerate(res.labels):
        for si, s in enumerate(res.seeds):
            for bi, b in enumerate(res.benchmarks):
                rows.append({"model_label": lab, "seed": s, "benchmark_id": b,
                             "official_score": float(res.point_seed_scores[li, si, bi])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Stage B


def stage_b(cfg: StudyConfig, *, log: Log = print) -> dict[str, Any]:
    out = cfg.work("tables", "stage_b")
    metrics = mt.build_metrics(cfg)
    keys = macro_benchmarks(cfg, metrics)
    binary = binary_benchmarks(cfg)
    seeds = [int(s) for s in cfg.raw["archive"]["seeds"]]
    ga = assignments(cfg)
    rows = load_rows(cfg, labels=ARCHIVE_LABELS, seeds=seeds, benchmarks=keys)
    log(f"[stage_b] {len(rows)} archive rows, {len(keys)} macro benchmarks, {len(binary)} binary-eligible")
    summary: dict[str, Any] = {"benchmarks": keys, "binary_benchmarks": binary, "seeds": seeds}

    # official scores: full split (reproduction check against the archived aggregates) and each partition
    aggs = ig.load_aggregates(cfg)
    for part in ("all", "development", "confirmation"):
        sub = restrict(rows, ga, part)
        res = run_bootstrap(cfg, sub, metrics, ga, labels=ARCHIVE_LABELS, seeds=seeds, benchmarks=keys,
                            comparisons=["T-B", "V@P0-B", "T-V@P0"])
        if part == "all":
            st_tab = seed_table(res)
            a = aggs[aggs["source"] == "archive"][["model_label", "seed", "benchmark_id", "official_value"]].copy()
            a["seed"] = pd.to_numeric(a["seed"]).astype(int)
            m = st_tab.merge(a, on=["model_label", "seed", "benchmark_id"], how="left", validate="one_to_one")
            if m["official_value"].isna().any():
                raise AnalysisError("archived aggregate missing for a full-split slice")
            err = float((m["official_score"] - m["official_value"]).abs().max())
            if not err <= REPRODUCTION_TOLERANCE:
                raise AnalysisError(f"full-split official scores differ from the archived aggregates by up to {err:.3g} "
                                    f"(tolerance {REPRODUCTION_TOLERANCE:g} percentage points)")
            summary["full_split_max_abs_error_vs_archive"] = err
        _write(res.scores, out / f"official_scores.{part}.csv", partition=part)
        _write(res.differences, out / f"differences.{part}.csv", partition=part)
        _write(seed_table(res), out / f"seed_scores.{part}.csv", partition=part)
        _write_json(out / f"bootstrap.{part}.json", res.summary())
        ov = res.scores[res.scores["level"] == "overall"]
        summary[f"overall_{part}"] = {r.model_label: [round(r.estimate, 3), round(r.ci_low, 3), round(r.ci_high, 3)]
                                      for r in ov.itertuples()}
        dv = res.differences[res.differences["level"] == "overall"]
        summary[f"differences_{part}"] = {r.comparison: [round(r.estimate, 3), round(r.ci_low, 3), round(r.ci_high, 3),
                                                         float(r.p_value)] for r in dv.itertuples()}
        log(f"[stage_b] {part}: {summary[f'overall_{part}']} {summary[f'differences_{part}']}")

    # behaviour tables on binary-eligible rows (full split, all seeds)
    brows = with_binary(rows[rows["benchmark_id"].isin(binary)], metrics, binary)
    matrix = bh.build_outcome_matrix(brows, expected_benchmarks=binary)
    spec = bh.BootstrapSpec.from_config(cfg.raw["statistics"])
    tables = bh.behavior_tables(matrix, categories=cfg.categories(), base="B", trace="T", vero="V@P0",
                                components=components_of(brows, ga), uncertainty=spec)
    summary["behavior_table_sha256"] = bh.write_tables(tables, out)
    _write_json(out / "summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ merge tiers


def _tier_local(cfg: StudyConfig, tier: str) -> dict[str, list[int]]:
    local: dict[str, list[int]] = {}
    for label, _, _, seed in tier_runs(cfg, tier):
        local.setdefault(label, []).append(int(seed))
    return local


def capture_tables(cfg: StudyConfig, rows: pd.DataFrame, metrics: dict[str, Any], ga: pd.DataFrame,
                   merges: Sequence[str], vero: str, seeds: Sequence[int],
                   benchmarks: Sequence[str] | None = None) -> pd.DataFrame:
    binary = list(benchmarks) if benchmarks is not None else binary_benchmarks(cfg)
    brows = with_binary(rows[rows["benchmark_id"].isin(binary)], metrics, binary)
    matrix = bh.build_outcome_matrix(brows, expected_benchmarks=binary)
    spec = bh.BootstrapSpec.from_config(cfg.raw["statistics"])
    variants = bh.CAPTURE_VARIANTS if len(seeds) > 1 else ("contemporaneous",)
    frames = []
    for merge in merges:
        # capture_damage labels its rows with merge/trace/vero/base itself
        frames.append(bh.capture_damage(matrix, merge=merge, trace="T", vero=vero, base="B",
                                        categories=cfg.categories(), variants=variants,
                                        components=components_of(brows, ga), uncertainty=spec))
    return pd.concat(frames, ignore_index=True)


def screen(cfg: StudyConfig, *, log: Log = print) -> dict[str, Any]:
    """Development screen (seed 42): P1 models with the archived parents, the P2 diagnostic separately."""

    out = cfg.work("tables", "screen")
    metrics = mt.build_metrics(cfg)
    keys = macro_benchmarks(cfg, metrics)
    ga = assignments(cfg)
    local = _tier_local(cfg, "development")
    seeds = [int(s) for s in cfg.raw["sampling"]["development_seeds"]]
    stale = ingested_run_problems(cfg, tier_runs(cfg, "development"))
    if stale:
        raise AnalysisError("development screen inputs are not current: " + "; ".join(stale[:8]))
    summary: dict[str, Any] = {"benchmarks": keys, "seeds": seeds}
    p1 = [l for l in local if l.endswith("@P1")]
    p2 = [l for l in local if l.endswith("@P2")]
    rows = load_rows(cfg, labels=[*ARCHIVE_LABELS, *p1, *p2], seeds=seeds, benchmarks=keys,
                     local={l: local[l] for l in [*p1, *p2]}, subset="development_screen")
    rows = restrict(rows, ga, "development_screen")
    groups = {
        "P1": ([*ARCHIVE_LABELS, *p1], ["M050-T", "M050-V@P1", "M050-CV050", "M050-CT050", "CV050-B", "CT050-B",
                                        "V@P1-V@P0", "V@P1-B", "T-B"]),
        "P2": (["V@P0", *p2], ["M050@P2-T@P2", "M050@P2-V@P0", "M050@P2-CV050@P2", "M050@P2-CT050@P2",
                               "T@P2-B@P2", "V@P0-B@P2"]),
    }
    for name, (labels, comps) in groups.items():
        res = run_bootstrap(cfg, rows, metrics, ga, labels=labels, seeds=seeds, benchmarks=keys, comparisons=comps)
        _write(res.scores, out / f"scores.{name}.csv", tier="development_screen", protocol_group=name)
        _write(res.differences, out / f"differences.{name}.csv", tier="development_screen", protocol_group=name)
        _write(seed_table(res), out / f"seed_scores.{name}.csv", tier="development_screen", protocol_group=name)
        _write_json(out / f"bootstrap.{name}.json", res.summary())
        ov = res.scores[res.scores["level"] == "overall"]
        summary[f"overall_{name}"] = {r.model_label: [round(r.estimate, 3), round(r.ci_low, 3), round(r.ci_high, 3)]
                                      for r in ov.itertuples()}
        dv = res.differences[res.differences["level"] == "overall"]
        summary[f"differences_{name}"] = {r.comparison: [round(r.estimate, 3), round(r.ci_low, 3), round(r.ci_high, 3),
                                                         float(r.p_value)] for r in dv.itertuples()}
    cd = capture_tables(cfg, rows[rows["model_label"].isin([*ARCHIVE_LABELS, *p1])], metrics, ga,
                        merges=["M050@P1", "CV050@P1", "CT050@P1"], vero="V@P1", seeds=seeds)
    _write(cd, out / "merge_capture_damage.csv", tier="development_screen")
    _write_json(out / "summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ selection lock


def _complete(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text(encoding="utf-8")).get("complete"))


def verify_lock(cfg: StudyConfig, *, design_path: Path | None = None,
                runs: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check that ``reports/selection_lock.json`` still holds (DESIGN.md A4); returns (lock, record for the summary).

    ``evaluation.verify_lock`` recomputes every frozen field (DESIGN.md may only have grown by appended amendments,
    which are recorded). With ``runs`` (the confirmation analysis) the locked runs must also be ingested from their
    current state and served at the locked view revision, and each confirmed model's smoke and development runs must
    be complete (a confirmation run counts only then).
    """

    design_path = design_path or PROJECT_ROOT / "DESIGN.md"
    try:
        report = ev.verify_lock(cfg, design_path)
    except (RuntimeError, OSError) as error:
        raise AnalysisError(f"the selection lock does not hold: {error}") from error
    lock = json.loads(cfg.work("reports", LOCK_NAME).read_text(encoding="utf-8"))
    problems: list[str] = []
    if runs:
        locked_runs = [(r["label"], r["slug"], r["subset"], int(r["seed"])) for r in lock["confirmation_runs"]]
        problems += ingested_run_problems(cfg, locked_runs)
        index = registered_local_runs(cfg)
        for label, _, subset, seed in locked_runs:
            entry = index.get(local_run_name(label, subset, seed)) or {}
            report_path = cfg.work("aligned", entry["report"]) if entry.get("report") else None
            served = (json.loads(report_path.read_text(encoding="utf-8")).get("runtime_revision")
                      if report_path is not None and report_path.is_file() else None)
            want = ((lock.get("views") or {}).get(label) or {}).get("registered_revision")
            if entry and served != want:
                problems.append(f"{label} seed {seed} was served at {served}, the lock froze {want}")
        selected = set(lock["selected_candidates"])
        for tier in ("smoke", "development"):
            for label, slug, subset, seed in tier_runs(cfg, tier):
                if label in selected and not _complete(run_dir(cfg, slug, subset) / f"validation.seed{seed}.json"):
                    problems.append(f"{label} {tier} seed {seed} is not complete (precondition, DESIGN.md A4)")
    if problems:
        raise AnalysisError("confirmation inputs do not satisfy the lock: " + "; ".join(problems[:10]))
    record = {"locked_at": lock.get("locked_at"), "design_sha256_locked": lock.get("design_sha256"),
              "design_sha256_current": sha256_file(design_path),
              "amendments_after_lock": list(report.get("post_lock_amendments", [])),
              "runner_code_changed_since_lock": list(report.get("runner_code_changed", [])),
              "fields_not_recorded_by_lock": list(report.get("not_recorded", [])), "runs_checked": bool(runs)}
    return lock, record


# ------------------------------------------------------------------------------------------------ confirmation


def confirm(cfg: StudyConfig, *, log: Log = print) -> dict[str, Any]:
    lock, lock_record = verify_lock(cfg, runs=True)
    if lock_record["amendments_after_lock"]:
        log(f"[confirm] DESIGN.md amendments appended after the lock: {lock_record['amendments_after_lock']}")
    out = cfg.work("tables", "confirmation")
    metrics = mt.build_metrics(cfg)
    keys = macro_benchmarks(cfg, metrics)
    ga = assignments(cfg)
    local: dict[str, list[int]] = {}
    for r in lock["confirmation_runs"]:
        local.setdefault(r["label"], []).append(int(r["seed"]))
    seeds = sorted({s for v in local.values() for s in v})
    labels = [*ARCHIVE_LABELS, *local]
    family = lock["primary_family"]
    comps = [*family, "CV050-B", "CT050-B", "V@P1-B", "T-B", "V@P1-V@P0", "M050-B"]
    rows = load_rows(cfg, labels=labels, seeds=seeds, benchmarks=keys, local=local, subset="confirmation")
    rows = restrict(rows, ga, "confirmation")
    res = run_bootstrap(cfg, rows, metrics, ga, labels=labels, seeds=seeds, benchmarks=keys, comparisons=comps,
                        family=family)
    _write(res.scores, out / "scores.csv", tier="confirmation")
    _write(res.differences, out / "differences.csv", tier="confirmation")
    _write(seed_table(res), out / "seed_scores.csv", tier="confirmation")
    _write_json(out / "bootstrap.json", res.summary())
    summary: dict[str, Any] = {"benchmarks": keys, "seeds": seeds, "family": family, "lock": lock_record}
    ov = res.scores[res.scores["level"] == "overall"]
    summary["overall"] = {r.model_label: [round(r.estimate, 3), round(r.ci_low, 3), round(r.ci_high, 3)]
                          for r in ov.itertuples()}
    dv = res.differences[res.differences["level"] == "overall"]
    summary["differences"] = {r.comparison: {k: (float(getattr(r, k)) if getattr(r, k, None) is not None else None)
                                             for k in ("estimate", "ci_low", "ci_high", "p_value", "p_holm")
                                             if hasattr(r, k)} for r in dv.itertuples()}
    cd = capture_tables(cfg, rows[rows["model_label"].isin([*ARCHIVE_LABELS[:2], *local])], metrics, ga,
                        merges=["M050@P1", "CV050@P1", "CT050@P1"], vero="V@P1", seeds=seeds)
    _write(cd, out / "merge_capture_damage.csv", tier="confirmation")

    # environment sensitivity: seed 42 only, archived B/T vs the existing local reruns of B/T
    sens = {}
    for which, loc in (("archived_BT", {}), ("local_rerun_BT", {"B": [42], "T": [42]})):
        r42 = load_rows(cfg, labels=["B", "T", *local], seeds=[42], benchmarks=keys,
                        local={**{l: [42] for l in local}, **loc}, subset="confirmation")
        r42 = restrict(r42, ga, "confirmation")
        rs = run_bootstrap(cfg, r42, metrics, ga, labels=["B", "T", *local], seeds=[42], benchmarks=keys,
                           comparisons=[c for c in comps if "V@P0" not in c])
        _write(rs.differences, out / f"sensitivity_seed42.{which}.csv", tier="confirmation", parents=which)
        d = rs.differences[rs.differences["level"] == "overall"]
        sens[which] = {r.comparison: [round(r.estimate, 3), round(r.ci_low, 3), round(r.ci_high, 3)]
                       for r in d.itertuples()}
    summary["environment_sensitivity_seed42"] = sens

    # amendment A7: the primary family again without the confirmation rows that have a development twin
    twin_free = rows[~rows["content_key"].isin(set(ga.loc[ga["in_dev_twin"], "content_key"]))]
    rt = run_bootstrap(cfg, twin_free, metrics, ga, labels=labels, seeds=seeds, benchmarks=keys,
                       comparisons=comps, family=family)
    _write(rt.differences, out / "sensitivity_without_dev_twins.csv", tier="confirmation",
           rows_dropped=int(rows["content_key"].nunique() - twin_free["content_key"].nunique()))
    d = rt.differences[rt.differences["level"] == "overall"]
    summary["sensitivity_without_dev_twins"] = {
        "rows_dropped": int(rows["content_key"].nunique() - twin_free["content_key"].nunique()),
        "differences": {r.comparison: [round(r.estimate, 3), round(r.ci_low, 3), round(r.ci_high, 3)]
                        for r in d.itertuples()}}
    _write_json(out / "summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ base reference

BASE_REFERENCE = "B (reference)"
# V@P1's LaTeX-wrapped option letters are scored 0 on these two benchmarks, which moves rows into the TRACE-only group
FORMAT_AFFECTED = ("puzzlevqa", "visualpuzzles")


def with_base_reference(rows: pd.DataFrame) -> pd.DataFrame:
    """Append a copy of B's rows labelled ``BASE_REFERENCE``, so B can be scored like a merge (capture_damage needs
    four distinct labels)."""

    ref = rows[rows["model_label"] == "B"].copy()
    ref["model_label"] = BASE_REFERENCE
    return pd.concat([rows, ref], ignore_index=True)


def capture_base_reference(cfg: StudyConfig, *, log: Log = print) -> pd.DataFrame:
    """Post-hoc, written after the confirmation tables were read (analysis_report.md §4.4); descriptive, outside the
    locked family. The base model's own capture/damage rates on the confirmation groups, beside the three merges: the
    rate a merge would reach by doing nothing. Computed on the 21 binary benchmarks and again without the two
    format-affected puzzle benchmarks."""

    lock, _ = verify_lock(cfg, runs=True)
    local: dict[str, list[int]] = {}
    for r in lock["confirmation_runs"]:
        local.setdefault(r["label"], []).append(int(r["seed"]))
    seeds = sorted({s for v in local.values() for s in v})
    metrics = mt.build_metrics(cfg)
    ga = assignments(cfg)
    binary = binary_benchmarks(cfg)
    rows = load_rows(cfg, labels=["B", "T", *local], seeds=seeds, benchmarks=binary, local=local,
                     subset="confirmation")
    rows = with_base_reference(restrict(rows, ga, "confirmation"))
    merges = ["M050@P1", "CV050@P1", "CT050@P1", BASE_REFERENCE]
    frames = []
    for name, benches in (("all_binary", binary), ("without_" + "_".join(FORMAT_AFFECTED),
                                                    [b for b in binary if b not in FORMAT_AFFECTED])):
        cd = capture_tables(cfg, rows[rows["benchmark_id"].isin(benches)], metrics, ga, merges=merges, vero="V@P1",
                            seeds=seeds, benchmarks=benches)
        cd.insert(0, "benchmark_set", name)
        frames.append(cd)
        log(f"[capture-reference] {name}: {len(benches)} benchmarks")
    table = pd.concat(frames, ignore_index=True)
    # overall and category rows only: the benchmark-level rows of the three merges are in merge_capture_damage.csv
    table = table[table["level"].isin(["overall", "category"])].reset_index(drop=True)
    _write(table, cfg.work("tables", "confirmation", "merge_capture_damage.base_reference.csv"), tier="confirmation",
           post_hoc=True)
    return table


# ------------------------------------------------------------------------------------------------ answer format

FORMAT_COLUMNS = ["model_label", "seed", "benchmark_id", "content_key", "source_ordinal", "source_index",
                  "source_row_sha256", "source_record_sha256", "archived_row_score", "official_excluded",
                  "extraction_status", "extraction_value_json", "judge_output", "response_text", "source", "run_id"]
FORMAT_TABLE_DIR = {"development": "screen", "confirmation": "confirmation"}


def answer_format_table(cfg: StudyConfig, tier: str, *, log: Log = print) -> pd.DataFrame:
    """Diagnostic, never a score (handoff 8.3): per letter-answer benchmark, model and seed of a tier, the official
    accuracy, the share of responses ending in a LaTeX-wrapped option letter, and the share scored 0 whose leniently
    read final letter is the correct option. Parents come from the archive, new models from the tier's local runs."""

    from . import diagnostics as dg

    subset = TIER_SUBSET[tier]
    benches = list(dg.LETTER_ANSWER_BENCHMARKS)
    metrics = mt.build_metrics(cfg, benches)
    ga = assignments(cfg)
    local = _tier_local(cfg, tier)
    seeds = sorted({s for v in local.values() for s in v})
    out = []
    for b in benches:
        meta = mt.load_benchmark_meta(cfg, b)
        for label in [*ARCHIVE_LABELS, *local]:
            for seed in (seeds if label in ARCHIVE_LABELS else local[label]):
                if label in ARCHIVE_LABELS:
                    r = ig.load_aligned(cfg, columns=FORMAT_COLUMNS, labels=[label], seeds=[seed], benchmarks=[b])
                else:
                    r = ig.load_aligned(cfg, columns=FORMAT_COLUMNS, benchmarks=[b], include_archive=False,
                                        local_runs=[local_run_name(label, subset, seed)])
                if r["official_excluded"].dtype != bool:
                    r["official_excluded"] = r["official_excluded"].map(lambda v: str(v) == "True")
                r = restrict(r, ga, subset).reset_index(drop=True)
                res = dg.answer_format_sensitivity(r, metrics[b].row_scores(r), meta, b)
                out.append({"model_label": label, "seed": int(seed), **res})
        log(f"[format] {b}: {len(out)} slices")
    table = pd.DataFrame(out)
    _write(table, cfg.work("tables", FORMAT_TABLE_DIR[tier], "answer_format_sensitivity.csv"), tier=subset)
    return table
