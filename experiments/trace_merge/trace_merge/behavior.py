"""Behavioural transitions on binary-eligible content units (handoff 5.1, 5.4, 8.2; DESIGN.md section 8).

Every function here is a pure function of the normalized aligned table (one row per model_label x seed x content row)
to which the caller has added a 0/1 ``binary_correct`` column for binary-eligible benchmarks only. Which benchmarks are
eligible is the metric registry's decision, not this module's: WeMath's strict metric and any fractional score must
never reach these tables, so a value outside {0, 1} fails instead of being thresholded.

Design choices that are easy to get wrong, fixed here once:

* A unit is a content row (benchmark_id, content_key). Decoding seeds are repeated measurements of the same unit, not
  extra units: seed-averaged rows use mean-per-seed counts over the same unit set, so an unconditional seed-averaged
  rate equals the mean of the per-seed rates, and a conditional one is the ratio of the summed counts.
* A missing generation is not a wrong answer. A unit lacking any required (model_label, seed) outcome fails the call
  (``missing="raise"``, default) or is dropped and counted in ``n_units_dropped_missing`` (``missing="drop"``).
  Rows flagged ``official_excluded`` fail the matrix build by default, like ``metrics.archived_binary_scores``,
  because their official treatment is not defined; ``exclusions="drop"`` drops and counts them instead.
  ``coverage_table`` records exclusions, missingness and benchmarks without any binary outcome.
* Rates are percentages of an explicit denominator that is always printed next to them. Conditional rates
  (corrections / base errors, regressions / base successes) are never on the same footing as population percentage
  points; the net change is the difference of the two unconditional rates and equals the accuracy difference.
* Category and overall rows come in two flavours: ``pooled_rows`` (ratio of summed counts, dominated by large
  benchmarks) and ``benchmark_macro`` (equal-weight mean of within-benchmark rates, the suite's weighting). Persistence
  and capture/damage tables are pooled only because their conditional groups can be empty inside a benchmark.
  A category with only some of its suite benchmarks present is named as partial (``category_scope_name``).
* The either-parent-correct score is a hindsight selection diagnostic over two parental responses and is labelled so
  in every row that carries it.
* Uncertainty (handoff 6.2): with a ``BootstrapSpec`` and component ids, the headline rows of every table (the
  seed-averaged or pooled rows, or the single-seed rows when only one seed exists) get percentile intervals from a
  bootstrap over connected components that uses the same replicate stream as ``statistics.paired_cluster_bootstrap``.
  All labels and seeds of a unit move with its component, so paired changes keep their pairing. These intervals are
  descriptive; McNemar tests are not used because rows sharing an image or seeds are not independent pairs.
"""

from __future__ import annotations

import io
import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import sha256_bytes
from .statistics import category_scope_name, iter_weight_blocks

VALUE_COLUMN = "binary_correct"
KEY_COLUMNS = ("model_label", "seed", "benchmark_id", "content_key")
SINGLE_SEED = "single_seed"
MEAN_OVER_SEEDS = "mean_over_seeds"
POOLED_PAIRS = "pooled_over_ordered_seed_pairs"
SINGLE_CONDITION = "single_condition"
POOLED_CONDITIONS = "pooled_over_conditions"
WITHIN_BENCHMARK = "within_benchmark"
POOLED_ROWS = "pooled_rows"
BENCHMARK_MACRO = "benchmark_macro"
HINDSIGHT_NOTE = ("hindsight selection diagnostic over two parental responses; not an implementable router, not a "
                  "compute-matched baseline, not an upper bound on a merged model")
SWAP_NULL_NOTE = ("label-swap null: expected persistence if, on every unit and seed where the two pattern labels "
                  "disagree, which label holds which value were a fair coin flip independent across seeds; keeps each "
                  "unit's observed disagreement (difficulty, decoding instability) and removes only a stable "
                  "direction. The eval-seed base rate is descriptive, not a noise reference: item heterogeneity alone "
                  "puts persistence above it")
NO_NULL_NOTE = "no label-swap null: the pattern does not have exactly two labels with opposite values"
CAPTURE_VARIANTS = ("contemporaneous", "independent_repeat", "stable_other_seeds")
# group name, trace parent value, vero parent value, metric name, merge outcome that counts as the event
CAPTURE_GROUPS = (
    ("trace_only", 1, 0, "trace_only_capture", 1),
    ("vero_only", 0, 1, "vero_only_capture", 1),
    ("shared_success", 1, 1, "shared_success_damage", 0),
    ("joint_failure", 0, 0, "joint_failure_rescue", 1),
)
UNSTABLE_GROUP = "unstable_parent_pattern"
NVALID_SUFFIX = "__nvalid"
CI_SUFFIXES = ("_ci_low", "_ci_high")


class BehaviorError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------- uncertainty
@dataclass(frozen=True)
class BootstrapSpec:
    """Cluster-bootstrap settings for the behaviour diagnostics (DESIGN.md section 7; handoff 5.2, 6.2).

    Components are drawn with replacement with ``statistics.iter_weight_blocks`` (the primary bootstrap's stream).
    A replicate whose draw leaves a rate's denominator empty leaves that rate undefined; an interval is reported only
    when at most ``max_invalid_fraction`` of the replicates are undefined (otherwise NaN), so a conditional rate in a
    tiny group never gets an interval that silently conditions on the group being drawn.
    ``bootstrap_min_valid_replicates`` gives the smallest replicate count behind any interval of a row.
    ``min_groups`` (the exploratory minimum of independent groups per slice) adds ``below_min_groups``.
    """

    n_replicates: int = 10_000
    seed: int = 314159
    confidence: float = 0.95
    max_invalid_fraction: float = 0.01
    min_groups: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_replicates) < 2:
            raise BehaviorError("n_replicates must be at least 2")
        if not 0 < float(self.confidence) < 1:
            raise BehaviorError("confidence must be in (0, 1)")
        if not 0 <= float(self.max_invalid_fraction) <= 1:
            raise BehaviorError("max_invalid_fraction must be in [0, 1]")

    @classmethod
    def from_config(cls, statistics: Mapping[str, Any], **overrides: Any) -> "BootstrapSpec":
        """From the config's ``statistics`` block (replicates, seed, confidence, exploratory minimum)."""

        base: dict[str, Any] = {"n_replicates": int(statistics["bootstrap_replicates"]),
                                "seed": int(statistics["seed"]), "confidence": float(statistics["confidence"])}
        if statistics.get("exploratory_min_groups_per_slice") is not None:
            base["min_groups"] = int(statistics["exploratory_min_groups_per_slice"])
        base.update(overrides)
        return cls(**base)


# ---------------------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class OutcomeMatrix:
    """0/1 outcomes with one row per content unit and one column per (model_label, seed).

    ``values`` is indexed by (benchmark_id, content_key) and holds NaN where a label/seed has no scored row for the
    unit; that NaN is missingness, never an error. ``excluded`` counts officially excluded rows that were removed
    (only with ``exclusions="drop"``); ``ineligible`` lists benchmarks present in the input without any 0/1 value.
    """

    values: pd.DataFrame
    excluded: pd.DataFrame
    ineligible: tuple[str, ...] = ()

    def labels(self) -> list[str]:
        return sorted(set(self.values.columns.get_level_values("model_label")))

    def seeds(self, label: str) -> list[int]:
        cols = self.values.columns
        return sorted(int(s) for lab, s in cols if lab == label)

    def common_seeds(self, labels: Iterable[str]) -> list[int]:
        sets = [set(self.seeds(lab)) for lab in labels]
        return sorted(set.intersection(*sets)) if sets else []

    def column(self, label: str, seed: int) -> np.ndarray:
        try:
            return self.values[(label, int(seed))].to_numpy(dtype=float)
        except KeyError:
            raise BehaviorError(f"no outcomes at all for model_label={label!r} seed={seed}") from None


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    """Normalise a boolean column that may have been stored as strings ('True'/'False') in an archive.

    A null flag is ambiguous (excluded or not?) and fails rather than defaulting either way.
    """

    if series.dtype == bool:
        return series
    mapping = {True: True, False: False, "True": True, "False": False, "true": True, "false": False, 1: True, 0: False}
    out = series.map(lambda v: None if pd.isna(v) else mapping.get(v, None))
    if out.isna().any():
        bad = sorted({repr(v) for v in series[out.isna()].head(5)})
        raise BehaviorError(f"{name} has non-boolean values: {bad}")
    return out.astype(bool)


def build_outcome_matrix(aligned: pd.DataFrame, *, labels: Sequence[str] | None = None,
                         seeds: Sequence[int] | None = None, benchmarks: Sequence[str] | None = None,
                         expected_benchmarks: Sequence[str] | None = None, exclusions: str = "raise",
                         value_col: str = VALUE_COLUMN) -> OutcomeMatrix:
    """Pivot the aligned table into units x (model_label, seed) without ever accepting an ambiguous join.

    Benchmarks whose ``value_col`` is null on every row are not binary-eligible; they are left out and listed in
    ``OutcomeMatrix.ineligible``. Pass ``expected_benchmarks`` (the registry's binary-eligible set) so that a benchmark
    nulled by a failed upstream join fails here instead of vanishing from every table. Inside an eligible benchmark a
    null is a missing outcome. Duplicate (model_label, seed, benchmark_id, content_key) rows and a content_key under
    two benchmarks both fail, because either would silently double-count a unit. Officially excluded rows fail unless
    ``exclusions="drop"`` (then they are removed and counted).
    """

    if exclusions not in ("raise", "drop"):
        raise BehaviorError("exclusions must be 'raise' or 'drop'")
    need = set(KEY_COLUMNS) | {value_col}
    absent = sorted(need - set(aligned.columns))
    if absent:
        raise BehaviorError(f"aligned table lacks columns {absent}")
    cols = list(KEY_COLUMNS) + [value_col] + (["official_excluded"] if "official_excluded" in aligned.columns else [])
    df = aligned.loc[:, cols].copy()
    if labels is not None:
        df = df[df["model_label"].isin(list(labels))]
    try:
        df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    except (ValueError, TypeError) as exc:
        raise BehaviorError(f"seed must be integer-valued: {exc}") from None
    if seeds is not None:
        df = df[df["seed"].isin([int(s) for s in seeds])]
    if benchmarks is not None:
        df = df[df["benchmark_id"].isin(list(benchmarks))]

    excluded = pd.DataFrame(columns=["model_label", "seed", "benchmark_id", "n_excluded"])
    if "official_excluded" in df.columns:
        exc = _as_bool(df["official_excluded"], "official_excluded")
        if exc.any():
            excluded = (df[exc].groupby(["model_label", "seed", "benchmark_id"]).size()
                        .rename("n_excluded").reset_index())
            if exclusions == "raise":
                by_bench = excluded.groupby("benchmark_id")["n_excluded"].sum().to_dict()
                raise BehaviorError(f"{int(exc.sum())} rows are flagged official_excluded (by benchmark: {by_bench}); "
                                    "their official treatment is not defined (metrics refuses them too). Pass "
                                    "exclusions='drop' to drop and count them")
        df = df[~exc].drop(columns=["official_excluded"])

    try:
        vals = pd.to_numeric(df[value_col], errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise BehaviorError(f"{value_col} must be numeric 0/1 or null: {exc}") from None
    bad = vals.notna() & ~vals.isin([0.0, 1.0])
    if bad.any():
        raise BehaviorError(f"{value_col} has {int(bad.sum())} values outside {{0, 1}} (e.g. {vals[bad].iloc[0]}); "
                            "fractional or grouped scores must not enter binary diagnostics")
    eligible = vals.notna().groupby(df["benchmark_id"]).any()
    ineligible = tuple(sorted(str(b) for b in eligible.index[~eligible.to_numpy(dtype=bool)]))
    have = set(str(b) for b in eligible.index[eligible.to_numpy(dtype=bool)])
    if expected_benchmarks is not None:
        expected = set(expected_benchmarks)
        lacking = sorted(expected - have)
        if lacking:
            raise BehaviorError(f"expected binary-eligible benchmarks without any 0/1 outcome: {lacking} (all-null in "
                                f"the input: {sorted(set(lacking) & set(ineligible))}; absent: "
                                f"{sorted(set(lacking) - set(ineligible))})")
        unexpected = sorted(have - expected)
        if unexpected:
            raise BehaviorError(f"benchmarks with 0/1 outcomes outside expected_benchmarks: {unexpected}")
    keep = df["benchmark_id"].map(eligible).fillna(False).to_numpy(dtype=bool)
    df, vals = df[keep], vals[keep]
    if df.empty:
        raise BehaviorError("no binary-eligible rows after filtering")

    dup = df.duplicated(list(KEY_COLUMNS))
    if dup.any():
        first = df[dup].iloc[0]
        raise BehaviorError(f"{int(dup.sum())} duplicate model_label x seed x content rows, e.g. "
                            f"{first['model_label']}/{first['seed']}/{first['benchmark_id']}/{first['content_key']}")
    pairs = df.drop_duplicates(["benchmark_id", "content_key"])
    if pairs["content_key"].duplicated().any():
        raise BehaviorError("a content_key appears under more than one benchmark_id")

    index = pd.MultiIndex.from_frame(df[["benchmark_id", "content_key", "model_label", "seed"]])
    wide = pd.Series(vals.to_numpy(), index=index).unstack(["model_label", "seed"])
    wide = wide.sort_index().sort_index(axis=1).astype(float)
    return OutcomeMatrix(values=wide, excluded=excluded.reset_index(drop=True), ineligible=ineligible)


def missingness_table(matrix: OutcomeMatrix) -> pd.DataFrame:
    """Per (model_label, seed, benchmark): units in the matrix, units with an outcome, and units without one."""

    v = matrix.values
    bench = v.index.get_level_values("benchmark_id")
    present = v.notna().groupby(bench).sum()
    total = v.groupby(bench).size()
    rows = []
    for (label, seed) in v.columns:
        for b in present.index:
            n_present = int(present.loc[b, (label, seed)])
            rows.append({"model_label": label, "seed": int(seed), "benchmark_id": b, "n_units": int(total.loc[b]),
                         "n_present": n_present, "n_missing": int(total.loc[b]) - n_present})
    return pd.DataFrame(rows)


def coverage_table(matrix: OutcomeMatrix) -> pd.DataFrame:
    """What entered the behaviour tables: per-slice presence, missingness and exclusions, plus ineligible benchmarks.

    Written next to the transition tables so a reader can see which rows and benchmarks every rate is (not) about.
    """

    cov = missingness_table(matrix)
    cov["n_excluded"] = 0
    if len(matrix.excluded):
        exc = matrix.excluded.assign(seed=matrix.excluded["seed"].astype(int))
        cov = cov.drop(columns=["n_excluded"]).merge(exc, on=["model_label", "seed", "benchmark_id"], how="outer")
        cov["n_excluded"] = cov["n_excluded"].fillna(0).astype(int)
    cov["status"] = "binary_eligible"
    if matrix.ineligible:
        extra = pd.DataFrame({"benchmark_id": list(matrix.ineligible), "status": "no_binary_outcome"})
        cov = pd.concat([cov, extra], ignore_index=True)
    for col in ("n_units", "n_present", "n_missing", "seed"):
        cov[col] = cov[col].astype("Int64")
    return cov.sort_values(["status", "benchmark_id", "model_label", "seed"], kind="stable").reset_index(drop=True)


def _complete_units(matrix: OutcomeMatrix, cells: Sequence[tuple[str, int]], missing: str) -> np.ndarray:
    """Boolean mask of units that have every required (label, seed) outcome; fails on gaps unless dropping."""

    if missing not in ("raise", "drop"):
        raise BehaviorError("missing must be 'raise' or 'drop'")
    block = np.column_stack([matrix.column(lab, s) for lab, s in cells])
    ok = ~np.isnan(block).any(axis=1)
    if missing == "raise" and not ok.all():
        bench = matrix.values.index.get_level_values("benchmark_id").to_numpy()
        gaps = pd.Series(~ok).groupby(bench).sum()
        gaps = gaps[gaps > 0]
        per_cell = {f"{lab}@seed{s}": int(np.isnan(block[:, i]).sum()) for i, (lab, s) in enumerate(cells)}
        raise BehaviorError(f"{int((~ok).sum())} units lack a required outcome (by benchmark: {gaps.to_dict()}; by "
                            f"cell: {per_cell}); filter the aligned table to the evaluated units or pass "
                            "missing='drop' to drop and count them")
    return ok


def _resolve_seeds(matrix: OutcomeMatrix, labels: Sequence[str], seeds: Sequence[int] | None) -> list[int]:
    if seeds is None:
        common = matrix.common_seeds(labels)
        if not common:
            raise BehaviorError(f"labels {list(labels)} share no seed")
        return common
    out = sorted({int(s) for s in seeds})
    for lab in labels:
        lacking = sorted(set(out) - set(matrix.seeds(lab)))
        if lacking:
            raise BehaviorError(f"{lab} has no outcomes for seeds {lacking}")
    return out


# ---------------------------------------------------------------------------------------------------------- scopes
def overall_scope_name(n_benchmarks: int) -> str:
    """The overall row is over the binary-eligible benchmarks only, never the 24-benchmark suite."""

    return f"{n_benchmarks}-benchmark binary-eligible set"


def _scopes(index: pd.MultiIndex, categories: Mapping[str, Sequence[str]] | None,
            extra_scopes: Mapping[str, pd.Series] | None) -> list[tuple[str, np.ndarray]]:
    """Scope labels per unit for every reporting level (benchmark, category, optional slices, overall).

    An extra scope is a Series indexed by content_key that must cover every unit of the matrix: a unit the caller
    did not label carries an explicit NaN (counted in ``n_units_unlabelled``), while a unit absent from the index
    fails, so a slice keyed by the wrong identifier (e.g. sample_id) cannot silently vanish. Multi-label axes
    (handoff 5.2 ``operation``) are passed as one scope per label, e.g. ``{"operation:counting": yes/no Series}``.
    """

    bench = index.get_level_values("benchmark_id").to_numpy(dtype=object)
    present = set(bench)
    out: list[tuple[str, np.ndarray]] = [("benchmark", bench)]
    if categories is not None:
        b2c: dict[str, str] = {}
        names: dict[str, str] = {}
        for cat, members in categories.items():
            for b in members:
                if b in b2c:
                    raise BehaviorError(f"benchmark {b!r} is in two categories")
                b2c[b] = cat
            names[cat] = category_scope_name(cat, len([b for b in members if b in present]), len(members))
        unknown = sorted(present - set(b2c))
        if unknown:
            raise BehaviorError(f"benchmarks without a category: {unknown}")
        out.append(("category", np.array([names[b2c[b]] for b in bench], dtype=object)))
    keys = pd.Series(index.get_level_values("content_key"))
    for name, series in (extra_scopes or {}).items():
        if name in ("benchmark", "category", "overall"):
            raise BehaviorError(f"extra scope name {name!r} is reserved")
        if series.index.has_duplicates:
            raise BehaviorError(f"extra scope {name!r} maps a content_key twice")
        absent = ~keys.isin(series.index)
        if absent.any():
            raise BehaviorError(f"extra scope {name!r} has no entry for {int(absent.sum())} of {len(keys)} units "
                                f"(e.g. {keys[absent].head(3).tolist()}); index it by content_key and give unlabelled "
                                "units an explicit NaN")
        out.append((name, keys.map(series).to_numpy(dtype=object)))
    out.append(("overall", np.full(len(bench), overall_scope_name(len(present)), dtype=object)))
    return out


def _ratio(num: pd.Series, den: pd.Series, scale: float = 100.0) -> pd.Series:
    num = num.astype(float)
    den = den.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den.to_numpy() > 0, scale * num.to_numpy() / den.to_numpy(), np.nan)
    return pd.Series(out, index=num.index)


RateFn = Callable[[pd.DataFrame], tuple[pd.DataFrame, list[str]]]


@dataclass
class _BootJob:
    """One summarized frame to bootstrap: unit -> group codes, and for macros the group -> row map."""

    frame_pos: int
    codes: np.ndarray            # per kept unit; -1 = not in this level
    n_groups: int
    parent: np.ndarray | None    # macro only: (scope, benchmark) pair -> row of the frame
    n_rows: int


def _summarize(counts: pd.DataFrame, bench: np.ndarray, scopes: list[tuple[str, np.ndarray]], rate_fn: RateFn, *,
               macro: bool, components: np.ndarray | None, dropped: np.ndarray,
               uncertainty: BootstrapSpec | None = None) -> pd.DataFrame:
    """Aggregate per-unit counts to every scope; optional equal-weight benchmark macro for multi-benchmark scopes.

    ``counts`` rows are the kept units (aligned with ``bench`` and each scope array after masking); ``dropped`` gives,
    for all units, whether they were removed for missingness so each scope can report how many it lost. With
    ``uncertainty``, every row gets bootstrap intervals for its rate columns (see ``BootstrapSpec``).
    """

    if uncertainty is not None and components is None:
        raise BehaviorError("bootstrap intervals need component ids (pass components=...)")
    frames: list[pd.DataFrame] = []
    jobs: list[_BootJob] = []
    for level, scope_all in scopes:
        keep_scope = pd.notna(scope_all)
        drop_counts = pd.Series(dropped[keep_scope]).groupby(scope_all[keep_scope]).sum()
        n_unlabelled = int((~keep_scope & ~dropped).sum())
        scope = scope_all[~dropped]
        valid = pd.notna(scope)
        c = counts.loc[valid]
        s = scope[valid]
        b = bench[valid]
        if len(c) == 0:
            continue
        pooled = c.groupby(s, sort=True).sum()
        rates, _ = rate_fn(pooled)
        rates.insert(0, "aggregation", WITHIN_BENCHMARK if level == "benchmark" else POOLED_ROWS)
        rates["benchmark_count"] = pd.Series(b).groupby(s).nunique().reindex(rates.index).to_numpy()
        if components is not None:
            comp = components[~dropped][valid]
            rates["n_groups"] = pd.Series(comp).groupby(s).nunique().reindex(rates.index).to_numpy()
        rates["n_units_dropped_missing"] = drop_counts.reindex(rates.index).fillna(0).astype(int).to_numpy()
        rates["n_units_unlabelled"] = n_unlabelled
        rates.insert(0, "scope", rates.index.astype(str))
        rates.insert(0, "level", level)
        frames.append(rates.reset_index(drop=True))
        if uncertainty is not None:
            codes = np.full(len(counts), -1, dtype=np.int64)
            codes[valid] = pd.Index(pooled.index).get_indexer(s)
            jobs.append(_BootJob(len(frames) - 1, codes, len(pooled), None, len(pooled)))
        if macro and level != "benchmark":
            per_b = c.groupby([s, b], sort=True).sum()
            rb, rate_cols = rate_fn(per_b)
            mean = rb[rate_cols].groupby(level=0).mean()
            undefined = rb[rate_cols].isna().groupby(level=0).any()
            mean = mean.mask(undefined)
            summed = per_b.groupby(level=0).sum()
            mrates, _ = rate_fn(summed)
            mrates[rate_cols] = mean[rate_cols]
            mrates.insert(0, "aggregation", BENCHMARK_MACRO)
            mrates["benchmark_count"] = rb.groupby(level=0).size().reindex(mrates.index).to_numpy()
            if components is not None:
                mrates["n_groups"] = rates.set_index("scope")["n_groups"].reindex(mrates.index.astype(str)).to_numpy()
            mrates["n_units_dropped_missing"] = drop_counts.reindex(mrates.index).fillna(0).astype(int).to_numpy()
            mrates["n_units_unlabelled"] = n_unlabelled
            mrates.insert(0, "scope", mrates.index.astype(str))
            mrates.insert(0, "level", level)
            frames.append(mrates.reset_index(drop=True))
            if uncertainty is not None:
                codes = np.full(len(counts), -1, dtype=np.int64)
                codes[valid] = per_b.index.get_indexer(pd.MultiIndex.from_arrays([s, b]))
                parent = pd.Index(mrates.index).get_indexer(per_b.index.get_level_values(0))
                jobs.append(_BootJob(len(frames) - 1, codes, len(per_b), parent, len(mrates)))
    if jobs:
        _attach_intervals(frames, jobs, counts, components[~dropped], rate_fn, uncertainty)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _attach_intervals(frames: list[pd.DataFrame], jobs: list[_BootJob], counts: pd.DataFrame,
                      components: np.ndarray, rate_fn: RateFn, spec: BootstrapSpec) -> None:
    """Percentile intervals of every rate column of the bootstrapped frames (in place).

    Per replicate, a group's counts are the multiplicity-weighted sums over its components' units, and its rates are
    ``rate_fn`` of those sums, so conditional rates stay ratios of resampled counts. A macro row is the mean over its
    benchmarks' replicate rates and is undefined when any member is, exactly like its point estimate.
    """

    from scipy import sparse

    comp_codes, comp_uniques = pd.factorize(pd.Series(components, dtype=object), sort=True)
    g = len(comp_uniques)
    vals = counts.to_numpy(dtype=float)
    k = vals.shape[1]
    offsets, rows_i, cols_i, data = [], [], [], []
    total = 0
    for job in jobs:
        sel = job.codes >= 0
        cc = comp_codes[sel]
        rows_i.append(np.repeat(cc, k))
        cols_i.append((total + job.codes[sel][:, None] * k + np.arange(k)[None, :]).ravel())
        data.append(vals[sel].ravel())
        offsets.append(total)
        total += job.n_groups * k
    agg = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows_i), np.concatenate(cols_i))),
                            shape=(g, total))
    agg_t = agg.T.tocsr()
    reps_all = np.empty((spec.n_replicates, total))
    start = 0
    for w in iter_weight_blocks(g, spec.n_replicates, spec.seed):
        r = w.shape[0]
        reps_all[start:start + r] = (agg_t @ w.T.astype(np.float64)).T
        start += r
    n_rep = spec.n_replicates
    q = [50 * (1 - spec.confidence), 50 * (1 + spec.confidence)]
    for job, off in zip(jobs, offsets):
        tot = reps_all[:, off:off + job.n_groups * k].reshape(n_rep * job.n_groups, k)
        out, rate_cols = rate_fn(pd.DataFrame(tot, columns=counts.columns))
        reps = out[rate_cols].to_numpy(dtype=float).reshape(n_rep, job.n_groups, len(rate_cols))
        if job.parent is not None:
            reps = np.stack([reps[:, job.parent == i, :].mean(axis=1) for i in range(job.n_rows)], axis=1)
        frame = frames[job.frame_pos]
        point = frame[rate_cols].to_numpy(dtype=float)
        n_valid = np.isfinite(reps).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(reps, q, axis=0)
        ok = np.isfinite(point) & (n_rep - n_valid <= spec.max_invalid_fraction * n_rep + 1e-9)
        lo, hi = np.where(ok, lo, np.nan), np.where(ok, hi, np.nan)
        new: dict[str, Any] = {}
        for j, col in enumerate(rate_cols):
            new[f"{col}_ci_low"] = lo[:, j]
            new[f"{col}_ci_high"] = hi[:, j]
            new[f"{col}{NVALID_SUFFIX}"] = np.where(np.isfinite(point[:, j]), n_valid[:, j], np.nan)
        new["bootstrap_replicates"] = n_rep
        new["bootstrap_seed"] = int(spec.seed)
        if spec.min_groups is not None and "n_groups" in frame.columns:
            new["below_min_groups"] = frame["n_groups"].to_numpy() < spec.min_groups
        frames[job.frame_pos] = pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)


def _collapse_validity(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the per-column valid-replicate counts by their row minimum (``bootstrap_min_valid_replicates``)."""

    cols = [c for c in frame.columns if c.endswith(NVALID_SUFFIX)]
    if not cols:
        return frame
    frame = frame.copy()
    frame["bootstrap_min_valid_replicates"] = frame[cols].min(axis=1, skipna=True)
    return frame.drop(columns=cols)


def _components_array(matrix: OutcomeMatrix, components: pd.Series | None) -> np.ndarray | None:
    if components is None:
        return None
    keys = pd.Series(matrix.values.index.get_level_values("content_key"))
    mapped = keys.map(components)
    if mapped.isna().any():
        raise BehaviorError(f"{int(mapped.isna().sum())} units have no component id")
    return mapped.to_numpy(dtype=object)


def _seed_rows(matrix: OutcomeMatrix, seeds: Sequence[int], counts_for_seed: Callable[[int, np.ndarray], pd.DataFrame],
               rate_fn: RateFn, ok: np.ndarray, *, categories, extra_scopes, components, macro: bool,
               uncertainty: BootstrapSpec | None) -> pd.DataFrame:
    """Per-seed rows plus seed-averaged rows (mean-per-seed counts over one fixed unit set).

    Bootstrap intervals go on the seed-averaged rows, or on the single-seed rows when only one seed is evaluated.
    """

    scopes = _scopes(matrix.values.index, categories, extra_scopes)
    bench = matrix.values.index.get_level_values("benchmark_id").to_numpy(dtype=object)[ok]
    comps = _components_array(matrix, components)
    frames = []
    total = None
    single = len(seeds) == 1
    for s in seeds:
        counts = counts_for_seed(int(s), ok)
        total = counts if total is None else total + counts
        out = _summarize(counts, bench, scopes, rate_fn, macro=macro, components=comps, dropped=~ok,
                         uncertainty=uncertainty if single else None)
        out.insert(2, "seed", int(s))
        out.insert(3, "seed_scope", SINGLE_SEED)
        out.insert(4, "n_seeds", 1)
        frames.append(out)
    if not single:
        out = _summarize(total / len(seeds), bench, scopes, rate_fn, macro=macro, components=comps, dropped=~ok,
                         uncertainty=uncertainty)
        out.insert(2, "seed", pd.NA)
        out.insert(3, "seed_scope", MEAN_OVER_SEEDS)
        out.insert(4, "n_seeds", len(seeds))
        frames.append(out)
    res = pd.concat(frames, ignore_index=True)
    res["seed"] = res["seed"].astype("Int64")
    return res


# ----------------------------------------------------------------------------------------- corrections/regressions
def corrections_regressions(matrix: OutcomeMatrix, *, base: str = "B", models: Sequence[str] = ("T", "V@P0"),
                            seeds: Sequence[int] | None = None, categories: Mapping[str, Sequence[str]] | None = None,
                            extra_scopes: Mapping[str, pd.Series] | None = None, missing: str = "raise",
                            components: pd.Series | None = None,
                            uncertainty: BootstrapSpec | None = None) -> pd.DataFrame:
    """Corrections (B=0, m=1) and regressions (B=1, m=0) of every model against the base.

    All of ``base`` and ``models`` are required on the same units and seeds, so every row carries the absolute
    accuracy of each of them (``acc_<label>_pct``) next to the transition rates, which guards against reading a
    delta without its ceiling or floor. ``net_delta_pp`` is the paired accuracy change; with ``uncertainty`` its
    interval is the component-bootstrap interval of that paired change.
    """

    refs = [base] + [m for m in models if m != base]
    seeds = _resolve_seeds(matrix, refs, seeds)
    ok = _complete_units(matrix, [(lab, s) for lab in refs for s in seeds], missing)
    frames = []
    for model in models:
        if model == base:
            raise BehaviorError("a model cannot be compared with itself as base")

        def counts_for_seed(s: int, mask: np.ndarray, model: str = model) -> pd.DataFrame:
            b = matrix.column(base, s)[mask]
            y = matrix.column(model, s)[mask]
            data = {"n_units": np.ones_like(b), "base_success": b, "base_error": 1 - b, "model_success": y,
                    "corrections": (1 - b) * y, "regressions": b * (1 - y)}
            for lab in refs:
                data[f"success::{lab}"] = matrix.column(lab, s)[mask]
            return pd.DataFrame(data)

        def rate_fn(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
            n = g["n_units"]
            out = pd.DataFrame(index=g.index)
            out["n_units"] = n
            out["n_base_success"] = g["base_success"]
            out["n_base_error"] = g["base_error"]
            out["n_corrections"] = g["corrections"]
            out["n_regressions"] = g["regressions"]
            out["acc_base_pct"] = _ratio(g["base_success"], n)
            out["acc_model_pct"] = _ratio(g["model_success"], n)
            for lab in refs:
                out[f"acc_{lab}_pct"] = _ratio(g[f"success::{lab}"], n)
            out["correction_rate_pct"] = _ratio(g["corrections"], n)
            out["regression_rate_pct"] = _ratio(g["regressions"], n)
            out["net_delta_pp"] = out["correction_rate_pct"] - out["regression_rate_pct"]
            out["correction_given_base_error_pct"] = _ratio(g["corrections"], g["base_error"])
            out["regression_given_base_success_pct"] = _ratio(g["regressions"], g["base_success"])
            return out, [c for c in out.columns if c.endswith("_pct") or c.endswith("_pp")]

        res = _seed_rows(matrix, seeds, counts_for_seed, rate_fn, ok, categories=categories,
                         extra_scopes=extra_scopes, components=components, macro=True, uncertainty=uncertainty)
        res.insert(0, "model_label", model)
        res.insert(0, "base_label", base)
        frames.append(_collapse_validity(res))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------------------------- joint 8-cell table
def joint_outcomes(matrix: OutcomeMatrix, *, labels: Sequence[str] = ("B", "T", "V@P0"),
                   seeds: Sequence[int] | None = None, categories: Mapping[str, Sequence[str]] | None = None,
                   extra_scopes: Mapping[str, pd.Series] | None = None, missing: str = "raise",
                   components: pd.Series | None = None, uncertainty: BootstrapSpec | None = None) -> pd.DataFrame:
    """All 2^k outcome cells for k labels (k=3 gives the eight (B, T, V) cells), long format, one row per cell.

    Cells are listed in binary order with the first label as the most significant bit, so for (B, T, V) the T-only
    successes split into B-wrong (010) and B-already-right (110) cells, which the parent-overlap table also reports.
    """

    labels = list(labels)
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise BehaviorError("joint_outcomes needs at least two distinct labels")
    seeds = _resolve_seeds(matrix, labels, seeds)
    ok = _complete_units(matrix, [(lab, s) for lab in labels for s in seeds], missing)
    codes = list(product((0, 1), repeat=len(labels)))
    names = ["".join(map(str, c)) for c in codes]

    def counts_for_seed(s: int, mask: np.ndarray) -> pd.DataFrame:
        cols = [matrix.column(lab, s)[mask] for lab in labels]
        data = {"n_units": np.ones(int(mask.sum()))}
        for code, name in zip(codes, names):
            ind = np.ones(int(mask.sum()))
            for v, col in zip(code, cols):
                ind = ind * (col if v == 1 else 1 - col)
            data[f"cell::{name}"] = ind
        for lab, col in zip(labels, cols):
            data[f"success::{lab}"] = col
        return pd.DataFrame(data)

    def rate_fn(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        n = g["n_units"]
        out = pd.DataFrame(index=g.index)
        out["n_units"] = n
        for lab in labels:
            out[f"acc_{lab}_pct"] = _ratio(g[f"success::{lab}"], n)
        for name in names:
            out[f"n::{name}"] = g[f"cell::{name}"]
            out[f"rate::{name}_pct"] = _ratio(g[f"cell::{name}"], n)
        return out, [c for c in out.columns if c.endswith("_pct")]

    wide = _seed_rows(matrix, seeds, counts_for_seed, rate_fn, ok, categories=categories, extra_scopes=extra_scopes,
                      components=components, macro=True, uncertainty=uncertainty)
    base_cols = [c for c in wide.columns if not c.startswith("n::") and not c.startswith("rate::")]
    frames = []
    for code, name in zip(codes, names):
        part = wide[base_cols].copy()
        part.insert(0, "cell", "|".join(f"{lab}={v}" for lab, v in zip(labels, code)))
        for i, (lab, v) in enumerate(zip(labels, code), start=1):
            part.insert(i, f"label_{i}", lab)
        for i, (lab, v) in enumerate(zip(labels, code), start=1):
            part[f"outcome_{i}"] = v
        part["n_cell"] = wide[f"n::{name}"]
        part["cell_rate_pct"] = wide[f"rate::{name}_pct"]
        for suffix in CI_SUFFIXES + (NVALID_SUFFIX,):
            if f"rate::{name}_pct{suffix}" in wide.columns:
                part[f"cell_rate_pct{suffix}"] = wide[f"rate::{name}_pct{suffix}"]
        part["cell_code"] = name
        frames.append(_collapse_validity(part))
    out = pd.concat(frames, ignore_index=True)
    order = ["level", "scope", "aggregation", "seed_scope", "seed", "cell_code"]
    return out.sort_values(order, kind="stable", na_position="last").reset_index(drop=True)


# -------------------------------------------------------------------------------------------------- parent overlap
def parent_overlap(matrix: OutcomeMatrix, *, base: str = "B", trace: str = "T", vero: str = "V@P0",
                   seeds: Sequence[int] | None = None, categories: Mapping[str, Sequence[str]] | None = None,
                   extra_scopes: Mapping[str, pd.Series] | None = None, missing: str = "raise",
                   components: pd.Series | None = None, uncertainty: BootstrapSpec | None = None) -> pd.DataFrame:
    """Direct overlap of the TRACE-side and VERO-side successes, split by whether the base was already right."""

    labels = [base, trace, vero]
    if len(set(labels)) != 3:
        raise BehaviorError("base, trace and vero labels must differ")
    seeds = _resolve_seeds(matrix, labels, seeds)
    ok = _complete_units(matrix, [(lab, s) for lab in labels for s in seeds], missing)

    def counts_for_seed(s: int, mask: np.ndarray) -> pd.DataFrame:
        b = matrix.column(base, s)[mask]
        t = matrix.column(trace, s)[mask]
        v = matrix.column(vero, s)[mask]
        t_only, v_only = t * (1 - v), (1 - t) * v
        return pd.DataFrame({
            "n_units": np.ones_like(b), "base_success": b, "trace_success": t, "vero_success": v,
            "trace_only": t_only, "vero_only": v_only, "both_correct": t * v, "both_wrong": (1 - t) * (1 - v),
            "trace_only_base_correct": t_only * b, "trace_only_base_wrong": t_only * (1 - b),
            "vero_only_base_correct": v_only * b, "vero_only_base_wrong": v_only * (1 - b),
        })

    def rate_fn(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        n = g["n_units"]
        out = pd.DataFrame(index=g.index)
        out["n_units"] = n
        out["acc_base_pct"] = _ratio(g["base_success"], n)
        out["acc_trace_pct"] = _ratio(g["trace_success"], n)
        out["acc_vero_pct"] = _ratio(g["vero_success"], n)
        for name in ("trace_only", "vero_only", "both_correct", "both_wrong", "trace_only_base_correct",
                     "trace_only_base_wrong", "vero_only_base_correct", "vero_only_base_wrong"):
            out[f"n_{name}"] = g[name]
            out[f"{name}_pct"] = _ratio(g[name], n)
        out["agreement_pct"] = _ratio(g["both_correct"] + g["both_wrong"], n)
        out["hindsight_either_parent_correct_pct"] = _ratio(n - g["both_wrong"], n)
        return out, [c for c in out.columns if c.endswith("_pct")]

    res = _seed_rows(matrix, seeds, counts_for_seed, rate_fn, ok, categories=categories, extra_scopes=extra_scopes,
                     components=components, macro=True, uncertainty=uncertainty)
    res.insert(0, "vero_label", vero)
    res.insert(0, "trace_label", trace)
    res.insert(0, "base_label", base)
    res["hindsight_note"] = HINDSIGHT_NOTE
    return _collapse_validity(res)


# ----------------------------------------------------------------------------------------------------- persistence
def default_persistence_patterns(base: str = "B", trace: str = "T", vero: str = "V@P0") -> dict[str, dict[str, int]]:
    """T-only/V-only parent sets plus each parent's corrections and regressions against the base."""

    return {
        "trace_only": {trace: 1, vero: 0},
        "vero_only": {trace: 0, vero: 1},
        "trace_correction": {base: 0, trace: 1},
        "vero_correction": {base: 0, vero: 1},
        "trace_regression": {base: 1, trace: 0},
        "vero_regression": {base: 1, vero: 0},
    }


def persistence(matrix: OutcomeMatrix, *, patterns: Mapping[str, Mapping[str, int]] | None = None,
                seeds: Sequence[int] | None = None, categories: Mapping[str, Sequence[str]] | None = None,
                extra_scopes: Mapping[str, pd.Series] | None = None, missing: str = "raise",
                components: pd.Series | None = None, uncertainty: BootstrapSpec | None = None) -> pd.DataFrame:
    """Cross-repeat persistence: define a set on one seed, measure whether its pattern recurs on another seed.

    For every ordered seed pair (d, e) with d != e: ``n_defined`` units match the pattern at d, ``n_persisting`` of
    them match it again at e. Rows pooled over all ordered pairs use mean-per-pair counts and carry the bootstrap
    intervals. These rows are separate from same-seed descriptive tables by construction.

    The reference is ``null_persistence_pct``, not the base rate. Even with no complementarity at all (two identical
    models), units differ in difficulty, so a set selected on one draw recurs above the marginal frequency
    (E[q^2]/E[q] with q the per-unit pattern probability). For a two-label pattern with opposite values (T-only,
    corrections, ...), the label-swap null (``SWAP_NULL_NOTE``) estimates that persistence from the data: with
    a_s = 1 when the two labels disagree at seed s, null persistence = sum(a_d * a_e) / (2 * sum(a_d)) over units and
    ordered pairs. ``persistence_excess_pp`` = observed - null is the evidence of a stable directional difference.
    ``eval_base_rate_pct`` remains as a descriptive frequency. Every row carries the accuracy of every label in any
    pattern (for the defaults: base, TRACE and VERO) on the definition and evaluation seeds.
    """

    patterns = dict(patterns) if patterns is not None else default_persistence_patterns()
    if not patterns:
        raise BehaviorError("no persistence patterns")
    labels = sorted({lab for p in patterns.values() for lab in p})
    for name, p in patterns.items():
        if not p or any(int(v) not in (0, 1) for v in p.values()):
            raise BehaviorError(f"pattern {name!r} must map labels to 0/1")
    seeds = _resolve_seeds(matrix, labels, seeds)
    if len(seeds) < 2:
        raise BehaviorError("persistence needs at least two seeds")
    ok = _complete_units(matrix, [(lab, s) for lab in labels for s in seeds], missing)
    scopes = _scopes(matrix.values.index, categories, extra_scopes)
    bench = matrix.values.index.get_level_values("benchmark_id").to_numpy(dtype=object)[ok]
    comps = _components_array(matrix, components)
    pairs = [(d, e) for d in seeds for e in seeds if d != e]
    frames = []
    for name, pattern in patterns.items():
        plabels = list(pattern)
        swap = len(plabels) == 2 and int(pattern[plabels[0]]) != int(pattern[plabels[1]])

        def holds(seed: int, lab: str) -> np.ndarray:
            col = matrix.column(lab, seed)[ok]
            return col if int(pattern[lab]) == 1 else 1 - col

        def discord(seed: int) -> np.ndarray:
            return (matrix.column(plabels[0], seed)[ok] != matrix.column(plabels[1], seed)[ok]).astype(float)

        def rate_fn(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
            out = pd.DataFrame(index=g.index)
            out["n_units"] = g["n_units"]
            out["n_defined"] = g["defined"]
            out["n_persisting"] = g["persisting"]
            out["n_pattern_eval"] = g["pattern_eval"]
            out["defined_rate_pct"] = _ratio(g["defined"], g["n_units"])
            out["persistence_pct"] = _ratio(g["persisting"], g["defined"])
            out["null_persistence_pct"] = _ratio(g["null_persisting"], g["null_defined"])
            out["persistence_excess_pp"] = out["persistence_pct"] - out["null_persistence_pct"]
            out["eval_base_rate_pct"] = _ratio(g["pattern_eval"], g["n_units"])
            for lab in plabels:
                out[f"{lab}_holds_pct"] = _ratio(g[f"holds::{lab}"], g["defined"])
            for lab in labels:
                out[f"acc_{lab}_def_pct"] = _ratio(g[f"success_def::{lab}"], g["n_units"])
                out[f"acc_{lab}_eval_pct"] = _ratio(g[f"success_eval::{lab}"], g["n_units"])
            return out, [c for c in out.columns if c.endswith("_pct") or c.endswith("_pp")]

        def counts_for_pair(d: int, e: int) -> pd.DataFrame:
            defined = np.prod([holds(d, lab) for lab in plabels], axis=0)
            at_e = {lab: holds(e, lab) for lab in plabels}
            pattern_eval = np.prod(list(at_e.values()), axis=0)
            data = {"n_units": np.ones_like(defined), "defined": defined, "persisting": defined * pattern_eval,
                    "pattern_eval": pattern_eval}
            if swap:
                a_d, a_e = discord(d), discord(e)
                data["null_defined"], data["null_persisting"] = a_d / 2.0, a_d * a_e / 4.0
            else:
                data["null_defined"] = data["null_persisting"] = np.zeros_like(defined)
            for lab in plabels:
                data[f"holds::{lab}"] = defined * at_e[lab]
            for lab in labels:
                data[f"success_def::{lab}"] = matrix.column(lab, d)[ok]
                data[f"success_eval::{lab}"] = matrix.column(lab, e)[ok]
            return pd.DataFrame(data)

        total = None
        for d, e in pairs:
            counts = counts_for_pair(d, e)
            total = counts if total is None else total + counts
            out = _summarize(counts, bench, scopes, rate_fn, macro=False, components=comps, dropped=~ok)
            out.insert(2, "def_seed", d)
            out.insert(3, "eval_seed", e)
            out.insert(4, "seed_scope", SINGLE_SEED)
            out.insert(5, "n_seed_pairs", 1)
            frames.append(out.assign(set_name=name, pattern=json.dumps(pattern, sort_keys=True)))
        out = _summarize(total / len(pairs), bench, scopes, rate_fn, macro=False, components=comps, dropped=~ok,
                         uncertainty=uncertainty)
        out.insert(2, "def_seed", pd.NA)
        out.insert(3, "eval_seed", pd.NA)
        out.insert(4, "seed_scope", POOLED_PAIRS)
        out.insert(5, "n_seed_pairs", len(pairs))
        frames.append(out.assign(set_name=name, pattern=json.dumps(pattern, sort_keys=True)))
        for f in frames[-len(pairs) - 1:]:
            f["null_reference"] = SWAP_NULL_NOTE if swap else NO_NULL_NOTE
    res = pd.concat(frames, ignore_index=True)
    res["def_seed"] = res["def_seed"].astype("Int64")
    res["eval_seed"] = res["eval_seed"].astype("Int64")
    front = ["set_name", "pattern"]
    return _collapse_validity(res[front + [c for c in res.columns if c not in front]])


# --------------------------------------------------------------------------------------------- merge capture/damage
def capture_damage(matrix: OutcomeMatrix, *, merge: str, trace: str = "T", vero: str = "V@P1", base: str = "B",
                   merge_seeds: Sequence[int] | None = None, parent_seeds: Sequence[int] | None = None,
                   variants: Sequence[str] = CAPTURE_VARIANTS, categories: Mapping[str, Sequence[str]] | None = None,
                   extra_scopes: Mapping[str, pd.Series] | None = None, missing: str = "raise",
                   components: pd.Series | None = None, uncertainty: BootstrapSpec | None = None) -> pd.DataFrame:
    """Which parental successes a merged model keeps, and what it loses or rescues (handoff 8.2).

    Parent groups at a definition seed d: trace_only (T=1, V=0), vero_only (T=0, V=1), shared_success (T=1, V=1),
    joint_failure (T=0, V=0). With the merge outcome at seed e:

    * trace_only_capture = P(M=1 | trace_only), vero_only_capture = P(M=1 | vero_only),
    * shared_success_damage = P(M=0 | shared_success), joint_failure_rescue = P(M=1 | joint_failure).

    Variants: ``contemporaneous`` (d = e, groups chosen on the same draw that the merge is compared against, so a
    parent's lucky failure inflates the group); ``independent_repeat`` (d != e, one row per ordered pair); and
    ``stable_other_seeds`` (the pattern must hold on every parent seed other than e; needs at least two such seeds,
    units without a stable pattern form ``unstable_parent_pattern``). ``contribution_pp`` is the event's share of the
    whole population in percentage points; ``merge_correct_contribution_pp`` sums over groups to the merge accuracy.
    ``acc_merge_pct`` is measured on the merge seed; ``acc_base_pct``, ``acc_trace_pct`` and ``acc_vero_pct`` on the
    definition seed(s) (their mean for the stable variant), so every row shows the base, TRACE and VERO absolute
    level (handoff 5.1). Parents are parameters so the same table serves ('T', 'V@P1') and ('T', 'V@P0').
    Bootstrap intervals go on each variant's pooled rows, or on its single condition when it has only one.
    """

    unknown = sorted(set(variants) - set(CAPTURE_VARIANTS))
    if unknown:
        raise BehaviorError(f"unknown capture variants {unknown}")
    if len({merge, trace, vero, base}) != 4:
        raise BehaviorError("merge, trace, vero and base labels must differ")
    merge_seeds = _resolve_seeds(matrix, [merge], merge_seeds)
    parent_seeds = _resolve_seeds(matrix, [trace, vero, base], parent_seeds)
    conditions: list[tuple[str, tuple[int, ...], int]] = []
    for e in merge_seeds:
        others = [d for d in parent_seeds if d != e]
        if "contemporaneous" in variants and e in parent_seeds:
            conditions.append(("contemporaneous", (e,), e))
        if "independent_repeat" in variants:
            conditions.extend(("independent_repeat", (d,), e) for d in others)
        if "stable_other_seeds" in variants and len(others) >= 2:
            conditions.append(("stable_other_seeds", tuple(others), e))
    if not conditions:
        raise BehaviorError(f"no capture condition is computable for merge seeds {merge_seeds} and parent seeds "
                            f"{parent_seeds} with variants {list(variants)}")
    used_parent = sorted({d for _, ds, _ in conditions for d in ds})
    used_merge = sorted({e for _, _, e in conditions})
    cells = [(merge, e) for e in used_merge] + [(lab, d) for lab in (trace, vero, base) for d in used_parent]
    ok = _complete_units(matrix, cells, missing)
    scopes = _scopes(matrix.values.index, categories, extra_scopes)
    bench = matrix.values.index.get_level_values("benchmark_id").to_numpy(dtype=object)[ok]
    comps = _components_array(matrix, components)
    group_names = [g[0] for g in CAPTURE_GROUPS]
    n_conditions = Counter(variant for variant, _, _ in conditions)

    def counts_for(def_seeds: tuple[int, ...], e: int, stable: bool) -> pd.DataFrame:
        m = matrix.column(merge, e)[ok]
        ts = [matrix.column(trace, d)[ok] for d in def_seeds]
        vs = [matrix.column(vero, d)[ok] for d in def_seeds]
        bs = [matrix.column(base, d)[ok] for d in def_seeds]
        data = {"n_units": np.ones_like(m), "merge_success": m, "base_success": np.mean(bs, axis=0),
                "trace_success": np.mean(ts, axis=0), "vero_success": np.mean(vs, axis=0)}
        assigned = np.zeros_like(m)
        for gname, tv, vv, _, event in CAPTURE_GROUPS:
            member = np.ones_like(m)
            for t, v in zip(ts, vs):
                member = member * (t if tv == 1 else 1 - t) * (v if vv == 1 else 1 - v)
            assigned = assigned + member
            data[f"in::{gname}"] = member
            data[f"event::{gname}"] = member * (m if event == 1 else 1 - m)
            data[f"mcorrect::{gname}"] = member * m
        if stable:
            un = 1 - assigned
            data[f"in::{UNSTABLE_GROUP}"] = un
            data[f"event::{UNSTABLE_GROUP}"] = un * m
            data[f"mcorrect::{UNSTABLE_GROUP}"] = un * m
        return pd.DataFrame(data)

    def make_rate_fn(groups: list[str]) -> RateFn:
        def rate_fn(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
            n = g["n_units"]
            out = pd.DataFrame(index=g.index)
            out["n_units"] = n
            out["acc_merge_pct"] = _ratio(g["merge_success"], n)
            out["acc_base_pct"] = _ratio(g["base_success"], n)
            out["acc_trace_pct"] = _ratio(g["trace_success"], n)
            out["acc_vero_pct"] = _ratio(g["vero_success"], n)
            for gname in groups:
                out[f"{gname}::n_group"] = g[f"in::{gname}"]
                out[f"{gname}::group_share_pct"] = _ratio(g[f"in::{gname}"], n)
                out[f"{gname}::n_event"] = g[f"event::{gname}"]
                out[f"{gname}::conditional_rate_pct"] = _ratio(g[f"event::{gname}"], g[f"in::{gname}"])
                out[f"{gname}::contribution_pp"] = _ratio(g[f"event::{gname}"], n)
                out[f"{gname}::merge_correct_contribution_pp"] = _ratio(g[f"mcorrect::{gname}"], n)
            return out, [c for c in out.columns if c.endswith("_pct") or c.endswith("_pp")]
        return rate_fn

    def to_long(wide: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        base_cols = [c for c in wide.columns if "::" not in c]
        meta = {g[0]: (g[3], "merge_correct" if g[4] == 1 else "merge_wrong") for g in CAPTURE_GROUPS}
        meta[UNSTABLE_GROUP] = ("unstable_group_merge_accuracy", "merge_correct")
        parts = []
        for gname in groups:
            part = wide[base_cols].copy()
            part["group"] = gname
            part["metric"] = meta[gname][0]
            part["event"] = meta[gname][1]
            for field in ("n_group", "group_share_pct", "n_event", "conditional_rate_pct", "contribution_pp",
                          "merge_correct_contribution_pp"):
                part[field] = wide[f"{gname}::{field}"]
                for suffix in CI_SUFFIXES + (NVALID_SUFFIX,):
                    if f"{gname}::{field}{suffix}" in wide.columns:
                        part[f"{field}{suffix}"] = wide[f"{gname}::{field}{suffix}"]
            parts.append(_collapse_validity(part))
        return pd.concat(parts, ignore_index=True)

    frames = []
    totals: dict[str, pd.DataFrame] = {}
    for variant, def_seeds, e in conditions:
        stable = variant == "stable_other_seeds"
        groups = group_names + ([UNSTABLE_GROUP] if stable else [])
        counts = counts_for(def_seeds, e, stable)
        totals[variant] = counts if variant not in totals else totals[variant] + counts
        wide = _summarize(counts, bench, scopes, make_rate_fn(groups), macro=False, components=comps, dropped=~ok,
                          uncertainty=uncertainty if n_conditions[variant] == 1 else None)
        long = to_long(wide, groups)
        long.insert(0, "variant", variant)
        long.insert(1, "parent_seeds", ",".join(map(str, def_seeds)))
        long.insert(2, "merge_seed", e)
        long.insert(3, "seed_scope", SINGLE_CONDITION)
        long.insert(4, "n_conditions", 1)
        frames.append(long)
    for variant, total in totals.items():
        if n_conditions[variant] < 2:
            continue
        stable = variant == "stable_other_seeds"
        groups = group_names + ([UNSTABLE_GROUP] if stable else [])
        wide = _summarize(total / n_conditions[variant], bench, scopes, make_rate_fn(groups), macro=False,
                          components=comps, dropped=~ok, uncertainty=uncertainty)
        long = to_long(wide, groups)
        long.insert(0, "variant", variant)
        long.insert(1, "parent_seeds", "pooled")
        long.insert(2, "merge_seed", pd.NA)
        long.insert(3, "seed_scope", POOLED_CONDITIONS)
        long.insert(4, "n_conditions", n_conditions[variant])
        frames.append(long)
    res = pd.concat(frames, ignore_index=True)
    res["merge_seed"] = res["merge_seed"].astype("Int64")
    res.insert(0, "base_label", base)
    res.insert(0, "vero_label", vero)
    res.insert(0, "trace_label", trace)
    res.insert(0, "merge_label", merge)
    return res


# --------------------------------------------------------------------------------------------------------- bundles
def behavior_tables(matrix: OutcomeMatrix, *, categories: Mapping[str, Sequence[str]] | None, base: str = "B",
                    trace: str = "T", vero: str = "V@P0", seeds: Sequence[int] | None = None,
                    missing: str = "raise", components: pd.Series | None = None,
                    extra_scopes: Mapping[str, pd.Series] | None = None,
                    uncertainty: BootstrapSpec | None = None) -> dict[str, pd.DataFrame]:
    """The Stage B parent tables (handoff 5.5) for one (base, trace, vero) triple, plus their coverage record."""

    common = dict(seeds=seeds, categories=categories, extra_scopes=extra_scopes, missing=missing,
                  components=components, uncertainty=uncertainty)
    return {
        "corrections_regressions": corrections_regressions(matrix, base=base, models=(trace, vero), **common),
        "joint_outcomes": joint_outcomes(matrix, labels=(base, trace, vero), **common),
        "parent_overlap": parent_overlap(matrix, base=base, trace=trace, vero=vero, **common),
        "persistence": persistence(matrix, patterns=default_persistence_patterns(base, trace, vero), **common),
        "behavior_coverage": coverage_table(matrix),
    }


def write_tables(tables: Mapping[str, pd.DataFrame], out_dir: Path) -> dict[str, str]:
    """Write each table as ``<name>.csv`` atomically; returns {file name: sha256} for the caller's manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, frame in tables.items():
        buf = io.StringIO()
        frame.to_csv(buf, index=False, float_format="%.10g")
        data = buf.getvalue().encode("utf-8")
        path = out_dir / f"{name}.csv"
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        digests[path.name] = sha256_bytes(data)
    return digests
