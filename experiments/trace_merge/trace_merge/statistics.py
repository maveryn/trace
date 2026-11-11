"""Equal-weight macros and the paired cluster bootstrap over connected components (handoff 6.2; DESIGN.md section 7).

The suite's headline number is the unweighted mean of the official benchmark percentages, not pooled row accuracy, so
every macro here averages finished benchmark scores. Uncertainty comes from resampling the independent units that
grouping.py freezes (connected image/problem components) with replacement:

* one global draw per replicate: a component that spans two benchmarks keeps the same multiplicity in both, so their
  coupling survives instead of being duplicated independently;
* the same draw is applied to every model and every decoding seed, so comparisons are paired and seeds are repeated
  measurements of the same content, not extra content;
* each benchmark's official score is recomputed per replicate from additive per-component statistics
  (``BenchmarkMetric.unit_stats``) with the metric's own ``finalize``, so grouped and stratified official metrics are
  resampled as metrics, not as row means.

Intervals are conditional on these checkpoints and this suite; they do not cover training-run variability or new
benchmark populations. The replicate stream depends only on (seed, n_replicates): replicates are drawn in fixed blocks
of ``RNG_BLOCK`` from child streams of ``numpy.random.SeedSequence(seed)``, so memory batching never changes results.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

RNG_BLOCK = 250
SUITE_BENCHMARKS = 24
PRIMARY_PROTOCOL_SUFFIX = "@P1"
MAX_KEPT_WEIGHTS = 50_000_000
P_VALUE_METHOD = ("two-sided percentile-inversion bootstrap p-value: p = min(1, 2 * (min(#{d* <= 0}, #{d* >= 0}) + 1) "
                  "/ (R + 1)) over the R finite replicate differences d*; p < alpha exactly when 0 lies outside the "
                  "equal-tailed (1 - alpha) percentile interval, up to the add-one Monte Carlo correction that keeps "
                  "p >= 2 / (R + 1)")
CI_METHOD = "equal-tailed percentile interval of the replicate distribution (numpy linear interpolation)"


class StatisticsError(ValueError):
    pass


class MetricLike(Protocol):
    """The part of metrics.BenchmarkMetric this module relies on."""

    benchmark_id: str
    kind: str
    official_metric_id: str

    def row_scores(self, rows: pd.DataFrame, meta: pd.DataFrame | None) -> pd.Series: ...

    def unit_stats(self, rows: pd.DataFrame, scores: pd.Series, unit: pd.Series) -> tuple[np.ndarray, np.ndarray]: ...

    def finalize(self, totals: np.ndarray) -> float: ...


# ----------------------------------------------------------------------------------------------------------- macros
def macro_name(n_benchmarks: int, n_suite: int = SUITE_BENCHMARKS) -> str:
    """Name a macro by its benchmark count so a subset mean is never presented as the suite's 24-benchmark metric."""

    return f"{n_suite}-benchmark macro" if n_benchmarks == n_suite else f"{n_benchmarks}-benchmark subset macro"


def benchmark_macro(scores: Mapping[str, float], benchmarks: Sequence[str] | None = None) -> float:
    """Equal-weight mean of official benchmark scores (percent); fails on a missing or non-finite benchmark."""

    keys = list(benchmarks) if benchmarks is not None else list(scores)
    if not keys:
        raise StatisticsError("no benchmarks to average")
    bad = [k for k in keys if k not in scores or not np.isfinite(float(scores[k]))]
    if bad:
        raise StatisticsError(f"missing or non-finite benchmark scores: {bad}")
    return float(np.mean([float(scores[k]) for k in keys]))


def category_scope_name(category: str, n_present: int, n_members: int) -> str:
    """Name a category scope by its coverage, so a one-benchmark remnant is never read as the whole category.

    A complete category keeps its suite name; a partial one becomes e.g. ``"Visual Math (partial: 1 of 4
    benchmarks)"``, the category counterpart of ``macro_name``'s subset renaming.
    """

    if n_present == n_members:
        return category
    return f"{category} (partial: {n_present} of {n_members} benchmarks)"


def _category_members(categories: Mapping[str, Sequence[str]], benchmarks: Sequence[str],
                      allow_partial: bool) -> dict[str, list[str]]:
    """{scope name: present member benchmarks}; partial categories are renamed by ``category_scope_name``."""

    present = set(benchmarks)
    seen: dict[str, str] = {}
    out: dict[str, list[str]] = {}
    for cat, members in categories.items():
        for b in members:
            if b in seen:
                raise StatisticsError(f"benchmark {b!r} is in categories {seen[b]!r} and {cat!r}")
            seen[b] = cat
        have = [b for b in members if b in present]
        if len(have) != len(members) and not allow_partial:
            raise StatisticsError(f"category {cat!r} lacks {sorted(set(members) - present)}; pass allow_partial=True "
                                  "to report it as a partial category macro")
        if have:
            out[category_scope_name(cat, len(have), len(members))] = have
    stray = sorted(present - set(seen))
    if stray:
        raise StatisticsError(f"benchmarks outside every category: {stray}")
    return out


def category_macros(scores: Mapping[str, float], categories: Mapping[str, Sequence[str]], *,
                    allow_partial: bool = False) -> dict[str, float]:
    """Equal-weight mean of the benchmarks of each category that are present in ``scores``.

    Keys are scope names: a partial category is renamed (see ``category_scope_name``).
    """

    members = _category_members(categories, list(scores), allow_partial)
    return {cat: benchmark_macro(scores, have) for cat, have in members.items()}


def macro_table(scores: pd.DataFrame, categories: Mapping[str, Sequence[str]] | None = None, *,
                value_col: str = "official_value", benchmarks: Sequence[str] | None = None,
                seeds: Sequence[int] | None = None, allow_partial_categories: bool = False,
                n_suite: int = SUITE_BENCHMARKS) -> pd.DataFrame:
    """Overall and category macros per model and seed, plus seed means, from per-benchmark official scores.

    Every (model_label, seed) must carry exactly the same benchmark set; the seed mean is the mean of per-seed macros
    (the release's order of operations, identical to the macro of seed-mean benchmark scores). Seed means of labels
    evaluated on different seed sets are not comparable (a one-seed value carries more decoding noise), so all labels
    must share one seed set; pass ``seeds`` to restrict the table to a common set (every label must have all of them).
    """

    need = {"model_label", "seed", "benchmark_id", value_col}
    if need - set(scores.columns):
        raise StatisticsError(f"scores lack columns {sorted(need - set(scores.columns))}")
    df = scores[list(need)].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    if df.duplicated(["model_label", "seed", "benchmark_id"]).any():
        raise StatisticsError("duplicate model_label x seed x benchmark scores")
    if seeds is not None:
        want = sorted({int(s) for s in seeds})
        df = df[df["seed"].isin(want)]
        lacking = {lab: sorted(set(want) - set(part["seed"])) for lab, part in df.groupby("model_label")}
        lacking = {lab: s for lab, s in lacking.items() if s}
        if lacking:
            raise StatisticsError(f"labels lack requested seeds: {lacking}")
    seed_sets = {lab: tuple(sorted(set(part["seed"]))) for lab, part in df.groupby("model_label")}
    if len(set(seed_sets.values())) > 1:
        raise StatisticsError(f"labels have different seed sets {seed_sets}; their seed means are not comparable. "
                              "Pass seeds=[...] for a common set or build one table per seed set")
    bset = list(benchmarks) if benchmarks is not None else sorted(df["benchmark_id"].unique())
    df = df[df["benchmark_id"].isin(bset)]
    members = _category_members(categories, bset, allow_partial_categories) if categories is not None else {}
    rows: list[dict[str, Any]] = []
    for label, part in df.groupby("model_label", sort=True):
        per_seed: dict[int, dict[str, float]] = {}
        for seed, sp in part.groupby("seed", sort=True):
            vals = dict(zip(sp["benchmark_id"], sp[value_col].astype(float)))
            if set(vals) != set(bset):
                raise StatisticsError(f"{label} seed {seed} lacks benchmarks {sorted(set(bset) - set(vals))}")
            per_seed[int(seed)] = vals
        scoped: dict[tuple[str, str], list[float]] = {}
        for seed, vals in per_seed.items():
            entries = [("overall", macro_name(len(bset), n_suite), benchmark_macro(vals, bset), len(bset))]
            entries += [("category", cat, benchmark_macro(vals, have), len(have)) for cat, have in members.items()]
            for level, scope, value, count in entries:
                rows.append({"model_label": label, "seed": seed, "seed_scope": "single_seed", "n_seeds": 1,
                             "level": level, "scope": scope, "value": value, "benchmark_count": count})
                scoped.setdefault((level, scope), []).append(value)
        for (level, scope), values in scoped.items():
            count = len(bset) if level == "overall" else len(members[scope])
            rows.append({"model_label": label, "seed": pd.NA, "seed_scope": "mean_over_seeds",
                         "n_seeds": len(values), "level": level, "scope": scope, "value": float(np.mean(values)),
                         "benchmark_count": count})
    out = pd.DataFrame(rows)
    out["seed"] = out["seed"].astype("Int64")
    return out


def macro_vs_pooled(rows: pd.DataFrame, value_col: str = "binary_correct") -> dict[str, Any]:
    """Equal-weight benchmark macro and pooled row accuracy of the same 0/1 rows, both in percent."""

    if rows.empty:
        raise StatisticsError("no rows")
    per = rows.groupby("benchmark_id")[value_col].agg(["mean", "size"])
    return {"macro_pct": float(100 * per["mean"].mean()), "pooled_pct": float(100 * rows[value_col].mean()),
            "per_benchmark_pct": {k: float(100 * v) for k, v in per["mean"].items()},
            "rows_per_benchmark": {k: int(v) for k, v in per["size"].items()}}


def demonstrate_macro_vs_pooled() -> dict[str, Any]:
    """Why the suite metric is not pooled accuracy: 900 rows at 90% and 100 rows at 10% give 82% pooled, 50% macro."""

    big = pd.DataFrame({"benchmark_id": "large", "binary_correct": [1] * 810 + [0] * 90})
    small = pd.DataFrame({"benchmark_id": "small", "binary_correct": [1] * 10 + [0] * 90})
    return macro_vs_pooled(pd.concat([big, small], ignore_index=True))


# ------------------------------------------------------------------------------------------------- multiplicity etc.
def holm_adjust(pvalues: Mapping[str, float] | Sequence[float]) -> dict[str, float] | list[float]:
    """Holm step-down adjustment for one prespecified family (ties keep input order; adjusted values are monotone)."""

    if isinstance(pvalues, Mapping):
        keys = list(pvalues)
        vals = np.array([float(pvalues[k]) for k in keys])
    else:
        keys = None
        vals = np.array([float(p) for p in pvalues])
    if vals.size == 0:
        return {} if keys is not None else []
    if not np.all(np.isfinite(vals)) or np.any(vals < 0) or np.any(vals > 1):
        raise StatisticsError(f"p-values must be finite and in [0, 1]: {vals.tolist()}")
    m = vals.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(vals, kind="stable")):
        running = max(running, min(1.0, (m - rank) * vals[idx]))
        adj[idx] = running
    if keys is not None:
        return {k: float(a) for k, a in zip(keys, adj)}
    return [float(a) for a in adj]


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b (A right, B wrong) and c (A wrong, B right).

    Valid only when the paired binary units are independent: one outcome per model per independent image/problem
    group from a single decoding draw. Rows that share an image or a problem family, rows pooled over decoding seeds,
    and grouped or stratified official metrics violate that; use ``paired_cluster_bootstrap`` for those.
    """

    from scipy.stats import binom

    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise StatisticsError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))


def bootstrap_p_value(replicates: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (see ``P_VALUE_METHOD``). NaN replicates are ignored."""

    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    le = int(np.count_nonzero(r <= 0))
    ge = int(np.count_nonzero(r >= 0))
    return float(min(1.0, 2.0 * (min(le, ge) + 1) / (r.size + 1)))


def percentile_interval(replicates: np.ndarray, confidence: float) -> tuple[float, float]:
    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    a = 1.0 - confidence
    lo, hi = np.percentile(r, [100 * a / 2, 100 * (1 - a / 2)])
    return float(lo), float(hi)


def parse_comparison(spec: str | Sequence[str]) -> tuple[str, str]:
    """'A-B' or (A, B) -> (A, B). Model labels never contain '-', so a split on it is unambiguous."""

    if isinstance(spec, str):
        parts = spec.split("-")
        if len(parts) != 2 or not all(parts):
            raise StatisticsError(f"comparison {spec!r} is not of the form 'A-B'")
        return parts[0], parts[1]
    a, b = spec
    return str(a), str(b)


def resolve_label(name: str, labels: Sequence[str], primary_suffix: str = PRIMARY_PROTOCOL_SUFFIX) -> str:
    """Map a family name to an evaluated label: exact match, else the primary-protocol label (``M050`` -> ``M050@P1``)."""

    if name in labels:
        return name
    if f"{name}{primary_suffix}" in labels:
        return f"{name}{primary_suffix}"
    raise StatisticsError(f"comparison label {name!r} is not among the evaluated labels {list(labels)}")


def resolve_comparisons(specs: Sequence[str | Sequence[str]], labels: Sequence[str], *,
                        unique: bool = False) -> list[tuple[str, str]]:
    """Resolve comparison specs to label pairs, dropping repeats (first occurrence kept).

    With ``unique=True`` a repeated pair is an error instead: a multiplicity family that lists one hypothesis twice
    (e.g. ``M050-T`` and ``M050@P1-T``) would inflate the Holm family size.
    """

    out: list[tuple[str, str]] = []
    for spec in specs:
        a, b = parse_comparison(spec)
        pair = (resolve_label(a, labels), resolve_label(b, labels))
        if pair[0] == pair[1]:
            raise StatisticsError(f"comparison {spec!r} compares a label with itself")
        if pair in out:
            if unique:
                raise StatisticsError(f"comparison {spec!r} repeats {pair[0]}-{pair[1]} in a family")
            continue
        out.append(pair)
    return out


# --------------------------------------------------------------------------------------------- unit-stat collection
@dataclass(frozen=True)
class SliceStats:
    """Additive per-component statistics of one (model_label, seed, benchmark) slice.

    ``content_sha256`` fingerprints the slice's sorted content-key set. Identical component ids do not prove that
    two slices hold the same rows (a slice can lose one row of a multi-row component and keep the component), so
    the bootstrap compares these fingerprints when present.
    """

    unit_ids: np.ndarray
    stats: np.ndarray
    content_sha256: str | None = None


SliceKey = tuple[str, int, str]


def content_fingerprint(content_keys: Sequence[str]) -> str:
    """sha256 of the sorted, newline-joined content keys of one slice."""

    return hashlib.sha256("\n".join(sorted(str(k) for k in content_keys)).encode("utf-8")).hexdigest()


def component_map(assignments: pd.DataFrame) -> pd.Series:
    """content_key -> component_id from splits/group_assignments.parquet, failing on conflicting assignments."""

    need = {"content_key", "component_id"}
    if need - set(assignments.columns):
        raise StatisticsError(f"group assignments lack {sorted(need - set(assignments.columns))}")
    pairs = assignments[["content_key", "component_id"]].drop_duplicates()
    if pairs["content_key"].duplicated().any():
        raise StatisticsError("a content_key has more than one component_id")
    return pd.Series(pairs["component_id"].astype(str).to_numpy(), index=pairs["content_key"].to_numpy())


def collect_unit_stats(rows: pd.DataFrame, metrics: Mapping[str, MetricLike], assignments: pd.DataFrame, *,
                       labels: Sequence[str], seeds: Sequence[int], benchmarks: Sequence[str],
                       meta: Mapping[str, pd.DataFrame] | None = None) -> dict[SliceKey, SliceStats]:
    """Call each benchmark's metric on every (label, seed, benchmark) slice of the aligned table.

    Pairing is checked on content rows, not only on components: every (label, seed) slice of a benchmark must hold
    exactly the same content-key set, because a slice that lost one row of a multi-row component keeps the same
    component ids while its benchmark score silently changes denominator. Pass only the columns the metrics need;
    response texts are not required for row-level archived scores.
    """

    need = {"model_label", "seed", "benchmark_id", "content_key"}
    if need - set(rows.columns):
        raise StatisticsError(f"rows lack columns {sorted(need - set(rows.columns))}")
    comp = component_map(assignments)
    seeds_int = [int(s) for s in seeds]
    df = rows[rows["model_label"].isin(list(labels)) & rows["benchmark_id"].isin(list(benchmarks))]
    df = df.assign(seed=pd.to_numeric(df["seed"], errors="raise").astype(int))
    df = df[df["seed"].isin(seeds_int)]
    groups = df.groupby(["model_label", "seed", "benchmark_id"], sort=False).indices
    out: dict[SliceKey, SliceStats] = {}
    reference: dict[str, tuple[SliceKey, set[str]]] = {}
    for label in labels:
        for seed in seeds_int:
            for bench in benchmarks:
                key = (label, seed, bench)
                if key not in groups:
                    raise StatisticsError(f"no rows for {key}")
                sub = df.iloc[groups[key]]
                if sub["content_key"].duplicated().any():
                    raise StatisticsError(f"duplicate content rows in {key}")
                keys = set(sub["content_key"].astype(str))
                if bench not in reference:
                    reference[bench] = (key, keys)
                elif keys != reference[bench][1]:
                    ref_key, ref_keys = reference[bench]
                    only_here, only_ref = sorted(keys - ref_keys), sorted(ref_keys - keys)
                    raise StatisticsError(
                        f"{bench}: {key} and {ref_key} hold different content rows ({len(only_here)} only in the "
                        f"former, e.g. {only_here[:3]}; {len(only_ref)} only in the latter, e.g. {only_ref[:3]}); "
                        "paired resampling needs every model and seed on the same rows")
                unit = sub["content_key"].map(comp)
                if unit.isna().any():
                    raise StatisticsError(f"{int(unit.isna().sum())} rows of {key} have no component id")
                metric = metrics[bench]
                if metric.kind == "unsupported":
                    raise StatisticsError(f"{bench} has no accepted metric adapter")
                scores = metric.row_scores(sub, (meta or {}).get(bench))
                if not scores.index.equals(sub.index):
                    raise StatisticsError(f"{bench}: row_scores is not aligned to its rows")
                ids, stats = metric.unit_stats(sub, scores, unit)
                out[key] = SliceStats(np.asarray(ids), np.asarray(stats, dtype=float), content_fingerprint(keys))
    return out


# --------------------------------------------------------------------------------------------------------- engine
# What a metric raises when its official value is undefined on a resampled row set (metrics.MetricError is a
# ValueError: "no rows", "split(s) absent", ...). On the full data the same call is strict, so a structural error
# (wrong statistic width, unbound metadata) surfaces at the point estimate and is never mistaken for a replicate gap.
UNDEFINED_ERRORS: tuple[type[BaseException], ...] = (ValueError, ArithmeticError)


def _finalize_rows(metric: MetricLike, totals: np.ndarray, errors: list[str] | None) -> np.ndarray:
    """Row-by-row ``finalize``; with ``errors`` given, an undefined row becomes NaN and its message is recorded."""

    out = np.empty(totals.shape[0])
    for i, row in enumerate(totals):
        if errors is None:
            out[i] = float(metric.finalize(row))
            continue
        try:
            out[i] = float(metric.finalize(row))
        except UNDEFINED_ERRORS as exc:
            out[i] = np.nan
            if len(errors) < 3:
                errors.append(f"{type(exc).__name__}: {exc}")
    return out


def _finalize_many(metric: MetricLike, totals: np.ndarray, cache: dict[int, str],
                   errors: list[str] | None = None) -> np.ndarray:
    """Finalize many total vectors at once.

    Uses ``finalize_many`` when the metric has it; otherwise probes once whether ``finalize`` already broadcasts over
    a leading axis (checked against row-by-row calls on k+3 rows, a count that a non-broadcasting implementation
    cannot satisfy by accident), and falls back to a Python loop, which is slower but always correct.

    ``errors=None`` is the strict mode used for point estimates: any failure propagates. With a list (replicate mode),
    a batch call that raises one of ``UNDEFINED_ERRORS`` is retried row by row so that only the undefined rows become
    NaN (metrics.py raises ``MetricError`` for a replicate that drew no rows or an empty official stratum, rather than
    returning NaN); the first messages are kept for the caller's error report.
    """

    n, k = totals.shape
    fn = getattr(metric, "finalize_many", None)
    if callable(fn):
        try:
            out = np.asarray(fn(totals), dtype=float)
        except UNDEFINED_ERRORS:
            if errors is None:
                raise
            return _finalize_rows(metric, totals, errors)
        if out.shape != (n,):
            raise StatisticsError(f"{metric.benchmark_id}: finalize_many returned shape {out.shape}, expected {(n,)}")
        return out
    mode = cache.get(id(metric))
    if mode is None:
        probe = totals[np.arange(k + 3) % n]
        expected = _finalize_rows(metric, probe, errors)
        try:
            got = np.asarray(metric.finalize(probe), dtype=float)
            same = got.shape == expected.shape and np.allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        except Exception:  # noqa: BLE001 - any failure just means "not vectorized"
            same = False
        mode = "vectorized" if same else "loop"
        cache[id(metric)] = mode
    if mode == "vectorized":
        try:
            out = np.asarray(metric.finalize(totals), dtype=float)
        except UNDEFINED_ERRORS:
            if errors is None:
                raise
            out = None
        if out is not None and out.shape == (n,):
            return out
    return _finalize_rows(metric, totals, errors)


@dataclass
class _BenchmarkBlock:
    positions: np.ndarray       # global component positions of this benchmark's units
    stacked: np.ndarray         # [n_units, L * S * k]
    k: int


def _prepare(unit_stats: Mapping[SliceKey, SliceStats | tuple], labels: Sequence[str], seeds: Sequence[int],
             benchmarks: Sequence[str]) -> tuple[np.ndarray, dict[str, _BenchmarkBlock]]:
    per_bench: dict[str, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]] = {}
    for bench in benchmarks:
        slices = {}
        ref_ids: np.ndarray | None = None
        k: int | None = None
        prints: dict[SliceKey, str | None] = {}
        for label in labels:
            for seed in seeds:
                key = (label, int(seed), bench)
                if key not in unit_stats:
                    raise StatisticsError(f"unit stats missing for {key}")
                item = unit_stats[key]
                if isinstance(item, SliceStats):
                    ids, stats = item.unit_ids, item.stats
                    prints[key] = item.content_sha256
                else:
                    ids, stats = item
                    prints[key] = None
                ids = np.asarray(ids).astype(str)
                stats = np.asarray(stats, dtype=float)
                if stats.ndim != 2 or stats.shape[0] != ids.size:
                    raise StatisticsError(f"{key}: stats must be [n_units, k] matching unit ids, got {stats.shape}")
                if not np.all(np.isfinite(stats)):
                    raise StatisticsError(f"{key}: non-finite unit stats")
                if np.unique(ids).size != ids.size:
                    raise StatisticsError(f"{key}: duplicate unit ids")
                order = np.argsort(ids, kind="stable")
                ids, stats = ids[order], stats[order]
                if ref_ids is None:
                    ref_ids, k = ids, stats.shape[1]
                elif not np.array_equal(ids, ref_ids):
                    raise StatisticsError(f"{bench}: {key} covers a different component set than the first slice; "
                                          "paired resampling needs every model and seed on the same units")
                elif stats.shape[1] != k:
                    raise StatisticsError(f"{bench}: inconsistent statistic width across slices")
                slices[(label, int(seed))] = (ids, stats)
        given = {p for p in prints.values() if p is not None}
        if given and (len(given) > 1 or None in prints.values()):
            detail = {f"{lab}/{s}": (p[:12] if p else None) for (lab, s, _), p in prints.items()}
            raise StatisticsError(f"{bench}: slices hold different content-row sets (fingerprints {detail}); paired "
                                  "resampling needs every model and seed on the same rows")
        per_bench[bench] = slices
    all_ids = np.unique(np.concatenate([next(iter(s.values()))[0] for s in per_bench.values()]))
    blocks: dict[str, _BenchmarkBlock] = {}
    for bench, slices in per_bench.items():
        ids = next(iter(slices.values()))[0]
        pos = np.searchsorted(all_ids, ids)
        stacked = np.concatenate([slices[(label, int(seed))][1] for label in labels for seed in seeds], axis=1)
        blocks[bench] = _BenchmarkBlock(positions=pos, stacked=stacked, k=next(iter(slices.values()))[1].shape[1])
    return all_ids, blocks


def _finalize_block(metric: MetricLike, totals: np.ndarray, n_l: int, n_s: int, k: int,
                    cache: dict[int, str], errors: list[str] | None = None) -> np.ndarray:
    r = totals.shape[0]
    flat = totals.reshape(r * n_l * n_s, k)
    return _finalize_many(metric, flat, cache, errors).reshape(r, n_l, n_s)


def iter_weight_blocks(n_components: int, n_replicates: int, seed: int):
    """Yield the bootstrap's component multiplicities block by block ([r, G] int arrays, r <= ``RNG_BLOCK``).

    The stream is the one ``paired_cluster_bootstrap`` uses, so other resampled diagnostics (behavior.py) can share
    the same replicate definition without materialising an [R, G] array.
    """

    for block_rng, r in _replicate_blocks(n_replicates, seed):
        yield _draw_block(block_rng, r, n_components)


def draw_component_weights(n_components: int, n_replicates: int, seed: int) -> np.ndarray:
    """Multiplicity of each component in each replicate ([R, G] int), exactly as the bootstrap draws them."""

    blocks = list(iter_weight_blocks(n_components, n_replicates, seed))
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, n_components), dtype=np.int64)


def _replicate_blocks(n_replicates: int, seed: int):
    n_blocks = math.ceil(n_replicates / RNG_BLOCK)
    children = np.random.SeedSequence(int(seed)).spawn(n_blocks)
    for i, child in enumerate(children):
        yield np.random.Generator(np.random.PCG64(child)), min(RNG_BLOCK, n_replicates - i * RNG_BLOCK)


def _draw_block(rng: np.random.Generator, r: int, g: int) -> np.ndarray:
    idx = rng.integers(0, g, size=(r, g))
    flat = (idx + (np.arange(r, dtype=np.int64) * g)[:, None]).ravel()
    return np.bincount(flat, minlength=r * g).reshape(r, g)


@dataclass
class BootstrapResult:
    labels: list[str]
    seeds: list[int]
    benchmarks: list[str]
    categories: dict[str, list[str]]
    n_components: int
    n_replicates: int
    seed: int
    confidence: float
    point_seed_scores: np.ndarray            # [L, S, B] official scores per seed
    point_benchmark_scores: np.ndarray       # [L, B] seed means
    replicate_benchmark_scores: np.ndarray   # [R, L, B] seed means per replicate
    scores: pd.DataFrame
    differences: pd.DataFrame
    replicate_seed_scores: np.ndarray | None = None   # [R, L, S, B] when requested
    weights: np.ndarray | None = None                 # [R, G] when requested
    component_ids: np.ndarray | None = None

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable description (no replicate arrays; NaN written as null)."""

        def clean(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = []
            for rec in records:
                out.append({k: (None if isinstance(v, float) and not math.isfinite(v) else
                                (int(v) if isinstance(v, np.integer) else
                                 (float(v) if isinstance(v, np.floating) else
                                  (bool(v) if isinstance(v, np.bool_) else v))))
                            for k, v in rec.items()})
            return out

        return {
            "labels": self.labels, "seeds": self.seeds, "benchmarks": self.benchmarks,
            "categories": self.categories, "n_components": self.n_components, "n_replicates": self.n_replicates,
            "seed": self.seed, "confidence": self.confidence, "rng_block": RNG_BLOCK,
            "resampling": "components drawn with replacement globally; same weights for every model and seed",
            "ci_method": CI_METHOD, "p_value_method": P_VALUE_METHOD,
            "scores": clean(self.scores.to_dict("records")),
            "differences": clean(self.differences.to_dict("records")),
        }


def _scope_reduce(arr: np.ndarray, benchmarks: Sequence[str], members: Mapping[str, list[str]],
                  n_suite: int) -> list[tuple[str, str, int, np.ndarray]]:
    """[..., B] benchmark scores -> (level, scope, benchmark_count, [...] values) for overall, categories, benchmarks."""

    idx = {b: i for i, b in enumerate(benchmarks)}
    out = [("overall", macro_name(len(benchmarks), n_suite), len(benchmarks), arr.mean(axis=-1))]
    for cat, have in members.items():
        out.append(("category", cat, len(have), arr[..., [idx[b] for b in have]].mean(axis=-1)))
    for b in benchmarks:
        out.append(("benchmark", b, 1, arr[..., idx[b]]))
    return out


def _check_invalid(score_rows: list[dict[str, Any]], diff_rows: list[dict[str, Any]], n_replicates: int,
                   max_invalid_fraction: float, errors: Mapping[str, list[str]]) -> None:
    """Fail when any reported score or difference rests on too few defined replicates.

    Checked per row, not per benchmark: a macro or a difference is undefined on the union of its members'
    undefined replicates, which can exceed the threshold even when every benchmark stays within it.
    """

    bad: dict[str, float] = {}
    for rec in score_rows + diff_rows:
        frac = 1.0 - rec["n_valid_replicates"] / n_replicates
        if frac > max_invalid_fraction:
            who = rec.get("comparison", rec.get("model_label"))
            name = f"{who} {rec['level']}:{rec['scope']}"
            bad[name] = max(bad.get(name, 0.0), round(frac, 6))
    if bad:
        shown = dict(sorted(bad.items(), key=lambda kv: -kv[1])[:8])
        why = {b: e[0] for b, e in errors.items() if e}
        raise StatisticsError(f"replicates with an undefined score exceed max_invalid_fraction={max_invalid_fraction} "
                              f"in {len(bad)} scope(s), worst {shown}; first undefined-score errors by benchmark: "
                              f"{why}. The metric's finalize must follow the official scorer on resampled subsets "
                              "(e.g. empty strata) or the threshold must be raised explicitly")


def paired_cluster_bootstrap(unit_stats: Mapping[SliceKey, SliceStats | tuple], metrics: Mapping[str, MetricLike], *,
                             labels: Sequence[str], seeds: Sequence[int], benchmarks: Sequence[str],
                             categories: Mapping[str, Sequence[str]] | None = None,
                             comparisons: Sequence[str | Sequence[str]] = (),
                             family: Sequence[str | Sequence[str]] = (), n_replicates: int = 10_000,
                             confidence: float = 0.95, seed: int = 314159, n_suite: int = SUITE_BENCHMARKS,
                             allow_partial_categories: bool = False, max_invalid_fraction: float = 0.0,
                             keep_weights: bool = False, keep_seed_replicates: bool = False) -> BootstrapResult:
    """Paired cluster bootstrap of official benchmark scores, macros and model differences.

    ``unit_stats[(label, seed, benchmark)]`` holds that slice's additive per-component statistics; all slices of a
    benchmark must cover the same components. Per replicate: draw G components with replacement from the global
    component set, weight every slice's statistics by the same multiplicities, finalize each benchmark score per seed,
    average over seeds, then form overall/category macros and differences. ``family`` comparisons (a subset of the
    overall-macro comparisons, added if absent) receive Holm-adjusted p-values; everything else is unadjusted and
    descriptive.

    A replicate can leave a benchmark score undefined (no drawn unit, or a stratum the official scorer divides by drawn
    empty); a ``finalize`` that raises ``MetricError`` (or returns NaN) there marks that replicate undefined. A macro
    or difference is undefined whenever any of its benchmarks or labels is, so its invalid fraction is that of the
    union. Percentiles over only the defined replicates would be conditional on that event, so by default any such
    replicate fails the call (``max_invalid_fraction=0``) naming the scopes; a caller that accepts the conditioning
    raises the threshold explicitly, and then every score and difference row, macros included, must stay within it.
    ``n_valid_replicates`` reports the count actually used.
    """

    labels = list(labels)
    seeds = [int(s) for s in seeds]
    benchmarks = list(benchmarks)
    for name, seq in (("labels", labels), ("seeds", seeds), ("benchmarks", benchmarks)):
        if not seq or len(set(seq)) != len(seq):
            raise StatisticsError(f"{name} must be non-empty and unique")
    if not 0 < confidence < 1:
        raise StatisticsError("confidence must be in (0, 1)")
    if n_replicates < 1:
        raise StatisticsError("n_replicates must be positive")
    if not 0 <= max_invalid_fraction <= 1:
        raise StatisticsError("max_invalid_fraction must be in [0, 1]")
    for b in benchmarks:
        if b not in metrics:
            raise StatisticsError(f"no metric for {b}")
        if metrics[b].kind == "unsupported":
            raise StatisticsError(f"{b} has no accepted metric adapter and cannot enter a macro")
    members = (_category_members(categories, benchmarks, allow_partial_categories)
               if categories is not None else {})
    pairs = resolve_comparisons(comparisons, labels)
    fam = resolve_comparisons(family, labels, unique=True)
    for p in fam:
        if p not in pairs:
            pairs.append(p)

    comp_ids, blocks = _prepare(unit_stats, labels, seeds, benchmarks)
    g = comp_ids.size
    n_l, n_s, n_b = len(labels), len(seeds), len(benchmarks)
    cache: dict[int, str] = {}

    point_seed = np.empty((n_l, n_s, n_b))
    for j, b in enumerate(benchmarks):
        blk = blocks[b]
        totals = blk.stacked.sum(axis=0, keepdims=True)
        try:
            point_seed[:, :, j] = _finalize_block(metrics[b], totals, n_l, n_s, blk.k, cache)[0]
        except UNDEFINED_ERRORS as exc:
            raise StatisticsError(f"{b}: the official score is not computable on the full rows ({exc})") from exc
    if not np.all(np.isfinite(point_seed)):
        bad = np.argwhere(~np.isfinite(point_seed))[0]
        raise StatisticsError(f"non-finite point score for {labels[bad[0]]}/seed {seeds[bad[1]]}/{benchmarks[bad[2]]}")
    point_bench = point_seed.mean(axis=1)

    if keep_weights and n_replicates * g > MAX_KEPT_WEIGHTS:
        raise StatisticsError("keep_weights is meant for small diagnostic runs")
    rep_bench = np.empty((n_replicates, n_l, n_b))
    rep_seed = np.empty((n_replicates, n_l, n_s, n_b)) if keep_seed_replicates else None
    kept = [] if keep_weights else None
    errors: dict[str, list[str]] = {b: [] for b in benchmarks}
    start = 0
    for w in iter_weight_blocks(g, n_replicates, seed):
        r = w.shape[0]
        if kept is not None:
            kept.append(w.astype(np.int32))
        wf = w.astype(np.float64)
        for j, b in enumerate(benchmarks):
            blk = blocks[b]
            totals = wf[:, blk.positions] @ blk.stacked
            vals = _finalize_block(metrics[b], totals, n_l, n_s, blk.k, cache, errors[b])
            rep_bench[start:start + r, :, j] = vals.mean(axis=2)
            if rep_seed is not None:
                rep_seed[start:start + r, :, :, j] = vals
        start += r

    score_rows = []
    point_scopes = _scope_reduce(point_bench, benchmarks, members, n_suite)
    rep_scopes = _scope_reduce(rep_bench, benchmarks, members, n_suite)
    for (level, scope, count, pv), (_, _, _, rv) in zip(point_scopes, rep_scopes):
        for li, label in enumerate(labels):
            r = rv[:, li]
            lo, hi = percentile_interval(r, confidence)
            finite = r[np.isfinite(r)]
            score_rows.append({"model_label": label, "level": level, "scope": scope, "benchmark_count": count,
                               "n_seeds": n_s, "estimate": float(pv[li]), "ci_low": lo, "ci_high": hi,
                               "bootstrap_se": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                               "n_valid_replicates": int(finite.size)})
    diff_rows = []
    pos = {lab: i for i, lab in enumerate(labels)}
    for a, b in pairs:
        for (level, scope, count, pv), (_, _, _, rv) in zip(point_scopes, rep_scopes):
            d = rv[:, pos[a]] - rv[:, pos[b]]
            lo, hi = percentile_interval(d, confidence)
            finite = d[np.isfinite(d)]
            diff_rows.append({"comparison": f"{a}-{b}", "label_a": a, "label_b": b, "level": level, "scope": scope,
                              "benchmark_count": count, "estimate": float(pv[pos[a]] - pv[pos[b]]), "ci_low": lo,
                              "ci_high": hi,
                              "bootstrap_se": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                              "p_value": bootstrap_p_value(d), "n_valid_replicates": int(finite.size),
                              "family_member": level == "overall" and (a, b) in fam})
    diffs = pd.DataFrame(diff_rows)
    if diffs.empty:
        diffs = pd.DataFrame(columns=["comparison", "label_a", "label_b", "level", "scope", "benchmark_count",
                                      "estimate", "ci_low", "ci_high", "bootstrap_se", "p_value",
                                      "n_valid_replicates", "family_member"])
    _check_invalid(score_rows, diff_rows, n_replicates, max_invalid_fraction, errors)
    diffs["p_holm"] = np.nan
    if fam:
        sel = diffs["family_member"].astype(bool)
        pv_fam = diffs.loc[sel, "p_value"].to_numpy(dtype=float)
        if not np.all(np.isfinite(pv_fam)):
            raise StatisticsError("a family comparison has no finite bootstrap p-value")
        diffs.loc[sel, "p_holm"] = holm_adjust(list(pv_fam))
    return BootstrapResult(
        labels=labels, seeds=seeds, benchmarks=benchmarks, categories={k: list(v) for k, v in members.items()},
        n_components=int(g), n_replicates=int(n_replicates), seed=int(seed), confidence=float(confidence),
        point_seed_scores=point_seed, point_benchmark_scores=point_bench, replicate_benchmark_scores=rep_bench,
        scores=pd.DataFrame(score_rows), differences=diffs, replicate_seed_scores=rep_seed,
        weights=np.concatenate(kept, axis=0) if kept is not None else None, component_ids=comp_ids,
    )


def bootstrap_from_rows(rows: pd.DataFrame, metrics: Mapping[str, MetricLike], assignments: pd.DataFrame, *,
                        labels: Sequence[str], seeds: Sequence[int], benchmarks: Sequence[str],
                        meta: Mapping[str, pd.DataFrame] | None = None, **kwargs: Any) -> BootstrapResult:
    """collect_unit_stats + paired_cluster_bootstrap in one call."""

    stats = collect_unit_stats(rows, metrics, assignments, labels=labels, seeds=seeds, benchmarks=benchmarks,
                               meta=meta)
    return paired_cluster_bootstrap(stats, metrics, labels=labels, seeds=seeds, benchmarks=benchmarks, **kwargs)
