"""Macros and the paired cluster bootstrap on synthetic fixtures (handoff 6.2, 11.1).

Most tests use tiny stand-in metrics that follow the BenchmarkMetric interface (additive per-unit statistics plus a
finalize that turns summed statistics into a percent score); the undefined-replicate contract is also checked with
the real metrics.py classes, which raise MetricError instead of returning NaN.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest

from trace_merge import statistics as st
from trace_merge.config import canonical_json, load_config, sha256_bytes


class RowMean:
    """Binary row mean: per-unit [sum, n]; optionally a non-broadcasting finalize to exercise the loop fallback."""

    kind = "binary_row"
    official_metric_id = "primary"

    def __init__(self, benchmark_id: str, vectorized: bool = True):
        self.benchmark_id = benchmark_id
        self.vectorized = vectorized

    def row_scores(self, rows, meta):
        return rows["archived_row_score"].astype(float)

    def unit_stats(self, rows, scores, unit):
        g = pd.DataFrame({"u": unit.to_numpy(), "s": scores.to_numpy()}).groupby("u")["s"].agg(["sum", "size"])
        return g.index.to_numpy(), g.to_numpy(dtype=float)

    def finalize(self, totals):
        t = np.asarray(totals, dtype=float)
        if self.vectorized:
            return 100.0 * t[..., 0] / t[..., 1]
        return 100.0 * t[0] / t[1]          # wrong on a 2-D input: must be detected, not trusted

    def official_score(self, rows, scores):
        _, s = self.unit_stats(rows, scores, pd.Series(np.arange(len(rows)), index=rows.index))
        return float(self.finalize(s.sum(axis=0)))


class StratumMean(RowMean):
    """Mean of per-stratum means (TableVQABench/PuzzleVQA style): per-unit [sum_j, n_j] for each stratum j."""

    kind = "aggregate_only_rederived"

    def __init__(self, benchmark_id: str, strata: list[str]):
        super().__init__(benchmark_id)
        self.strata = strata

    def unit_stats(self, rows, scores, unit):
        cols = {}
        for s in self.strata:
            m = (rows["stratum"] == s).to_numpy()
            cols[f"sum_{s}"] = scores.to_numpy() * m
            cols[f"n_{s}"] = m.astype(float)
        g = pd.DataFrame({"u": unit.to_numpy(), **cols}).groupby("u").sum()
        return g.index.to_numpy(), g.to_numpy(dtype=float)

    def finalize(self, totals):
        t = np.asarray(totals, dtype=float)
        sums, ns = t[..., 0::2], t[..., 1::2]
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * np.mean(sums / ns, axis=-1)


class Sum(RowMean):
    """Unnormalised sum of row scores, so a replicate value is the weighted count it was built from."""

    def unit_stats(self, rows, scores, unit):
        g = pd.DataFrame({"u": unit.to_numpy(), "s": scores.to_numpy()}).groupby("u")["s"].sum()
        return g.index.to_numpy(), g.to_numpy(dtype=float)[:, None]

    def finalize(self, totals):
        return np.asarray(totals, dtype=float)[..., 0]


def make_rows(spec: dict[tuple[str, int, str], list[float]], comps: dict[str, list[str]],
              strata: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rows for each (label, seed, benchmark) with per-benchmark content keys and component assignments."""

    rows = []
    assign = []
    for bench, cl in comps.items():
        for i, c in enumerate(cl):
            assign.append({"benchmark_id": bench, "content_key": f"{bench}:{i}", "component_id": c})
    for (label, seed, bench), scores in spec.items():
        for i, s in enumerate(scores):
            rec = {"model_label": label, "seed": seed, "benchmark_id": bench, "content_key": f"{bench}:{i}",
                   "archived_row_score": s}
            if strata and bench in strata:
                rec["stratum"] = strata[bench][i]
            rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(assign)


def random_spec(rng, labels, seeds, sizes):
    return {(lab, s, b): rng.integers(0, 2, n).astype(float).tolist()
            for lab in labels for s in seeds for b, n in sizes.items()}


# ------------------------------------------------------------------------------------------------------- macros
def test_macro_differs_from_pooled_on_unbalanced_data():
    demo = st.demonstrate_macro_vs_pooled()
    assert demo["macro_pct"] == pytest.approx(50.0)
    assert demo["pooled_pct"] == pytest.approx(82.0)
    assert demo["rows_per_benchmark"] == {"large": 900, "small": 100}


def test_benchmark_and_category_macros():
    scores = {"a": 10.0, "b": 30.0, "c": 50.0, "d": 70.0}
    cats = {"x": ["a", "b"], "y": ["c", "d"]}
    assert st.benchmark_macro(scores) == pytest.approx(40.0)
    assert st.category_macros(scores, cats) == {"x": 20.0, "y": 60.0}
    with pytest.raises(st.StatisticsError, match="lacks"):
        st.category_macros({"a": 10.0, "c": 50.0, "d": 70.0}, cats)
    assert st.category_macros({"a": 10.0, "c": 50.0, "d": 70.0}, cats, allow_partial=True) == {
        "x (partial: 1 of 2 benchmarks)": 10.0, "y": 60.0}
    with pytest.raises(st.StatisticsError, match="non-finite"):
        st.benchmark_macro({"a": float("nan")})
    assert st.macro_name(24) == "24-benchmark macro"
    assert st.macro_name(23) == "23-benchmark subset macro"


def test_macro_table_seed_means_and_coverage():
    rows = []
    for label, base in (("B", 0.0), ("T", 5.0)):
        for seed, shift in ((42, 0.0), (43, 2.0)):
            for bench, val in (("a", 10.0), ("b", 30.0), ("c", 50.0), ("d", 70.0)):
                rows.append({"model_label": label, "seed": seed, "benchmark_id": bench,
                             "official_value": val + base + shift})
    df = pd.DataFrame(rows)
    out = st.macro_table(df, {"x": ["a", "b"], "y": ["c", "d"]}, n_suite=4)
    t_mean = out.query("model_label == 'T' and seed_scope == 'mean_over_seeds' and level == 'overall'").iloc[0]
    assert t_mean["value"] == pytest.approx(46.0) and t_mean["scope"] == "4-benchmark macro"
    assert out.query("model_label == 'B' and seed == 43 and scope == 'x'")["value"].iloc[0] == pytest.approx(22.0)
    with pytest.raises(st.StatisticsError, match="lacks benchmarks"):
        st.macro_table(df.iloc[1:], n_suite=4)


def test_macro_table_refuses_unequal_seed_sets():
    df = pd.DataFrame([{"model_label": lab, "seed": s, "benchmark_id": b, "official_value": 50.0 + (s - 42)}
                       for lab, ss in (("A", (42, 43, 44)), ("B", (42,))) for s in ss for b in "xy"])
    with pytest.raises(st.StatisticsError, match="different seed sets"):
        st.macro_table(df, n_suite=2)
    out = st.macro_table(df, n_suite=2, seeds=[42])
    assert set(out["n_seeds"]) == {1} and set(out.loc[out["seed_scope"] == "single_seed", "seed"]) == {42}
    with pytest.raises(st.StatisticsError, match="lack requested seeds"):
        st.macro_table(df, n_suite=2, seeds=[42, 43])


def test_partial_category_scope_is_renamed_in_bootstrap():
    rng = np.random.default_rng(0)
    comps = {b: [f"{b}{i}" for i in range(10)] for b in "ab"}
    rows, assign = make_rows(random_spec(rng, ["A"], [42], {b: 10 for b in "ab"}), comps)
    res = st.bootstrap_from_rows(rows, {b: RowMean(b) for b in "ab"}, assign, labels=["A"], seeds=[42],
                                 benchmarks=["a", "b"], categories={"K": ["a", "c"], "L": ["b"]},
                                 allow_partial_categories=True, n_replicates=10)
    cats = res.scores.query("level == 'category'").set_index("scope")["benchmark_count"].to_dict()
    assert cats == {"K (partial: 1 of 2 benchmarks)": 1, "L": 1}
    assert set(res.categories) == set(cats)


# ------------------------------------------------------------------------------------------ multiplicity/tests
def test_holm_known_example():
    adj = st.holm_adjust({"h1": 0.01, "h2": 0.04, "h3": 0.03, "h4": 0.005})
    assert adj == pytest.approx({"h1": 0.03, "h2": 0.06, "h3": 0.06, "h4": 0.02})
    assert st.holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    with pytest.raises(st.StatisticsError):
        st.holm_adjust([0.1, float("nan")])


def test_mcnemar_exact():
    assert st.mcnemar_exact(0, 5) == pytest.approx(2 * 0.5 ** 5)
    assert st.mcnemar_exact(4, 4) == pytest.approx(1.0)
    assert st.mcnemar_exact(0, 0) == 1.0


def test_bootstrap_p_value_construction():
    assert st.bootstrap_p_value(np.full(999, 3.0)) == pytest.approx(2 / 1000)
    assert st.bootstrap_p_value(np.zeros(99)) == 1.0
    r = np.concatenate([np.full(25, -1.0), np.full(75, 1.0)])
    assert st.bootstrap_p_value(r) == pytest.approx(2 * 26 / 101)


def test_resolve_family_labels():
    labels = ["T", "V@P1", "M050@P1", "CV050@P1", "CT050@P1"]
    fam = st.resolve_comparisons(["M050-T", "M050-V@P1", "M050-CV050", "M050-CT050"], labels)
    assert fam == [("M050@P1", "T"), ("M050@P1", "V@P1"), ("M050@P1", "CV050@P1"), ("M050@P1", "CT050@P1")]
    assert st.resolve_comparisons(["M050-T", "M050@P1-T"], labels) == [("M050@P1", "T")]
    with pytest.raises(st.StatisticsError, match="repeats"):
        st.resolve_comparisons(["M050-T", "M050@P1-T"], labels, unique=True)
    with pytest.raises(st.StatisticsError):
        st.resolve_comparisons(["M050-V@P0"], labels)
    with pytest.raises(st.StatisticsError):
        st.parse_comparison("a-b-c")


# ------------------------------------------------------------------------------------------------ bootstrap
def two_benchmark_fixture(seed=0, labels=("A", "B"), seeds=(42, 43)):
    rng = np.random.default_rng(seed)
    # every benchmark has enough components that a replicate drawing none of them is practically impossible
    comps = {"x": [f"c{i // 2}" for i in range(20)], "y": [f"c{100 + i}" for i in range(12)]}
    spec = random_spec(rng, labels, seeds, {"x": 20, "y": 12})
    rows, assign = make_rows(spec, comps)
    metrics = {"x": RowMean("x"), "y": RowMean("y")}
    return rows, assign, metrics


def test_finalize_of_sums_equals_direct_score_with_unit_weights():
    rng = np.random.default_rng(1)
    strata = {"s": [("p", "q", "r")[i % 3] for i in range(30)]}
    comps = {"x": [f"c{i // 2}" for i in range(20)], "s": [f"d{i // 4}" for i in range(30)]}
    spec = random_spec(rng, ["A", "B"], [42, 43], {"x": 20, "s": 30})
    rows, assign = make_rows(spec, comps, strata)
    metrics = {"x": RowMean("x"), "s": StratumMean("s", ["p", "q", "r"])}
    res = st.bootstrap_from_rows(rows, metrics, assign, labels=["A", "B"], seeds=[42, 43], benchmarks=["x", "s"],
                                 n_replicates=10, seed=3)
    for li, lab in enumerate(["A", "B"]):
        for si, seed in enumerate([42, 43]):
            sx = rows.query("model_label == @lab and seed == @seed and benchmark_id == 'x'")
            assert res.point_seed_scores[li, si, 0] == pytest.approx(100 * sx["archived_row_score"].mean())
            assert res.point_seed_scores[li, si, 0] == pytest.approx(
                metrics["x"].official_score(sx, sx["archived_row_score"]))
            ss = rows.query("model_label == @lab and seed == @seed and benchmark_id == 's'")
            direct = 100 * ss.groupby("stratum")["archived_row_score"].mean().mean()
            assert res.point_seed_scores[li, si, 1] == pytest.approx(direct)
    overall = res.scores.query("model_label == 'A' and level == 'overall'").iloc[0]
    assert overall["estimate"] == pytest.approx(res.point_seed_scores[0].mean(axis=0).mean())
    assert overall["scope"] == "2-benchmark subset macro"


def test_bootstrap_is_reproducible_and_seed_dependent():
    rows, assign, metrics = two_benchmark_fixture()
    kw = dict(labels=["A", "B"], seeds=[42, 43], benchmarks=["x", "y"], comparisons=["A-B"], n_replicates=600)
    r1 = st.bootstrap_from_rows(rows, metrics, assign, seed=11, **kw)
    r2 = st.bootstrap_from_rows(rows, metrics, assign, seed=11, **kw)
    r3 = st.bootstrap_from_rows(rows, metrics, assign, seed=12, **kw)
    np.testing.assert_array_equal(r1.replicate_benchmark_scores, r2.replicate_benchmark_scores)
    pd.testing.assert_frame_equal(r1.differences, r2.differences)
    assert not np.array_equal(r1.replicate_benchmark_scores, r3.replicate_benchmark_scores)
    # input row order does not matter
    r4 = st.bootstrap_from_rows(rows.sample(frac=1, random_state=5), metrics, assign, seed=11, **kw)
    np.testing.assert_array_equal(r1.replicate_benchmark_scores, r4.replicate_benchmark_scores)


def test_same_weights_for_every_model_and_seed():
    rows, assign, metrics = two_benchmark_fixture(labels=("A", "B", "C"), seeds=(42, 43, 44))
    labels, seeds, benches = ["A", "B", "C"], [42, 43, 44], ["x", "y"]
    stats = st.collect_unit_stats(rows, metrics, assign, labels=labels, seeds=seeds, benchmarks=benches)
    res = st.paired_cluster_bootstrap(stats, metrics, labels=labels, seeds=seeds, benchmarks=benches,
                                      n_replicates=300, seed=7, keep_weights=True, keep_seed_replicates=True)
    np.testing.assert_array_equal(res.weights, st.draw_component_weights(res.n_components, 300, 7))
    assert (res.weights.sum(axis=1) == res.n_components).all()
    pos = {c: i for i, c in enumerate(res.component_ids)}
    for li, lab in enumerate(labels):
        for si, seed in enumerate(seeds):
            for bi, b in enumerate(benches):
                ids, s = stats[(lab, seed, b)].unit_ids, stats[(lab, seed, b)].stats
                w = res.weights[:, [pos[str(i)] for i in ids]].astype(float)
                expect = metrics[b].finalize(w @ s)
                np.testing.assert_allclose(res.replicate_seed_scores[:, li, si, bi], expect)
    np.testing.assert_allclose(res.replicate_benchmark_scores, res.replicate_seed_scores.mean(axis=2))


def test_all_seeds_and_variants_of_a_component_move_together():
    comps = {"x": ["c0", "c0", "c0", "c1", "c2", "c3", "c4", "c5"]}    # three question variants share c0
    a42 = [1, 1, 1, 0, 0, 0, 0, 0]
    spec = {("A", 42, "x"): a42, ("A", 43, "x"): a42, ("B", 42, "x"): a42, ("B", 43, "x"): a42}
    rows, assign = make_rows(spec, comps)
    res = st.bootstrap_from_rows(rows, {"x": Sum("x")}, assign, labels=["A", "B"], seeds=[42, 43],
                                 benchmarks=["x"], comparisons=["A-B"], n_replicates=500, seed=2,
                                 keep_weights=True, keep_seed_replicates=True)
    c0 = list(res.component_ids).index("c0")
    for li in range(2):
        for si in range(2):
            np.testing.assert_array_equal(res.replicate_seed_scores[:, li, si, 0], 3 * res.weights[:, c0])
    d = res.differences.query("level == 'overall'").iloc[0]
    assert (d["estimate"], d["ci_low"], d["ci_high"], d["p_value"]) == (0.0, 0.0, 0.0, 1.0)
    assert res.weights[:, c0].std() > 0


def test_component_spanning_two_benchmarks_is_coupled():
    comps = {"x": ["shared"] + [f"x{i}" for i in range(9)], "y": ["shared"] + [f"y{i}" for i in range(5)]}
    spec = {("A", 42, "x"): [1] + [0] * 9, ("A", 42, "y"): [1] + [0] * 5}
    rows, assign = make_rows(spec, comps)
    res = st.bootstrap_from_rows(rows, {"x": Sum("x"), "y": Sum("y")}, assign, labels=["A"], seeds=[42],
                                 benchmarks=["x", "y"], n_replicates=1000, seed=4, keep_weights=True)
    assert res.n_components == 15
    rep = res.replicate_benchmark_scores[:, 0, :]
    np.testing.assert_array_equal(rep[:, 0], rep[:, 1])
    shared = list(res.component_ids).index("shared")
    np.testing.assert_array_equal(rep[:, 0], res.weights[:, shared])
    assert rep[:, 0].std() > 0


def test_clear_difference_ci_pvalue_and_holm_family():
    comps = {"x": [f"c{i}" for i in range(30)]}
    spec = {("A", 42, "x"): [1.0] * 30, ("B", 42, "x"): [0.0] * 30, ("C", 42, "x"): [1.0] * 15 + [0.0] * 15}
    rows, assign = make_rows(spec, comps)
    res = st.bootstrap_from_rows(rows, {"x": RowMean("x")}, assign, labels=["A", "B", "C"], seeds=[42],
                                 benchmarks=["x"], comparisons=["C-B"], family=["A-B", "A-C"], n_replicates=999,
                                 seed=9)
    d = res.differences.query("level == 'overall'").set_index("comparison")
    ab = d.loc["A-B"]
    assert (ab["estimate"], ab["ci_low"], ab["ci_high"]) == (100.0, 100.0, 100.0)
    assert ab["p_value"] == pytest.approx(2 / 1000)
    assert bool(ab["family_member"]) and not bool(d.loc["C-B", "family_member"])
    assert np.isnan(d.loc["C-B", "p_holm"])
    fam = d[d["family_member"].astype(bool)]
    assert fam["p_holm"].to_dict() == pytest.approx(st.holm_adjust(fam["p_value"].to_dict()))
    # a repeated comparison does not enlarge the Holm family
    dup = st.bootstrap_from_rows(rows, {"x": RowMean("x")}, assign, labels=["A", "B", "C"], seeds=[42],
                                 benchmarks=["x"], comparisons=["A-B", "A-B", "C-B"], family=["A-B", "A-C"],
                                 n_replicates=999, seed=9)
    assert int(dup.differences["family_member"].astype(bool).sum()) == 2
    assert len(dup.differences.query("level == 'overall'")) == 3
    assert set(res.differences["level"]) == {"overall", "benchmark"}
    summary = res.summary()
    assert summary["p_value_method"] == st.P_VALUE_METHOD and summary["n_replicates"] == 999


def test_vectorized_and_loop_finalize_agree():
    rows, assign, _ = two_benchmark_fixture()
    kw = dict(labels=["A", "B"], seeds=[42, 43], benchmarks=["x", "y"], n_replicates=300, seed=5)
    fast = st.bootstrap_from_rows(rows, {"x": RowMean("x"), "y": RowMean("y")}, assign, **kw)
    slow = st.bootstrap_from_rows(rows, {"x": RowMean("x", vectorized=False), "y": RowMean("y", False)}, assign,
                                  **kw)
    np.testing.assert_allclose(fast.replicate_benchmark_scores, slow.replicate_benchmark_scores)


def test_category_macros_inside_bootstrap():
    rng = np.random.default_rng(3)
    comps = {b: [f"{b}{i}" for i in range(10)] for b in "abc"}
    spec = random_spec(rng, ["A"], [42], {b: 10 for b in "abc"})
    rows, assign = make_rows(spec, comps)
    metrics = {b: RowMean(b) for b in "abc"}
    res = st.bootstrap_from_rows(rows, metrics, assign, labels=["A"], seeds=[42], benchmarks=list("abc"),
                                 categories={"k1": ["a", "b"], "k2": ["c"]}, n_replicates=50, seed=1)
    k1 = res.scores.query("level == 'category' and scope == 'k1'").iloc[0]
    assert k1["estimate"] == pytest.approx(res.point_benchmark_scores[0, :2].mean())
    with pytest.raises(st.StatisticsError, match="lacks"):
        st.bootstrap_from_rows(rows, metrics, assign, labels=["A"], seeds=[42], benchmarks=["a", "c"],
                               categories={"k1": ["a", "b"], "k2": ["c"]}, n_replicates=5)


def test_undefined_replicate_scores_fail_closed_unless_accepted():
    # stratum "rare" lives in a single component, so many replicates draw it zero times
    comps = {"s": ["c0"] + [f"c{i}" for i in range(1, 20)]}
    strata = {"s": ["rare"] + ["common"] * 19}
    spec = {("A", 42, "s"): [1.0] + [0.0, 1.0] * 9 + [1.0]}
    rows, assign = make_rows(spec, comps, strata)
    metrics = {"s": StratumMean("s", ["rare", "common"])}
    kw = dict(labels=["A"], seeds=[42], benchmarks=["s"], n_replicates=400, seed=1)
    with pytest.raises(st.StatisticsError, match=r"undefined score exceed .*'A benchmark:s'"):
        st.bootstrap_from_rows(rows, metrics, assign, **kw)
    res = st.bootstrap_from_rows(rows, metrics, assign, max_invalid_fraction=1.0, **kw)
    n_valid = res.scores.query("level == 'benchmark'")["n_valid_replicates"].iloc[0]
    assert 0 < n_valid < 400


class RaisingStratumMean(StratumMean):
    """Stratum mean whose finalize raises on an empty stratum, as the pinned scorers (and metrics.py) do."""

    def finalize(self, totals):
        t = np.asarray(totals, dtype=float)
        if t.ndim != 1:
            raise TypeError("scalar only")
        if not (t[1::2] > 0).all():
            raise ValueError("stratum absent; the official scorer is undefined there")
        return 100.0 * float(np.mean(t[0::2] / t[1::2]))


def test_raising_finalize_counts_as_undefined_replicate():
    comps = {"s": ["c0"] + [f"c{i}" for i in range(1, 20)]}
    strata = {"s": ["rare"] + ["common"] * 19}
    rows, assign = make_rows({("A", 42, "s"): [1.0] + [0.0, 1.0] * 9 + [1.0]}, comps, strata)
    kw = dict(labels=["A"], seeds=[42], benchmarks=["s"], n_replicates=400, seed=1)
    nan_metric = st.bootstrap_from_rows(rows, {"s": StratumMean("s", ["rare", "common"])}, assign,
                                        max_invalid_fraction=1.0, keep_seed_replicates=True, **kw)
    raising = st.bootstrap_from_rows(rows, {"s": RaisingStratumMean("s", ["rare", "common"])}, assign,
                                     max_invalid_fraction=1.0, keep_seed_replicates=True, **kw)
    np.testing.assert_array_equal(nan_metric.replicate_seed_scores, raising.replicate_seed_scores)
    with pytest.raises(st.StatisticsError, match="stratum absent"):
        st.bootstrap_from_rows(rows, {"s": RaisingStratumMean("s", ["rare", "common"])}, assign, **kw)


def test_real_metric_errors_become_counted_undefined_replicates():
    """metrics.py raises MetricError (not NaN) when a replicate draws no rows of a benchmark."""

    from trace_merge import metrics as mt

    comps = {"erqa": ["e0", "e1"], "blink": [f"b{i}" for i in range(60)]}
    rng = np.random.default_rng(0)
    spec = {("A", 42, b): rng.integers(0, 2, len(c)).astype(float).tolist() for b, c in comps.items()}
    rows, assign = make_rows(spec, comps)
    rows["official_excluded"] = False
    metrics = {b: mt.get_metric(b) for b in comps}
    kw = dict(labels=["A"], seeds=[42], benchmarks=list(comps), n_replicates=500, seed=1)
    with pytest.raises(st.StatisticsError, match=r"exceed max_invalid_fraction=0\.0.*erqa"):
        st.bootstrap_from_rows(rows, metrics, assign, **kw)
    res = st.bootstrap_from_rows(rows, metrics, assign, max_invalid_fraction=1.0, keep_weights=True, **kw)
    pos = [list(res.component_ids).index(c) for c in ("e0", "e1")]
    drawn = int((res.weights[:, pos].sum(axis=1) > 0).sum())
    valid = res.scores.query("level == 'benchmark'").set_index("scope")["n_valid_replicates"]
    assert valid["erqa"] == drawn < 500 and valid["blink"] == 500
    assert res.scores.query("level == 'overall'")["n_valid_replicates"].iloc[0] == drawn
    # a score that cannot be computed on the full rows is a hard error, not an undefined replicate
    strata_rows = rows[rows["benchmark_id"] == "blink"].assign(stratum="common")
    never = {"blink": RaisingStratumMean("blink", ["common", "never"])}
    with pytest.raises(st.StatisticsError, match="blink: the official score is not computable on the full rows"):
        st.bootstrap_from_rows(strata_rows, never, assign, labels=["A"], seeds=[42], benchmarks=["blink"],
                               n_replicates=10, max_invalid_fraction=1.0)


def test_macro_invalid_fraction_is_checked_on_the_union():
    comps = {"s": ["r1"] + [f"s{i}" for i in range(30)], "t": ["r2"] + [f"t{i}" for i in range(30)]}
    strata = {"s": ["rare"] + ["common"] * 30, "t": ["rare"] + ["common"] * 30}
    spec = {("A", 42, "s"): [1] + [0, 1] * 15, ("A", 42, "t"): [0] + [1, 0] * 15}
    rows, assign = make_rows(spec, comps, strata)
    metrics = {b: StratumMean(b, ["rare", "common"]) for b in "st"}
    kw = dict(labels=["A"], seeds=[42], benchmarks=["s", "t"], n_replicates=2000, seed=3)
    with pytest.raises(st.StatisticsError, match="overall"):
        st.bootstrap_from_rows(rows, metrics, assign, max_invalid_fraction=0.40, **kw)
    res = st.bootstrap_from_rows(rows, metrics, assign, max_invalid_fraction=0.70, **kw)
    valid = res.scores.set_index("scope")["n_valid_replicates"]
    assert 1 - valid["s"] / 2000 < 0.40 and 1 - valid["2-benchmark subset macro"] / 2000 > 0.40


def test_lost_row_inside_a_shared_component_fails():
    comps = {"x": ["c0", "c0", "c1", "c2", "c3", "c4"]}
    spec = {("A", 42, "x"): [1, 0, 1, 1, 0, 0], ("B", 42, "x"): [1, 0, 1, 1, 0, 0]}
    rows, assign = make_rows(spec, comps)
    lost = rows[~((rows["model_label"] == "B") & (rows["content_key"] == "x:1"))]
    kw = dict(labels=["A", "B"], seeds=[42], benchmarks=["x"])
    with pytest.raises(st.StatisticsError, match=r"different content rows \(0 only in the former.*x:1"):
        st.bootstrap_from_rows(lost, {"x": RowMean("x")}, assign, n_replicates=10, **kw)
    # unit stats built elsewhere: the content fingerprints catch it although the component ids agree
    stats = st.collect_unit_stats(rows, {"x": RowMean("x")}, assign, **kw)
    sub = lost[lost["model_label"] == "B"]
    unit = sub["content_key"].map(st.component_map(assign))
    ids, vals = RowMean("x").unit_stats(sub, sub["archived_row_score"].astype(float), unit)
    np.testing.assert_array_equal(ids, stats[("B", 42, "x")].unit_ids)
    tampered = {**stats, ("B", 42, "x"): st.SliceStats(ids, vals, st.content_fingerprint(sub["content_key"]))}
    with pytest.raises(st.StatisticsError, match="different content-row sets"):
        st.paired_cluster_bootstrap(tampered, {"x": RowMean("x")}, n_replicates=10, **kw)
    unprinted = {**stats, ("B", 42, "x"): st.SliceStats(ids, vals)}
    with pytest.raises(st.StatisticsError, match="different content-row sets"):
        st.paired_cluster_bootstrap(unprinted, {"x": RowMean("x")}, n_replicates=10, **kw)


def test_unpaired_or_incomplete_inputs_fail():
    rows, assign, metrics = two_benchmark_fixture()
    stats = st.collect_unit_stats(rows, metrics, assign, labels=["A", "B"], seeds=[42, 43], benchmarks=["x", "y"])
    broken = dict(stats)
    s = broken[("B", 43, "x")]
    broken[("B", 43, "x")] = st.SliceStats(s.unit_ids[1:], s.stats[1:])
    with pytest.raises(st.StatisticsError, match="different component set"):
        st.paired_cluster_bootstrap(broken, metrics, labels=["A", "B"], seeds=[42, 43], benchmarks=["x", "y"],
                                    n_replicates=5)
    with pytest.raises(st.StatisticsError, match="no rows"):
        st.collect_unit_stats(rows, metrics, assign, labels=["A", "Z"], seeds=[42], benchmarks=["x"])
    with pytest.raises(st.StatisticsError, match="no component id"):
        st.collect_unit_stats(rows, metrics, assign[assign["content_key"] != "x:0"], labels=["A"], seeds=[42],
                              benchmarks=["x"])
    with pytest.raises(st.StatisticsError, match="duplicate content rows"):
        st.collect_unit_stats(pd.concat([rows, rows.iloc[:1]]), metrics, assign, labels=["A"], seeds=[42],
                              benchmarks=["x"])


# ------------------------------------------------------------------------------------- real data (opt-in only)
@pytest.mark.skipif(not os.environ.get("TRACE_MERGE_WORK_ROOT"), reason="real-data check needs TRACE_MERGE_WORK_ROOT")
def test_real_archive_point_estimates_match_aggregates():
    import pyarrow.parquet as pq

    cfg = load_config()
    if not cfg.archive_run_root.is_dir():
        pytest.skip("archive snapshot not present")
    labels = {"B": cfg.role("B")["archived_model_id"], "T": cfg.role("T")["archived_model_id"],
              "V@P0": cfg.role("V")["archived_model_id"]}
    benches, seeds = ["erqa", "treebench"], [int(s) for s in cfg.raw["archive"]["seeds"]]
    root = cfg.archive_run_root / "data" / "scores" / f"run={cfg.raw['archive']['run_id']}"
    cols = ["score_scope", "score_value", "source_ordinal", "source_index", "source_row_sha256",
            "source_record_sha256", "media_set_sha256"]
    parts, aggs = [], {}
    for label, model_id in labels.items():
        for seed in seeds:
            for bench in benches:
                (path,) = glob.glob(str(root / f"model={model_id}" / f"seed={seed}" / f"benchmark={bench}" / "*.parquet"))
                df = pq.ParquetFile(path).read(columns=cols).to_pandas()
                aggs[(label, seed, bench)] = float(df.loc[df["score_scope"] == "aggregate", "score_value"].iloc[0])
                df = df[df["score_scope"] == "row"]
                keys = [sha256_bytes(canonical_json([bench, int(o), str(i), r, s]).encode())
                        for o, i, r, s in zip(df["source_ordinal"], df["source_index"], df["source_row_sha256"],
                                              df["source_record_sha256"])]
                parts.append(pd.DataFrame({"model_label": label, "seed": seed, "benchmark_id": bench,
                                           "content_key": keys, "archived_row_score": df["score_value"].to_numpy(),
                                           "media_set_sha256": df["media_set_sha256"].to_numpy()}))
    rows = pd.concat(parts, ignore_index=True)
    ga_path = cfg.work("splits", "group_assignments.parquet")
    if not ga_path.is_file():
        pytest.skip("splits/group_assignments.parquet not frozen yet")
    assign = pd.read_parquet(ga_path, columns=["content_key", "component_id"])
    assert rows["content_key"].isin(set(assign["content_key"])).all()
    res = st.bootstrap_from_rows(rows, {b: RowMean(b) for b in benches}, assign, labels=list(labels), seeds=seeds,
                                 benchmarks=benches, comparisons=["T-B", "V@P0-B"], n_replicates=200, seed=314159)
    for li, lab in enumerate(labels):
        for si, seed in enumerate(seeds):
            for bi, b in enumerate(benches):
                assert res.point_seed_scores[li, si, bi] == pytest.approx(aggs[(lab, seed, b)], abs=1e-6)
