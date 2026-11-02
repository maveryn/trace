"""Behavioural transition tables on synthetic fixtures with hand-computed answers (handoff 11.1)."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trace_merge import behavior as bh
from trace_merge.config import canonical_json, load_config, sha256_bytes

# (B, T, V) cell -> count; 40 units with every cell non-empty and distinct.
CELLS = {"000": 5, "001": 3, "010": 7, "011": 2, "100": 4, "101": 1, "110": 6, "111": 12}


def expand_cells(cells: dict[str, int]) -> tuple[list[int], list[int], list[int]]:
    b, t, v = [], [], []
    for code, n in cells.items():
        b += [int(code[0])] * n
        t += [int(code[1])] * n
        v += [int(code[2])] * n
    return b, t, v


def aligned(outcomes: dict[str, dict[int, list[int]]], benchmarks: list[str] | str = "bench_a",
            prefix: str = "k") -> pd.DataFrame:
    rows = []
    for label, by_seed in outcomes.items():
        for seed, vals in by_seed.items():
            bench = [benchmarks] * len(vals) if isinstance(benchmarks, str) else benchmarks
            for i, v in enumerate(vals):
                rows.append({"model_label": label, "seed": seed, "benchmark_id": bench[i],
                             "content_key": f"{prefix}{i:04d}", "binary_correct": v, "official_excluded": False})
    return pd.DataFrame(rows)


def btv_frame(seed: int = 42, vero: str = "V@P0") -> pd.DataFrame:
    b, t, v = expand_cells(CELLS)
    return aligned({"B": {seed: b}, "T": {seed: t}, vero: {seed: v}})


def overall(df: pd.DataFrame, **filters) -> pd.DataFrame:
    sel = (df["level"] == "overall")
    for k, v in filters.items():
        sel &= df[k] == v
    return df[sel]


# ------------------------------------------------------------------------------------------------ eight cells
def test_joint_outcomes_known_eight_cells():
    m = bh.build_outcome_matrix(btv_frame())
    out = bh.joint_outcomes(m, labels=("B", "T", "V@P0"))
    rows = out[(out["level"] == "benchmark") & (out["seed"] == 42)]
    assert len(rows) == 8
    got = dict(zip(rows["cell_code"], rows["n_cell"]))
    assert got == {k: float(v) for k, v in CELLS.items()}
    assert rows["n_units"].unique().tolist() == [40]
    np.testing.assert_allclose(rows.set_index("cell_code")["cell_rate_pct"].loc["111"], 100 * 12 / 40)
    assert rows["cell_rate_pct"].sum() == pytest.approx(100.0)
    cell = rows.set_index("cell_code").loc["010"]
    assert cell["cell"] == "B=0|T=1|V@P0=0" and cell["outcome_2"] == 1
    assert cell["acc_B_pct"] == pytest.approx(100 * 23 / 40)


def test_corrections_regressions_conditional_vs_unconditional():
    m = bh.build_outcome_matrix(btv_frame())
    out = bh.corrections_regressions(m, base="B", models=("T", "V@P0"))
    t = out[(out["model_label"] == "T") & (out["level"] == "benchmark")].iloc[0]
    assert (t["n_units"], t["n_base_error"], t["n_base_success"]) == (40, 17, 23)
    assert (t["n_corrections"], t["n_regressions"]) == (9, 5)
    assert t["correction_rate_pct"] == pytest.approx(100 * 9 / 40)            # population rate
    assert t["correction_given_base_error_pct"] == pytest.approx(100 * 9 / 17)  # conditional on B errors
    assert t["regression_rate_pct"] == pytest.approx(100 * 5 / 40)
    assert t["regression_given_base_success_pct"] == pytest.approx(100 * 5 / 23)
    assert t["net_delta_pp"] == pytest.approx(10.0)
    assert t["net_delta_pp"] == pytest.approx(t["acc_model_pct"] - t["acc_base_pct"])
    assert (t["acc_B_pct"], t["acc_T_pct"], t["acc_V@P0_pct"]) == pytest.approx((57.5, 67.5, 45.0))
    v = out[(out["model_label"] == "V@P0") & (out["level"] == "benchmark")].iloc[0]
    assert (v["n_corrections"], v["n_regressions"]) == (5, 10)
    assert v["net_delta_pp"] == pytest.approx(-12.5)


def test_parent_overlap_and_hindsight_label():
    m = bh.build_outcome_matrix(btv_frame())
    r = bh.parent_overlap(m, base="B", trace="T", vero="V@P0").query("level == 'benchmark'").iloc[0]
    assert (r["n_trace_only"], r["n_vero_only"], r["n_both_correct"], r["n_both_wrong"]) == (13, 4, 14, 9)
    assert (r["n_trace_only_base_correct"], r["n_trace_only_base_wrong"]) == (6, 7)
    assert (r["n_vero_only_base_correct"], r["n_vero_only_base_wrong"]) == (1, 3)
    assert r["agreement_pct"] == pytest.approx(100 * 23 / 40)
    assert r["hindsight_either_parent_correct_pct"] == pytest.approx(100 * 31 / 40)
    assert "hindsight" in r["hindsight_note"] and "not an upper bound" in r["hindsight_note"]
    assert (r["acc_base_pct"], r["acc_trace_pct"], r["acc_vero_pct"]) == pytest.approx((57.5, 67.5, 45.0))


# --------------------------------------------------------------------------------------- scopes and weighting
def test_category_pooled_and_benchmark_macro_differ():
    b, t, v = expand_cells(CELLS)
    n_small = 10
    frame = aligned({"B": {42: b + [1] * n_small}, "T": {42: t + [0] * n_small}, "V@P0": {42: v + [1] * n_small}},
                    benchmarks=["bench_a"] * 40 + ["bench_b"] * n_small)
    m = bh.build_outcome_matrix(frame)
    cats = {"cat": ["bench_a", "bench_b"]}
    out = bh.corrections_regressions(m, models=("T",), categories=cats)
    per_b = out[(out["level"] == "benchmark")].set_index("scope")
    pooled = out[(out["level"] == "category") & (out["aggregation"] == "pooled_rows")].iloc[0]
    macro = out[(out["level"] == "category") & (out["aggregation"] == "benchmark_macro")].iloc[0]
    assert per_b.loc["bench_b", "regression_rate_pct"] == pytest.approx(100.0)
    assert pooled["regression_rate_pct"] == pytest.approx(100 * (5 + 10) / 50)
    assert macro["regression_rate_pct"] == pytest.approx((12.5 + 100.0) / 2)
    assert macro["benchmark_count"] == 2 and pooled["n_units"] == 50 and macro["n_units"] == 50
    # bench_b has no base errors, so its conditional correction rate is undefined and the macro must say so
    assert np.isnan(per_b.loc["bench_b", "correction_given_base_error_pct"])
    assert np.isnan(macro["correction_given_base_error_pct"])
    assert overall(out, aggregation="pooled_rows")["n_units"].iloc[0] == 50


def test_seed_average_uses_ratio_of_sums_for_conditional_rates():
    frame = aligned({"B": {42: [0, 0, 1, 1], 43: [0, 1, 1, 1]}, "T": {42: [1, 0, 1, 1], 43: [1, 1, 0, 1]}})
    m = bh.build_outcome_matrix(frame)
    out = bh.corrections_regressions(m, models=("T",)).query("level == 'benchmark'").set_index("seed_scope")
    s42 = out[out["seed"] == 42].iloc[0]
    s43 = out[out["seed"] == 43].iloc[0]
    mean = out.loc["mean_over_seeds"]
    assert s42["correction_given_base_error_pct"] == pytest.approx(50.0)
    assert s43["correction_given_base_error_pct"] == pytest.approx(100.0)
    assert mean["correction_rate_pct"] == pytest.approx((s42["correction_rate_pct"] + s43["correction_rate_pct"]) / 2)
    assert mean["correction_given_base_error_pct"] == pytest.approx(100 * 2 / 3)   # not the 75% mean of rates
    assert mean["regression_given_base_success_pct"] == pytest.approx(100 * 1 / 5)
    assert (mean["n_base_error"], mean["n_corrections"], mean["n_seeds"]) == (1.5, 1.0, 2)
    assert pd.isna(mean["seed"])


def test_extra_scopes_and_component_support():
    frame = btv_frame()
    m = bh.build_outcome_matrix(frame)
    keys = sorted(frame["content_key"].unique())
    slices = pd.Series(["chart" if i % 2 else "photo" for i in range(len(keys))], index=keys)
    comps = pd.Series([f"c{i // 4}" for i in range(len(keys))], index=keys)
    out = bh.parent_overlap(m, extra_scopes={"visual_domain": slices}, components=comps)
    dom = out[(out["level"] == "visual_domain") & (out["aggregation"] == "pooled_rows")].set_index("scope")
    assert set(dom.index) == {"chart", "photo"}
    assert dom.loc["chart", "n_units"] == 20
    assert out.query("level == 'overall'")["n_groups"].tolist() == [10, 10]
    assert out.query("level == 'overall'")["scope"].unique().tolist() == ["1-benchmark binary-eligible set"]


def test_extra_scopes_must_cover_every_unit():
    frame = btv_frame()
    m = bh.build_outcome_matrix(frame)
    keys = sorted(frame["content_key"].unique())
    wrong_key = pd.Series(["chart"] * 40, index=[f"sample-{i}" for i in range(40)])
    with pytest.raises(bh.BehaviorError, match="no entry for 40 of 40 units"):
        bh.parent_overlap(m, extra_scopes={"visual_domain": wrong_key})
    with pytest.raises(bh.BehaviorError, match="no entry for 20 of 40"):
        bh.parent_overlap(m, extra_scopes={"visual_domain": pd.Series(["chart"] * 20, index=keys[:20])})
    # explicit NaN = annotated sample did not cover the unit: allowed and counted
    partial = pd.Series(["chart"] * 20 + [np.nan] * 20, index=keys)
    row = bh.parent_overlap(m, extra_scopes={"visual_domain": partial}).query("level == 'visual_domain'").iloc[0]
    assert (row["n_units"], row["n_units_unlabelled"]) == (20, 20)


def test_partial_categories_are_named_as_partial():
    b, t, v = expand_cells(CELLS)
    frame = aligned({"B": {42: b}, "T": {42: t}, "V@P0": {42: v}}, benchmarks=["bench_a"] * 20 + ["bench_b"] * 20)
    m = bh.build_outcome_matrix(frame)
    cats = {"Visual Math": ["bench_a", "bench_c", "bench_d"], "Other": ["bench_b"]}
    out = bh.corrections_regressions(m, models=("T",), categories=cats)
    scopes = set(out.loc[out["level"] == "category", "scope"])
    assert scopes == {"Visual Math (partial: 1 of 3 benchmarks)", "Other"}


# ----------------------------------------------------------------------------------------------- persistence
def test_persistence_defined_on_one_seed_measured_on_another():
    frame = aligned({
        "B": {42: [0] * 8, 43: [0] * 8},
        "T": {42: [1, 1, 1, 1, 0, 0, 0, 0], 43: [1, 1, 0, 1, 1, 0, 0, 0]},
        "V@P0": {42: [0, 0, 0, 0, 0, 1, 1, 0], 43: [0, 0, 0, 1, 0, 1, 0, 0]},
    })
    m = bh.build_outcome_matrix(frame)
    patterns = {"trace_only": {"T": 1, "V@P0": 0}, "vero_only": {"T": 0, "V@P0": 1}}
    out = bh.persistence(m, patterns=patterns).query("level == 'benchmark'")
    t = out[out["set_name"] == "trace_only"]
    fwd = t[(t["def_seed"] == 42) & (t["eval_seed"] == 43)].iloc[0]
    assert (fwd["n_defined"], fwd["n_persisting"]) == (4, 2)
    assert fwd["persistence_pct"] == pytest.approx(50.0)
    assert fwd["eval_base_rate_pct"] == pytest.approx(100 * 3 / 8)
    assert fwd["T_holds_pct"] == pytest.approx(75.0) and fwd["V@P0_holds_pct"] == pytest.approx(75.0)
    assert (fwd["acc_T_def_pct"], fwd["acc_T_eval_pct"]) == pytest.approx((50.0, 50.0))
    back = t[(t["def_seed"] == 43) & (t["eval_seed"] == 42)].iloc[0]
    assert back["persistence_pct"] == pytest.approx(100 * 2 / 3)
    pooled = t[t["seed_scope"] == bh.POOLED_PAIRS].iloc[0]
    assert pooled["persistence_pct"] == pytest.approx(100 * 4 / 7) and pooled["n_seed_pairs"] == 2
    v = out[(out["set_name"] == "vero_only") & (out["def_seed"] == 42)].iloc[0]
    assert v["persistence_pct"] == pytest.approx(50.0)


def test_persistence_label_swap_null_known_values():
    # T/V disagree on units 0-3 at seed 42 and on units 0, 1, 4 at seed 43
    frame = aligned({
        "B": {42: [0] * 6, 43: [0] * 6},
        "T": {42: [1, 1, 0, 1, 0, 0], 43: [1, 0, 0, 0, 1, 0]},
        "V@P0": {42: [0, 0, 1, 0, 0, 0], 43: [0, 1, 0, 0, 0, 0]},
    })
    m = bh.build_outcome_matrix(frame)
    out = bh.persistence(m, patterns={"trace_only": {"T": 1, "V@P0": 0}, "both": {"T": 1, "V@P0": 1}})
    pooled = out[(out["level"] == "benchmark") & (out["seed_scope"] == bh.POOLED_PAIRS)].set_index("set_name")
    # null = sum(a_d a_e) / (2 sum a_d) over both ordered pairs: (2 + 2) / (2 * (4 + 3))
    assert pooled.loc["trace_only", "null_persistence_pct"] == pytest.approx(100 * 4 / 14)
    assert pooled.loc["trace_only", "persistence_excess_pp"] == pytest.approx(
        pooled.loc["trace_only", "persistence_pct"] - 100 * 4 / 14)
    assert "label-swap null" in pooled.loc["trace_only", "null_reference"]
    assert np.isnan(pooled.loc["both", "null_persistence_pct"])
    assert pooled.loc["both", "null_reference"] == bh.NO_NULL_NOTE
    assert "persistence_lift" not in out.columns
    # every row shows the accuracy of every pattern label on both seeds
    assert {"acc_B_def_pct", "acc_T_def_pct", "acc_V@P0_eval_pct"} <= set(bh.persistence(m).columns)


def test_persistence_null_matches_identical_models_not_base_rate():
    rng = np.random.default_rng(7)
    n = 20000
    p = rng.beta(0.5, 0.5, n)                     # shared item difficulty, no complementarity at all
    outs = {lab: {s: (rng.random(n) < p).astype(int).tolist() for s in (42, 43, 44)} for lab in ("T", "V@P0")}
    m = bh.build_outcome_matrix(aligned(outs))
    r = bh.persistence(m, patterns={"trace_only": {"T": 1, "V@P0": 0}}).query(
        "level == 'overall' and seed_scope == @bh.POOLED_PAIRS").iloc[0]
    assert r["persistence_pct"] > r["eval_base_rate_pct"] + 4          # heterogeneity alone beats the base rate
    assert abs(r["persistence_excess_pp"]) < 1.0                        # the null absorbs it


def test_persistence_needs_two_seeds():
    m = bh.build_outcome_matrix(btv_frame())
    with pytest.raises(bh.BehaviorError, match="two seeds"):
        bh.persistence(m)


# -------------------------------------------------------------------------------------------- capture/damage
GROUP_FIXTURE = {  # (T, V) -> merge outcomes
    (1, 0): [1, 1, 1, 0, 0],
    (0, 1): [1, 0, 0, 0],
    (1, 1): [1] * 6 + [0] * 2,
    (0, 0): [1, 0, 0],
}


def capture_frame(vero: str = "V@P1", seeds: tuple[int, ...] = (42, 43)) -> pd.DataFrame:
    t, v, mo = [], [], []
    for (tv, vv), outs in GROUP_FIXTURE.items():
        t += [tv] * len(outs)
        v += [vv] * len(outs)
        mo += outs
    base = [1] * 5 + [0] * 15            # B right on the first five units (the T-only group at seed 42)
    parents: dict[str, dict[int, list[int]]] = {"T": {42: t}, vero: {42: v}, "B": {42: base}}
    if 43 in seeds:  # parents swap roles on seed 43
        parents["T"][43], parents[vero][43], parents["B"][43] = v, t, base
    if 44 in seeds:
        parents["T"][44], parents[vero][44], parents["B"][44] = v, t, base
    return aligned({**parents, "M050@P1": {42: mo}})


def test_capture_damage_contemporaneous_known_values():
    m = bh.build_outcome_matrix(capture_frame())
    out = bh.capture_damage(m, merge="M050@P1", trace="T", vero="V@P1", parent_seeds=[42, 43])
    c = out[(out["variant"] == "contemporaneous") & (out["level"] == "benchmark")].set_index("metric")
    assert c.loc["trace_only_capture", "n_group"] == 5
    assert c.loc["trace_only_capture", "conditional_rate_pct"] == pytest.approx(60.0)
    assert c.loc["vero_only_capture", "conditional_rate_pct"] == pytest.approx(25.0)
    assert c.loc["shared_success_damage", "conditional_rate_pct"] == pytest.approx(25.0)
    assert c.loc["shared_success_damage", "event"] == "merge_wrong"
    assert c.loc["joint_failure_rescue", "conditional_rate_pct"] == pytest.approx(100 / 3)
    assert c["contribution_pp"].to_dict() == pytest.approx({
        "trace_only_capture": 15.0, "vero_only_capture": 5.0, "shared_success_damage": 10.0,
        "joint_failure_rescue": 5.0})
    assert c["n_units"].unique().tolist() == [20]
    assert c["acc_merge_pct"].iloc[0] == pytest.approx(55.0)
    assert (c["acc_base_pct"].iloc[0], c["acc_trace_pct"].iloc[0], c["acc_vero_pct"].iloc[0]) == pytest.approx(
        (25.0, 65.0, 60.0))
    assert set(c["base_label"]) == {"B"}
    assert c["merge_correct_contribution_pp"].sum() == pytest.approx(55.0)
    assert c["group_share_pct"].sum() == pytest.approx(100.0)


def test_capture_damage_independent_repeat_uses_other_seed_groups():
    m = bh.build_outcome_matrix(capture_frame())
    out = bh.capture_damage(m, merge="M050@P1", trace="T", vero="V@P1", parent_seeds=[42, 43])
    r = out[(out["variant"] == "independent_repeat") & (out["level"] == "benchmark")]
    assert r["parent_seeds"].unique().tolist() == ["43"] and r["merge_seed"].unique().tolist() == [42]
    r = r.set_index("metric")
    # on seed 43 the parents swapped, so the TRACE-only group is the seed-42 VERO-only group
    assert r.loc["trace_only_capture", "n_group"] == 4
    assert r.loc["trace_only_capture", "conditional_rate_pct"] == pytest.approx(25.0)
    assert r.loc["vero_only_capture", "conditional_rate_pct"] == pytest.approx(60.0)
    assert r.loc["shared_success_damage", "conditional_rate_pct"] == pytest.approx(25.0)


def test_capture_damage_parameterized_on_archived_vero_and_stable_variant():
    m = bh.build_outcome_matrix(capture_frame(vero="V@P0", seeds=(42, 43, 44)))
    out = bh.capture_damage(m, merge="M050@P1", trace="T", vero="V@P0")
    assert set(out["vero_label"]) == {"V@P0"}
    assert set(out["variant"]) == {"contemporaneous", "independent_repeat", "stable_other_seeds"}
    st = out[(out["variant"] == "stable_other_seeds") & (out["level"] == "overall")]
    assert st["parent_seeds"].unique().tolist() == ["43,44"]
    assert st["n_group"].sum() == pytest.approx(20)
    assert st["merge_correct_contribution_pp"].sum() == pytest.approx(st["acc_merge_pct"].iloc[0])
    ind = out[(out["variant"] == "independent_repeat") & (out["level"] == "overall")]
    assert set(ind["seed_scope"]) == {bh.SINGLE_CONDITION, bh.POOLED_CONDITIONS}


def test_capture_damage_without_other_parent_seed_is_explicit():
    frame = capture_frame(seeds=(42,))
    m = bh.build_outcome_matrix(frame)
    out = bh.capture_damage(m, merge="M050@P1", trace="T", vero="V@P1")
    assert set(out["variant"]) == {"contemporaneous"}
    with pytest.raises(bh.BehaviorError, match="no capture condition"):
        bh.capture_damage(m, merge="M050@P1", variants=("independent_repeat",))


# ------------------------------------------------------------------------------------------------ fail closed
def test_duplicate_rows_fail():
    frame = btv_frame()
    with pytest.raises(bh.BehaviorError, match="duplicate"):
        bh.build_outcome_matrix(pd.concat([frame, frame.iloc[:1]], ignore_index=True))


def test_fractional_scores_fail():
    frame = btv_frame().astype({"binary_correct": float})
    frame.loc[0, "binary_correct"] = 0.5
    with pytest.raises(bh.BehaviorError, match="outside"):
        bh.build_outcome_matrix(frame)


def test_missing_generation_raises_or_is_counted():
    frame = btv_frame()
    frame = frame[~((frame["model_label"] == "T") & (frame["content_key"] == "k0003"))]
    m = bh.build_outcome_matrix(frame)
    assert bh.missingness_table(m).query("model_label == 'T'")["n_missing"].sum() == 1
    with pytest.raises(bh.BehaviorError, match="lack a required outcome"):
        bh.joint_outcomes(m)
    out = bh.joint_outcomes(m, missing="drop")
    row = out[(out["level"] == "benchmark")].iloc[0]
    assert row["n_units"] == 39 and row["n_units_dropped_missing"] == 1


def test_official_exclusions_fail_unless_dropped_and_null_flag_fails():
    frame = btv_frame()
    frame.loc[frame["content_key"] == "k0000", "official_excluded"] = True
    with pytest.raises(bh.BehaviorError, match="official_excluded"):
        bh.build_outcome_matrix(frame)
    frame["official_excluded"] = frame["official_excluded"].map({True: "True", False: "False"})  # archive strings
    m = bh.build_outcome_matrix(frame, exclusions="drop")
    assert int(m.excluded["n_excluded"].sum()) == 3
    assert len(m.values) == 39
    cov = bh.coverage_table(m)
    assert int(cov["n_excluded"].sum()) == 3 and set(cov["status"]) == {"binary_eligible"}
    frame["official_excluded"] = frame["official_excluded"].astype(object)
    frame.loc[1, "official_excluded"] = None
    with pytest.raises(bh.BehaviorError, match="non-boolean"):
        bh.build_outcome_matrix(frame)


def test_non_binary_benchmarks_are_left_out_but_recorded():
    frame = btv_frame()
    other = btv_frame().assign(benchmark_id="wemath", binary_correct=np.nan)
    other["content_key"] = "w" + other["content_key"]
    both = pd.concat([frame, other], ignore_index=True)
    m = bh.build_outcome_matrix(both)
    assert set(m.values.index.get_level_values("benchmark_id")) == {"bench_a"}
    assert m.ineligible == ("wemath",)
    cov = bh.coverage_table(m)
    assert cov.loc[cov["benchmark_id"] == "wemath", "status"].tolist() == ["no_binary_outcome"]
    # a benchmark the registry calls binary-eligible but whose outcomes were nulled by a failed join must fail
    with pytest.raises(bh.BehaviorError, match="without any 0/1 outcome"):
        bh.build_outcome_matrix(both, expected_benchmarks=["bench_a", "wemath"])
    unexpected = both.assign(binary_correct=both["binary_correct"].fillna(1))
    with pytest.raises(bh.BehaviorError, match="outside expected_benchmarks"):
        bh.build_outcome_matrix(unexpected, expected_benchmarks=["bench_a"])
    assert len(bh.build_outcome_matrix(both, expected_benchmarks=["bench_a"]).values) == 40


# --------------------------------------------------------------------------------------------- uncertainty
def test_bootstrap_intervals_match_the_primary_bootstrap():
    """Same seed and components: accuracy and paired-change intervals equal statistics.paired_cluster_bootstrap's."""

    from trace_merge import metrics as mt
    from trace_merge import statistics as st

    b, t, v = expand_cells(CELLS)
    frame = aligned({"B": {42: b}, "T": {42: t}, "V@P0": {42: v}}, benchmarks="erqa")
    keys = sorted(frame["content_key"].unique())
    comps = pd.Series([f"c{i // 2}" for i in range(len(keys))], index=keys)   # pairs of rows share a component
    spec = bh.BootstrapSpec(n_replicates=400, seed=11, max_invalid_fraction=1.0)
    m = bh.build_outcome_matrix(frame)
    cr = bh.corrections_regressions(m, models=("T",), components=comps, uncertainty=spec)
    row = cr[(cr["level"] == "benchmark")].iloc[0]

    rows = frame.assign(archived_row_score=frame["binary_correct"].astype(float))
    assign = pd.DataFrame({"content_key": comps.index, "component_id": comps.to_numpy()})
    res = st.bootstrap_from_rows(rows, {"erqa": mt.get_metric("erqa")}, assign, labels=["B", "T"], seeds=[42],
                                 benchmarks=["erqa"], comparisons=["T-B"], n_replicates=400, seed=11)
    sb = res.scores.query("level == 'benchmark' and model_label == 'B'").iloc[0]
    assert (row["acc_B_pct_ci_low"], row["acc_B_pct_ci_high"]) == pytest.approx((sb["ci_low"], sb["ci_high"]))
    d = res.differences.query("level == 'benchmark'").iloc[0]
    assert row["net_delta_pp"] == pytest.approx(d["estimate"])
    assert (row["net_delta_pp_ci_low"], row["net_delta_pp_ci_high"]) == pytest.approx((d["ci_low"], d["ci_high"]))
    assert row["bootstrap_replicates"] == 400 and row["bootstrap_min_valid_replicates"] == 400


def test_bootstrap_intervals_on_headline_rows_and_reproducible():
    b, t, v = expand_cells(CELLS)
    frame = aligned({"B": {42: b, 43: t}, "T": {42: t, 43: v}, "V@P0": {42: v, 43: b}},
                    benchmarks=["bench_a"] * 20 + ["bench_b"] * 20)
    keys = sorted(frame["content_key"].unique())
    comps = pd.Series([f"c{i // 2}" for i in range(len(keys))], index=keys)
    m = bh.build_outcome_matrix(frame)
    spec = bh.BootstrapSpec(n_replicates=300, seed=3, min_groups=15)
    cats = {"cat": ["bench_a", "bench_b"]}
    po = bh.parent_overlap(m, categories=cats, components=comps, uncertainty=spec)
    again = bh.parent_overlap(m, categories=cats, components=comps, uncertainty=spec)
    pd.testing.assert_frame_equal(po, again)
    head = po[po["seed_scope"] == bh.MEAN_OVER_SEEDS]
    assert head["trace_only_pct_ci_low"].notna().all() and po.loc[po["seed_scope"] == bh.SINGLE_SEED,
                                                                 "trace_only_pct_ci_low"].isna().all()
    assert (head["trace_only_pct_ci_low"] <= head["trace_only_pct"]).all()
    assert (head["trace_only_pct"] <= head["trace_only_pct_ci_high"]).all()
    macro = head[(head["level"] == "category") & (head["aggregation"] == bh.BENCHMARK_MACRO)]
    assert len(macro) == 1 and macro["agreement_pct_ci_low"].notna().all()
    assert head.set_index("scope").loc["bench_a", "below_min_groups"]                 # 10 groups < 15
    assert not head[head["level"] == "overall"]["below_min_groups"].any()             # 20 groups
    # conditional rates whose denominator a replicate can draw empty get no interval past the threshold
    cd_frame = capture_frame()
    cd_keys = sorted(cd_frame["content_key"].unique())
    cd = bh.capture_damage(bh.build_outcome_matrix(cd_frame), merge="M050@P1", trace="T", vero="V@P1",
                           components=pd.Series(cd_keys, index=cd_keys),
                           uncertainty=bh.BootstrapSpec(n_replicates=500, seed=1))
    con = cd[(cd["variant"] == "contemporaneous") & (cd["level"] == "benchmark")].set_index("metric")
    assert con["contribution_pp_ci_low"].notna().all()
    small = con.loc["joint_failure_rescue"]                                           # 3 of 20 units
    assert np.isnan(small["conditional_rate_pct_ci_low"]) and small["bootstrap_min_valid_replicates"] < 500
    with pytest.raises(bh.BehaviorError, match="component ids"):
        bh.parent_overlap(m, uncertainty=spec)


def test_bootstrap_spec_from_config():
    spec = bh.BootstrapSpec.from_config({"bootstrap_replicates": 10000, "seed": 314159, "confidence": 0.95,
                                         "exploratory_min_groups_per_slice": 100})
    assert (spec.n_replicates, spec.seed, spec.confidence, spec.min_groups) == (10000, 314159, 0.95, 100)
    with pytest.raises(bh.BehaviorError):
        bh.BootstrapSpec(confidence=1.5)


def test_behavior_tables_bundle_and_write(tmp_path: Path):
    b, t, v = expand_cells(CELLS)
    frame = aligned({"B": {42: b, 43: t}, "T": {42: t, 43: v}, "V@P0": {42: v, 43: b}})
    m = bh.build_outcome_matrix(frame)
    tables = bh.behavior_tables(m, categories={"cat": ["bench_a"]})
    assert set(tables) == {"corrections_regressions", "joint_outcomes", "parent_overlap", "persistence",
                           "behavior_coverage"}
    digests = bh.write_tables(tables, tmp_path)
    for name, digest in digests.items():
        assert sha256_bytes((tmp_path / name).read_bytes()) == digest
    back = pd.read_csv(tmp_path / "joint_outcomes.csv")
    assert len(back) == len(tables["joint_outcomes"])


# ------------------------------------------------------------------------------------- real data (opt-in only)
def _archive_frame(cfg, benchmarks: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    labels = {"B": cfg.role("B")["archived_model_id"], "T": cfg.role("T")["archived_model_id"],
              "V@P0": cfg.role("V")["archived_model_id"]}
    cols = ["score_scope", "score_value", "excluded", "source_ordinal", "source_index", "source_row_sha256",
            "source_record_sha256", "benchmark_id"]
    parts = []
    root = cfg.archive_run_root / "data" / "scores" / f"run={cfg.raw['archive']['run_id']}"
    for label, model_id in labels.items():
        for seed in cfg.raw["archive"]["seeds"]:
            for bench in benchmarks:
                (path,) = glob.glob(str(root / f"model={model_id}" / f"seed={seed}" / f"benchmark={bench}" / "*.parquet"))
                df = pq.ParquetFile(path).read(columns=cols).to_pandas()
                agg = float(df.loc[df["score_scope"] == "aggregate", "score_value"].iloc[0])
                df = df[df["score_scope"] == "row"].copy()
                df["content_key"] = [sha256_bytes(canonical_json([bench, int(o), str(i), r, s]).encode())
                                     for o, i, r, s in zip(df["source_ordinal"], df["source_index"],
                                                           df["source_row_sha256"], df["source_record_sha256"])]
                parts.append(pd.DataFrame({"model_label": label, "seed": int(seed), "benchmark_id": bench,
                                           "content_key": df["content_key"], "binary_correct": df["score_value"],
                                           "official_excluded": df["excluded"], "archived_aggregate": agg}))
    return pd.concat(parts, ignore_index=True)


@pytest.mark.skipif(not os.environ.get("TRACE_MERGE_WORK_ROOT"), reason="real-data check needs TRACE_MERGE_WORK_ROOT")
def test_real_archive_accuracies_reproduce_aggregates():
    cfg = load_config()
    if not cfg.archive_run_root.is_dir():
        pytest.skip("archive snapshot not present")
    frame = _archive_frame(cfg, ["erqa", "treebench"])
    m = bh.build_outcome_matrix(frame)
    out = bh.corrections_regressions(m, models=("T", "V@P0"))
    bench = out[(out["level"] == "benchmark") & (out["seed_scope"] == bh.SINGLE_SEED)]
    agg = frame.drop_duplicates(["model_label", "seed", "benchmark_id"]).set_index(
        ["model_label", "seed", "benchmark_id"])["archived_aggregate"]
    for _, r in bench.iterrows():
        for lab in ("B", "T", "V@P0"):
            assert r[f"acc_{lab}_pct"] == pytest.approx(agg[(lab, int(r["seed"]), r["scope"])], abs=1e-6)
    joint = bh.joint_outcomes(m)
    sums = joint.groupby(["level", "scope", "aggregation", "seed_scope", "seed"], dropna=False)["n_cell"].sum()
    units = joint.groupby(["level", "scope", "aggregation", "seed_scope", "seed"], dropna=False)["n_units"].first()
    np.testing.assert_allclose(sums.to_numpy(), units.to_numpy())
    # frozen components: every archived unit of these benchmarks has one, and intervals bracket the estimates
    ga_path = cfg.work("splits", "group_assignments.parquet")
    if not ga_path.is_file():
        pytest.skip("splits/group_assignments.parquet not frozen yet")
    ga = pd.read_parquet(ga_path, columns=["content_key", "component_id"])
    comps = ga.drop_duplicates("content_key").set_index("content_key")["component_id"]
    po = bh.parent_overlap(m, components=comps, uncertainty=bh.BootstrapSpec(n_replicates=200, seed=314159))
    head = po[(po["seed_scope"] == bh.MEAN_OVER_SEEDS) & (po["level"] == "overall")]
    assert ((head["agreement_pct_ci_low"] <= head["agreement_pct"])
            & (head["agreement_pct"] <= head["agreement_pct_ci_high"])).all()
