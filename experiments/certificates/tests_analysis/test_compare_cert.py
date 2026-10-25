"""Tests of analysis/compare_cert.py on synthetic cert_scores rows with known effects.

    TRACE_TOOL_ROOT=<trace-tool checkout> python -m pytest experiments/certificates/tests_analysis -q

trace-tool (``trace_tool.compare``: mcnemar_exact, holm, paired_bootstrap) is found through
``$TRACE_TOOL_SRC`` or ``$TRACE_TOOL_ROOT/src``; without it the module is skipped.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compare_cert", HERE.parent / "analysis" / "compare_cert.py")
CC = importlib.util.module_from_spec(_spec)
sys.modules["compare_cert"] = CC
_spec.loader.exec_module(CC)

try:
    MCNEMAR, HOLM = CC.load_trace_tool(None)
except CC.CompareError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"trace-tool not available: {exc}", allow_module_level=True)
TT = importlib.import_module("trace_tool.compare")

ADAPTERS = (
    "selected_set.chess_piece_count",
    "filtered_mean.table_filtered_column_mean",
    "weighted_sum.lever_side_torque",
    "path.rpg_movement_cost",
)
SEEDS = (42, 43, 44)


def row(iid, seed, *, split="e1_test", adapter=ADAPTERS[0], scene=None, cell="c0", A=0.0, G=0.0, M=0.0, F=1.0,
        joint=0.0, tokens=100, condition="cert_v1", ext=None):
    return {"instance_id": iid, "cert_extension_id": ext or f"ext:{iid}", "adapter_id": adapter, "split": split,
            "cell": cell, "scene_group": scene or f"sg:{iid}", "condition": condition, "cond": "cert_px", "seed": seed,
            "A": float(A), "G": float(G), "M": float(M), "F": float(F), "joint": float(joint), "completion_tokens": int(tokens)}


def write(root: Path, rows) -> Path:
    """zs_eval/ckpt_eval layout: <root>/<split>/<condition>/seed<k>/cert/cert_scores.jsonl."""
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["condition"], r["seed"])].append(r)
    for (split, cond, seed), rs in by.items():
        d = root / split / cond / f"seed{seed}" / "cert"
        d.mkdir(parents=True, exist_ok=True)
        with (d / "cert_scores.jsonl").open("a") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    return root


def simulate(rng, *, n_per_adapter=300, eff_A=0.0, eff_G=0.0, eff_tokens=0.0, splits=("e1_test",), cluster_size=1,
             seeds=SEEDS, cells=("c0", "c1")):
    """Paired arms X and Y: Bernoulli A with P(X) = P(Y) + eff_A per row, G = base + eff_G + noise,
    M drawn from the same distribution in both arms (true effect 0), tokens shifted by eff_tokens."""
    xs, ys = [], []
    for split in splits:
        for ai, adapter in enumerate(ADAPTERS):
            for s in range(n_per_adapter):
                scene = f"sg:{split}:{ai}:{s}"
                for j in range(cluster_size):
                    iid = f"id:{split}:{ai}:{s}:{j}"
                    pa, g, m, tok = rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.7), rng.uniform(0.1, 0.9), rng.normal(300, 50)
                    cell = cells[s % len(cells)]
                    for seed in seeds:
                        common = dict(split=split, adapter=adapter, scene=scene, cell=cell)
                        ya, xa = rng.random() < pa, rng.random() < pa + eff_A
                        ys.append(row(iid, seed, A=ya, G=g + rng.normal(0, 0.05), M=float(rng.random() < m), F=1.0,
                                      joint=0.0, tokens=tok + rng.normal(0, 20), **common))
                        xs.append(row(iid, seed, A=xa, G=g + eff_G + rng.normal(0, 0.05), M=float(rng.random() < m), F=1.0,
                                      joint=0.0, tokens=tok + eff_tokens + rng.normal(0, 20), **common))
    return xs, ys


def settings(n_boot=2000, n_perm=2000):
    return CC.Settings(n_boot=n_boot, n_perm=n_perm, seed=0)


def compare(tmp_path, arms, *, pairs=(), st=None, metrics=CC.DEFAULT_METRICS, condition="cert_v1", cells=True):
    specs = []
    for name, rows in arms.items():
        specs.append(f"{name}={write(tmp_path / name.replace('@', '_s'), rows)}")
    ds = CC.load_dataset(specs, condition=condition, metrics=metrics)
    st = st or settings()
    chosen = CC.default_pairs(ds, list(pairs), {}, True)
    comps = CC.run_comparisons(ds, chosen, st, MCNEMAR, cells=cells)
    fams = CC.evaluate_families(comps, {}, st, HOLM)
    return ds, {c["name"]: c for c in comps}, fams


# ============================================================================== exact values


def test_hand_computed_pairing_seed_means_and_mcnemar(tmp_path):
    a, b = ADAPTERS[0], ADAPTERS[1]
    X = [row("id1", 42, adapter=a, A=1, G=0.5), row("id1", 43, adapter=a, A=0, G=0.7),
         row("id2", 42, adapter=b, A=1, G=0.2), row("id2", 43, adapter=b, A=1, G=0.4),
         row("id2", 44, adapter=b, A=1, G=0.9),  # seed 44 only in X: not used
         row("id3", 42, adapter=a, A=0, G=0.1), row("id3", 43, adapter=a, A=0, G=0.3),
         row("idX", 42, adapter=a, A=1, G=1.0)]  # unpaired
    Y = [row("id1", 42, adapter=a, A=0, G=0.1), row("id1", 43, adapter=a, A=0, G=0.3),
         row("id2", 42, adapter=b, A=1, G=0.2), row("id2", 43, adapter=b, A=0, G=0.2),
         row("id3", 42, adapter=a, A=1, G=0.1), row("id3", 43, adapter=a, A=0, G=0.1),
         row("idY", 42, adapter=b, A=0, G=0.0)]
    _, comps, _ = compare(tmp_path, {"Y": Y, "X": X}, st=settings(500, 500))
    assert list(comps) == ["X - Y"]  # no P/G/A arms: every arm vs the first one given
    entry = comps["X - Y"]["splits"]["e1_test"]
    pairing = entry["pairing"][0]
    assert (pairing["rows"], pairing["x_only"], pairing["y_only"], pairing["decoding_seeds"]) == (3, 1, 1, [42, 43])
    A, G = entry["scopes"]["all"]["A"], entry["scopes"]["all"]["G"]
    # per row over seeds 42, 43: A of X = .5, 1, 0 and of Y = 0, .5, .5 -> X - Y = (.5 + .5 - .5) / 3
    assert A["n_rows"] == 3 and A["mean_x"] == pytest.approx(0.5) and A["mean_y"] == pytest.approx(1 / 3)
    assert A["delta"] == pytest.approx(1 / 6)
    assert G["delta"] == pytest.approx(0.2)  # G of X = .6, .3, .2 and of Y = .2, .2, .1
    # McNemar on seed 42: b = Y-only successes (id3), c = X-only successes (id1)
    assert A["mcnemar"]["b_y_only"] == 1 and A["mcnemar"]["c_x_only"] == 1 and A["mcnemar"]["n"] == 3
    assert A["p"] == pytest.approx(MCNEMAR(1, 1)) and A["test"].startswith("mcnemar_exact")
    assert G["mcnemar"] is None and G["test"] == "cluster_signflip" and G["p"] == G["perm_p"]
    # per adapter scope only holds that adapter's rows
    assert entry["scopes"][f"adapter:{a}"]["A"]["n_rows"] == 2
    assert entry["scopes"][f"adapter:{b}"]["A"]["delta"] == pytest.approx(0.5)


def test_mcnemar_orientation_matches_trace_tool(tmp_path):
    rng = np.random.default_rng(1)
    xs, ys = simulate(rng, n_per_adapter=40, eff_A=0.2, seeds=(42,))
    _, comps, _ = compare(tmp_path, {"Y": ys, "X": xs}, st=settings(200, 200))
    res = comps["X - Y"]["splits"]["e1_test"]["scopes"]["all"]["A"]
    xa = {r["instance_id"]: r["A"] for r in xs}
    ya = {r["instance_id"]: r["A"] for r in ys}
    b = sum(1 for i in xa if ya[i] == 1 and xa[i] == 0)
    c = sum(1 for i in xa if xa[i] == 1 and ya[i] == 0)
    assert (res["mcnemar"]["b_y_only"], res["mcnemar"]["c_x_only"]) == (b, c)
    assert res["p"] == TT.mcnemar_exact(b, c)
    assert res["delta"] == pytest.approx((c - b) / len(xa))


# ============================================================================== known effects


def test_known_effects_recovered_with_default_resamples(tmp_path):
    rng = np.random.default_rng(7)
    xs, ys = simulate(rng, n_per_adapter=300, eff_A=0.15, eff_G=0.10, eff_tokens=-20.0)
    t0 = time.time()
    _, comps, _ = compare(tmp_path, {"Y": ys, "X": xs}, st=CC.Settings(), cells=False)  # 10,000 / 10,000, seed 0
    elapsed = time.time() - t0
    res = comps["X - Y"]["splits"]["e1_test"]["scopes"]["all"]
    for metric, truth in (("A", 0.15), ("G", 0.10), ("tokens", -20.0), ("M", 0.0)):
        r = res[metric]
        assert r["n_rows"] == 1200 and r["n_clusters"] == 1200
        assert r["ci"][0] < truth < r["ci"][1], (metric, r)
    assert abs(res["A"]["delta"] - 0.15) < 0.04
    assert abs(res["G"]["delta"] - 0.10) < 0.01
    assert abs(res["tokens"]["delta"] + 20.0) < 3.0
    assert res["A"]["p"] < 1e-6 and res["A"]["perm_p"] == pytest.approx(1 / 10001)
    assert res["G"]["p"] == pytest.approx(1 / 10001) and res["tokens"]["p"] == pytest.approx(1 / 10001)
    assert res["F"]["delta"] == 0 and res["F"]["mcnemar"]["b_y_only"] == 0 and res["F"]["p"] == 1.0
    assert elapsed < 120


def test_null_effect_ci_covers_zero_per_adapter(tmp_path):
    rng = np.random.default_rng(11)
    xs, ys = simulate(rng, n_per_adapter=150)
    _, comps, _ = compare(tmp_path, {"Y": ys, "X": xs})
    scopes = comps["X - Y"]["splits"]["e1_test"]["scopes"]
    assert {s for s in scopes if s.startswith("adapter:")} == {f"adapter:{a}" for a in ADAPTERS}
    assert len([s for s in scopes if s.startswith("cell:")]) == 8  # 4 adapters x 2 cells
    assert scopes[f"cell:{ADAPTERS[0]}/c0"]["G"]["n_rows"] == 75
    misses = [(name, m) for name, res in scopes.items() if not name.startswith("cell:")
              for m in ("A", "G", "M", "tokens") if not res[m]["ci"][0] <= 0 <= res[m]["ci"][1]]
    assert len(misses) <= 3, misses  # 20 nominal 95% intervals under the null


# ============================================================================== clustering and strata


def test_cluster_bootstrap_is_wider_than_row_bootstrap_for_correlated_clusters():
    rng = np.random.default_rng(3)
    n_clusters, size = 200, 8
    d = np.repeat(rng.normal(0.0, 1.0, n_clusters), size)  # identical rows within a cluster
    X, Y = d[:, None], np.zeros((len(d), 1))
    cl = CC.make_clusters(X, Y, np.repeat(np.arange(n_clusters), size).astype(str), np.array(["s"] * len(d), dtype=object))
    boot = CC.cluster_bootstrap(cl, 4000, np.random.default_rng(0))
    lo, hi = CC.percentile_ci(boot)
    rlo, rhi = TT.paired_bootstrap(d, n_boot=4000, seed=0)  # trace-tool's row-level bootstrap
    ratio = (hi[0] - lo[0]) / (rhi - rlo)
    assert 2.2 < ratio < 3.6  # ~sqrt(8) = 2.83


def test_stratified_resampling_keeps_adapter_shares_fixed():
    # stratum a: 10 clusters with difference 1; stratum b: 30 clusters with difference 0
    d = np.array([1.0] * 10 + [0.0] * 30)
    X, Y = d[:, None], np.zeros((40, 1))
    ids = np.arange(40).astype(str)
    strat = CC.make_clusters(X, Y, ids, np.array(["a"] * 10 + ["b"] * 30, dtype=object))
    boot = CC.cluster_bootstrap(strat, 2000, np.random.default_rng(0))
    assert np.allclose(boot, 0.25)  # the stratified estimate cannot move
    flat = CC.make_clusters(X, Y, ids, np.array(["a"] * 40, dtype=object))
    boot = CC.cluster_bootstrap(flat, 2000, np.random.default_rng(0))
    assert boot.std() > 0.03  # unstratified resampling moves the adapter mix


def test_cluster_spanning_two_strata_is_rejected():
    X, Y = np.ones((2, 1)), np.zeros((2, 1))
    with pytest.raises(CC.CompareError):
        CC.make_clusters(X, Y, np.array(["c", "c"]), np.array(["a", "b"], dtype=object))


def test_multi_row_clusters_from_files(tmp_path):
    rng = np.random.default_rng(5)
    xs, ys = simulate(rng, n_per_adapter=30, cluster_size=3, eff_G=0.05)
    _, comps, _ = compare(tmp_path, {"Y": ys, "X": xs}, st=settings(500, 500), cells=False)
    r = comps["X - Y"]["splits"]["e1_test"]["scopes"]["all"]["G"]
    assert r["n_rows"] == 360 and r["n_clusters"] == 120


# ============================================================================== sign-flip permutation


def test_signflip_extremes():
    zeros = CC.make_clusters(np.zeros((30, 2)), np.zeros((30, 2)), np.arange(30).astype(str), np.array(["s"] * 30, dtype=object))
    assert np.all(CC.signflip_p(zeros, 999, np.random.default_rng(0)) == 1.0)
    ones = CC.make_clusters(np.ones((60, 1)), np.zeros((60, 1)), np.arange(60).astype(str), np.array(["s"] * 60, dtype=object))
    assert CC.signflip_p(ones, 999, np.random.default_rng(0))[0] == pytest.approx(1 / 1000)


def test_signflip_is_calibrated_under_the_null():
    rng = np.random.default_rng(2024)
    rejections, n_sets = 0, 300
    for _ in range(n_sets):
        d = rng.normal(0, 1, 40)
        cl = CC.make_clusters(d[:, None], np.zeros((40, 1)), np.arange(40).astype(str), np.array(["s"] * 40, dtype=object))
        rejections += CC.signflip_p(cl, 199, rng)[0] <= 0.05
    assert 0.02 <= rejections / n_sets <= 0.09


# ============================================================================== families and Holm


def test_preregistered_families_use_holm_within_family(tmp_path):
    rng = np.random.default_rng(9)
    splits = ("e1_test", "e2_test", "e1_dev")
    p_rows, g_rows = simulate(rng, n_per_adapter=100, eff_A=0.2, splits=splits)  # (X with the effect, Y)
    # arm A: the G rows with answers degraded at random
    a_rows = [dict(r, A=float(r["A"] and rng.random() < 0.8)) for r in g_rows]
    _, comps, fams = compare(tmp_path, {"G": g_rows, "P": p_rows, "A": a_rows})
    assert list(comps) == ["P - G", "P - A", "G - A"]
    primary, secondary = fams
    assert primary["family"] == "primary" and primary["complete"]
    assert [(t["comparison"], t["split"], t["metric"]) for t in primary["tests"]] == [
        ("P - G", "e1_test", "A"), ("P - G", "e2_test", "A")]
    raw = [t["result"]["p"] for t in primary["tests"]]
    assert [t["holm_p"] for t in primary["tests"]] == pytest.approx(HOLM(raw))
    assert len(secondary["tests"]) == 4 and secondary["complete"]
    raw = [t["result"]["p"] for t in secondary["tests"]]
    assert [t["holm_p"] for t in secondary["tests"]] == pytest.approx(HOLM(raw))
    assert all(t["decision"] == "X > Y" for t in primary["tests"])  # 20 pp on 400 rows
    # e1_dev is compared (descriptive) but is not in any family
    assert "e1_dev" in comps["P - G"]["splits"]
    assert all(t["split"] != "e1_dev" for f in fams for t in f["tests"])


def test_family_roles_can_be_mapped_to_arm_names(tmp_path):
    rng = np.random.default_rng(6)
    p_rows, g_rows = simulate(rng, n_per_adapter=30, eff_A=0.3, splits=("e1_test", "e2_test"))
    gdir, pdir = write(tmp_path / "g", g_rows), write(tmp_path / "p", p_rows)
    js = tmp_path / "c.json"
    assert CC.main(["--arm", f"g3b-s1={gdir}", "--arm", f"p3b-s1={pdir}", "--family-arms", "P=p3b-s1,G=g3b-s1",
                    "--n-boot", "300", "--n-perm", "300", "--no-cells", "--out-json", str(js)]) == 0
    doc = json.loads(js.read_text())
    assert [c["name"] for c in doc["comparisons"]] == ["p3b-s1 - g3b-s1"]
    primary = doc["families"][0]
    assert primary["applicable"] and primary["complete"]
    assert [(t["comparison"], t["role"]) for t in primary["tests"]] == [("p3b-s1 - g3b-s1", "P - G")] * 2
    assert not doc["families"][1]["applicable"]  # no arm A


def test_family_incomplete_when_an_arm_is_missing(tmp_path):
    rng = np.random.default_rng(4)
    g_rows, p_rows = simulate(rng, n_per_adapter=20, splits=("e1_test",))
    _, comps, fams = compare(tmp_path, {"G": g_rows, "P": p_rows}, st=settings(300, 300))
    primary, secondary = fams
    assert primary["applicable"] and not primary["complete"] and sum(t["missing"] for t in primary["tests"]) == 1  # no e2_test
    assert all(t["missing"] for t in secondary["tests"])  # no arm A


# ============================================================================== training seeds


def _seeded_arms(effects):
    """P@k = G@k + effects[k] exactly on G (zero instance noise), random A."""
    rng = np.random.default_rng(12)
    arms = {}
    for k, eff in enumerate(effects, start=1):
        g_rows, _ = simulate(rng, n_per_adapter=25, splits=("e1_test", "e2_test"))
        arms[f"G@{k}"] = g_rows
        arms[f"P@{k}"] = [dict(r, G=r["G"] + eff, A=float(rng.random() < 0.5)) for r in g_rows]
    return arms


def test_two_level_bootstrap_mean_of_seed_differences_and_sign_rule(tmp_path):
    _, comps, fams = compare(tmp_path, _seeded_arms((0.2, 0.1, -0.05)), st=settings(3000, 300), cells=False)
    comp = comps["P - G"]
    assert comp["training_seed_pairs"] == [["1", "1"], ["2", "2"], ["3", "3"]]
    r = comp["splits"]["e1_test"]["scopes"]["all"]["G"]
    assert r["K"] == 3 and r["delta"] == pytest.approx((0.2 + 0.1 - 0.05) / 3)
    assert [s["delta"] for s in r["per_seed"]] == pytest.approx([0.2, 0.1, -0.05])
    assert r["sign_agree"] == 2 and r["sign_needed"] == 2
    # all three seeds resampled as -0.05 has probability 1/27 > 2.5%: the CI reaches -0.05
    assert r["ci"][0] == pytest.approx(-0.05) and r["ci"][1] == pytest.approx(0.2)
    assert not r["claim"] and "two-level" in r["ci_scope"]
    # per-seed Holm in the primary family: one adjusted p per training seed
    t = fams[0]["tests"][0]
    assert len(t["holm_p_per_seed"]) == 3
    by_seed = [[x["result"]["per_seed"][i]["p"] for x in fams[0]["tests"]] for i in range(3)]
    for i, ps in enumerate(by_seed):
        assert [x["holm_p_per_seed"][i] for x in fams[0]["tests"]] == pytest.approx(HOLM(ps))


def test_two_level_claim_when_all_seeds_agree(tmp_path):
    _, comps, _ = compare(tmp_path, _seeded_arms((0.2, 0.1, 0.15)), st=settings(3000, 300), cells=False)
    r = comps["P - G"]["splits"]["e1_test"]["scopes"]["all"]["G"]
    assert r["sign_agree"] == 3 and r["claim"]
    assert 0.1 - 1e-9 <= r["ci"][0] <= r["ci"][1] <= 0.2 + 1e-9


def test_single_untagged_baseline_is_shared_by_every_training_seed(tmp_path):
    rng = np.random.default_rng(8)
    base, p1 = simulate(rng, n_per_adapter=20, eff_G=0.1)
    _, p2 = simulate(np.random.default_rng(8), n_per_adapter=20, eff_G=0.1)
    _, comps, _ = compare(tmp_path, {"base": base, "P@1": p1, "P@2": p2}, pairs=[("P", "base")], st=settings(300, 300))
    assert comps["P - base"]["training_seed_pairs"] == [["1", ""], ["2", ""]]
    assert comps["P - base"]["splits"]["e1_test"]["scopes"]["all"]["G"]["K"] == 2


def test_unmatched_training_seeds_are_refused():
    runs_x = {"1": CC.Run("P", "1"), "2": CC.Run("P", "2")}
    runs_y = {"3": CC.Run("G", "3"), "4": CC.Run("G", "4")}
    with pytest.raises(CC.CompareError):
        CC.tseed_pairs(runs_x, runs_y)


# ============================================================================== inputs


def test_rows_of_other_conditions_are_ignored(tmp_path):
    X = [row("i1", 42, A=1), row("i1", 42, A=0, condition="ans_v1"), row("i2", 42, A=1)]
    Y = [row("i1", 42, A=0), row("i2", 42, A=0), row("i2", 42, A=1, condition="ans_v1")]
    ds, comps, _ = compare(tmp_path, {"Y": Y, "X": X}, st=settings(100, 100))
    assert ds.skipped_condition == 2
    assert comps["X - Y"]["splits"]["e1_test"]["scopes"]["all"]["A"]["delta"] == 1.0


def test_different_frozen_data_is_refused(tmp_path):
    X = [row("i1", 42, ext="ext:a")]
    Y = [row("i1", 42, ext="ext:b")]
    with pytest.raises(CC.CompareError, match="frozen data"):
        compare(tmp_path, {"Y": Y, "X": X})


def test_duplicate_rows_are_refused(tmp_path):
    root = write(tmp_path / "X", [row("i1", 42)])
    extra = tmp_path / "extra.jsonl"
    extra.write_text(json.dumps(row("i1", 42)) + "\n")
    with pytest.raises(CC.CompareError, match="twice"):
        CC.load_dataset([f"X={root}", f"X={extra}"])


def test_bad_trace_tool_src_is_refused(tmp_path):
    with pytest.raises(CC.CompareError):
        CC.load_trace_tool(str(tmp_path))


# ============================================================================== CLI


def test_cli_writes_json_and_markdown_deterministically(tmp_path):
    rng = np.random.default_rng(21)
    p_rows, g_rows = simulate(rng, n_per_adapter=20, eff_A=0.1, splits=("e1_test", "e2_test"))
    gdir, pdir = write(tmp_path / "g", g_rows), write(tmp_path / "p", p_rows)
    outs = []
    for i in range(2):
        js, md = tmp_path / f"c{i}.json", tmp_path / f"c{i}.md"
        rc = CC.main(["--arm", f"G={gdir}", "--arm", f"P={pdir}", "--n-boot", "500", "--n-perm", "500",
                      "--out-json", str(js), "--out-md", str(md)])
        assert rc == 0
        outs.append((js.read_text(), md.read_text()))
    assert outs[0] == outs[1]
    doc = json.loads(outs[0][0])
    assert doc["version"] == CC.VERSION and doc["config"]["n_boot"] == 500
    assert [c["name"] for c in doc["comparisons"]] == ["P - G"]
    text = outs[0][1]
    for needle in ("## Pre-registered families", "## Arm means", "## P - G", "### e1_test", "per cell:", "instance sampling only"):
        assert needle in text
    assert math.isfinite(doc["comparisons"][0]["splits"]["e2_test"]["scopes"]["all"]["A"]["delta"])


def test_cli_usage_error_returns_2(tmp_path, capsys):
    assert CC.main(["--arm", "bad-spec", "--out-json", str(tmp_path / "x.json")]) == 2
    assert "NAME[@TRAINING_SEED]=PATH" in capsys.readouterr().err
