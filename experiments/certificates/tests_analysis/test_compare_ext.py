"""Tests of analysis/compare_ext.py: the frozen per-row rules, the macro statistics, and an end-to-end ``analyze``
on a synthetic trace_eval_v1 layout (summaries + item tables, no xlsx).

    TRACE_TOOL_ROOT=<trace-tool checkout> python -m pytest experiments/certificates/tests_analysis/test_compare_ext.py -q

trace-tool (``trace_tool.compare.holm``) is found through ``$TRACE_TOOL_SRC`` or ``$TRACE_TOOL_ROOT/src``; without
it the end-to-end test is skipped (the rule and statistics tests need numpy only).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compare_ext", HERE.parent / "analysis" / "compare_ext.py")
CE = importlib.util.module_from_spec(_spec)
sys.modules["compare_ext"] = CE
_spec.loader.exec_module(CE)


# ============================================================================== per-row rules


def test_extract_unambiguous_abcd():
    f = CE.extract_unambiguous_abcd
    assert f("Answer: B") == "B"
    assert f("blah\nanswer:c") == "C"
    assert f("(B) green") == "Z"  # a bare option is not an explicit final answer for the pinned scorer
    assert f("<answer>(D)</answer>") == "D"
    assert f("so \\boxed{\\text{A}}") == "A"
    assert f("Answer: A ... Answer: B") == "Z"  # conflicting explicit candidates stay unresolved
    assert f(float("nan")) == "Z"


def test_puzzle_and_count_rules():
    assert CE.puzzlevqa_hit("Answer: B", ["red", "green", "blue"], "green") == 1
    assert CE.puzzlevqa_hit("(B) green", ["red", "green", "blue"], "green") == 0
    with pytest.raises(CE.ExtError):
        CE.puzzlevqa_hit("Answer: A", ["red"], "green")
    assert CE.visualpuzzles_hit("Answer: d", "D") == 1
    assert CE.countbenchqa_hit("3.0", "3") == 1  # substring rule of CountBenchQA.evaluate
    assert CE.countbenchqa_hit("9.0", "10") == 0


def test_anls_and_chartqapro():
    assert CE.anls_score("hello", ["hello"]) == 1.0
    assert CE.anls_score("abcd", ["wxyz"]) == 0.0
    assert abs(CE.anls_score("hellp", ["hello"]) - 0.8) < 1e-12
    # numeric within 5% relative change, year flags exact, list answers element-wise
    assert CE.chartqapro_score(["100"], "Factoid", ["NO"], "104") == 1.0
    assert CE.chartqapro_score(["100"], "Factoid", ["NO"], "106") == 0.0
    assert CE.chartqapro_score(["2019"], "Factoid", ["YES"], "2020") == 0.0
    assert CE.chartqapro_score(["['a', 'b']"], "Factoid", ["NO", "NO"], "['a', 'c']") == 0.5
    # Conversational keeps only the last year flag
    assert CE.chartqapro_score(["x", "2019"], "Conversational", ["NO", "YES"], "2019") == 1.0


def test_mathvista_post_check():
    mc = {"question_type": "multi_choice", "answer_option": "B", "choices": "['3', '4', '5']", "answer": "4",
          "answer_type": "text", "res": "B"}
    assert CE.mathvista_hit(mc) == 1
    assert CE.mathvista_hit({**mc, "res": "4"}) == 1  # can_infer_text on the choice values
    # can_infer_text gives up on text longer than twice the choices' total length
    assert CE.mathvista_hit({**mc, "res": "the answer is 4"}) == 0
    free = {"question_type": "free_form", "answer_type": "float", "answer": "1.5", "res": "1.50", "choices": None}
    assert CE.mathvista_hit(free) == 1
    assert CE.mathvista_hit({**free, "answer_type": "integer", "answer": "2", "res": "2.5"}) == 0
    # text answers compare str(None) with the answer in the pinned code: never correct
    assert CE.mathvista_hit({**free, "answer_type": "text", "answer": "cat", "res": "cat"}) == 0


def test_mathvision_is_equal_semantics():
    eq = lambda a, b: CE.mathv_is_equal(a, b)[0]  # noqa: E731
    assert eq("60", "60")
    assert eq("0.3333333", "1/3")
    assert eq("2^3", "1")  # Python eval: XOR
    # bare eval inside is_equal sees its locals: the response "a" is the already-evaluated ground truth
    assert eq("a", "17")
    assert not eq("b", "17")
    assert not eq("sqrt(2)", "1.4142135")
    assert not eq("x", "y")
    hit, unresolved = CE.mathvision_hit({"answer": "C", "choices": "['1', '2', '3']", "res": "C"})
    assert hit == 1 and not unresolved


def test_wemath_groups_match_official_formula():
    rows = []
    # 2-step group g1: all right -> CM (1); g2: steps right, composite wrong -> IG (0.5)
    for gid, (s1, s2, m) in {"g1": (1, 1, 1), "g2": (1, 1, 0), "g3": (0, 1, 1), "g4": (0, 0, 0)}.items():
        rows += [{"ID": gid, "key": "2steps_1", "hit": s1}, {"ID": gid, "key": "2steps_2", "hit": s2},
                 {"ID": gid, "key": "2steps_multi", "hit": m}]
    rows += [{"ID": "t1", "key": f"3steps_{i}", "hit": 1} for i in (1, 2, 3)] + [{"ID": "t1", "key": "3steps_multi", "hit": 1}]
    rows += [{"ID": "x", "key": "1steps_1", "hit": 1}]  # one-step rows do not enter the strict score
    vals, counts = CE.wemath_groups(rows, total=5)
    assert vals == {"2:g1": 1.0, "2:g2": 0.5, "2:g3": 0.0, "2:g4": 0.0, "3:t1": 1.0}
    assert counts["CM_strict"] == 2 and counts["IG"] == 1 and counts["RM_strict"] == 1 and counts["IK"] == 1
    # official: (total - 0.5 IG - RM - IK) / total equals the group mean when groups == total
    assert abs(counts["score_strict_x100"] - 100 * np.mean(list(vals.values()))) < 1e-12
    with pytest.raises(CE.ExtError):
        CE.wemath_groups([{"ID": "z", "key": "2steps_1", "hit": 1}, {"ID": "z", "key": "2steps_multi", "hit": 1}])


def test_tablevqa_weighted_score():
    entry = {"values": [1.0, 0.5, 1.0, 0.0, 1.0, 0.0], "strata": ["fintabnetqa", "fintabnetqa", "vtabfact", "vwtq",
                                                                 "vwtq_syn", "vwtq_syn"],
             "weights": dict(CE.TABLEVQA_W)}
    # fintabnet mean 0.75 (weight 0.4), vtabfact 1, vwtq 0, vwtq_syn 0.5 (0.2 each)
    assert abs(CE._reconstructed_score(entry) - 100 * (0.4 * 0.75 + 0.2 * 1 + 0.2 * 0 + 0.2 * 0.5)) < 1e-12


# ============================================================================== statistics


def _bench(key, x, y, strata=None, weights=None):
    if strata is None:
        return CE.BenchArrays(key, {"-": (x, y)}, {"-": 1.0})
    by = {}
    for h in sorted(set(strata)):
        idx = [i for i, s in enumerate(strata) if s == h]
        by[h] = ([x[i] for i in idx], [y[i] for i in idx])
    return CE.BenchArrays(key, by, weights)


def test_bench_boot_constant_difference():
    rng = np.random.default_rng(0)
    b = _bench("k", [1.0] * 50 + [0.0] * 50, [0.0] * 50 + [0.0] * 50)
    mx, my = b.means()
    assert (mx, my) == (50.0, 0.0)
    # resampled differences vary, centred on 50
    reps = b.boot(2000, rng)
    assert abs(reps.mean() - 50) < 2.0 and reps.std() > 1.0
    c = _bench("c", [0.7] * 30, [0.2] * 30)
    assert np.allclose(c.boot(100, rng), 50.0)
    cc, d = c.coef_diff()
    assert abs(float((cc * d).sum()) - 50.0) < 1e-9


def test_signflip_and_macro_groups():
    rng = np.random.default_rng(1)
    n = 400
    strong = _bench("a", list((rng.random(n) < 0.7).astype(float)), list((rng.random(n) < 0.4).astype(float)))
    x = list((rng.random(n) < 0.5).astype(float))
    null = _bench("b", x, x)
    benches = {"a": strong, "b": null}
    groups = {"A": ("a",), "B": ("b",), "AB": ("a", "b")}
    p = CE.signflip_groups(benches, groups, 2000, np.random.default_rng(2))
    assert p["A"] < 0.01
    assert p["B"] == 1.0  # all differences are zero
    assert p["AB"] < 0.01
    st = {"n_boot": 500, "n_perm": 500, "seed": 0}
    out, reps = CE.seed_stats(benches, groups, st, [7])
    assert abs(out["AB"]["delta"] - (out["A"]["delta"] + out["B"]["delta"]) / 2) < 1e-9
    assert out["B"]["ci"] == [0.0, 0.0]
    assert out["A"]["ci"][0] > 0


def test_boot_p_and_ci():
    arr = np.array([1.0, 2.0, 3.0, -1.0])
    assert abs(CE._boot_p(arr) - 2 * (1 + 1) / 5) < 1e-12
    assert CE._boot_p(np.ones(99)) == 2 * 1 / 100


# ============================================================================== end to end (synthetic layout)

CATS = {"Cat1": CE.NATURAL_KEYS[:5], "Cat2": CE.NATURAL_KEYS[5:], "Cat3": CE.SYNTHETIC_KEYS[:8], "Cat4": CE.SYNTHETIC_KEYS[8:]}
CAT_OF = {k: c for c, ks in CATS.items() for k in ks}


def _write_run(root: Path, out: Path, served: str, seed: int, probs: dict, n: int, rng, complete=True):
    d = root / "runs" / served / "full"
    d.mkdir(parents=True, exist_ok=True)
    created = f"2026-10-20T00:00:{seed % 60:02d}+00:00"
    benches, items = [], {}
    for k in CE.ALL_KEYS:
        p = probs.get(k, 0.4)
        v = (rng.random(n) < p).astype(float)
        benches.append({"key": k, "category": CAT_OF[k], "score": 100 * float(v.mean()),
                        "group": "natural" if k in CE.NATURAL_KEYS else "chart_diagram_synthetic",
                        "det_fail_rate": 0.1, "evaluator_exception_scored_incorrect": 0})
        items[k] = {"method": "synthetic", "ids": list(range(n)), "values": v.tolist(), "valid": True, "abs_diff": 0.0,
                    "official": 100 * float(v.mean()), "reconstructed": 100 * float(v.mean()), "n_items": n,
                    "row_digest": f"digest-{k}"}
        if k == "puzzlevqa":
            items[k]["extractor_unresolved_rate"] = 0.5
    if not complete:
        benches = benches[:-1]
    sc = {b["key"]: b["score"] for b in benches}
    macro = {"overall_24": float(np.mean(list(sc.values()))),
             "natural_9": float(np.mean([sc[k] for k in CE.NATURAL_KEYS])),
             "chart_diagram_synthetic_15": float(np.mean([sc[k] for k in CE.SYNTHETIC_KEYS if k in sc])),
             "mmstar": sc["mmstar"], "overall_category_mean": None,
             "categories": {c: float(np.mean([sc[k] for k in ks if k in sc])) for c, ks in CATS.items()}}
    summary = {"served": served, "seed": seed, "complete": complete, "benchmarks_scored": len(benches),
               "created": created, "macro": macro,
               "groups": {"natural": list(CE.NATURAL_KEYS), "chart_diagram_synthetic": list(CE.SYNTHETIC_KEYS),
                          "reported_alone": ["mmstar"]},
               "extraction": {"det_fail_rate": 0.1, "judge_needed_rate_proxy": 0.12 if "p-" in served else 0.10,
                              "cert_json_rate": 0.0, "json_answer_rate": 0.0, "length_cap_rate": 0.0},
               "judge_needed_protocol": {"mathvision": 0.47, "mathvista": 0.26},
               "evaluator_exception_scored_incorrect": {"total": 1, "by_benchmark": {"mme_reasoning": 1}},
               "benchmarks": benches}
    (d / f"summary.seed{seed}.json").write_text(json.dumps(summary))
    if complete:
        CE._write_json(CE._items_path(out, served, seed),
                       {"schema": CE.ITEMS_SCHEMA, "served": served, "seed": seed, "summary_created": created,
                        "benchmarks": items}, gz=True)


@pytest.fixture()
def layout(tmp_path):
    root, out = tmp_path / "ext", tmp_path / "out"
    rng = np.random.default_rng(5)
    n = 300
    nat_up = {k: 0.55 for k in CE.NATURAL_KEYS}  # P better than G on natural benchmarks in every training seed
    for t in (1, 2, 3):
        _write_run(root, out, f"rl3b-p-s{t}-300", 42, nat_up, n, rng)
        _write_run(root, out, f"rl3b-g-s{t}-300", 42, {}, n, rng)
    _write_run(root, out, "rl3b-p-s1-300", 43, nat_up, n, rng)
    _write_run(root, out, "rl3b-g-s1-300", 43, {}, n, rng)
    _write_run(root, out, "rl3b-a-s1-300", 42, {}, n, rng)
    _write_run(root, out, "rl3b-a-s3-300", 42, {}, n, rng, complete=False)
    _write_run(root, out, "cert-sft-3b-s564", 42, {}, n, rng)
    return root, out


def test_discover_lists_pending(layout):
    root, _ = layout
    inv = CE.discover(root)
    pend = {(p["served"], p["seed"]): p["status"] for p in inv["pending"]}
    assert pend[("rl3b-a-s3-300", 42)].startswith("incomplete")
    assert pend[("rl3b-ggated-s1-300", 42)] == "missing"
    assert ("rl3b-p-s1-300", 43) not in pend and ("rl3b-p-s2-300", 43) in pend


def test_analyze_end_to_end(layout):
    cc = CE._load_compare_cert()
    try:
        cc.load_trace_tool(None)
    except cc.CompareError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"trace-tool not available: {exc}")
    root, out = layout
    rc = CE.main(["analyze", "--ext-root", str(root), "--out-dir", str(out), "--n-boot", "300", "--n-perm", "300"])
    assert rc == 0
    doc = json.loads((out / "compare_ext.json").read_text())
    fam = {t["group"]: t for t in doc["families"]["external"]["tests"]}
    nat, syn = fam["natural_9"], fam["chart_diagram_synthetic_15"]
    assert nat["result"]["K"] == 3 and syn["result"]["K"] == 3
    assert nat["result"]["delta"] > 5 and nat["decision"] == "X > Y" and nat["result"]["sign_agree"] == 3
    assert syn["decision"] == "no claim"
    assert nat["holm_p"] <= 2 * nat["result"]["p"] + 1e-12
    pg = next(c for c in doc["comparisons"] if c["name"] == "P - G")
    assert pg["per_seed"][0]["decoding_seeds"] == [42, 43]  # seed 1 pools 42 and 43
    assert pg["per_seed"][1]["decoding_seeds"] == [42]
    # secondary: A has one complete training seed (s3 incomplete) -> K = 1, matched with P@1 / G@1
    pa = next(c for c in doc["comparisons"] if c["name"] == "P - A")
    assert pa["K"] == 1 and pa["training_seed_pairs"] == [["1", "1"]]
    assert next(c for c in doc["comparisons"] if c["name"] == "P7 - G7")["status"] == "pending"
    # the reconstructed paired macros equal the summaries' macros
    assert doc["macro_checks"]["max_abs_diff_points"] < 1e-9
    fs = {(e["served"], e["seed"]): e for e in doc["format_shift"]["runs"]}
    assert fs[("rl3b-p-s1-300", 42)]["abcd_unresolved_puzzlevqa"] == 0.5
    flagged = {(f["comparison"], f["rate"]) for f in doc["format_shift"]["flagged"]}
    assert ("P - G", "judge_needed_rate_proxy") not in flagged  # 2.0 pp is not above the 2 pp threshold
    md = (out / "compare_ext.md").read_text()
    assert "Pre-registered external family" in md and "pending" in md


# ============================================================================== q25 subset, provenance, re-keying


def test_archive_rows_rekey_subset_positions(tmp_path):
    gen = {3: "h3", 7: "h7", 9: "h9"}
    keys = {("3", "h3"): 3, ("7", "h7"): 7, ("9", "h9"): 9}
    p = tmp_path / "score.jsonl"
    # q25 deterministic route: ordinals are positions 0..n-1, source_index and row hash are the true ones
    rows = [{"source_ordinal": i, "source_index": str(o), "source_row_hash": gen[o], "score": float(i)}
            for i, o in enumerate(sorted(gen))]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    stats = {}
    got = [(o, r["score"]) for o, r in CE._archive_rows(str(p), gen, keys, stats)]
    assert got == [(3, 0.0), (7, 1.0), (9, 2.0)]
    assert stats == {"rekeyed": 3, "rekeyed_as_position": True}
    # full-run rows that already carry the generation ordinal pass unchanged
    p.write_text("\n".join(json.dumps({"source_ordinal": o, "source_index": str(o), "source_row_hash": h})
                           for o, h in gen.items()))
    stats = {}
    assert [o for o, _ in CE._archive_rows(str(p), gen, keys, stats)] == [3, 7, 9] and stats == {}
    p.write_text(json.dumps({"source_ordinal": 0, "source_index": "5", "source_row_hash": "zz"}))
    with pytest.raises(CE.ExtError):
        list(CE._archive_rows(str(p), gen, keys, {}))


def test_scorer_counter_locations():
    assert CE._scorer_counter({"artifacts": {"mathverse_all_retries_failed_scored_incorrect": 2}},
                              "mathverse_all_retries_failed_scored_incorrect") == 2
    assert CE._scorer_counter({"evaluation": {"extraction_failed_scored_incorrect": 0}},
                              "extraction_failed_scored_incorrect") == 0
    assert CE._scorer_counter({"score": 1.0}, "extraction_failed_scored_incorrect") is None  # predates fix-3


def _subset(rows_by_bench):
    rows = {k: dict(v) for k, v in rows_by_bench.items()}
    return {"name": "q25", "path": "-", "row_list_sha256": "sha", "rows": rows,
            "digests": {k: CE._row_digest(v.items()) for k, v in rows.items()}}


def test_restrict_to_subset_keeps_whole_wemath_groups():
    sub = _subset({"wemath": {0: "a", 1: "b", 2: "c"}, "erqa": {1: "x", 3: "y"}})
    we = {"values": {"2:g1": (1.0, "-"), "2:g2": (0.5, "-")}, "weights": {"-": 1.0}, "valid": True,
          "group_members": {"2:g1": [0, 1, 2], "2:g2": [3, 4, 5]},
          "subset_digest": {"q25": sub["digests"]["wemath"]}, "subset_rows_match": {"q25": True}}
    r, err = CE.restrict_to_subset(we, "wemath", sub)
    assert err is None and set(r["values"]) == {"2:g1"} and r["n_items"] == 1
    bad = dict(we, group_members={"2:g1": [0, 1, 2], "2:g2": [2, 4, 5]})  # a group cut by the subset
    assert CE.restrict_to_subset(bad, "wemath", sub)[1]
    er = {"values": {str(i): (float(i % 2), "-") for i in range(5)}, "weights": {"-": 1.0}, "valid": True,
          "subset_digest": {"q25": sub["digests"]["erqa"]}, "subset_rows_match": {"q25": True}}
    r, err = CE.restrict_to_subset(er, "erqa", sub)
    assert err is None and set(r["values"]) == {"1", "3"} and CE.bench_score(r) == 100.0
    assert CE.restrict_to_subset(dict(er, subset_digest={"q25": "other"}), "erqa", sub)[1]


def _write_q25_layout(root: Path, out: Path, rng, n_full=40):
    """Subset = first quarter of every benchmark (WeMath: whole groups); full runs for P/G/S and q25 runs at
    step 100 for P/G seed 1 only."""
    nq = n_full // 4
    sub_rows = {}
    for k in CE.ALL_KEYS:
        sub_rows[k] = [{"ordinal": o, "source_row_hash": f"{k}-{o}"} for o in range(nq if k != "wemath" else 3 * 3)]
    subset_doc = {"mode": "q25", "row_list_sha256": "rowsha", "benchmarks": {
        k: {"total": n_full, "selected": len(v), "rows": v} for k, v in sub_rows.items()}}
    (root / "subsets").mkdir(parents=True, exist_ok=True)
    (root / "subsets" / "trace_eval_v1_q25.rows.json").write_text(json.dumps(subset_doc))
    sub = CE.load_subset(root / "subsets" / "trace_eval_v1_q25.rows.json")

    def items_for(subset_run: bool):
        items = {}
        for k in CE.ALL_KEYS:
            if k == "wemath":
                ng = 3 if subset_run else n_full // 3
                ids = [f"2:g{i}" for i in range(ng)]
                vals = list(rng.choice([0.0, 0.5, 1.0], size=ng))
                gm = {f"2:g{i}": [3 * i, 3 * i + 1, 3 * i + 2] for i in range(ng)}
            else:
                n = nq if subset_run else n_full
                ids, vals, gm = list(range(n)), list((rng.random(n) < 0.5).astype(float)), None
            e = {"method": "synthetic", "ids": ids, "values": [float(v) for v in vals], "valid": True, "abs_diff": 0.0,
                 "official": 100 * float(np.mean(vals)), "reconstructed": 100 * float(np.mean(vals)),
                 "n_items": len(ids), "group_members": gm,
                 "row_digest": sub["digests"][k] if subset_run else f"digest-{k}"}
            if subset_run:
                e["subset_exact"] = {"q25": True}
            else:
                e["subset_digest"] = {"q25": sub["digests"][k]}
                e["subset_rows_match"] = {"q25": True}
            items[k] = e
        return items

    def write(served, subset_run):
        d = root / "runs" / served / ("q25" if subset_run else "full")
        d.mkdir(parents=True, exist_ok=True)
        items = items_for(subset_run)
        sc = {k: 100 * float(np.mean(e["values"])) for k, e in items.items()}
        macro = {"overall_24": float(np.mean(list(sc.values()))),
                 "natural_9": float(np.mean([sc[k] for k in CE.NATURAL_KEYS])),
                 "chart_diagram_synthetic_15": float(np.mean([sc[k] for k in CE.SYNTHETIC_KEYS])),
                 "mmstar": sc["mmstar"], "overall_category_mean": None,
                 "categories": {c: float(np.mean([sc[k] for k in ks])) for c, ks in CATS.items()}}
        summary = {"served": served, "seed": 42, "complete": True, "benchmarks_scored": 24, "subset":
                   "q25" if subset_run else "full", "row_list_sha256": "rowsha" if subset_run else None,
                   "created": "2026-10-24T00:00:00+00:00", "macro": macro,
                   "groups": {"natural": list(CE.NATURAL_KEYS), "chart_diagram_synthetic": list(CE.SYNTHETIC_KEYS)},
                   "extraction": {"det_fail_rate": 0.1, "judge_needed_rate_proxy": 0.1, "cert_json_rate": 0.0,
                                  "json_answer_rate": 0.0, "length_cap_rate": 0.0},
                   "judge_needed_protocol": {}, "mathverse_all_retries_failed_scored_incorrect": {"total": 1},
                   "benchmarks": [{"key": k, "category": CAT_OF[k], "score": sc[k],
                                   "group": "natural" if k in CE.NATURAL_KEYS else "chart_diagram_synthetic"}
                                  for k in CE.ALL_KEYS]}
        (d / "summary.seed42.json").write_text(json.dumps(summary))
        CE._write_json(CE._items_path(out, served, 42, "q25" if subset_run else "full"),
                       {"schema": CE.ITEMS_SCHEMA, "served": served, "seed": 42, "summary_created": summary["created"],
                        "benchmarks": items, "provenance": {"score": {"d9319e7db480": 24}},
                        "scorer_counters": {"mathverse": {"mathverse_all_retries_failed_scored_incorrect": 1}}},
                       gz=True)
        return items

    full = {s: write(s, False) for s in ("rl3b-p-s1-300", "rl3b-g-s1-300", "cert-sft-3b-s564")}
    write("rl3b-p-s1-100", True)
    write("rl3b-g-s1-100", True)
    (root / "runs" / "rl3b-g-s1-200" / "q25").mkdir(parents=True)  # in progress: no summary yet
    return sub, full, nq


def test_q25_section_end_to_end(tmp_path):
    cc = CE._load_compare_cert()
    try:
        cc.load_trace_tool(None)
    except cc.CompareError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"trace-tool not available: {exc}")
    root, out = tmp_path / "ext", tmp_path / "out"
    sub, full, nq = _write_q25_layout(root, out, np.random.default_rng(3))
    rc = CE.main(["analyze", "--ext-root", str(root), "--out-dir", str(out), "--n-boot", "200", "--n-perm", "200"])
    assert rc == 0
    doc = json.loads((out / "compare_ext.json").read_text())
    q = doc["transfer_over_training_q25"]
    assert q["problems"] == [] and q["n_items_consistent"]
    pend = {p["served"]: p["status"] for p in q["pending"]}
    assert pend["rl3b-g-s1-200"].startswith("in progress") and pend["rl3b-a-s1-100"] == "missing"
    # step 300 = the full run restricted to the subset rows (WeMath: the first 3 whole groups)
    lv = q["levels"]["rl3b-p-s1-300"]
    for k in ("erqa", "wemath"):
        n = 3 if k == "wemath" else nq
        assert abs(lv["benchmarks"][k] - 100 * float(np.mean(full["rl3b-p-s1-300"][k]["values"][:n]))) < 1e-9
    # q25 runs reproduce their summary macros; P - G at 100 and 300 are paired, step 200 pending
    assert q["q25_runs_max_abs_diff_official_macro"] < 1e-9
    pg = {c["step"]: c for c in q["comparisons"] if c["pair"] == "P - G"}
    assert pg[100]["status"] == "ok" and pg[300]["status"] == "ok" and pg[200]["status"] == "pending"
    d300 = pg[300]["combined"]["natural_9"]["delta"]
    want = np.mean([lv["benchmarks"][k] - q["levels"]["rl3b-g-s1-300"]["benchmarks"][k] for k in CE.NATURAL_KEYS])
    assert abs(d300 - want) < 1e-9
    ps = {c["step"]: c for c in q["comparisons"] if c["pair"] == "P - S"}
    assert ps[100]["status"] == "ok" and ps[100]["training_seed_pairs"] == [["1", ""]]
    prov = {(r["served"], r["subset"]): r for r in doc["provenance"]["runs"]}
    assert prov[("rl3b-p-s1-100", "q25")]["score_code"] == {"d9319e7db480": 24}
    assert prov[("rl3b-p-s1-100", "q25")]["counters_agree"]
    md = (out / "compare_ext.md").read_text()
    assert "Transfer over training on trace_eval_v1_q25" in md and "rl3b-g-s1-200" in md


# ============================================================================== training hardware, decoding-seed shift


def test_training_hardware_and_crossed_pairs(tmp_path):
    def run(arm, t, gpu, n=2, step=300):
        rd = tmp_path / "rl" / "3b" / arm / f"s{t}"
        (rd / "hf" / f"global_step_{step}").mkdir(parents=True, exist_ok=True)
        (rd / "run_receipt.json").write_text(json.dumps({"attempts": [
            {"status": "ok", "resumed_from_step": 0, "last_step": 300,
             "gpu": {"name": gpu, "uuids": [f"u{i}" for i in range(n)]}}]}))
        sp = tmp_path / "runs" / f"rl3b-{arm.lower()}-s{t}-{step}" / "full" / "summary.seed42.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"served": f"rl3b-{arm.lower()}-s{t}-{step}",
                                  "model": str(rd / "hf" / f"global_step_{step}")}))
        return str(sp)

    sps = [run("P", 1, "NVIDIA H200 NVL"), run("G", 1, "NVIDIA H200 NVL"),
           run("P_gated", 1, "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"),
           run("P", 1, "NVIDIA H200 NVL", step=100)]
    base = tmp_path / "runs" / "qwen2.5-vl-3b-instruct" / "full" / "summary.seed42.json"
    base.parent.mkdir(parents=True)
    base.write_text(json.dumps({"served": "qwen2.5-vl-3b-instruct", "model": "/models/qwen"}))
    hw = CE.training_hardware(sps + [str(base)])
    assert hw["rl3b-p-s1-300"] == "NVIDIA H200 NVL x2" == hw["rl3b-p-s1-100"]
    assert hw["rl3b-p_gated-s1-300"].startswith("NVIDIA RTX PRO 6000") and "qwen2.5-vl-3b-instruct" not in hw
    comps = [{"per_seed": [{"tseeds": ["1", "1"], "x_served": "rl3b-p_gated-s1-300", "y_served": "rl3b-p-s1-300"}]},
             {"per_seed": [{"tseeds": ["1", "1"], "x_served": "rl3b-p-s1-300", "y_served": "rl3b-g-s1-300"}]},
             {"per_seed": [{"tseeds": ["1", ""], "x_served": "rl3b-p-s1-300", "y_served": "cert-sft-3b-s564"}]}]
    CE.annotate_hardware(comps, hw)
    assert len(comps[0]["hardware_crossed"]) == 1 and comps[1]["hardware_crossed"] == []
    assert comps[2]["hardware_crossed"] == [] and comps[2]["per_seed"][0]["hardware"][1] is None
    assert "training hardware differs" in CE._hw_note(comps[0]) and CE._hw_note(comps[1]) == ""


def test_decoding_seed_shift():
    def e(served, seed, nat, bench):
        return {"served": served, "seed": seed, "complete": True, "natural_9": nat, "chart_diagram_synthetic_15": 30.0,
                "overall_24": 40.0, "benchmarks": {k: bench.get(k, 50.0) for k in CE.ALL_KEYS}}

    runs = [e("rl3b-p-s1-300", 42, 50.0, {"phyx_mini_mc": 30.0}), e("rl3b-p-s1-300", 43, 51.0, {"phyx_mini_mc": 38.0}),
            e("rl3b-g-s1-300", 42, 49.0, {"phyx_mini_mc": 31.0}), e("rl3b-g-s1-300", 43, 49.5, {"phyx_mini_mc": 37.0}),
            e("cert-sft-3b-s564", 42, 48.0, {})]
    sh = CE.decoding_seed_shift({"runs": runs})
    assert list(sh["seeds"]) == ["43"]
    d = sh["seeds"]["43"]
    assert d["models"] == ["rl3b-g-s1-300", "rl3b-p-s1-300"]  # MODELS order; S has no seed 43
    r = d["rows"]["benchmark:phyx_mini_mc"]
    assert r["mean"] == 7.0 and r["min"] == 6.0 and r["max"] == 8.0 and r["n_up"] == 2 and r["n_down"] == 0
    assert d["rows"]["natural_9"]["mean"] == 0.75 and d["rows"]["chart_diagram_synthetic_15"]["n_up"] == 0
    md = "\n".join(CE._render_seed_shift(sh))
    assert "Decoding-seed shift" in md and "+7.00 [+6.00, +8.00] (0/2)" in md
