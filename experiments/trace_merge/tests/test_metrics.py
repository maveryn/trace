"""Tests for the metric registry and official-metric adapters.

Synthetic fixtures cover every metric kind. Tests that call the pinned VLMEvalKit scorers run only when the
prepared protocol clone is importable; the real-data reproduction test runs only with TRACE_MERGE_WORK_ROOT set.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trace_merge import metrics as M
from trace_merge.config import load_config, sha256_file, write_json

RNG = np.random.default_rng(20261101)
CLONE = os.environ.get("TRACE_MERGE_PROTOCOL_CLONE")


def _protocol_available() -> bool:
    """Whether the pinned scorers can be imported, decided without touching sys.path or the environment."""

    if importlib.util.find_spec("vlmeval") is not None:
        return True
    return bool(CLONE) and (Path(CLONE) / "external/VLMEvalKit/vlmeval").is_dir()


needs_protocol = pytest.mark.skipif(not _protocol_available(), reason="pinned protocol clone not importable")


@pytest.fixture
def protocol(monkeypatch):
    """The clone's import paths and environment for one test; sys.path and the environment are restored after."""

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "1"))
    if CLONE:
        monkeypatch.setenv("LMUData", os.environ.get("LMUData", str(Path(CLONE) / "rlvr/evaluation/.work/LMUData")))
        M.ensure_protocol_imports(Path(CLONE))


def _meta(n: int, **cols: list) -> pd.DataFrame:
    df = pd.DataFrame({"source_ordinal": np.arange(n), "source_index": [f"i{k}" for k in range(n)]})
    for k, v in cols.items():
        df[k] = v
    return df


def _rows(n: int, bench: str | None = None, **cols: list) -> pd.DataFrame:
    df = pd.DataFrame({"source_ordinal": np.arange(n), "source_index": [f"i{k}" for k in range(n)]})
    if bench is not None:
        df.insert(0, "benchmark_id", bench)
    for k, v in cols.items():
        df[k] = v
    return df


def _fake_clone(root: Path, benches: list[str]) -> Path:
    """Stand-in pinned scorer files at the specs' code_refs paths (only their digests matter here)."""

    clone = root / "clone"
    for b in benches:
        for ref in M.SPECS[b].code_refs:
            f = clone / ref.split("::")[0]
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists():
                f.write_text(f"# {f.name}\n")
    return clone


def _accepted(cfg, bench: str, verify: bool = True) -> dict:
    return {"accepted": True, "max_abs_error_pp": 0.0, "slices": [{}] * 9,
            "provenance": M.reproduction_provenance(cfg, bench, verify=verify)}


# --------------------------------------------------------------------------------------------- registry/specs
def test_specs_cover_the_suite_with_expected_kinds():
    cfg = load_config()
    assert set(cfg.benchmark_keys()) == set(M.SPECS)
    assert len(M.BINARY_ROW_BENCHMARKS) == 16 and M.GROUPED_ROW_BENCHMARKS == ("wemath",)
    assert set(M.AGGREGATE_ONLY_BENCHMARKS) == {"chartqapro", "tablevqabench", "mathvision", "mathvista",
                                                "countbenchqa", "puzzlevqa", "visualpuzzles"}
    for spec in M.SPECS.values():
        assert spec.kind in M.KINDS and spec.code_refs
        assert all("::" in r or r.endswith(".py") for r in spec.code_refs)
        if spec.kind != "binary_row":
            assert spec.meta_columns


def test_registry_is_json_and_marks_reproduction(tmp_path):
    cfg = load_config(work_root=tmp_path)
    reg = M.build_metric_registry(cfg)
    json.dumps(reg)
    assert reg["accepted_benchmarks"] == []
    assert len(reg["kinds"]["binary_row"]) == 16 and reg["kinds"]["grouped_row"] == ["wemath"]
    assert not reg["benchmarks"]["tablevqabench"]["row_diagnostics"]["supported"]
    assert reg["benchmarks"]["mathvision"]["row_diagnostics"]["supported"]
    assert all(not k.startswith("/") for e in reg["benchmarks"].values() for k in e["code_sha256"])
    assert "amendment" in reg["benchmarks"]["wemath"]["subset_semantics"]
    fake = {"benchmarks": {"blink": _accepted(cfg, "blink"),
                           "wemath": {"accepted": False, "max_abs_error_pp": 1.0, "slices": [{}] * 9},
                           "mmstar": _accepted(cfg, "mmstar", verify=False),
                           "treebench": {"accepted": True, "max_abs_error_pp": 0.0, "slices": [{}] * 9}}}
    reg = M.build_metric_registry(cfg, fake)
    json.dumps(reg)
    assert reg["accepted_benchmarks"] == ["blink"]
    status = {b: e["reproduction"]["status"] for b, e in reg["benchmarks"].items()}
    assert status["wemath"] == "failed" and status["erqa"] == "not_run"
    assert status["mmstar"] == "unverified" and status["treebench"] == "unverified"    # no part digests / provenance
    assert reg["benchmarks"]["blink"]["reproduction"]["provenance"]["metrics_code_sha256"] == M.metrics_code_sha256()


def test_build_metrics_requires_a_current_verified_reproduction(tmp_path, monkeypatch):
    clone = _fake_clone(tmp_path, ["blink", "erqa", "mmstar", "treebench"])
    cfg = load_config(work_root=tmp_path / "wr", protocol_clone=clone)
    with pytest.raises(M.MetricError, match="reproduce"):
        M.build_metrics(cfg, ["blink"])
    repro = {"benchmarks": {"blink": _accepted(cfg, "blink"), "erqa": {"accepted": False, "slices": []},
                            "mmstar": _accepted(cfg, "mmstar", verify=False)}}
    write_json(tmp_path / "wr" / "manifests" / "metric_registry.json", M.build_metric_registry(cfg, repro))
    out = M.build_metrics(cfg, ["blink", "erqa"])
    assert out["blink"].kind == "binary_row"
    assert out["erqa"].kind == "unsupported"                     # a failed adapter: full-suite reporting only
    with pytest.raises(M.MetricError, match="no accepted adapter"):
        out["erqa"].finalize(np.array([1.0, 2.0]))
    with pytest.raises(M.MetricError, match="no accepted adapter"):
        out["erqa"].campaign_value(np.array([1.0, 2.0]))
    for bench, status in (("mmstar", "unverified"), ("treebench", "not_run")):   # process errors raise
        with pytest.raises(M.MetricError, match=status):
            M.build_metrics(cfg, [bench])
    with pytest.raises(M.MetricError, match="protocol clone"):
        M.build_metrics(dataclasses.replace(cfg, protocol_clone=None), ["blink"])
    # evidence goes stale when the adapter module or a pinned scorer file changes after the reproduction
    with monkeypatch.context() as mp:
        mp.setattr(M, "metrics_code_sha256", lambda: "0" * 64)
        with pytest.raises(M.MetricError, match=r"stale \(metrics.py"):
            M.build_metrics(cfg, ["blink"])
    (clone / M.SPECS["blink"].code_refs[0].split("::")[0]).write_text("# edited\n")
    with pytest.raises(M.MetricError, match="stale.*pinned scorer files"):
        M.build_metrics(cfg, ["blink"])
    assert M.build_metric_registry(cfg, repro)["benchmarks"]["blink"]["reproduction"]["status"] == "stale"


# --------------------------------------------------------------------------------------------- binary_row
def test_binary_row_mean_is_additive_and_scale_invariant():
    n = 240
    s = RNG.integers(0, 2, n).astype(float)
    rows = _rows(n, "blink", archived_row_score=s, official_excluded=False)
    unit = pd.Series(RNG.integers(0, 41, n).astype(str), index=rows.index)
    m = M.get_metric("blink")
    scores = m.row_scores(rows, None)
    assert scores.index.equals(rows.index)
    official = m.official_score(rows, scores)
    assert official == pytest.approx(100 * s.mean(), abs=1e-12)
    ids, stats = m.unit_stats(rows, scores, unit)
    assert stats.shape == (len(set(unit)), 2)
    assert m.finalize(stats.sum(axis=0)) == pytest.approx(official, abs=1e-12)
    assert np.allclose(m.finalize_many(stats[:5]), [m.finalize(t) for t in stats[:5]])
    keep = set(ids[: len(ids) // 3])
    sub = unit.isin(keep).to_numpy()
    sub_total = stats[[i in keep for i in ids]].sum(axis=0)
    assert m.finalize(sub_total) == pytest.approx(100 * s[sub].mean(), abs=1e-12)
    w = RNG.integers(0, 3, len(ids)).astype(float)       # bootstrap-style multiplicities
    weights = w[pd.factorize(unit, sort=True)[0]].astype(int)
    assert m.finalize(w @ stats) == pytest.approx(100 * np.repeat(s, weights).mean())
    assert m.campaign_value(stats.sum(axis=0)) == official


@pytest.mark.parametrize("bad", [{"archived_row_score": [1.0, None]}, {"archived_row_score": [1.0, 0.5]},
                                 {"archived_row_score": [1.0, 0.0], "official_excluded": [False, True]}])
def test_binary_row_fails_closed(bad):
    rows = _rows(2, **bad)
    with pytest.raises(M.MetricError):
        M.get_metric("erqa").row_scores(rows, None)


def test_unit_alignment_and_missing_units_fail():
    rows = _rows(3, archived_row_score=[1.0, 0.0, 1.0])
    m = M.get_metric("erqa")
    scores = m.row_scores(rows)
    with pytest.raises(M.MetricError):
        m.unit_stats(rows, scores, pd.Series(["a", None, "b"], index=rows.index))
    with pytest.raises(M.MetricError):
        m.unit_stats(rows, scores, pd.Series(["a", "b", "c"], index=[5, 6, 7]))
    with pytest.raises(M.MetricError):
        m.finalize(np.array([0.0, 0.0]))


def test_benchmark_macro_differs_from_pooled_accuracy_on_unbalanced_fixture():
    """Each benchmark's value comes from its own unit statistics and finalize; benchmarks then weigh equally."""

    values = {}
    for bench, hits, misses in (("blink", 810, 90), ("erqa", 10, 90)):
        m = M.get_metric(bench)
        r = _rows(hits + misses, bench, archived_row_score=[1.0] * hits + [0.0] * misses)
        unit = pd.Series((r["source_ordinal"] // 7).astype(str), index=r.index)
        _, stats = m.unit_stats(r, m.row_scores(r), unit)
        values[bench] = m.finalize(stats.sum(axis=0))
    assert values == pytest.approx({"blink": 90.0, "erqa": 10.0})
    assert sum(values.values()) / 2 == pytest.approx(50.0)
    assert 100 * (810 + 10) / 1000 == pytest.approx(82.0)        # pooled item accuracy of the same rows


def test_metrics_reject_repeated_or_foreign_rows():
    m = M.get_metric("blink")
    rows = _rows(3, "blink", archived_row_score=[1.0, 0.0, 0.0])
    scores = m.row_scores(rows)
    assert m.official_score(rows, scores) == pytest.approx(100 / 3)
    dup = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)    # would read 50.0
    with pytest.raises(M.MetricError, match="repeated source_ordinal"):
        m.row_scores(dup)
    with pytest.raises(M.MetricError, match="repeated"):
        m.official_score(dup, pd.Series([1.0, 0.0, 0.0, 1.0]))
    with pytest.raises(M.MetricError, match="repeated"):
        m.unit_stats(dup, pd.Series([1.0, 0.0, 0.0, 1.0]), pd.Series(list("abca")))
    with pytest.raises(M.MetricError, match="other benchmarks"):
        m.row_scores(rows.assign(benchmark_id=["blink", "erqa", "blink"]))
    with pytest.raises(M.MetricError, match="several seed"):
        m.row_scores(rows.assign(seed=[42, 43, 42]))
    keyed = rows.drop(columns="source_ordinal").assign(content_key=["a", "b", "a"])
    with pytest.raises(M.MetricError, match="repeated content_key"):
        m.row_scores(keyed)
    cb = M.get_metric("countbenchqa", meta=_meta(3, answer=[1, 2, 3]))
    r = _rows(3, "countbenchqa", extraction_status="resolved", extraction_value_json=["1", "2", "4"])
    assert cb.official_score(r, cb.row_scores(r)) == pytest.approx(200 / 3)
    with pytest.raises(M.MetricError, match="repeated"):                # would read 75.0
        cb.row_scores(pd.concat([r, r.iloc[[0]]], ignore_index=True))


# --------------------------------------------------------------------------------------- split macro (TVQA)
def _tvqa_fixture():
    split = ["fintabnetqa"] * 4 + ["vtabfact"] * 3 + ["vwtq"] * 5 + ["vwtq_syn"] * 3
    primary = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    exact = [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0]
    n = len(split)
    meta = _meta(n, answer=["a"] * n, split=split)

    def components(preds, m, diag):
        return pd.DataFrame({"primary": np.array(primary, float), "exact": np.array(exact, float)})

    rows = _rows(n, extraction_status="resolved", extraction_value_json=json.dumps("x"))
    return M.SplitMacroMetric(M.SPECS["tablevqabench"], meta, components=components), rows, primary, exact


def test_split_macro_reproduces_the_pinned_formula_and_is_additive():
    m, rows, primary, exact = _tvqa_fixture()
    scores = m.row_scores(rows)
    official = m.official_score(rows, scores)

    def acc(c, n):
        return round((c + 1e-9) / (n + 1e-9), 8) * 100

    expected = (acc(3, 4) + acc(2, 4) + acc(2, 3) + acc(3, 5) + acc(1, 3)) / 5
    assert official == expected
    assert official != pytest.approx(100 * np.mean(primary))    # not a pooled row mean
    unit = pd.Series(["u%d" % (k % 4) for k in range(len(rows))], index=rows.index)
    _, stats = m.unit_stats(rows, scores, unit)
    assert m.finalize(stats.sum(axis=0)) == official
    assert m.finalize_many(stats.sum(axis=0)[None, :])[0] == official


def test_split_macro_missing_split_and_bad_components_fail():
    m, rows, _, _ = _tvqa_fixture()
    scores = m.row_scores(rows)
    keep = (m.aligned_meta(rows)["split"] != "vtabfact").to_numpy()
    with pytest.raises(M.MetricError, match="absent"):
        m.official_score(rows[keep], scores[keep])
    with pytest.raises(M.MetricError):
        m.row_contributions(rows, scores.where(scores.index != 5, 0.5))   # vtabfact row cannot be 0.5


# ------------------------------------------------------------------------------------------------- WeMath
def _wemath_fixture(outcomes2: list[tuple], outcomes3: list[tuple]):
    ids, keys, hits = [], [], []
    for g, (s1, s2, multi) in enumerate(outcomes2):
        for k, h in zip(M.WEMATH_KEYS["2steps"], (s1, s2, multi)):
            ids.append(f"2stpes_{g}"), keys.append(k), hits.append(float(h))
    for g, (s1, s2, s3, multi) in enumerate(outcomes3):
        for k, h in zip(M.WEMATH_KEYS["3steps"], (s1, s2, s3, multi)):
            ids.append(f"3stpes_{g}"), keys.append(k), hits.append(float(h))
    n = len(ids)
    perm = RNG.permutation(n)                     # row order must not matter
    meta = _meta(n, ID=[ids[i] for i in perm], key=[keys[i] for i in perm])
    rows = _rows(n, archived_row_score=[hits[i] for i in perm])
    return M.get_metric("wemath", meta=meta), rows


def test_wemath_strict_counts_and_formula():
    # 2-step: CM, IG, RM, IK ; 3-step: CM, IK
    m, rows = _wemath_fixture([(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 0, 0)], [(1, 1, 1, 1), (1, 0, 1, 0)])
    scores = m.row_scores(rows)
    totals = m.row_contributions(rows, scores).sum(axis=0)
    assert list(totals) == [6, 2, 1, 1, 2]
    g, cm, ig, rm, ik = totals
    assert m.finalize_unrounded(totals) == pytest.approx(100 * (cm + 0.5 * ig) / g)
    # G != 525: the groups-present value, without the pinned two-decimal display rounding
    assert m.finalize(totals) == m.finalize_unrounded(totals) == pytest.approx(100 * 2.5 / 6)
    assert m.official_score(rows, scores) == pytest.approx(41.6666667)
    # the pinned fixed denominator counts the 519 absent groups as mastered; that is the campaign's subset value
    assert m.pinned_fixed525_score(totals) == float("{:.2%}".format((525 - 0.5 - 1 - 2) / 525)[:-1]) == 99.33
    assert m.campaign_value(totals) == 99.33
    assert m.official_score(rows, scores) != pytest.approx(100 * scores.mean())


def test_wemath_full_split_keeps_the_pinned_rounding():
    full = np.array([525.0, 200.0, 41.0, 100.0, 184.0])
    assert M.WeMathStrictMetric(M.SPECS["wemath"]).finalize(full) == float(
        "{:.2%}".format((525 - 0.5 * 41 - 100 - 184) / 525)[:-1]) == 42.0
    m = M.WeMathStrictMetric(M.SPECS["wemath"])
    for totals in (np.array([525.0, 237.0, 3.0, 140.0, 145.0]), np.array([524.0, 237.0, 3.0, 140.0, 144.0])):
        g, cm, ig, rm, ik = totals
        raw = 100 * (cm + 0.5 * ig) / g
        assert m.finalize_unrounded(totals) == pytest.approx(raw)
        expected = round(raw, 2) if g == 525 else raw
        assert m.finalize(totals) == pytest.approx(expected, abs=1e-12)
        assert m.campaign_value(totals) == float("{:.2%}".format((525 - 0.5 * ig - rm - ik) / 525)[:-1])
    assert np.allclose(m.finalize_many(np.array([[525.0, 237.0, 3.0, 140.0, 145.0], [6.0, 2.0, 1.0, 1.0, 2.0]])),
                       [45.43, 100 * 2.5 / 6])


def test_wemath_subset_scale_and_units():
    m, rows = _wemath_fixture([(1, 1, 1)] * 6 + [(0, 0, 0)] * 2 + [(1, 1, 0), (0, 1, 1)], [(1, 1, 1, 0)] * 2)
    scores = m.row_scores(rows)
    meta = m.aligned_meta(rows)
    unit = meta["ID"].map(lambda s: "U" + s[-1])          # whole groups per unit
    ids, stats = m.unit_stats(rows, scores, unit)
    assert m.finalize(stats.sum(axis=0)) == m.official_score(rows, scores)
    mastered = rows.index[meta["ID"].isin(["2stpes_0", "2stpes_1"]).to_numpy()]
    assert m.official_score(rows.loc[mastered], scores.loc[mastered]) == 100.0
    # a mixed subset: the groups-present value, which the pinned fixed-525 value is not
    mixed = rows.index[meta["ID"].isin(["2stpes_0", "2stpes_6", "2stpes_8", "2stpes_9", "3stpes_0"]).to_numpy()]
    totals = m.row_contributions(rows.loc[mixed], scores.loc[mixed]).sum(axis=0)
    assert list(totals) == [5, 1, 2, 1, 1]
    assert m.official_score(rows.loc[mixed], scores.loc[mixed]) == pytest.approx(100 * (1 + 0.5 * 2) / 5)
    assert m.campaign_value(totals) == float("{:.2%}".format((525 - 0.5 * 2 - 1 - 1) / 525)[:-1])
    assert m.campaign_value(totals) != pytest.approx(m.finalize(totals))
    bootstrap = np.array([2.0, 0.0, 1.0, 3.0]) @ stats[[list(ids).index(u) for u in ("U0", "U6", "U8", "U9")]]
    assert m.finalize(bootstrap) == pytest.approx(m.finalize_unrounded(bootstrap))
    bad_unit = pd.Series(np.where(rows.index % 2 == 0, "a", "b"), index=rows.index)
    with pytest.raises(M.MetricError, match="span several units"):
        m.unit_stats(rows, scores, bad_unit)
    partial = rows.index[(meta["key"] != "2steps_2").to_numpy()]
    with pytest.raises(M.MetricError, match="incomplete"):
        m.official_score(rows.loc[partial], scores.loc[partial])


# ------------------------------------------------------------------------------------------------- views
def test_cell_views_follow_the_pinned_io():
    rows = pd.DataFrame({
        "extraction_status": ["resolved", "invalid", "resolved", "resolved", "resolved"],
        "extraction_value_json": [json.dumps("B"), "null", json.dumps("None"), json.dumps(10), json.dumps("60")],
        "judge_output": [None, None, json.dumps(["None"]), json.dumps(["10"]), json.dumps(["x", "  "])],
    })
    off = M.official_route_view(rows)
    assert off[0] == "B" and math.isnan(off[1]) and off[2] == "None" and off[3] == 10
    diag: dict = {}
    res = M.judged_res_view(rows, diag)
    assert res[0] == "B" and math.isnan(res[1]) and math.isnan(res[2]) and res[3] == 10 and math.isnan(res[4])
    assert diag == {"invalid_extraction_as_nan": 1, "na_string_res_as_nan": 1, "blank_judge_res_as_nan": 1}
    assert M.scored_prediction_view(rows) == ["B", "", "None", "10", "60"]
    long = pd.DataFrame({"extraction_status": "resolved",
                         "extraction_value_json": [json.dumps("a" * M.XLSX_MAX_STRING), json.dumps("a_x000C_b")]})
    diag = {}
    M.scored_prediction_view(long, diag)
    assert diag == {"xlsx_max_length_prediction": 1, "xlsx_escape_in_prediction": 1}


def test_xlsx_cell_text_matches_writer_round_trip():
    assert M.xlsx_cell_text("\x0crac{36}{13}") == "_x000C_rac{36}{13}"
    assert M.xlsx_cell_text(7) == 7 and M.xlsx_cell_text("plain") == "plain"
    assert len(M.xlsx_cell_text("a" * 40000)) == M.XLSX_MAX_STRING
    with pytest.raises(M.MetricError):
        M.xlsx_cell_text("=1+2")
    if importlib.util.find_spec("xlsxwriter") is None:
        pytest.skip("xlsxwriter not importable")
    values = ["x\x0cy", "_x000C_", "tab\tok", "a\x01b", "plain"]
    buf = io.BytesIO()
    pd.DataFrame({"a": values}).to_excel(buf, index=False, engine="xlsxwriter")
    buf.seek(0)
    assert pd.read_excel(buf)["a"].tolist() == [M.xlsx_cell_text(v) for v in values]


def test_metadata_binding_fails_closed():
    meta = _meta(3, answer=["1", "2", "3"])
    meta["source_record_sha256"] = ["a", "b", "c"]
    m = M.get_metric("countbenchqa", meta=meta)
    ok = _rows(2, source_record_sha256=["a", "b"])
    assert m.aligned_meta(ok)["answer"].tolist() == ["1", "2"]
    with pytest.raises(M.MetricError, match="source_index"):
        m.aligned_meta(ok.assign(source_index=["i0", "zz"]))
    with pytest.raises(M.MetricError, match="no metadata"):
        m.aligned_meta(_rows(4))
    with pytest.raises(M.MetricError, match="record"):
        m.aligned_meta(ok.assign(source_record_sha256=["a", "x"]))
    hashed = M.get_metric("countbenchqa", meta=meta.assign(source_row_sha256=["r0", "r1", "r2"]))
    spool_like = _rows(2, source_row_sha256=["r0", "r1"])     # a local run's spool carries the row hash only
    assert hashed.aligned_meta(spool_like)["answer"].tolist() == ["1", "2"]
    with pytest.raises(M.MetricError, match="source_row_sha256"):
        hashed.aligned_meta(spool_like.assign(source_row_sha256=["r0", "r2"]))
    with pytest.raises(M.MetricError, match="not bound"):
        M.get_metric("countbenchqa").row_scores(_rows(1, extraction_status="resolved", extraction_value_json="1"))


# ----------------------------------------------------------------------------------- aggregate-only adapters
def test_aggregate_only_adapter_on_a_tiny_fabricated_case():
    """CountBenchQA's pinned rule needs no judge or VLMEvalKit import: str(answer) in str(prediction)."""

    meta = _meta(5, answer=[3, 12, 2, 7, 5])
    rows = _rows(5, extraction_status=["resolved", "resolved", "resolved", "invalid", "resolved"],
                 extraction_value_json=["3", json.dumps("12 apples"), "12", "null", json.dumps("five")])
    m = M.get_metric("countbenchqa", meta=meta)
    scores = m.row_scores(rows)
    assert scores.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]     # "2" in "12" counts, as in the pinned rule
    assert m.official_score(rows, scores) == pytest.approx(60.0)
    assert m.last_diagnostics == {"invalid_extraction_as_nan": 1}
    _, stats = m.unit_stats(rows, scores, pd.Series(list("aabbc"), index=rows.index))
    assert m.finalize(stats.sum(axis=0)) == pytest.approx(60.0)


def test_rederived_metric_rejects_out_of_range_scorer_output():
    meta = _meta(2, answer=["a", "b"])
    rows = _rows(2, extraction_status="resolved", extraction_value_json=json.dumps("a"))
    m = M.RederivedPooledMetric(M.SPECS["countbenchqa"], meta, scorer=lambda p, mm, d: np.array([1.0, 2.0]))
    with pytest.raises(M.MetricError, match=r"\[0, 1\]"):
        m.row_scores(rows)


def test_spool_reader_loads_only_scoring_stages_and_checks_rows(tmp_path):
    ext = [{"source_ordinal": i, "source_index": f"i{i}", "source_row_hash": f"h{i}", "judge_response": None,
            "normalized_extraction": {"status": "RESOLVED", "value": str(i)}} for i in range(3)]
    score = [{"source_ordinal": i, "source_index": f"i{i}", "score": s, "excluded": False}
             for i, s in enumerate((True, "incorrect", 1.0))]
    score.append({"source_ordinal": -1, "source_index": "__aggregate__", "score": 66.7})
    spool = tmp_path / "run" / "hf_archive"
    (spool / "ready").mkdir(parents=True)
    (spool / "payloads").mkdir()
    for stage, recs in (("generation", [{"response": "x" * 100}]), ("extraction", ext), ("score", score)):
        payload = spool / "payloads" / f"{stage}.jsonl"
        payload.write_text("".join(json.dumps(r) + "\n" for r in recs))
        desc = {"identity": {"seed": 42, "stage": stage, "benchmark": "erqa"}, "payload_path": f"payloads/{stage}.jsonl",
                "payload_sha256": sha256_file(payload)}
        (spool / "ready" / f"{stage}.ready.json").write_text(json.dumps(desc))
    index = M._spool_index(tmp_path / "run", 42)
    assert set(index) == {("erqa", "extraction"), ("erqa", "score")}      # generation payloads never parsed
    assert M._spool_index(tmp_path / "run", 43) == {}
    records = M._spool_records(index, "erqa")
    rows = M.spool_slice_rows(records, "erqa")
    assert rows["archived_row_score"].tolist() == [1.0, 0.0, 1.0]
    assert set(rows["benchmark_id"]) == {"erqa"} and rows["source_row_sha256"].tolist() == ["h0", "h1", "h2"]
    assert M.get_metric("erqa").official_score(rows, M.get_metric("erqa").row_scores(rows)) == pytest.approx(200 / 3)
    with pytest.raises(M.MetricError, match="repeats"):
        M.spool_slice_rows({**records, ("erqa", "score"): score + [score[0]]}, "erqa")
    with pytest.raises(M.MetricError, match="lacks"):
        M.spool_slice_rows({**records, ("erqa", "score"): score + [{"source_ordinal": 9, "source_index": "i9",
                                                                     "score": 1}]}, "erqa")
    (spool / "payloads" / "score.jsonl").write_text("tampered\n")
    with pytest.raises(M.MetricError, match="digest"):
        M._spool_records(index, "erqa")


def test_ensure_protocol_imports_puts_pinned_dependencies_first(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    for p in M.protocol_paths(clone):
        p.mkdir(parents=True)
    deps = M.protocol_paths(clone)[0]
    (deps / "tm_fake_dep").mkdir()
    (deps / "tm_fake_dep" / "__init__.py").write_text("__version__ = '2.0'\n")
    info = deps / "tm_fake_dep-2.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: tm-fake-dep\nVersion: 2.0\n")
    (info / "RECORD").write_text("tm_fake_dep/__init__.py,,\ntm_fake_dep-2.0.dist-info/METADATA,,\n")
    site = str(tmp_path / "env" / "site-packages")
    monkeypatch.setattr(sys, "path", ["", site])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("LMUData", "unused")
    M.ensure_protocol_imports(clone)
    expected = ["", *[str(p) for p in M.protocol_paths(clone)], site]   # where the documented PYTHONPATH puts them
    assert sys.path == expected
    M.ensure_protocol_imports(clone)
    assert sys.path == expected                                          # idempotent
    sys.path[:] = ["", site, *[str(p) for p in M.protocol_paths(clone)]]   # appended after site-packages: moved
    M.ensure_protocol_imports(clone)
    assert sys.path == expected
    stale = types.ModuleType("tm_fake_dep")
    stale.__file__ = str(tmp_path / "env" / "site-packages" / "tm_fake_dep" / "__init__.py")
    stale.__version__ = "1.0"
    monkeypatch.setitem(sys.modules, "tm_fake_dep", stale)
    with pytest.raises(M.MetricError, match=r"tm_fake_dep 1.0 \(pinned 2.0\)"):
        M.ensure_protocol_imports(clone)
    stale.__version__ = "2.0"                                            # same version from elsewhere is fine
    M.ensure_protocol_imports(clone)


@needs_protocol
def test_pinned_scorers_on_fabricated_rows(protocol):
    puz = M.get_metric("puzzlevqa", meta=_meta(3, options=["['red', 'blue', 'green', 'pink']"] * 3,
                                               answer=["blue", "green", "red"], category=["c"] * 3))
    rows = _rows(3, extraction_status=["resolved", "resolved", "invalid"],
                 extraction_value_json=[json.dumps("so... Answer: B"), json.dumps("Answer: A"), "null"])
    assert puz.row_scores(rows).tolist() == [1.0, 0.0, 0.0]

    vp = M.get_metric("visualpuzzles", meta=_meta(2, answer=["c", "D"], category=["x"] * 2, difficulty=["easy"] * 2))
    rows = _rows(2, extraction_status="resolved", extraction_value_json=[json.dumps("\\boxed{C}"), json.dumps("A")])
    assert vp.row_scores(rows).tolist() == [1.0, 0.0]

    # MathVision: a 'None' judge output is NaN for the scorer, and can_infer(NaN) matches option text 'A'
    mv = M.get_metric("mathvision", meta=_meta(3, answer=["A", "61", "B"],
                                               choices=["['A', 'B', 'C']", "[]", "['A', 'B']"],
                                               category=["c"] * 3))
    rows = _rows(3, extraction_status="resolved", extraction_value_json=[json.dumps("None"), json.dumps("61"),
                                                                           json.dumps("A")],
                 judge_output=[json.dumps(["None"]), json.dumps(["61"]), None])
    assert mv.row_scores(rows).tolist() == [1.0, 1.0, 0.0]
    assert mv.last_diagnostics.get("na_string_res_as_nan") == 1

    cq = M.get_metric("chartqapro", meta=_meta(2, answer=["['100']", "['Paris']"], question_type=["Factoid"] * 2,
                                               year=["['NO']", "['NO']"]))
    rows = _rows(2, extraction_status="resolved", extraction_value_json=[json.dumps("103"), json.dumps("Paris")])
    assert cq.row_scores(rows).tolist() == [1.0, 1.0]       # 3% relative error is within the 5% tolerance

    split = ["fintabnetqa", "vtabfact", "vwtq", "vwtq_syn"]
    tv = M.get_metric("tablevqabench", meta=_meta(4, answer=["$3,852 million", "1", "2004", "Apple|Banana"],
                                                  split=split))
    rows = _rows(4, extraction_status="resolved",
                 extraction_value_json=[json.dumps("3852"), json.dumps("True"), json.dumps("2004"),
                                        json.dumps("Apple||Banana")])
    s = tv.row_scores(rows)
    assert s.tolist() == [0.5, 1.0, 1.0, 1.0]               # fintabnetqa: relieved but not exact
    assert tv.official_score(rows, s) == pytest.approx((100 + 0 + 100 + 100 + 100) / 5)


@needs_protocol
def test_wemath_matches_the_pinned_implementation_on_a_full_size_fixture(tmp_path, protocol):
    wemath = M._pinned("vlmeval.dataset.utils.wemath")
    out2 = [tuple(RNG.integers(0, 2, 3)) for _ in range(360)]
    out3 = [tuple(RNG.integers(0, 2, 4)) for _ in range(165)]
    m, rows = _wemath_fixture(out2, out3)
    meta = m.aligned_meta(rows)
    book = tmp_path / "wemath.xlsx"
    pd.DataFrame({"ID": meta["ID"], "key": meta["key"], "hit": rows["archived_row_score"].astype(int)}).to_excel(
        book, index=False)
    with M._quiet():
        pinned = wemath.wemath_evaluate_models(str(book))
    value = float(str(pinned["Score (Strict)"][0]).rstrip("%"))
    assert m.official_score(rows, m.row_scores(rows)) == value


# ---------------------------------------------------------------------------------------------- real data
@pytest.mark.skipif(not os.environ.get("TRACE_MERGE_WORK_ROOT"), reason="TRACE_MERGE_WORK_ROOT not set")
def test_real_archive_reproduction(protocol):
    cfg = load_config()
    try:
        cfg.archive_run_root
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"archive snapshot unavailable: {error}")
    if not (cfg.archive_run_root / "metadata" / "manifest.json").is_file():
        pytest.skip("archive snapshot unavailable")
    M._verify_archive_manifest(cfg)
    results = M._results_json_index(cfg)
    benches = ["erqa", "treebench"]
    if cfg.protocol_clone is not None and cfg.dataset_manifest.is_file():
        benches.append("wemath")
        if _protocol_available():
            benches.append("countbenchqa")
    for bench in benches:
        r = M.reproduce_benchmark(cfg, bench, results=results)
        assert len(r["slices"]) == 9
        assert r["max_abs_error_pp"] <= M.REPRODUCTION_TOLERANCE_PP, bench
        assert r["results_json_max_abs_error_pp"] == 0.0
        assert r["accepted"] and r["provenance"]["part_digests_verified"]
        if bench == "wemath":   # full split: the study's value, the pinned value and the campaign value coincide
            assert all(s["group_totals"]["groups"] == M.WEMATH_PINNED_GROUPS for s in r["slices"])
            assert all(s["campaign_abs_error_pp"] == 0.0 for s in r["slices"])
    audit = cfg.work("audit", "metric_reproduction.json")
    if audit.is_file():
        data = json.loads(audit.read_text(encoding="utf-8"))
        assert len(data["summary"]["accepted"]) == 24 and not data["summary"]["not_accepted"]
        registry = M.build_metric_registry(cfg, data)
        stale = [b for b, e in registry["benchmarks"].items() if e["reproduction"]["status"] != "accepted"]
        assert not stale, f"audit is not current for {stale}; rerun `python -m trace_merge.metrics reproduce`"
