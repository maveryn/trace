"""Analysis orchestration tests: partitions, binary rows, row loading, capture, the lock check, screen and confirm.

Everything runs on a synthetic work root under pytest's tmp_path (model outcomes are drawn so that every model differs
from every other); the last tests read a real work root and skip without TRACE_MERGE_WORK_ROOT.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_merge import analysis as an  # noqa: E402
from trace_merge import cli, config, grouping, reports  # noqa: E402
from trace_merge import evaluation as ev  # noqa: E402
from trace_merge import ingest as ig  # noqa: E402
from trace_merge import metrics as mt  # noqa: E402
from trace_merge import statistics as st  # noqa: E402

ROWS_PER_BENCH = 40   # dev screen 14 singleton components per benchmark, so no bootstrap draw misses one
NOT_BINARY = ("chartqapro", "tablevqabench", "wemath")
CONFIRMED = ("V@P1", "M050@P1", "CV050@P1", "CT050@P1")


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("TVM_RUNS", raising=False)
    c = config.load_config(work_root=tmp_path / "work", protocol_clone=tmp_path / "clone")
    c.raw["statistics"]["bootstrap_replicates"] = 200          # small, deterministic
    return c


def _keys(cfg):
    return {b: [ig.content_key(b, i, str(i), f"row{b}{i}", f"rec{b}{i}") for i in range(ROWS_PER_BENCH)]
            for b in cfg.benchmark_keys()}


def _ga(cfg) -> pd.DataFrame:
    recs = []
    for b, keys in _keys(cfg).items():
        for i, k in enumerate(keys):
            comp = f"{b}:{i}" if i not in (17, 39) else f"{b}:{i - 1}"     # two 2-row components
            recs.append({"benchmark_id": b, "content_key": k, "component_id": comp,
                         "split_role": "development" if i < 16 else "confirmation",
                         "in_dev_screen": i < 14, "in_smoke": i == 0})
    return pd.DataFrame(recs)


def _rows(cfg, label: str, seed: int, *, source: str = "archive", run_id: str = "arch", keep=None,
          always: float | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(zlib.crc32(f"{label}|{seed}".encode()))
    recs = []
    for b, keys in _keys(cfg).items():
        for i, k in enumerate(keys):
            if keep is not None and k not in keep:
                continue
            score = float(always) if always is not None else float(rng.integers(0, 2))
            recs.append({"model_label": label, "model_role": label.split("@")[0], "protocol_id": "P0_native",
                         "source": source, "run_id": run_id, "seed": seed, "benchmark_id": b, "content_key": k,
                         "source_ordinal": i, "source_index": str(i), "source_row_sha256": f"row{b}{i}",
                         "source_record_sha256": f"rec{b}{i}", "archived_row_score": score,
                         "official_excluded": False, "extraction_status": "resolved", "extraction_value_json": "1",
                         "judge_output": None})
    return pd.DataFrame(recs)


def _write_splits(cfg) -> None:
    if cfg.work("splits", "split_manifest.json").is_file():
        return                                               # frozen once, like the real split
    ga_path = cfg.work("splits", "group_assignments.parquet")
    ga_path.parent.mkdir(parents=True, exist_ok=True)
    _ga(cfg).to_parquet(ga_path, index=False)
    config.write_json(cfg.work("splits", "split_manifest.json"),
                      {"rule": grouping.split_rule(cfg.raw["split"]),
                       "group_assignments_sha256": config.sha256_file(ga_path),
                       "row_lists": {"confirmation": {"row_list_sha256": "a" * 64}}})
    # amendment A7 bootstrap units; no twins in the synthetic world, so units equal components
    twins = cfg.work("splits", "_twins.csv")
    pd.DataFrame(columns=list(grouping.TWIN_COLUMNS)).to_csv(twins, index=False)
    grouping.freeze_bootstrap_units(cfg, twins, log=lambda *_: None)


def _write_work(cfg, archive: pd.DataFrame, local: dict[str, pd.DataFrame]) -> None:
    root = cfg.require_work_root()
    (root / "aligned" / "local_runs").mkdir(parents=True, exist_ok=True)
    archive.to_parquet(root / "aligned" / "predictions.parquet", index=False)
    index = {"local_runs": {}}
    for name, df in local.items():
        df.to_parquet(root / "aligned" / "local_runs" / f"{name}.parquet", index=False)
        index["local_runs"][name] = {"path": f"local_runs/{name}.parquet", "rows": len(df)}
    (root / "aligned" / "index.json").write_text(json.dumps(index))
    _write_splits(cfg)


def _slug(label: str) -> str:
    return ev.VIEW_SPECS[label][0]


def _validate_run(cfg, label: str, subset: str, seed: int, complete: bool = True) -> Path:
    """A run directory with a validation file and one spool descriptor of its seed (the tier's seeds share it)."""

    rd = ev.run_dir(cfg, _slug(label), subset)
    config.write_json(rd / f"validation.seed{seed}.json", {"complete": complete, "label": label})
    config.write_json(rd / "hf_archive" / "ready" / f"{seed}.ready.json", {"identity": {"seed": seed}})
    return rd


def _register(cfg, name: str, *, runtime_revision: str, fingerprint: str | None) -> None:
    index_path = cfg.work("aligned", "index.json")
    index = json.loads(index_path.read_text())
    config.write_json(cfg.work("aligned", "local_runs", f"{name}.report.json"),
                      {"runtime_revision": runtime_revision, "source_fingerprint": fingerprint})
    index["local_runs"][name]["report"] = f"local_runs/{name}.report.json"
    index_path.write_text(json.dumps(index))


def _ingest_tier_fixture(cfg, tier: str) -> None:
    """Validate and register every run of a tier as freshly ingested, served at its view's registered revision."""

    for label, slug, subset, seed in ev.tier_runs(cfg, tier):
        rd = _validate_run(cfg, label, subset, seed)
        _register(cfg, an.local_run_name(label, subset, seed), runtime_revision=f"r:{slug}",
                  fingerprint=an.run_fingerprint(rd, seed))


def _metrics(cfg):
    return {b: mt.BinaryRowMetric(mt.SPECS[b]) for b in cfg.benchmark_keys()}


def _binary(cfg):
    return [b for b in cfg.benchmark_keys() if b not in NOT_BINARY]


def _lock(cfg) -> dict:
    """Every input the lock reads, then the lock itself (before any confirmation output exists)."""

    _write_splits(cfg)
    config.write_json(cfg.work("manifests", "metric_registry.json"), {"accepted_benchmarks": cfg.benchmark_keys()})
    for mid in cfg.merges:
        config.write_json(cfg.work("merges", mid, "merge_manifest.json"),
                          {"content_revision": f"sha256set:{mid}", "recipe_sha256": mid})
    for _, slug, _, _ in ev.tier_runs(cfg, "confirmation"):
        config.write_json(cfg.work("views", slug, "view_manifest.json"), {"slug": slug})
        config.write_json(cfg.work("views", slug, ".trace_model_revision.json"), {"immutable_revision": f"r:{slug}"})
    return ev.lock_selection(cfg, config.PROJECT_ROOT / "DESIGN.md")


def _confirmation_work(cfg) -> None:
    _lock(cfg)
    arch = pd.concat([_rows(cfg, l, s) for l in an.ARCHIVE_LABELS for s in (42, 43, 44)], ignore_index=True)
    conf = set(_ga(cfg).loc[lambda g: g.split_role == "confirmation", "content_key"])
    local = {f"{l}.confirmation.s{s}": _rows(cfg, l, s, source="local_run", run_id=l, keep=conf)
             for l in CONFIRMED for s in (42, 43, 44)}
    local["B_seed42_existing_local_rerun"] = _rows(cfg, "B", 42, source="existing_local_rerun", always=1.0)
    local["T_seed42_existing_local_rerun"] = _rows(cfg, "T", 42, source="existing_local_rerun", always=0.0)
    _write_work(cfg, arch, local)
    _ingest_tier_fixture(cfg, "confirmation")
    for tier in ("smoke", "development"):
        for label, _, subset, seed in ev.tier_runs(cfg, tier):
            _validate_run(cfg, label, subset, seed)


def _strict_json(path: Path):
    def refuse(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


# ---------------------------------------------------------------------------------------------------- restrict
def test_restrict_selects_exactly_the_frozen_partitions(cfg):
    ga = _ga(cfg)
    rows = _rows(cfg, "B", 42)
    expect = {"all": set(ga.content_key),
              "development": set(ga.loc[ga.split_role == "development", "content_key"]),
              "confirmation": set(ga.loc[ga.split_role == "confirmation", "content_key"]),
              "development_screen": set(ga.loc[ga.in_dev_screen, "content_key"]),
              "smoke": set(ga.loc[ga.in_smoke, "content_key"])}
    for part, keys in expect.items():
        got = an.restrict(rows, ga, part)
        assert set(got.content_key) == keys, part
        assert got.index.equals(pd.RangeIndex(len(got)))
    assert expect["development"].isdisjoint(expect["confirmation"])
    assert expect["development_screen"] <= expect["development"] and expect["smoke"] <= expect["development_screen"]


def test_restrict_fails_on_unknown_partition_and_foreign_keys(cfg):
    ga = _ga(cfg)
    rows = _rows(cfg, "B", 42)
    with pytest.raises(an.AnalysisError, match="unknown partition"):
        an.restrict(rows, ga, "dev")
    bad = pd.concat([rows, rows.iloc[[0]].assign(content_key="not-in-split")], ignore_index=True)
    with pytest.raises(an.AnalysisError, match="absent from the frozen split"):
        an.restrict(bad, ga, "confirmation")


def test_assignments_refuse_a_parquet_that_differs_from_the_split_manifest(cfg):
    _write_splits(cfg)
    assert len(an.assignments(cfg)) == len(_ga(cfg))
    _ga(cfg).iloc[:-1].to_parquet(cfg.work("splits", "group_assignments.parquet"), index=False)
    with pytest.raises(an.AnalysisError, match="group_assignments_sha256"):
        an.assignments(cfg)


# ---------------------------------------------------------------------------------------------------- with_binary
class _SliceCheckingMetric:
    kind = "binary_row"

    def __init__(self, bench, values=None):
        self.bench, self.values, self.calls = bench, values, []

    def row_scores(self, rows, meta=None):
        assert rows["benchmark_id"].nunique() == 1 and rows["model_label"].nunique() == 1 and rows["seed"].nunique() == 1
        self.calls.append(len(rows))
        if self.values is not None:
            return pd.Series(self.values, index=rows.index)
        return rows["archived_row_score"].astype(float)


def test_with_binary_scores_only_binary_benchmarks_one_slice_at_a_time(cfg):
    rows = pd.concat([_rows(cfg, "B", 42), _rows(cfg, "T", 42), _rows(cfg, "B", 43)], ignore_index=True)
    binary = _binary(cfg)
    metrics = {b: _SliceCheckingMetric(b) for b in cfg.benchmark_keys()}
    out = an.with_binary(rows, metrics, binary)
    assert len(binary) == 21
    inside = out["benchmark_id"].isin(binary)
    assert out.loc[inside, "binary_correct"].notna().all()
    assert out.loc[~inside, "binary_correct"].isna().all()
    assert (out.loc[inside, "binary_correct"] == out.loc[inside, "archived_row_score"]).all()
    for b in NOT_BINARY:
        assert metrics[b].calls == []                                # never scored for binary diagnostics
    assert all(len(metrics[b].calls) == 3 for b in binary)          # B/42, T/42, B/43 separately
    assert "binary_correct" not in rows.columns                     # input untouched


def test_with_binary_refuses_fractional_row_scores(cfg):
    rows = _rows(cfg, "B", 42)
    metrics = {b: _SliceCheckingMetric(b) for b in cfg.benchmark_keys()}
    metrics["blink"] = _SliceCheckingMetric("blink", values=[0.5] * ROWS_PER_BENCH)
    with pytest.raises(an.AnalysisError, match="0/1 row scores"):
        an.with_binary(rows, metrics, ["blink"])


# ---------------------------------------------------------------------------------------------------- load_rows
def test_load_rows_refuses_duplicate_archive_rows(cfg):
    arch = pd.concat([_rows(cfg, "B", 42), _rows(cfg, "B", 42).iloc[:1]], ignore_index=True)
    _write_work(cfg, arch, {})
    with pytest.raises(an.AnalysisError, match="duplicate"):
        an.load_rows(cfg, labels=["B"], seeds=[42], benchmarks=cfg.benchmark_keys())


def test_load_rows_local_rerun_replaces_only_its_own_archive_seed(cfg):
    arch = pd.concat([_rows(cfg, l, s) for l in an.ARCHIVE_LABELS for s in (42, 43)], ignore_index=True)
    rerun = _rows(cfg, "B", 42, source="existing_local_rerun", run_id="rerun", always=1.0)
    _write_work(cfg, arch, {"B_seed42_existing_local_rerun": rerun})
    keys = cfg.benchmark_keys()
    a = an.load_rows(cfg, labels=["B", "T"], seeds=[42], benchmarks=keys)
    r = an.load_rows(cfg, labels=["B", "T"], seeds=[42], benchmarks=keys, local={"B": [42]})
    assert set(a.loc[a.model_label == "B", "source"]) == {"archive"}
    assert set(r.loc[r.model_label == "B", "source"]) == {"existing_local_rerun"}
    assert len(r) == len(a) == 2 * len(keys) * ROWS_PER_BENCH
    assert not r.duplicated(["model_label", "seed", "content_key"]).any()
    # the override is per (label, seed): B's archived seed 43 stays
    r2 = an.load_rows(cfg, labels=["B"], seeds=[42, 43], benchmarks=keys, local={"B": [42]})
    assert r2.groupby("seed")["source"].agg(set).to_dict() == {42: {"existing_local_rerun"}, 43: {"archive"}}
    assert len(r2) == 2 * len(keys) * ROWS_PER_BENCH


def test_load_rows_refuses_duplicates_inside_a_local_run_and_missing_runs(cfg):
    arch = _rows(cfg, "B", 42)
    loc = _rows(cfg, "V@P1", 42, source="local_run", run_id="v")
    _write_work(cfg, arch, {"V@P1.confirmation.s42": pd.concat([loc, loc.iloc[:2]], ignore_index=True)})
    keys = cfg.benchmark_keys()
    with pytest.raises(an.AnalysisError, match="duplicate"):
        an.load_rows(cfg, labels=["B", "V@P1"], seeds=[42], benchmarks=keys, local={"V@P1": [42]},
                     subset="confirmation")
    with pytest.raises(an.AnalysisError, match="not ingested"):
        an.load_rows(cfg, labels=["V@P1"], seeds=[43], benchmarks=keys, local={"V@P1": [43]}, subset="confirmation")
    with pytest.raises(an.AnalysisError, match="needs a subset"):
        an.load_rows(cfg, labels=["V@P1"], seeds=[42], benchmarks=keys, local={"V@P1": [42]})


def test_load_rows_seed_filter_empty_requests_and_optional_columns(cfg):
    arch = _rows(cfg, "B", 42)
    loc = pd.concat([_rows(cfg, "V@P1", 42, source="local_run"), _rows(cfg, "V@P1", 43, source="local_run")],
                    ignore_index=True)
    _write_work(cfg, arch, {"V@P1.confirmation.s42": loc})
    keys = cfg.benchmark_keys()
    r = an.load_rows(cfg, labels=["V@P1"], seeds=[42], benchmarks=keys, local={"V@P1": [42]}, subset="confirmation")
    assert sorted(r["seed"].unique()) == [42]
    with pytest.raises(an.AnalysisError, match="no rows requested"):
        an.load_rows(cfg, labels=["V@P1"], seeds=[42], benchmarks=keys)
    slim = an.load_rows(cfg, labels=["B"], seeds=[42], benchmarks=keys,
                        columns=["model_label", "seed", "benchmark_id", "content_key"])
    assert list(slim.columns) == ["model_label", "seed", "benchmark_id", "content_key"]


def test_row_columns_carry_no_response_text():
    assert not {"response_text", "finish_reason", "completion_tokens"} & set(an.ROW_COLUMNS)


# ---------------------------------------------------------------------------------------------------- statistics glue
def test_run_bootstrap_reports_partial_categories_when_an_adapter_is_missing(cfg):
    keys = [b for b in cfg.benchmark_keys() if b != "blink"]
    rows = pd.concat([_rows(cfg, l, 42) for l in ("B", "T")], ignore_index=True)
    res = an.run_bootstrap(cfg, rows[rows.benchmark_id.isin(keys)], _metrics(cfg), _ga(cfg), labels=["B", "T"],
                           seeds=[42], benchmarks=keys, comparisons=["T-B"], n_replicates=50)
    cats = set(res.scores.loc[res.scores.level == "category", "scope"])
    assert any("(partial" in c for c in cats)


def test_json_safe_writes_strict_json(tmp_path):
    path = tmp_path / "s.json"
    an._write_json(path, {"a": float("nan"), "b": [np.float64(1.5), np.int64(2), float("inf")], "c": {"d": None}})
    assert _strict_json(path) == {"a": None, "b": [1.5, 2, None], "c": {"d": None}}


# ---------------------------------------------------------------------------------------------------- ingest_tier
def test_ingest_tier_reingests_runs_whose_directory_changed(cfg, monkeypatch):
    _write_work(cfg, _rows(cfg, "B", 42), {})
    label, slug, subset, seed = ev.tier_runs(cfg, "smoke")[0]
    rd = _validate_run(cfg, label, subset, seed)
    calls = []

    def fake_ingest(c, run_dir, lab, role, s, source, log=print):
        calls.append(lab)
        df = _rows(cfg, lab, s, source="local_run").iloc[:3]
        df.attrs["ingest_report"] = {"runtime_revision": "r"}
        return df

    def fake_write(c, df, name):
        rel = f"local_runs/{name}.report.json"
        config.write_json(c.work("aligned", rel), dict(df.attrs["ingest_report"]))
        index = json.loads(c.work("aligned", "index.json").read_text())
        index["local_runs"][name] = {"path": f"local_runs/{name}.parquet", "rows": len(df), "report": rel}
        c.work("aligned", "index.json").write_text(json.dumps(index))

    monkeypatch.setattr(ig, "ingest_local_run", fake_ingest)
    monkeypatch.setattr(ig, "write_local_run", fake_write)
    name = an.local_run_name(label, subset, seed)
    others = {an.local_run_name(l, sub, s) for l, _, sub, s in ev.tier_runs(cfg, "smoke")} - {name}
    st1 = an.ingest_tier(cfg, "smoke", log=lambda m: None)
    assert st1[name].startswith("ingested") and all(st1[o] == "not_complete" for o in others)
    assert an.ingested_fingerprint(cfg, name) == an.run_fingerprint(rd, seed)
    assert an.ingest_tier(cfg, "smoke", log=lambda m: None)[name] == "already_ingested"
    config.write_json(rd / f"validation.seed{seed}.json", {"complete": True, "rerun": 2})     # repeated run
    assert an.ingested_run_problems(cfg, [(label, slug, subset, seed)])
    assert an.ingest_tier(cfg, "smoke", log=lambda m: None)[name].startswith("re-ingested")
    assert not an.ingested_run_problems(cfg, [(label, slug, subset, seed)])
    assert an.ingest_tier(cfg, "smoke", force=True, log=lambda m: None)[name].startswith("re-ingested")
    assert calls == [label] * 3
    # another seed's descriptor in the shared spool does not change this seed's fingerprint
    fp = an.run_fingerprint(rd, seed)
    config.write_json(rd / "hf_archive" / "ready" / "other.ready.json", {"identity": {"seed": seed + 1}})
    assert an.run_fingerprint(rd, seed) == fp


# ---------------------------------------------------------------------------------------------------- capture
def test_capture_tables_runs_on_valid_input(cfg, monkeypatch):
    monkeypatch.setattr(an, "binary_benchmarks", lambda c: _binary(c))
    rows = pd.concat([_rows(cfg, l, 42) for l in ("B", "T", "V@P1", "M050@P1")], ignore_index=True)
    cd = an.capture_tables(cfg, rows, _metrics(cfg), _ga(cfg), merges=["M050@P1"], vero="V@P1", seeds=[42])
    assert set(cd["merge_label"]) == {"M050@P1"} and set(cd["vero_label"]) == {"V@P1"}
    assert set(cd["base_label"]) == {"B"} and set(cd["trace_label"]) == {"T"}
    assert list(cd.columns).count("merge_label") == 1


def test_base_reference_scores_the_base_like_a_merge(cfg, monkeypatch):
    monkeypatch.setattr(an, "binary_benchmarks", lambda c: _binary(c))
    rows = pd.concat([_rows(cfg, l, 42) for l in ("B", "T", "V@P1")], ignore_index=True)
    rows = an.with_base_reference(rows)
    assert (rows["model_label"] == an.BASE_REFERENCE).sum() == (rows["model_label"] == "B").sum()
    cd = an.capture_tables(cfg, rows, _metrics(cfg), _ga(cfg), merges=[an.BASE_REFERENCE], vero="V@P1", seeds=[42])
    brows = an.with_binary(rows[rows["benchmark_id"].isin(_binary(cfg))], _metrics(cfg), _binary(cfg))
    wide = brows[brows["model_label"].isin(["B", "T", "V@P1"])].pivot(index="content_key", columns="model_label",
                                                                    values="binary_correct")
    o = cd[(cd["level"] == "overall") & (cd["variant"] == "contemporaneous")].set_index("group")
    checked = 0
    for group, mask in (("trace_only", (wide["T"] == 1) & (wide["V@P1"] == 0)),
                        ("joint_failure", (wide["T"] == 0) & (wide["V@P1"] == 0))):
        if mask.any():
            assert o.loc[group, "n_group"] == mask.sum()
            # capture is P(B = 1 | group); damage-type groups would report P(B = 0 | group)
            assert o.loc[group, "conditional_rate_pct"] == pytest.approx(100 * wide.loc[mask, "B"].mean())
            checked += 1
    assert checked, "the fixture must populate at least one parent-defined group"
    # restricting the benchmark set is passed through, not re-derived from the registry
    one = _binary(cfg)[:1]
    cd1 = an.capture_tables(cfg, rows[rows["benchmark_id"].isin(one)], _metrics(cfg), _ga(cfg),
                            merges=[an.BASE_REFERENCE], vero="V@P1", seeds=[42], benchmarks=one)
    assert set(cd1.loc[cd1["level"] == "benchmark", "scope"]) <= set(one)


# ---------------------------------------------------------------------------------------------------- lock
def test_verify_lock_holds_and_records_appended_amendments(cfg, tmp_path):
    design = tmp_path / "DESIGN.md"
    design.write_text("# d\n\n## 1. Frozen\n\ntext\n\n## Amendments\n\n**A1 (t).** one.\n", encoding="utf-8")
    _lock(cfg)                                               # every lock input, locked against the repo's DESIGN.md
    cfg.work("reports", ev.LOCK_NAME).unlink()
    ev.lock_selection(cfg, design)                           # relocked against the synthetic design
    _, rec = an.verify_lock(cfg, design_path=design)
    assert rec["amendments_after_lock"] == [] and rec["runs_checked"] is False
    design.write_text(design.read_text() + "\n**A2 (t2).** two.\n", encoding="utf-8")
    assert an.verify_lock(cfg, design_path=design)[1]["amendments_after_lock"] == ["A2"]
    design.write_text(design.read_text().replace("text", "edited"), encoding="utf-8")
    with pytest.raises(an.AnalysisError, match="does not hold"):
        an.verify_lock(cfg, design_path=design)


def test_verify_lock_refuses_changed_inputs_and_missing_lock(cfg):
    with pytest.raises(an.AnalysisError, match="lock-selection"):
        an.verify_lock(cfg)
    _lock(cfg)
    an.verify_lock(cfg)
    config.write_json(cfg.work("merges", "M050", "merge_manifest.json"),
                      {"content_revision": "sha256set:M050", "recipe_sha256": "rebuilt"})
    with pytest.raises(an.AnalysisError, match="merge_recipe_sha256"):
        an.verify_lock(cfg)


def test_verify_lock_runs_need_fresh_ingests_locked_revisions_and_complete_preconditions(cfg):
    _confirmation_work(cfg)
    lock, rec = an.verify_lock(cfg, runs=True)
    assert rec["runs_checked"] and [r["label"] for r in lock["confirmation_runs"]][:3] == ["V@P1"] * 3
    # served at another revision
    name = an.local_run_name("M050@P1", "confirmation", 43)
    rd = ev.run_dir(cfg, _slug("M050@P1"), "confirmation")
    _register(cfg, name, runtime_revision="r:other", fingerprint=an.run_fingerprint(rd, 43))
    with pytest.raises(an.AnalysisError, match="lock froze r:m050-p1"):
        an.verify_lock(cfg, runs=True)
    _register(cfg, name, runtime_revision="r:m050-p1", fingerprint=an.run_fingerprint(rd, 43))
    an.verify_lock(cfg, runs=True)
    # a repeated run that was not ingested again
    config.write_json(rd / "validation.seed43.json", {"complete": True, "rerun": 2})
    with pytest.raises(an.AnalysisError, match="different state"):
        an.verify_lock(cfg, runs=True)
    _register(cfg, name, runtime_revision="r:m050-p1", fingerprint=an.run_fingerprint(rd, 43))
    # a confirmed model whose smoke run is not complete (DESIGN.md A4)
    _validate_run(cfg, "CT050@P1", "smoke", 42, complete=False)
    with pytest.raises(an.AnalysisError, match="CT050@P1 smoke seed 42 is not complete"):
        an.verify_lock(cfg, runs=True)


def test_locked_design_is_a_prefix_of_the_current_design():
    """The real lock hashed DESIGN.md through A4; every later change must be an appended amendment."""

    text = (config.PROJECT_ROOT / "DESIGN.md").read_text(encoding="utf-8")
    locked = "ee0fe4767a28cd18dd5afdc614b2983482704903d8050e0c52495b823d698252"
    assert ev.amendments_after(text, locked)[:2] == ["A5", "A6"]


# ---------------------------------------------------------------------------------------------------- end to end
def test_confirm_family_holm_capture_and_environment_sensitivity(cfg, monkeypatch):
    family = list(cfg.raw["statistics"]["primary_family"])
    assert family == ["M050-T", "M050-V@P1", "M050-CV050", "M050-CT050"]   # DESIGN.md section 7
    _confirmation_work(cfg)
    monkeypatch.setattr(mt, "build_metrics", lambda c: _metrics(c))
    monkeypatch.setattr(an, "binary_benchmarks", lambda c: _binary(c))
    calls = {}
    real_capture = an.capture_tables

    def spy(*a, **k):
        calls["capture"] = (a, k)
        return real_capture(*a, **k)

    monkeypatch.setattr(an, "capture_tables", spy)
    summary = an.confirm(cfg, log=lambda m: None)
    out = cfg.work("tables", "confirmation")
    d = pd.read_csv(out / "differences.csv")
    fam = d[d["p_holm"].notna()]
    assert sorted(fam["comparison"]) == sorted(["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1"])
    assert set(fam["level"]) == {"overall"} and fam["family_member"].all()
    assert d.loc[~d.index.isin(fam.index), "family_member"].eq(False).all()
    np.testing.assert_allclose(fam["p_holm"], st.holm_adjust(list(fam["p_value"])))
    assert summary["seeds"] == [42, 43, 44]
    assert summary["lock"]["runs_checked"] and summary["lock"]["amendments_after_lock"] == []
    assert sorted(pd.read_csv(out / "seed_scores.csv")["seed"].unique()) == [42, 43, 44]
    # capture/damage gets (T, V@P1) with base B and never V@P0, and is written
    (_, c_rows, *_), kw = calls["capture"]
    assert kw["vero"] == "V@P1" and kw["merges"] == ["M050@P1", "CV050@P1", "CT050@P1"]
    assert set(c_rows["model_label"]) == {"B", "T", *CONFIRMED}
    cd = pd.read_csv(out / "merge_capture_damage.csv")
    assert set(cd["merge_label"]) == {"M050@P1", "CV050@P1", "CT050@P1"} and set(cd["vero_label"]) == {"V@P1"}
    # sensitivity: the rerun table uses the existing reruns (B all correct, T all wrong), the other the archive
    arch = pd.read_csv(out / "sensitivity_seed42.archived_BT.csv")
    rer = pd.read_csv(out / "sensitivity_seed42.local_rerun_BT.csv")
    get = lambda t, c: float(t.loc[(t.comparison == c) & (t.level == "overall"), "estimate"].iloc[0])  # noqa: E731
    assert get(rer, "T-B") == pytest.approx(-100.0)
    assert get(arch, "T-B") != pytest.approx(-100.0)
    assert get(arch, "M050@P1-CV050@P1") == pytest.approx(get(rer, "M050@P1-CV050@P1"))   # no B/T involved
    assert not any("V@P0" in c for c in rer["comparison"])
    s = _strict_json(out / "summary.json")                    # strict JSON: non-family p_holm is null, not NaN
    assert s["differences"]["M050@P1-B"]["p_holm"] is None
    _strict_json(out / "bootstrap.json")


def test_confirm_requires_the_lock(cfg):
    _write_work(cfg, _rows(cfg, "B", 42), {})
    with pytest.raises(an.AnalysisError, match="lock-selection"):
        an.confirm(cfg, log=lambda m: None)


def test_screen_uses_seed_42_separates_protocol_groups_and_needs_fresh_ingests(cfg, monkeypatch):
    arch = pd.concat([_rows(cfg, l, s) for l in an.ARCHIVE_LABELS for s in (42, 43)], ignore_index=True)
    dev = set(_ga(cfg).loc[lambda g: g.in_dev_screen, "content_key"])
    labels = cfg.raw["evaluation_matrix"]["development"]["models"]
    local = {f"{l}.development_screen.s42": _rows(cfg, l, 42, source="local_run", run_id=l, keep=dev) for l in labels}
    _write_work(cfg, arch, local)
    _ingest_tier_fixture(cfg, "development")
    monkeypatch.setattr(mt, "build_metrics", lambda c: _metrics(c))
    monkeypatch.setattr(an, "binary_benchmarks", lambda c: _binary(c))
    calls = {}
    real_capture = an.capture_tables

    def spy(*a, **k):
        calls["capture"] = (a, k)
        return real_capture(*a, **k)

    monkeypatch.setattr(an, "capture_tables", spy)
    summary = an.screen(cfg, log=lambda m: None)
    out = cfg.work("tables", "screen")
    p1 = pd.read_csv(out / "scores.P1.csv")
    p2 = pd.read_csv(out / "scores.P2.csv")
    assert set(p1["model_label"]) == {"B", "T", "V@P0", *CONFIRMED}
    assert set(p2["model_label"]) == {"V@P0", "B@P2", "T@P2", "M050@P2", "CV050@P2", "CT050@P2"}
    assert set(pd.read_csv(out / "seed_scores.P1.csv")["seed"]) == {42} and summary["seeds"] == [42]
    assert pd.read_csv(out / "differences.P1.csv")["p_holm"].isna().all()   # nothing is confirmatory in the screen
    (_, c_rows, *_), kw = calls["capture"]
    assert kw["vero"] == "V@P1" and kw["seeds"] == [42]
    assert not any(l.endswith("@P2") for l in c_rows["model_label"].unique())
    assert set(pd.read_csv(out / "merge_capture_damage.csv")["merge_label"]) == {"M050@P1", "CV050@P1", "CT050@P1"}
    _strict_json(out / "summary.json")
    # a development run repeated after its ingest blocks the screen until it is ingested again
    rd = ev.run_dir(cfg, _slug("CV050@P2"), "development_screen")
    config.write_json(rd / "validation.seed42.json", {"complete": True, "rerun": 2})
    with pytest.raises(an.AnalysisError, match="CV050@P2.development_screen.s42 was ingested from a different state"):
        an.screen(cfg, log=lambda m: None)


# ---------------------------------------------------------------------------------------------------- command line
@pytest.mark.parametrize("argv", [["analyze", "--dry-run"], ["lock-selection", "--dry-run"], ["screen", "--dry-run"],
                                  ["ingest", "--dry-run"], ["confirm", "--force"], ["lock-selection", "--force"]])
def test_cli_refuses_flags_a_command_does_not_implement(argv, tmp_path, capsys):
    assert cli.main([*argv, "--work-root", str(tmp_path / "w")]) == 2
    assert "does not support" in capsys.readouterr().out
    assert not (tmp_path / "w").exists()                     # refused before anything is created


def test_cli_plan_eval_confirmation_checks_the_lock(tmp_path, capsys):
    rc = cli.main(["plan-eval", "--tier", "confirmation", "--work-root", str(tmp_path / "w"),
                   "--env-file", str(tmp_path / "env.sh"), "--jobs-dir", str(tmp_path / "jobs")])
    assert rc == 4 and "lock-selection" in capsys.readouterr().out
    assert not (tmp_path / "w" / "dispatch" / "wave_confirmation.tasks").exists()


# ---------------------------------------------------------------------------------------------------- config
@pytest.mark.parametrize("edit, message", [
    (lambda r: r["evaluation_matrix"]["development"].update(seeds=[43]), "sampling.development_seeds"),
    (lambda r: r["evaluation_matrix"]["confirmation"].update(seeds=[42, 43]), "sampling.confirmation_seeds"),
    (lambda r: r["statistics"].update(multiplicity="bonferroni"), "holm"),
    (lambda r: r["statistics"].update(primary_family=["M050-T", "M050-T"]), "primary_family"),
    (lambda r: r["statistics"].update(primary_family=["M050"]), "primary_family"),
    (lambda r: r.pop("sampling"), "sampling"),
])
def test_config_validates_seeds_and_the_primary_family(edit, message):
    raw = copy.deepcopy(config.load_config().raw)
    config.validate(raw)
    edit(raw)
    with pytest.raises(config.ConfigError, match=message):
        config.validate(raw)


# ---------------------------------------------------------------------------------------------------- reports
def _machine(*parts: str) -> str:
    return "/".join(["", *parts])                            # built at run time: no machine path in this file


def test_export_replaces_whole_paths_in_json_and_writes_nothing_when_a_file_is_dirty(cfg, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    scratch = _machine("scratch", "general", "nfs1", "u0", "work", "t.parquet")
    home = _machine("uufs", "chpc.utah.edu", "common", "home", "u0", "x")
    (src / "a.json").write_text(json.dumps({"p": scratch, "q": f"see {home}/y.csv", "r": "a/b"}), encoding="utf-8")
    (src / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    dest = tmp_path / "results"
    hashes = reports.export_results(cfg, {"a.json": src / "a.json", "b.csv": src / "b.csv"}, dest=dest)
    exported = (dest / "a.json").read_text(encoding="utf-8")
    assert json.loads(exported) == {"p": "<local>", "q": "see <local>", "r": "a/b"}
    assert "general" not in exported and "chpc" not in exported and set(hashes) == {"a.json", "b.csv"}
    for bad in (_machine("proj", "lab", "u0", "f"), _machine("Users", "u0", "f"), _machine("opt", "x"),
                "hf_" + "A1b2" * 8):
        out = tmp_path / f"out{abs(hash(bad))}"
        (src / "c.csv").write_text(f"x\n{bad}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="nothing written"):
            reports.export_results(cfg, {"b.csv": src / "b.csv", "c.csv": src / "c.csv"}, dest=out)
        assert not out.exists()


def test_entity_colors_are_distinct_and_cover_family_and_p2_labels():
    assert reports.COLORS["V@P0"] != reports.COLORS["V@P1"]
    assert len(set(reports.COLORS.values())) == len(reports.COLORS)
    assert reports.color_of("M050") == reports.color_of("M050@P2") == reports.COLORS["M050@P1"]
    assert reports.color_of("B@P2") == reports.COLORS["B"] and reports.color_of("T@P2") == reports.COLORS["T"]
    assert reports.color_of("unknown") == reports.INK_2


def test_parent_overlap_figure_follows_suite_order(cfg, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    cats = list(cfg.categories())
    rows = [{"seed_scope": "mean_over_seeds", "aggregation": "pooled_rows", "level": "category",
             "scope": c if i else f"{c} (partial: 2 of 4 benchmarks)", "n_units": 100 + i, "both_correct_pct": 40.0,
             "trace_only_pct": 10.0, "vero_only_pct": 10.0, "both_wrong_pct": 40.0} for i, c in enumerate(cats)]
    rows.append({**rows[1], "level": "overall", "scope": "21-benchmark binary-eligible set", "n_units": 999})
    seen = {}
    import matplotlib.axes

    real = matplotlib.axes.Axes.set_yticklabels

    def record(self, labels, *a, **k):
        seen["labels"] = list(labels)
        return real(self, labels, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_yticklabels", record)
    reports.fig_parent_overlap(pd.DataFrame(rows[::-1]), tmp_path / "o.png", "caption", category_order=cats)
    got = [l.split("\n")[0] for l in seen["labels"]]
    assert got == [cats[0], *cats[1:], "21-benchmark binary-eligible set"]
    assert all("rows/seed" in l for l in seen["labels"])


def test_bundle_keeps_aggregates_and_leaves_row_level_files_in_the_work_root(cfg):
    root = cfg.work()
    keep = ["audit/audit.md", "manifests/model_roles.json", "splits/split_manifest.json", "reports/merge_screen.md",
            "tables/evaluation_costs.csv", "tables/screen/scores.P1.csv", "figures/screen_model_scores.P1.png",
            "merges/M050/merge_manifest.json", "tests/test_report.txt"]
    drop = ["splits/rows.confirmation.json", "splits/group_assignments.parquet", "aligned/predictions.parquet",
            "aligned/local_runs/M050@P1.confirmation.s42.parquet", "runs/m050-p1/confirmation/scores.seed42.json",
            "merges/M050/model-00001-of-00005.safetensors", "dispatch/env.sh", "logs/fill_all.log"]
    for rel in keep + drop:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")
    assert set(reports.bundle_files(cfg)) == set(keep)


def test_contrast_and_capture_figures_render_with_readable_labels(cfg, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    cats = list(cfg.categories())
    comps = ["M050@P1-T", "M050@P1-V@P1"]
    diffs = pd.DataFrame([{"comparison": c, "level": lvl, "scope": sc, "estimate": 1.0, "ci_low": 0.0, "ci_high": 2.0,
                           "p_holm": 0.01}
                          for c in comps for lvl, sc in [("overall", "24-benchmark macro"),
                                                         *[("category", k) for k in cats]]])
    legends = []
    import matplotlib.axes

    real = matplotlib.axes.Axes.legend

    def record(self, *a, **k):
        leg = real(self, *a, **k)
        legends.append([t.get_text() for t in leg.get_texts()])
        return leg

    monkeypatch.setattr(matplotlib.axes.Axes, "legend", record)
    reports.fig_category_contrasts(diffs, tmp_path / "c.png", comps, cats, "t", "caption")
    assert legends[-1] == ["M050 − TRACE (Holm p = 0.01)", "M050 − VERO (backbone template) (Holm p = 0.01)"]
    # the lock's unresolved family names ("M050-T") match no row: refuse instead of drawing an empty figure
    with pytest.raises(ValueError, match="no overall or category rows"):
        reports.fig_category_contrasts(diffs, tmp_path / "e.png", ["M050-T"], cats, "t", "caption")
    cd = pd.DataFrame([{"level": "overall", "variant": "contemporaneous", "seed_scope": "single_condition",
                        "aggregation": "pooled_rows", "metric": m, "merge_label": g, "n_group": 10 + i,
                        "conditional_rate_pct": 50.0, "conditional_rate_pct_ci_low": 40.0,
                        "conditional_rate_pct_ci_high": 60.0}
                       for i, (m, _) in enumerate(reports.CAPTURE_EVENTS) for g in reports.MERGES])
    reports.fig_capture(cd, tmp_path / "d.png", reports.MERGES, "contemporaneous", "t", "caption")
    assert (tmp_path / "c.png").stat().st_size > 0 and (tmp_path / "d.png").stat().st_size > 0
    assert legends[-1] == [reports.DISPLAY[m] for m in reports.MERGES]


# ---------------------------------------------------------------------------------------------------- real work root
REAL = bool(os.environ.get("TRACE_MERGE_WORK_ROOT"))


@pytest.mark.skipif(not REAL, reason="real-data check needs TRACE_MERGE_WORK_ROOT")
def test_real_lock_and_split_hold_read_only():
    cfg = config.load_config()
    if not cfg.work("reports", ev.LOCK_NAME).is_file():
        pytest.skip("no selection lock yet")
    lock, rec = an.verify_lock(cfg)
    assert rec["amendments_after_lock"][:2] == ["A5", "A6"]
    assert lock["primary_family"] == cfg.raw["statistics"]["primary_family"]
    assert len(an.assignments(cfg)) > 0


def test_write_costs_counts_a_shared_run_directory_once(tmp_path, monkeypatch):
    import types

    rows = [{"view_slug": "m050-p1", "tier": "confirmation", "seed": s, "attempts": 2, "successful_attempts": 1,
             "wall_hours": 1.0, "gpu_hours": 2.0, "completion_tokens": 10, "run_dir_gib": 1.5} for s in (42, 43, 44)]
    rows.append({"view_slug": "m050-p1", "tier": "smoke", "seed": 42, "attempts": 1, "successful_attempts": 1,
                 "wall_hours": 0.1, "gpu_hours": 0.1, "completion_tokens": 1, "run_dir_gib": 0.25})
    monkeypatch.setattr(reports, "evaluation_costs", lambda cfg: pd.DataFrame(rows))
    (tmp_path / "tables").mkdir()
    reports.write_costs(types.SimpleNamespace(work=lambda *p: tmp_path.joinpath(*p)))
    totals = json.loads((tmp_path / "tables" / "evaluation_costs.totals.json").read_text())
    # the three seeds of one model and tier share one run directory
    assert totals["run_dirs_gib"] == pytest.approx(1.75)
    assert totals["gpu_hours"] == pytest.approx(6.1) and totals["attempts"] == 7
    assert totals["failed_or_incomplete_attempts"] == 3
