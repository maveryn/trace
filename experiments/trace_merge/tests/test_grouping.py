"""Split tests: connected groups, partition integrity, determinism, screen coverage (handoff 6.1, 11.1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_merge import grouping  # noqa: E402


def _manifest(spec: dict[str, list[list[str]]]) -> dict:
    """spec: benchmark -> per-row list of image hashes."""
    ds = {}
    for key, rows in spec.items():
        ds[key] = {"status": "ready", "row_media": [
            {"index": str(i + 1), "ordinal": i, "source_row_hash": f"{key}-r{i}", "source_record_sha256": f"{key}-c{i}",
             "media_set_sha256": "|".join(imgs), "media": [{"sha256": h} for h in imgs]} for i, imgs in enumerate(rows)]}
    return {"datasets": ds}


def _df(spec, families=None, columns=None, strata=None):
    m = _manifest(spec)
    keys = list(spec)
    df = grouping.build_rows(m, keys, {k: len(v) for k, v in spec.items()}, families or {}, strata or {},
                             columns=columns or {})
    df["component_id"] = grouping.connected_components(df)
    return df


def test_shared_images_connect_rows_within_and_across_benchmarks():
    df = _df({"a": [["x"], ["x", "y"], ["z"]], "b": [["y"], ["w"]]})
    comp = dict(zip(zip(df.benchmark_id, df.source_ordinal), df.component_id))
    assert comp[("a", 0)] == comp[("a", 1)] == comp[("b", 0)]  # x links a0-a1, y links a1-b0
    assert comp[("a", 2)] != comp[("a", 0)] and comp[("b", 1)] != comp[("a", 0)]


def test_family_columns_connect_rows_without_shared_images():
    cols = {"wemath": [{"ID": "1"}, {"ID": "1"}, {"ID": "2"}]}
    df = _df({"wemath": [["p"], ["q"], ["r"]]}, families={"wemath": ["ID"]}, columns=cols)
    assert df.component_id.iat[0] == df.component_id.iat[1] != df.component_id.iat[2]


def test_partition_keeps_components_whole_and_is_deterministic():
    spec = {"a": [[f"i{i // 3}"] for i in range(300)], "b": [[f"j{i}"] for i in range(200)] + [["i0"]]}
    df = _df(spec)
    r1 = grouping.assign_partition(df, 271828, 0.2)
    r2 = grouping.assign_partition(df, 271828, 0.2)
    assert r1.equals(r2)
    df["split_role"] = r1
    assert df.groupby("component_id")["split_role"].nunique().max() == 1
    for key, sub in df.groupby("benchmark_id"):
        frac = (sub.split_role == "development").mean()
        assert 0.15 <= frac <= 0.27, (key, frac)
    r3 = grouping.assign_partition(df, 1, 0.2)
    assert not r3.equals(r1)  # the seed matters


def test_screen_takes_whole_components_and_covers_strata():
    spec = {"t": [[f"k{i // 2}"] for i in range(400)]}
    cols = {"t": [{"split": "rare" if i == 399 else "common"} for i in range(400)]}
    df = _df(spec, columns=cols, strata={"t": ["split"]})
    df["split_role"] = "development"
    scr = grouping.select_screen(df, 7, 20, {"t": ["split"]})
    df["s"] = scr
    assert df.groupby("component_id")["s"].nunique().max() == 1
    assert {s["split"] for s in df.loc[scr, "strata"]} == {"common", "rare"}
    assert 20 <= int(scr.sum()) <= 24


def test_content_key_is_stable_and_field_sensitive():
    k = grouping.content_key("a", 0, "1", "r", "c")
    assert k == grouping.content_key("a", 0, "1", "r", "c")
    assert k != grouping.content_key("a", 1, "1", "r", "c") and k != grouping.content_key("b", 0, "1", "r", "c")


def test_rowlist_hash_and_format():
    df = _df({"a": [["x"], ["y"]]})
    df["split_role"] = "development"
    rl = grouping.rowlist(df, pd.Series([True, False]), ["a"], "t", {"rule": "r"})
    assert rl["schema"] == grouping.ROWLIST_SCHEMA and rl["benchmarks"]["a"]["selected"] == 1
    row = rl["benchmarks"]["a"]["rows"][0]
    assert {"ordinal", "index", "source_row_hash", "row_id"} <= set(row)


def test_empty_family_value_does_not_join_unrelated_rows():
    cols = {"wemath": [{"ID": ""}, {"ID": " "}, {"ID": "7"}, {"ID": "7"}]}
    df = _df({"wemath": [["p"], ["q"], ["r"], ["s"]]}, families={"wemath": ["ID"]}, columns=cols)
    assert df.families.iat[0] == [] and df.families.iat[2] == ["wemath:ID=7"]
    assert df.component_id.iat[0] != df.component_id.iat[1]
    assert df.component_id.iat[2] == df.component_id.iat[3]


def _toy_study(tmp_path, monkeypatch):
    from trace_merge import config

    spec = {"a": [[f"x{i // 2}"] for i in range(30)], "b": [[f"y{i}"] for i in range(19)] + [["x0"]]}
    cfg = config.load_config(work_root=tmp_path / "work", protocol_clone=tmp_path / "clone")
    config.ensure_work_tree(cfg)
    config.write_json(cfg.dataset_manifest, {**_manifest(spec), "view_snapshot_sha256": {"trace_eval_v1": "s"}})
    suite = {"suite_id": "toy", "benchmarks": [{"key": k, "rows": len(v)} for k, v in spec.items()]}
    monkeypatch.setattr(config.StudyConfig, "suite", lambda self: suite)
    return cfg


def test_refreeze_compares_inputs_rule_and_assignments(tmp_path, monkeypatch):
    cfg = _toy_study(tmp_path, monkeypatch)
    logs: list[str] = []
    first = grouping.freeze_splits(cfg, log=logs.append)
    assert first["rule"] == grouping.split_rule(cfg.raw["split"])
    assert grouping.freeze_splits(cfg, log=logs.append) == first and "already frozen" in logs[-1]
    ga_path, man_path = cfg.work("splits", "group_assignments.parquet"), cfg.work("splits", "split_manifest.json")
    ga, ga_bytes, man = pd.read_parquet(ga_path), ga_path.read_bytes(), json.loads(man_path.read_text())
    ga.assign(component_rank=ga["component_rank"] / 2).to_parquet(ga_path, index=False)  # same row lists
    with pytest.raises(RuntimeError, match="group_assignments_sha256"):
        grouping.freeze_splits(cfg, log=logs.append)
    grouping.write_json(man_path, {**man, "group_assignments_sha256": grouping.sha256_file(ga_path)})
    with pytest.raises(RuntimeError, match=r"'group_assignments'"):
        grouping.freeze_splits(cfg, log=logs.append)
    ga_path.write_bytes(ga_bytes)
    grouping.write_json(man_path, man)
    data = json.loads(cfg.dataset_manifest.read_text())
    grouping.write_json(cfg.dataset_manifest, {**data, "note": "rebuilt"})  # same rows, different manifest
    with pytest.raises(RuntimeError, match="dataset_manifest_sha256"):
        grouping.freeze_splits(cfg, log=logs.append)


def test_bootstrap_units_merge_twin_components_and_flag_development_twins():
    ga = pd.DataFrame({"benchmark_id": ["a", "a", "b", "b"], "source_ordinal": [0, 1, 0, 1],
                       "content_key": ["ka0", "ka1", "kb0", "kb1"], "component_id": ["c1", "c2", "c3", "c4"],
                       "split_role": ["development", "confirmation", "confirmation", "confirmation"]})
    twins = pd.DataFrame({"bench_a": ["a"], "ordinal_a": [0], "bench_b": ["b"], "ordinal_b": [0]})
    u = grouping.bootstrap_units(ga, twins).set_index("content_key")
    assert u.loc["ka0", "bootstrap_unit"] == u.loc["kb0", "bootstrap_unit"]
    assert u["bootstrap_unit"].nunique() == 3
    assert bool(u.loc["kb0", "in_dev_twin"]) and not u.loc[["ka0", "ka1", "kb1"], "in_dev_twin"].any()
    with pytest.raises(ValueError):
        grouping.bootstrap_units(ga, pd.DataFrame({"bench_a": ["a"], "ordinal_a": [9], "bench_b": ["b"], "ordinal_b": [0]}))
