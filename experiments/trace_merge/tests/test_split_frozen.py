"""Read-only checks of the frozen split and lock of a real work root (skipped unless TRACE_MERGE_WORK_ROOT and
TRACE_MERGE_PROTOCOL_CLONE are set): the split rebuilds bit for bit, its invariants hold, every row list a run directory
carries equals the frozen one, and the selection lock still holds. Nothing under the work root is written."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_merge import config, evaluation, grouping  # noqa: E402

REAL = bool(os.environ.get("TRACE_MERGE_WORK_ROOT") and os.environ.get("TRACE_MERGE_PROTOCOL_CLONE"))
pytestmark = pytest.mark.skipif(not REAL, reason="needs the frozen work root and the protocol clone")
SUBSETS = ["development_screen", "confirmation", "development", "smoke"]


@pytest.fixture(scope="module")
def real():
    c = config.load_config()
    sc = c.raw["split"]
    seed, frac, target = int(sc["seed"]), float(sc["development_fraction"]), int(sc["screen_target_rows_per_benchmark"])
    fam = {k: list(v) for k, v in (sc.get("family_columns") or {}).items()}
    strata = {k: list(v) for k, v in (sc.get("cover_strata") or {}).items()}
    suite = c.suite()
    keys = [b["key"] for b in suite["benchmarks"]]
    man = json.loads(c.dataset_manifest.read_text(encoding="utf-8"))
    df = grouping.build_rows(man, keys, {b["key"]: int(b["rows"]) for b in suite["benchmarks"]}, fam, strata)
    df["component_id"] = grouping.connected_components(df)
    df["component_rank"] = df["component_id"].map(lambda x: grouping.rank_fraction(seed, x))
    df["split_role"] = grouping.assign_partition(df, seed, frac)
    df["in_dev_screen"] = grouping.select_screen(df, seed, target, strata)
    df["in_smoke"] = grouping.select_screen(df[df["in_dev_screen"]].copy(), seed, 4, strata).reindex(
        df.index, fill_value=False)
    frozen = json.loads(c.work("splits", "split_manifest.json").read_text(encoding="utf-8"))
    return c, keys, strata, df, frozen


def test_real_rebuild_reproduces_frozen_split(real):
    c, keys, _, df, frozen = real
    assert frozen["rule"] == grouping.split_rule(c.raw["split"])
    assert frozen["dataset_manifest_sha256"] == config.sha256_file(c.dataset_manifest)
    masks = {"development_screen": df["in_dev_screen"], "confirmation": df["split_role"] == "confirmation",
             "development": df["split_role"] == "development", "smoke": df["in_smoke"]}
    for name, mask in masks.items():
        assert grouping.rowlist(df, mask, keys, name, {"rule": ""})["row_list_sha256"] == \
            frozen["row_lists"][name]["row_list_sha256"], name
    ga_path = c.work("splits", "group_assignments.parquet")
    assert config.sha256_file(ga_path) == frozen["group_assignments_sha256"]
    assert pd.read_parquet(ga_path).equals(df[grouping.ASSIGNMENT_COLUMNS].reset_index(drop=True))


def test_real_split_invariants(real):
    _, _, strata, df, _ = real
    assert df["content_key"].is_unique
    assert df.groupby("component_id")["split_role"].nunique().max() == 1
    assert not (df["in_dev_screen"] & (df["split_role"] != "development")).any()
    assert not (df["in_smoke"] & ~df["in_dev_screen"]).any()
    for key, sub in df.groupby("benchmark_id"):
        dev = int((sub["split_role"] == "development").sum())
        assert round(0.2 * len(sub)) <= dev <= round(0.2 * len(sub)) + int(sub.groupby("component_id").size().max())
        for col in ["in_dev_screen", "in_smoke"]:
            assert sub.groupby("component_id")[col].nunique().max() == 1, (key, col)  # whole within a benchmark
    for key, cols in strata.items():
        sub = df[df["benchmark_id"] == key]
        for col in cols:
            allv = {s[col] for s in sub["strata"]}
            assert {s[col] for s in sub.loc[sub["in_dev_screen"], "strata"]} == allv, (key, col)
            assert {s[col] for s in sub.loc[sub["in_smoke"], "strata"]} == allv, (key, col)


def test_real_frozen_row_lists_are_self_consistent(real):
    """The generation filter selects by (ordinal, index, source_row_hash) while rows-check hashes (benchmark, row_id):
    the frozen lists must tie the two together (row_id rule, unique ordinals, counts, content keys)."""

    c, keys, _, df, frozen = real
    by_key = {(r.benchmark_id, r.source_ordinal): r for r in df.itertuples(index=False)}
    for name in SUBSETS:
        rl = json.loads(c.work("splits", f"rows.{name}.json").read_text(encoding="utf-8"))
        assert rl["row_list_sha256"] == frozen["row_lists"][name]["row_list_sha256"]
        for key in keys:
            entry = rl["benchmarks"][key]
            unique = df.loc[df["benchmark_id"] == key, "source_index"].is_unique
            ordinals = [int(r["ordinal"]) for r in entry["rows"]]
            assert len(set(ordinals)) == len(ordinals) == entry["selected"], (name, key)
            for r in entry["rows"]:
                src = by_key[(key, int(r["ordinal"]))]
                assert (r["index"], r["source_row_hash"], r["content_key"]) == \
                    (src.source_index, src.source_row_sha256, src.content_key), (name, key, r["ordinal"])
                assert r["row_id"] == (src.source_index if unique else f"{src.source_ordinal}:{src.source_index}")


def test_real_run_directories_carry_the_frozen_row_lists(real):
    c, *_, frozen = real
    for name in SUBSETS:
        want = json.loads(c.work("splits", f"rows.{name}.json").read_text(encoding="utf-8"))
        for copy in sorted(c.work("runs").glob(f"*/{name}/rows.{name}.json")):
            assert json.loads(copy.read_text(encoding="utf-8")) == want, copy.parent
            for plan in sorted(copy.parent.glob("plan.seed*.json")):
                p = json.loads(plan.read_text(encoding="utf-8"))
                assert p["row_list_sha256"] == frozen["row_lists"][name]["row_list_sha256"], plan
                assert {k: int(v) for k, v in p["expected_rows"].items()} == \
                    {k: int(v["selected"]) for k, v in want["benchmarks"].items()}, plan


def test_real_selection_lock_still_holds():
    c = config.load_config()
    if not c.work("reports", evaluation.LOCK_NAME).is_file():
        pytest.skip("no selection lock yet")
    rep = evaluation.verify_lock(c, config.PROJECT_ROOT / "DESIGN.md", strict=False)
    assert rep["ok"], rep
