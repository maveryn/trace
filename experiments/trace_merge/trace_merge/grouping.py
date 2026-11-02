"""Connected image/problem groups and the frozen development/confirmation partition (DESIGN.md section 4).

Rows are nodes; two rows are connected when they share any image (per-image sha256, across all 24 benchmarks) or a
declared problem family (WeMath ``ID``; an empty family value declares no family). Connected components are the
independent units for the split and for every bootstrap. The partition is a pure function of the prepared data and the
config seed, computed before any merged-model output exists, and is written with the hashes needed to prove it was not
changed afterwards.

Image identity is byte-level: the same picture stored with a different encoding in two benchmarks (e.g. MathVista items
re-encoded in MMStar) is not joined. The development screen and the smoke list are chosen per benchmark and keep
components whole within each benchmark; a cross-benchmark component can have only some of its benchmarks' rows on the
screen (all of its rows are development rows either way). Smoke takes at least 4 rows per benchmark, so whole components
and stratum coverage make it larger than 4 x 24.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .config import StudyConfig, canonical_json, sha256_bytes, sha256_file, write_json

ROWLIST_SCHEMA = "trace-cert-ext-eval-subset-v1"  # row-list format read by the generation filter (jobs/eval_lib.py)
SPLIT_SCHEMA = "trace-merge-split-v1"


def content_key(benchmark_id: str, ordinal: int, index: str, row_sha: str, record_sha: str) -> str:
    return sha256_bytes(canonical_json([benchmark_id, int(ordinal), str(index), row_sha, record_sha]).encode("utf-8"))


def rank_fraction(seed: int, key: str) -> float:
    h = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0**64


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def load_tsv_columns(manifest: dict[str, Any], key: str, columns: Iterable[str]) -> list[dict[str, str]]:
    """Per-ordinal values of licensed TSV columns, row order checked against the dataset manifest."""

    receipt = manifest["datasets"][key]
    path = Path(receipt["metadata_files"][0]["path"])
    cols = list(columns)
    table = pd.read_csv(path, sep="\t", usecols=["index", *cols], dtype=str, keep_default_na=False)
    rows = receipt["row_media"]
    if len(table) != len(rows) or [str(i) for i in table["index"]] != [str(r["index"]) for r in rows]:
        raise ValueError(f"{key}: {path.name} row order does not match the dataset manifest")
    return [{c: str(v[c]) for c in cols} for v in table.to_dict("records")]


def build_rows(manifest: dict[str, Any], keys: list[str], expected_rows: dict[str, int],
               family_columns: dict[str, list[str]], strata_columns: dict[str, list[str]],
               columns: dict[str, list[dict[str, str]]] | None = None) -> pd.DataFrame:
    recs = []
    columns = columns if columns is not None else {}
    for key in keys:
        receipt = manifest["datasets"].get(key) or {}
        rows = receipt.get("row_media") or []
        if receipt.get("status") != "ready" or len(rows) != expected_rows[key]:
            raise ValueError(f"{key}: manifest receipt not ready or has {len(rows)} != {expected_rows[key]} rows")
        need = list(dict.fromkeys([*family_columns.get(key, []), *strata_columns.get(key, [])]))
        cols = columns.get(key)
        if need and cols is None:
            cols = load_tsv_columns(manifest, key, need)
        for i, r in enumerate(rows):
            if int(r["ordinal"]) != i:
                raise ValueError(f"{key}: manifest ordinals are not 0..n-1")
            images = sorted({m["sha256"] for m in (r.get("media") or []) if m.get("sha256")})
            rec = {
                "benchmark_id": key,
                "source_ordinal": int(r["ordinal"]),
                "source_index": str(r["index"]),
                "source_row_sha256": str(r["source_row_hash"]),
                "source_record_sha256": str(r["source_record_sha256"]),
                "media_set_sha256": str(r.get("media_set_sha256") or ""),
                "images": images,
                "families": [f"{key}:{c}={cols[i][c]}" for c in family_columns.get(key, []) if str(cols[i][c]).strip()],
                "strata": {c: cols[i][c] for c in strata_columns.get(key, [])},
            }
            rec["content_key"] = content_key(key, rec["source_ordinal"], rec["source_index"],
                                             rec["source_row_sha256"], rec["source_record_sha256"])
            recs.append(rec)
    df = pd.DataFrame.from_records(recs)
    if df["content_key"].duplicated().any():
        raise ValueError("duplicate content keys in the prepared dataset")
    return df


def connected_components(df: pd.DataFrame) -> pd.Series:
    uf = UnionFind(len(df))
    first_by_token: dict[str, int] = {}
    for i, (images, fams) in enumerate(zip(df["images"], df["families"])):
        for tok in [f"img:{h}" for h in images] + [f"fam:{f}" for f in fams]:
            j = first_by_token.setdefault(tok, i)
            if j != i:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(len(df))]
    members: dict[int, list[str]] = defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(df["content_key"].iat[i])
    comp_id = {r: sha256_bytes("\n".join(sorted(keys)).encode("utf-8")) for r, keys in members.items()}
    return pd.Series([comp_id[r] for r in roots], index=df.index, name="component_id")


def assign_partition(df: pd.DataFrame, seed: int, dev_fraction: float) -> pd.Series:
    """Cross-benchmark components by rank threshold; then per benchmark fill to round(fraction * rows) in rank order."""

    comp_bench = df.groupby("component_id")["benchmark_id"].nunique()
    rank = {c: rank_fraction(seed, c) for c in comp_bench.index}
    dev: set[str] = {c for c, nb in comp_bench.items() if nb > 1 and rank[c] < dev_fraction}
    for key, sub in df.groupby("benchmark_id", sort=False):
        target = round(dev_fraction * len(sub))
        have = int(sub["component_id"].isin(dev).sum())
        singles = sorted((c for c in sub["component_id"].unique() if comp_bench[c] == 1), key=lambda c: (rank[c], c))
        sizes = sub.groupby("component_id").size()
        for c in singles:
            if have >= target:
                break
            dev.add(c)
            have += int(sizes[c])
    return df["component_id"].map(lambda c: "development" if c in dev else "confirmation").rename("split_role")


def select_screen(df: pd.DataFrame, seed: int, target: int, strata_columns: dict[str, list[str]],
                  per_bench_cap: int | None = None) -> pd.Series:
    """Development screen: per benchmark, development components in rank order until >= target rows (whole components),
    then the best-ranked development component carrying each missing value of the scorer's per-stratum columns.

    Components are kept whole within each benchmark only: the rows of a cross-benchmark component in another benchmark
    are chosen by that benchmark's own ranking (frozen behaviour of the v1 screen and smoke lists)."""

    chosen = pd.Series(False, index=df.index)
    dev = df[df["split_role"] == "development"]
    for key, sub in dev.groupby("benchmark_id", sort=False):
        comps = sorted(sub["component_id"].unique(), key=lambda c: (rank_fraction(seed, c), c))
        by_comp = sub.groupby("component_id").groups
        n = 0
        picked: list[str] = []
        goal = target if per_bench_cap is None else min(target, per_bench_cap)
        for c in comps:
            if n >= goal:
                break
            picked.append(c)
            n += len(by_comp[c])
        for col in strata_columns.get(key, []):
            have = {s[col] for s in sub.loc[sub["component_id"].isin(picked), "strata"]}
            for c in comps:
                vals = {s[col] for s in sub.loc[by_comp[c], "strata"]}
                if vals - have:
                    picked.append(c)
                    have |= vals
        chosen.loc[sub.index[sub["component_id"].isin(picked)]] = True
    return chosen


def rowlist(df: pd.DataFrame, mask: pd.Series, suite_keys: list[str], name: str, meta: dict[str, Any]) -> dict[str, Any]:
    sel = df[mask]
    out: dict[str, Any] = {"schema": ROWLIST_SCHEMA, "mode": name, "rule": meta.get("rule"), **{k: v for k, v in meta.items() if k != "rule"},
                           "benchmarks": {}}
    lines = []
    for key in suite_keys:
        sub = sel[sel["benchmark_id"] == key].sort_values("source_ordinal")
        total = int((df["benchmark_id"] == key).sum())
        unique = df.loc[df["benchmark_id"] == key, "source_index"].is_unique
        rows = []
        for r in sub.itertuples(index=False):
            row_id = r.source_index if unique else f"{r.source_ordinal}:{r.source_index}"
            # "rank" is the generation filter's field name (jobs/eval_lib.py); here it carries the component id
            rows.append({"ordinal": int(r.source_ordinal), "index": r.source_index, "source_row_hash": r.source_row_sha256,
                         "row_id": row_id, "rank": r.component_id, "content_key": r.content_key})
            lines.append(f"{key}\t{row_id}\n")
        out["benchmarks"][key] = {"total": total, "selected": len(rows), "rows": rows}
    out["total_selected"] = int(mask.sum())
    out["row_list_sha256"] = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return out


ASSIGNMENT_COLUMNS = ["benchmark_id", "source_ordinal", "source_index", "source_row_sha256", "source_record_sha256",
                      "media_set_sha256", "content_key", "component_id", "component_rank", "split_role",
                      "in_dev_screen", "in_smoke"]


def split_rule(split_cfg: dict[str, Any]) -> str:
    """The rule text recorded in the split manifest; a function of the config's ``split`` section only."""

    seed = int(split_cfg["seed"])
    frac = float(split_cfg["development_fraction"])
    target = int(split_cfg["screen_target_rows_per_benchmark"])
    families = {k: list(v) for k, v in (split_cfg.get("family_columns") or {}).items()}
    strata = {k: list(v) for k, v in (split_cfg.get("cover_strata") or {}).items()}
    return ("components: union of rows sharing any image sha256 (global) or a family column value "
            f"{families}; rank = blake2b('{seed}|'+component_id); cross-benchmark components -> development if rank < "
            f"{frac}; then per benchmark single-benchmark components in rank order until development rows >= "
            f"round({frac}*rows); screen: development components in rank order until >= {target} rows per benchmark, "
            f"plus coverage of {strata}")


def freeze_splits(cfg: StudyConfig, dry_run: bool = False, log: Callable[[str], None] = print) -> dict[str, Any]:
    split_cfg = cfg.raw["split"]
    seed = int(split_cfg["seed"])
    frac = float(split_cfg["development_fraction"])
    target = int(split_cfg["screen_target_rows_per_benchmark"])
    families = {k: list(v) for k, v in (split_cfg.get("family_columns") or {}).items()}
    strata = {k: list(v) for k, v in (split_cfg.get("cover_strata") or {}).items()}
    suite = cfg.suite()
    keys = [b["key"] for b in suite["benchmarks"]]
    expected = {b["key"]: int(b["rows"]) for b in suite["benchmarks"]}
    manifest_path = cfg.dataset_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = cfg.work("splits")
    manifest_out = out_dir / "split_manifest.json"
    df = build_rows(manifest, keys, expected, families, strata)
    df["component_id"] = connected_components(df)
    df["component_rank"] = df["component_id"].map(lambda c: rank_fraction(seed, c))
    df["split_role"] = assign_partition(df, seed, frac)
    df["in_dev_screen"] = select_screen(df, seed, target, strata)
    df["in_smoke"] = select_screen(df[df["in_dev_screen"]].copy(), seed, 4, strata).reindex(df.index, fill_value=False)
    comp_sizes = df.groupby("component_id").size()
    comp_bench = df.groupby("component_id")["benchmark_id"].nunique()
    per_bench = {}
    for key in keys:
        sub = df[df["benchmark_id"] == key]
        per_bench[key] = {
            "rows": int(len(sub)), "components": int(sub["component_id"].nunique()),
            "development_rows": int((sub["split_role"] == "development").sum()),
            "confirmation_rows": int((sub["split_role"] == "confirmation").sum()),
            "development_fraction": round(float((sub["split_role"] == "development").mean()), 4),
            "dev_screen_rows": int(sub["in_dev_screen"].sum()),
            "dev_screen_components": int(sub.loc[sub["in_dev_screen"], "component_id"].nunique()),
            "smoke_rows": int(sub["in_smoke"].sum()),
            "largest_component_rows": int(sub.groupby("component_id").size().max()),
            "rows_in_cross_benchmark_components": int(sub["component_id"].map(comp_bench).gt(1).sum()),
        }
    rule = split_rule(split_cfg)
    common = {"rule": rule, "seed": seed, "suite_id": suite["suite_id"],
              "dataset_manifest_sha256": sha256_file(manifest_path),
              "dataset_snapshot_sha256": (manifest.get("view_snapshot_sha256") or {}).get("trace_eval_v1"),
              "config_sha256": cfg.sha256}
    lists = {
        "development_screen": rowlist(df, df["in_dev_screen"], keys, "development_screen", common),
        "confirmation": rowlist(df, df["split_role"] == "confirmation", keys, "confirmation", common),
        "development": rowlist(df, df["split_role"] == "development", keys, "development", common),
        "smoke": rowlist(df, df["in_smoke"], keys, "smoke", common),
    }
    summary = {
        "schema": SPLIT_SCHEMA, **common,
        "rows": int(len(df)), "components": int(df["component_id"].nunique()),
        "cross_benchmark_components": int((comp_bench > 1).sum()),
        "largest_component_rows": int(comp_sizes.max()),
        "singleton_components": int((comp_sizes == 1).sum()),
        "development_rows": int((df["split_role"] == "development").sum()),
        "confirmation_rows": int((df["split_role"] == "confirmation").sum()),
        "dev_screen_rows": int(df["in_dev_screen"].sum()),
        "smoke_rows": int(df["in_smoke"].sum()),
        "row_lists": {k: {"file": f"rows.{k}.json", "total_selected": v["total_selected"],
                          "row_list_sha256": v["row_list_sha256"]} for k, v in lists.items()},
        "per_benchmark": per_bench,
    }
    if dry_run:
        return {"dry_run": True, **{k: v for k, v in summary.items() if k != "per_benchmark"}}
    if manifest_out.is_file():
        old = json.loads(manifest_out.read_text(encoding="utf-8"))
        same = all(old.get("row_lists", {}).get(k, {}).get("row_list_sha256") == v["row_list_sha256"]
                   for k, v in summary["row_lists"].items())
        if not same:
            raise RuntimeError(f"{manifest_out} is frozen with different row lists; the inputs or rule changed")
        # the row-list hashes cover (benchmark, row_id) only: also compare the inputs, the rule and the assignments
        diff = [k for k in ("rule", "seed", "suite_id", "dataset_manifest_sha256") if old.get(k) != common[k]]
        ga_path = out_dir / "group_assignments.parquet"
        if not ga_path.is_file() or sha256_file(ga_path) != old.get("group_assignments_sha256"):
            diff.append("group_assignments_sha256")
        elif not pd.read_parquet(ga_path).equals(df[ASSIGNMENT_COLUMNS].reset_index(drop=True)):
            diff.append("group_assignments")
        if diff:
            raise RuntimeError(f"{manifest_out} is frozen with identical row lists but different {diff}")
        log("[splits] already frozen with identical row lists, inputs, rule and group assignments")
        return {k: v for k, v in old.items() if k != "per_benchmark"}
    for k, v in lists.items():
        write_json(out_dir / f"rows.{k}.json", v)
    df[ASSIGNMENT_COLUMNS].to_parquet(out_dir / "group_assignments.parquet", index=False)
    summary["group_assignments_sha256"] = sha256_file(out_dir / "group_assignments.parquet")
    summary["frozen_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    write_json(manifest_out, summary)
    return {k: v for k, v in summary.items() if k != "per_benchmark"}


# ------------------------------------------------------------------------------------ bootstrap clusters (amendment A7)
# The partition joins rows on byte-identical image files. Some benchmarks re-encode the same picture (MathVista and
# MMStar share 285 images with the same question and answer; MME-Reasoning and PuzzleVQA share 67), so those twins sit
# in different components, occasionally on different sides of the frozen partition. The partition stays frozen (runs
# were in flight); for every bootstrap the twins' components are merged into one resampling unit, so twins are never
# drawn as independent groups. ``in_dev_twin`` marks confirmation rows with a development twin (sensitivity analysis).

TWIN_COLUMNS = ("bench_a", "ordinal_a", "bench_b", "ordinal_b")


def bootstrap_units(ga: pd.DataFrame, twins: pd.DataFrame) -> pd.DataFrame:
    """content_key -> bootstrap_unit (union of components joined by twin pairs) and in_dev_twin."""

    missing = [c for c in TWIN_COLUMNS if c not in twins.columns]
    if missing:
        raise ValueError(f"twin list lacks columns {missing}")
    by_row = ga.set_index(["benchmark_id", "source_ordinal"]) if len(twins) else None
    comps = sorted(ga["component_id"].unique())
    pos = {c: i for i, c in enumerate(comps)}
    uf = UnionFind(len(comps))
    pairs = []
    for a, oa, b, ob in twins[list(TWIN_COLUMNS)].itertuples(index=False):
        ka, kb = (str(a), int(oa)), (str(b), int(ob))
        if ka not in by_row.index or kb not in by_row.index:
            raise ValueError(f"twin pair {ka} / {kb} is not in the frozen assignments")
        ra, rb = by_row.loc[ka], by_row.loc[kb]
        uf.union(pos[ra["component_id"]], pos[rb["component_id"]])
        pairs.append((ra["content_key"], rb["content_key"], ra["split_role"], rb["split_role"]))
    members: dict[int, list[str]] = defaultdict(list)
    for c in comps:
        members[uf.find(pos[c])].append(c)
    unit_of = {c: sha256_bytes("\n".join(sorted(members[uf.find(pos[c])])).encode("utf-8")) for c in comps}
    dev_twin = set()
    for ka, kb, ra, rb in pairs:
        if ra == "confirmation" and rb == "development":
            dev_twin.add(ka)
        if rb == "confirmation" and ra == "development":
            dev_twin.add(kb)
    keep = [c for c in ("content_key", "benchmark_id", "source_ordinal", "component_id", "split_role") if c in ga.columns]
    out = ga[keep].copy()
    out["bootstrap_unit"] = out["component_id"].map(unit_of)
    out["in_dev_twin"] = out["content_key"].isin(dev_twin)
    return out


def freeze_bootstrap_units(cfg: StudyConfig, twins_csv: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Write splits/bootstrap_units.parquet + splits/bootstrap_units.json from a verified twin list (once)."""

    out_dir = cfg.work("splits")
    manifest = out_dir / "bootstrap_units.json"
    ga_path = out_dir / "group_assignments.parquet"
    twins = pd.read_csv(twins_csv)
    units = bootstrap_units(pd.read_parquet(ga_path), twins)
    summary = {
        "schema": "trace-merge-bootstrap-units-v1",
        "rule": "bootstrap units = components joined by verified cross-benchmark near-duplicate image twins "
                "(same question and answer); the frozen partition is unchanged",
        "group_assignments_sha256": sha256_file(ga_path),
        "twin_pairs": int(len(twins)),
        "twin_pairs_by_benchmarks": ({f"{a}|{b}": int(n) for (a, b), n in
                                      twins.groupby(["bench_a", "bench_b"]).size().items()} if len(twins) else {}),
        "components": int(units["component_id"].nunique()),
        "bootstrap_units": int(units["bootstrap_unit"].nunique()),
        "confirmation_rows_with_development_twin": int(units["in_dev_twin"].sum()),
        "confirmation_rows_with_development_twin_by_benchmark":
            ({k: int(v) for k, v in units[units["in_dev_twin"]].groupby("benchmark_id").size().items()}
             if "benchmark_id" in units.columns else {}),
    }
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in summary} != summary:
            raise RuntimeError(f"{manifest} is frozen with a different twin list or split")
        log("[bootstrap-units] already frozen")
        return old
    twins.to_csv(out_dir / "near_duplicate_twins.csv", index=False)
    units.to_parquet(out_dir / "bootstrap_units.parquet", index=False)
    summary["twins_csv_sha256"] = sha256_file(out_dir / "near_duplicate_twins.csv")
    summary["bootstrap_units_sha256"] = sha256_file(out_dir / "bootstrap_units.parquet")
    summary["frozen_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    write_json(manifest, summary)
    return summary
