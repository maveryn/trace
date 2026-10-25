"""7B replication report (DESIGN 7.3 stage 3, 7.4, 8): P7 vs G7 vs S7, base/legacy 7B context, 3B side by side.

    nice -n 10 "$TRACE_CERT_PYTHON" experiments/certificates/analysis/cert_7b.py \\
        --runs "$TRACE_CERT_RUNS" --out "$TRACE_CERT_RUNS/results/final/cert_7b" \\
        [--trace-tool-src "$TRACE_TOOL_ROOT/src"] [--n-boot 10000] [--n-perm 10000]
    ... --render-only      # re-render cert_7b.md from an existing cert_7b.json

Inputs (all read-only, paths relative to --runs):

- certificate rows (``cert_v1`` and ``ans_v1``, E1-test, E2-test, E1-dev, decoding seeds 42-44):
  P7 ``eval/ckpt/rl7b-p-s1-300``, G7 ``eval/ckpt/rl7b-g-s1-300``, S7 ``eval/ckpt/cert-sft-7b-s564``,
  base 7B ``eval/zs/qwen2.5-vl-7b-instruct``, legacy TRACE-7B ``eval/zs/trace-qwen2.5-vl-7b``; 3B seed 1:
  ``rl3b-{p,g}-s1-300``, ``cert-sft-3b-s564``, ``eval/zs/{qwen2.5-vl-3b-instruct,trace-qwen2.5-vl-3b}``;
  steps 100/200 (E1-dev only): ``rl{7b,3b}-{p,g}-s1-{100,200}``; 3B training seeds 2-3 (``rl3b-{p,g}-s{2,3}-300``,
  E1-test/E2-test) only for the training-seed spread reference (P3 - G3 per seed, each a single-seed comparison).
- TRACE released validation (``trace_val/{ans_v1,no_tool}/seed*/scores.jsonl``) of every ckpt_eval dir,
  including base/legacy through ``eval/ckpt/base-*`` (used only once ``validation.json`` and
  ``validation.trace_val.json`` say ok; until then those columns are reported as pending).
- external suite (descriptive context only): ``ext_eval/runs/<served>/full/summary.seed42.json``, plus the scorer
  code hash (``scoring/score_campaign_manifest_seed_42.json``) and the fix-2/fix-3 scored-incorrect counters.
- training receipts: ``rl/{7b,3b}/{P,G}/s1/{run_receipt.json,resolved_easyr1.yaml}``.

The numeric-only comparison (selected_set dropped) is a post hoc sensitivity (DESIGN 3.1/10 motivate it; DESIGN 8's
primary family uses all rows). The scale contrast is exploratory (one training seed per scale).

A model directory is used only when its ``validation.json`` has ``ok: true``; the rest are listed as pending.
Every statistic on certificate rows reuses ``compare_cert.py`` (imported, not modified): rows paired by
(split, instance_id), seed-averaged over the decoding seeds both arms have, paired cluster bootstrap over
``scene_group`` stratified by adapter (10,000 resamples, seed 0), exact McNemar on decoding seed 42 for
binary metrics, cluster sign-flip test otherwise, Holm only within the pre-registered family (P - G on A,
E1-test and E2-test). One training seed per arm at 7B (and in the 3B seed-1 side-by-side): every CI reflects
instance sampling only. The scale contrast (P7-G7) - (P3-G3) pairs the same frozen rows across scales.

TRACE validation: pairs by instance_id, seed-averaged over ans_v1 seeds 42-44, cluster bootstrap over the
TRACE task (2 instances per task) stratified by domain, McNemar on seed 42; judge-pending rows count as
wrong (``official`` is a lower bound) and the upper bound counts them as right.

Outputs in --out: ``cert_7b.json`` (every number), ``cert_7b.md`` (rendered from the JSON only),
``cmp_7b.{json,md}`` and ``cmp_3b_s1.{json,md}`` (full compare_cert documents with per-cell scopes).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.dont_write_bytecode = True  # never leave __pycache__ next to the shared analysis scripts

VERSION = "cert_7b-1.1.0"
HERE = Path(__file__).resolve().parent

CERT_SPLITS = ("e1_test", "e2_test", "e1_dev")
METRICS = ("A", "G", "M", "F", "joint", "tokens")
ERROR_CLASSES = ("correct", "grounding", "operand", "computation", "answer_inconsistent", "envelope")

# key -> (path under --runs, scale, label, external served name or None)
MODELS: Dict[str, Tuple[str, str, str, Optional[str]]] = {
    "P7": ("eval/ckpt/rl7b-p-s1-300", "7b", "P7 grounding+semantic RL (seed 1, step 300)", "rl7b-p-s1-300"),
    "G7": ("eval/ckpt/rl7b-g-s1-300", "7b", "G7 grounding RL (seed 1, step 300)", "rl7b-g-s1-300"),
    "S7": ("eval/ckpt/cert-sft-7b-s564", "7b", "S7 structured SFT (step 564)", "cert-sft-7b-s564"),
    "base7": ("eval/zs/qwen2.5-vl-7b-instruct", "7b", "base Qwen2.5-VL-7B (zero-shot eval)", "qwen2.5-vl-7b-instruct"),
    "legacy7": ("eval/zs/trace-qwen2.5-vl-7b", "7b", "legacy TRACE-7B (answer-only RL; zero-shot eval)", "trace-qwen2.5-vl-7b"),
    "base7_ck": ("eval/ckpt/base-qwen2.5-vl-7b", "7b", "base Qwen2.5-VL-7B (ckpt_eval)", None),
    "legacy7_ck": ("eval/ckpt/base-trace-qwen2.5-vl-7b", "7b", "legacy TRACE-7B (ckpt_eval)", None),
    "P7_100": ("eval/ckpt/rl7b-p-s1-100", "7b", "P7 step 100", None),
    "P7_200": ("eval/ckpt/rl7b-p-s1-200", "7b", "P7 step 200", None),
    "G7_100": ("eval/ckpt/rl7b-g-s1-100", "7b", "G7 step 100", None),
    "G7_200": ("eval/ckpt/rl7b-g-s1-200", "7b", "G7 step 200", None),
    "P3": ("eval/ckpt/rl3b-p-s1-300", "3b", "P3 grounding+semantic RL (seed 1, step 300)", "rl3b-p-s1-300"),
    "G3": ("eval/ckpt/rl3b-g-s1-300", "3b", "G3 grounding RL (seed 1, step 300)", "rl3b-g-s1-300"),
    "S3": ("eval/ckpt/cert-sft-3b-s564", "3b", "S3 structured SFT (step 564)", "cert-sft-3b-s564"),
    "base3": ("eval/zs/qwen2.5-vl-3b-instruct", "3b", "base Qwen2.5-VL-3B (zero-shot eval)", "qwen2.5-vl-3b-instruct"),
    "legacy3": ("eval/zs/trace-qwen2.5-vl-3b", "3b", "legacy TRACE-3B (zero-shot eval)", "trace-qwen2.5-vl-3b"),
    "base3_ck": ("eval/ckpt/base-qwen2.5-vl-3b", "3b", "base Qwen2.5-VL-3B (ckpt_eval)", None),
    "legacy3_ck": ("eval/ckpt/base-trace-qwen2.5-vl-3b", "3b", "legacy TRACE-3B (ckpt_eval)", None),
    "P3_100": ("eval/ckpt/rl3b-p-s1-100", "3b", "P3 step 100", None),
    "P3_200": ("eval/ckpt/rl3b-p-s1-200", "3b", "P3 step 200", None),
    "G3_100": ("eval/ckpt/rl3b-g-s1-100", "3b", "G3 step 100", None),
    "G3_200": ("eval/ckpt/rl3b-g-s1-200", "3b", "G3 step 200", None),
    # 3B training seeds 2-3: used only as a training-seed spread reference for the single-seed 7B effects
    "P3s2": ("eval/ckpt/rl3b-p-s2-300", "3b", "P3 seed 2, step 300 (seed-spread reference only)", None),
    "G3s2": ("eval/ckpt/rl3b-g-s2-300", "3b", "G3 seed 2, step 300 (seed-spread reference only)", None),
    "P3s3": ("eval/ckpt/rl3b-p-s3-300", "3b", "P3 seed 3, step 300 (seed-spread reference only)", None),
    "G3s3": ("eval/ckpt/rl3b-g-s3-300", "3b", "G3 seed 3, step 300 (seed-spread reference only)", None),
}
SEED_PAIRS_3B = [("P3", "G3"), ("P3s2", "G3s2"), ("P3s3", "G3s3")]
SS_ADAPTER = "selected_set.chess_piece_count"
# step-300 models re-used under step names (same directories)
STEP_ALIAS = {"P7_300": "P7", "G7_300": "G7", "P3_300": "P3", "G3_300": "G3"}
CONTEXT = {"7b": ["base7", "legacy7", "S7", "G7", "P7"], "3b": ["base3", "legacy3", "S3", "G3", "P3"]}
TRACE_VAL = {"7b": ["base7_ck", "legacy7_ck", "S7", "G7", "P7"], "3b": ["base3_ck", "legacy3_ck", "S3", "G3", "P3"]}
TV_PAIRS = {"7b": [("P7", "G7"), ("P7", "S7"), ("G7", "S7"), ("S7", "base7_ck"), ("legacy7_ck", "base7_ck")],
            "3b": [("P3", "G3"), ("P3", "S3"), ("G3", "S3"), ("S3", "base3_ck"), ("legacy3_ck", "base3_ck")]}
# same weights through the zero-shot and the ckpt_eval pipelines
REPRO_PAIRS = [("base3_ck", "base3"), ("legacy3_ck", "legacy3"), ("base7_ck", "base7"), ("legacy7_ck", "legacy7")]
PROMPT_MODELS = ["P7", "G7", "S7", "base7", "legacy7", "base7_ck", "legacy7_ck"]


# ============================================================================== utilities


def load_compare_cert():
    spec = importlib.util.spec_from_file_location("compare_cert", HERE / "compare_cert.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_cert"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def fnum(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def root_of(runs: Path, key: str) -> Path:
    return runs / MODELS[STEP_ALIAS.get(key, key)][0]


def readiness(runs: Path, key: str) -> Dict[str, Any]:
    root = root_of(runs, key)
    out: Dict[str, Any] = {"path": str(root), "exists": root.is_dir(), "validation_ok": False, "trace_val_ok": False,
                           "has_trace_val_plan": False}
    v = root / "validation.json"
    if v.exists():
        d = jload(v)
        out.update(validation_ok=d.get("ok") is True, validation_done=d.get("done"), validation_expected=d.get("expected"))
    plan = root / "plan.json"
    if plan.exists():
        p = jload(plan)
        out["has_trace_val_plan"] = any(r.get("split") == "trace_val" for r in p.get("runs", []))
        out["plan_expected_total"] = p.get("expected_total")
        out["plan_runs"] = len(p.get("runs", []))
    out["cert_score_files"] = len(list(root.glob("*/*/seed*/cert/cert_scores.jsonl"))) if root.is_dir() else 0
    tv = root / "validation.trace_val.json"
    if tv.exists():
        d = jload(tv)
        out["trace_val_validation"] = {"ok": d.get("ok"), "done": d.get("done"), "expected": d.get("expected")}
        out["trace_val_ok"] = bool(out["validation_ok"] and d.get("ok") is True)
    return out


# ============================================================================== certificate comparisons


def arm_specs(runs: Path, ready: Mapping[str, Mapping[str, Any]], name: str, splits: Sequence[str], cond: str) -> List[str]:
    if not ready[name]["validation_ok"]:
        return []
    root = root_of(runs, name)
    return [f"{name}={root / s / cond}" for s in splits if (root / s / cond).is_dir() and any((root / s / cond).rglob("cert_scores.jsonl"))]


def cc_build(CC, args, specs: Sequence[str], *, condition: str, splits: Sequence[str], metrics: Sequence[str],
             pairs: Sequence[Tuple[str, str]], family_arms: Optional[Mapping[str, str]] = None, prereg: bool = False,
             cells: bool = False) -> Dict[str, Any]:
    ns = argparse.Namespace(
        arm=list(specs), condition=condition, splits=",".join(splits), metrics=",".join(metrics),
        pairs=",".join(f"{x}:{y}" for x, y in pairs) or None,
        family_arms=",".join(f"{k}={v}" for k, v in (family_arms or {}).items()) or None,
        no_preregistered=not prereg, no_cells=not cells, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
        mcnemar_seed=42, alpha=0.05, cluster_key="scene_group", stratum_key="adapter_id", trace_tool_src=args.trace_tool_src)
    return CC.build(ns)


def cc_safe(CC, args, specs: Sequence[str], **kw: Any) -> Dict[str, Any]:
    """cc_build, or arm means only when there is nothing to pair (missing arms never raise)."""
    have = present(specs)
    if not have:
        return {"arm_means": {}, "comparisons": [], "families": [], "config": {"condition": kw["condition"]}}
    if not kw.get("pairs") and not kw.get("prereg"):
        ds = CC.load_dataset(specs, condition=kw["condition"], metrics=kw["metrics"], splits=list(kw["splits"]))
        return {"arm_means": CC.arm_means(ds), "comparisons": [], "families": [], "config": {"condition": kw["condition"]}}
    return cc_build(CC, args, specs, **kw)


def present(specs: Sequence[str]) -> set:
    return {s.split("=", 1)[0] for s in specs}


def keep_pairs(pairs: Sequence[Tuple[str, str]], have: set) -> Tuple[List[Tuple[str, str]], List[str]]:
    ok = [(x, y) for x, y in pairs if x in have and y in have]
    missing = [f"{x} - {y}" for x, y in pairs if not (x in have and y in have)]
    return ok, missing


def comp_of(doc: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    for c in doc.get("comparisons", []):
        if c["name"] == name:
            return c
    return None


def slim_comp(c: Mapping[str, Any], scopes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Keep 'all' and per-adapter scopes (cells stay in the cmp_*.json documents)."""
    out = {"name": c["name"], "x": c["x"], "y": c["y"], "splits": {}}
    for split, e in c["splits"].items():
        sc = {k: v for k, v in e["scopes"].items() if (k == "all" or k.startswith("adapter:")) and (scopes is None or k in scopes)}
        out["splits"][split] = {"pairing": e["pairing"], "scopes": sc}
    return out


def scale_contrast(CC, args, runs: Path, ready, contrasts: Sequence[Tuple[Tuple[str, str], Tuple[str, str]]],
                   splits: Sequence[str]) -> Dict[str, Any]:
    """(X7 - Y7) - (X3 - Y3) on the same frozen rows (both scales score identical instance ids)."""
    names = sorted({n for c in contrasts for pair in c for n in pair})
    specs: List[str] = []
    for n in names:
        specs += arm_specs(runs, ready, n, splits, "cert_v1")
    have = present(specs)
    out: Dict[str, Any] = {"metrics": list(METRICS), "ci_scope": "instance sampling only (1 training seed per arm and scale)",
                           "contrasts": [], "missing": []}
    if not specs:
        return out
    ds = CC.load_dataset(specs, condition="cert_v1", metrics=METRICS, splits=list(splits))
    st = CC.Settings(n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)
    for (x7, y7), (x3, y3) in contrasts:
        if not {x7, y7, x3, y3} <= have:
            out["missing"].append(f"({x7} - {y7}) - ({x3} - {y3})")
            continue
        entry: Dict[str, Any] = {"name": f"({x7} - {y7}) - ({x3} - {y3})", "splits": {}}
        for split in splits:
            p7 = CC.pair_runs(ds, ds.runs[x7][""], ds.runs[y7][""], split)
            p3 = CC.pair_runs(ds, ds.runs[x3][""], ds.runs[y3][""], split)
            if not p7.keys or not p3.keys:
                continue
            i7 = {k: i for i, k in enumerate(p7.keys)}
            i3 = {k: i for i, k in enumerate(p3.keys)}
            keys = [k for k in p7.keys if k in i3]
            a7 = np.array([i7[k] for k in keys])
            a3 = np.array([i3[k] for k in keys])
            d7 = (p7.X - p7.Y)[a7]
            d3 = (p3.X - p3.Y)[a3]
            scopes: Dict[str, Any] = {}
            adapters = p7.adapter[a7]
            for scope in ["all"] + [f"adapter:{a}" for a in sorted(set(adapters.tolist()))]:
                mask = np.ones(len(keys), bool) if scope == "all" else adapters == scope.split(":", 1)[1]
                cl = CC.make_clusters(d7[mask], d3[mask], p7.cluster[a7][mask], p7.stratum[a7][mask])
                n = cl.n_rows
                delta = CC._ratio(cl.S.sum(axis=0), n)
                boot = CC.cluster_bootstrap(cl, st.n_boot, np.random.default_rng([st.seed, 11]))
                lo, hi = CC.percentile_ci(boot)
                perm = CC.signflip_p(cl, st.n_perm, np.random.default_rng([st.seed, 12]))
                res = {}
                for j, m in enumerate(METRICS):
                    res[m] = {"n_rows": int(n[j]), "effect_7b": fnum(CC._ratio(cl.SX.sum(axis=0), n)[j]),
                              "effect_3b": fnum(CC._ratio(cl.SY.sum(axis=0), n)[j]), "delta": fnum(delta[j]),
                              "ci": [fnum(lo[j]), fnum(hi[j])], "perm_p": fnum(perm[j]), "test": "cluster_signflip"}
                scopes[scope] = res
            entry["splits"][split] = {"rows": len(keys), "scopes": scopes}
        out["contrasts"].append(entry)
    return out


def numeric_only(CC, args, runs: Path, ready, pairs: Sequence[Tuple[str, str]], splits: Sequence[str],
                 exclude: str = SS_ADAPTER) -> Dict[str, Any]:
    """P - G (and RL - S) without the selected_set control adapter. Post hoc sensitivity, added after the per-adapter
    results were seen; not a pre-registered family. Motivation: DESIGN 3.1 and 10 call selected_set a plumbing/control
    adapter that re-expresses G and exclude it from go/no-go, whereas DESIGN 8's primary family is defined on all rows.
    Uses compare_cert.compare_scope with a custom row mask."""
    names = sorted({n for pr in pairs for n in pr})
    specs: List[str] = []
    for n in names:
        specs += arm_specs(runs, ready, n, splits, "cert_v1")
    have = present(specs)
    out: Dict[str, Any] = {"excluded_adapter": exclude, "pairs": {}, "missing": [],
                           "status": "post hoc sensitivity (not pre-registered); descriptive",
                           "rationale": "DESIGN 3.1 and 10: selected_set is a control adapter that re-expresses G and is "
                                        "excluded from go/no-go; DESIGN 8's primary family itself uses all rows"}
    if not specs:
        return out
    mcnemar, _ = CC.load_trace_tool(args.trace_tool_src)
    ds = CC.load_dataset(specs, condition="cert_v1", metrics=METRICS, splits=list(splits))
    st = CC.Settings(n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)
    for x, y in pairs:
        if not {x, y} <= have:
            out["missing"].append(f"{x} - {y}")
            continue
        res = {}
        for split in splits:
            p = CC.pair_runs(ds, ds.runs[x][""], ds.runs[y][""], split, 42)
            if not p.keys:
                continue
            res[split] = CC.compare_scope(ds, [(("", ""), p)], lambda q: q.adapter != exclude, st, mcnemar)
        out["pairs"][f"{x} - {y}"] = res
    return out


def seed_spread_3b(CC, args, runs: Path, ready, core7: Mapping[str, Any], no7: Mapping[str, Any],
                   splits: Sequence[str] = ("e1_test", "e2_test")) -> Dict[str, Any]:
    """P3 - G3 for 3B training seeds 1-3, each a single-seed comparison with the same code, as a reference scale for
    the one-seed 7B effects. Descriptive; the 3B multi-seed analysis itself (two-level bootstrap) is cert_3b's."""
    names = sorted({n for pr in SEED_PAIRS_3B for n in pr})
    specs: List[str] = []
    for n in names:
        specs += arm_specs(runs, ready, n, splits, "cert_v1")
    have = present(specs)
    pairs, missing = keep_pairs(SEED_PAIRS_3B, have)
    out: Dict[str, Any] = {"purpose": "training-seed spread of P - G at 3B (seeds 1-3, step 300), to scale the single-seed 7B effects",
                           "pairs": [f"{x} - {y}" for x, y in pairs], "missing": missing, "per_seed": {}, "summary": {}}
    if len(pairs) < 2:
        return out
    d = cc_build(CC, args, specs, condition="cert_v1", splits=splits, metrics=METRICS, pairs=pairs)
    no = numeric_only(CC, args, runs, ready, pairs, splits)
    c7 = next((c for c in core7.get("comparisons", []) if c["name"] == "P7 - G7"), None)
    for x, y in pairs:
        name = f"{x} - {y}"
        c = comp_of(d, name) or {"splits": {}}
        ent: Dict[str, Any] = {}
        for split in splits:
            sc = ((c["splits"].get(split) or {}).get("scopes")) or {}
            row = {k: {m: v[m] for m in ("A", "M", "joint")} for k, v in sc.items() if k == "all" or k.startswith("adapter:")}
            nr = (no["pairs"].get(name) or {}).get(split)
            if nr:
                row["numeric_only"] = {m: nr[m] for m in ("A", "M", "joint")}
            ent[split] = row
        out["per_seed"][name] = ent
    for split in splits:
        scopes = sorted({k for e in out["per_seed"].values() for k in (e.get(split) or {})})
        for scope in scopes:
            for m in ("A", "M", "joint"):
                vals = [((out["per_seed"][n].get(split) or {}).get(scope) or {}).get(m, {}).get("delta") for n in out["pairs"]]
                if any(v is None for v in vals):
                    continue
                if scope == "numeric_only":
                    r7 = ((no7.get("pairs") or {}).get("P7 - G7") or {}).get(split, {}).get(m)
                else:
                    r7 = ((((c7 or {}).get("splits") or {}).get(split) or {}).get("scopes") or {}).get(scope, {}).get(m)
                v7 = (r7 or {}).get("delta")
                lo, hi = min(vals), max(vals)
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
                out["summary"][f"{split}/{scope}/{m}"] = {
                    "split": split, "scope": scope, "metric": m, "seeds": [n.split(" - ")[0] for n in out["pairs"]],
                    "deltas_3b": vals, "min_3b": lo, "max_3b": hi, "range_width_3b": hi - lo, "sd_3b": sd,
                    "mean_3b": float(np.mean(vals)), "n_positive_3b": sum(v > 0 for v in vals), "n_seeds_3b": len(vals),
                    "delta_7b_s1": v7, "ci_7b_s1": (r7 or {}).get("ci"),
                    "delta_7b_inside_3b_range": None if v7 is None else bool(lo <= v7 <= hi),
                    "abs_delta_7b_over_sd_3b": None if v7 is None or not sd else abs(v7) / sd}
    return out


def selected_set_decomposition(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Row-weighted split of P - G on A into the selected_set and the numeric-adapter parts (exact: a mean over rows)."""
    out: Dict[str, Any] = {}
    for scale, name in (("7b", "P7 - G7"), ("3b", "P3 - G3")):
        c = next((x for x in (doc.get(f"core_{scale}") or {}).get("comparisons", []) if x["name"] == name), None)
        nr = ((doc.get("numeric_only") or {}).get("pairs") or {}).get(name) or {}
        if not c:
            continue
        for split in ("e1_test", "e2_test"):
            sc = ((c["splits"].get(split) or {}).get("scopes")) or {}
            a, s, n = (sc.get("all") or {}).get("A"), (sc.get(f"adapter:{SS_ADAPTER}") or {}).get("A"), (nr.get(split) or {}).get("A")
            if not (a and s and n):
                continue
            cs = s["delta"] * s["n_rows"] / a["n_rows"]
            cn = n["delta"] * n["n_rows"] / a["n_rows"]
            out[f"{name}/{split}"] = {
                "comparison": name, "split": split, "metric": "A", "delta_all": a["delta"], "n_all": a["n_rows"],
                "delta_selected_set": s["delta"], "n_selected_set": s["n_rows"], "contribution_selected_set": cs,
                "delta_numeric": n["delta"], "n_numeric": n["n_rows"], "contribution_numeric": cn,
                "share_selected_set": cs / a["delta"] if a["delta"] else None,
                "sum_check_abs_diff": abs(cs + cn - a["delta"])}
    return out


# ============================================================================== error classes and 2x2


def error_classes(runs: Path, ready, models: Sequence[str], splits: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in models:
        if not ready[m]["validation_ok"]:
            continue
        root = root_of(runs, m)
        for split in splits:
            files = sorted((root / split / "cert_v1").glob("seed*/cert/cert_scores.jsonl"))
            if not files:
                continue
            cnt: Counter = Counter()
            by_ad: Dict[str, Counter] = defaultdict(Counter)
            n = a1m = a0m1 = 0
            for f in files:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("condition") != "cert_v1":
                        continue
                    ec = str(r.get("error_class"))
                    cnt[ec] += 1
                    by_ad[str(r.get("adapter_id"))][ec] += 1
                    n += 1
                    A, M = r.get("A"), r.get("M")
                    if A is not None and M is not None:
                        a1m += int(A >= 1.0 and M < 1.0)
                        a0m1 += int(A < 1.0 and M >= 1.0)
            out.setdefault(m, {})[split] = {
                "rows_pooled_over_seeds": n, "files": len(files), "counts": dict(sorted(cnt.items())),
                "rates": {k: v / n for k, v in sorted(cnt.items())},
                "A1_M_lt1": a1m, "A0_M1": a0m1, "A1_M_lt1_rate": a1m / n if n else None, "A0_M1_rate": a0m1 / n if n else None,
                "by_adapter": {a: {"n": sum(c.values()), "rates": {k: v / sum(c.values()) for k, v in sorted(c.items())}}
                               for a, c in sorted(by_ad.items())},
            }
    return out


def error_class_paired(CC, args, runs: Path, ready, pairs: Sequence[Tuple[str, str]], splits: Sequence[str]) -> Dict[str, Any]:
    """Paired difference of error-class indicator rates (seed-averaged rows, cluster bootstrap, sign-flip p)."""
    cols = list(ERROR_CLASSES) + ["A1_M_lt1", "A0_M1"]

    def load(m: str) -> Tuple[Dict[Tuple[str, str], Dict[int, np.ndarray]], Dict[Tuple[str, str], Tuple[str, str, str]]]:
        rows: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}
        meta: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
        root = root_of(runs, m)
        for split in splits:
            for f in sorted((root / split / "cert_v1").glob("seed*/cert/cert_scores.jsonl")):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("condition") != "cert_v1":
                        continue
                    k = (r["split"], r["instance_id"])
                    ec = str(r.get("error_class"))
                    A, M = r.get("A"), r.get("M")
                    v = [float(ec == c) for c in ERROR_CLASSES]
                    v.append(float(A is not None and M is not None and A >= 1.0 and M < 1.0))
                    v.append(float(A is not None and M is not None and A < 1.0 and M >= 1.0))
                    rows.setdefault(k, {})[int(r["seed"])] = np.array(v)
                    meta[k] = (str(r.get("scene_group") or r["instance_id"]), str(r.get("adapter_id")), str(r.get("adapter_id")))
        return rows, meta

    cache: Dict[str, Any] = {}
    out: Dict[str, Any] = {"columns": cols, "pairs": {}}
    for x, y in pairs:
        if not (ready[x]["validation_ok"] and ready[y]["validation_ok"]):
            continue
        for m in (x, y):
            if m not in cache:
                cache[m] = load(m)
        (rx, mx), (ry, _) = cache[x], cache[y]
        res: Dict[str, Any] = {}
        for split in splits:
            keys = sorted(k for k in rx if k[0] == split and k in ry)
            if not keys:
                continue
            X = np.zeros((len(keys), len(cols)))
            Y = np.zeros((len(keys), len(cols)))
            for i, k in enumerate(keys):
                common = sorted(set(rx[k]) & set(ry[k]))
                X[i] = np.mean([rx[k][s] for s in common], axis=0)
                Y[i] = np.mean([ry[k][s] for s in common], axis=0)
            cl = CC.make_clusters(X, Y, np.array([mx[k][0] for k in keys], dtype=object), np.array([mx[k][1] for k in keys], dtype=object))
            n = cl.n_rows
            delta = CC._ratio(cl.S.sum(axis=0), n)
            boot = CC.cluster_bootstrap(cl, args.n_boot, np.random.default_rng([args.seed, 21]))
            lo, hi = CC.percentile_ci(boot)
            perm = CC.signflip_p(cl, args.n_perm, np.random.default_rng([args.seed, 22]))
            res[split] = {"rows": len(keys), "cols": {c: {"mean_x": fnum(CC._ratio(cl.SX.sum(axis=0), n)[j]),
                                                          "mean_y": fnum(CC._ratio(cl.SY.sum(axis=0), n)[j]),
                                                          "delta": fnum(delta[j]), "ci": [fnum(lo[j]), fnum(hi[j])],
                                                          "perm_p": fnum(perm[j])} for j, c in enumerate(cols)}}
        out["pairs"][f"{x} - {y}"] = res
    return out


# ============================================================================== TRACE validation


def load_trace_val(runs: Path, m: str) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    out: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    for f in sorted((root_of(runs, m) / "trace_val").glob("*/seed*/scores.jsonl")):
        cond, seed = f.parent.parent.name, int(f.parent.name.replace("seed", ""))
        rows = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["instance_id"]] = {
                "task": r["task"], "domain": r["domain"], "off": float(r["official_correct"]),
                "strict": float(r["strict_correct"]), "det": float(r["deterministic_correct"]),
                "pend": float(r.get("official_source") == "judge_pending"), "tok": float(r.get("completion_tokens") or 0),
                "complete": float(r.get("stop_reason") == "answer"),
            }
        out.setdefault(cond, {})[seed] = rows
    return out


def trace_val_protocol(runs: Path, m: str) -> Dict[str, Any]:
    """Token cap, decoding temperature and seeds of each TRACE-val condition, from its run_config.json files."""
    out: Dict[str, Any] = {}
    for f in sorted((root_of(runs, m) / "trace_val").glob("*/seed*/run_config.json")):
        cond, seed = f.parent.parent.name, int(f.parent.name.replace("seed", ""))
        rc = jload(f)
        e = out.setdefault(cond, {"seeds": [], "max_total_tokens": set(), "temperature": set()})
        e["seeds"].append(seed)
        e["max_total_tokens"].add(dig(rc, "settings.max_total_tokens"))
        e["temperature"].add(dig(rc, "settings.decoding_params.temperature"))
    return {c: {"seeds": sorted(v["seeds"]), "max_total_tokens": sorted(v["max_total_tokens"], key=str),
                "temperature": sorted(v["temperature"], key=str)} for c, v in out.items()}


def trace_val_summary(tv: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cond, by_seed in tv.items():
        per_seed = {}
        allrows: List[Mapping[str, Any]] = []
        for seed, rows in sorted(by_seed.items()):
            vals = list(rows.values())
            allrows += vals
            n = len(vals)
            per_seed[str(seed)] = {
                "n": n, "official_lower": sum(r["off"] for r in vals) / n,
                "official_upper": sum(min(1.0, r["off"] + r["pend"]) for r in vals) / n,
                "strict": sum(r["strict"] for r in vals) / n, "deterministic": sum(r["det"] for r in vals) / n,
                "judge_pending": int(sum(r["pend"] for r in vals)), "tokens_mean": sum(r["tok"] for r in vals) / n,
                "completion_rate": sum(r["complete"] for r in vals) / n}
        n = len(allrows)
        dom: Dict[str, List[float]] = defaultdict(list)
        for r in allrows:
            dom[r["domain"]].append(r["off"])
        out[cond] = {
            "seeds": sorted(by_seed), "per_seed": per_seed,
            "pooled": {"n": n, "official_lower": sum(r["off"] for r in allrows) / n,
                       "official_upper": sum(min(1.0, r["off"] + r["pend"]) for r in allrows) / n,
                       "strict": sum(r["strict"] for r in allrows) / n, "deterministic": sum(r["det"] for r in allrows) / n,
                       "judge_pending": int(sum(r["pend"] for r in allrows)), "tokens_mean": sum(r["tok"] for r in allrows) / n,
                       "completion_rate": sum(r["complete"] for r in allrows) / n},
            "by_domain_official_lower": {d: sum(v) / len(v) for d, v in sorted(dom.items())},
        }
    return out


def trace_val_pair(CC, args, mcnemar, X, Y, cond: str = "ans_v1") -> Optional[Dict[str, Any]]:
    if cond not in X or cond not in Y:
        return None
    seeds = sorted(set(X[cond]) & set(Y[cond]))
    if not seeds:
        return None
    keys = set.intersection(*[set(X[cond][s]) for s in seeds], *[set(Y[cond][s]) for s in seeds])
    keys = sorted(keys)
    cols = ("off", "strict", "pend", "tok")
    Xv = np.array([[np.mean([X[cond][s][k][c] for s in seeds]) for c in cols] for k in keys])
    Yv = np.array([[np.mean([Y[cond][s][k][c] for s in seeds]) for c in cols] for k in keys])
    first = X[cond][seeds[0]]
    cl = CC.make_clusters(Xv, Yv, np.array([first[k]["task"] for k in keys], dtype=object),
                          np.array([first[k]["domain"] for k in keys], dtype=object))
    n = cl.n_rows
    delta = CC._ratio(cl.S.sum(axis=0), n)
    boot = CC.cluster_bootstrap(cl, args.n_boot, np.random.default_rng([args.seed, 31]))
    lo, hi = CC.percentile_ci(boot)
    perm = CC.signflip_p(cl, args.n_perm, np.random.default_rng([args.seed, 32]))
    mc = None
    if 42 in X[cond] and 42 in Y[cond]:
        x42 = np.array([X[cond][42][k]["off"] for k in keys])
        y42 = np.array([Y[cond][42][k]["off"] for k in keys])
        b, c, nn = CC.mcnemar_counts(x42, y42)
        mc = {"seed": 42, "b_y_only": b, "c_x_only": c, "n": nn, "p": float(mcnemar(b, c)), "outcome": "official_lower"}
    px, py = float(Xv[:, 2].mean()), float(Yv[:, 2].mean())
    d_off = fnum(delta[0])
    return {
        "condition": cond, "decoding_seeds": seeds, "rows": len(keys), "n_clusters_tasks": int(cl.S.shape[0]),
        "mean_x_official_lower": float(Xv[:, 0].mean()), "mean_y_official_lower": float(Yv[:, 0].mean()),
        "delta_official_lower": d_off, "ci_official_lower": [fnum(lo[0]), fnum(hi[0])], "perm_p_official_lower": fnum(perm[0]),
        "mcnemar": mc,
        "pending_rate_x": px, "pending_rate_y": py,
        "delta_bounds_if_pending_resolved": [d_off - py, d_off + px] if d_off is not None else None,
        "delta_strict": fnum(delta[1]), "ci_strict": [fnum(lo[1]), fnum(hi[1])],
        "delta_tokens": fnum(delta[3]),
        "ci_scope": "instance sampling only (1 training seed); clusters = TRACE task (2 instances each), strata = domain",
    }


# ============================================================================== external (descriptive)


EXT_COUNTERS = ("evaluator_exception_scored_incorrect", "mathverse_all_retries_failed_scored_incorrect",
                "extraction_failed_scored_incorrect")
FIX3_COUNTERS = ("mathverse_all_retries_failed_scored_incorrect", "extraction_failed_scored_incorrect")


def external(runs: Path, keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        served = MODELS[k][3]
        if not served:
            continue
        p = runs / "ext_eval" / "runs" / served / "full" / "summary.seed42.json"
        if not p.exists():
            out[k] = {"served": served, "status": "pending (no summary.seed42.json)", "path": str(p)}
            continue
        d = jload(p)
        mac, ex = d.get("macro") or {}, d.get("extraction") or {}
        complete = d.get("complete") is True
        man = p.parent / "scoring" / "score_campaign_manifest_seed_42.json"
        code_hash = dig(jload(man), "contract.trace_eval_code_provenance.sha256") if man.exists() else None
        counters: Dict[str, Any] = {}
        for key in EXT_COUNTERS:
            v = d.get(key)
            counters[key] = None if v is None else (v.get("total") if isinstance(v, dict) else v)
        # upper bound on the overall_24 shift if every fix-3 row had been scored correct (benchmark score = row accuracy)
        bound = 0.0
        n_bench = 0
        for b in d.get("benchmarks") or []:
            n_bench += 1
            cnt = sum(int(b.get(kk) or 0) for kk in FIX3_COUNTERS)
            if cnt and b.get("rows"):
                bound += 100.0 * cnt / b["rows"]
        fix3_rows = None if all(counters[kk] is None for kk in FIX3_COUNTERS) else sum(int(counters[kk] or 0) for kk in FIX3_COUNTERS)
        out[k] = {
            "code_hash": code_hash, "code_hash_short": code_hash[:12] if code_hash else None, "scoring_manifest": str(man),
            "counters": counters, "fix3_rows": fix3_rows, "fix3_recorded": fix3_rows is not None,
            "fix3_overall_24_shift_bound_pp": (bound / n_bench if n_bench else None) if fix3_rows is not None else None,
            "served": served, "path": str(p), "sha256": sha256_file(p), "created": d.get("created"), "complete": complete,
            "status": "complete" if complete else f"pending (incomplete: {d.get('benchmarks_scored')}/24 benchmarks scored)",
            "overall_24": mac.get("overall_24") if complete else None, "natural_9": mac.get("natural_9") if complete else None,
            "chart_diagram_synthetic_15": mac.get("chart_diagram_synthetic_15") if complete else None,
            "mmstar": mac.get("mmstar") if complete else None,
            "det_fail_rate": ex.get("det_fail_rate") if complete else None, "cert_json_rate": ex.get("cert_json_rate") if complete else None,
            "judge_needed_rate_proxy": ex.get("judge_needed_rate_proxy") if complete else None,
        }
    return out


# ============================================================================== training match (run receipts)


def _ts(x: Optional[str]) -> Optional[float]:
    if not x:
        return None
    from datetime import datetime
    return datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def _resolved_rollout(p: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """(data.rollout_batch_size, worker.rollout.n, trainer.max_steps) from the run's resolved EasyR1 config."""
    if not p.exists():
        return None, None, None
    try:
        import yaml  # type: ignore
        y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - missing PyYAML or a bad file: report the fields as unknown
        return None, None, None
    g = lambda *ks: dig(y, ".".join(ks))  # noqa: E731
    return g("data", "rollout_batch_size"), g("worker", "rollout", "n"), g("trainer", "max_steps")


def training_match(runs: Path) -> Dict[str, Any]:
    """P vs G of one seed: same init, data, prompt, seed, GPU class? Rollout length and wall time per arm."""
    out: Dict[str, Any] = {}
    for scale, pair in (("7b", ("P", "G")), ("3b", ("P", "G"))):
        arms: Dict[str, Any] = {}
        for a in pair:
            f = runs / "rl" / scale / a / "s1" / "run_receipt.json"
            if not f.exists():
                arms[a] = {"path": str(f), "status": "missing"}
                continue
            r = jload(f)
            ident, data, md, summ = r.get("identity") or {}, r.get("data") or {}, r.get("model_dir") or {}, r.get("summary") or {}
            atts = r.get("attempts") or []
            hours = sum(((_ts(t.get("ended")) or 0) - (_ts(t.get("started")) or 0)) / 3600 for t in atts if t.get("ended") and t.get("started"))
            resp = (summ.get("reward_mean") or {}).get("resp_len")
            rb, rn, ms = _resolved_rollout(f.parent / "resolved_easyr1.yaml")
            per_step = rb * rn if rb and rn else None
            arms[a] = {
                "rollout_batch_size": rb, "rollout_n": rn, "max_steps": ms, "rollouts_per_step": per_step,
                "path": str(f), "status": r.get("status"), "seed": ident.get("seed"), "config_id": ident.get("config_id"),
                "n_gpus": r.get("n_gpus"), "gpu": (r.get("gpu") or {}).get("name"), "hosts": sorted({t.get("host") for t in atts}),
                "attempts": len(atts), "resumed_from_steps": [t.get("resumed_from_step") for t in atts],
                "last_step": atts[-1].get("last_step") if atts else None,
                "init_path": md.get("path"), "init_config_sha256": md.get("config_sha256"),
                "train_sha256": (data.get("train") or {}).get("sha256"), "val_sha256": (data.get("val") or {}).get("sha256"),
                "store_sha256": (data.get("store") or {}).get("sha256"), "system_prompt_sha256": (data.get("system_prompt") or {}).get("sha256"),
                "wall_hours": round(hours, 2), "gpu_hours": round(hours * (r.get("n_gpus") or 0), 2),
                "step_seconds_median": summ.get("step_seconds_median"), "train_resp_len_mean": resp,
                "rollout_tokens_approx": int(resp * per_step * (atts[-1].get("last_step") or 0)) if resp and atts and per_step else None,
            }
        same = ["seed", "config_id", "n_gpus", "gpu", "init_path", "init_config_sha256", "train_sha256", "val_sha256", "store_sha256",
                "system_prompt_sha256", "last_step", "rollout_batch_size", "rollout_n", "max_steps"]
        vals = [arms[a] for a in pair if arms[a].get("status") != "missing"]
        matched = {k: len({json.dumps(v.get(k)) for v in vals}) == 1 for k in same} if len(vals) == 2 else {}
        out[scale] = {"arms": arms, "matched_fields": matched, "all_matched": bool(matched) and all(matched.values())}
    return out


# ============================================================================== comparability audit


RUN_FIELDS = {
    "protocol": ["condition", "dataset.sha256", "dataset.split", "settings.system_prompt_sha256", "settings.decoding",
                 "settings.decoding_params.temperature", "settings.decoding_params.top_p", "settings.decoding_params.top_k",
                 "settings.decoding_params.repetition_penalty", "settings.decoding_params.presence_penalty",
                 "settings.max_tokens_per_turn", "settings.max_total_tokens", "settings.max_turns", "settings.min_pixels",
                 "settings.max_pixels", "settings.coords", "settings.patch_factor", "settings.family", "settings.tools",
                 "settings.oracle_crop", "seed"],
    "software": ["trace_tool_version", "code_sha256", "git_commit", "git_dirty", "python"],
}
RECEIPT_FIELDS = ["scorer_version", "provenance.cert_code_sha256", "provenance.git_head", "store.sha256", "export_receipt.sha256",
                  "provenance.resource_sha256.cert_prompts_v1.json", "provenance.resource_sha256.cert_system_v1.txt",
                  "checks.prompt_check", "checks.dataset_frozen", "checks.settings_problems", "responses.endpoint_errors",
                  "responses.rejected", "run.coord_frame"]
VLLM_DROP = {"model_tag", "model", "port", "revision", "tokenizer", "served_model_name", "host", "api_server_count",
             "data_parallel_size", "tensor_parallel_size", "tokenizer_revision"}


def dig(d: Any, dotted: str) -> Any:
    cur = d
    parts = dotted.split(".")
    i = 0
    while i < len(parts):
        if not isinstance(cur, dict):
            return None
        # keys may themselves contain dots (resource file names)
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if k in cur:
                cur = cur[k]
                i = j
                break
        else:
            return None
    return cur


def vllm_info(root: Path) -> Dict[str, Any]:
    versions, argsets, nodes = set(), set(), set()
    for f in sorted((root / "logs").glob("vllm.*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"version (\d+\.\d+\.\d+)", txt):
            versions.add(m.group(1))
        for m in re.finditer(r"non-default args: (\{.*\})", txt):
            try:
                a = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                continue
            argsets.add(json.dumps({k: v for k, v in a.items() if k not in VLLM_DROP}, sort_keys=True))
    for f in sorted((root / "logs").glob("job.*/claims.txt")):
        for m in re.finditer(r"claimed trc_[a-z]+_\d+_([A-Za-z0-9]+)_", f.read_text(encoding="utf-8", errors="replace")):
            nodes.add(m.group(1))
    tc = root / "template_check.json"
    tcd = jload(tc) if tc.exists() else None
    return {"vllm_versions": sorted(versions), "vllm_args": sorted(argsets), "eval_nodes": sorted(nodes),
            "template_check": None if tcd is None else {k: tcd.get(k) for k in ("template_equal", "token_ids_equal", "geometry_equal", "no_server_resize")}}


def prompt_fingerprints(path: Path) -> Dict[str, Tuple[str, int, str]]:
    """instance_id -> (sha256 of system + user text + image size, final_prompt_tokens, sha256 of final_response)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        parts = []
        for msg in r.get("transcript") or []:
            if msg.get("role") not in ("system", "user"):
                continue
            c = msg.get("content")
            if isinstance(c, str):
                parts.append(f"{msg['role']}:{c}")
            else:
                for x in c or []:
                    if x.get("type") == "text":
                        parts.append(f"{msg['role']}:text:{x.get('text')}")
                    elif x.get("type") == "image":
                        parts.append(f"{msg['role']}:image:{x.get('size')}")
            if msg.get("role") == "user":
                break  # first user turn only (single-turn conditions)
        h = hashlib.sha256("\n".join(parts).encode()).hexdigest()
        resp = hashlib.sha256(str(r.get("final_response")).encode()).hexdigest()
        out[r["instance_id"]] = (h, int(r.get("final_prompt_tokens") or -1), resp)
    return out


def comparability(runs: Path, ready, models: Sequence[str]) -> Dict[str, Any]:
    use = [m for m in models if ready[m]["validation_ok"]]
    groups: Dict[Tuple[str, str], Dict[str, Dict[str, set]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    n_runs = Counter()
    for m in use:
        root = root_of(runs, m)
        for cfg in sorted(root.glob("*/*/seed*/run_config.json")):
            split, cond = cfg.parent.parent.parent.name, cfg.parent.parent.name
            if split not in CERT_SPLITS or cond not in ("cert_v1", "ans_v1"):
                continue
            n_runs[m] += 1
            rc = jload(cfg)
            for cat, fields in RUN_FIELDS.items():
                for f in fields:
                    groups[(split, cond)][f"{cat}:{f}"][m].add(json.dumps(dig(rc, f), sort_keys=True))
            rec = cfg.parent / "cert" / "score_receipt.json"
            if rec.exists():
                rd = jload(rec)
                for f in RECEIPT_FIELDS:
                    groups[(split, cond)][f"scoring:{f}"][m].add(json.dumps(dig(rd, f), sort_keys=True))
                ad = rd.get("adapters") or {}
                flat = sorted(f"{v.get('id')}={v.get('verifier_version')}/{v.get('schema_id')}" for v in ad.values() if isinstance(v, dict))
                groups[(split, cond)]["scoring:adapters.verifier_versions"][m].add(json.dumps(flat))
                pc = dig(rd, "checks.prompts_checked")
                rows = dig(rd, "responses.scored")
                groups[(split, cond)]["scoring:prompts_checked_equals_rows"][m].add(json.dumps(pc == rows if pc is not None else None))
    fields_out: Dict[str, Any] = {}
    differing: List[Dict[str, Any]] = []
    for (split, cond), fields in sorted(groups.items()):
        for f, by_model in sorted(fields.items()):
            sets = {frozenset(v) for v in by_model.values()}
            entry = {"split": split, "cond": cond, "field": f, "identical": len(sets) == 1,
                     "values": {m: sorted(v) for m, v in by_model.items()}}
            fields_out[f"{split}/{cond}/{f}"] = entry
            if not entry["identical"]:
                differing.append(entry)
    subsets = {"7b": [m for m in use if MODELS[STEP_ALIAS.get(m, m)][1] == "7b"],
               "3b": [m for m in use if MODELS[STEP_ALIAS.get(m, m)][1] == "3b"]}
    verdict = {}
    for name, ms in subsets.items():
        bad = sorted({e["field"] for e in fields_out.values()
                      if len({frozenset(v) for m, v in e["values"].items() if m in ms}) > 1})
        verdict[name] = {"models": ms, "all_fields_identical": not bad, "differing_fields": bad}
    serving = {m: vllm_info(root_of(runs, m)) for m in use}
    serving_versions = sorted({v for s in serving.values() for v in s["vllm_versions"]})
    serving_args = sorted({a for s in serving.values() for a in s["vllm_args"]})
    by_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: {"identical": 0, "differing": 0})
    for e in fields_out.values():
        by_cat[e["field"].split(":", 1)[0]]["identical" if e["identical"] else "differing"] += 1
    values_of = {f: sorted({json.loads(v) for e in fields_out.values() if e["field"] == f for vals in e["values"].values() for v in vals}, key=str)
                 for f in ("protocol:settings.decoding_params.temperature", "protocol:settings.decoding_params.top_p",
                           "protocol:settings.max_total_tokens", "protocol:seed")}
    return {"models": use, "run_configs_read": dict(n_runs), "field_checks": len(fields_out), "by_category": dict(by_cat),
            "values_of": values_of,
            "differing": differing, "verdict": verdict, "serving": serving, "vllm_versions_all": serving_versions, "vllm_args_all": serving_args,
            "vllm_identical": len(serving_versions) == 1 and len(serving_args) == 1}


def prompt_identity(runs: Path, ready, models: Sequence[str], ref: str = "P7") -> Dict[str, Any]:
    """Seed-42 prompts (system + user text + image size) and prompt token counts against the reference model."""
    use = [m for m in models if ready[m]["validation_ok"]]
    out: Dict[str, Any] = {"reference": ref, "seed": 42, "by_model": {}}
    if ref not in use:
        return out
    cache: Dict[Tuple[str, str, str], Dict[str, Tuple[str, int, str]]] = {}

    def fp(m: str, split: str, cond: str) -> Optional[Dict[str, Tuple[str, int, str]]]:
        p = root_of(runs, m) / split / cond / "seed42" / "responses.jsonl"
        if not p.exists():
            return None
        key = (m, split, cond)
        if key not in cache:
            cache[key] = prompt_fingerprints(p)
        return cache[key]

    for m in use:
        res = {}
        for split in CERT_SPLITS:
            for cond in ("cert_v1", "ans_v1"):
                a, b = fp(ref, split, cond), fp(m, split, cond)
                if a is None or b is None:
                    continue
                keys = sorted(set(a) & set(b))
                res[f"{split}/{cond}"] = {
                    "rows": len(keys), "prompt_text_identical": sum(a[k][0] == b[k][0] for k in keys),
                    "prompt_tokens_identical": sum(a[k][1] == b[k][1] for k in keys)}
        out["by_model"][m] = res
    return out


def repro_identical_responses(runs: Path, ready, pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    out = {}
    for x, y in pairs:
        if not (ready[x]["validation_ok"] and ready[y]["validation_ok"]):
            continue
        res = {}
        for split in CERT_SPLITS:
            for cond in ("cert_v1", "ans_v1"):
                for seed in (42, 43, 44):
                    px = root_of(runs, x) / split / cond / f"seed{seed}" / "responses.jsonl"
                    py = root_of(runs, y) / split / cond / f"seed{seed}" / "responses.jsonl"
                    if not (px.exists() and py.exists()):
                        continue
                    a, b = prompt_fingerprints(px), prompt_fingerprints(py)
                    keys = sorted(set(a) & set(b))
                    res[f"{split}/{cond}/seed{seed}"] = {"rows": len(keys), "identical_response": sum(a[k][2] == b[k][2] for k in keys),
                                                        "identical_prompt": sum(a[k][0] == b[k][0] for k in keys)}
        out[f"{x} - {y}"] = res
    return out


# ============================================================================== cross-checks


def crosscheck_means(runs: Path, ready, docs: Sequence[Mapping[str, Any]], cond_tag: Mapping[str, str]) -> Dict[str, Any]:
    """compare_cert arm means against the pipeline's own summary.<split>.json (pooled; equal when every row has 3 seeds)."""
    checks, worst = [], 0.0
    for doc in docs:
        cond = doc["config"]["condition"]
        for arm, by in doc["arm_means"].items():
            if arm not in MODELS and arm not in STEP_ALIAS:
                continue
            for split, v in by["-"].items():
                p = root_of(runs, arm) / f"summary.{split}.json"
                if not p.exists():
                    continue
                ent = [e for e in jload(p).get("all", []) if e.get("cond") == cond_tag[cond]]
                if not ent:
                    continue
                for m in ("A", "G", "M", "F", "joint"):
                    if m in v["means"] and v["means"][m] is not None and ent[0].get(m) is not None:
                        d = abs(v["means"][m] - ent[0][m])
                        worst = max(worst, d)
                        checks.append({"arm": arm, "split": split, "cond": cond, "metric": m, "abs_diff": d})
    return {"n_checks": len(checks), "max_abs_diff": worst, "ok": worst < 1e-9, "checks_failing": [c for c in checks if c["abs_diff"] >= 1e-9]}


def crosscheck_stage1(runs: Path, core3: Mapping[str, Any]) -> Dict[str, Any]:
    """The 3B seed-1 primary family must reproduce results/stage1_interim/cmp_PGS.json (same rows, code and seed)."""
    f = runs / "results" / "stage1_interim" / "cmp_PGS.json"
    if not f.exists() or not core3.get("families"):
        return {"available": False, "path": str(f)}
    old = {(t["split"], t["metric"]): t for fam in jload(f)["families"] if fam["family"] == "primary" for t in fam["tests"] if t.get("result")}
    rows, worst = [], 0.0
    for fam in core3["families"]:
        if fam["family"] != "primary":
            continue
        for t in fam["tests"]:
            o = old.get((t["split"], t["metric"]))
            if not o or not t.get("result"):
                continue
            a, b = t["result"], o["result"]
            diffs = [abs(a["delta"] - b["delta"]), abs(a["ci"][0] - b["ci"][0]), abs(a["ci"][1] - b["ci"][1]),
                     abs(a["p"] - b["p"]), abs(t["holm_p"] - o["holm_p"])]
            worst = max(worst, max(diffs))
            rows.append({"split": t["split"], "metric": t["metric"], "ours": [a["delta"], a["ci"], a["p"], t["holm_p"]],
                         "stage1_interim": [b["delta"], b["ci"], b["p"], o["holm_p"]], "max_abs_diff": max(diffs)})
    return {"available": True, "path": str(f), "sha256": sha256_file(f), "rows": rows, "max_abs_diff": worst, "ok": bool(rows) and worst < 1e-9}


def crosscheck_spread(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The seed-1 row of the 3B seed spread is computed in a separate compare_cert call; it must equal core_3b's P3 - G3."""
    ss = (doc.get("seed_spread_3b") or {}).get("per_seed", {}).get("P3 - G3")
    c3 = next((c for c in (doc.get("core_3b") or {}).get("comparisons", []) if c["name"] == "P3 - G3"), None)
    if not ss or not c3:
        return {"available": False}
    worst, n = 0.0, 0
    for split, scopes in ss.items():
        for scope, ms in scopes.items():
            if scope == "numeric_only":
                ref = ((doc.get("numeric_only") or {}).get("pairs", {}).get("P3 - G3") or {}).get(split) or {}
            else:
                ref = ((c3["splits"].get(split) or {}).get("scopes") or {}).get(scope) or {}
            for m, r in ms.items():
                o = ref.get(m)
                if not o:
                    continue
                for a, b in ((r["delta"], o["delta"]), (r["ci"][0], o["ci"][0]), (r["ci"][1], o["ci"][1])):
                    worst = max(worst, abs(a - b))
                n += 1
    return {"available": True, "n_checks": n, "max_abs_diff": worst, "ok": n > 0 and worst < 1e-12}


def crosscheck_trace_val(runs: Path, tvs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    out, worst = [], 0.0
    for m, s in tvs.items():
        p = root_of(runs, m) / "summary.trace_val.json"
        if not p.exists():
            continue
        d = jload(p).get("by_cond", {})
        for cond, v in s.items():
            if cond in d and d[cond].get("official_acc_lower") is not None:
                diff = abs(v["pooled"]["official_lower"] - d[cond]["official_acc_lower"])
                worst = max(worst, diff)
                out.append({"model": m, "cond": cond, "ours": v["pooled"]["official_lower"], "pipeline": d[cond]["official_acc_lower"],
                            "pending_ours": v["pooled"]["judge_pending"], "pending_pipeline": d[cond].get("judge_pending")})
    return {"checks": out, "max_abs_diff": worst, "ok": worst < 5e-4 and all(c["pending_ours"] == c["pending_pipeline"] for c in out)}


# ============================================================================== build


def build(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    t0 = time.time()
    runs = Path(args.runs)
    CC = load_compare_cert()
    mcnemar, holm = CC.load_trace_tool(args.trace_tool_src)
    keys = list(MODELS) + list(STEP_ALIAS)
    ready = {k: readiness(runs, k) for k in keys}
    side: Dict[str, Dict[str, Any]] = {}
    doc: Dict[str, Any] = {
        "version": VERSION, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"runs": str(runs), "n_boot": args.n_boot, "n_perm": args.n_perm, "seed": args.seed, "mcnemar_seed": 42,
                   "cluster_key": "scene_group", "stratum_key": "adapter_id", "metrics": list(METRICS),
                   "ci_scope": "instance sampling only: one training seed per arm (7B and the 3B seed-1 side by side)"},
        "code": {"cert_7b.py": sha256_file(Path(__file__).resolve()), "compare_cert.py": sha256_file(HERE / "compare_cert.py"),
                 "compare_cert_version": CC.VERSION},
        "models": {k: {"label": MODELS[STEP_ALIAS.get(k, k)][2], "scale": MODELS[STEP_ALIAS.get(k, k)][1], **ready[k]} for k in keys},
    }

    # ---- 1. core comparisons with cells (P - G family, RL vs SFT) at each scale
    for scale, (P, G, S), name in (("7b", ("P7", "G7", "S7"), "cmp_7b"), ("3b", ("P3", "G3", "S3"), "cmp_3b_s1")):
        specs = [s for a in (P, G, S) for s in arm_specs(runs, ready, a, CERT_SPLITS, "cert_v1")]
        have = present(specs)
        pairs, missing = keep_pairs([(P, G), (P, S), (G, S)], have)
        if len(have) >= 2 and pairs:
            d = cc_build(CC, args, specs, condition="cert_v1", splits=CERT_SPLITS, metrics=METRICS, pairs=pairs,
                         family_arms={"P": P, "G": G}, prereg=True, cells=True)
            side[name] = d
            doc[f"core_{scale}"] = {"arms": [P, G, S], "missing_pairs": missing, "families": d["families"],
                                    "arm_means": d["arm_means"], "comparisons": [slim_comp(c) for c in d["comparisons"]],
                                    "full_document": f"{name}.json"}
        else:
            doc[f"core_{scale}"] = {"arms": [P, G, S], "pending": True, "missing_pairs": missing}

    # ---- 2. context: base / legacy / S / G / P on cert_v1 (means and descriptive pairs, no cells)
    ctx_pairs = {"7b": [("S7", "base7"), ("legacy7", "base7"), ("P7", "legacy7"), ("G7", "legacy7")],
                 "3b": [("S3", "base3"), ("legacy3", "base3"), ("P3", "legacy3"), ("G3", "legacy3")]}
    for scale in ("7b", "3b"):
        specs = [s for a in CONTEXT[scale] for s in arm_specs(runs, ready, a, CERT_SPLITS, "cert_v1")]
        have = present(specs)
        pairs, missing = keep_pairs(ctx_pairs[scale], have)
        d = ctx_doc = cc_safe(CC, args, specs, condition="cert_v1", splits=CERT_SPLITS, metrics=METRICS + ("within_2048",), pairs=pairs)
        doc[f"context_{scale}"] = {"arms": CONTEXT[scale], "missing_pairs": missing, "arm_means": d["arm_means"],
                                   "comparisons": [slim_comp(c, ["all"]) for c in d["comparisons"]]}
        # answer-only prompt on the same frozen rows
        specs = [s for a in CONTEXT[scale] for s in arm_specs(runs, ready, a, CERT_SPLITS, "ans_v1")]
        have = present(specs)
        P, G, S, base = CONTEXT[scale][4], CONTEXT[scale][3], CONTEXT[scale][2], CONTEXT[scale][0]
        pairs, missing = keep_pairs([(P, G), (S, base)], have)
        d = cc_safe(CC, args, specs, condition="ans_v1", splits=CERT_SPLITS, metrics=("A", "tokens"), pairs=pairs)
        doc[f"answer_only_{scale}"] = {"arms": CONTEXT[scale], "missing_pairs": missing, "arm_means": d["arm_means"],
                                       "comparisons": [slim_comp(c, ["all"]) for c in d["comparisons"]]}
        side[f"_ans_{scale}"] = d
        side[f"_ctx_{scale}"] = ctx_doc

    # ---- 3. scale contrast on the same rows
    doc["scale_contrast"] = scale_contrast(CC, args, runs, ready, [
        (("P7", "G7"), ("P3", "G3")), (("P7", "S7"), ("P3", "S3")), (("G7", "S7"), ("G3", "S3"))], CERT_SPLITS)

    doc["numeric_only"] = numeric_only(CC, args, runs, ready, [("P7", "G7"), ("P3", "G3"), ("P7", "S7"), ("G7", "S7")], ("e1_test", "e2_test"))
    doc["selected_set_decomposition"] = selected_set_decomposition(doc)
    doc["seed_spread_3b"] = seed_spread_3b(CC, args, runs, ready, doc.get("core_7b") or {}, doc["numeric_only"])

    # ---- 4. steps 100/200/300 on E1-dev (S = step 0)
    for scale in ("7b", "3b"):
        n = scale[0]
        arms = [f"S{n}"] + [f"{a}{n}_{k}" for a in ("G", "P") for k in (100, 200, 300)]
        specs = [s for a in arms for s in arm_specs(runs, ready, a, ("e1_dev",), "cert_v1")]
        have = present(specs)
        want = [(f"P{n}_{k}", f"G{n}_{k}") for k in (100, 200, 300)] + [(f"{a}{n}_{k}", f"S{n}") for a in ("G", "P") for k in (100, 200, 300)]
        pairs, missing = keep_pairs(want, have)
        d = cc_safe(CC, args, specs, condition="cert_v1", splits=("e1_dev",), metrics=METRICS, pairs=pairs)
        doc[f"steps_{scale}"] = {"arms": arms, "missing_pairs": missing, "arm_means": d["arm_means"],
                                 "comparisons": [slim_comp(c, ["all"]) for c in d["comparisons"]]}

    # ---- 5. error classes (RQ3)
    ec_models = ["base7", "legacy7", "S7", "G7", "P7", "S3", "G3", "P3"]
    doc["error_classes"] = error_classes(runs, ready, ec_models, CERT_SPLITS)
    doc["error_class_paired"] = error_class_paired(CC, args, runs, ready, [("P7", "G7"), ("P3", "G3"), ("P7", "S7"), ("G7", "S7")],
                                                   ("e1_test", "e2_test"))

    # ---- 6. TRACE validation
    tv_data: Dict[str, Any] = {}
    tv_sum: Dict[str, Any] = {}
    for scale in ("7b", "3b"):
        for m in TRACE_VAL[scale]:
            if ready[m]["trace_val_ok"]:
                tv_data[m] = load_trace_val(runs, m)
                tv_sum[m] = trace_val_summary(tv_data[m])
    tv_pairs: Dict[str, Any] = {}
    tv_missing: List[str] = []
    for scale in ("7b", "3b"):
        for x, y in TV_PAIRS[scale]:
            if x in tv_data and y in tv_data:
                tv_pairs[f"{x} - {y}"] = trace_val_pair(CC, args, mcnemar, tv_data[x], tv_data[y])
            else:
                tv_missing.append(f"{x} - {y}")
    doc["trace_val"] = {"models": {s: TRACE_VAL[s] for s in TRACE_VAL}, "summary": tv_sum, "pairs": tv_pairs, "pending_pairs": tv_missing,
                        "protocol": {m: trace_val_protocol(runs, m) for m in tv_data},
                        "pending_models": [m for s in TRACE_VAL for m in TRACE_VAL[s] if m not in tv_data],
                        "note": "official = lower bound (judge-pending rows count as wrong); upper bound counts them as right"}

    # ---- 7. external context (descriptive; seed 42)
    doc["external"] = {"seed": 42, "models": external(runs, CONTEXT["7b"] + CONTEXT["3b"]),
                       "note": "descriptive context only; no paired statistics here; values as on disk when this ran"}

    doc["training_match"] = training_match(runs)

    # ---- 8. comparability audit
    audit_models = ["P7", "G7", "S7", "base7", "legacy7", "base7_ck", "legacy7_ck", "P7_100", "P7_200", "G7_100", "G7_200",
                    "P3", "G3", "S3", "base3", "legacy3", "base3_ck", "legacy3_ck"]
    doc["comparability"] = comparability(runs, ready, audit_models)
    doc["comparability"]["prompt_identity"] = prompt_identity(runs, ready, PROMPT_MODELS)
    doc["comparability"]["prompt_identity_3b"] = prompt_identity(runs, ready, ["P3", "G3", "S3", "base3", "legacy3", "base3_ck", "legacy3_ck"], ref="P3")
    rp_specs: List[str] = []
    for x, y in REPRO_PAIRS:
        rp_specs += arm_specs(runs, ready, x, CERT_SPLITS, "cert_v1") + arm_specs(runs, ready, y, CERT_SPLITS, "cert_v1")
    have = present(rp_specs)
    pairs, missing = keep_pairs(REPRO_PAIRS, have)
    repro: Dict[str, Any] = {"pairs_missing": missing}
    if pairs:
        d = cc_build(CC, args, rp_specs, condition="cert_v1", splits=CERT_SPLITS, metrics=METRICS, pairs=pairs)
        repro["comparisons"] = [slim_comp(c, ["all"]) for c in d["comparisons"]]
        repro["arm_means"] = d["arm_means"]
    repro["identical_responses"] = repro_identical_responses(runs, ready, pairs)
    doc["comparability"]["pipeline_reproducibility"] = repro

    # ---- 9. cross-checks against the pipeline's own summaries
    cond_tag = {"cert_v1": "cert_px", "ans_v1": "ans"}
    docs_for_check = [d for k, d in side.items() if k.startswith("cmp_") or k.startswith("_ans_") or k.startswith("_ctx_")]
    doc["crosschecks"] = {"cert_means_vs_pipeline_summaries": crosscheck_means(runs, ready, docs_for_check, cond_tag),
                          "trace_val_vs_pipeline_summaries": crosscheck_trace_val(runs, tv_sum),
                          "core_3b_vs_stage1_interim": crosscheck_stage1(runs, doc.get("core_3b") or {}),
                          "seed_spread_s1_vs_core_3b": crosscheck_spread(doc)}

    # ---- 10. pending
    pend = []
    for k in keys:
        r = ready[k]
        if not r["validation_ok"]:
            what = "certificate splits" + (" and TRACE validation" if r["has_trace_val_plan"] else "")
            pend.append({"model": k, "what": f"{what} (validation.json not ok yet)", "path": r["path"],
                         "progress": f"{r['cert_score_files']} cert_scores files of {r.get('plan_runs', '?')} planned runs"})
        elif r["has_trace_val_plan"] and not r["trace_val_ok"] and k in TRACE_VAL["7b"] + TRACE_VAL["3b"]:
            pend.append({"model": k, "what": "TRACE validation (validation.trace_val.json not ok yet)", "path": r["path"]})
    for k, e in doc["external"]["models"].items():
        if not e.get("complete"):
            pend.append({"model": k, "what": f"external suite seed 42: {e['status']}", "path": e["path"]})
    doc["pending"] = pend
    doc["runtime_s"] = round(time.time() - t0, 1)
    return doc, {k: v for k, v in side.items() if k.startswith("cmp_")}


# ============================================================================== markdown (from the JSON only)


def pc(v: Optional[float], d: int = 1) -> str:
    return "-" if v is None else f"{100 * v:.{d}f}"


def pp(v: Optional[float], d: int = 2) -> str:
    return "-" if v is None else f"{100 * v:+.{d}f}"


def tk(v: Optional[float], signed: bool = False) -> str:
    return "-" if v is None else (f"{v:+.1f}" if signed else f"{v:.0f}")


def pv(v: Optional[float]) -> str:
    if v is None:
        return "-"
    if v == 0:
        return "<1e-300"
    return f"{v:.2g}" if v < 0.001 else f"{v:.3f}"


def fmt_delta(m: str, r: Optional[Mapping[str, Any]], with_p: bool = True) -> str:
    if not r or r.get("delta") is None:
        return "-"
    lo, hi = r["ci"]
    if m == "tokens":
        s = f"{r['delta']:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    else:
        s = f"{pp(r['delta'])} [{pp(lo)}, {pp(hi)}]"
    if with_p and r.get("p") is not None:
        s += f" p={pv(r['p'])}"
    return s


def mean_of(doc: Mapping[str, Any], block: str, arm: str, split: str, m: str) -> Optional[float]:
    try:
        return doc[block]["arm_means"][arm]["-"][split]["means"][m]
    except (KeyError, TypeError):
        return None


def comp(doc: Mapping[str, Any], block: str, name: str) -> Optional[Mapping[str, Any]]:
    for c in (doc.get(block) or {}).get("comparisons", []):
        if c["name"] == name:
            return c
    return None


def scope_res(c: Optional[Mapping[str, Any]], split: str, scope: str = "all") -> Mapping[str, Any]:
    if not c:
        return {}
    return ((c.get("splits") or {}).get(split) or {}).get("scopes", {}).get(scope) or {}


def family_row(doc: Mapping[str, Any], scale: str, split: str) -> Optional[Mapping[str, Any]]:
    for fam in (doc.get(f"core_{scale}") or {}).get("families", []):
        if fam["family"] == "primary":
            for t in fam["tests"]:
                if t["split"] == split and t["metric"] == "A":
                    return t
    return None


def _dci(r: Optional[Mapping[str, Any]], key: str = "delta", ci: str = "ci") -> str:
    if not r or r.get(key) is None:
        return "pending"
    return f"{pp(r[key])} pp [{pp(r[ci][0])}, {pp(r[ci][1])}]"


def _get(d: Any, *path: Any) -> Any:
    for k in path:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def _mcbc(r: Optional[Mapping[str, Any]]) -> str:
    m = (r or {}).get("mcnemar")
    return f"b={m['b_y_only']} G-only, c={m['c_x_only']} P-only, n={m['n']}" if m else "no McNemar"


def _seeds_txt(seeds: Optional[Sequence[int]]) -> str:
    s = sorted(seeds or [])
    if not s:
        return "?"
    return f"{s[0]}-{s[-1]}" if s == list(range(s[0], s[-1] + 1)) and len(s) > 1 else ", ".join(map(str, s))


def decoding_seeds_txt(doc: Mapping[str, Any]) -> str:
    return _seeds_txt(_get(doc, "core_7b", "arm_means", "P7", "-", "e1_test", "decoding_seeds"))


def temperature_txt(doc: Mapping[str, Any]) -> str:
    v = _get(doc, "comparability", "values_of", "protocol:settings.decoding_params.temperature") or []
    return "/".join(str(x) for x in v) if v else "?"


def _dec(t: Mapping[str, Any]) -> str:
    return t["decision"].replace("X > Y", "P > G").replace("X < Y", "P < G")


def _excl0(ci: Optional[Sequence[Optional[float]]]) -> Optional[bool]:
    if not ci or ci[0] is None or ci[1] is None:
        return None
    return ci[0] > 0 or ci[1] < 0


def _share_word(share: Optional[float]) -> str:
    if share is None:
        return "-"
    if share > 1.0:
        return "more than all"
    if share >= 0.95:
        return "almost all"
    if share >= 0.5:
        return "most"
    if share > 0:
        return "part"
    return "none"


def _spread(doc: Mapping[str, Any], split: str, scope: str, m: str = "A") -> Optional[Mapping[str, Any]]:
    return _get(doc, "seed_spread_3b", "summary", f"{split}/{scope}/{m}")


def _spread_txt(s: Mapping[str, Any]) -> str:
    vals = ", ".join(f"s{i + 1} {pp(v)}" for i, v in enumerate(s["deltas_3b"]))
    sd = f", SD {100 * s['sd_3b']:.2f} pp" if s.get("sd_3b") is not None else ""
    return f"{vals} (range width {100 * s['range_width_3b']:.2f} pp{sd})"


def _inside_txt(s: Mapping[str, Any]) -> str:
    v = s.get("delta_7b_s1")
    if v is None:
        return "7B pending"
    if s["delta_7b_inside_3b_range"]:
        return f"the 7B value {pp(v)} lies inside the 3B seed range"
    side = "above" if v > s["max_3b"] else "below"
    gap = v - s["max_3b"] if side == "above" else s["min_3b"] - v
    return f"the 7B value {pp(v)} lies {side} the 3B seed range by {100 * gap:.2f} pp"


def key_numbers(doc: Mapping[str, Any]) -> List[str]:
    """Generated bullets: every number is a lookup into the JSON document."""
    L = ["## Key numbers (generated from cert_7b.json)", ""]
    for scale in ("7b", "3b"):
        parts = []
        for split in ("e1_test", "e2_test"):
            t = family_row(doc, scale, split)
            r = (t or {}).get("result")
            parts.append(f"{split} {_dci(r)}, McNemar p {pv(r.get('p'))} ({_mcbc(r)}), Holm p {pv(t.get('holm_p'))}: {_dec(t)}"
                         if r else f"{split} pending")
        L.append(f"- Primary family, P - G on A at {scale.upper()} (1 training seed; CI = instance sampling only): " + "; ".join(parts) + ".")
    dec = doc.get("selected_set_decomposition") or {}
    for name in ("P7 - G7", "P3 - G3"):
        e = dec.get(f"{name}/e2_test")
        if not e:
            continue
        c = comp(doc, f"core_{name[1]}b", name)
        ss = scope_res(c, "e2_test", f"adapter:{SS_ADAPTER}").get("A")
        nr = _get(doc, "numeric_only", "pairs", name, "e2_test", "A")
        L.append(f"- E2-test {name} on A, {pp(e['delta_all'])} pp over {e['n_all']} rows, split by adapter (exact row-weighted sum): "
                 f"selected_set ({e['n_selected_set']} rows; P - G {_dci(ss)}, McNemar p {pv((ss or {}).get('p'))}) contributes "
                 f"{pp(e['contribution_selected_set'])} pp ({_share_word(e['share_selected_set'])} of it, "
                 f"{100 * e['share_selected_set']:.0f}%); the numeric adapters ({e['n_numeric']} rows; P - G {_dci(nr)}, "
                 f"McNemar p {pv((nr or {}).get('p'))}) contribute {pp(e['contribution_numeric'])} pp.")
    no = (doc.get("numeric_only") or {}).get("pairs", {})
    for name in ("P7 - G7", "P3 - G3"):
        if name in no:
            parts = [f"{split} {_dci(no[name][split]['A'])} (McNemar p {pv(no[name][split]['A'].get('p'))}, n {no[name][split]['A']['n_rows']})"
                     for split in ("e1_test", "e2_test") if split in no[name]]
            L.append(f"- {name} on A without the selected_set control adapter (post hoc sensitivity, not pre-registered): " + "; ".join(parts) + ".")
    sp = doc.get("seed_spread_3b") or {}
    if sp.get("summary"):
        for split, scope, label in (("e1_test", "all", "E1-test all rows"), ("e2_test", "all", "E2-test all rows"),
                                    ("e2_test", f"adapter:{SS_ADAPTER}", "E2-test selected_set"),
                                    ("e2_test", "numeric_only", "E2-test numeric adapters"),
                                    ("e1_test", "adapter:filtered_mean.table_filtered_column_mean", "E1-test table filtered mean")):
            s = _spread(doc, split, scope)
            if s:
                L.append(f"- 3B training-seed spread of P3 - G3 on A, {label} (step 300, each seed single-seed): {_spread_txt(s)}; {_inside_txt(s)}.")
    sc = next((c for c in (doc.get("scale_contrast") or {}).get("contrasts", []) if c["name"].startswith("(P7 - G7)")), None)
    if sc:
        parts = []
        for split, scope in (("e1_test", "all"), ("e2_test", "all"), ("e1_test", "adapter:filtered_mean.table_filtered_column_mean")):
            r = (((sc["splits"].get(split) or {}).get("scopes") or {}).get(scope) or {}).get("A")
            if r:
                parts.append(f"{split}{'' if scope == 'all' else ' table'} {_dci(r)} (sign-flip p {pv(r.get('perm_p'))})")
        L.append("- Scale contrast (P7 - G7) - (P3 - G3) on A, same rows, seed 1 at each scale (exploratory; CI = instance sampling only): "
                 + "; ".join(parts) + ".")
    for name in ("P7 - S7", "G7 - S7"):
        c = comp(doc, "core_7b", name)
        if c:
            L.append(f"- {name} on A: " + "; ".join(f"{split} {_dci(scope_res(c, split).get('A'))} (McNemar p {pv(scope_res(c, split).get('A', {}).get('p'))})"
                                                    for split in ("e1_test", "e2_test")) + ".")
    tv = doc.get("trace_val") or {}
    rows = []
    for m in (tv.get("models") or {}).get("7b", []):
        a = _get(tv, "summary", m, "ans_v1", "pooled")
        if a:
            rows.append(f"{m} {pc(a['official_lower'], 2)}% ({a['judge_pending']}/{a['n']} judge-pending; upper {pc(a['official_upper'], 2)}%)")
    if rows:
        L.append("- TRACE released validation at 7B, official accuracy (lower bound; ans_v1 seeds "
                 f"{_seeds_txt(_get(tv, 'protocol', 'P7', 'ans_v1', 'seeds'))}): " + "; ".join(rows) + ".")
    tvp = tv.get("pairs", {})
    parts = [f"{n} {_dci(tvp[n], 'delta_official_lower', 'ci_official_lower')} (McNemar p {pv((tvp[n].get('mcnemar') or {}).get('p'))})"
             for n in [f"{x} - {y}" for x, y in TV_PAIRS["7b"]] if tvp.get(n)]
    if parts:
        L.append("- TRACE validation paired at 7B: " + "; ".join(parts) + ".")
    steps = []
    for k in (100, 200, 300):
        r = scope_res(comp(doc, "steps_7b", f"P7_{k} - G7_{k}"), "e1_dev").get("A")
        if r:
            steps.append(f"step {k} {_dci(r)}, McNemar p {pv(r.get('p'))} ({_mcbc(r)}), sign-flip p {pv(r.get('perm_p'))}")
    if steps:
        L.append("- E1-dev P7 - G7 on A by training step: " + "; ".join(steps) + ".")
    rp = _get(doc, "comparability", "pipeline_reproducibility") or {}
    parts = []
    for name in ("base7_ck - base7", "legacy7_ck - legacy7"):
        c = next((x for x in rp.get("comparisons", []) if x["name"] == name), None)
        if c:
            parts.append(f"{name} " + ", ".join(f"{split} {_dci(scope_res(c, split).get('A'))}" for split in ("e1_test", "e2_test")))
    if parts:
        L.append("- Same 7B weights through the zero-shot and the checkpoint pipelines, A: " + "; ".join(parts) + ".")
    ex = _get(doc, "external", "models") or {}
    parts = [f"{m} {ex[m]['overall_24']:.2f}" for m in CONTEXT["7b"] if (ex.get(m) or {}).get("overall_24") is not None]
    if parts:
        L.append(f"- External trace_eval_v1 overall (24 benchmarks, seed {_get(doc, 'external', 'seed')}, descriptive): " + "; ".join(parts) + ".")
    L.append("")
    return L


def reading(doc: Mapping[str, Any]) -> List[str]:
    """Interpretation, generated from the JSON (the words that depend on numbers are chosen from the numbers)."""
    L = ["## Reading the results (generated; exploratory where marked)", ""]
    t1, t2 = family_row(doc, "7b", "e1_test"), family_row(doc, "7b", "e2_test")
    if t1 and t2 and t1.get("result") and t2.get("result"):
        r1, r2 = t1["result"], t2["result"]
        e1_ci = _excl0(r1["ci"])
        L.append(f"- Primary family at 7B (one training seed). E2-test: P7 - G7 {_dci(r2)}, Holm p {pv(t2['holm_p'])}, decision "
                 f"'{_dec(t2)}' at the instance level. E1-test: {_dci(r1)}, Holm p {pv(t1['holm_p'])}, decision '{_dec(t1)}'"
                 + (f" (the all-seed bootstrap CI excludes 0 while the pre-registered seed-{doc['config']['mcnemar_seed']} McNemar test does not reject)" if e1_ci and not t1["significant"] else "")
                 + ". DESIGN 8 needs a training-seed CI that excludes 0 and at least 2 of 3 seeds agreeing in sign, so neither "
                 "split is a P > G or P < G claim.")
    e = (doc.get("selected_set_decomposition") or {}).get("P7 - G7/e2_test")
    if e:
        L.append(f"- {_share_word(e['share_selected_set']).capitalize()} of the 7B E2 difference comes from the held-out chess "
                 f"program (selected_set): {pp(e['contribution_selected_set'])} of {pp(e['delta_all'])} pp "
                 f"({100 * e['share_selected_set']:.0f}%), with {pp(e['contribution_numeric'])} pp from the numeric adapters. "
                 "DESIGN 3.1 and 10 treat selected_set as a control adapter that re-expresses G and keep it out of go/no-go; the "
                 "numeric-only comparison is a post hoc sensitivity, not a pre-registered test.")
    spreads = [(lab, _spread(doc, split, scope)) for lab, split, scope in
               (("E2-test all rows", "e2_test", "all"), ("E2-test selected_set", "e2_test", f"adapter:{SS_ADAPTER}"),
                ("E2-test numeric adapters", "e2_test", "numeric_only"), ("E1-test all rows", "e1_test", "all"))]
    spreads = [(lab, s) for lab, s in spreads if s and s.get("abs_delta_7b_over_sd_3b") is not None]
    if spreads:
        parts = [f"{lab}: 3B {_spread_txt(s)}, {s['n_positive_3b']} of {s['n_seeds_3b']} seeds positive; {_inside_txt(s)}, "
                 f"{s['abs_delta_7b_over_sd_3b']:.1f} x the 3B seed SD from zero" for lab, s in spreads]
        big = [lab for lab, s in spreads if s["abs_delta_7b_over_sd_3b"] >= 2.0]
        small = [lab for lab, s in spreads if s["abs_delta_7b_over_sd_3b"] < 2.0]
        concl = []
        if small:
            concl.append(f"for {', '.join(small)} the 7B single-seed difference is less than two 3B-seed SDs from zero, the size "
                         "of seed-to-seed variation")
        if big:
            concl.append(f"for {', '.join(big)} it is at least two 3B-seed SDs from zero, so the 3B seed variation does not cover "
                         "it, but with one 7B seed it cannot be separated from 7B seed variation")
        L.append("- Training-seed variation, P - G on A (3B is the only scale with 3 seeds; each seed a single-seed comparison). "
                 + "; ".join(parts) + ". Reading: " + "; ".join(concl) + ". A second 7B training seed is needed for any claim.")
    sc = next((c for c in (doc.get("scale_contrast") or {}).get("contrasts", []) if c["name"].startswith("(P7 - G7)")), None)
    tb = _spread(doc, "e1_test", "adapter:filtered_mean.table_filtered_column_mean")
    if sc and tb:
        r = (((sc["splits"].get("e1_test") or {}).get("scopes") or {}).get("adapter:filtered_mean.table_filtered_column_mean") or {}).get("A")
        if r:
            L.append(f"- Scale contrasts are exploratory. Example: E1-test table filtered mean, P - G on A is {pp(r['effect_7b'])} pp at 7B "
                     f"and {pp(r['effect_3b'])} pp at 3B seed 1 (contrast {_dci(r)}, sign-flip p {pv(r['perm_p'])}), but across 3B "
                     f"seeds the same cell gives {_spread_txt(tb)}; {_inside_txt(tb)}. The contrast's CI and p resample instances "
                     "with one training seed per scale, so they do not show a scale-dependent mechanism.")
    steps = [(k, scope_res(comp(doc, "steps_7b", f"P7_{k} - G7_{k}"), "e1_dev").get("A")) for k in (100, 200, 300)]
    steps = [(k, r) for k, r in steps if r]
    if steps:
        k0, r0 = steps[0]
        L.append(f"- Training steps (E1-dev): P7 - G7 on A is " + "; ".join(f"step {k} {pp(r['delta'])} pp (McNemar p {pv(r['p'])})" for k, r in steps)
                 + f". At step {k0} the bootstrap CI {_dci(r0)} " + ("excludes" if _excl0(r0["ci"]) else "includes")
                 + f" 0 and the cluster sign-flip p is {pv(r0['perm_p'])}, while McNemar on seed {doc['config']['mcnemar_seed']} gives p {pv(r0['p'])}"
                 + ("; with no correction across steps and no consistent sign, this is not evidence of an early P advantage." if
                    len({r['delta'] > 0 for _, r in steps}) > 1 else "."))
    tvp = _get(doc, "trace_val", "pairs") or {}
    sb, lb, pg_ = tvp.get("S7 - base7_ck"), tvp.get("legacy7_ck - base7_ck"), tvp.get("P7 - G7")
    gs, ps = tvp.get("G7 - S7"), tvp.get("P7 - S7")
    tvd = lambda r: _dci(r, "delta_official_lower", "ci_official_lower")  # noqa: E731
    if sb and lb:
        L.append(f"- TRACE released validation (answer-only prompt): S7 - base7_ck is {tvd(sb)} (McNemar p "
                 f"{pv((sb.get('mcnemar') or {}).get('p'))}) and legacy7_ck - base7_ck is {tvd(lb)}. Neither RL arm moves far from S7"
                 + (f": G7 - S7 {tvd(gs)}, P7 - S7 {tvd(ps)}" if gs and ps else "")
                 + (f", P7 - G7 {tvd(pg_)} (McNemar p {pv((pg_.get('mcnemar') or {}).get('p'))})" if pg_ else "") + ".")
    rp = _get(doc, "comparability", "pipeline_reproducibility") or {}
    parts, all_incl = [], True
    for name in ("base7_ck - base7", "legacy7_ck - legacy7"):
        c = next((x for x in rp.get("comparisons", []) if x["name"] == name), None)
        if not c:
            continue
        for split in ("e1_test", "e2_test"):
            r = scope_res(c, split).get("A")
            if r:
                parts.append(f"{name} {split} {_dci(r)}")
                all_incl = all_incl and _excl0(r["ci"]) is False
    if parts:
        c = next((x for x in rp.get("comparisons", []) if x["name"] == "base7_ck - base7"), None)
        tok = scope_res(c, "e2_test").get("tokens") if c else None
        L.append("- Pipeline check at 7B, same weights through the zero-shot and the checkpoint pipelines, A: " + "; ".join(parts)
                 + (f". Completion tokens, base7_ck - base7 on E2-test: {tok['delta']:+.1f} [{tok['ci'][0]:+.1f}, {tok['ci'][1]:+.1f}]" if tok else "")
                 + (". Every accuracy CI includes 0, so the zero-shot base/legacy rows serve as accuracy context" if all_incl else
                    ". At least one accuracy CI excludes 0: read the zero-shot base/legacy rows with that pipeline offset in mind")
                 + ("; the token CI excludes 0, so completion-token counts are less comparable across the two pipelines."
                    if tok and _excl0(tok["ci"]) else "."))
    L.append("")
    return L


def render(doc: Mapping[str, Any]) -> str:
    L: List[str] = []
    tm7 = _get(doc, "training_match", "7b", "arms") or {}
    p7 = tm7.get("P") or {}
    hw = f"{p7.get('n_gpus')} x {p7.get('gpu')}" if p7.get("gpu") else "?"
    same_hw = bool(_get(doc, "training_match", "7b", "matched_fields", "gpu")) and bool(_get(doc, "training_match", "7b", "matched_fields", "n_gpus"))
    fam = next((f for f in (doc.get("core_7b") or {}).get("families", []) if f["family"] == "primary"), None)
    n_tests = len((fam or {}).get("tests", []))
    mseed = doc["config"]["mcnemar_seed"]
    rollout = (f"{p7['max_steps']} GRPO steps, {p7['rollout_batch_size']} prompts x {p7['rollout_n']} rollouts per step"
               if p7.get("max_steps") and p7.get("rollout_batch_size") and p7.get("rollout_n") else "GRPO")
    L += ["# TRACE certificates: 7B replication (DESIGN 7.3 stage 3)", "",
          f"Generated {doc['generated_at']} by `cert_7b.py` ({doc['version']}; compare_cert {doc['code']['compare_cert_version']}). "
          "Every number below is read from `cert_7b.json`; `cmp_7b.md` and `cmp_3b_s1.md` hold the full per-cell comparisons.", "",
          f"**Scope of the uncertainty.** P7 and G7 are one training run each (seed {p7.get('seed', '?')}, same 7B SFT init, "
          f"{'same ' + hw if same_hw else hw + ' (GPU differs between arms)'}), and "
          "the 3B side by side uses the seed-1 pair. CIs reflect instance sampling only: the 95% CIs and p-values describe "
          f"resampling of evaluation instances (clusters = `scene_group`, strata = adapter, {doc['config']['n_boot']:,} resamples) and "
          "say nothing about training-seed variance. DESIGN 8's claim rule (CI over training seeds excludes 0 and at least 2 of 3 seeds "
          "agree in sign) cannot be applied with one 7B seed, so no row below is a P > G claim. The 3B training-seed spread "
          "(section 1b) shows how large seed-to-seed differences are at 3B.", "",
          f"Conventions: rows are the mean over decoding seeds {decoding_seeds_txt(doc)} (cert_v1, temperature {temperature_txt(doc)}); "
          "differences are X - Y in percentage points (tokens: completion tokens); p is exact McNemar on decoding seed "
          f"{mseed} for A/F/joint and a cluster sign-flip test for G/M/tokens. Holm is applied only inside the pre-registered "
          f"primary family (P - G on A over E1-test and E2-test, {n_tests} tests per scale). The family decision uses the "
          f"pre-registered seed-{mseed} McNemar test, while the CI uses all decoding seeds, so the two can disagree near the boundary.", "",
          f"Arms: S = structured SFT warm start; G = S + grounding RL; P = S + grounding and semantic-certificate RL ({rollout}); "
          "base = Qwen2.5-VL-7B-Instruct; legacy = released TRACE-7B (answer-only RL). Suffix 7 = 7B, 3 = 3B seed 1; "
          "P3s2/G3s2/P3s3/G3s3 = 3B training seeds 2 and 3.", ""]

    L += key_numbers(doc)
    L += reading(doc)

    # pending
    if doc["pending"]:
        L += ["## Pending inputs (re-run `run.sh` to pick them up)", ""]
        for p in doc["pending"]:
            L.append(f"- `{p['model']}`: {p['what']}" + (f" ({p['progress']})" if p.get("progress") else ""))
        L.append("")

    # headline
    L += ["## 1. Primary family at 7B, with the 3B seed-1 pair beside it", "",
          f"| scale | split | P mean A | G mean A | P - G on A [95% CI] | McNemar p (seed {mseed}) | Holm p | decision |", "|---|---|---|---|---|---|---|---|"]
    for scale in ("7b", "3b"):
        for split in ("e1_test", "e2_test"):
            t = family_row(doc, scale, split)
            if not t or not t.get("result"):
                L.append(f"| {scale.upper()} | {split} | pending | | | | | |")
                continue
            r = t["result"]
            L.append(f"| {scale.upper()} | {split} | {pc(r['mean_x'])} | {pc(r['mean_y'])} | {pp(r['delta'])} [{pp(r['ci'][0])}, {pp(r['ci'][1])}] | "
                     f"{pv(r.get('p'))} | {pv(t.get('holm_p'))} | {t['decision'].replace('X > Y', 'P > G').replace('X < Y', 'P < G')} (1 training seed) |")
    L.append("")
    sc = doc.get("scale_contrast") or {}
    pg = next((c for c in sc.get("contrasts", []) if c["name"].startswith("(P7 - G7)")), None)
    if pg:
        L += ["Scale contrast on the same frozen rows, (P7 - G7) - (P3 - G3), cluster bootstrap and sign-flip p. Exploratory: one "
              "training seed per scale and a CI over instances only, so it does not separate scale from training-seed variation "
              "(section 1b).", "",
              "| split | metric | P - G at 7B | P - G at 3B | difference [95% CI] | p |", "|---|---|---|---|---|---|"]
        for split in ("e1_test", "e2_test"):
            for m in ("A", "M", "joint", "G", "F"):
                r = ((pg["splits"].get(split) or {}).get("scopes", {}).get("all") or {}).get(m)
                if r:
                    L.append(f"| {split} | {m} | {pp(r['effect_7b'])} | {pp(r['effect_3b'])} | {pp(r['delta'])} [{pp(r['ci'][0])}, {pp(r['ci'][1])}] | {pv(r['perm_p'])} |")
        L.append("")

    tm = doc.get("training_match") or {}
    if tm:
        per_step = sorted({str(v.get("rollouts_per_step")) for t in tm.values() for v in (t.get("arms") or {}).values()})
        L += ["Training side (from `rl/<scale>/<arm>/s1/run_receipt.json` and `resolved_easyr1.yaml`): rollout tokens are approximate "
              f"(mean training response length x {'/'.join(per_step)} rollouts per step x steps).", "",
              "| scale | arm | GPU | host | attempts | wall h | median s/step | mean rollout length | rollout tokens (approx) | matched with the other arm |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for scale in ("7b", "3b"):
            t = tm.get(scale) or {}
            mf = t.get("matched_fields") or {}
            note = ("all of seed, config, GPU class, init, train/val data, store, system prompt, steps" if t.get("all_matched")
                    else "differs: " + ", ".join(k for k, v in mf.items() if not v)) if mf else "-"
            for a, v in (t.get("arms") or {}).items():
                if v.get("status") == "missing":
                    L.append(f"| {scale.upper()} | {a} | missing | | | | | | | |")
                    continue
                L.append(f"| {scale.upper()} | {a} | {v['n_gpus']} x {v['gpu']} | {', '.join(h for h in v['hosts'] if h)} | {v['attempts']} | "
                         f"{v['wall_hours']:.1f} | {v['step_seconds_median']} | {v['train_resp_len_mean']:.1f} | {v['rollout_tokens_approx']:,} | {note} |")
        L.append("")

    no = doc.get("numeric_only") or {}
    if no.get("pairs"):
        L += [f"Post hoc sensitivity without the selected_set control adapter (`{no['excluded_adapter']}`), added after the "
              "per-adapter results were seen and not pre-registered. Motivation: DESIGN 3.1 and 10 call selected_set a control "
              "adapter that re-expresses G and keep it out of go/no-go; DESIGN 8's primary family itself is defined on all rows. "
              "Table, lever and path rows, descriptive, Δ [95% CI] p.", "",
              "| comparison | split | n | A | M | joint |", "|---|---|---|---|---|---|"]
        for name in [x for x in ("P7 - G7", "P3 - G3", "P7 - S7", "G7 - S7") if x in no["pairs"]]:
            for split, r in no["pairs"][name].items():
                L.append(f"| {name} | {split} | {r['A']['n_rows']} | {fmt_delta('A', r['A'])} | {fmt_delta('M', r['M'])} | {fmt_delta('joint', r['joint'])} |")
        L.append("")
    dec = doc.get("selected_set_decomposition") or {}
    if dec:
        L += ["Row-weighted split of P - G on A into the selected_set part and the numeric-adapter part (contribution = Δ x n / n_all; "
              "the two contributions add up to the all-rows Δ):", "",
              "| comparison | split | Δ all rows (n) | selected_set Δ (n) | selected_set contribution | numeric Δ (n) | numeric contribution | selected_set share |",
              "|---|---|---|---|---|---|---|---|"]
        for e in dec.values():
            share = "-" if e["share_selected_set"] is None else "{:.0f}%".format(100 * e["share_selected_set"])
            L.append(f"| {e['comparison']} | {e['split']} | {pp(e['delta_all'])} ({e['n_all']}) | {pp(e['delta_selected_set'])} ({e['n_selected_set']}) | "
                     f"{pp(e['contribution_selected_set'])} | {pp(e['delta_numeric'])} ({e['n_numeric']}) | {pp(e['contribution_numeric'])} | {share} |")
        L.append("")
    sp = doc.get("seed_spread_3b") or {}
    if sp.get("per_seed"):
        L += ["## 1b. Training-seed spread at 3B (reference scale for the one-seed 7B effects)", "",
              "P3 - G3 at step 300 for 3B training seeds 1-3, each computed as a single-seed comparison with the same code (Δ [95% CI, "
              "instance sampling] McNemar p). The 3B multi-seed estimate itself (two-level bootstrap, DESIGN 8 claim rule) is in "
              "`results/final/cert_3b`. Last columns: the 7B seed-1 P7 - G7 and whether it falls inside the range of the three 3B seeds.", ""]
        for split in ("e1_test", "e2_test"):
            scopes = sorted({k for e in sp["per_seed"].values() for k in (e.get(split) or {})}, key=lambda s: (s != "all", s != "numeric_only", s))
            L += [f"### {split}", "", "| scope | metric | " + " | ".join(sp["pairs"]) + " | 3B range (SD) | P7 - G7 | inside 3B range |",
                  "|---|---|" + "---|" * (len(sp["pairs"]) + 3)]
            for scope in scopes:
                for m in ("A", "M", "joint"):
                    cells = [fmt_delta(m, ((sp["per_seed"][n].get(split) or {}).get(scope) or {}).get(m)) for n in sp["pairs"]]
                    s = (sp.get("summary") or {}).get(f"{split}/{scope}/{m}")
                    rng = f"[{pp(s['min_3b'])}, {pp(s['max_3b'])}] ({100 * s['sd_3b']:.2f})" if s and s.get("sd_3b") is not None else "-"
                    v7 = pp(s["delta_7b_s1"]) if s and s.get("delta_7b_s1") is not None else "-"
                    ins = "-" if not s or s.get("delta_7b_inside_3b_range") is None else ("yes" if s["delta_7b_inside_3b_range"] else "no")
                    L.append(f"| {scope.split(':', 1)[-1].split('.')[0]} | {m} | " + " | ".join(cells) + f" | {rng} | {v7} | {ins} |")
            L.append("")
        if sp.get("missing"):
            L += [f"missing: {', '.join(sp['missing'])}", ""]

    # arm means
    L += ["## 2. Certificate-prompt results (cert_v1): 7B rows, then the 3B seed-1 rows", "",
          "Base Qwen2.5-VL-7B and legacy TRACE-7B rows come from the zero-shot evaluation (`eval/zs`); section 9 shows that its "
          "protocol, prompts and scorer are the same as the checkpoint evaluation's. `A (ans)` is accuracy on the same rows under the "
          "answer-only prompt (`ans_v1`). Means in %, tokens = mean completion tokens.", ""]
    for split in CERT_SPLITS:
        L += [f"### {split}", "", "| model | n | A | G | M | F | joint | tokens | within 2048 | A (ans) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for scale in ("7b", "3b"):
            for a in doc[f"context_{scale}"]["arms"]:
                v = ((doc[f"context_{scale}"]["arm_means"].get(a) or {}).get("-") or {}).get(split)
                if not v:
                    L.append(f"| {a} | pending | | | | | | | | |")
                    continue
                mm = v["means"]
                L.append(f"| {a} | {v['rows']} | {pc(mm['A'])} | {pc(mm['G'])} | {pc(mm['M'])} | {pc(mm['F'])} | {pc(mm['joint'])} | "
                         f"{tk(mm['tokens'])} | {pc(mm.get('within_2048'))} | {pc(mean_of(doc, f'answer_only_{scale}', a, split, 'A'))} |")
        L.append("")

    # RL vs SFT and P - G per metric
    L += ["## 3. Paired differences at 7B and 3B (all rows; descriptive except section 1)", "",
          "| scale | comparison | split | A | G | M | F | joint | tokens |", "|---|---|---|---|---|---|---|---|---|"]
    for scale, (P, G, S) in (("7b", ("P7", "G7", "S7")), ("3b", ("P3", "G3", "S3"))):
        for name in (f"{P} - {G}", f"{P} - {S}", f"{G} - {S}"):
            c = comp(doc, f"core_{scale}", name)
            for split in CERT_SPLITS:
                r = scope_res(c, split)
                if not r:
                    continue
                L.append(f"| {scale.upper()} | {name} | {split} | " + " | ".join(fmt_delta(m, r.get(m)) for m in METRICS) + " |")
    L.append("")
    for scale, (S, base, legacy, P, G) in (("7b", ("S7", "base7", "legacy7", "P7", "G7")), ("3b", ("S3", "base3", "legacy3", "P3", "G3"))):
        rows = []
        for name in (f"{S} - {base}", f"{legacy} - {base}", f"{P} - {legacy}", f"{G} - {legacy}"):
            c = comp(doc, f"context_{scale}", name)
            for split in ("e1_test", "e2_test"):
                r = scope_res(c, split)
                if r:
                    rows.append(f"| {scale.upper()} | {name} | {split} | " + " | ".join(fmt_delta(m, r.get(m)) for m in ("A", "M", "joint")) + " |")
        if rows:
            if scale == "7b":
                L += ["Context pairs (cert_v1, all rows):", "", "| scale | comparison | split | A | M | joint |", "|---|---|---|---|---|---|"]
            L += rows
    L.append("")

    # per adapter
    L += ["## 4. P - G per adapter, 7B beside 3B (cert_v1; Δ [95% CI] p)", ""]
    for split in ("e1_test", "e2_test"):
        L += [f"### {split}", "", "| adapter | metric | P7 - G7 | P3 - G3 | (P7-G7) - (P3-G3) |", "|---|---|---|---|---|"]
        c7, c3 = comp(doc, "core_7b", "P7 - G7"), comp(doc, "core_3b", "P3 - G3")
        scopes = sorted(k for k in ((c7 or {}).get("splits", {}).get(split, {}).get("scopes", {})) if k.startswith("adapter:"))
        for s in scopes:
            for m in ("A", "M", "joint"):
                r7 = scope_res(c7, split, s).get(m)
                r3 = scope_res(c3, split, s).get(m)
                rd = (((pg or {}).get("splits", {}).get(split) or {}).get("scopes", {}).get(s) or {}).get(m) if pg else None
                dd = f"{pp(rd['delta'])} [{pp(rd['ci'][0])}, {pp(rd['ci'][1])}] p={pv(rd['perm_p'])}" if rd else "-"
                L.append(f"| {s.split(':', 1)[1].split('.')[0]} | {m} | {fmt_delta(m, r7)} | {fmt_delta(m, r3)} | {dd} |")
        L.append("")

    # error classes
    L += ["## 5. Where the errors are (RQ3): first failing stage, cert_v1, rows pooled over seeds (%)", "",
          "Classes follow the scorer's first failing stage: envelope, grounding, operand, computation, answer_inconsistent (M = 1, A = 0), "
          "correct (M = 1, A = 1). `A1 M<1` = right answer with an imperfect certificate.", ""]
    for split in ("e1_test", "e2_test"):
        L += [f"### {split}", "", "| model | rows | " + " | ".join(ERROR_CLASSES) + " | A1 M<1 | A0 M1 |", "|---|---|" + "---|" * (len(ERROR_CLASSES) + 2)]
        for m in [x for x in ("base7", "legacy7", "S7", "G7", "P7", "S3", "G3", "P3") if x in doc["error_classes"]]:
            v = doc["error_classes"][m].get(split)
            if v:
                L.append(f"| {m} | {v['rows_pooled_over_seeds']} | " + " | ".join(pc(v["rates"].get(c, 0.0)) for c in ERROR_CLASSES) +
                         f" | {pc(v['A1_M_lt1_rate'])} | {pc(v['A0_M1_rate'])} |")
        L.append("")
    L += ["Paired differences of class rates (seed-averaged rows, cluster bootstrap, sign-flip p; descriptive):", "",
          "| comparison | split | " + " | ".join(ERROR_CLASSES) + " |", "|---|---|" + "---|" * len(ERROR_CLASSES)]
    ecp = doc["error_class_paired"]["pairs"]
    for name in [x for x in ("P7 - G7", "P3 - G3", "P7 - S7", "G7 - S7") if x in ecp]:
        for split, v in ecp[name].items():
            L.append(f"| {name} | {split} | " + " | ".join(
                f"{pp(v['cols'][c]['delta'])} [{pp(v['cols'][c]['ci'][0])}, {pp(v['cols'][c]['ci'][1])}] p={pv(v['cols'][c]['perm_p'])}" for c in ERROR_CLASSES) + " |")
    L.append("")

    # steps
    s7 = _get(doc, "steps_7b", "arm_means", "S7", "-", "e1_dev") or {}
    L += [f"## 6. Training steps on E1-dev (cert_v1, {s7.get('rows', '?')} rows, decoding seeds {_seeds_txt(s7.get('decoding_seeds'))}; "
          "step 0 = the SFT init)", "",
          "P - G columns: Δ [95% CI] and McNemar p on decoding seed "
          f"{doc['config']['mcnemar_seed']} (A) or sign-flip p (M); one comparison per step, no correction across steps.", ""]
    for scale in ("7b", "3b"):
        n = scale[0]
        blk = doc.get(f"steps_{scale}") or {}
        L += [f"### {scale.upper()}", "", "| step | arm | A | G | M | F | joint | tokens | P - G on A | P - G on M | arm - S on A |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        s = mean_of(doc, f"steps_{scale}", f"S{n}", "e1_dev", "A")
        if s is not None:
            mm = {m: mean_of(doc, f"steps_{scale}", f"S{n}", "e1_dev", m) for m in METRICS}
            L.append(f"| 0 | S{n} | {pc(mm['A'])} | {pc(mm['G'])} | {pc(mm['M'])} | {pc(mm['F'])} | {pc(mm['joint'])} | {tk(mm['tokens'])} | | | |")
        for k in (100, 200, 300):
            for a in ("G", "P"):
                arm = f"{a}{n}_{k}"
                mm = {m: mean_of(doc, f"steps_{scale}", arm, "e1_dev", m) for m in METRICS}
                if mm["A"] is None:
                    L.append(f"| {k} | {arm} | pending | | | | | | | | |")
                    continue
                pg_c = comp(doc, f"steps_{scale}", f"P{n}_{k} - G{n}_{k}") if a == "P" else None
                rs = scope_res(pg_c, "e1_dev") if pg_c else {}
                vs = scope_res(comp(doc, f"steps_{scale}", f"{arm} - S{n}"), "e1_dev")
                L.append(f"| {k} | {arm} | {pc(mm['A'])} | {pc(mm['G'])} | {pc(mm['M'])} | {pc(mm['F'])} | {pc(mm['joint'])} | {tk(mm['tokens'])} | "
                         f"{fmt_delta('A', rs.get('A')) if rs else ''} | {fmt_delta('M', rs.get('M')) if rs else ''} | {fmt_delta('A', vs.get('A'))} |")
        if blk.get("missing_pairs"):
            L.append(f"\nmissing: {', '.join(blk['missing_pairs'])}")
        L.append("")

    # trace val
    tv = doc["trace_val"]
    first = next((r for r in tv["pairs"].values() if r), None) or {}
    proto = {}
    for m, pr in (tv.get("protocol") or {}).items():
        for cond, v in pr.items():
            e = proto.setdefault(cond, {"seeds": set(), "cap": set(), "temp": set()})
            e["seeds"].update(v["seeds"])
            e["cap"].update(map(str, v["max_total_tokens"]))
            e["temp"].update(map(str, v["temperature"]))
    pa, pn = proto.get("ans_v1") or {}, proto.get("no_tool") or {}
    L += [f"## 7. TRACE released validation (answer-only prompt, {first.get('rows', '?'):,} rows, {first.get('n_clusters_tasks', '?'):,} tasks)"
          if first else "## 7. TRACE released validation (answer-only prompt)", "",
          "`official` counts judge-pending rows as wrong (lower bound); `upper` counts them as right. ans_v1 pooled over decoding seeds "
          f"{_seeds_txt(sorted(pa.get('seeds', [])))} ({'/'.join(sorted(pa.get('cap', []))) or '?'}-token cap, temperature "
          f"{'/'.join(sorted(pa.get('temp', []))) or '?'}); `no_tool` is TRACE's own protocol run "
          f"({'/'.join(sorted(pn.get('cap', []))) or '?'}-token cap, seed {_seeds_txt(sorted(pn.get('seeds', [])))}). "
          "Base and legacy columns come from `eval/ckpt/base-*` (ckpt_eval pipeline)"
          + (" and appear once those runs validate." if any(m.endswith("_ck") for m in tv.get("pending_models", [])) else "."), "",
          "| scale | model | official (lower) | upper | strict | judge pending / rows | no_tool official | tokens |", "|---|---|---|---|---|---|---|---|"]
    for scale in ("7b", "3b"):
        for m in tv["models"][scale]:
            s = tv["summary"].get(m)
            if not s or "ans_v1" not in s:
                L.append(f"| {scale.upper()} | {m} | pending | | | | | |")
                continue
            a = s["ans_v1"]["pooled"]
            nt = (s.get("no_tool") or {}).get("pooled")
            L.append(f"| {scale.upper()} | {m} | {pc(a['official_lower'], 2)} | {pc(a['official_upper'], 2)} | {pc(a['strict'], 2)} | {a['judge_pending']} / {a['n']} | "
                     f"{pc(nt['official_lower'], 2) if nt else '-'} | {a['tokens_mean']:.1f} |")
    L += ["", "Paired (cluster = TRACE task, strata = domain; Δ range = the point estimate if every judge-pending row of X or of Y turned out right):", "",
          f"| comparison | X official | Y official | Δ official [95% CI] | McNemar p (seed {(first.get('mcnemar') or {}).get('seed', '?')}) | Δ range if pending resolved | Δ strict [95% CI] |", "|---|---|---|---|---|---|---|"]
    order = [f"{x} - {y}" for sc in ("7b", "3b") for x, y in TV_PAIRS[sc]]
    for name in [n for n in order if n in tv["pairs"]]:
        r = tv["pairs"][name]
        if not r:
            continue
        b = r["delta_bounds_if_pending_resolved"]
        L.append(f"| {name} | {pc(r['mean_x_official_lower'], 2)} | {pc(r['mean_y_official_lower'], 2)} | {pp(r['delta_official_lower'])} "
                 f"[{pp(r['ci_official_lower'][0])}, {pp(r['ci_official_lower'][1])}] | {pv((r.get('mcnemar') or {}).get('p'))} | "
                 f"[{pp(b[0])}, {pp(b[1])}] | {pp(r['delta_strict'])} [{pp(r['ci_strict'][0])}, {pp(r['ci_strict'][1])}] |")
    for name in [n for n in order if n in tv["pending_pairs"]]:
        L.append(f"| {name} | pending | | | | | |")
    L.append("")

    # external
    ex = doc["external"]["models"]
    hashes = sorted({e.get("code_hash_short") for e in ex.values() if e.get("code_hash_short")})
    L += [f"## 8. External suite trace_eval_v1 (full, seed {doc['external']['seed']}; descriptive context)", "",
          "Macro scores as written by the external-eval pipeline (`summary.seed42.json`); paired external statistics belong to the "
          "external-transfer analysis. Scorer provenance: `code hash` is `contract.trace_eval_code_provenance.sha256` of each run's "
          "scoring manifest" + (f" (values here: {', '.join(hashes)})" if hashes else "") + ". The post-release fix-3 (a judge row "
          "with no usable output after all attempts is scored incorrect, as upstream VLMEvalKit does, instead of failing the "
          "benchmark) changed this hash; runs scored before it carry no fix-3 counters, and under that code such a row would have "
          "failed its benchmark, so a complete pre-fix-3 run had none. `fix-3 rows` = `mathverse_all_retries_failed_scored_incorrect` "
          "+ `extraction_failed_scored_incorrect`; `max shift` = the overall (24) change if every such row had been scored correct "
          "(benchmark score = row accuracy). `fix-2 rows` = `evaluator_exception_scored_incorrect`.", "",
          "| scale | model | status | code hash | overall (24) | natural (9) | chart/diagram/synthetic (15) | MMStar | det. extraction fail | cert-shaped JSON | fix-2 rows | fix-3 rows (max shift, pp) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for scale in ("7b", "3b"):
        for m in CONTEXT[scale]:
            e = ex.get(m)
            if not e:
                continue
            f2 = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
            f3 = ("not recorded" if not e.get("fix3_recorded") else
                  f"{e['fix3_rows']} ({e['fix3_overall_24_shift_bound_pp']:.4f})" if e.get("fix3_overall_24_shift_bound_pp") is not None else str(e["fix3_rows"]))
            f2r = (e.get("counters") or {}).get("evaluator_exception_scored_incorrect")
            L.append(f"| {scale.upper()} | {m} | {e['status']} | {e.get('code_hash_short') or '-'} | {f2(e.get('overall_24'))} | {f2(e.get('natural_9'))} | "
                     f"{f2(e.get('chart_diagram_synthetic_15'))} | {f2(e.get('mmstar'))} | {pc(e.get('det_fail_rate'))} | {pc(e.get('cert_json_rate'))} | "
                     f"{'-' if f2r is None else f2r} | {f3} |")
    L.append("")

    # comparability
    cp = doc["comparability"]
    L += ["## 9. Are the zero-shot base/legacy rows comparable with the checkpoint evaluations?", "",
          f"Audit over {sum(cp['run_configs_read'].values())} run configs of {len(cp['models'])} model directories "
          f"({cp['field_checks']} field checks, grouped by split and condition):", ""]
    L += ["| category | fields identical across all models | fields differing |", "|---|---|---|"]
    for cat, v in sorted(cp["by_category"].items()):
        L.append(f"| {cat} | {v['identical']} | {v['differing']} |")
    L.append("")
    for name, v in cp.get("verdict", {}).items():
        L.append(f"- {name.upper()} model dirs ({', '.join(v['models'])}): " +
                 ("every protocol, scoring and software field identical." if v["all_fields_identical"]
                  else "differ in " + ", ".join(f"`{f}`" for f in v["differing_fields"]) + "."))
    L.append("")
    if any(e["field"] == "scoring:provenance.cert_code_sha256" for e in cp["differing"]):
        L += ["A different `provenance.cert_code_sha256` under the same `scorer_version`, store and verifier versions means those rows were "
              "scored before a later edit of the certificate package; the same-weights rows at the end of this section measure whether "
              "the pipeline difference moves any metric.", ""]
    if cp["differing"]:
        L += ["Differing fields (value -> models; every split and condition listed):", ""]
        byfield: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        where: Dict[str, set] = defaultdict(set)
        for e in cp["differing"]:
            where[e["field"]].add(f"{e['split']}/{e['cond']}")
            for m, vals in e["values"].items():
                byfield[e["field"]][", ".join(v.strip('"') for v in vals)].add(m)
        for f, inv in byfield.items():
            parts = "; ".join(f"{val[:24] + ('...' if len(val) > 24 else '')} -> {', '.join(sorted(ms))}" for val, ms in sorted(inv.items(), key=lambda kv: -len(kv[1])))
            L.append(f"- `{f}` in {len(where[f])} split/condition groups: {parts}")
        L.append("")
    L.append(f"Serving: vLLM versions {', '.join(cp['vllm_versions_all'])}; non-default serving arguments (model, port and parallelism removed) "
             f"{'identical' if len(cp['vllm_args_all']) == 1 else 'differ'} across models; eval nodes: " +
             "; ".join(f"{m} {','.join(s['eval_nodes'])}" for m, s in cp["serving"].items()) + ".")
    tc = {m: s["template_check"] for m, s in cp["serving"].items() if s["template_check"]}
    if tc:
        ok = all(all(v.values()) for v in tc.values())
        L.append(f"Checkpoint chat template and processor geometry equal the base model's (template_check.json): "
                 f"{'yes for all ' + str(len(tc)) + ' checkpoint dirs' if ok else 'NO for ' + ', '.join(m for m, v in tc.items() if not all(v.values()))}.")
    L.append("")
    for blk, title in (("prompt_identity", "7B"), ("prompt_identity_3b", "3B")):
        pi = cp.get(blk) or {}
        if not pi.get("by_model"):
            continue
        L += [f"Prompt identity at {title}, decoding seed 42, against `{pi['reference']}` (system + user text + image size; prompt token counts):", "",
              "| model | " + " | ".join(f"{s}/{c}" for s in CERT_SPLITS for c in ("cert_v1", "ans_v1")) + " |", "|---|" + "---|" * 6]
        for m, res in pi["by_model"].items():
            cells = []
            for s in CERT_SPLITS:
                for c in ("cert_v1", "ans_v1"):
                    v = res.get(f"{s}/{c}")
                    cells.append("-" if not v else f"{v['prompt_text_identical']}/{v['rows']} text, {v['prompt_tokens_identical']}/{v['rows']} tokens")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    rp = cp["pipeline_reproducibility"]
    L += ["Same weights through both pipelines (ckpt_eval - zero-shot eval; cert_v1, all rows, descriptive):", "",
          "| comparison | split | A | G | M | F | joint | tokens |", "|---|---|---|---|---|---|---|---|"]
    for c in rp.get("comparisons", []):
        for split in CERT_SPLITS:
            r = scope_res(c, split)
            if r:
                L.append(f"| {c['name']} | {split} | " + " | ".join(fmt_delta(m, r.get(m)) for m in METRICS) + " |")
    for name in rp.get("pairs_missing", []):
        L.append(f"| {name} | pending | | | | | | |")
    L.append("")
    for name, res in rp.get("identical_responses", {}).items():
        tot = sum(v["rows"] for v in res.values())
        same = sum(v["identical_response"] for v in res.values())
        samep = sum(v["identical_prompt"] for v in res.values())
        x, y = name.split(" - ")
        nx, ny = set(_get(cp, "serving", x, "eval_nodes") or []), set(_get(cp, "serving", y, "eval_nodes") or [])
        where = "on different eval nodes" if nx and ny and not (nx & ny) else "on overlapping eval nodes" if nx and ny else "on unrecorded nodes"
        if tot:
            L.append(f"- {name}: identical prompts on {samep}/{tot} rows; byte-identical responses on {same}/{tot} rows "
                     f"(sampling at temperature {temperature_txt(doc)} {where}, so responses are not expected to match).")
    L.append("")
    cc = doc["crosschecks"]
    sp_cc = cc.get("seed_spread_s1_vs_core_3b") or {}
    L += ["## 10. Cross-checks", "",
          f"- compare_cert arm means vs the pipeline's `summary.<split>.json`: {cc['cert_means_vs_pipeline_summaries']['n_checks']} checks, "
          f"max |diff| {cc['cert_means_vs_pipeline_summaries']['max_abs_diff']:.2g} ({'ok' if cc['cert_means_vs_pipeline_summaries']['ok'] else 'MISMATCH'}).",
          f"- TRACE-val pooled official accuracy and judge-pending counts vs `summary.trace_val.json`: {len(cc['trace_val_vs_pipeline_summaries']['checks'])} checks, "
          f"max |diff| {cc['trace_val_vs_pipeline_summaries']['max_abs_diff']:.2g} ({'ok' if cc['trace_val_vs_pipeline_summaries']['ok'] else 'MISMATCH'}).",
          (f"- 3B seed-1 primary family (Δ, CI, McNemar p, Holm p) vs `results/stage1_interim/cmp_PGS.json`: "
           f"{len(cc['core_3b_vs_stage1_interim']['rows'])} tests, max |diff| {cc['core_3b_vs_stage1_interim']['max_abs_diff']:.2g} "
           f"({'ok' if cc['core_3b_vs_stage1_interim']['ok'] else 'MISMATCH'})."
           if (cc.get("core_3b_vs_stage1_interim") or {}).get("available") else "- stage-1 interim comparison file not found."),
          (f"- Seed-1 rows of the 3B seed spread (separate compare_cert call) vs the core 3B P3 - G3 (Δ and CI): {sp_cc['n_checks']} checks, "
           f"max |diff| {sp_cc['max_abs_diff']:.2g} ({'ok' if sp_cc['ok'] else 'MISMATCH'})." if sp_cc.get("available")
           else "- 3B seed-spread cross-check not available."),
          (f"- E2/E1 selected_set decomposition: contributions add up to the all-rows Δ (max |diff| "
           f"{max(e['sum_check_abs_diff'] for e in doc['selected_set_decomposition'].values()):.2g})."
           if doc.get("selected_set_decomposition") else ""), "",
          f"Runtime {doc['runtime_s']} s.", ""]
    return "\n".join(L)


def render_readme(doc: Mapping[str, Any]) -> str:
    p7 = _get(doc, "training_match", "7b", "arms", "P") or {}
    hw = f"{p7.get('n_gpus')} x {p7.get('gpu')}" if p7.get("gpu") else "?"
    sizes = ", ".join(f"{s} ({n:,})" if isinstance(n, int) else f"{s} (?)"
                      for s, n in ((s, _get(doc, "core_7b", "arm_means", "P7", "-", s, "rows")) for s in CERT_SPLITS))
    rp_missing = _get(doc, "comparability", "pipeline_reproducibility", "pairs_missing") or []
    L = ["# cert_7b: 7B replication of the certificate RL arms (DESIGN 7.3 stage 3)", "",
         f"Generated {doc['generated_at']} by `$TRACE_ROOT/experiments/certificates/analysis/cert_7b.py` ({doc['version']}). "
         "Regenerate everything with `bash run.sh` (CPU only, a few minutes; `CERT_7B_OUT=<dir> bash run.sh` writes elsewhere). "
         "This README is written by the script too.", "",
         "## Question", "",
         "Does the 3B seed-1 picture (P = grounding + semantic certificate RL vs G = grounding RL, both from the same SFT init S) "
         f"replicate at 7B (P7 vs G7 vs S7, one training seed each, {hw}), and how do base Qwen2.5-VL-7B and legacy "
         "TRACE-7B compare on the same frozen splits and on TRACE's released validation set? With one 7B training seed this is "
         "answered at the instance level only; 3B training seeds 1-3 give the reference scale of seed-to-seed variation.", "",
         "## Files", "",
         "- `cert_7b.md`: the report (every number is read from `cert_7b.json`).",
         "- `cert_7b.json`: every number, plus input readiness, the comparability audit, cross-checks and the pending list.",
         "- `cmp_7b.{json,md}`, `cmp_3b_s1.{json,md}`: full `compare_cert.py` documents (P, G, S; all rows, per adapter, per cell).",
         "- `run.sh`: the exact commands; `run.log`: output of the last run.", "",
         "## Inputs (read-only)", "", "| key | directory | used | note |", "|---|---|---|---|"]
    for k in [x for x in list(MODELS) + list(STEP_ALIAS) if x in doc["models"]]:
        m = doc["models"][k]
        used = "yes" if m["validation_ok"] else "pending"
        note = m["label"] + (", TRACE-val ok" if m.get("trace_val_ok") else "")
        L.append(f"| {k} | `{m['path']}` | {used} | {note} |")
    L += ["", "External context: `ext_eval/runs/<served>/full/summary.seed42.json` (descriptive only).", "",
          "## Method", "",
          f"- Certificate rows (`cert_v1`; `ans_v1` for the answer-only column): {sizes} rows, decoding seeds "
          f"{decoding_seeds_txt(doc)} at temperature {temperature_txt(doc)}. Statistics come from `compare_cert.py` (imported unchanged): "
          "rows paired by instance, seed-averaged, paired cluster bootstrap over `scene_group` stratified by adapter, exact McNemar on "
          f"decoding seed {doc['config']['mcnemar_seed']} for A/F/joint, cluster sign-flip for G/M/tokens, Holm only inside the primary "
          "family (P - G on A, E1-test and E2-test).",
          "- One training seed per arm, so every CI reflects instance sampling only. The scale contrast (P7 - G7) - (P3 - G3) pairs the "
          "same frozen rows across scales (the 3B and 7B runs score identical instance ids); it is exploratory because it compares one "
          "training seed per scale. The 3B seed spread (P3 - G3 for 3B training seeds 1-3, each as a single-seed comparison) shows "
          "how large seed-to-seed differences are.",
          "- The numeric-only comparison (selected_set rows dropped) is a post hoc sensitivity, motivated by DESIGN 3.1 and 10 "
          "(selected_set is a control adapter that re-expresses G and is excluded from go/no-go); it is not pre-registered.",
          "- Base and legacy 7B certificate rows come from the zero-shot evaluation (`eval/zs`). The audit in section 9 of `cert_7b.md` "
          "compares every run config, score receipt, serving argument and seed-42 prompt with the checkpoint evaluations, and the "
          "same-weights rows compare the two pipelines directly"
          + (f" (pending: {', '.join(rp_missing)})." if rp_missing else " at both scales."),
          "- TRACE released validation: `trace_val/ans_v1` and the `no_tool` protocol run of each ckpt_eval dir (seeds in section 7 of "
          "the report). Judge-pending rows count as wrong (official = lower bound); the upper bound counts them as right. Paired by "
          "instance, cluster = TRACE task, strata = domain, McNemar on decoding seed 42.",
          "- External suite: descriptive only; each run's scorer code hash and fix-2/fix-3 counters are reported next to its scores.",
          "- Cross-checks: compare_cert arm means are compared with each eval dir's own `summary.<split>.json`, and the TRACE-val "
          "accuracy and pending counts with `summary.trace_val.json` (section 10 of the report).", "",
          "## Pending (picked up automatically by re-running `run.sh`)", ""]
    if doc["pending"]:
        for pnd in doc["pending"]:
            L.append(f"- `{pnd['model']}`: {pnd['what']}" + (f" ({pnd['progress']})" if pnd.get("progress") else "") + f"; `{pnd['path']}`")
    else:
        L.append("- nothing")
    L += ["", "Not used here: G_gated / P_gated and external seeds 43/44 (other analyses). 3B seeds 2-3 enter only the seed-spread "
          "reference; their own analysis is `results/final/cert_3b`.", ""]
    return "\n".join(L)


# ============================================================================== main


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.environ.get("TRACE_CERT_RUNS"), help="campaign root ($TRACE_CERT_RUNS)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-tool-src", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        if not args.runs:
            ap.error("--runs or $TRACE_CERT_RUNS is required")
        doc, side = build(args)
        CC = sys.modules["compare_cert"]
        for name, d in side.items():
            (out / f"{name}.json").write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
            (out / f"{name}.md").write_text(CC.markdown(d))
        (out / "cert_7b.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    doc = jload(out / "cert_7b.json")
    (out / "cert_7b.md").write_text(render(doc))
    (out / "README.md").write_text(render_readme(doc))
    print(f"wrote {out / 'cert_7b.json'} and {out / 'cert_7b.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
