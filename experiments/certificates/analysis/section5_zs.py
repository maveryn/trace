"""Section 5 report: zero-shot certificates across models plus the CPU digest (DESIGN 7.1, 8).

    python experiments/certificates/analysis/section5_zs.py \\
        --zs-root "$TRACE_CERT_RUNS/eval/zs" \\
        --rescore "$TRACE_CERT_RUNS/results/final/section5/rescore" \\
        --cpu ref="$TRACE_CERT_RUNS/results/section5_cpu_v2_gate" \\
        --cpu v2="$TRACE_CERT_RUNS/results/section5_cpu_v2" \\
        --cpu current="$TRACE_CERT_RUNS/results/final/section5/cpu_current" \\
        --out-json section5_zs.json --out-md section5_zs.md [--trace-tool-src "$TRACE_TOOL_ROOT/src"] \\
        [--design "$TRACE_TOOL_ROOT/external/cert_notes/DESIGN.md"]

Read-only on every input. Every number in the Markdown is rendered from the JSON document this
script writes (no hand-typed values).

Inputs
- ``--zs-root``: one directory per model (``plan.json``, ``validation.json``,
  ``<split>/<cond>_v1/seed<k>/cert/{cert_scores.jsonl, score_receipt.json}``, ``run_config.json``,
  ``summary.<split>.json``). A model whose ``validation.json`` is missing or not ok, or that lacks
  a run, is listed as pending and left out of the tables.
- ``--rescore``: ``rows.jsonl.gz`` and ``rescore_summary.json`` from ``section5_zs_rescore.py``
  (declared-frame reproduction, other-frame scores, coordinate ranges, identity sensitivity).
- ``--cpu NAME=DIR``: Section 5 CPU result dirs (``section5_cpu.py``: ``summary.json``,
  ``receipt.json``); the first one is the reference, the others are reproductions.

Method
- Rows: one per (model, split, condition, decoding seed, instance); conditions ``ans`` (answer
  only), ``ann`` (legacy answer + annotation) and ``cert`` (certificate prompt; px frame for
  Qwen2.5-VL, rel1000 for Qwen3.5). Levels are pooled over decoding seeds 42-44.
- CIs (DESIGN 8): per instance the mean over decoding seeds; scene_group clusters resampled within
  adapter strata, 10,000 resamples, seed 0, percentile 95% (``compare_cert.cluster_bootstrap``).
  They reflect instance sampling only (zero-shot models have no training seed).
- Paired contrasts (same model, two conditions; or two models, same condition): mean per-instance
  difference, same bootstrap; binary metrics add exact McNemar on the decoding-seed-42 outcome
  (``trace_tool.compare.mcnemar_exact``), continuous metrics the cluster sign-flip permutation test
  (10,000 flips; a p at the floor 1/(n_perm+1) is flagged ``p_at_floor`` and shown as "<=floor",
  and a Holm p driven by a floor p is flagged ``p_holm_floor_limited``). Holm
  (``trace_tool.compare.holm``) is applied within each condition-contrast family (cert-ans,
  ann-ans, cert-ann) and over the cross-model table; the frame re-readings are unadjusted. All of
  Section 5 is descriptive: none of these families is pre-registered.
- Qwen3.5 frame evidence per adapter (``frames.frame_evidence``): from the paired px_source -
  rel1000 contrasts of G and M on both test splits and both Qwen3.5 sizes. ``decisive`` = the G and
  M CIs exclude 0 in the same direction in every cell; ``leans`` = the G CI does in every cell and
  no M CI points the other way; else ``unclear``.
- Constants shown in the Markdown (decoding settings, slow-row threshold, verifier budget, go/no-go
  1(b) thresholds, identity thresholds) are read from their sources into ``constants`` /
  ``decoding_summary`` of the JSON first.
- Cross-check: pooled means, rates, token totals, A x (M==1) counts and error classes are compared
  with each model's ``summary.<split>.json`` (written by the scorer's ``summarize``); any difference
  is listed under ``crosscheck`` and the exit code is 1.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import compare_cert as cc  # noqa: E402  (read-only reuse of the DESIGN 8 statistics)

VERSION = "section5_zs-1.1.0"
SPLITS = ("e1_dev", "e1_test", "e2_test")
PRIMARY_SPLITS = ("e1_test", "e2_test")
SPLIT_LABEL = {"e1_dev": "E1-dev", "e1_test": "E1-test", "e2_test": "E2-test"}
CONDS = ("ans", "ann", "cert")
COND_DIR = {"ans": "ans_v1", "ann": "ann_v1", "cert": "cert_v1"}
SEEDS = (42, 43, 44)
# known models: key -> (label, kind, size); others are appended in name order
MODEL_INFO = {
    "qwen2.5-vl-3b-instruct": ("Qwen2.5-VL-3B", "base", "3B"),
    "trace-qwen2.5-vl-3b": ("TRACE-3B (legacy)", "legacy TRACE", "3B"),
    "qwen2.5-vl-7b-instruct": ("Qwen2.5-VL-7B", "base", "7B"),
    "trace-qwen2.5-vl-7b": ("TRACE-7B (legacy)", "legacy TRACE", "7B"),
    "qwen3.5-4b": ("Qwen3.5-4B", "base", "4B"),
    "qwen3.5-9b": ("Qwen3.5-9B", "base", "9B"),
}
CROSS_PAIRS = (
    ("trace-qwen2.5-vl-3b", "qwen2.5-vl-3b-instruct"),
    ("trace-qwen2.5-vl-7b", "qwen2.5-vl-7b-instruct"),
    ("qwen2.5-vl-7b-instruct", "qwen2.5-vl-3b-instruct"),
    ("qwen3.5-9b", "qwen3.5-4b"),
)
ADAPTERS = (
    "selected_set.chess_piece_count",
    "filtered_mean.table_filtered_column_mean",
    "weighted_sum.lever_side_torque",
    "path.rpg_movement_cost",
)
ADAPTER_LABEL = {
    "selected_set.chess_piece_count": "chess (selected_set)",
    "filtered_mean.table_filtered_column_mean": "table (filtered_mean)",
    "weighted_sum.lever_side_torque": "lever (weighted_sum)",
    "path.rpg_movement_cost": "rpg (path)",
}
ERROR_CLASSES = ("envelope", "grounding", "operand", "computation", "answer_inconsistent", "correct")
BINARY = {"A", "A_legacy", "F", "joint", "valid", "complete", "trunc", "w2048", "envelope"}
LEVEL_CI_METRICS = {
    "ans": ("A",),
    "ann": ("A", "G", "G_px"),
    "cert": ("A", "G", "M", "F", "valid", "joint"),
}
KEEP = ("instance_id", "adapter_id", "cell", "scene_group", "seed", "A", "A_legacy", "G", "G_id", "G_px", "G_frame",
        "G_shown", "M", "M_shown", "F", "joint", "valid_schema", "envelope_error", "error_class", "format_reasons",
        "completion_tokens", "stop_reason", "truncated", "within_2048", "verifier_ms", "slow_row", "coord_frame",
        "cond", "scorer_version", "row_index")


class ReportError(RuntimeError):
    pass


# The report keeps about 270,000 row dicts in memory (200,448 stored rows + 66,816 re-scored rows). The
# shared login node caps this account's memory, so rows are stored compactly: top-level None values of the stored
# rows are dropped (every read of them uses .get), repeated strings are interned, 0.0 / 1.0 share one object and
# lists become tuples. Values and therefore every number in the report are unchanged.
_CANON_FLOATS = {0.0: 0.0, 1.0: 1.0}
# re-scoring row fields this report never reads (kept in rows.jsonl.gz, dropped on load)
RESCORE_UNUSED = ("max_x", "max_y", "annotation_source", "shown_wh", "family", "declared_frame", "alt_frame",
                  "reproduced", "row_index", "cell", "joint_px_shown")


def _compact(v: Any) -> Any:
    if isinstance(v, str):
        return sys.intern(v)
    if isinstance(v, float):
        return _CANON_FLOATS.get(v, v)
    if isinstance(v, list):
        return tuple(_compact(x) for x in v)
    if isinstance(v, dict):
        return {sys.intern(k): (None if x is None else _compact(x)) for k, x in v.items()}
    return v


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, str):
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def metric_value(row: Mapping[str, Any], m: str) -> Optional[float]:
    if m == "valid":
        return _num(row.get("valid_schema"))
    if m == "complete":
        return 1.0 if row.get("stop_reason") == "answer" else 0.0
    if m == "trunc":
        return 1.0 if row.get("truncated") else 0.0
    if m == "w2048":
        return _num(row.get("within_2048"))
    if m == "envelope":
        return 1.0 if row.get("envelope_error") is not None else 0.0
    if m == "tokens":
        return _num(row.get("completion_tokens"))
    return _num(row.get(m))


def mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def quantiles(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if not v.size:
        return {"mean": None, "median": None, "p90": None, "p99": None, "max": None}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


# ===================================================================================== loading


def discover_models(zs_root: Path, names: Sequence[str]) -> List[str]:
    found = sorted(p.name for p in zs_root.iterdir() if (p / "plan.json").is_file())
    if names:
        found = [n for n in names if n in found]
    known = [m for m in MODEL_INFO if m in found]
    return known + [m for m in found if m not in MODEL_INFO]


def load_model(zs_root: Path, model: str) -> Dict[str, Any]:
    root = zs_root / model
    plan = json.loads((root / "plan.json").read_text())
    val_path = root / "validation.json"
    validation = json.loads(val_path.read_text()) if val_path.is_file() else None
    info: Dict[str, Any] = {
        "key": model,
        "label": MODEL_INFO.get(model, (model, "other", "?"))[0],
        "kind": MODEL_INFO.get(model, (model, "other", "?"))[1],
        "size": MODEL_INFO.get(model, (model, "other", "?"))[2],
        "model_id": plan.get("model"),
        "family": plan.get("family"),
        "frame": plan.get("frame"),
        "expected_rows": plan.get("expected_total"),
        "validation_ok": bool(validation and validation.get("ok")),
        "validation_done": validation.get("done") if validation else None,
        "validation_expected": validation.get("expected") if validation else None,
        "validation_problems": sum(len(r.get("problems") or []) for r in (validation or {}).get("runs", [])),
        "missing_runs": [],
        "runs": 0,
        "rows": 0,
        "endpoint_errors": 0,
        "rejected": 0,
        "scoring_code_sha256": Counter(),
        "scorer_versions": Counter(),
        "revision": None,
        "revisions": [],
        "decoding": None,
        "decoding_variants": [],
        "decoding_consistent": None,
        "n_run_configs": 0,
        "missing_run_configs": [],
        "coords_by_cond": {},
        "files": {},
    }
    dec_variants: Counter = Counter()
    revisions: set = set()
    coords: Dict[str, set] = defaultdict(set)
    rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for split in SPLITS:
        for cond in CONDS:
            for seed in SEEDS:
                run = root / split / COND_DIR[cond] / f"seed{seed}"
                scores = run / "cert" / "cert_scores.jsonl"
                receipt = run / "cert" / "score_receipt.json"
                if not scores.is_file() or not receipt.is_file():
                    info["missing_runs"].append(f"{split}/{COND_DIR[cond]}/seed{seed}")
                    continue
                rec = json.loads(receipt.read_text())
                resp = rec.get("responses") or {}
                info["endpoint_errors"] += int(resp.get("endpoint_errors") or 0)
                info["rejected"] += int(resp.get("rejected") or 0)
                info["scoring_code_sha256"][(rec.get("provenance") or {}).get("cert_code_sha256", "?")] += 1
                info["files"][f"{split}/{COND_DIR[cond]}/seed{seed}/cert/cert_scores.jsonl"] = sha256_file(scores)
                if (run / "run_config.json").is_file():  # every run_config, checked for consistency
                    rc = json.loads((run / "run_config.json").read_text())
                    revisions.add(str(rc.get("model_revision")))
                    st = rc.get("settings") or {}
                    dec = {
                        "preset": st.get("decoding"), **(st.get("decoding_params") or {}),
                        "max_tokens_per_turn": st.get("max_tokens_per_turn"),
                        "max_total_tokens": st.get("max_total_tokens"),
                        "enable_thinking": (st.get("chat_template_kwargs") or {}).get("enable_thinking"),
                    }
                    dec_variants[json.dumps(dec, sort_keys=True)] += 1
                    coords[cond].add(str(st.get("coords")))
                    info["n_run_configs"] += 1
                else:
                    info["missing_run_configs"].append(f"{split}/{COND_DIR[cond]}/seed{seed}")
                info["runs"] += 1
                with open(scores, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        row = {k: _compact(v) for k in KEEP if (v := r.get(k)) is not None}
                        row["cond_raw"] = row.get("cond")
                        row["cond"] = cond
                        row["split"] = split
                        row["seed"] = int(row["seed"]) if row.get("seed") is not None else seed
                        info["scorer_versions"][row.get("scorer_version")] += 1
                        rows[(split, cond)].append(row)
                        info["rows"] += 1
    info["scoring_code_sha256"] = dict(info["scoring_code_sha256"])
    info["scorer_versions"] = dict(info["scorer_versions"])
    info["decoding_variants"] = [{"settings": json.loads(k), "runs": v} for k, v in sorted(dec_variants.items())]
    info["decoding_consistent"] = len(dec_variants) == 1
    info["decoding"] = info["decoding_variants"][0]["settings"] if len(dec_variants) == 1 else None
    info["revisions"] = sorted(revisions)
    info["revision"] = info["revisions"][0] if len(revisions) == 1 else None
    info["coords_by_cond"] = {c: sorted(v) for c, v in coords.items()}
    info["complete"] = info["validation_ok"] and not info["missing_runs"]
    return {"info": info, "rows": rows}


# ===================================================================================== statistics


class Stats:
    def __init__(self, n_boot: int, n_perm: int, seed: int, mcnemar_seed: int, trace_tool_src: Optional[str]):
        self.n_boot, self.n_perm, self.seed, self.mcnemar_seed = n_boot, n_perm, seed, mcnemar_seed
        self.mcnemar, self.holm = cc.load_trace_tool(trace_tool_src)

    def instance_matrix(
        self, rows: Sequence[Mapping[str, Any]], metrics: Sequence[str],
        seeds: Optional[set] = None, value: Callable[[Mapping[str, Any], str], Optional[float]] = metric_value,
    ) -> Tuple[List[str], Dict[str, Tuple[str, str]], Dict[str, Dict[int, np.ndarray]]]:
        by: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        meta: Dict[str, Tuple[str, str]] = {}
        for r in rows:
            if seeds is not None and r["seed"] not in seeds:
                continue
            iid = r["instance_id"]
            meta[iid] = (str(r.get("scene_group") or iid), str(r["adapter_id"]))
            by[iid][int(r["seed"])] = np.array(
                [np.nan if (v := value(r, m)) is None else v for m in metrics], dtype=float)
        return sorted(by), meta, by

    @staticmethod
    def _seed_mean(per_seed: Dict[int, np.ndarray], seeds: Sequence[int], M: int) -> np.ndarray:
        arr = np.array([per_seed[s] for s in seeds if s in per_seed], dtype=float)
        if not arr.size:
            return np.full(M, np.nan)
        with np.errstate(invalid="ignore"):
            cnt = (~np.isnan(arr)).sum(axis=0)
            tot = np.nansum(arr, axis=0)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    def level(self, rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        ids, meta, by = self.instance_matrix(rows, metrics)
        M = len(metrics)
        X = np.array([self._seed_mean(by[i], sorted(by[i]), M) for i in ids])
        Y = np.where(np.isnan(X), np.nan, 0.0)
        cl = cc.make_clusters(X, Y, np.array([meta[i][0] for i in ids]), np.array([meta[i][1] for i in ids]))
        boot = cc.cluster_bootstrap(cl, self.n_boot, np.random.default_rng([self.seed, 1]))
        lo, hi = cc.percentile_ci(boot)
        est = cc._ratio(cl.S.sum(axis=0), cl.n_rows)
        return {m: {"mean_instance": _f(est[j]), "ci": [_f(lo[j]), _f(hi[j])], "n_instances": int(cl.n_rows[j])}
                for j, m in enumerate(metrics)}

    def paired(
        self, xrows: Sequence[Mapping[str, Any]], yrows: Sequence[Mapping[str, Any]], metrics: Sequence[str],
        xvalue: Callable = metric_value, yvalue: Callable = metric_value,
    ) -> Dict[str, Dict[str, Any]]:
        """X - Y per metric over instances present in both (seed means over shared seeds)."""
        xi, xmeta, xb = self.instance_matrix(xrows, metrics, value=xvalue)
        yi, _, yb = self.instance_matrix(yrows, metrics, value=yvalue)
        ids = sorted(set(xi) & set(yi))
        M = len(metrics)
        X, Y, X42, Y42 = [], [], [], []
        for i in ids:
            shared = sorted(set(xb[i]) & set(yb[i]))
            X.append(self._seed_mean(xb[i], shared, M))
            Y.append(self._seed_mean(yb[i], shared, M))
            X42.append(xb[i].get(self.mcnemar_seed, np.full(M, np.nan)) if self.mcnemar_seed in shared else np.full(M, np.nan))
            Y42.append(yb[i].get(self.mcnemar_seed, np.full(M, np.nan)) if self.mcnemar_seed in shared else np.full(M, np.nan))
        X, Y, X42, Y42 = (np.array(a, dtype=float).reshape(len(ids), M) for a in (X, Y, X42, Y42))
        cl = cc.make_clusters(X, Y, np.array([xmeta[i][0] for i in ids]), np.array([xmeta[i][1] for i in ids]))
        n = cl.n_rows
        delta = cc._ratio(cl.S.sum(axis=0), n)
        mx, my = cc._ratio(cl.SX.sum(axis=0), n), cc._ratio(cl.SY.sum(axis=0), n)
        boot = cc.cluster_bootstrap(cl, self.n_boot, np.random.default_rng([self.seed, 1]))
        lo, hi = cc.percentile_ci(boot)
        perm = cc.signflip_p(cl, self.n_perm, np.random.default_rng([self.seed, 2]))
        floor = self.perm_floor
        out: Dict[str, Dict[str, Any]] = {}
        for j, m in enumerate(metrics):
            res: Dict[str, Any] = {
                "n_instances": int(n[j]), "n_clusters": int(np.sum(cl.N[:, j] > 0)),
                "mean_x": _f(mx[j]), "mean_y": _f(my[j]), "delta": _f(delta[j]),
                "ci": [_f(lo[j]), _f(hi[j])], "perm_p": _f(perm[j]), "mcnemar": None,
            }
            res["perm_p_at_floor"] = res["perm_p"] is not None and res["perm_p"] <= floor * (1 + 1e-9)
            if m in BINARY:
                b, c, nn = cc.mcnemar_counts(X42[:, j], Y42[:, j])
                if nn:
                    res["mcnemar"] = {"seed": self.mcnemar_seed, "b_y_only": b, "c_x_only": c, "n": nn,
                                      "p": float(self.mcnemar(b, c))}
            if res["mcnemar"] is not None:
                res["test"], res["p"] = f"mcnemar_exact(seed {self.mcnemar_seed})", res["mcnemar"]["p"]
            else:
                res["test"], res["p"] = "cluster_signflip", res["perm_p"]
            res["p_at_floor"] = res["test"] == "cluster_signflip" and bool(res["perm_p_at_floor"])
            out[m] = res
        return out

    @property
    def perm_floor(self) -> float:
        """Smallest sign-flip p: (1 + 0) / (1 + n_perm) (``compare_cert.signflip_p``)."""
        return 1.0 / (self.n_perm + 1)

    def holm_family(self, entries: List[Dict[str, Any]]) -> None:
        ps = [e["p"] if e.get("p") is not None else 1.0 for e in entries]
        for e, a in zip(entries, self.holm(ps) if ps else []):
            e["p_holm"] = float(a)
            e["family_size"] = len(entries)
        # Holm step-down: adjusted p_(i) = max_{j<=i} min(1, (m-j+1) p_(j)). Flag the entries whose own
        # p sits at the permutation floor or whose adjusted value is set by such a p.
        order = sorted(range(len(entries)), key=lambda i: ps[i])
        run_val, run_floor = -1.0, False
        for rank, i in enumerate(order):
            v = min(1.0, (len(entries) - rank) * ps[i])
            at = bool(entries[i].get("p_at_floor"))
            if v > run_val * (1 + 1e-12):
                run_val, run_floor = v, at
            elif at and abs(v - run_val) <= 1e-12 * max(1.0, run_val):
                run_floor = True
            entries[i]["p_holm_floor_limited"] = bool((run_floor or at) and run_val < 1.0)


# ===================================================================================== levels


def describe(rows: Sequence[Mapping[str, Any]], cond: str) -> Dict[str, Any]:
    """Pooled (all decoding seeds) description of one model x split x condition."""
    n = len(rows)
    out: Dict[str, Any] = {
        "n": n,
        "n_instances": len({r["instance_id"] for r in rows}),
        "seeds": sorted({int(r["seed"]) for r in rows}),
    }
    keys = ["A", "A_legacy", "complete", "trunc", "w2048"]
    if cond == "ann":
        keys += ["G", "G_px", "G_id"]
        out["G_frame"] = sorted({str(r.get("G_frame")) for r in rows})
    if cond == "cert":
        keys += ["G", "G_id", "G_shown", "M", "M_shown", "F", "joint", "valid", "envelope"]
    out["mean"] = {k: mean([metric_value(r, k) for r in rows]) for k in keys}
    tok = [metric_value(r, "tokens") for r in rows]
    out["tokens"] = {**quantiles(tok), "total": int(sum(t for t in tok if t is not None))}
    vm = [r.get("verifier_ms") for r in rows if r.get("verifier_ms") is not None]
    out["verifier_ms"] = {**quantiles(vm), "slow_rows": sum(1 for r in rows if r.get("slow_row"))}
    out["stop_reasons"] = dict(Counter(str(r.get("stop_reason")) for r in rows))
    per_seed = {}
    for s in out["seeds"]:
        rs = [r for r in rows if int(r["seed"]) == s]
        per_seed[str(s)] = {k: mean([metric_value(r, k) for r in rs]) for k in
                            (["A"] + (["G"] if cond == "ann" else []) + (["G", "M", "F", "joint"] if cond == "cert" else []))}
        per_seed[str(s)]["n"] = len(rs)
    out["per_seed"] = per_seed
    if cond == "cert":
        c = Counter()
        for r in rows:
            a = (r.get("A") or 0.0) >= 1.0
            m = r.get("M")
            m = 0.0 if m is None else float(m)
            if a and m >= 1.0:
                c["A1_M1"] += 1
            elif a and m > 0.0:
                c["A1_Mpartial"] += 1
            elif a:
                c["A1_M0"] += 1
            elif m >= 1.0:
                c["A0_M1"] += 1
            elif m > 0.0:
                c["A0_Mpartial"] += 1
            else:
                c["A0_M0"] += 1
        tbb = {k: int(c.get(k, 0)) for k in ("A1_M1", "A1_Mpartial", "A1_M0", "A0_M1", "A0_Mpartial", "A0_M0")}
        tbb["A1_Mlt1"] = tbb["A1_Mpartial"] + tbb["A1_M0"]
        tbb["A0_Mlt1"] = tbb["A0_Mpartial"] + tbb["A0_M0"]
        n_a1 = tbb["A1_M1"] + tbb["A1_Mlt1"]
        n_m1 = tbb["A1_M1"] + tbb["A0_M1"]
        tbb["share_Mlt1_given_A1"] = tbb["A1_Mlt1"] / n_a1 if n_a1 else None
        tbb["share_A0_given_M1"] = tbb["A0_M1"] / n_m1 if n_m1 else None
        tbb["share_Mpartial_given_A0"] = tbb["A0_Mpartial"] / (n - n_a1) if n - n_a1 else None
        out["two_by_two"] = tbb
        ec = Counter(str(r.get("error_class")) for r in rows)
        out["error_classes"] = {k: int(ec.get(k, 0)) for k in ERROR_CLASSES}
        extra = {k: v for k, v in ec.items() if k not in ERROR_CLASSES}
        if extra:
            out["error_classes_other"] = dict(extra)
        out["envelope_errors"] = dict(Counter(str(r["envelope_error"]) for r in rows if r.get("envelope_error") is not None))
        fr = Counter()
        for r in rows:
            for x in r.get("format_reasons") or []:
                fr[x] += 1
        out["format_reasons"] = dict(fr.most_common())
        out["F0_rows"] = sum(1 for r in rows if (r.get("F") or 0.0) < 1.0)
    return out


# ===================================================================================== cross-check


def crosscheck(zs_root: Path, model: str, levels: Mapping[str, Any], by_adapter: Mapping[str, Any]) -> Dict[str, Any]:
    problems: List[str] = []
    n_checked = 0
    max_diff = 0.0
    cond_of = lambda raw: "cert" if str(raw).startswith("cert") else str(raw)  # noqa: E731
    pairs = [("A", "A"), ("A_legacy", "A_legacy"), ("G", "G"), ("G_id", "G_id"), ("G_px", "G_px"), ("G_shown", "G_shown"),
             ("M", "M"), ("M_shown", "M_shown"), ("F", "F"), ("joint", "joint"), ("complete", "completion_rate"),
             ("trunc", "truncation_rate"), ("valid", "valid_schema_rate"), ("w2048", "within_2048_rate"),
             ("envelope", "envelope_failure_rate")]

    def cmp(ours: Mapping[str, Any], theirs: Mapping[str, Any], where: str) -> None:
        nonlocal n_checked, max_diff
        if int(theirs.get("n", -1)) != int(ours["n"]):
            problems.append(f"{where}: n {theirs.get('n')} != {ours['n']}")
        for mine, key in pairs:
            if key in theirs and theirs[key] is not None:
                v = ours["mean"].get(mine)
                n_checked += 1
                if v is None:
                    problems.append(f"{where}: {mine} missing (summary {theirs[key]})")
                    continue
                d = abs(float(v) - float(theirs[key]))
                max_diff = max(max_diff, d)
                if d > 1e-9:
                    problems.append(f"{where}: {mine} {v} != {theirs[key]}")
        tk = theirs.get("tokens") or {}
        if "total" in tk:
            n_checked += 1
            if int(tk["total"]) != int(ours["tokens"]["total"]):
                problems.append(f"{where}: tokens total {tk['total']} != {ours['tokens']['total']}")
        if "A_x_M1" in theirs and "two_by_two" in ours:
            for k, v in theirs["A_x_M1"].items():
                n_checked += 1
                if int(ours["two_by_two"].get(k, -1)) != int(v):
                    problems.append(f"{where}: {k} {ours['two_by_two'].get(k)} != {v}")
        if "error_classes" in theirs and "error_classes" in ours:
            for k, v in theirs["error_classes"].items():
                n_checked += 1
                if int(ours["error_classes"].get(k, 0)) != int(v):
                    problems.append(f"{where}: error class {k} {ours['error_classes'].get(k)} != {v}")

    for split in SPLITS:
        p = zs_root / model / f"summary.{split}.json"
        if not p.is_file():
            problems.append(f"missing {p}")
            continue
        s = json.loads(p.read_text())
        for e in s.get("all", []):
            cond = cond_of(e["cond"])
            if cond in levels.get(split, {}):
                cmp(levels[split][cond], e, f"{model}/{split}/{cond}")
        for e in s.get("by_adapter_split", []):
            cond = cond_of(e["cond"])
            ours = by_adapter.get(split, {}).get(cond, {}).get(e["adapter_id"])
            if ours is not None and e.get("split") == split:
                cmp(ours, e, f"{model}/{split}/{cond}/{e['adapter_id']}")
    return {"n_checked": n_checked, "max_abs_diff": max_diff, "problems": problems}


# ===================================================================================== frames


def load_rescore(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summ = json.loads((path / "rescore_summary.json").read_text())
    rows = []
    with gzip.open(path / "rows.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                for k in RESCORE_UNUSED:
                    r.pop(k, None)
                rows.append(_compact(r))  # keeps None values (e.g. M_px_shown after a shown_error)
    return summ, rows


def frame_values(r: Mapping[str, Any], family: str) -> Dict[str, Optional[float]]:
    """G, M and joint of one cert row read in each of three frames: the declared contract frame,
    the other contract frame and the shown frame (``px_shown``). Shown-frame values are the
    scorer's own ``G_shown``/``M_shown`` where it logged them (Qwen2.5-VL), else the re-scoring's
    ``G_px_shown``/``M_px_shown`` (same scorer path with the server-reported shown size; it
    reproduces the scorer's Qwen2.5-VL values exactly, see ``frames.rescore.shown_replication``).
    Shown-frame joint = A = F = 1 and M_shown = 1 (A and F do not depend on the frame)."""
    st, alt = r["stored"], r["alt"]
    out = {"G_decl": st["G"], "G_alt": alt["G"], "M_decl": st["M"], "M_alt": alt["M"],
           "joint_decl": st["joint"], "joint_alt": alt["joint"], "G_shown": r.get("G_shown_stored"),
           "M_shown": r.get("M_shown_stored"), "joint_shown": None}
    if out["G_shown"] is None:
        out["G_shown"] = r.get("G_px_shown")
    if out["M_shown"] is None:
        out["M_shown"] = r.get("M_px_shown")
    if out["M_shown"] is not None:
        out["joint_shown"] = 1.0 if (st["A"] == 1.0 and st["F"] == 1.0 and out["M_shown"] == 1.0) else 0.0
    m_cands = [v for v in (out["M_decl"], out["M_alt"], out["M_shown"]) if v is not None]
    out["M_best"] = max(m_cands) if m_cands else None
    j_cands = [v for v in (out["joint_decl"], out["joint_alt"], out["joint_shown"]) if v is not None]
    out["joint_best"] = max(j_cands) if j_cands else None
    return out


def scale_gap(wh: Sequence[Any]) -> float:
    """How far a source image is from 1000 x 1000: max over axes of |size / 1000 - 1| (0 = the
    px_source and rel1000 readings coincide)."""
    return max(abs(float(wh[0]) / 1000.0 - 1.0), abs(float(wh[1]) / 1000.0 - 1.0))


def best_frame(r: Mapping[str, Any], names: Tuple[str, str]) -> str:
    """Which public contract frame (declared or the other one) gives the higher G for this row."""
    g = {names[0]: r["G_decl"], names[1]: r["G_alt"]}
    top = max(g.values())
    if top <= 0:
        return "none"
    winners = [k for k, v in g.items() if v >= top - 1e-12]
    return winners[0] if len(winners) == 1 else "tie"


def frame_block(rows: Sequence[Mapping[str, Any]], family: str, declared: str) -> Dict[str, Any]:
    alt = "px_source" if declared == "rel1000" else "rel1000"
    shown_name = "px_shown"
    names = (declared, alt)
    vals = [dict(frame_values(r, family), **{k: r[k] for k in ("n_coords", "le_1000", "in_source", "in_shown")}) for r in rows]
    wc = [v for v in vals if v["n_coords"]]
    bf = Counter(best_frame(v, names) for v in vals)
    sizes = Counter("x".join(str(int(x)) for x in r["source_wh"]) for r in rows)
    gaps = np.asarray([scale_gap(r["source_wh"]) for r in rows], dtype=float)
    return {
        "n": len(vals),
        "declared": declared, "alt": alt,
        "G": {declared: mean([v["G_decl"] for v in vals]), alt: mean([v["G_alt"] for v in vals]),
              shown_name: mean([v["G_shown"] for v in vals])},
        "M": {declared: mean([v["M_decl"] for v in vals]), alt: mean([v["M_alt"] for v in vals]),
              **({shown_name: mean([v["M_shown"] for v in vals])} if any(v["M_shown"] is not None for v in vals) else {}),
              "best_of_frames": mean([v["M_best"] for v in vals])},
        "joint": {declared: mean([v["joint_decl"] for v in vals]), alt: mean([v["joint_alt"] for v in vals]),
                  **({shown_name: mean([v["joint_shown"] for v in vals])} if any(v["joint_shown"] is not None for v in vals) else {}),
                  "best_of_frames": mean([v["joint_best"] for v in vals])},
        "best_of_frames_over": [declared, alt] + ([shown_name] if any(v["M_shown"] is not None for v in vals) else []),
        "rows_M_shown_missing": sum(1 for v in vals if v["M_shown"] is None),
        "scale_gap": ({"median": float(np.median(gaps)), "p10": float(np.percentile(gaps, 10)),
                       "p90": float(np.percentile(gaps, 90))} if gaps.size else None),
        "rows_with_coords": len(wc),
        "frac_le_1000": mean([1.0 if v["le_1000"] else 0.0 for v in wc]),
        "frac_in_source": mean([1.0 if v["in_source"] else 0.0 for v in wc]),
        "frac_in_shown": mean([1.0 if v["in_shown"] else 0.0 for v in wc]),
        "best_frame_by_G": {k: int(bf.get(k, 0)) for k in (declared, alt, "tie", "none")},
        "source_sizes_top": dict(sizes.most_common(5)),
        "n_source_sizes": len(sizes),
        "source_size_mode": sizes.most_common(1)[0][0] if sizes else None,
    }


# ===================================================================================== CPU digest


def cpu_digest(specs: Sequence[Tuple[str, Path]]) -> Dict[str, Any]:
    runs = []
    loaded = []
    for name, d in specs:
        sp, rp = d / "summary.json", d / "receipt.json"
        if not sp.is_file():
            runs.append({"name": name, "dir": str(d), "missing": True})
            continue
        s = json.loads(sp.read_text())
        r = json.loads(rp.read_text()) if rp.is_file() else {}
        loaded.append((name, s, r))
        n_types = sum(len(v) for v in s["corruptions"].values())
        runs.append({
            "name": name, "dir": str(d), "missing": False,
            "cert_code_sha256": s.get("cert_code_sha256"), "scorer_version": s.get("scorer_version"),
            "workers": s.get("workers"), "elapsed_s": s.get("elapsed_s"), "splits": s.get("splits"),
            "rows": s.get("rows"), "roundtrip_ok_total": s.get("roundtrip_ok_total"),
            "corruptions_scored": s.get("corruptions_scored"), "corruption_types": n_types,
            "corruption_expectations_unmet": s.get("corruption_expectations_unmet"),
            "outputs_sha256": (r.get("outputs") or {}),
            "summary_sha256": sha256_file(sp),
            "timing_all": s.get("timing_all"),
            "path_cap": s.get("path_cap"),
        })
    if not loaded:
        return {"runs": runs}
    ref_name, ref, _ = loaded[0]
    agree = {}
    for name, s, r in loaded[1:]:
        agree[name] = {
            "roundtrip_equal": s["roundtrip"] == ref["roundtrip"],
            "corruptions_equal": s["corruptions"] == ref["corruptions"],
            "go_no_go_equal": s["go_no_go_1b"] == ref["go_no_go_1b"],
            "corruptions_jsonl_sha_equal": (r.get("outputs") or {}).get("corruptions.jsonl")
            == (next(x for x in runs if x["name"] == ref_name)["outputs_sha256"]).get("corruptions.jsonl"),
            "differing_summary_keys": sorted(k for k in set(s) | set(ref) if s.get(k) != ref.get(k)),
        }
    per_adapter = {}
    for a in ADAPTERS:
        rt = ref["roundtrip"].get(a, {})
        cor = ref["corruptions"].get(a, {})
        gng = ref["go_no_go_1b"].get(a)
        tm = (ref.get("timing") or {}).get(a, {})
        per_adapter[a] = {
            "roundtrip": rt,
            "corruption_types": len(cor),
            "corrupted_responses": sum(int(v.get("rows", 0)) for v in cor.values()),
            "min_expect_ok_rate": min((v["expect_ok_rate"] for v in cor.values()), default=None),
            "min_intended_fail_rate": min((v["intended_fail_rate"] for v in cor.values()), default=None),
            "types_keep_G_lower_M_all_rows": sorted(k for k, v in cor.items() if v.get("keeps_g_lowers_m_rate") == 1.0),
            "go_no_go_1b": gng,
            "timing": tm,
            "corruptions": cor,
        }
    return {
        "reference": ref_name,
        "runs": runs,
        "agreement_with_reference": agree,
        "per_adapter": per_adapter,
        "timing_all": ref.get("timing_all"),
        "go_no_go_1b_pass_all_numeric": all(
            (ref["go_no_go_1b"].get(a) or {}).get("pass") for a in ADAPTERS if a in ref["go_no_go_1b"]),
        "numeric_adapters": [a for a in ADAPTERS if a in ref["go_no_go_1b"]],
    }


# ===================================================================================== constants


def read_constants(design: Optional[Path]) -> Dict[str, Any]:
    """Thresholds the Markdown quotes, read from their sources (never typed into the Markdown)."""
    from trace_tasks.experimental.certificates.evaluate import TOKEN_CAP
    from trace_tasks.experimental.certificates.identity import DEFAULT_IDENTITY
    from trace_tasks.experimental.certificates.scoring import SLOW_ROW_MS

    cpu_path = HERE / "section5_cpu.py"
    cpu_src = cpu_path.read_text()
    m_types = re.search(r"^GNG_MIN_TYPES\s*=\s*(\d+)", cpu_src, re.M)
    m_tol = re.search(r'abs\(c\["dG"\]\)\s*<\s*([0-9.]+)', cpu_src)
    budget = None
    if design is not None and design.is_file():
        m_b = re.search(r"Budget:\s*\u2264\s*([0-9.]+)\s*ms per row", design.read_text(encoding="utf-8"))
        budget = float(m_b.group(1)) if m_b else None
    return {
        "slow_row_ms": {"value": float(SLOW_ROW_MS), "source": "trace_tasks.experimental.certificates.scoring.SLOW_ROW_MS"},
        "within_token_cap": {"value": int(TOKEN_CAP), "source": "trace_tasks.experimental.certificates.evaluate.TOKEN_CAP"},
        "verifier_budget_ms": {"value": budget, "source": f"{design} ('Budget: <= X ms per row')" if design else None},
        "go_no_go_min_content_types": {"value": int(m_types.group(1)) if m_types else None,
                                       "source": f"{cpu_path} GNG_MIN_TYPES"},
        "go_no_go_keep_G_tol": {"value": float(m_tol.group(1)) if m_tol else None,
                                "source": f"{cpu_path} (keeps G: abs(dG) < X)"},
        "identity_frozen": {**DEFAULT_IDENTITY.to_dict(),
                            "source": "trace_tasks.experimental.certificates.identity.DEFAULT_IDENTITY"},
        "decoding_seeds": list(SEEDS),
    }


def ci_level() -> float:
    """Coverage of ``compare_cert.percentile_ci``, measured on a uniform grid (not typed here)."""
    lo, hi = cc.percentile_ci(np.linspace(0.0, 1.0, 100001)[:, None])
    return round(float(hi[0] - lo[0]), 6)


def decoding_summary(infos: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Decoding settings over every run_config.json of the complete models."""
    vals: Dict[str, set] = defaultdict(set)
    think: Dict[str, set] = defaultdict(set)
    for i in infos:
        for v in i.get("decoding_variants") or []:
            d = v["settings"]
            for k in ("preset", "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty",
                      "max_tokens_per_turn", "max_total_tokens"):
                vals[k].add(d.get(k))
            think[str(i.get("family"))].add(d.get("enable_thinking"))
    return {
        "n_run_configs": sum(int(i.get("n_run_configs") or 0) for i in infos),
        "missing_run_configs": sum(len(i.get("missing_run_configs") or []) for i in infos),
        "consistent_within_models": all(i.get("decoding_consistent") for i in infos),
        "values": {k: sorted(v, key=str) for k, v in vals.items()},
        "enable_thinking_by_family": {f: sorted(v, key=str) for f, v in think.items()},
    }


# ===================================================================================== build


def build(args: argparse.Namespace) -> Dict[str, Any]:
    zs_root = Path(args.zs_root).resolve()
    st = Stats(args.n_boot, args.n_perm, args.seed, args.mcnemar_seed, args.trace_tool_src)
    models = discover_models(zs_root, [m for m in args.models.split(",") if m])
    data: Dict[str, Dict[str, Any]] = {}
    infos, pending = [], []
    for m in models:
        d = load_model(zs_root, m)
        infos.append(d["info"])
        if d["info"]["complete"]:
            data[m] = d
        else:
            pending.append({"model": m, "reason": "validation not ok" if not d["info"]["validation_ok"] else
                            f"missing runs: {d['info']['missing_runs'][:6]}"})
    complete = [m for m in models if m in data]
    print(f"models complete: {complete}; pending: {[p['model'] for p in pending]}", flush=True)

    # ---- levels
    levels: Dict[str, Any] = {}
    by_adapter: Dict[str, Any] = {}
    checks: Dict[str, Any] = {}
    for m in complete:
        levels[m], by_adapter[m] = {}, {}
        for split in SPLITS:
            levels[m][split], by_adapter[m][split] = {}, {}
            for cond in CONDS:
                rows = data[m]["rows"].get((split, cond), [])
                if not rows:
                    continue
                lv = describe(rows, cond)
                lv["ci"] = st.level(rows, LEVEL_CI_METRICS[cond])
                levels[m][split][cond] = lv
                for a in ADAPTERS:
                    ra = [r for r in rows if r["adapter_id"] == a]
                    if ra:
                        by_adapter[m][split].setdefault(cond, {})[a] = describe(ra, cond)
        checks[m] = crosscheck(zs_root, m, levels[m], by_adapter[m])
        print(f"levels {m}: crosscheck {checks[m]['n_checked']} values, {len(checks[m]['problems'])} problems", flush=True)

    # ---- condition contrasts (same model)
    contrasts: Dict[str, List[Dict[str, Any]]] = {"cert_minus_ans": [], "ann_minus_ans": [], "cert_minus_ann": []}
    for fam, (cx, cy) in (("cert_minus_ans", ("cert", "ans")), ("ann_minus_ans", ("ann", "ans")),
                          ("cert_minus_ann", ("cert", "ann"))):
        for m in complete:
            for split in PRIMARY_SPLITS:
                xr, yr = data[m]["rows"].get((split, cx)), data[m]["rows"].get((split, cy))
                if not xr or not yr:
                    continue
                res = st.paired(xr, yr, ["A"])["A"]
                res.update({"model": m, "split": split, "x": cx, "y": cy, "metric": "A"})
                contrasts[fam].append(res)
        st.holm_family(contrasts[fam])

    # ---- cross-model contrasts (same condition)
    cross: List[Dict[str, Any]] = []
    for xm, ym in CROSS_PAIRS:
        if xm not in data or ym not in data:
            continue
        for split in PRIMARY_SPLITS:
            ans = st.paired(data[xm]["rows"][(split, "ans")], data[ym]["rows"][(split, "ans")], ["A"])
            cert = st.paired(data[xm]["rows"][(split, "cert")], data[ym]["rows"][(split, "cert")],
                             ["A", "G", "M", "F", "joint"])
            for metric, res in [("A_ans", ans["A"])] + [(f"{k}_cert", v) for k, v in cert.items()]:
                res.update({"x": xm, "y": ym, "split": split, "metric": metric})
                # G/M/joint of a rel1000 (Qwen3.5) model mix grounding with frame use (section 9.1);
                # A and F do not depend on the frame.
                res["frame_confounded"] = metric in ("G_cert", "M_cert", "joint_cert") and "rel1000" in (
                    data[xm]["info"]["frame"], data[ym]["info"]["frame"])
                cross.append(res)
    st.holm_family(cross)

    # ---- frames
    rsumm, rrows = load_rescore(Path(args.rescore))
    stored_shown: Dict[Tuple[str, str, int, str], Tuple[Any, Any, Any]] = {}
    for m in complete:
        for split in SPLITS:
            for r in data[m]["rows"].get((split, "cert"), []):
                stored_shown[(m, split, int(r["seed"]), r["instance_id"])] = (r.get("G_shown"), r.get("M_shown"), r.get("G_id"))
    by_ms: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rrows:
        if r["model"] not in data:
            continue
        gs, ms, gid = stored_shown.get((r["model"], r["split"], int(r["seed"]), r["instance_id"]), (None, None, None))
        r["G_shown_stored"], r["M_shown_stored"], r["G_id_stored"] = gs, ms, gid
        by_ms[(r["model"], r["split"])].append(r)
    n_cert_rows = {m: sum(len(data[m]["rows"].get((s, "cert"), [])) for s in SPLITS) for m in complete}
    n_rescored = {m: sum(len(by_ms.get((m, s), [])) for s in SPLITS) for m in complete}
    frames: Dict[str, Any] = {"by_model": {}, "contrasts": [], "ann": {}}
    for m in complete:
        fam, declared = data[m]["info"]["family"], ("rel1000" if data[m]["info"]["frame"] == "rel1000" else "px_source")
        frames["by_model"][m] = {}
        for split in SPLITS:
            rs = by_ms.get((m, split), [])
            if not rs:
                continue
            blk = {"all": frame_block(rs, fam, declared)}
            for a in ADAPTERS:
                ra = [r for r in rs if r["adapter_id"] == a]
                if ra:
                    blk[a] = frame_block(ra, fam, declared)
            frames["by_model"][m][split] = blk
        # paired: other frame - declared (and shown - declared for Qwen2.5-VL), on G, M, joint
        for split in PRIMARY_SPLITS:
            rs = by_ms.get((m, split), [])
            if not rs:
                continue
            for scope in ("all",) + ADAPTERS:
                rsc = rs if scope == "all" else [r for r in rs if r["adapter_id"] == scope]
                if not rsc:
                    continue
                alt_rows = [dict(r, G=r["alt"]["G"], M=r["alt"]["M"], joint=r["alt"]["joint"]) for r in rsc]
                dec_rows = [dict(r, G=r["stored"]["G"], M=r["stored"]["M"], joint=r["stored"]["joint"]) for r in rsc]
                res = st.paired(alt_rows, dec_rows, ["G", "M", "joint"])
                for metric, v in res.items():
                    v.update({"model": m, "split": split, "scope": scope, "metric": metric,
                              "x": "px_source" if declared == "rel1000" else "rel1000", "y": declared})
                    frames["contrasts"].append(v)
                if fam == "qwen2.5-vl":
                    sh_rows = [dict(r, G=r["G_shown_stored"], M=r["M_shown_stored"]) for r in rsc]
                    res = st.paired(sh_rows, dec_rows, ["G", "M"])
                    for metric, v in res.items():
                        v.update({"model": m, "split": split, "scope": scope, "metric": metric, "x": "px_shown", "y": declared})
                        frames["contrasts"].append(v)
        # ann: family-native frame vs px_source
        frames["ann"][m] = {}
        for split in SPLITS:
            rs = data[m]["rows"].get((split, "ann"), [])
            if rs:
                native = sorted({str(r.get("G_frame")) for r in rs})
                res = st.paired([dict(r, G=r.get("G_px")) for r in rs], rs, ["G"])["G"]
                res.update({"x": "px_source", "y": "/".join(native)})
                frames["ann"][m][split] = {"native_frame": native, "G_native": levels[m][split]["ann"]["mean"]["G"],
                                           "G_px_source": levels[m][split]["ann"]["mean"]["G_px"], "px_minus_native": res}
    frames["rescore"] = {
        "version": rsumm.get("version"), "scorer_version": rsumm.get("scorer_version"),
        "cert_code_sha256": rsumm.get("cert_code_sha256"), "n_rows": rsumm.get("n_rows"),
        "n_mismatch_total": rsumm.get("n_mismatch_total"), "absent_runs": rsumm.get("absent_runs"),
        "rows_jsonl_gz_sha256": rsumm.get("rows_jsonl_gz_sha256"),
        "rows_jsonl_sha256": rsumm.get("rows_jsonl_sha256"),
        "gzip_header": rsumm.get("gzip_header"),
        "shown_replication": rsumm.get("shown_replication"),
        "af_frame_dependent_total": rsumm.get("af_frame_dependent_total"),
        "m_px_shown_errors_total": rsumm.get("m_px_shown_errors_total"),
        "iou_sensitivity": rsumm.get("iou_sensitivity"),
        "containment_threshold": rsumm.get("containment_threshold"),
        "rows_cover_all_cert_rows": all(n_cert_rows[m] == n_rescored[m] for m in complete),
        "cert_rows": n_cert_rows, "rescored_rows": n_rescored,
    }
    frames["note"] = qwen35_note(frames, levels, complete)
    frames["frame_diag_hints"] = {}
    for m in complete:
        hints = {}
        for tag, root in (("full_e1_dev_seed42", zs_root), ("smoke_20_rows", zs_root.parent / "zs_smoke")):
            fp = root / m / "e1_dev" / "cert_v1" / "seed42" / "cert" / "frame_diag.json"
            if fp.is_file():
                fd = json.loads(fp.read_text())
                agg = fd.get("aggregate") or {}
                hints[tag] = {"path": str(fp), "hint": fd.get("hint"), "n": agg.get("n"),
                              "G_rel1000": agg.get("G_rel1000"), "G_px_source": agg.get("G_px_source"),
                              "G_px_shown": agg.get("G_px_shown")}
        frames["frame_diag_hints"][m] = hints
    frames["frame_evidence"] = frame_evidence(frames, {m: data[m]["info"]["frame"] for m in complete}, complete)
    fe = frames["frame_evidence"]
    px_any, rel_any = bool(fe["decisive_px_source"]), bool(fe["decisive_rel1000"])
    frames["qwen35_verdict"] = {
        "obedience": "mixed" if (px_any and rel_any) else ("rel1000" if rel_any else ("px_source" if px_any else "unknown")),
        "obedience_basis": "adapters with decisive frame evidence (frames.frame_evidence)",
        "max_M_best_of_frames": max((n["M_best"] for n in frames["note"]), default=None),
        "max_joint_best_of_frames": max((n["joint_best"] for n in frames["note"]), default=None),
        "best_of_frames_over": sorted({f for n in frames["note"] for f in n["best_of_frames_over"]}),
    }
    frames["all_models_max"] = {
        "joint": max((levels[m][s]["cert"]["mean"]["joint"], m, s) for m in complete for s in PRIMARY_SPLITS) if complete else None,
        "M": max((levels[m][s]["cert"]["mean"]["M"], m, s) for m in complete for s in PRIMARY_SPLITS) if complete else None,
    }

    # ---- identity sensitivity (E1-dev)
    constants = read_constants(Path(args.design).resolve() if args.design else None)
    frozen_iou = float(constants["identity_frozen"]["iou_threshold"])
    sens_iou = [float(x) for x in (rsumm.get("iou_sensitivity") or [])]
    iou_tags = [f"{t:.1f}" for t in sorted({frozen_iou, *sens_iou})]
    frozen_tag = f"{frozen_iou:.1f}"
    ident: Dict[str, Any] = {}
    for m in complete:
        rs = by_ms.get((m, "e1_dev"), [])
        rs = [r for r in rs if all(f"iou{t}" in r for t in iou_tags if t != frozen_tag)]
        if not rs or not sens_iou:
            continue
        ent = {"n": len(rs), "tags": iou_tags, "frozen": frozen_tag}
        for tag in iou_tags:
            get = ((lambda r: dict(r["stored"], G_id=r["G_id_stored"])) if tag == frozen_tag
                   else (lambda r, _t=tag: r[f"iou{_t}"]))
            ent[tag] = {"M": mean([get(r)["M"] for r in rs]), "G_id": mean([get(r).get("G_id") for r in rs]),
                        "joint": mean([get(r)["joint"] for r in rs]),
                        "rows_M_changed_vs_frozen": sum(1 for r in rs if abs(get(r)["M"] - r["stored"]["M"]) > 1e-12),
                        "rows_joint_changed_vs_frozen": sum(1 for r in rs if get(r)["joint"] != r["stored"]["joint"])}
        ident[m] = ent

    # ---- CPU digest
    cpu = cpu_digest([(n, Path(p).resolve()) for n, p in (s.split("=", 1) for s in args.cpu)])

    headline = make_headline(levels, contrasts, frames, ident, cpu, complete)

    for info in infos:
        info["scoring_code_sha256"] = {k[:12]: v for k, v in info["scoring_code_sha256"].items()}
    doc = {
        "version": VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": " ".join([sys.executable] + sys.argv),
        "inputs": {"zs_root": str(zs_root), "rescore": str(Path(args.rescore).resolve()),
                   "cpu": {n: str(Path(p).resolve()) for n, p in (s.split("=", 1) for s in args.cpu)},
                   "compare_cert_sha256": sha256_file(HERE / "compare_cert.py"),
                   "this_script_sha256": sha256_file(Path(__file__).resolve())},
        "settings": {"n_boot": args.n_boot, "n_perm": args.n_perm, "perm_floor": st.perm_floor,
                     "seed": args.seed, "mcnemar_seed": args.mcnemar_seed,
                     "holm_families": ["contrasts.cert_minus_ans", "contrasts.ann_minus_ans", "contrasts.cert_minus_ann",
                                       "cross_model"],
                     "unadjusted": ["frames.contrasts", "frames.ann"],
                     "cluster": "scene_group", "strata": "adapter_id", "ci": "percentile",
                     "ci_level": ci_level(),
                     "seed_aggregation": "mean over decoding seeds per instance (pooled means are over rows)"},
        "models": infos,
        "decoding_summary": decoding_summary([i for i in infos if i["key"] in complete]),
        "constants": constants,
        "complete_models": complete,
        "pending": pending,
        "headline": headline,
        "levels": levels,
        "by_adapter": by_adapter,
        "contrasts": contrasts,
        "cross_model": cross,
        "frames": frames,
        "identity_sensitivity": ident,
        "cpu": cpu,
        "crosscheck": checks,
        "crosscheck_problems_total": sum(len(c["problems"]) for c in checks.values()),
    }
    return doc


def _range(items: Sequence[Tuple[float, str, str]]) -> Dict[str, Any]:
    items = [i for i in items if i[0] is not None]
    if not items:
        return {}
    lo, hi = min(items), max(items)
    return {"min": lo[0], "min_model": lo[1], "min_split": lo[2], "max": hi[0], "max_model": hi[1], "max_split": hi[2]}


def make_headline(levels: Mapping[str, Any], contrasts: Mapping[str, Any], frames: Mapping[str, Any],
                  ident: Mapping[str, Any], cpu: Mapping[str, Any], complete: Sequence[str]) -> Dict[str, Any]:
    """Summary numbers for the key-findings list (all derived from the sections below)."""
    if not complete:
        return {}
    cert = [(m, s, levels[m][s]["cert"]) for m in complete for s in PRIMARY_SPLITS]
    h: Dict[str, Any] = {
        "joint": _range([(c["mean"]["joint"], m, s) for m, s, c in cert]),
        "M": _range([(c["mean"]["M"], m, s) for m, s, c in cert]),
        "A_ans": _range([(levels[m][s]["ans"]["mean"]["A"], m, s) for m in complete for s in PRIMARY_SPLITS]),
        "A_ans_e1": _range([(levels[m]["e1_test"]["ans"]["mean"]["A"], m, "e1_test") for m in complete]),
        "M_e1": _range([(levels[m]["e1_test"]["cert"]["mean"]["M"], m, "e1_test") for m in complete]),
        "share_Mlt1_given_A1": _range([(c["two_by_two"]["share_Mlt1_given_A1"], m, s) for m, s, c in cert]),
        "grounding_share": _range([(c["error_classes"]["grounding"] / c["n"], m, s) for m, s, c in cert]),
        "envelope_share": _range([(c["error_classes"]["envelope"] / c["n"], m, s) for m, s, c in cert]),
        "correct_share": _range([(c["error_classes"]["correct"] / c["n"], m, s) for m, s, c in cert]),
        "F": _range([(c["mean"]["F"], m, s) for m, s, c in cert]),
        "trunc_cert": _range([(c["mean"]["trunc"], m, s) for m, s, c in cert]),
        "A1_M1_total": sum(c["two_by_two"]["A1_M1"] for _, _, c in cert),
        "rows_cert_total": sum(c["n"] for _, _, c in cert),
    }
    cost = {}
    for fam in ("cert_minus_ans", "ann_minus_ans"):
        ents = contrasts.get(fam, [])
        cost[fam] = {
            "delta": _range([(e["delta"], e["model"], e["split"]) for e in ents]),
            "n_tests": len(ents),
            "n_negative": sum(1 for e in ents if (e["delta"] or 0) < 0),
            "n_holm_sig_negative": sum(1 for e in ents if (e["delta"] or 0) < 0 and e.get("p_holm", 1) < 0.05),
            "n_ci_below_zero": sum(1 for e in ents if e["ci"][1] is not None and e["ci"][1] < 0),
        }
    h["cost"] = cost
    h["qwen35"] = frames.get("qwen35_verdict")
    if ident:
        e0 = next(iter(ident.values()))
        fz, others = e0["frozen"], [t for t in e0["tags"] if t != e0["frozen"]]
        h["identity_tags"] = {"frozen": fz, "others": others, "lowest": min(others, key=float)}
        h["identity_M_shift"] = {m: {t: (e[t]["M"] - e[fz]["M"]) for t in others} for m, e in ident.items()}
        h["identity_M_max_lowest"] = _range([(e[h["identity_tags"]["lowest"]]["M"], m, "e1_dev") for m, e in ident.items()])
        h["identity_M_shift_range"] = _range([(e[t]["M"] - e[fz]["M"], m, "e1_dev") for m, e in ident.items() for t in others])
    ref = next((r for r in cpu.get("runs", []) if r.get("name") == cpu.get("reference")), None)
    if ref:
        h["cpu"] = {"roundtrip_ok": ref["roundtrip_ok_total"], "rows": ref["rows"], "corruptions": ref["corruptions_scored"],
                    "types": ref["corruption_types"], "unmet": ref["corruption_expectations_unmet"],
                    "verifier_median_ms": (cpu.get("timing_all") or {}).get("verifier_ms", {}).get("median"),
                    "verifier_p99_ms": (cpu.get("timing_all") or {}).get("verifier_ms", {}).get("p99"),
                    "go_no_go_1b_pass": cpu.get("go_no_go_1b_pass_all_numeric"),
                    "reproductions_identical": all(a["roundtrip_equal"] and a["corruptions_equal"] and a["go_no_go_equal"]
                                                   for a in cpu.get("agreement_with_reference", {}).values()),
                    "n_reproductions": len(cpu.get("agreement_with_reference", {}))}
    return h


def _ci_sign(ci: Sequence[Any]) -> int:
    if not ci or ci[0] is None or ci[1] is None:
        return 0
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


FRAME_EVIDENCE_RULE = (
    "per adapter, the paired px_source - rel1000 contrasts of G and M (cluster-bootstrap percentile CIs) in every "
    "Qwen3.5 model x test split cell: decisive = the G and M CIs exclude 0 in the same direction in every cell; "
    "leans = the G CI excludes 0 in the same direction in every cell and no M CI excludes 0 in the other direction; "
    "unclear = otherwise")


def frame_evidence(frames: Mapping[str, Any], frame_of: Mapping[str, str], complete: Sequence[str]) -> Dict[str, Any]:
    """Which contract frame Qwen3.5 (declared rel1000) coordinates fit, per adapter (FRAME_EVIDENCE_RULE)."""
    idx = {(c["model"], c["split"], c["scope"], c["metric"]): c for c in frames["contrasts"]
           if c["x"] == "px_source" and c["y"] == "rel1000"}
    models = [m for m in complete if frame_of.get(m) == "rel1000"]
    out: Dict[str, Any] = {"rule": FRAME_EVIDENCE_RULE, "models": models, "splits": list(PRIMARY_SPLITS), "per_adapter": {}}
    for a in ADAPTERS:
        cells = []
        for m in models:
            for split in PRIMARY_SPLITS:
                g, mm = idx.get((m, split, a, "G")), idx.get((m, split, a, "M"))
                blk = frames["by_model"].get(m, {}).get(split, {}).get(a)
                if not g or not mm or not blk:
                    continue
                cells.append({"model": m, "split": split, "dG": g["delta"], "ciG": g["ci"], "pG": g["perm_p"],
                              "dM": mm["delta"], "ciM": mm["ci"], "pM": mm["perm_p"],
                              "sG": _ci_sign(g["ci"]), "sM": _ci_sign(mm["ci"]),
                              "scale_gap_median": (blk.get("scale_gap") or {}).get("median"),
                              "source_size_mode": blk.get("source_size_mode"), "n_source_sizes": blk.get("n_source_sizes"),
                              "rows_G_higher_px": blk["best_frame_by_G"].get("px_source", 0),
                              "rows_G_higher_rel": blk["best_frame_by_G"].get("rel1000", 0)})
        if not cells:
            continue
        signs = {c["sG"] for c in cells}
        sgn = next(iter(signs)) if len(signs) == 1 else 0
        if sgn and all(c["sM"] == sgn for c in cells):
            strength = "decisive"
        elif sgn and not any(c["sM"] == -sgn for c in cells):
            strength = "leans"
        else:
            strength = "unclear"
        direction = {1: "px_source", -1: "rel1000"}.get(sgn) if strength != "unclear" else None
        dg = [c["dG"] for c in cells if c["dG"] is not None]
        dm = [c["dM"] for c in cells if c["dM"] is not None]
        gaps = [c["scale_gap_median"] for c in cells if c["scale_gap_median"] is not None]
        out["per_adapter"][a] = {
            "strength": strength, "direction": direction, "n_cells": len(cells),
            "G_delta_range": [min(dg), max(dg)] if dg else None, "M_delta_range": [min(dm), max(dm)] if dm else None,
            "n_G_ci_px": sum(1 for c in cells if c["sG"] == 1), "n_G_ci_rel": sum(1 for c in cells if c["sG"] == -1),
            "n_M_ci_px": sum(1 for c in cells if c["sM"] == 1), "n_M_ci_rel": sum(1 for c in cells if c["sM"] == -1),
            "M_against_G": [c for c in cells if c["sG"] and c["sM"] == -c["sG"]],
            "scale_gap_median_range": [min(gaps), max(gaps)] if gaps else None,
            "rows_G_higher_px": sum(c["rows_G_higher_px"] for c in cells),
            "rows_G_higher_rel": sum(c["rows_G_higher_rel"] for c in cells),
            "cells": cells,
        }
    pa = out["per_adapter"]
    for strength in ("decisive", "leans"):
        for direction in ("px_source", "rel1000"):
            out[f"{strength}_{direction}"] = [a for a, e in pa.items() if e["strength"] == strength and e["direction"] == direction]
    out["unclear"] = [a for a, e in pa.items() if e["strength"] == "unclear"]
    return out


def qwen35_note(frames: Mapping[str, Any], levels: Mapping[str, Any], complete: Sequence[str]) -> List[Dict[str, Any]]:
    """Data for the Qwen3.5 frame-confound note (rendered in the Markdown)."""
    out = []
    for m in complete:
        bm = frames["by_model"].get(m, {})
        any_blk = next(iter(bm.values()), {}).get("all") if bm else None
        if not any_blk or any_blk["declared"] != "rel1000":
            continue
        for split in PRIMARY_SPLITS:
            blk = bm.get(split)
            if not blk:
                continue
            per = {}
            for a in ADAPTERS:
                b = blk.get(a)
                if b:
                    per[a] = {"G_rel1000": b["G"]["rel1000"], "G_px_source": b["G"]["px_source"],
                              "M_rel1000": b["M"]["rel1000"], "M_px_source": b["M"]["px_source"],
                              "M_best": b["M"]["best_of_frames"], "frac_in_source": b["frac_in_source"],
                              "frac_le_1000": b["frac_le_1000"], "source_size_mode": b["source_size_mode"],
                              "n_source_sizes": b["n_source_sizes"], "scale_gap": b.get("scale_gap"),
                              # point comparison of G only (no CI, no M); the per-adapter verdict is frames.frame_evidence
                              "G_point_higher_as": "px_source" if (b["G"]["px_source"] or 0) > (b["G"]["rel1000"] or 0) else "rel1000"}
            a_all = blk["all"]
            out.append({
                "model": m, "split": split, "per_adapter": per,
                "G_rel1000": a_all["G"]["rel1000"], "G_px_source": a_all["G"]["px_source"],
                "M_rel1000": a_all["M"]["rel1000"], "M_px_source": a_all["M"]["px_source"],
                "M_px_shown": a_all["M"].get("px_shown"),
                "M_best": a_all["M"]["best_of_frames"], "joint_rel1000": a_all["joint"]["rel1000"],
                "joint_best": a_all["joint"]["best_of_frames"], "best_of_frames_over": a_all["best_of_frames_over"],
                "frac_le_1000": a_all["frac_le_1000"], "frac_in_source": a_all["frac_in_source"],
                "best_frame_by_G": a_all["best_frame_by_G"],
                "A_cert": levels[m][split]["cert"]["mean"]["A"], "A_ans": levels[m][split]["ans"]["mean"]["A"],
                "adapters_G_point_higher_px": [a for a, v in per.items() if v["G_point_higher_as"] == "px_source"],
                "adapters_G_point_higher_rel": [a for a, v in per.items() if v["G_point_higher_as"] == "rel1000"],
            })
    return out


# ===================================================================================== markdown


def f3(v: Any) -> str:
    return "-" if v is None else f"{float(v):.3f}"


def pct(v: Any, d: int = 1) -> str:
    return "-" if v is None else f"{100.0 * float(v):.{d}f}%"


def pp(v: Any, d: int = 1) -> str:
    return "-" if v is None else f"{100.0 * float(v):+.{d}f}"


def ci_pp(ci: Sequence[Any], d: int = 1) -> str:
    if not ci or ci[0] is None:
        return "-"
    return f"[{100.0 * ci[0]:+.{d}f}, {100.0 * ci[1]:+.{d}f}]"


def ci3(ci: Sequence[Any]) -> str:
    if not ci or ci[0] is None:
        return "-"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def pv(v: Any, floor: Optional[float] = None) -> str:
    """p value; a sign-flip p at its permutation floor (pass ``floor``) is shown as "<=floor"."""
    if v is None:
        return "-"
    v = float(v)
    if floor is not None and v <= floor * (1 + 1e-9):
        return f"<={floor:.1e}"
    return f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"


def p_raw(e: Mapping[str, Any], floor: float) -> str:
    """The entry's test p (McNemar exact, or sign-flip with its floor marked)."""
    return pv(e.get("p"), floor if e.get("test") == "cluster_signflip" else None)


def p_holm(e: Mapping[str, Any]) -> str:
    return pv(e.get("p_holm")) + (" (floor)" if e.get("p_holm_floor_limited") else "")


def num(v: Any) -> str:
    """A constant as written in its source (2.0 -> 2, 0.01 -> 0.01); '?' when it could not be read."""
    return "?" if v is None else f"{float(v):g}"


def one_or_all(vals: Sequence[Any]) -> str:
    return "?" if not vals else (f"{vals[0]:,}" if len(vals) == 1 and isinstance(vals[0], int) and not isinstance(vals[0], bool)
                                 else " / ".join(str(v) for v in vals))


def intc(v: Any) -> str:
    return "-" if v is None else f"{int(v):,}"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: Optional[str] = None) -> List[str]:
    al = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if c == "r" else "---" for c in al]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def levels_m(doc: Mapping[str, Any], model: str, split: str = "e1_test") -> Optional[float]:
    return doc["levels"][model][split]["cert"]["mean"]["M"]


def best_m(doc: Mapping[str, Any], model: str, split: str = "e1_test") -> Optional[float]:
    return doc["frames"]["by_model"][model][split]["all"]["M"]["best_of_frames"]


def fe_list(doc: Mapping[str, Any], key: str) -> str:
    return ", ".join(ADAPTER_LABEL.get(a, a) for a in doc["frames"]["frame_evidence"].get(key, [])) or "none"


def fe_rest(doc: Mapping[str, Any]) -> str:
    fe = doc["frames"]["frame_evidence"]
    parts = [f"{ADAPTER_LABEL.get(a, a)} leans {e['direction']}" if e["strength"] == "leans" else f"{ADAPTER_LABEL.get(a, a)} unclear"
             for a, e in fe["per_adapter"].items() if e["strength"] != "decisive"]
    return "; ".join(parts) or "none"


def markdown(doc: Mapping[str, Any]) -> str:
    L: List[str] = []
    models = doc["complete_models"]
    info = {i["key"]: i for i in doc["models"]}
    lab = lambda m: info[m]["label"]  # noqa: E731
    lv = doc["levels"]
    st = doc["settings"]
    ds = doc["decoding_summary"]
    kc = doc["constants"]
    floor = st["perm_floor"]
    cil = f"{100.0 * st['ci_level']:g}%"
    frames_by: Dict[str, List[str]] = defaultdict(list)
    for m in models:
        frames_by[str(info[m]["frame"])].append(lab(m))
    thinking = "; ".join(f"{fam} chat-template enable_thinking = {str(v[0]).lower()}"
                         for fam, v in ds["enable_thinking_by_family"].items() if v and v != [None])
    L += [f"# Section 5: zero-shot certificates across {len(models)} models",
          "",
          f"Generated {doc['generated_utc']} by `{doc['version']}` from the raw `cert_scores.jsonl` rows under "
          f"`{doc['inputs']['zs_root']}`, the re-scoring in `{doc['inputs']['rescore']}` and the Section 5 CPU result "
          f"directories listed in section 11. Every number below is read from `section5_zs.json`.",
          "",
          "Conditions (DESIGN 7.1): `ans` = TRACE answer-only prompt; `ann` = legacy answer + annotation prompt; "
          "`cert` = certificate prompt in the frame each model's plan declares ("
          + "; ".join(f"{fr}: {', '.join(labs)}" for fr, labs in frames_by.items()) + "). "
          f"Decoding, read from all {intc(ds['n_run_configs'])} `run_config.json` files "
          f"({'one setting per model' if ds['consistent_within_models'] else 'NOT one setting per model; see models[].decoding_variants'}): "
          f"preset `{one_or_all(ds['values'].get('preset', []))}`, temperature {one_or_all(ds['values'].get('temperature', []))}, "
          f"top-p {one_or_all(ds['values'].get('top_p', []))}, top-k {one_or_all(ds['values'].get('top_k', []))}, "
          f"{one_or_all(ds['values'].get('max_tokens_per_turn', []))}-token budget per response"
          + (f"; {thinking}" if thinking else "") + ". "
          f"Levels are pooled over decoding seeds {', '.join(str(s) for s in kc['decoding_seeds'])}. "
          f"{cil} CIs (percentile): cluster bootstrap over scene groups within adapter strata "
          f"({st['n_boot']:,} resamples, seed {st['seed']}), instance sampling only. Paired tests: exact McNemar on the "
          f"seed-{st['mcnemar_seed']} outcome (binary metrics) or cluster sign-flip permutation ({st['n_perm']:,} flips; "
          f"G, M); a sign-flip p at its floor 1/({st['n_perm']:,}+1) is shown as {pv(floor, floor)}, and a Holm p set by "
          "such a value is marked (floor): it is limited by the number of flips, not by the data. Holm is applied within "
          "each condition-contrast table of section 2 and over the cross-model table of section 8; the frame contrasts "
          "of section 9 are unadjusted. Everything in Section 5 is descriptive (no pre-registered family).",
          ""]
    hl = doc.get("headline") or {}
    if hl:
        c1 = hl["cost"]["cert_minus_ans"]
        c2 = hl["cost"]["ann_minus_ans"]
        cpu_h = hl.get("cpu") or {}
        q = hl.get("qwen35") or {}
        L += ["## Key findings", "",
              f"1. **{'No zero-shot model writes verifiable certificates' if hl['joint']['max'] < 0.1 else 'Zero-shot certificate quality'}.** Across {len(models)} models x E1-test/E2-test, "
              f"joint (A and F and M = 1) is at most {f3(hl['joint']['max'])} ({lab(hl['joint']['max_model'])}, "
              f"{SPLIT_LABEL[hl['joint']['max_split']]}) and M at most {f3(hl['M']['max'])} "
              f"({lab(hl['M']['max_model'])}, {SPLIT_LABEL[hl['M']['max_split']]}); only {intc(hl['A1_M1_total'])} of "
              f"{intc(hl['rows_cert_total'])} `cert` rows have a correct answer with a fully verified certificate.",
              f"2. **Correct answers come with certificates that do not verify.** Among rows with A = 1, the share with "
              f"M < 1 ranges from {pct(hl['share_Mlt1_given_A1']['min'])} ({lab(hl['share_Mlt1_given_A1']['min_model'])}) "
              f"to {pct(hl['share_Mlt1_given_A1']['max'])} ({lab(hl['share_Mlt1_given_A1']['max_model'])}). The first "
              f"failing stage is grounding in {pct(hl['grounding_share']['min'])}-{pct(hl['grounding_share']['max'])} of "
              f"rows and the envelope in {pct(hl['envelope_share']['min'])}-{pct(hl['envelope_share']['max'])}.",
              f"3. **{'Asking for a certificate lowers accuracy' if c1['n_negative'] == c1['n_tests'] else 'Asking for a certificate changes accuracy'}.** A(cert) - A(ans) is negative in {c1['n_negative']} of "
              f"{c1['n_tests']} model x split cells ({pp(c1['delta']['min'])} to {pp(c1['delta']['max'])} pp); "
              f"{c1['n_ci_below_zero']} CIs exclude 0 and {c1['n_holm_sig_negative']} survive Holm (McNemar, seed {st['mcnemar_seed']}). "
              f"The legacy annotation prompt costs less: {pp(c2['delta']['min'])} to {pp(c2['delta']['max'])} pp, "
              f"{c2['n_holm_sig_negative']} of {c2['n_tests']} significant after Holm.",
              f"4. **Accuracy and certificate quality rank models differently.** On E1-test, answer-only A runs from "
              f"{f3(hl['A_ans_e1']['min'])} ({lab(hl['A_ans_e1']['min_model'])}) to {f3(hl['A_ans_e1']['max'])} "
              f"({lab(hl['A_ans_e1']['max_model'])}); the highest M is {f3(hl['M_e1']['max'])} "
              f"({lab(hl['M_e1']['max_model'])}){' while the most accurate model reaches M ' + f3(levels_m(doc, hl['A_ans_e1']['max_model'])) + ' (at most ' + f3(best_m(doc, hl['A_ans_e1']['max_model'])) + ' with the best of ' + str(len(doc['frames']['by_model'][hl['A_ans_e1']['max_model']]['e1_test']['all']['best_of_frames_over'])) + ' frames per row)' if hl['M_e1']['max_model'] != hl['A_ans_e1']['max_model'] else ''}. "
              f"Strict format F ranges {f3(hl['F']['min'])}-{f3(hl['F']['max'])} over both test splits.",
              f"5. **Qwen3.5 frame confound.** Frame obedience is `{q.get('obedience')}`"
              + (f": its boxes decisively fit source pixels on {fe_list(doc, 'decisive_px_source')} and rel1000 on "
                 f"{fe_list(doc, 'decisive_rel1000')}; elsewhere the evidence is weaker ({fe_rest(doc)}; section 9.1). "
                 if q.get('obedience') == 'mixed' else " (section 9.1). ")
              + f"Its G and M are reported as frame-confounded; choosing the best of {len(q.get('best_of_frames_over') or [])} "
              f"frames ({', '.join(q.get('best_of_frames_over') or [])}) per row caps its M at {f3(q.get('max_M_best_of_frames'))}.",
              ]
        if cpu_h:
            L += [f"6. **Evaluation layer (CPU).** Oracle round trip {intc(cpu_h['roundtrip_ok'])}/{intc(cpu_h['rows'])}; "
                  f"{intc(cpu_h['corruptions'])} corrupted responses of {cpu_h['types']} types with {cpu_h['unmet']} unmet "
                  f"expectations; go/no-go 1(b) {'passes on every numeric adapter' if cpu_h['go_no_go_1b_pass'] else 'FAILS on a numeric adapter'}; verifier "
                  f"median {cpu_h['verifier_median_ms']:.2f} ms, p99 {cpu_h['verifier_p99_ms']:.2f} ms per row; "
                  f"{'the' if cpu_h['reproductions_identical'] else 'NOT all'} {cpu_h['n_reproductions']} reproductions with other "
                  f"certificate-code versions {'give identical round-trip, corruption and go/no-go results' if cpu_h['reproductions_identical'] else 'agree'}."]
        if hl.get("identity_M_shift_range"):
            r = hl["identity_M_shift_range"]
            r3 = hl["identity_M_max_lowest"]
            tg = hl["identity_tags"]
            L += [f"7. **Identity threshold.** Moving the IoU threshold from {tg['frozen']} to {' or '.join(tg['others'])} "
                  f"shifts E1-dev M by {pp(r['min'])} to {pp(r['max'])} pp (section 10); even at IoU {tg['lowest']} the "
                  f"highest E1-dev M is {f3(r3['max'])} ({lab(r3['max_model'])})."]
        L += [""]
    if doc["pending"]:
        L += ["**Pending (left out of every table):** " + "; ".join(f"{p['model']} ({p['reason']})" for p in doc["pending"]), ""]
    else:
        L += ["**Pending:** none. All zero-shot models found under the root are complete and validated.", ""]

    # ---- inputs
    L += ["## 0. Inputs and completeness", ""]
    rows = []
    for i in doc["models"]:
        rows.append([i["label"], f"`{i['model_id']}`@`{(i['revision'] or '?')[:8]}`", i["family"], i["frame"],
                     f"{intc(i['rows'])} / {intc(i['expected_rows'])}", i["runs"],
                     "ok" if i["validation_ok"] else "NOT OK", i["endpoint_errors"], i["rejected"],
                     ", ".join(f"`{k}` x{v}" for k, v in i["scoring_code_sha256"].items())])
    L += table(["model", "HF id @ revision", "family", "frame", "rows scored / expected", "runs", "validation",
                "endpoint errors", "rejected", "scoring code sha256 (runs)"], rows, "llllrrlrrl")
    rs = doc["frames"]["rescore"]
    L += [f"Re-scoring check: all {intc(rs['n_rows'])} `cert` rows were scored again with the current scorer "
          f"(`{rs['scorer_version']}`, certificate code `{(rs['cert_code_sha256'] or '')[:12]}`) and "
          f"{intc(rs['n_mismatch_total'])} differ from the stored rows (A, A_legacy, G, G_id, G_shown, M, M_shown, F, joint, "
          "error class, valid schema), so the runs scored with older code shas above are unaffected. "
          f"The shown-frame re-reading (section 9) reproduces the scorer's own Qwen2.5-VL `G_shown` / `M_shown` on "
          f"{intc((rs.get('shown_replication') or {}).get('n_compared'))} rows with "
          f"{intc((rs.get('shown_replication') or {}).get('n_G_mismatch'))} / {intc((rs.get('shown_replication') or {}).get('n_M_mismatch'))} "
          f"differences; A and F differ between frames on {intc(rs.get('af_frame_dependent_total'))} rows. "
          f"Summary cross-check: {sum(c['n_checked'] for c in doc['crosscheck'].values()):,} pooled values compared with "
          f"each model's `summary.<split>.json`, {doc['crosscheck_problems_total']} differences.", ""]

    # ---- 1 headline
    L += ["## 1. Headline: accuracy and certificate quality", ""]
    for split in PRIMARY_SPLITS:
        L += [f"**{SPLIT_LABEL[split]}** ({intc(lv[models[0]][split]['cert']['n_instances'])} instances x "
              f"{len(lv[models[0]][split]['cert']['seeds'])} seeds = {intc(lv[models[0]][split]['cert']['n'])} rows per "
              "model and condition)", ""]
        rows = []
        for m in models:
            c = {e["model"]: e for e in doc["contrasts"]["cert_minus_ans"] if e["split"] == split}.get(m, {})
            cm = lv[m][split]["cert"]["mean"]
            rows.append([lab(m), f3(lv[m][split]["ans"]["mean"]["A"]), f3(lv[m][split]["ann"]["mean"]["A"]), f3(cm["A"]),
                         f"{pp(c.get('delta'))} {ci_pp(c.get('ci'))}", f3(cm["G"]), f3(cm["M"]), f3(cm["F"]),
                         f3(cm["valid"]), f3(cm["joint"]), f3(lv[m][split]["cert"]["two_by_two"]["share_Mlt1_given_A1"])])
        L += table(["model", "A ans", "A ann", "A cert", f"A cert - ans, pp [{cil} CI]", "G", "M", "F", "valid schema",
                    "joint", "P(M<1 given A=1)"], rows)
    L += ["G for `cert` is read in the declared frame. Qwen3.5 G and M are frame-confounded (section 9.1); compare "
          "Qwen3.5 with the other families on A only.", ""]

    # ---- 2 certificate cost
    L += ["## 2. Cost of asking for a certificate or an annotation (accuracy, paired within model)", ""]
    for fam, title in (("cert_minus_ans", "A(cert) - A(ans)"), ("ann_minus_ans", "A(ann) - A(ans)"),
                       ("cert_minus_ann", "A(cert) - A(ann)")):
        ents = doc["contrasts"][fam]
        if not ents:
            continue
        rows = [[lab(e["model"]), SPLIT_LABEL[e["split"]], f3(e["mean_x"]), f3(e["mean_y"]), pp(e["delta"]),
                 ci_pp(e["ci"]), f"{e['mcnemar']['c_x_only']} / {e['mcnemar']['b_y_only']}" if e["mcnemar"] else "-",
                 p_raw(e, floor), p_holm(e)] for e in ents]
        L += [f"**{title}** (Holm over {ents[0]['family_size']} tests)", ""]
        L += table(["model", "split", f"A {ents[0]['x']}", f"A {ents[0]['y']}", "delta pp",
                    f"{cil} CI pp", f"seed-{st['mcnemar_seed']} discordant (x only / y only)", "McNemar p", "Holm p"], rows, "llrrrrrrr")

    # ---- 3 certificate quality with CIs
    L += [f"## 3. Certificate quality with {cil} CIs (`cert` condition)", ""]
    rows = []
    for split in PRIMARY_SPLITS:
        for m in models:
            c = lv[m][split]["cert"]
            ci = c["ci"]
            rows.append([lab(m), SPLIT_LABEL[split], f"{f3(c['mean']['A'])} {ci3(ci['A']['ci'])}",
                         f"{f3(c['mean']['G'])} {ci3(ci['G']['ci'])}", f"{f3(c['mean']['M'])} {ci3(ci['M']['ci'])}",
                         f"{f3(c['mean']['F'])} {ci3(ci['F']['ci'])}", f"{f3(c['mean']['valid'])} {ci3(ci['valid']['ci'])}",
                         f"{f3(c['mean']['joint'])} {ci3(ci['joint']['ci'])}", f3(c["mean"]["envelope"])])
    L += table(["model", "split", "A", "G", "M", "F", "valid schema", "joint", "envelope failure"], rows, "llrrrrrrr")
    rows = []
    for split in PRIMARY_SPLITS:
        for m in models:
            ps = lv[m][split]["cert"]["per_seed"]
            rows.append([lab(m), SPLIT_LABEL[split]] + [" / ".join(f3(ps[s][k]) for s in sorted(ps)) for k in ("A", "G", "M", "F", "joint")])
    L += [f"Per decoding seed ({' / '.join(str(x) for x in kc['decoding_seeds'])}):", ""]
    L += table(["model", "split", "A", "G", "M", "F", "joint"], rows, "llrrrrr")

    # ---- 4 generation facts
    L += ["## 4. Completion, truncation, tokens and verifier time", ""]
    rows = []
    for split in PRIMARY_SPLITS:
        for m in models:
            for cond in CONDS:
                c = lv[m][split][cond]
                rows.append([lab(m), SPLIT_LABEL[split], cond, f3(c["mean"]["complete"]), f3(c["mean"]["trunc"]),
                             f3(c["mean"]["w2048"]), f"{c['tokens']['mean']:.0f}", f"{c['tokens']['median']:.0f}",
                             f"{c['tokens']['p99']:.0f}", intc(c["tokens"]["total"]),
                             f"{c['verifier_ms']['median']:.2f} / {c['verifier_ms']['p99']:.2f} / {c['verifier_ms']['max']:.1f}",
                             c["verifier_ms"]["slow_rows"]])
    L += table(["model", "split", "cond", "completion (stop=answer)", "truncated", f"within {kc['within_token_cap']['value']:,} tokens", "tokens mean",
                "median", "p99", "total generated", "verifier ms median / p99 / max",
                f"rows > {num(kc['slow_row_ms']['value'])} ms"], rows, "lllrrrrrrrrr")
    L += ["Verifier time is the scorer's wall time per row on the CPU scoring host (all signals; `ans` scores A "
          "only). Scores never depend on it.", ""]

    # ---- 5 2x2
    L += ["## 5. Answer x certificate: A x (M == 1) (`cert` condition, pooled rows)", ""]
    rows = []
    for split in PRIMARY_SPLITS:
        for m in models:
            t = lv[m][split]["cert"]["two_by_two"]
            rows.append([lab(m), SPLIT_LABEL[split], intc(t["A1_M1"]), intc(t["A1_Mlt1"]),
                         f"{intc(t['A1_Mpartial'])} / {intc(t['A1_M0'])}", intc(t["A0_M1"]), intc(t["A0_Mlt1"]),
                         f"{intc(t['A0_Mpartial'])} / {intc(t['A0_M0'])}", pct(t["share_Mlt1_given_A1"]),
                         pct(t["share_Mpartial_given_A0"]), pct(t["share_A0_given_M1"])])
    L += table(["model", "split", "A=1, M=1", "A=1, M<1", "of which 0<M<1 / M=0", "A=0, M=1", "A=0, M<1",
                "of which 0<M<1 / M=0", "M<1 among A=1", "0<M<1 among A=0", "A=0 among M=1"], rows, "llrrrrrrrrr")
    L += ["`A=1, M<1` is a correct answer with a certificate that does not verify; `A=0, 0<M<1` is a wrong answer "
          "whose certificate is partly correct; `A=0, M=1` (a fully verified certificate with a wrong final answer) "
          "is the `answer_inconsistent` error class.", ""]

    # ---- 6 error classes
    L += ["## 6. Error classes (`cert`, first failing stage; DESIGN 5)", ""]
    rows = []
    for split in PRIMARY_SPLITS:
        for m in models:
            c = lv[m][split]["cert"]
            ec = c["error_classes"]
            n = c["n"]
            rows.append([lab(m), SPLIT_LABEL[split]] + [f"{intc(ec[k])} ({pct(ec[k] / n)})" for k in ERROR_CLASSES])
    L += table(["model", "split"] + list(ERROR_CLASSES), rows, "llrrrrrr")
    rows = []
    for m in models:
        c = lv[m]["e1_test"]["cert"]
        env = ", ".join(f"{k} {v}" for k, v in sorted(c["envelope_errors"].items(), key=lambda kv: -kv[1])) or "-"
        fr = ", ".join(f"{k} {v}" for k, v in list(c["format_reasons"].items())[:6]) or "-"
        rows.append([lab(m), intc(c["F0_rows"]), env, fr])
    L += ["Envelope errors and the most frequent strict-format (F) failure reasons, E1-test `cert` (a row can carry "
          "several format reasons):", ""]
    L += table(["model", "rows with F=0", "envelope errors", "top format reasons"], rows, "lrll")

    # ---- 7 per adapter
    L += ["## 7. Per adapter", ""]
    for split in PRIMARY_SPLITS:
        rows = []
        for m in models:
            for a in ADAPTERS:
                ba = doc["by_adapter"][m][split]
                if a not in ba.get("cert", {}):
                    continue
                c = ba["cert"][a]["mean"]
                rows.append([lab(m), ADAPTER_LABEL[a], f3(ba["ans"][a]["mean"]["A"]), f3(ba["ann"][a]["mean"]["A"]),
                             f3(c["A"]), f3(c["G"]), f3(c["M"]), f3(c["F"]), f3(c["valid"]), f3(c["joint"]),
                             f3(c["complete"]), f"{ba['cert'][a]['tokens']['mean']:.0f}"])
        L += [f"**{SPLIT_LABEL[split]}**", ""]
        L += table(["model", "adapter", "A ans", "A ann", "A cert", "G", "M", "F", "valid", "joint", "completion",
                    "tokens mean (cert)"], rows, "llrrrrrrrrrr")

    # ---- 8 cross model
    L += ["## 8. Cross-model contrasts within a family (paired by instance, descriptive)", ""]
    rows = []
    for e in doc["cross_model"]:
        rows.append([f"{lab(e['x'])} - {lab(e['y'])}", SPLIT_LABEL[e["split"]], e["metric"], f3(e["mean_x"]), f3(e["mean_y"]),
                     pp(e["delta"]), ci_pp(e["ci"]), e["test"].split("(")[0], p_raw(e, floor), p_holm(e),
                     "yes" if e.get("frame_confounded") else ""])
    if rows:
        L += [f"Holm over all {doc['cross_model'][0]['family_size']} tests of this table. `_ans` = answer-only "
              "prompt, `_cert` = certificate prompt.", ""]
        L += table(["x - y", "split", "metric", "x", "y", "delta pp", f"{cil} CI pp", "test", "p", "Holm p", "frame-confounded"],
                   rows, "lllrrrrlrrl")
        fc = [e for e in doc["cross_model"] if e.get("frame_confounded")]
        if fc:
            fe = doc["frames"]["frame_evidence"]
            dec = fe.get("decisive_px_source", []) + fe.get("decisive_rel1000", [])
            ex = "; ".join(
                f"{ADAPTER_LABEL[a]}, {SPLIT_LABEL[c['split']]}: " + ", ".join(
                    f"{lab(c2['model'])} {pp(c2['dG'])}" for c2 in fe["per_adapter"][a]["cells"] if c2["split"] == c["split"])
                for a in dec for c in fe["per_adapter"][a]["cells"][:1])
            L += ["Rows marked frame-confounded compare G, M or joint of models that declare rel1000 (Qwen3.5). "
                  "The two Qwen3.5 sizes share the declared frame but use the frames differently (G read as source pixels "
                  f"minus G read as rel1000, pp: {ex}), so a size difference in G, M or joint mixes grounding skill with "
                  "frame use. A and F do not depend on the frame.", ""]

    # ---- 9 frames
    L += ["## 9. Coordinate frames", ""]
    rs9 = doc["frames"]["rescore"]
    L += ["Each `cert` response was re-scored with the other public frame (px_source <-> rel1000), and G and M also "
          "in the frame the server showed the model (`px_shown`, from the reported shown image size). For Qwen2.5-VL "
          "the shown-frame G and M are the scorer's own diagnostics (`G_shown`, `M_shown`); for Qwen3.5 the re-scoring "
          "computes them with the same scorer path, which reproduces the Qwen2.5-VL values "
          f"({intc((rs9.get('shown_replication') or {}).get('n_compared'))} rows, "
          f"{intc((rs9.get('shown_replication') or {}).get('n_M_mismatch'))} M differences). Shown-frame joint = A = F = 1 "
          f"and M_shown = 1 (A and F do not depend on the frame: {intc(rs9.get('af_frame_dependent_total'))} rows differ). "
          "`best of frames` takes, per row, the highest M (or joint) over the declared, the other and the shown frame "
          f"(shown-frame M could not be computed on {intc(rs9.get('m_px_shown_errors_total'))} rows; those use the two "
          "contract frames): an optimistic bound on what a frame fix could recover, not an estimate. Coordinate ranges use every "
          "number pair of the annotation the scorer reads.", ""]
    rows = []
    for split in PRIMARY_SPLITS:
        for m in models:
            b = doc["frames"]["by_model"][m].get(split, {}).get("all")
            if not b:
                continue
            d, a = b["declared"], b["alt"]
            rows.append([lab(m), SPLIT_LABEL[split], d, f3(b["G"][d]), f3(b["G"][a]), f3(b["G"]["px_shown"]),
                         f3(b["M"][d]), f3(b["M"][a]), f3(b["M"].get("px_shown")), f3(b["M"]["best_of_frames"]),
                         f3(b["joint"][d]), f3(b["joint"]["best_of_frames"]),
                         f"{pct(b['frac_le_1000'])} / {pct(b['frac_in_source'])}",
                         " / ".join(str(b["best_frame_by_G"][k]) for k in (d, a, "tie", "none"))])
    L += table(["model", "split", "declared", "G declared", "G other", "G shown", "M declared", "M other", "M shown",
                "M best of frames", "joint declared", "joint best", "coords <=1000 / inside source image",
                "rows where G is higher as declared / other frame / tie / zero in both"], rows, "lllrrrrrrrrrrl")

    L += ["### 9.1 Qwen3.5 frame-confound note", ""]
    notes = doc["frames"]["note"]
    if not notes:
        L += ["No Qwen3.5 model is complete.", ""]
    for n in notes:
        if n["split"] != "e1_test":
            continue
        per = n["per_adapter"]
        lever = per.get("weighted_sum.lever_side_torque", {})
        rpg = per.get("path.rpg_movement_cost", {})
        L += [f"- **{lab(n['model'])}, E1-test.** Its coordinates lie in 0-1000 on {pct(n['frac_le_1000'])} of rows "
              f"with coordinates, but {pct(n['frac_in_source'])} also lie inside the source image, so range alone "
              f"cannot tell the frames apart. Re-scored per adapter: on the lever "
              f"({lever.get('source_size_mode')} images) G is {f3(lever.get('G_px_source'))} read as source pixels and "
              f"{f3(lever.get('G_rel1000'))} read as rel1000 (M {f3(lever.get('M_px_source'))} vs "
              f"{f3(lever.get('M_rel1000'))}; {pct(lever.get('frac_in_source'))} of rows with coordinates fit inside the "
              f"image); on rpg G is {f3(rpg.get('G_rel1000'))} read as rel1000 and {f3(rpg.get('G_px_source'))} read as "
              f"pixels. "
              f"Overall M is {f3(n['M_rel1000'])} as declared, {f3(n['M_px_source'])} as source pixels, "
              f"{f3(n.get('M_px_shown'))} as shown pixels and at most {f3(n['M_best'])} with the best of these "
              f"{len(n['best_of_frames_over'])} frames chosen per row (joint {f3(n['joint_rel1000'])} vs at most "
              f"{f3(n['joint_best'])}); A is {f3(n['A_cert'])} (cert) and {f3(n['A_ans'])} (answer-only)."]
    e2 = [n for n in notes if n["split"] == "e2_test"]
    if e2:
        L += ["- E2-test: " + "; ".join(
            f"{lab(n['model'])} lever G {f3(n['per_adapter'].get('weighted_sum.lever_side_torque', {}).get('G_px_source'))} "
            f"(px) vs {f3(n['per_adapter'].get('weighted_sum.lever_side_torque', {}).get('G_rel1000'))} (rel1000), "
            f"rpg {f3(n['per_adapter'].get('path.rpg_movement_cost', {}).get('G_rel1000'))} (rel1000) vs "
            f"{f3(n['per_adapter'].get('path.rpg_movement_cost', {}).get('G_px_source'))} (px), M best-of-frames "
            f"{f3(n['M_best'])} vs declared {f3(n['M_rel1000'])}" for n in e2) + "."]
    fe = doc["frames"].get("frame_evidence") or {}
    if fe.get("per_adapter"):
        parts = []
        for a, e in fe["per_adapter"].items():
            verdict = {"decisive": f"decisive, {e['direction']}", "leans": f"leans {e['direction']}",
                       "unclear": "unclear"}[e["strength"]]
            against = "; ".join(f"M points the other way for {lab(c['model'])} {SPLIT_LABEL[c['split']]} "
                                f"({pp(c['dM'], 2)} pp {ci_pp(c['ciM'], 2)})" for c in e["M_against_G"])
            gr, mr, sg = e["G_delta_range"], e["M_delta_range"], e["scale_gap_median_range"]
            parts.append(
                f"{ADAPTER_LABEL[a]}: **{verdict}** (G {pp(gr[0])} to {pp(gr[1])} pp; G CI toward pixels / rel1000 in "
                f"{e['n_G_ci_px']} / {e['n_G_ci_rel']} of {e['n_cells']} cells, M CI in {e['n_M_ci_px']} / {e['n_M_ci_rel']}; "
                f"M {pp(mr[0], 2)} to {pp(mr[1], 2)} pp; rows where G is higher as pixels / rel1000 "
                f"{intc(e['rows_G_higher_px'])} / {intc(e['rows_G_higher_rel'])}; median frame scale gap "
                f"{pct(sg[0], 0)}{'' if abs(sg[1] - sg[0]) < 0.005 else '-' + pct(sg[1], 0)}"
                + (f"; {against}" if against else "") + ")")
        L += [f"- **Which frame fits, per adapter** (pixels minus rel1000 in {len(fe['models']) * len(fe['splits'])} cells = "
              f"{len(fe['models'])} models x {len(fe['splits'])} test splits; rule: {fe['rule']}; the frame scale gap is max over axes of "
              "|source size / 1000 - 1|, so a small gap means the two readings nearly coincide): " + "; ".join(parts) + "."]
    if notes:
        vd = doc["frames"]["qwen35_verdict"]
        mx = doc["frames"]["all_models_max"]
        reading = {
            "mixed": "Qwen3.5 does not follow the rel1000 instruction consistently: the frame that fits changes with "
                     f"the task (decisively source pixels on {fe_list(doc, 'decisive_px_source')}, rel1000 on "
                     f"{fe_list(doc, 'decisive_rel1000')}), so its declared-frame G and M mix a frame error with a "
                     "grounding error. Where the source image is close to 1000 x 1000 the two readings nearly coincide, "
                     "so small G differences there say little about the frame the model used.",
            "rel1000": "Where the frame evidence is decisive, Qwen3.5 coordinates read as rel1000, as instructed.",
            "px_source": "Where the frame evidence is decisive, Qwen3.5 coordinates read as source pixels, against the instruction.",
        }.get(vd["obedience"], "The frame evidence is inconclusive.")
        hints = []
        for m in models:
            for tag, h in doc["frames"]["frame_diag_hints"].get(m, {}).items():
                if info[m]["family"] == "qwen3.5":
                    hints.append(f"{lab(m)} {tag.replace('_', ' ')} (n={h['n']}): \"{h['hint']}\" "
                                 f"(G rel1000 {f3(h['G_rel1000'])}, px {f3(h['G_px_source'])})")
        L += [f"- **Reading.** {reading} Per DESIGN 10 and UNRESOLVED 2, Qwen3.5 G and M are therefore reported as "
              "frame-confounded and only A is compared across families (and, within Qwen3.5, the 9B - 4B G/M/joint "
              "contrasts of section 8 are frame-confounded too, since the sizes use the frames differently). Choosing "
              f"the best of the {len(vd.get('best_of_frames_over') or [])} frames ({', '.join(vd.get('best_of_frames_over') or [])}) per row (an "
              f"optimistic bound) lifts Qwen3.5 M to at most {f3(vd['max_M_best_of_frames'])} and joint to at most "
              f"{f3(vd['max_joint_best_of_frames'])}; across all {len(models)} models the highest declared-frame M is "
              f"{f3(mx['M'][0])} ({lab(mx['M'][1])}, {SPLIT_LABEL[mx['M'][2]]}) and the highest joint "
              f"{f3(mx['joint'][0])} ({lab(mx['joint'][1])}, {SPLIT_LABEL[mx['joint'][2]]}), so the low zero-shot "
              "certificate quality does not hinge on the frame."]
        ann_q35 = [(m, sp, a) for m in models if info[m]["family"] == "qwen3.5"
                   for sp, a in doc["frames"]["ann"].get(m, {}).items() if sp in PRIMARY_SPLITS]
        if ann_q35:
            L += ["- Under the legacy `ann` prompt, which names no frame, Qwen3.5 boxes read better as source pixels "
                  "than in its assumed native rel1000 frame: " + "; ".join(
                      f"{lab(m)} {SPLIT_LABEL[sp]} G {f3(a['G_px_source'])} (px) vs {f3(a['G_native'])} (rel1000)"
                      for m, sp, a in ann_q35) + "." + (" So pixel output is also its default when no frame is given."
                                                         if all((a["G_px_source"] or 0) > (a["G_native"] or 0) for _, _, a in ann_q35) else "")]
        if hints:
            L += ["- The automatic frame-obedience hint (`zs_report.py frame-diag`) pools all adapters and only "
                  "compares pooled G, so it did not flag this: " + "; ".join(hints) + "."]
        L += [""]
    rows = []
    for c in doc["frames"]["contrasts"]:
        if c["metric"] not in ("G", "M"):
            continue
        if info[c["model"]]["family"] != "qwen3.5" and c["x"] != "px_shown":
            continue
        rows.append([lab(c["model"]), SPLIT_LABEL[c["split"]], ADAPTER_LABEL.get(c["scope"], c["scope"]), c["metric"],
                     f"{c['x']} - {c['y']}", f3(c["mean_x"]), f3(c["mean_y"]), pp(c["delta"]), ci_pp(c["ci"]), pv(c["perm_p"], floor)])
    L += ["Paired frame contrasts (Qwen3.5: pixels minus rel1000; Qwen2.5-VL family: shown frame minus source "
          "pixels), cluster bootstrap CI and sign-flip p, unadjusted (no Holm):", ""]
    L += table(["model", "split", "scope", "metric", "x - y", "x", "y", "delta pp", f"{cil} CI pp", "sign-flip p"], rows, "lllllrrrrr")
    rows = []
    for m in models:
        for split in PRIMARY_SPLITS:
            a = doc["frames"]["ann"][m].get(split)
            if a:
                c = a["px_minus_native"]
                rows.append([lab(m), SPLIT_LABEL[split], "/".join(a["native_frame"]), f3(a["G_native"]), f3(a["G_px_source"]),
                             pp(c["delta"]), ci_pp(c["ci"]), pv(c["perm_p"], floor)])
    L += ["Legacy `ann` prompt (states no frame): G in the family-native frame (primary, DESIGN 5) and in source pixels "
          "(sign-flip p unadjusted):", ""]
    L += table(["model", "split", "native frame", "G native", "G px_source", "px - native pp", f"{cil} CI pp", "sign-flip p"], rows, "lllrrrrr")

    # per-adapter frame table for Qwen3.5
    rows = []
    bof: set = set()
    for m in models:
        if info[m]["family"] != "qwen3.5":
            continue
        for split in PRIMARY_SPLITS:
            for a in ADAPTERS:
                b = doc["frames"]["by_model"][m].get(split, {}).get(a)
                if not b:
                    continue
                bof.update(b.get("best_of_frames_over") or [])
                rows.append([lab(m), SPLIT_LABEL[split], ADAPTER_LABEL[a], f3(b["G"]["rel1000"]), f3(b["G"]["px_source"]),
                             f3(b["G"]["px_shown"]), f3(b["M"]["rel1000"]), f3(b["M"]["px_source"]), f3(b["M"].get("px_shown")),
                             f3(b["M"]["best_of_frames"]), pct((b.get("scale_gap") or {}).get("median"), 0),
                             pct(b["frac_le_1000"]), pct(b["frac_in_source"]),
                             " / ".join(str(b["best_frame_by_G"][k]) for k in ("rel1000", "px_source", "tie", "none")),
                             ", ".join(f"{k} ({v})" for k, v in list(b["source_sizes_top"].items())[:2])
                             + (f", +{b['n_source_sizes'] - 2} other sizes" if b["n_source_sizes"] > 2 else "")])
    if rows:
        L += ["Qwen3.5 per adapter:", ""]
        L += table(["model", "split", "adapter", "G rel1000", "G px_source", "G px_shown", "M rel1000", "M px_source",
                    "M px_shown", f"M best of {len(bof)} frames", "median frame scale gap", "coords <=1000", "inside source",
                    "rows where G is higher as rel1000 / px / tie / zero", "source image sizes (rows)"], rows, "lllrrrrrrrrrrll")

    # ---- 10 identity
    L += ["## 10. Identity-rule sensitivity (E1-dev `cert`, DESIGN 2.3)", ""]
    rows = []
    for m in models:
        e = doc["identity_sensitivity"].get(m)
        if not e:
            continue
        rows.append([lab(m), intc(e["n"])] + [f"{f3(e[t]['M'])} / {f3(e[t]['G_id'])} / {f3(e[t]['joint'])}" for t in e["tags"]]
                    + [" / ".join(intc(e[t]["rows_M_changed_vs_frozen"]) for t in e["tags"] if t != e["frozen"])])
    e0 = next(iter(doc["identity_sensitivity"].values()), None)
    if e0:
        idc = doc["constants"]["identity_frozen"]
        others = [t for t in e0["tags"] if t != e0["frozen"]]
        L += [f"M / G_id / joint with the identity IoU threshold at {', '.join(e0['tags'])} ({e0['frozen']} frozen, "
              f"DESIGN 2.3), containment threshold {num(doc['frames']['rescore'].get('containment_threshold'))} throughout "
              f"(frozen value {num(idc['containment_threshold'])}), declared frame:", ""]
        L += table(["model", "rows"] + [f"IoU {t}" + (" (frozen)" if t == e0["frozen"] else "") for t in e0["tags"]]
                   + [f"rows whose M changes ({' / '.join(others)})"], rows, "lr" + "r" * (len(e0["tags"]) + 1))

    # ---- 11 CPU digest
    cpu = doc["cpu"]
    L += ["## 11. Section 5 CPU digest (oracle round trip, corruption suites, verifier time)", ""]
    rows = []
    for r in cpu["runs"]:
        if r.get("missing"):
            rows.append([r["name"], f"`{r['dir']}`", "missing", "", "", "", "", "", ""])
            continue
        rows.append([r["name"] + (" (reference)" if r["name"] == cpu.get("reference") else ""), f"`{r['dir']}`",
                     f"`{(r['cert_code_sha256'] or '')[:12]}`", r["scorer_version"], r["workers"],
                     f"{intc(r['roundtrip_ok_total'])} / {intc(r['rows'])}",
                     f"{intc(r['corruptions_scored'])} ({r['corruption_types']} types)", intc(r["corruption_expectations_unmet"]),
                     f"`{(r['outputs_sha256'].get('corruptions.jsonl') or '')[:12]}`"])
    L += table(["run", "dir", "cert code", "scorer", "workers", "round trip ok / rows", "corrupted responses",
                "unmet expectations", "corruptions.jsonl sha256"], rows, "llllrrrrl")
    for name, ag in cpu.get("agreement_with_reference", {}).items():
        L += [f"- `{name}` vs reference: round trip equal {ag['roundtrip_equal']}, corruption aggregates equal "
              f"{ag['corruptions_equal']}, go/no-go equal {ag['go_no_go_equal']}, corruptions.jsonl byte-identical "
              f"{ag['corruptions_jsonl_sha_equal']}; summary keys that differ: {', '.join(ag['differing_summary_keys']) or 'none'}."]
    L += [""]
    rows = []
    for a, p in cpu.get("per_adapter", {}).items():
        rt = p["roundtrip"]
        paths = f"{intc(rt.get('optimal_paths_checked'))} / {intc(rt.get('optimal_paths_failed'))}" if "optimal_paths_checked" in rt else "n/a"
        rows.append([ADAPTER_LABEL[a], intc(rt.get("rows")), intc(rt.get("oracle_all_one")), intc(rt.get("oracle_all_resolved")),
                     intc(rt.get("sft_AMF_one")), f"{f3(rt.get('sft_G_mean'))} ({f3(rt.get('sft_G_min'))})", intc(rt.get("schema_ok")),
                     paths, intc(rt.get("M_shown_lt1")), f3(rt.get("G_shown_mean"))])
    L += ["**(a) Oracle round trip** (reference run):", ""]
    L += table(["adapter", "rows", "oracle A=G=M=F=1", "all boxes resolved", "SFT target A=M=F=1", "SFT G mean (min)",
                "schema ok", "optimal paths checked / failed", "perfect answer read in shown frame: M<1 rows",
                "shown-frame G mean"], rows, "lrrrrrrrrr")
    rows = []
    for a, p in cpu.get("per_adapter", {}).items():
        g = p["go_no_go_1b"]
        rows.append([ADAPTER_LABEL[a], p["corruption_types"], intc(p["corrupted_responses"]), f3(p["min_expect_ok_rate"]),
                     f3(p["min_intended_fail_rate"]),
                     (f"{g['count_content_all_rows']} content / {g['count_all_rows']} all" if g else "n/a (control)"),
                     ("yes" if g and g.get("pass") else ("no" if g else "-"))])
    L += ["**(b) Corruption suite** (`expect` = every declared check/signal expectation held; `intended` = every "
          "check meant to fail failed; go/no-go 1(b) = corruption types that keep G "
          f"(|dG| < {num(kc['go_no_go_keep_G_tol']['value'])}) and lower M on every applicable row, >= "
          f"{num(kc['go_no_go_min_content_types']['value'])} content types required per numeric adapter):", ""]
    L += table(["adapter", "types", "corrupted responses", "min expect rate", "min intended rate",
                "keep G, lower M (types)", "go/no-go 1(b) pass"], rows, "lrrrrrl")
    rows = []
    for a, p in cpu.get("per_adapter", {}).items():
        t = p["timing"]
        rows.append([ADAPTER_LABEL[a], f"{t['verifier_ms']['median']:.2f}", f"{t['verifier_ms']['p99']:.2f}",
                     f"{t['verifier_ms']['max']:.2f}", f"{t['cpu_ms']['median']:.2f}", f"{t['cpu_ms']['p99']:.2f}",
                     f"{t['verifier_ms_shown']['median']:.2f}", f"{t['verifier_ms_shown']['p99']:.2f}"])
    ta = cpu.get("timing_all") or {}
    if ta:
        rows.append(["all", f"{ta['verifier_ms']['median']:.2f}", f"{ta['verifier_ms']['p99']:.2f}", f"{ta['verifier_ms']['max']:.2f}",
                     f"{ta['cpu_ms']['median']:.2f}", f"{ta['cpu_ms']['p99']:.2f}", "", ""])
    L += [f"**(c) Verifier time per row** on the oracle object (ms, reference run; DESIGN budget "
          f"{num(kc['verifier_budget_ms']['value'])} ms):", ""]
    L += table(["adapter", "wall median", "wall p99", "wall max", "CPU median", "CPU p99", "+ shown frame median",
                "+ shown frame p99"], rows, "lrrrrrrr")

    # ---- appendix
    L += ["## Appendix A. E1-dev (calibration split) levels", ""]
    rows = []
    for m in models:
        c = lv[m]["e1_dev"]
        cm = c["cert"]["mean"]
        rows.append([lab(m), f3(c["ans"]["mean"]["A"]), f3(c["ann"]["mean"]["A"]), f3(cm["A"]), f3(cm["G"]), f3(cm["M"]),
                     f3(cm["F"]), f3(cm["valid"]), f3(cm["joint"]), f3(cm["complete"]), f"{c['cert']['tokens']['mean']:.0f}"])
    L += table(["model", "A ans", "A ann", "A cert", "G", "M", "F", "valid", "joint", "completion", "tokens mean"], rows)
    L += ["## Appendix B. Corruption suite, every type (reference run)", ""]
    for a, p in cpu.get("per_adapter", {}).items():
        rows = [[k, intc(v["rows"]), f3(v["A"]), f3(v["G"]), f3(v["M"]), f3(v["F"]), f3(v["expect_ok_rate"]),
                 f3(v["intended_fail_rate"]), f3(v["keeps_g_lowers_m_rate"])] for k, v in sorted(p["corruptions"].items())]
        L += [f"*{ADAPTER_LABEL[a]}*", ""]
        L += table(["corruption", "rows", "A", "G", "M", "F", "expect", "intended", "G kept, M down"], rows)
    return "\n".join(L).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zs-root", required=True)
    ap.add_argument("--rescore", required=True)
    ap.add_argument("--cpu", action="append", default=[], help="NAME=DIR of a section5_cpu.py output (first = reference)")
    ap.add_argument("--models", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--trace-tool-src", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mcnemar-seed", type=int, default=42)
    ap.add_argument("--design", default=None, help="DESIGN.md (read for the verifier time budget only)")
    args = ap.parse_args(argv)
    doc = build(args)
    Path(args.out_json).write_text(json.dumps(doc, indent=1, sort_keys=False, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)) + "\n")
    doc2 = json.loads(Path(args.out_json).read_text())  # render from the written JSON only
    Path(args.out_md).write_text(markdown(doc2))
    print(f"wrote {args.out_json} and {args.out_md}; crosscheck problems {doc['crosscheck_problems_total']}, "
          f"rescore mismatches {doc['frames']['rescore']['n_mismatch_total']}")
    sr = doc["frames"]["rescore"].get("shown_replication") or {}
    bad = (doc["crosscheck_problems_total"] or doc["frames"]["rescore"]["n_mismatch_total"]
           or not doc["frames"]["rescore"]["rows_cover_all_cert_rows"]
           or sr.get("n_G_mismatch") or sr.get("n_M_mismatch"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
