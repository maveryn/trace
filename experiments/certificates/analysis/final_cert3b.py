"""Final 3B in-domain certificate analysis helpers (results/final/cert_3b; DESIGN 7.4, 8; RQ1/RQ3).

The pre-registered families come from ``compare_cert.py`` itself (run separately, see the output
dir's ``run.sh``). This module adds what compare_cert does not compute, reusing its loading,
pairing and statistics code unchanged (imported, never edited):

    python final_cert3b.py extras   --model P@1=<eval/ckpt/root> ... --pairs P:G,... --out-json X.json
    python final_cert3b.py traceval --ckpt-root <eval/ckpt> --model S=<root> ... --pairs ... --out-json X.json
    python final_cert3b.py training --rl-root <rl> --out-json X.json
    python final_cert3b.py inputs   --runs-root <runs> --need <root> --need-tv <root> --receipt 3b/P/s1 ... --out-json X.json
    python final_cert3b.py report   --dir <out dir> --out-md REPORT.md

``extras``  Per-row derived metrics of the certificate condition (``cert_v1``) read from every
            ``cert_scores.jsonl`` under each model root: the first-failing-stage error class as
            one-hot rates (envelope, grounding, operand, computation, answer_inconsistent, correct;
            ``classify_error`` of the scorer), the A x M cells (right answer with an untrue
            certificate, ...), stage truth scores (mean of the applicable family-level truth checks
            mapped to grounding / operand / computation by each adapter's ``family_stages``),
            local consistency (arithmetic and answer-vs-certificate checks; rows with a parsed
            certificate only), every family's truth score, truncation and schema validity; and the
            answer-only prompt (``ans_v1``) A and tokens on the same rows. Paired comparisons use
            ``compare_cert.run_comparisons`` (decoding-seed means per row, cluster bootstrap
            stratified by adapter, McNemar on seed 42 for 0/1 metrics, cluster sign-flip otherwise,
            two-level bootstrap and the sign rule with several training seeds). All of it is
            descriptive: the only Holm-adjusted families are compare_cert's.
``traceval`` TRACE released validation (``ans_v1`` seeds 42-44, ``no_tool`` protocol seed 42) of every
            ``<ckpt-root>/*/summary.trace_val.json`` whose name matches ``--include`` (judge_pending
            rows count as wrong in the official lower bound; the upper bound counts them right),
            plus paired per-row comparisons of the named models (cluster = task, stratum =
            domain; a model is paired only when its ``validation.trace_val.json`` is ok). Besides
            lower-minus-lower and upper-minus-upper, every pair gets the judge-outcome range of the
            difference: X lower - Y upper (least favourable to X) and X upper - Y lower.
``training`` RL run receipts: hardware, attempts, wall hours, step seconds and mean rollout length.
``inputs``  Status of every input root and RL receipt the analysis reads (exit 1 if any is not ok).
``report``  Markdown report built only from the JSON documents in the output dir.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_cert as cc  # noqa: E402  (same directory; read-only reuse)

VERSION = "final_cert3b-1.1.0"

ERROR_CLASSES = ("envelope", "grounding", "operand", "computation", "answer_inconsistent", "correct")
STAGES = ("grounding", "operand", "computation")
ARITH_LOCAL = (".count.local", ".sum.local", ".mean.local", ".product.local", ".total.local")

# name -> (binary, description); families ("fam:<id>") are added from the registry
BASE_CERT_METRICS: Dict[str, Tuple[bool, str]] = {
    "A": (True, "answer correct (cert prompt)"),
    "G": (False, "grounding score"),
    "M": (False, "semantic certificate score"),
    "F": (True, "strict format"),
    "joint": (True, "answer and certificate both correct"),
    "tokens": (False, "completion tokens"),
    "truncated": (True, "response truncated"),
    "valid_schema": (True, "certificate schema valid"),
    **{f"ec_{c}": (True, f"error class '{c}' (first failing stage)") for c in ERROR_CLASSES},
    "a1_m1": (True, "A = 1 and M = 1"),
    "a1_mlt1": (True, "A = 1 and M < 1 (right answer, untrue certificate)"),
    "a0_m1": (True, "A = 0 and M = 1"),
    "a0_mlt1": (True, "A = 0 and M < 1"),
    "g09_mlt1": (True, "G >= 0.9 and M < 1 (good boxes, untrue semantics; go/no-go 1a's row type)"),
    **{f"st_{s}": (False, f"mean truth score of the {s}-stage families") for s in STAGES},
    "loc_arith": (False, "local arithmetic consistency on the model's own operands (parsed certificates only)"),
    "loc_answer": (False, "answer consistent with the certificate (parsed certificates only)"),
}
ANS_METRICS: Dict[str, Tuple[bool, str]] = {
    "A_ans": (True, "answer correct under the answer-only prompt (ans_v1)"),
    "tokens_ans": (False, "completion tokens under the answer-only prompt"),
    "truncated_ans": (True, "truncated under the answer-only prompt"),
}
TV_METRICS: Dict[str, Tuple[bool, str]] = {
    "tv_acc_lo": (True, "official accuracy, judge_pending counted wrong (lower bound)"),
    "tv_acc_hi": (True, "official accuracy, judge_pending counted right (upper bound)"),
    "tv_strict": (True, "strict accuracy"),
    "tv_tokens": (False, "completion tokens"),
}
# judge-outcome range of a paired difference X - Y (pending rows resolved against / in favour of X)
RANGE_METRICS: Dict[str, Tuple[bool, str]] = {
    "tv_range_lo": (True, "X lower bound - Y upper bound: X's pending rows wrong, Y's right (least favourable to X)"),
    "tv_range_hi": (True, "X upper bound - Y lower bound: X's pending rows right, Y's wrong (most favourable to X)"),
}
Y_ROLE = "~y"  # suffix of the Y-role copy of an arm in the range dataset
FAILURE_CLASSES = tuple(c for c in ERROR_CLASSES if c != "correct")


def _finite(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _nan(v: Optional[float]) -> float:
    return float("nan") if v is None else float(v)


def _register(metrics: Mapping[str, Tuple[bool, str]]) -> None:
    """Make compare_cert's statistics treat our derived metrics like its own (runtime only)."""
    for name, (binary, _) in metrics.items():
        cc.METRICS.setdefault(name, (name, binary))


def load_stage_map(trace_src: Optional[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """family id -> stage, and adapter id -> its family ids (from the scorer's registry)."""
    src = trace_src or str(HERE.parents[2] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from trace_tasks.experimental.certificates import registry  # noqa: WPS433

    stage_of: Dict[str, str] = {}
    fams: Dict[str, List[str]] = {}
    for ad in registry.supported_adapters():
        fams[ad.adapter_id] = list(ad.m_families)
        for fam, stage in ad.family_stages.items():
            stage_of[fam] = stage
    return stage_of, fams


# ============================================================================== cert rows


def cert_values(row: Mapping[str, Any], names: Sequence[str], stage_of: Mapping[str, str]) -> np.ndarray:
    out: Dict[str, Optional[float]] = {}
    A, M = _finite(row.get("A")), _finite(row.get("M"))
    for k in ("A", "G", "M", "F", "joint"):
        out[k] = _finite(row.get(k))
    out["tokens"] = _finite(row.get("completion_tokens"))
    out["truncated"] = _finite(row.get("truncated"))
    out["valid_schema"] = _finite(row.get("valid_schema"))
    ec = row.get("error_class")
    for c in ERROR_CLASSES:
        out[f"ec_{c}"] = None if ec is None else float(ec == c)
    if A is not None and M is not None:
        a1, m1 = A >= 0.5, M >= 1.0
        out["a1_m1"], out["a1_mlt1"] = float(a1 and m1), float(a1 and not m1)
        out["a0_m1"], out["a0_mlt1"] = float((not a1) and m1), float((not a1) and not m1)
    G = out["G"]
    if G is not None and M is not None:
        out["g09_mlt1"] = float(G >= 0.9 and M < 1.0)
    stage_scores: Dict[str, List[float]] = defaultdict(list)
    arith: List[float] = []
    ans: List[float] = []
    for c in row.get("checks") or []:
        if not c.get("applicable"):
            continue
        s = _finite(c.get("score"))
        if s is None:
            continue
        cid, fam, kind = c.get("check_id"), c.get("family"), c.get("kind")
        if kind == "truth" and cid == fam and fam in stage_of:
            stage_scores[stage_of[fam]].append(s)
            out[f"fam:{fam}"] = s
        elif kind == "local" and isinstance(cid, str):
            if cid.endswith(ARITH_LOCAL):
                arith.append(s)
            elif cid.endswith(".answer.local"):
                ans.append(s)
    for st in STAGES:
        vals = stage_scores.get(st)
        out[f"st_{st}"] = float(np.mean(vals)) if vals else None
    out["loc_arith"] = float(np.mean(arith)) if arith else None
    out["loc_answer"] = float(np.mean(ans)) if ans else None
    return np.array([_nan(out.get(n)) for n in names], dtype=float)


def ans_values(row: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    out = {"A_ans": _finite(row.get("A")), "tokens_ans": _finite(row.get("completion_tokens")),
           "truncated_ans": _finite(row.get("truncated"))}
    return np.array([_nan(out.get(n)) for n in names], dtype=float)


def load_cert_datasets(
    specs: Sequence[str], cert_names: Sequence[str], stage_of: Mapping[str, str], splits: Optional[Sequence[str]],
) -> Tuple[cc.Dataset, cc.Dataset, Dict[str, Any]]:
    ds_c = cc.Dataset(metrics=tuple(cert_names))
    ds_a = cc.Dataset(metrics=tuple(ANS_METRICS))
    info: Dict[str, Any] = {}
    for spec in specs:
        arm, tseed, path = cc.parse_arm(spec)
        files = cc.find_score_files(path)
        rec = info.setdefault(arm, {}).setdefault(tseed or "-", {"paths": [], "files": 0, "validation_ok": []})
        rec["paths"].append(path)
        rec["files"] += len(files)
        vj = Path(path) / "validation.json"
        rec["validation_ok"].append(json.loads(vj.read_text()).get("ok") if vj.is_file() else None)
        for ds in (ds_c, ds_a):
            ds.runs.setdefault(arm, {}).setdefault(tseed, cc.Run(arm, tseed))
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for n, line in enumerate(fh):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cond = row.get("condition")
                    if cond == "cert_v1":
                        ds, vals = ds_c, None
                    elif cond == "ans_v1":
                        ds, vals = ds_a, None
                    else:
                        continue
                    split, iid = row.get("split"), row.get("instance_id")
                    if not split or not iid:
                        raise cc.CompareError(f"{f}:{n + 1}: row without split or instance_id")
                    if splits and split not in splits:
                        continue
                    key = (str(split), str(iid))
                    meta = cc.RowMeta(cluster=str(row.get("scene_group") or iid), stratum=str(row.get("adapter_id") or "-"),
                                      adapter=str(row.get("adapter_id") or "-"), cell=str(row.get("cell") or "-"),
                                      ext_id=row.get("cert_extension_id"))
                    for d in (ds_c, ds_a):  # one frozen dataset: metadata must agree everywhere
                        old = d.meta.setdefault(key, meta)
                        if old != meta:
                            raise cc.CompareError(f"{f}: {key} disagrees with an earlier file")
                    seed = cc._decoding_seed(row, f)
                    vals = cert_values(row, cert_names, stage_of) if ds is ds_c else ans_values(row, tuple(ANS_METRICS))
                    per = ds.runs[arm][tseed].rows.setdefault(key, {})
                    if seed in per:
                        raise cc.CompareError(f"{f}: {arm}@{tseed or '-'} has {key} seed {seed} twice")
                    per[seed] = vals
    for ds in (ds_c, ds_a):
        for arm in list(ds.runs):
            for t in list(ds.runs[arm]):
                if not ds.runs[arm][t].rows:
                    del ds.runs[arm][t]
            if not ds.runs[arm]:
                del ds.runs[arm]
    return ds_c, ds_a, info


def arm_means_by_adapter(ds: cc.Dataset) -> Dict[str, Any]:
    """arm -> tseed -> split -> adapter -> {rows, means}; row value = mean over its decoding seeds."""
    out: Dict[str, Any] = {}
    for arm, by_seed in ds.runs.items():
        for tseed, run in by_seed.items():
            acc: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
            for key, vals in run.rows.items():
                stack = np.stack(list(vals.values()))
                ok = ~np.isnan(stack)
                acc[(key[0], ds.meta[key].adapter)].append(cc._ratio(np.where(ok, stack, 0.0).sum(axis=0), ok.sum(axis=0).astype(float)))
            for (split, adapter), rows in acc.items():
                arr = np.stack(rows)
                means = {}
                for j, m in enumerate(ds.metrics):
                    col = arr[:, j][~np.isnan(arr[:, j])]
                    means[m] = cc._f(col.mean()) if col.size else None
                out.setdefault(arm, {}).setdefault(tseed or "-", {}).setdefault(split, {})[adapter] = {"rows": len(rows), "means": means}
    return out


def pooled(per_seed: Mapping[str, Mapping[str, Any]], metrics: Sequence[str], path: Sequence[str] = ()) -> Dict[str, Any]:
    """Mean over training seeds of per-seed means (each seed weighted equally) and the per-seed values."""
    out: Dict[str, Any] = {}
    seeds = sorted(per_seed)
    for m in metrics:
        vals = []
        for t in seeds:
            node: Any = per_seed[t]
            for p in path:
                node = (node or {}).get(p)
            v = None if node is None else node["means"].get(m)
            vals.append(v)
        known = [v for v in vals if v is not None]
        out[m] = {"mean": float(np.mean(known)) if len(known) == len(vals) and known else None,
                  "min": float(min(known)) if known else None, "max": float(max(known)) if known else None,
                  "per_seed": dict(zip(seeds, vals))}
    return out


def pooled_tables(means: Mapping[str, Any], by_adapter: Mapping[str, Any], metrics: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm, by_seed in means.items():
        splits = sorted({s for t in by_seed.values() for s in t})
        for split in splits:
            if not all(split in by_seed[t] for t in by_seed):
                continue
            rows = {t: by_seed[t][split]["rows"] for t in by_seed}
            ent = {"training_seeds": sorted(by_seed), "rows": rows, "metrics": pooled({t: by_seed[t][split] for t in by_seed}, metrics),
                   "adapters": {}}
            ads = sorted({a for t in by_adapter.get(arm, {}).values() for a in t.get(split, {})})
            for ad in ads:
                if all(ad in by_adapter[arm][t].get(split, {}) for t in by_adapter[arm]):
                    ent["adapters"][ad] = pooled({t: by_adapter[arm][t][split][ad] for t in by_adapter[arm]}, metrics)
            out.setdefault(arm, {})[split] = ent
    return out


def strip_comparisons(comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return comps  # every number is kept (the JSON is the record)


def cmd_extras(args: argparse.Namespace) -> int:
    mcnemar, _ = cc.load_trace_tool(args.trace_tool_src)
    stage_of, fams = load_stage_map(args.trace_src)
    fam_metrics = {f"fam:{f}": (False, f"truth score of family {f} ({stage_of[f]} stage)") for ad in sorted(fams) for f in fams[ad]}
    cert_metrics = {**BASE_CERT_METRICS, **fam_metrics}
    _register(cert_metrics)
    _register(ANS_METRICS)
    splits = [s for s in args.splits.split(",") if s] if args.splits else None
    ds_c, ds_a, info = load_cert_datasets(args.model, tuple(cert_metrics), stage_of, splits)
    pairs = [tuple(p.split(":", 1)) for p in args.pairs.split(",") if p] if args.pairs else []
    st = cc.Settings(n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)
    doc: Dict[str, Any] = {
        "version": VERSION, "compare_cert_version": cc.VERSION,
        "config": {"pairs": [list(p) for p in pairs], "n_boot": st.n_boot, "n_perm": st.n_perm, "seed": st.seed,
                   "mcnemar_seed": st.mcnemar_seed, "splits": splits, "cluster_key": "scene_group", "stratum_key": "adapter_id",
                   "argv": sys.argv[1:]},
        "models": info,
        "metrics": {k: {"binary": v[0], "description": v[1]} for k, v in {**cert_metrics, **ANS_METRICS}.items()},
        "stage_of_family": stage_of, "families_by_adapter": fams,
    }
    for label, ds in (("cert", ds_c), ("ans", ds_a)):
        sec: Dict[str, Any] = {}
        if not ds.runs:
            doc[label] = None
            continue
        means = cc.arm_means(ds)
        by_ad = arm_means_by_adapter(ds)
        sec["arm_means"] = means
        sec["arm_means_by_adapter"] = by_ad
        sec["pooled"] = pooled_tables(means, by_ad, ds.metrics)
        use = [(x, y) for x, y in pairs if x in ds.runs and y in ds.runs]
        sec["missing_pairs"] = [list(p) for p in pairs if p not in use]
        sec["comparisons"] = cc.run_comparisons(ds, use, st, mcnemar, cells=False) if use else []
        doc[label] = sec
    Path(args.out_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"extras: {len(ds_c.runs)} cert arms, {len(ds_a.runs)} ans arms, {len(pairs)} pairs -> {args.out_json}")
    return 0


# ============================================================================== TRACE validation


def _iso(s: Optional[str]) -> Optional[_dt.datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return _dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def traceval_table(ckpt_root: Path, expect: Sequence[str], include: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    models: Dict[str, Any] = {}
    names = sorted({p.name for p in ckpt_root.iterdir() if p.is_dir()} | set(expect))
    excluded = [n for n in names if include and not re.search(include, n)]
    for name in names:
        if name in excluded:
            continue
        d = ckpt_root / name
        ent: Dict[str, Any] = {"root": str(d)}
        summ, val = d / "summary.trace_val.json", d / "validation.trace_val.json"
        if not d.is_dir():
            ent["status"] = "pending (no eval root yet)"
        elif not summ.is_file():
            stem = name.rsplit("-", 1)[-1]
            ent["status"] = ("not in the matrix (steps 100/200: e1_dev only)" if stem in ("100", "200") else "pending (no summary.trace_val.json)")
        else:
            ok = json.loads(val.read_text()).get("ok") if val.is_file() else None
            s = json.loads(summ.read_text())
            ent["status"] = "ok" if ok else f"summary present, validation.trace_val.json ok={ok}"
            ent["validation_ok"] = ok
            ent["dataset"] = s.get("split")
            conds = {}
            for cond, v in (s.get("by_cond") or {}).items():
                n = v.get("n") or 0
                pend = v.get("judge_pending") or 0
                lo = v.get("official_acc_lower")
                conds[cond] = {**v, "official_acc_upper": (lo + pend / n) if (lo is not None and n) else None,
                               "judge_pending_rate": (pend / n) if n else None}
            ent["by_cond"] = conds
            ent["runs"] = [{**r, "official_acc_upper": (r["official_acc_lower"] + (r.get("judge_pending") or 0) / r["n"])
                            if r.get("official_acc_lower") is not None and r.get("n") else None} for r in s.get("runs") or []]
            jp = d / "trace_val" / "judge_pending.jsonl"
            ent["judge_pending_unique_keys"] = sum(1 for line in jp.open() if line.strip()) if jp.is_file() else None
        models[name] = ent
    return models, excluded


def range_dataset(ds: cc.Dataset) -> cc.Dataset:
    """Arm X keeps (lo, hi) and its Y-role copy ``X~y`` carries (hi, lo), so the pair X : Y~y gives per row
    tv_range_lo = x_lo - y_hi and tv_range_hi = x_hi - y_lo with compare_cert's own statistics."""
    j_lo, j_hi = ds.metrics.index("tv_acc_lo"), ds.metrics.index("tv_acc_hi")
    rd = cc.Dataset(metrics=tuple(RANGE_METRICS))
    rd.meta = ds.meta
    for arm, by in ds.runs.items():
        for t, run in by.items():
            for name, cols in ((arm, [j_lo, j_hi]), (arm + Y_ROLE, [j_hi, j_lo])):
                r = rd.runs.setdefault(name, {}).setdefault(t, cc.Run(name, t))
                r.rows = {k: {s: v[cols] for s, v in per.items()} for k, per in run.rows.items()}
    return rd


def dataset_stats(ds: cc.Dataset) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in sorted({k[0] for k in ds.meta}):
        metas = [m for k, m in ds.meta.items() if k[0] == split]
        out[split] = {"rows": len(metas), "tasks": len({m.cluster for m in metas}), "domains": len({m.stratum for m in metas})}
    for arm, by in ds.runs.items():
        for t, run in by.items():
            for split in out:
                seeds = sorted({s for k, per in run.rows.items() if k[0] == split for s in per})
                out[split].setdefault("decoding_seeds", {})[f"{arm}@{t or '-'}"] = seeds
    return out


def load_traceval_dataset(specs: Sequence[str]) -> Tuple[cc.Dataset, Dict[str, Any]]:
    names = tuple(TV_METRICS)
    ds = cc.Dataset(metrics=names)
    info: Dict[str, Any] = {}
    for spec in specs:
        arm, tseed, path = cc.parse_arm(spec)
        run = ds.runs.setdefault(arm, {}).setdefault(tseed, cc.Run(arm, tseed))
        files = sorted(Path(path, "trace_val").glob("*/seed*/scores.jsonl"))
        rec = info.setdefault(arm, {}).setdefault(tseed or "-", {"paths": [], "files": []})
        rec["paths"].append(path)
        for f in files:
            cond = f.parent.parent.name
            seed = int(f.parent.name.replace("seed", ""))
            split = f"trace_val:{cond}"
            rec["files"].append(str(f))
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    iid = str(row["instance_id"])
                    key = (split, iid)
                    meta = cc.RowMeta(cluster=str(row.get("task") or iid), stratum=str(row.get("domain") or "-"),
                                      adapter=str(row.get("domain") or "-"), cell=str(row.get("task") or "-"), ext_id=None)
                    old = ds.meta.setdefault(key, meta)
                    if old != meta:
                        raise cc.CompareError(f"{f}: {key} disagrees with an earlier file")
                    lo = _finite(row.get("official_correct"))
                    pend = row.get("official_source") == "judge_pending"
                    hi = 1.0 if pend else lo
                    vals = np.array([_nan(lo), _nan(hi), _nan(_finite(row.get("strict_correct"))),
                                     _nan(_finite(row.get("completion_tokens")))])
                    per = run.rows.setdefault(key, {})
                    if seed in per:
                        raise cc.CompareError(f"{f}: {arm}@{tseed or '-'} has {key} seed {seed} twice")
                    per[seed] = vals
        if not run.rows:
            raise cc.CompareError(f"{arm}@{tseed or '-'}: no trace_val scores under {path}")
    return ds, info


def cmd_traceval(args: argparse.Namespace) -> int:
    mcnemar, _ = cc.load_trace_tool(args.trace_tool_src)
    _register(TV_METRICS)
    _register(RANGE_METRICS)
    table, excluded = traceval_table(Path(args.ckpt_root), [e for e in (args.expect or []) if e], args.include)
    doc: Dict[str, Any] = {"version": VERSION, "config": {"ckpt_root": args.ckpt_root, "expect": args.expect or [],
                                                          "include": args.include, "argv": sys.argv[1:], "n_boot": args.n_boot,
                                                          "n_perm": args.n_perm, "seed": args.seed, "cluster_key": "task",
                                                          "stratum_key": "domain"},
                           "metrics": {k: {"binary": v[0], "description": v[1]} for k, v in {**TV_METRICS, **RANGE_METRICS}.items()},
                           "table": table, "excluded_by_include": excluded}
    specs = []
    skipped = []
    for spec in args.model or []:
        arm, tseed, path = cc.parse_arm(spec)
        summ, val = Path(path) / "summary.trace_val.json", Path(path) / "validation.trace_val.json"
        ok = json.loads(val.read_text()).get("ok") if val.is_file() else None
        if summ.is_file() and ok:
            specs.append(spec)
        else:
            skipped.append({"spec": spec, "reason": f"summary.trace_val.json present={summ.is_file()}, validation.trace_val.json ok={ok}"})
    doc["skipped_models"] = skipped
    if specs:
        ds, info = load_traceval_dataset(specs)
        st = cc.Settings(n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)
        pairs = [tuple(p.split(":", 1)) for p in args.pairs.split(",") if p] if args.pairs else []
        use = [(x, y) for x, y in pairs if x in ds.runs and y in ds.runs]
        doc["models"] = info
        doc["dataset"] = dataset_stats(ds)
        doc["missing_pairs"] = [list(p) for p in pairs if p not in use]
        doc["arm_means"] = cc.arm_means(ds)
        doc["comparisons"] = cc.run_comparisons(ds, use, st, mcnemar, cells=False) if use else []
        rd = range_dataset(ds)
        rcomps = cc.run_comparisons(rd, [(x, y + Y_ROLE) for x, y in use], st, mcnemar, cells=False) if use else []
        for c in rcomps:
            c["y"] = c["y"][: -len(Y_ROLE)]
            c["name"] = f"{c['x']} - {c['y']}"
        doc["range_comparisons"] = rcomps
    Path(args.out_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"traceval: {sum(1 for v in table.values() if v.get('status') == 'ok')} models with a table row -> {args.out_json}")
    return 0


# ============================================================================== training receipts


def cmd_training(args: argparse.Namespace) -> int:
    root = Path(args.rl_root)
    out: Dict[str, Any] = {}
    for scale in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for arm_dir in sorted((root / scale).iterdir()):
            if not arm_dir.is_dir():
                continue
            for seed_dir in sorted(arm_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("s"):
                    continue
                key = f"{scale}/{arm_dir.name}/{seed_dir.name}"
                rec = seed_dir / "run_receipt.json"
                if not rec.is_file():
                    prog = seed_dir / "progress.json"
                    p = json.loads(prog.read_text()) if prog.is_file() else {}
                    out[key] = {"status": "pending (no run_receipt.json)", "last_logged_step": p.get("last_logged_step"),
                                "max_steps": p.get("max_steps"), "host": p.get("host")}
                    continue
                r = json.loads(rec.read_text())
                summ = r.get("summary") or {}
                prog = seed_dir / "progress.json"
                progress = json.loads(prog.read_text()) if prog.is_file() else {}
                hours = 0.0
                hosts, gpus, resumes = [], [], []
                for a in r.get("attempts") or []:
                    t0, t1 = _iso(a.get("started")), _iso(a.get("ended"))
                    if t0 and t1:
                        hours += (t1 - t0).total_seconds() / 3600.0
                    hosts.append(a.get("host"))
                    gpus.append((a.get("gpu") or {}).get("name"))
                    resumes.append(a.get("resumed_from_step"))
                rm = summ.get("reward_mean") or {}
                steps = summ.get("logged_steps")
                n_rollouts = None
                try:
                    import yaml  # type: ignore

                    cfg = yaml.safe_load((seed_dir / "resolved_easyr1.yaml").read_text())
                    n_rollouts = int(cfg["data"]["rollout_batch_size"]) * int(cfg["worker"]["rollout"]["n"])
                except Exception:  # noqa: BLE001 - optional detail
                    n_rollouts = None
                resp = rm.get("resp_len")
                out[key] = {
                    "status": r.get("status"), "arm": r.get("arm"), "seed": r.get("seed"), "model": r.get("model"),
                    "gpu": (r.get("gpu") or {}).get("name"), "n_gpus": r.get("n_gpus"), "attempt_hosts": hosts, "attempt_gpus": gpus,
                    "attempts": len(r.get("attempts") or []), "resumed_from_steps": resumes, "wall_hours": hours,
                    "gpu_hours": hours * (r.get("n_gpus") or 0), "step_seconds_median": summ.get("step_seconds_median"),
                    "logged_steps": steps, "resp_len_mean": resp, "rollouts_per_step": n_rollouts,
                    "rollout_tokens_est": (resp * n_rollouts * steps) if (resp and n_rollouts and steps) else None,
                    "reward_mean": rm, "init": (r.get("model_dir") or {}).get("path"),
                    "gated": (r.get("reward_kwargs") or {}).get("gated"),
                    "last_logged_step": progress.get("last_logged_step"), "max_steps": progress.get("max_steps"),
                }
    doc = {"version": VERSION, "config": {"rl_root": str(root)}, "runs": out}
    Path(args.out_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"training: {len(out)} runs -> {args.out_json}")
    return 0


# ============================================================================== report


def _load(d: Path, name: str) -> Optional[Dict[str, Any]]:
    p = d / name
    return json.loads(p.read_text()) if p.is_file() else None


def pp(v: Optional[float], signed: bool = False, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{100 * v:+.{digits}f}" if signed else f"{100 * v:.{digits}f}"


def tok(v: Optional[float], signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def pv(v: Optional[float]) -> str:
    return "-" if v is None else (f"{v:.2g}" if v < 0.001 else f"{v:.3f}")


def fmt_metric(m: str, v: Optional[float], signed: bool = False) -> str:
    return tok(v, signed) if m.startswith("tokens") or m == "tv_tokens" else pp(v, signed)


def short_adapter(a: str) -> str:
    return {"selected_set.chess_piece_count": "chess (selected set, control)",
            "filtered_mean.table_filtered_column_mean": "table (filtered mean)",
            "weighted_sum.lever_side_torque": "lever (weighted sum)",
            "path.rpg_movement_cost": "rpg (path)"}.get(a, a)


def res_cell(m: str, r: Optional[Mapping[str, Any]]) -> str:
    """Δ [CI] with the per-training-seed sign agreement (k/K, * = claim rule) or the p value (K = 1)."""
    if not r or r.get("delta") is None:
        return "-"
    lo, hi = r["ci"]
    base = f"{fmt_metric(m, r['delta'], True)} [{fmt_metric(m, lo, True)}, {fmt_metric(m, hi, True)}]"
    if r.get("K", 1) > 1:
        return base + f" {r['sign_agree']}/{r['K']}" + (" *" if r.get("claim") else "")
    return base + f" p={pv(r.get('p'))}"


def per_seed_cell(m: str, ps: Mapping[str, Any]) -> str:
    lo, hi = ps["ci"]
    return f"{fmt_metric(m, ps['delta'], True)} [{fmt_metric(m, lo, True)}, {fmt_metric(m, hi, True)}] p={pv(ps.get('p'))}"


def comp_lookup(doc: Optional[Mapping[str, Any]], x: str, y: str, section: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    if not doc:
        return None
    comps = (doc.get(section) or {}).get("comparisons", []) if section else doc.get("comparisons", [])
    for c in comps:
        if c["x"] == x and c["y"] == y:
            return c
    return None


def scope_res(comp: Optional[Mapping[str, Any]], split: str, scope: str, m: str) -> Optional[Mapping[str, Any]]:
    if not comp:
        return None
    return (((comp.get("splits") or {}).get(split) or {}).get("scopes") or {}).get(scope, {}).get(m)


LABELS = {"a1_m1": "A=1,M=1", "a1_mlt1": "A=1,M<1", "a0_m1": "A=0,M=1", "a0_mlt1": "A=0,M<1", "g09_mlt1": "G>=0.9,M<1",
          "loc_arith": "loc arith", "loc_answer": "loc answer", "st_grounding": "stage grounding", "st_operand": "stage operand",
          "st_computation": "stage computation"}


def label(m: str) -> str:
    if m in LABELS:
        return LABELS[m]
    if m.startswith("ec_"):
        return "class " + m[3:]
    if m.startswith("fam:"):
        return m[4:]
    return m


def md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        L.append("| " + " | ".join(str(c).replace("|", "\\|") if i == 0 else str(c) for i, c in enumerate(r)) + " |")
    return L


MAIN_ARMS = ("A", "G", "P")
BORDER = 0.001  # a CI end within 0.1 pp of 0 is flagged as borderline


def hardware_by_run(tr: Optional[Mapping[str, Any]]) -> Dict[Tuple[str, str], str]:
    """(arm, 's<seed>') -> 'N x GPU' for every 3B RL receipt."""
    out: Dict[Tuple[str, str], str] = {}
    for key, r in ((tr or {}).get("runs") or {}).items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "3b":
            out[(parts[1], parts[2])] = f"{r.get('n_gpus') or '?'}x {r.get('gpu') or '?'}"
    return out


def hardware_sentence(tr: Optional[Mapping[str, Any]]) -> str:
    hw = hardware_by_run(tr)
    if not hw:
        return "Hardware: `training.json` missing."
    per_seed: Dict[str, List[str]] = {}
    for s in sorted({s for (a, s) in hw if a in MAIN_ARMS}):
        per_seed[s] = sorted({hw[(a, s)] for a in MAIN_ARMS if (a, s) in hw})
    parts = [f"seed {s[1:]} on {' / '.join(g)}" + ("" if len(g) == 1 else " (differs between arms)") for s, g in per_seed.items()]
    gated = [f"{a} seed {s[1:]} on {hw[(a, s)]}" for (a, s) in sorted(hw) if a.endswith("_gated")]
    txt = "Hardware (RL receipts, `training.json`): A/G/P " + "; ".join(parts) + "."
    if gated:
        txt += " Answer-gated: " + "; ".join(gated) + "."
    if len({tuple(v) for v in per_seed.values()}) > 1:
        txt += " Hardware is therefore nested in training seed, and the two-level CI mixes seed and hardware variation."
    txt += (" Within each seed the three arms ran on the same hardware class." if all(len(v) == 1 for v in per_seed.values())
            else " Not every seed has A/G/P on the same hardware class.")
    return txt


def ci_txt(m: str, r: Mapping[str, Any]) -> str:
    return f"{fmt_metric(m, r['delta'], True)} [{fmt_metric(m, r['ci'][0], True)}, {fmt_metric(m, r['ci'][1], True)}]"


def sign_txt(r: Mapping[str, Any]) -> str:
    if r.get("K", 1) > 1:
        return f"{r['sign_agree']}/{r['K']} seeds agree"
    return f"p={pv(r.get('p'))}, 1 seed (instance sampling only)"


def borderline(r: Optional[Mapping[str, Any]]) -> Optional[str]:
    """A CI end within BORDER of 0 (the claim could flip with another resampling seed)."""
    if not r or r.get("delta") is None or r["ci"][0] is None:
        return None
    lo, hi = r["ci"]
    near = lo if abs(lo) <= abs(hi) else hi
    if abs(near) < BORDER:
        return f"the CI end {pp(near, True)} is within {pp(BORDER)} pp of 0"
    return None


def judge_range_verdict(rlo: Optional[Mapping[str, Any]], rhi: Optional[Mapping[str, Any]]) -> str:
    """Direction that holds under every resolution of judge_pending rows (CI of the least favourable end)."""
    def holds(r: Optional[Mapping[str, Any]], positive: bool) -> bool:
        if not r or r.get("delta") is None or r["ci"][0] is None:
            return False
        lo, hi = r["ci"]
        side = lo > 0 if positive else hi < 0
        return bool(side and (r.get("claim") if r.get("K", 1) > 1 else True))
    if holds(rlo, True):
        return "X > Y under every judge outcome"
    if holds(rhi, False):
        return "X < Y under every judge outcome"
    return "direction not established under every judge outcome"


def report(d: Path) -> str:
    main = _load(d, "cmp_main.json")
    ctx = _load(d, "cmp_context.json")
    gated = _load(d, "cmp_gated.json")
    ex = _load(d, "extras_main.json")
    exc = _load(d, "extras_context.json")
    exg = _load(d, "extras_gated.json")
    exs = _load(d, "extras_steps.json")
    tv = _load(d, "trace_val.json")
    tr = _load(d, "training.json")
    inp = _load(d, "inputs.json") or {}
    L: List[str] = ["# Final 3B in-domain certificate comparison (RQ1, RQ3; DESIGN 7.4, 8)", ""]
    L += ["Generated by `final_cert3b.py report` from the JSON files in this directory; every number below is read from "
          "them (see `run.sh`). Percentages are percentage points; tokens are completion tokens. Arms: S = structured SFT "
          "(cert-sft-3b-s564), A = outcome-only RL, G = grounding RL, P = grounding + semantic certificate RL (all 300 GRPO "
          "steps from S); Pg / Gg = answer-gated P / G (stage 4). Only the two families in section 1 are pre-registered and "
          "Holm-controlled; everything else is descriptive and unadjusted, and a `*` (claim rule) outside section 1 is not "
          "multiplicity-controlled.", "", hardware_sentence(tr), ""]
    # ------------------------------------------------------------------ inputs
    L += ["## Input status", ""]
    if inp.get("items"):
        bad = [it for it in inp["items"] if it["status"] != "ok"]
        kinds = defaultdict(int)
        for it in inp["items"]:
            kinds[it["kind"]] += 1
        L.append(f"{inp['n_ok']} of {inp['n_items']} inputs ok at {inp['checked_at']} ("
                 + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) + "; list in `inputs.json`). "
                 + ("Nothing is pending." if not bad else "NOT ok: " + "; ".join(f"{it['what']}: {it['status']}" for it in bad) + "."))
        for note in inp.get("not_used_here", []):
            L.append(f"- Not used here: {note}.")
    else:
        L.append("- (inputs.json missing)")
    L.append("")
    # ------------------------------------------------------------------ summary
    L += ["## 0. Summary (rendered from the JSON; details in the numbered sections)", ""]
    prim = [t for f in (main or {}).get("families", []) if f["family"] == "primary" for t in f["tests"]]
    sec = [t for f in (main or {}).get("families", []) if f["family"] == "secondary" for t in f["tests"]]
    if prim:
        met_on = [t["split"] for t in prim if t.get("decision") == "X > Y"]
        against = [t["split"] for t in prim if t.get("decision") == "X < Y"]
        verdict = "MET" if len(met_on) == len(prim) else ("PARTLY MET (" + ", ".join(met_on) + ")" if met_on else "NOT MET")
        L.append(f"- **DESIGN 8 go/no-go criterion 2 (P > G on the primary family, answer accuracy A): {verdict}.** "
                 + "; ".join(f"{t['split']}: P - G {ci_txt('A', t['result'])}, {sign_txt(t['result'])}, decision '{t['decision']}'"
                             for t in prim if t.get("result")) + "."
                 + (f" The claim rule favours G on {', '.join(against)}." if against else "")
                 + (" Per DESIGN 8, this result is kept and the semantic-reward claim is narrowed." if not met_on else "")
                 + " (Criterion 1, the semantic signal being distinct from G, is judged on the S checkpoint and the corruption "
                   "suite and is not assessed here.)")
    if sec:
        L.append("- Secondary family (P - A, G - A on A): " + "; ".join(
            f"{t['comparison']} {t['split']} {ci_txt('A', t['result'])} {t['result'].get('sign_agree')}/{t['result'].get('K')} '{t['decision']}'"
            for t in sec if t.get("result")) + ".")
    edge = [(t, borderline(t.get("result"))) for t in prim + sec]
    edge = [(t, b) for t, b in edge if b]
    if edge:
        L.append("- Borderline family tests: " + "; ".join(f"{t['comparison']} {t['split']} ({b})" for t, b in edge)
                 + ". The bootstrap seed is fixed (0), so the decision is reproducible, but it rests on Monte Carlo noise at that edge.")
    if main:
        c = comp_lookup(main, "P", "G")
        items = []
        for split in ("e1_test", "e2_test"):
            for m in ("M", "joint"):
                r = scope_res(c, split, "all", m)
                if r:
                    items.append(f"{m} {split} {ci_txt(m, r)} {r['sign_agree']}/{r['K']}" + (" *" if r.get("claim") else ""))
        L.append("- Certificate quality P - G (descriptive, unadjusted; one of many metric x split cells): " + "; ".join(items) + ".")
        items = []
        for x in ("A", "G", "P"):
            for split in ("e1_test", "e2_test"):
                r = scope_res(comp_lookup(main, x, "S"), split, "all", "A")
                if r:
                    items.append(f"{x} - S {split} {ci_txt('A', r)} {r['sign_agree']}/{r['K']}" + (" *" if r.get("claim") else ""))
        L.append("- RL over SFT on A (descriptive, S treated as fixed): " + "; ".join(items) + ".")
    if gated and comp_lookup(gated, "Pg", "Gg"):
        c = comp_lookup(gated, "Pg", "Gg")
        items = []
        for split in ("e1_test", "e2_test"):
            items.append(f"{split}: " + ", ".join(f"{m} {ci_txt(m, scope_res(c, split, 'all', m))} p={pv(scope_res(c, split, 'all', m).get('p'))}"
                                                 for m in ("A", "M", "joint") if scope_res(c, split, "all", m)))
        ref = comp_lookup(gated, "P3", "G3")
        reft = ""
        if ref:
            reft = (" Ungated reference on the same hardware class, P3 - G3 (seed 3) A: "
                    + "; ".join(f"{s} {ci_txt('A', scope_res(ref, s, 'all', 'A'))}" for s in ("e1_test", "e2_test") if scope_res(ref, s, "all", "A")) + ".")
        L.append("- Matched stage-4 contrast P_gated - G_gated (seed 1, same hardware, descriptive, CI from instance sampling "
                 "only): " + "; ".join(items) + "." + reft)
    if tv and tv.get("range_comparisons"):
        items = []
        for x, y in (("S", "base"), ("legacy", "base"), ("A", "S"), ("G", "S"), ("P", "S"), ("P", "G")):
            rc = comp_lookup({"comparisons": tv["range_comparisons"]}, x, y)
            rlo, rhi = scope_res(rc, "trace_val:ans_v1", "all", "tv_range_lo"), scope_res(rc, "trace_val:ans_v1", "all", "tv_range_hi")
            if rlo and rhi:
                items.append(f"{x} - {y} [{pp(rlo['delta'], True)}, {pp(rhi['delta'], True)}] ({judge_range_verdict(rlo, rhi).replace('X', x).replace('Y', y)})")
        L.append("- TRACE released validation, `ans_v1`, range of the paired difference over judge_pending outcomes "
                 "(X lower - Y upper, X upper - Y lower): " + "; ".join(items) + ".")
    L.append("")
    # ------------------------------------------------------------------ families
    L += ["## 1. Pre-registered families (compare_cert, Holm within family)", ""]
    if main:
        cfg = main["config"]
        L.append(f"Condition `{cfg['condition']}`; {cfg['n_boot']} bootstrap resamples, cluster `{cfg['cluster_key']}` stratified by "
                 f"`{cfg['stratum_key']}`; McNemar exact on decoding seed {cfg['mcnemar_seed']} per training seed, Holm within the "
                 "family separately for each training-seed index; the claim rule needs the two-level 95% CI to exclude 0 and "
                 "the sign to agree in at least 2 of 3 training seeds. With 3 training seeds the family decision comes from the "
                 "claim rule; Holm enters only through the per-seed McNemar p values. Source: `cmp_main.json`.")
        L.append("")
        rows = []
        for fam in main["families"]:
            for t in fam["tests"]:
                r = t["result"]
                if r is None:
                    rows.append([fam["family"], t["comparison"], t["split"], t["metric"], "missing", "", "", "", "", "", ""])
                    continue
                per = r.get("per_seed") or []
                rows.append([
                    fam["family"], t["comparison"], t["split"], t["metric"],
                    pp(r["delta"], True), f"[{pp(r['ci'][0], True)}, {pp(r['ci'][1], True)}]",
                    ", ".join(pp(s["delta"], True) for s in per) or "-",
                    ", ".join(pv(s.get("p")) for s in per) or pv(r.get("p")),
                    ", ".join(pv(x) for x in t.get("holm_p_per_seed", [])) or pv(t.get("holm_p")),
                    f"{r.get('sign_agree', '-')}/{r.get('K', 1)} (need {r.get('sign_needed', '-')})",
                    t["decision"],
                ])
        L += md_table(["family", "X - Y", "split", "metric", "Δ (mean of seeds)", "two-level 95% CI", "per-seed Δ (s1, s2, s3)",
                       "per-seed McNemar p", "per-seed Holm p", "sign agree", "decision"], rows)
        L.append("")
        met_on = [t["split"] for t in prim if t.get("decision") == "X > Y"]
        L.append(f"**Go/no-go criterion 2 (DESIGN 8: P > G on the primary family) is {'met on ' + ', '.join(met_on) if met_on else 'not met'}** "
                 f"(decisions: {'; '.join(t['split'] + ': ' + str(t.get('decision')) for t in prim)})."
                 + ("" if met_on else " No split supports P > G on answer accuracy")
                 + ("" if met_on else (f"; the claim rule favours G on {', '.join(t['split'] for t in prim if t.get('decision') == 'X < Y')}."
                                       if any(t.get("decision") == "X < Y" for t in prim) else ", and none supports G > P under the claim rule.")))
        sig_seed = [(t, t["result"]["per_seed"][i], h) for t in prim for i, h in enumerate(t.get("holm_p_per_seed", [])) if h < cfg["alpha"]]
        if sig_seed:
            L.append("Individual training seeds with a Holm-significant primary McNemar test (not a family decision): "
                     + "; ".join(f"{t['split']} seed {ps['tseeds'][0]}: P - G {pp(ps['delta'], True)}, Holm p {pv(h)}"
                                 for t, ps, h in sig_seed) + ".")
        for t, b in edge:
            L.append(f"Borderline: {t['comparison']} {t['split']}: {b}.")
        L.append("")
    else:
        L += ["cmp_main.json missing.", ""]
    # ------------------------------------------------------------------ arm means
    L += ["## 2. Arm means per split (descriptive)", "",
          "Each cell is the mean over training seeds of the per-seed mean (row value = mean over decoding seeds), with the "
          "per-seed range in brackets. `A ans` is the same rows under the answer-only prompt (`ans_v1`). Source: `extras_main.json` "
          "(`cert.pooled`, `ans.pooled`), `extras_context.json` for base / legacy and `extras_gated.json` for Pg / Gg.", ""]
    met = [("A", "A"), ("G", "G"), ("M", "M"), ("F", "F"), ("joint", "joint"), ("tokens", "tokens"), ("truncated", "trunc."), ("valid_schema", "valid schema")]
    for split in ("e1_test", "e2_test", "e1_dev"):
        rows = []
        for src, arms in ((exc, ("base", "legacy")), (ex, ("S", "A", "G", "P")), (exg, ("Pg", "Gg"))):
            if not src or not src.get("cert"):
                continue
            for arm in arms:
                ent = src["cert"]["pooled"].get(arm, {}).get(split)
                if not ent:
                    continue
                ans = ((src.get("ans") or {}).get("pooled") or {}).get(arm, {}).get(split)
                nrows = sorted(set(ent["rows"].values()))
                row = [arm, str(len(ent["training_seeds"])), "/".join(str(n) for n in nrows)]
                for m, _ in met:
                    v = ent["metrics"][m]
                    rng = f" [{fmt_metric(m, v['min'])}-{fmt_metric(m, v['max'])}]" if len(ent["training_seeds"]) > 1 else ""
                    row.append(fmt_metric(m, v["mean"]) + rng)
                if ans:
                    v = ans["metrics"]["A_ans"]
                    row.append(pp(v["mean"]) + (f" [{pp(v['min'])}-{pp(v['max'])}]" if len(ans["training_seeds"]) > 1 else ""))
                else:
                    row.append("-")
                rows.append(row)
        if rows:
            L += [f"### {split}", ""] + md_table(["arm", "training seeds", "rows", *[n for _, n in met], "A ans"], rows) + [""]
    if ex and ex.get("cert"):
        fv = []
        for arm in ("S", "A", "G", "P"):
            for split in ("e1_test", "e2_test"):
                ent = ex["cert"]["pooled"].get(arm, {}).get(split)
                if ent:
                    v = ent["metrics"]["F"]
                    fv.append((v["mean"], v["min"], arm, split))
        if fv:
            lo_mean = min(fv)
            lo_seed = min(fv, key=lambda z: z[1])
            txt = (f"Strict format F on e1_test/e2_test: the seed-mean is at least {pp(lo_mean[0])} for S/A/G/P (lowest {lo_mean[2]} "
                   f"{lo_mean[3]}); the lowest single training seed is {pp(lo_seed[1])} ({lo_seed[2]} {lo_seed[3]}).")
            extra = []
            for src, arms in ((exg, ("Pg", "Gg")), (exc, ("base", "legacy"))):
                for arm in arms:
                    for split in ("e1_test", "e2_test"):
                        ent = ((src or {}).get("cert") or {}).get("pooled", {}).get(arm, {}).get(split)
                        if ent:
                            extra.append(f"{arm} {split} {pp(ent['metrics']['F']['mean'])}")
            if extra:
                txt += " Single-checkpoint models: " + ", ".join(extra) + "."
            L += [txt, ""]
    # ------------------------------------------------------------------ per-seed differences
    L += ["## 3. Paired differences per training seed (descriptive except section 1)", "",
          "Cells: Δ [95% CI] with p (McNemar seed 42 for A/joint, sign-flip for M/tokens) for each training seed, then the "
          "mean of seeds with its two-level CI and sign agreement (* = claim rule holds). X - S pairs use the single S checkpoint "
          "against each seed of X. Source: `cmp_main.json` (comparisons).", ""]
    if main:
        pairs = [("P", "G"), ("P", "A"), ("G", "A"), ("P", "S"), ("G", "S"), ("A", "S")]
        for m in ("A", "M", "joint", "tokens"):
            rows = []
            for x, y in pairs:
                c = comp_lookup(main, x, y)
                for split in ("e1_test", "e2_test", "e1_dev"):
                    r = scope_res(c, split, "all", m)
                    if not r:
                        continue
                    per = [per_seed_cell(m, s) for s in (r.get("per_seed") or [r])]
                    per = (per + ["-"] * 3)[:3]
                    rows.append([f"{x} - {y}", split] + per + [res_cell(m, r)])
            L += [f"### {m}", ""] + md_table(["pair", "split", "seed 1", "seed 2", "seed 3", "mean of seeds"], rows) + [""]
    # ------------------------------------------------------------------ per adapter
    L += ["## 4. Per-adapter breakdown (two-level, descriptive)", "",
          "Δ [two-level 95% CI] k/K (* = claim rule) for A, M and joint; chess is the selected-set control, the lever is "
          "near ceiling. Source: `cmp_main.json` (scopes `adapter:*`).", ""]
    if main:
        rows = []
        for x, y in (("P", "G"), ("P", "A"), ("G", "A"), ("P", "S"), ("G", "S"), ("A", "S")):
            c = comp_lookup(main, x, y)
            for split in ("e1_test", "e2_test"):
                for ad in ("filtered_mean.table_filtered_column_mean", "path.rpg_movement_cost", "selected_set.chess_piece_count", "weighted_sum.lever_side_torque"):
                    sc = f"adapter:{ad}"
                    if not scope_res(c, split, sc, "A"):
                        continue
                    rows.append([f"{x} - {y}", split, short_adapter(ad)] + [res_cell(m, scope_res(c, split, sc, m)) for m in ("A", "M", "joint")])
        L += md_table(["pair", "split", "adapter", "A", "M", "joint"], rows) + [""]
        hl = [r for r in rows if r[3].endswith("*") or r[4].endswith("*") or r[5].endswith("*")]
        L.append("Adapter-level results where the claim rule holds on A, M or joint (descriptive, unadjusted): "
                 + ("; ".join(f"{r[0]} {r[1]} {r[2]}" for r in hl) if hl else "none") + ".")
        pg = [r for r in hl if r[0] == "P - G"]
        L.append("Of these, P - G: " + ("; ".join(f"{r[1]} {r[2]}" for r in pg) if pg else "none") + ".")
        L.append("")
    # ------------------------------------------------------------------ RQ3
    L += ["## 5. RQ3: which errors change", "",
          "Error class = first failing stage of the certificate (scorer `classify_error`: envelope, grounding, operand, "
          "computation, answer_inconsistent, correct); rates are shares of rows. The five failure classes exclude `correct`, "
          "which is the joint-correct row (A = 1 and M = 1). Stage scores are mean truth scores of the "
          "families each adapter maps to that stage. `loc arith` is the local arithmetic consistency on the model's own "
          "operands (parsed certificates only). Truth families are gated on their inputs (a wrong operand also fails the "
          "dependent product / sum / total truth), so stage truth scores propagate upstream errors; the error class (first "
          "failing stage) and `loc arith` are the cleaner attributions. `A=1,M<1` = right answer with an untrue "
          "certificate; `G>=0.9,M<1` = well-placed boxes with untrue semantics. Source: `extras_main.json`.", ""]
    if ex and ex.get("cert"):
        cls = [f"ec_{c}" for c in ERROR_CLASSES] + ["a1_mlt1", "a0_m1", "g09_mlt1"]
        for split in ("e1_test", "e2_test"):
            rows = []
            for arm in ("S", "A", "G", "P"):
                ent = ex["cert"]["pooled"].get(arm, {}).get(split)
                if ent:
                    rows.append([arm] + [pp(ent["metrics"][m]["mean"]) for m in cls])
            L += [f"### Error-class rates, {split} (mean of training seeds)", ""]
            L += md_table(["arm"] + [label(m).replace("class ", "") for m in cls], rows) + [""]
        stage_m = ["st_grounding", "st_operand", "st_computation", "loc_arith", "loc_answer"] + cls
        for split in ("e1_test", "e2_test"):
            rows = []
            for x, y in (("P", "G"), ("P", "A"), ("G", "A"), ("P", "S"), ("G", "S"), ("A", "S")):
                c = comp_lookup(ex, x, y, "cert")
                if not scope_res(c, split, "all", "st_grounding"):
                    continue
                rows.append([f"{x} - {y}"] + [res_cell(m, scope_res(c, split, "all", m)) for m in stage_m])
            L += [f"### Paired differences in stage scores and error classes, {split} (all adapters)", ""]
            L += md_table(["pair"] + [label(m) for m in stage_m], rows) + [""]
        c = comp_lookup(ex, "P", "G", "cert")
        for split in ("e1_test", "e2_test"):
            fc = [m for m in FAILURE_CLASSES if (scope_res(c, split, "all", f"ec_{m}") or {}).get("claim")]
            rc, rj = scope_res(c, split, "all", "ec_correct"), scope_res(c, split, "all", "joint")
            same = rc and rj and rc["delta"] is not None and rj["delta"] is not None and abs(rc["delta"] - rj["delta"]) < 1e-12
            L.append(f"P - G, {split}: failure classes meeting the claim rule: {', '.join(fc) if fc else 'none'}. "
                     + (f"The `correct` class ({'identical to joint, ' if same else ''}{ci_txt('ec_correct', rc)} {rc['sign_agree']}/{rc['K']}"
                        + (" *" if rc.get("claim") else "") + ")" if rc else "")
                     + " (descriptive, unadjusted).")
        L.append("")
        # numeric adapters, per family
        fams = ex.get("families_by_adapter", {})
        for ad in ("filtered_mean.table_filtered_column_mean", "path.rpg_movement_cost", "selected_set.chess_piece_count"):
            fm = [f"fam:{f}" for f in fams.get(ad, [])]
            ms = ["ec_grounding", "ec_operand", "ec_computation", "ec_correct", "loc_arith"] + fm
            rows = []
            for split in ("e1_test", "e2_test"):
                for x, y in (("P", "G"), ("P", "A"), ("G", "A")):
                    c = comp_lookup(ex, x, y, "cert")
                    sc = f"adapter:{ad}"
                    if not scope_res(c, split, sc, "ec_correct"):
                        continue
                    rows.append([f"{x} - {y}", split] + [res_cell(m, scope_res(c, split, sc, m)) for m in ms])
            if rows:
                L += [f"### {short_adapter(ad)}: error classes and family truth scores", ""]
                L += md_table(["pair", "split"] + [label(m) for m in ms], rows) + [""]
        # arm means of family truth scores per adapter
        rows = []
        for split in ("e1_test", "e2_test"):
            for ad in ("filtered_mean.table_filtered_column_mean", "path.rpg_movement_cost", "selected_set.chess_piece_count", "weighted_sum.lever_side_torque"):
                for arm in ("S", "A", "G", "P"):
                    ent = ex["cert"]["pooled"].get(arm, {}).get(split, {}).get("adapters", {}).get(ad)
                    if not ent:
                        continue
                    rows.append([short_adapter(ad), split, arm] + [pp(ent[m]["mean"]) for m in ("A", "M", "G", "joint", "ec_grounding", "ec_operand", "ec_computation", "ec_correct")]
                                + [tok(ent["tokens"]["mean"])])
        L += ["### Per-adapter arm means (mean of training seeds)", ""]
        L += md_table(["adapter", "split", "arm", "A", "M", "G", "joint", "grounding err", "operand err", "computation err", "correct", "tokens"], rows) + [""]
    # ------------------------------------------------------------------ verbosity
    L += ["## 6. Verbosity and completion", "",
          "Completion tokens and truncation under the certificate prompt and the answer-only prompt; Δ tokens are paired "
          "(two-level CI, k/K). Source: `extras_main.json`.", ""]
    if ex and ex.get("cert"):
        rows = []
        for split in ("e1_test", "e2_test"):
            for arm in ("S", "A", "G", "P"):
                c_ent = ex["cert"]["pooled"].get(arm, {}).get(split)
                a_ent = ((ex.get("ans") or {}).get("pooled") or {}).get(arm, {}).get(split)
                if not c_ent:
                    continue
                rows.append([arm, split, tok(c_ent["metrics"]["tokens"]["mean"]), pp(c_ent["metrics"]["truncated"]["mean"]),
                             tok(a_ent["metrics"]["tokens_ans"]["mean"]) if a_ent else "-", pp(a_ent["metrics"]["truncated_ans"]["mean"]) if a_ent else "-",
                             pp(a_ent["metrics"]["A_ans"]["mean"]) if a_ent else "-"])
        L += md_table(["arm", "split", "tokens (cert)", "truncated % (cert)", "tokens (ans)", "truncated % (ans)", "A (ans)"], rows) + [""]
        rows = []
        for x, y in (("P", "G"), ("P", "A"), ("G", "A"), ("P", "S"), ("G", "S"), ("A", "S")):
            c = comp_lookup(ex, x, y, "cert")
            ca = comp_lookup(ex, x, y, "ans")
            for split in ("e1_test", "e2_test"):
                if not scope_res(c, split, "all", "tokens"):
                    continue
                rows.append([f"{x} - {y}", split, res_cell("tokens", scope_res(c, split, "all", "tokens")), res_cell("truncated", scope_res(c, split, "all", "truncated")),
                             res_cell("A_ans", scope_res(ca, split, "all", "A_ans")), res_cell("tokens_ans", scope_res(ca, split, "all", "tokens_ans"))])
        L += md_table(["pair", "split", "Δ tokens (cert)", "Δ truncated (cert)", "Δ A (ans prompt)", "Δ tokens (ans)"], rows) + [""]
    # ------------------------------------------------------------------ base / legacy context
    L += ["## 7. SFT contribution and base / legacy context (descriptive, one checkpoint each)", "",
          "base = Qwen2.5-VL-3B-Instruct, legacy = released TRACE-3B (answer-only RL), S = structured SFT. Single checkpoints, so "
          "CIs reflect instance sampling only. Source: `cmp_context.json` (cert prompt) and `extras_context.json` "
          "(answer-only prompt `A_ans`).", ""]
    if ctx:
        rows = []
        for c in ctx["comparisons"]:
            for split in ("e1_test", "e2_test"):
                r = scope_res(c, split, "all", "A")
                if not r:
                    continue
                ca = comp_lookup(exc, c["x"], c["y"], "ans")
                rows.append([c["name"], split] + [res_cell(m, scope_res(c, split, "all", m)) for m in ("A", "G", "M", "joint", "tokens")]
                            + [res_cell("A_ans", scope_res(ca, split, "all", "A_ans"))])
        L += md_table(["X - Y", "split", "A", "G", "M", "joint", "tokens", "A (ans prompt)"], rows) + [""]
    # ------------------------------------------------------------------ gated
    hw = hardware_by_run(tr)

    def hw_of(arm: str, seed: str) -> str:
        return hw.get((arm, seed), "unknown hardware")
    L += ["## 8. Answer-gated variant (stage 4; descriptive, one training seed)", "",
          f"Pg = P_gated seed 1 ({hw_of('P_gated', 's1')}), Gg = G_gated seed 1 ({hw_of('G_gated', 's1')}). "
          + ("**Pg - Gg is the matched stage-4 contrast** (same seed, init, data order and hardware class; only the reward differs). "
             if gated and comp_lookup(gated, "Pg", "Gg") else "G_gated is not available, so the matched contrast Pg - Gg is not computed. ")
          + f"For reference, P1/G1 are seed 1 ({hw_of('P', 's1')}; same seed, other hardware) and P3/G3 are seed 3 "
          f"({hw_of('P', 's3')}; same hardware, other seed); P1 - G1 and P3 - G3 are the ungated P - G contrasts at those settings, "
          "and comparisons of a gated with an ungated model confound gating with hardware or seed. One training seed, so every "
          "CI reflects instance sampling only and p is McNemar (seed 42) for A / joint and sign-flip otherwise. "
          "Source: `cmp_gated.json`, `extras_gated.json`.", ""]
    if gated:
        rows = []
        for c in gated["comparisons"]:
            for split in ("e1_test", "e2_test"):
                if not scope_res(c, split, "all", "A"):
                    continue
                cx = comp_lookup(exg, c["x"], c["y"], "cert")
                rows.append([c["name"], split] + [res_cell(m, scope_res(c, split, "all", m)) for m in ("A", "M", "joint", "tokens")]
                            + [res_cell(m, scope_res(cx, split, "all", m)) for m in ("ec_grounding", "ec_operand", "ec_computation")])
        L += md_table(["X - Y", "split", "A", "M", "joint", "tokens", "class grounding", "class operand", "class computation"], rows) + [""]
        c = comp_lookup(gated, "Pg", "Gg")
        if c:
            rows = []
            for split in ("e1_test", "e2_test"):
                for ad in ("filtered_mean.table_filtered_column_mean", "path.rpg_movement_cost", "selected_set.chess_piece_count", "weighted_sum.lever_side_torque"):
                    sc = f"adapter:{ad}"
                    if scope_res(c, split, sc, "A"):
                        rows.append([split, short_adapter(ad)] + [res_cell(m, scope_res(c, split, sc, m)) for m in ("A", "M", "joint")])
            L += ["Pg - Gg per adapter:", ""] + md_table(["split", "adapter", "A", "M", "joint"], rows) + [""]
    # ------------------------------------------------------------------ steps
    if exs and exs.get("cert"):
        pooled_c = exs["cert"]["pooled"]
        nrows = sorted({n for arm in pooled_c.values() for n in ((arm.get("e1_dev") or {}).get("rows") or {}).values()})
        dseeds = sorted({s for arm in (exs["cert"].get("arm_means") or {}).values() for t in arm.values()
                         for s in (t.get("e1_dev") or {}).get("decoding_seeds", [])})
        L += ["## 9. Training-step trajectory on E1-dev (descriptive)", "",
              f"Arm means on e1_dev ({'/'.join(str(n) for n in nrows)} rows"
              + (f", decoding seeds {', '.join(str(s) for s in dseeds)}" if dseeds else "")
              + ") at steps 100 / 200 / 300; S = step 0. Source: `extras_steps.json`.", ""]
        rows = []
        for arm in sorted(pooled_c):
            ent = pooled_c[arm].get("e1_dev")
            if not ent:
                continue
            per = ent["metrics"]["A"]["per_seed"]
            rows.append([arm, ", ".join(sorted(per))] + [fmt_metric(m, ent["metrics"][m]["mean"]) for m in ("A", "M", "G", "joint", "tokens")])
        L += md_table(["arm_step", "training seeds", "A", "M", "G", "joint", "tokens"], rows) + [""]
        rows = []
        for c in exs["cert"].get("comparisons", []):
            if scope_res(c, "e1_dev", "all", "A"):
                rows.append([c["name"]] + [res_cell(m, scope_res(c, "e1_dev", "all", m)) for m in ("A", "M", "joint")])
        if rows:
            L += ["Paired differences on e1_dev at the same step (Δ [CI] k/K, or p for single checkpoints):", ""]
            L += md_table(["X - Y", "A", "M", "joint"], rows) + [""]
    # ------------------------------------------------------------------ TRACE val
    L += ["## 10. TRACE released validation (breadth cost)", ""]
    if tv:
        ds_ = tv.get("dataset") or {}
        a_st, n_st = ds_.get("trace_val:ans_v1") or {}, ds_.get("trace_val:no_tool") or {}
        a_seeds = sorted({s for v in (a_st.get("decoding_seeds") or {}).values() for s in v})
        n_seeds = sorted({s for v in (n_st.get("decoding_seeds") or {}).values() for s in v})
        L += [f"Answer-only TRACE validation: {a_st.get('rows', '-')} rows ({a_st.get('tasks', '-')} tasks, {a_st.get('domains', '-')} domains) "
              f"per decoding seed. `ans_v1` pools decoding seeds {', '.join(map(str, a_seeds)) or '-'}; `no_tool` is TRACE's protocol "
              f"(2048-token cap), seed {', '.join(map(str, n_seeds)) or '-'}. judge_pending rows have no judge verdict: `lower` counts them "
              "wrong (official lower bound), `upper` counts them right. 3B models only"
              + (f" (`--include {tv['config'].get('include')}`; {len(tv.get('excluded_by_include') or [])} other eval roots, the 7B ones, are in results/final/cert_7b)"
                 if tv["config"].get("include") else "") + ". Source: `trace_val.json`.", ""]
        rows = []
        for name, ent in sorted(tv["table"].items()):
            if ent.get("status") != "ok":
                if "not in the matrix" in ent.get("status", ""):
                    continue
                rows.append([name, ent.get("status", "-")] + ["-"] * 8)
                continue
            a, nt = ent["by_cond"].get("ans_v1") or {}, ent["by_cond"].get("no_tool") or {}
            per = {r["seed"]: r for r in ent.get("runs", []) if r.get("cond") == "ans_v1"}
            rows.append([name, "ok", pp(a.get("official_acc_lower")), pp(a.get("official_acc_upper")), f"{a.get('judge_pending', '-')} / {a.get('n', '-')}",
                         ", ".join(f"{s}: {pp(per[s]['official_acc_lower'])}" for s in sorted(per)),
                         pp(nt.get("official_acc_lower")), pp(nt.get("official_acc_upper")), f"{nt.get('judge_pending', '-')} / {nt.get('n', '-')}",
                         f"{a.get('tokens_mean', 0):.1f}" if a else "-"])
        L += md_table(["model", "status", "ans_v1 lower", "ans_v1 upper", "ans_v1 pending / n", "per seed (lower)",
                       "no_tool lower", "no_tool upper", "no_tool pending / n", "tokens (ans_v1)"], rows) + [""]
        if tv.get("comparisons"):
            L += ["Paired per-row differences (cluster = task, stratum = domain). `lower - lower` and `upper - upper` are NOT "
                  "bounds on the difference: under every resolution of the judge_pending rows the difference lies in "
                  "[X lower - Y upper, X upper - Y lower], and the last column states the direction only when the CI of the "
                  "least favourable end (with the sign rule for 3 seeds) excludes 0.", ""]
            rows = []
            rcs = {"comparisons": tv.get("range_comparisons") or []}
            for c in tv["comparisons"]:
                rc = comp_lookup(rcs, c["x"], c["y"])
                for split in ("trace_val:ans_v1", "trace_val:no_tool"):
                    if not scope_res(c, split, "all", "tv_acc_lo"):
                        continue
                    rlo, rhi = scope_res(rc, split, "all", "tv_range_lo"), scope_res(rc, split, "all", "tv_range_hi")
                    rows.append([c["name"], split.split(":")[1]] + [res_cell(m, scope_res(c, split, "all", m)) for m in ("tv_acc_lo", "tv_acc_hi")]
                                + [res_cell("tv_range_lo", rlo), res_cell("tv_range_hi", rhi),
                                   judge_range_verdict(rlo, rhi).replace("X", c["x"]).replace("Y", c["y"]) if rlo and rhi else "-",
                                   res_cell("tv_tokens", scope_res(c, split, "all", "tv_tokens"))])
            L += md_table(["X - Y", "condition", "Δ lower - lower", "Δ upper - upper", "X lower - Y upper", "X upper - Y lower",
                           "direction under every judge outcome", "Δ tokens"], rows) + [""]
    # ------------------------------------------------------------------ training
    if tr:
        L += ["## 11. Training runs (matching and compute)", "",
              "From each RL run receipt: GPU class, attempts, wall hours, median step seconds and the mean rollout response "
              "length over the logged steps (rollout tokens = mean length x rollouts per step x steps). Source: `training.json`.", ""]
        rows = []
        for key, r in sorted(tr["runs"].items()):
            if not key.startswith("3b/"):
                continue
            if r.get("status") != "completed":
                rows.append([key, f"{r.get('status', '-')} (step {r.get('last_logged_step')}/{r.get('max_steps')})", r.get("gpu") or "-", "-", "-", "-", "-", "-", "-"])
                continue
            rows.append([key, r["status"], f"{r.get('n_gpus')}x {r.get('gpu') or '-'}", str(r.get("attempts")), f"{r['wall_hours']:.1f}",
                         f"{r['step_seconds_median']:.0f}" if r.get("step_seconds_median") else "-",
                         str(r.get("logged_steps") or "-"),
                         f"{r['resp_len_mean']:.0f}" if r.get("resp_len_mean") else "-",
                         f"{r['rollout_tokens_est'] / 1e6:.1f} M" if r.get("rollout_tokens_est") else "-"])
        L += md_table(["run", "status", "GPU", "attempts", "wall h", "step s (median)", "steps", "rollout len (mean)", "rollout tokens"], rows) + [""]
    return "\n".join(L) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    d = Path(args.dir)
    md = report(d)
    Path(args.out_md).write_text(md)
    print(f"report -> {args.out_md} ({len(md.splitlines())} lines)")
    return 0


def cmd_inputs(args: argparse.Namespace) -> int:
    """Status of every input root and RL receipt this analysis reads (inputs.json, summarized in the report).

    Exit status 1 when any is not ok, so run.sh stops instead of silently dropping an input."""
    root = Path(args.runs_root)
    items = []

    def ckpt(name: str, kind: str) -> None:
        d = root / "eval" / "ckpt" / name
        v = d / ("validation.trace_val.json" if kind == "trace_val" else "validation.json")
        ok = json.loads(v.read_text()).get("ok") if v.is_file() else None
        status = "ok" if ok else ("eval root exists, validation not ok" if d.is_dir() else "missing (no eval root)")
        items.append({"what": f"{name} ({'TRACE-val' if kind == 'trace_val' else 'cert evals'})", "kind": kind,
                      "status": status, "check": str(v) + " ok"})

    for name in args.need or []:
        ckpt(name, "cert")
    for name in args.need_tv or []:
        ckpt(name, "trace_val")
    for key in args.receipt or []:
        d = root / "rl" / key
        rec = d / "run_receipt.json"
        prog = json.loads((d / "progress.json").read_text()) if (d / "progress.json").is_file() else {}
        st = json.loads(rec.read_text()).get("status") if rec.is_file() else None
        items.append({"what": f"rl/{key} training receipt", "kind": "receipt", "check": str(rec) + " status",
                      "status": "ok" if st == "completed" else
                      f"not completed ({st or 'no receipt'}, step {prog.get('last_logged_step')}/{prog.get('max_steps')})"})
    bad = [it for it in items if it["status"] != "ok"]
    doc = {"version": VERSION, "checked_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "items": items,
           "n_items": len(items), "n_ok": len(items) - len(bad), "all_ok": not bad,
           "not_used_here": ["trace_eval_v1 / external-scorer outputs (ext_eval): no number in this analysis comes from them, "
                             "so the 2026-10-23/24 external-scorer fix-3 change of TRACE_EVAL_CODE_HASH does not apply here",
                             "7B models (results/final/cert_7b)"]}
    Path(args.out_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"inputs: {doc['n_ok']}/{doc['n_items']} ok")
    for it in bad:
        print(f"NOT OK: {it['what']}: {it['status']}", file=sys.stderr)
    return 1 if bad else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extras")
    e.add_argument("--model", action="append", required=True, help="NAME[@TRAINING_SEED]=<eval/ckpt/served root> (repeat)")
    e.add_argument("--pairs", help="X:Y,... (X - Y)")
    e.add_argument("--splits", help="comma list (default every split present)")
    e.add_argument("--out-json", required=True)
    t = sub.add_parser("traceval")
    t.add_argument("--ckpt-root", required=True)
    t.add_argument("--expect", action="append", help="eval root names that must appear (pending if absent)")
    t.add_argument("--model", action="append", help="NAME[@TRAINING_SEED]=<eval/ckpt/served root> for paired comparisons")
    t.add_argument("--pairs")
    t.add_argument("--include", help="regex: table rows only for eval roots whose name matches (others listed as excluded)")
    t.add_argument("--out-json", required=True)
    r = sub.add_parser("training")
    r.add_argument("--rl-root", required=True)
    r.add_argument("--out-json", required=True)
    q = sub.add_parser("inputs")
    q.add_argument("--runs-root", required=True)
    q.add_argument("--need", action="append", help="eval/ckpt root whose validation.json must be ok (repeat)")
    q.add_argument("--need-tv", action="append", help="eval/ckpt root whose validation.trace_val.json must be ok (repeat)")
    q.add_argument("--receipt", action="append", help="rl/<scale>/<arm>/s<seed> whose run_receipt.json must say completed")
    q.add_argument("--out-json", required=True)
    m = sub.add_parser("report")
    m.add_argument("--dir", required=True)
    m.add_argument("--out-md", required=True)
    for sp in (e, t):
        sp.add_argument("--n-boot", type=int, default=10000)
        sp.add_argument("--n-perm", type=int, default=10000)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--trace-tool-src", help="trace-tool src dir (default $TRACE_TOOL_SRC / $TRACE_TOOL_ROOT/src)")
    e.add_argument("--trace-src", help="TRACE src dir for the adapter registry (default: this checkout's src)")
    args = p.parse_args(argv)
    try:
        return {"extras": cmd_extras, "traceval": cmd_traceval, "training": cmd_training, "report": cmd_report,
                "inputs": cmd_inputs}[args.cmd](args)
    except cc.CompareError as exc:
        print(f"final_cert3b: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
