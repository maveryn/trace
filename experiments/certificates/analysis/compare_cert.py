"""Paired comparison of certificate evaluation runs (DESIGN 8, statistics).

    python experiments/certificates/analysis/compare_cert.py \\
        --arm P="$TRACE_CERT_RUNS/eval/ckpt/p3b-s1" --arm G="$TRACE_CERT_RUNS/eval/ckpt/g3b-s1" \\
        --arm A="$TRACE_CERT_RUNS/eval/ckpt/a3b-s1" --arm S="$TRACE_CERT_RUNS/eval/ckpt/s3b" \\
        --out-json cmp.json --out-md cmp.md [--trace-tool-src "$TRACE_TOOL_ROOT/src"]

    several training seeds per arm:  --arm P@1=<root> --arm P@2=<root> --arm P@3=<root> --arm G@1=...

An ``--arm NAME[@TRAINING_SEED]=PATH`` names one evaluated model; PATH is a ``cert_scores.jsonl``
file or a directory searched recursively for them (an ``eval/{zs,ckpt}/<served>`` root), and the
option may repeat to add paths. Rows are filtered to one trace-tool condition (``--condition``,
default ``cert_v1``) and read by ``instance_id`` within each split.

Method (DESIGN 8, D14):

- **Pairing and decoding seeds.** Two arms are compared on the rows (split, instance_id) both
  scored. Each row's value of a metric is its mean over the decoding seeds present in both arms
  with a value (42/43/44 by default); the paired difference per row is X - Y.
- **Unit and CI.** Rows are grouped into ``scene_group`` clusters; the paired cluster bootstrap
  resamples clusters with replacement within each adapter stratum (10,000 resamples, seed 0) and
  recomputes the row-weighted mean difference (percentile 95% CI).
- **Tests.** Binary metrics (A, F, joint): exact McNemar on the decoding-seed-42 outcome
  (``trace_tool.compare.mcnemar_exact``). Continuous metrics (G, M, tokens): two-sided cluster
  sign-flip permutation test of the mean difference (10,000 flips, seed 0, p = (1 + #|T*| >= |T|) /
  (1 + n)); it is also reported for the binary metrics.
- **Families (Holm within each, ``trace_tool.compare.holm``).** primary: P vs G on A for e1_test
  and e2_test; secondary: P vs A and G vs A on the same splits. Everything else (other pairs,
  splits, metrics, adapters, cells) is descriptive and unadjusted. The external family (natural vs
  chart/diagram/synthetic macro) is not computed here: external scores are not cert_scores rows.
  The family comparisons run whenever arms named P, G, A (or mapped with ``--family-arms
  P=<arm>,G=<arm>,A=<arm>``) are given; other comparisons (A vs S for the SFT contribution, base vs
  legacy, ...) are added with ``--pairs X:Y,...``; with neither, every arm is compared with the first.
- **Training seeds.** With matched training seeds (P@1 with G@1, ...), the estimate is the mean of
  the per-seed paired differences and the CI a two-level bootstrap (resample training seeds, then
  clusters within each). A directional claim needs that CI to exclude 0 and at least 2 of 3 seeds
  (generally max(2, ceil(2K/3)) of K) to agree in sign; per-seed tests are Holm-adjusted within each
  seed. With one training seed the CI reflects instance sampling only and is labelled so.

Outputs a JSON document (every number) and a Markdown report (families, arm means, and per pair
per split: all rows, per adapter, per cell). trace-tool is imported from ``--trace-tool-src``, else
``$TRACE_TOOL_SRC``, else ``$TRACE_TOOL_ROOT/src``, else the installed package.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VERSION = "compare_cert-1.0.0"

# metric name -> (cert_scores field, binary outcome)
METRICS: Dict[str, Tuple[str, bool]] = {
    "A": ("A", True),
    "G": ("G", False),
    "M": ("M", False),
    "F": ("F", True),
    "joint": ("joint", True),
    "tokens": ("completion_tokens", False),
    "A_legacy": ("A_legacy", True),
    "G_shown": ("G_shown", False),
    "M_shown": ("M_shown", False),
    "within_2048": ("within_2048", True),
}
DEFAULT_METRICS = ("A", "G", "M", "F", "joint", "tokens")

# DESIGN 8 pre-registered families: (name, [(X, Y) meaning X - Y], splits, metrics)
FAMILIES: Tuple[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("primary", (("P", "G"),), ("e1_test", "e2_test"), ("A",)),
    ("secondary", (("P", "A"), ("G", "A")), ("e1_test", "e2_test"), ("A",)),
)

CompareFns = Tuple[Callable[[int, int], float], Callable[[List[float]], List[float]]]


class CompareError(ValueError):
    """Inconsistent or missing inputs (never silently turned into a result)."""


# ============================================================================== trace-tool


def load_trace_tool(src: Optional[str] = None) -> CompareFns:
    """``trace_tool.compare.mcnemar_exact`` and ``holm``; the source dir comes from the CLI or env."""
    root = os.environ.get("TRACE_TOOL_ROOT")
    for cand in (src, os.environ.get("TRACE_TOOL_SRC"), os.path.join(root, "src") if root else None):
        if cand:
            if not Path(cand, "trace_tool", "compare.py").is_file():
                raise CompareError(f"no trace_tool/compare.py under {cand}")
            if cand not in sys.path:
                sys.path.insert(0, cand)
            break
    try:
        mod = importlib.import_module("trace_tool.compare")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise CompareError(f"cannot import trace_tool.compare ({exc}); pass --trace-tool-src or set TRACE_TOOL_ROOT") from exc
    return mod.mcnemar_exact, mod.holm


# ============================================================================== loading


@dataclass(frozen=True)
class RowMeta:
    cluster: str
    stratum: str
    adapter: str
    cell: str
    ext_id: Optional[str]


@dataclass
class Run:
    """One evaluated model (arm, training seed): (split, instance_id) -> decoding seed -> values."""

    arm: str
    tseed: str
    files: List[str] = field(default_factory=list)
    rows: Dict[Tuple[str, str], Dict[int, np.ndarray]] = field(default_factory=dict)


@dataclass
class Dataset:
    metrics: Tuple[str, ...]
    runs: Dict[str, Dict[str, Run]] = field(default_factory=dict)  # arm -> tseed -> Run
    meta: Dict[Tuple[str, str], RowMeta] = field(default_factory=dict)
    skipped_condition: int = 0


ARM_RE = re.compile(r"^([A-Za-z0-9_.+-]+?)(?:@([A-Za-z0-9_.-]+))?=(.+)$")


def parse_arm(spec: str) -> Tuple[str, str, str]:
    m = ARM_RE.match(spec)
    if not m:
        raise CompareError(f"--arm must be NAME[@TRAINING_SEED]=PATH, got {spec!r}")
    return m.group(1), m.group(2) or "", m.group(3)


def find_score_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        found = sorted(p.rglob("cert_scores.jsonl"))
        if found:
            return found
    raise CompareError(f"no cert_scores.jsonl at {path}")


def _decoding_seed(row: Mapping[str, Any], path: Path) -> int:
    seed = row.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    cfg = path.parent.parent / "run_config.json"  # <run>/cert/cert_scores.jsonl
    if cfg.exists():
        seed = json.loads(cfg.read_text()).get("seed")
        if isinstance(seed, int):
            return seed
    for part in reversed(path.parts):
        m = re.fullmatch(r"seed(\d+)", part)
        if m:
            return int(m.group(1))
    raise CompareError(f"{path}: row {row.get('instance_id')} has no decoding seed")


def _value(row: Mapping[str, Any], fieldname: str) -> float:
    v = row.get(fieldname)
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return float("nan")


def load_dataset(
    arm_specs: Sequence[str],
    *,
    condition: str = "cert_v1",
    metrics: Sequence[str] = DEFAULT_METRICS,
    splits: Optional[Sequence[str]] = None,
    cluster_key: str = "scene_group",
    stratum_key: str = "adapter_id",
) -> Dataset:
    for m in metrics:
        if m not in METRICS:
            raise CompareError(f"unknown metric {m!r}; known: {sorted(METRICS)}")
    ds = Dataset(metrics=tuple(metrics))
    fields = [METRICS[m][0] for m in metrics]
    for spec in arm_specs:
        arm, tseed, path = parse_arm(spec)
        run = ds.runs.setdefault(arm, {}).setdefault(tseed, Run(arm, tseed))
        for f in find_score_files(path):
            if str(f) in run.files:
                raise CompareError(f"{f} is given twice for {arm}@{tseed or '-'}")
            run.files.append(str(f))
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("condition") != condition and row.get("cond") != condition:
                    ds.skipped_condition += 1
                    continue
                split, iid = row.get("split"), row.get("instance_id")
                if not split or not iid:
                    raise CompareError(f"{f}:{n + 1}: row without split or instance_id")
                if splits and split not in splits:
                    continue
                key = (str(split), str(iid))
                meta = RowMeta(
                    cluster=str(row.get(cluster_key) or iid),
                    stratum=str(row.get(stratum_key) or "-"),
                    adapter=str(row.get("adapter_id") or "-"),
                    cell=str(row.get("cell") or "-"),
                    ext_id=row.get("cert_extension_id"),
                )
                old = ds.meta.setdefault(key, meta)
                if old != meta:
                    raise CompareError(f"{f}: {key} disagrees with an earlier file ({old} vs {meta}): not the same frozen data")
                seed = _decoding_seed(row, f)
                per_seed = run.rows.setdefault(key, {})
                if seed in per_seed:
                    raise CompareError(f"{f}: {arm}@{tseed or '-'} has {key} seed {seed} twice")
                per_seed[seed] = np.array([_value(row, fl) for fl in fields], dtype=float)
    for arm, by_seed in ds.runs.items():
        for tseed, run in by_seed.items():
            if not run.rows:
                raise CompareError(f"{arm}@{tseed or '-'}: no rows of condition {condition!r}")
    return ds


# ============================================================================== pairing


@dataclass
class Paired:
    """Rows of one split scored by both runs: per-row seed means and seed-42 outcomes."""

    keys: List[Tuple[str, str]]
    X: np.ndarray  # (R, M) seed-averaged values of arm X (NaN: no common seed with a value)
    Y: np.ndarray
    X42: np.ndarray  # (R, M) decoding seed 42 (NaN: absent in either arm)
    Y42: np.ndarray
    cluster: np.ndarray
    stratum: np.ndarray
    adapter: np.ndarray
    cell: np.ndarray
    x_only: int
    y_only: int
    seeds: List[int]


def pair_runs(ds: Dataset, x: Run, y: Run, split: str, mcnemar_seed: int = 42) -> Paired:
    xk = {k for k in x.rows if k[0] == split}
    yk = {k for k in y.rows if k[0] == split}
    keys = sorted(xk & yk)
    M = len(ds.metrics)
    X = np.full((len(keys), M), np.nan)
    Y = np.full((len(keys), M), np.nan)
    X42 = np.full((len(keys), M), np.nan)
    Y42 = np.full((len(keys), M), np.nan)
    seeds_used = set()
    for r, k in enumerate(keys):
        xs, ys = x.rows[k], y.rows[k]
        common = sorted(set(xs) & set(ys))
        seeds_used.update(common)
        if common:
            xv = np.stack([xs[s] for s in common])
            yv = np.stack([ys[s] for s in common])
            ok = ~np.isnan(xv) & ~np.isnan(yv)
            cnt = ok.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                X[r] = np.where(cnt > 0, np.where(ok, xv, 0.0).sum(axis=0) / cnt, np.nan)
                Y[r] = np.where(cnt > 0, np.where(ok, yv, 0.0).sum(axis=0) / cnt, np.nan)
        if mcnemar_seed in xs and mcnemar_seed in ys:
            X42[r], Y42[r] = xs[mcnemar_seed], ys[mcnemar_seed]
    metas = [ds.meta[k] for k in keys]
    return Paired(
        keys=keys, X=X, Y=Y, X42=X42, Y42=Y42,
        cluster=np.array([m.cluster for m in metas], dtype=object),
        stratum=np.array([m.stratum for m in metas], dtype=object),
        adapter=np.array([m.adapter for m in metas], dtype=object),
        cell=np.array([m.cell for m in metas], dtype=object),
        x_only=len(xk - yk), y_only=len(yk - xk), seeds=sorted(seeds_used),
    )


def tseed_pairs(xruns: Mapping[str, Run], yruns: Mapping[str, Run]) -> List[Tuple[str, str]]:
    """Matched training seeds; a single run on one side is shared by every run of the other."""
    common = sorted(set(xruns) & set(yruns))
    if common and not (len(common) == 1 and common[0] == "" and (len(xruns) > 1 or len(yruns) > 1)):
        return [(t, t) for t in common]
    if len(yruns) == 1:
        (only,) = yruns
        return [(t, only) for t in sorted(xruns)]
    if len(xruns) == 1:
        (only,) = xruns
        return [(only, t) for t in sorted(yruns)]
    raise CompareError(f"training seeds {sorted(xruns)} and {sorted(yruns)} share none; tag runs as NAME@SEED")


# ============================================================================== statistics


@dataclass
class Clusters:
    S: np.ndarray  # (C, M) sum over the cluster's rows of the paired difference (valid rows only)
    N: np.ndarray  # (C, M) number of valid rows
    SX: np.ndarray
    SY: np.ndarray
    stratum: np.ndarray  # (C,)

    @property
    def n_rows(self) -> np.ndarray:
        return self.N.sum(axis=0)


def make_clusters(X: np.ndarray, Y: np.ndarray, cluster: np.ndarray, stratum: np.ndarray) -> Clusters:
    D = X - Y
    V = ~np.isnan(D)
    uniq, idx = np.unique(cluster.astype(str), return_inverse=True)
    C, M = len(uniq), X.shape[1]
    S, N, SX, SY = (np.zeros((C, M)) for _ in range(4))
    np.add.at(S, idx, np.where(V, D, 0.0))
    np.add.at(N, idx, V.astype(float))
    np.add.at(SX, idx, np.where(V, X, 0.0))
    np.add.at(SY, idx, np.where(V, Y, 0.0))
    cst = np.empty(C, dtype=object)
    for i, c in enumerate(idx):
        if cst[c] is None:
            cst[c] = stratum[i]
        elif cst[c] != stratum[i]:
            raise CompareError(f"cluster {uniq[c]} spans strata {cst[c]} and {stratum[i]}")
    return Clusters(S, N, SX, SY, cst)


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def cluster_bootstrap(cl: Clusters, n_boot: int, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
    """(n_boot, M) row-weighted mean differences, clusters resampled with replacement within strata."""
    M = cl.S.shape[1]
    tot_s = np.zeros((n_boot, M))
    tot_n = np.zeros((n_boot, M))
    for h in sorted(set(cl.stratum.tolist())):
        idx = np.flatnonzero(cl.stratum == h)
        k = len(idx)
        Sh, Nh = cl.S[idx], cl.N[idx]
        for b0 in range(0, n_boot, chunk):
            b1 = min(n_boot, b0 + chunk)
            counts = rng.multinomial(k, np.full(k, 1.0 / k), size=b1 - b0).astype(float)
            tot_s[b0:b1] += counts @ Sh
            tot_n[b0:b1] += counts @ Nh
    return _ratio(tot_s, tot_n)


def signflip_p(cl: Clusters, n_perm: int, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
    """(M,) two-sided p of the mean paired difference under cluster-wise sign flips."""
    n = cl.N.sum(axis=0)
    obs = np.abs(_ratio(cl.S.sum(axis=0), n))
    C, M = cl.S.shape
    exceed = np.zeros(M)
    for b0 in range(0, n_perm, chunk):
        b1 = min(n_perm, b0 + chunk)
        signs = rng.integers(0, 2, size=(b1 - b0, C)).astype(float) * 2.0 - 1.0
        t = np.abs(_ratio(signs @ cl.S, n[None, :]))
        exceed += (t >= obs[None, :] - 1e-12 * np.maximum(1.0, obs[None, :])).sum(axis=0)
    p = (1.0 + exceed) / (1.0 + n_perm)
    return np.where(n > 0, p, np.nan)


def mcnemar_counts(x42: np.ndarray, y42: np.ndarray) -> Tuple[int, int, int]:
    """(b, c, n): b = Y-only successes, c = X-only successes, over rows with both outcomes."""
    ok = ~np.isnan(x42) & ~np.isnan(y42)
    xs, ys = x42[ok] > 0.5, y42[ok] > 0.5
    return int(np.sum(ys & ~xs)), int(np.sum(xs & ~ys)), int(ok.sum())


def percentile_ci(boot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.full(boot.shape[1], np.nan)
    hi = np.full(boot.shape[1], np.nan)
    for m in range(boot.shape[1]):
        col = boot[:, m]
        col = col[~np.isnan(col)]
        if col.size:
            lo[m], hi[m] = np.percentile(col, [2.5, 97.5])
    return lo, hi


def _f(x: Any) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass
class Settings:
    n_boot: int = 10000
    n_perm: int = 10000
    seed: int = 0
    mcnemar_seed: int = 42
    alpha: float = 0.05


def single_seed_stats(
    ds: Dataset, p: Paired, mask: np.ndarray, st: Settings, mcnemar: Callable[[int, int], float]
) -> Tuple[Dict[str, Dict[str, Any]], Clusters]:
    cl = make_clusters(p.X[mask], p.Y[mask], p.cluster[mask], p.stratum[mask])
    n = cl.n_rows
    delta = _ratio(cl.S.sum(axis=0), n)
    mx, my = _ratio(cl.SX.sum(axis=0), n), _ratio(cl.SY.sum(axis=0), n)
    boot = cluster_bootstrap(cl, st.n_boot, np.random.default_rng([st.seed, 1]))
    lo, hi = percentile_ci(boot)
    perm = signflip_p(cl, st.n_perm, np.random.default_rng([st.seed, 2]))
    out: Dict[str, Dict[str, Any]] = {}
    for j, m in enumerate(ds.metrics):
        binary = METRICS[m][1]
        res: Dict[str, Any] = {
            "n_rows": int(n[j]),
            "n_clusters": int(np.sum(cl.N[:, j] > 0)),
            "mean_x": _f(mx[j]), "mean_y": _f(my[j]), "delta": _f(delta[j]),
            "ci": [_f(lo[j]), _f(hi[j])], "perm_p": _f(perm[j]), "mcnemar": None,
        }
        if binary:
            b, c, nn = mcnemar_counts(p.X42[mask][:, j], p.Y42[mask][:, j])
            if nn:
                res["mcnemar"] = {"seed": st.mcnemar_seed, "b_y_only": b, "c_x_only": c, "n": nn, "p": float(mcnemar(b, c))}
        if res["mcnemar"] is not None:
            res["test"], res["p"] = f"mcnemar_exact(seed {st.mcnemar_seed})", res["mcnemar"]["p"]
        else:
            res["test"], res["p"] = "cluster_signflip", res["perm_p"]
        out[m] = res
    return out, cl


def two_level_bootstrap(cls: Sequence[Clusters], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, M): mean over K resampled training seeds of a cluster-bootstrap estimate of each."""
    K, M = len(cls), cls[0].S.shape[1]
    pick = rng.integers(0, K, size=(n_boot, K))
    vals = np.full((n_boot, K, M), np.nan)
    for k in range(K):
        where = np.argwhere(pick == k)
        if len(where):
            vals[where[:, 0], where[:, 1]] = cluster_bootstrap(cls[k], len(where), rng)
    return vals.mean(axis=1)


def sign_needed(K: int) -> int:
    return max(2, math.ceil(2 * K / 3)) if K > 1 else 1


def compare_scope(
    ds: Dataset, paired: Sequence[Tuple[Tuple[str, str], Paired]], scope: Callable[[Paired], np.ndarray],
    st: Settings, mcnemar: Callable[[int, int], float],
) -> Dict[str, Dict[str, Any]]:
    per_seed, cls = [], []
    for tpair, p in paired:
        res, cl = single_seed_stats(ds, p, scope(p), st, mcnemar)
        per_seed.append((tpair, res))
        cls.append(cl)
    K = len(per_seed)
    if K == 1:
        out = per_seed[0][1]
        for res in out.values():
            res["ci_scope"] = "instance sampling only (1 training seed)"
            res["K"] = 1
        return out
    boot = two_level_bootstrap(cls, st.n_boot, np.random.default_rng([st.seed, 3]))
    lo, hi = percentile_ci(boot)
    out = {}
    for j, m in enumerate(ds.metrics):
        deltas = [r[m]["delta"] for _, r in per_seed]
        known = [d for d in deltas if d is not None]
        delta = float(np.mean(known)) if len(known) == K else None
        sgn = 0 if delta is None else (1 if delta > 0 else -1 if delta < 0 else 0)
        agree = sum(1 for d in known if sgn and d * sgn > 0)
        ci = [_f(lo[j]), _f(hi[j])]
        excludes = ci[0] is not None and ci[1] is not None and (ci[0] > 0 or ci[1] < 0)
        out[m] = {
            "K": K,
            "n_rows": int(max(r[m]["n_rows"] for _, r in per_seed)),  # per training seed (see per_seed)
            "n_clusters": int(max(r[m]["n_clusters"] for _, r in per_seed)),
            "mean_x": _f(np.mean([r[m]["mean_x"] for _, r in per_seed])) if all(r[m]["mean_x"] is not None for _, r in per_seed) else None,
            "mean_y": _f(np.mean([r[m]["mean_y"] for _, r in per_seed])) if all(r[m]["mean_y"] is not None for _, r in per_seed) else None,
            "delta": delta,
            "ci": ci,
            "ci_scope": f"two-level bootstrap over {K} training seeds and clusters",
            "sign_agree": agree,
            "sign_needed": sign_needed(K),
            "claim": bool(excludes and agree >= sign_needed(K)),
            "test": per_seed[0][1][m]["test"],
            "p": None,
            "per_seed": [{"tseeds": list(tp), **{k: v for k, v in r[m].items()}} for tp, r in per_seed],
        }
    return out


# ============================================================================== comparisons


def default_pairs(ds: Dataset, pairs: Sequence[Tuple[str, str]], alias: Mapping[str, str], preregistered: bool) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if preregistered:
        for _, fam_pairs, _, _ in FAMILIES:
            for x, y in fam_pairs:
                ax, ay = alias.get(x, x), alias.get(y, y)
                if ax in ds.runs and ay in ds.runs and (ax, ay) not in out:
                    out.append((ax, ay))
    for x, y in pairs:
        for a in (x, y):
            if a not in ds.runs:
                raise CompareError(f"--pairs names arm {a!r}, which was not given (arms: {sorted(ds.runs)})")
        if (x, y) not in out:
            out.append((x, y))
    if not out:
        arms = list(ds.runs)
        if len(arms) < 2:
            raise CompareError("need at least two arms")
        out = [(a, arms[0]) for a in arms[1:]]
    return out


def scopes_of(p: Paired, cells: bool) -> List[Tuple[str, Callable[[Paired], np.ndarray]]]:
    out: List[Tuple[str, Callable[[Paired], np.ndarray]]] = [("all", lambda q: np.ones(len(q.keys), dtype=bool))]
    for a in sorted(set(p.adapter.tolist())):
        out.append((f"adapter:{a}", lambda q, a=a: q.adapter == a))
    if cells:
        for a, c in sorted(set(zip(p.adapter.tolist(), p.cell.tolist()))):
            out.append((f"cell:{a}/{c}", lambda q, a=a, c=c: (q.adapter == a) & (q.cell == c)))
    return out


def run_comparisons(
    ds: Dataset, pairs: Sequence[Tuple[str, str]], st: Settings, mcnemar: Callable[[int, int], float], cells: bool = True,
) -> List[Dict[str, Any]]:
    splits = sorted({k[0] for k in ds.meta})
    comps = []
    for x, y in pairs:
        tps = tseed_pairs(ds.runs[x], ds.runs[y])
        comp: Dict[str, Any] = {"x": x, "y": y, "name": f"{x} - {y}", "training_seed_pairs": [list(t) for t in tps], "splits": {}}
        for split in splits:
            paired = [(tp, pair_runs(ds, ds.runs[x][tp[0]], ds.runs[y][tp[1]], split, st.mcnemar_seed)) for tp in tps]
            if any(len(p.keys) == 0 for _, p in paired):
                continue
            first = paired[0][1]
            entry: Dict[str, Any] = {
                "pairing": [{"tseeds": list(tp), "rows": len(p.keys), "x_only": p.x_only, "y_only": p.y_only,
                             "decoding_seeds": p.seeds} for tp, p in paired],
                "scopes": {},
            }
            for name, fn in scopes_of(first, cells):
                if any(not fn(p).any() for _, p in paired):
                    continue
                entry["scopes"][name] = compare_scope(ds, paired, fn, st, mcnemar)
            comp["splits"][split] = entry
        comps.append(comp)
    return comps


def evaluate_families(
    comps: Sequence[Mapping[str, Any]], alias: Mapping[str, str], st: Settings, holm: Callable[[List[float]], List[float]],
) -> List[Dict[str, Any]]:
    by_pair = {(c["x"], c["y"]): c for c in comps}
    fams = []
    for name, fam_pairs, splits, metrics in FAMILIES:
        tests = []
        for x, y in fam_pairs:
            ax, ay = alias.get(x, x), alias.get(y, y)
            comp = by_pair.get((ax, ay))
            for split in splits:
                for m in metrics:
                    res = (((comp or {}).get("splits", {}).get(split) or {}).get("scopes", {}).get("all") or {}).get(m)
                    tests.append({"comparison": f"{ax} - {ay}", "role": f"{x} - {y}", "split": split, "metric": m,
                                  "result": res, "missing": res is None})
        present = [t for t in tests if not t["missing"]]
        # Holm within the family, separately for every training-seed index (K = 1: one group)
        groups: Dict[str, List[Tuple[Dict[str, Any], Optional[float]]]] = defaultdict(list)
        for t in present:
            r = t["result"]
            if r.get("K", 1) == 1:
                groups["-"].append((t, r.get("p")))
            else:
                for i, ps in enumerate(r["per_seed"]):
                    groups[f"seed#{i}"].append((t, ps.get("p")))
        for g, items in groups.items():
            ps = [1.0 if p is None else float(p) for _, p in items]
            adj = holm(ps) if ps else []
            for (t, _), a in zip(items, adj):
                if g == "-":
                    t["holm_p"] = float(a)
                else:
                    t.setdefault("holm_p_per_seed", []).append(float(a))
        for t in present:
            r = t["result"]
            if r.get("K", 1) == 1:
                t["significant"] = bool(t.get("holm_p") is not None and t["holm_p"] < st.alpha)
                t["decision"] = (("X > Y" if (r["delta"] or 0) > 0 else "X < Y") if t["significant"] else "not significant")
            else:
                t["decision"] = (("X > Y" if (r["delta"] or 0) > 0 else "X < Y") if r["claim"] else "no claim")
        fams.append({"family": name, "pairs": [f"{x} - {y}" for x, y in fam_pairs], "splits": list(splits), "metrics": list(metrics),
                     "applicable": bool(present), "complete": len(present) == len(tests), "tests": tests})
    return fams


def arm_means(ds: Dataset) -> Dict[str, Any]:
    """Descriptive per (arm, training seed, split): seed-averaged row means of every metric."""
    out: Dict[str, Any] = {}
    for arm, by_seed in ds.runs.items():
        for tseed, run in by_seed.items():
            per_split: Dict[str, List[np.ndarray]] = defaultdict(list)
            seeds: Dict[str, set] = defaultdict(set)
            for (split, _), vals in run.rows.items():
                stack = np.stack(list(vals.values()))
                ok = ~np.isnan(stack)
                per_split[split].append(_ratio(np.where(ok, stack, 0.0).sum(axis=0), ok.sum(axis=0).astype(float)))
                seeds[split].update(vals)
            for split, rows in per_split.items():
                arr = np.stack(rows)
                means = {}
                for j, m in enumerate(ds.metrics):
                    col = arr[:, j][~np.isnan(arr[:, j])]
                    means[m] = _f(col.mean()) if col.size else None
                out.setdefault(arm, {}).setdefault(tseed or "-", {})[split] = {
                    "rows": len(rows), "decoding_seeds": sorted(seeds[split]), "means": means}
    return out


# ============================================================================== report


def _num(m: str, v: Optional[float], signed: bool = True) -> str:
    if v is None:
        return "-"
    if m == "tokens":
        return f"{v:+.1f}" if signed else f"{v:.1f}"
    return f"{100 * v:+.2f}" if signed else f"{100 * v:.2f}"


def _p(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3g}"


def _md(text: Any) -> str:
    """Table-safe text: cell names such as ``above|card_table`` contain the column separator."""
    return str(text).replace("|", "\\|")


def _tp(tseeds: Sequence[str]) -> str:
    return "/".join(t or "-" for t in tseeds)


def _cell(m: str, r: Optional[Mapping[str, Any]]) -> str:
    if not r or r.get("delta") is None:
        return "-"
    lo, hi = r["ci"]
    p = r.get("p")
    if p is None and r.get("per_seed"):
        tail = f" {r['sign_agree']}/{r['K']}" + (" *" if r.get("claim") else "")
    else:
        tail = f" p={_p(p)}"
    return f"{_num(m, r['delta'])} [{_num(m, lo)}, {_num(m, hi)}]{tail}"


def markdown(doc: Mapping[str, Any]) -> str:
    metrics = doc["config"]["metrics"]
    L = ["# Certificate comparison (DESIGN 8)", "",
         f"Condition `{doc['config']['condition']}`; cluster `{doc['config']['cluster_key']}` stratified by "
         f"`{doc['config']['stratum_key']}`; {doc['config']['n_boot']} bootstrap resamples and {doc['config']['n_perm']} "
         f"sign flips (seed {doc['config']['seed']}); McNemar on decoding seed {doc['config']['mcnemar_seed']}; "
         "Holm within pre-registered families only. Differences are X - Y in percentage points "
         "(tokens: completion tokens). Rows are means over the decoding seeds both arms have.", ""]
    L += ["## Inputs", "", "| arm | training seed | files | rows per split | decoding seeds |", "|---|---|---|---|---|"]
    for arm, by_seed in doc["arms"].items():
        for tseed, info in by_seed.items():
            rows = ", ".join(f"{s} {v['rows']}" for s, v in sorted(info["splits"].items()))
            seeds = sorted({x for v in info["splits"].values() for x in v["decoding_seeds"]})
            L.append(f"| {_md(arm)} | {_md(tseed)} | {len(info['files'])} | {rows} | {', '.join(map(str, seeds))} |")
    L += ["", "## Pre-registered families", "",
          "| family | comparison (role) | split | metric | K | Δ | 95% CI | CI scope | test | p | Holm p | sign agree | decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in doc["families"]:
        if not fam["applicable"]:
            L.append(f"| {fam['family']} | not applicable: no {', '.join(fam['pairs'])} comparison on {', '.join(fam['splits'])} "
                     "(name the arms P/G/A or map them with --family-arms) | | | | | | | | | | | |")
            continue
        for t in fam["tests"]:
            r = t["result"]
            if r is None:
                L.append(f"| {fam['family']} | {t['comparison']} ({t['role']}) | {t['split']} | {t['metric']} | - | - | - | - | - | - | - | - | missing |")
                continue
            m = t["metric"]
            if r.get("K", 1) == 1:
                p, hp, sa = _p(r.get("p")), _p(t.get("holm_p")), "-"
            else:
                p = ", ".join(_p(s.get("p")) for s in r["per_seed"])
                hp = ", ".join(_p(x) for x in t.get("holm_p_per_seed", []))
                sa = f"{r['sign_agree']}/{r['K']} (need {r['sign_needed']})"
            L.append(f"| {fam['family']} | {t['comparison']} ({t['role']}) | {t['split']} | {m} | {r.get('K', 1)} | {_num(m, r['delta'])} | "
                     f"[{_num(m, r['ci'][0])}, {_num(m, r['ci'][1])}] | {r['ci_scope']} | {r['test']} | {p} | {hp} | {sa} | {t['decision']} |")
        if not fam["complete"]:
            L.append(f"| {fam['family']} | (incomplete: missing arms or splits) | | | | | | | | | | | |")
    L += ["", "## Arm means (descriptive; % except tokens)", ""]
    splits = sorted({s for by in doc["arm_means"].values() for t in by.values() for s in t})
    for split in splits:
        L += [f"### {split}", "", "| arm | training seed | rows | " + " | ".join(metrics) + " |", "|---|---|---|" + "---|" * len(metrics)]
        for arm, by in doc["arm_means"].items():
            for tseed, per in by.items():
                if split in per:
                    v = per[split]
                    L.append(f"| {_md(arm)} | {_md(tseed)} | {v['rows']} | " + " | ".join(_num(m, v["means"][m], signed=False) for m in metrics) + " |")
        L.append("")
    for comp in doc["comparisons"]:
        L += [f"## {comp['name']}", "", f"Training-seed pairs (X/Y, - = untagged): {', '.join(_tp(t) for t in comp['training_seed_pairs'])}. "
              "Cells: Δ [95% CI] and p (McNemar for A/F/joint, sign-flip otherwise); with several training seeds, "
              "sign agreement k/K and * when the claim rule holds. Descriptive except the family rows above.", ""]
        for split, entry in comp["splits"].items():
            pair_txt = "; ".join(f"{_tp(pp['tseeds'])}: {pp['rows']} paired rows (x only {pp['x_only']}, y only {pp['y_only']}), "
                                 f"decoding seeds {pp['decoding_seeds']}" for pp in entry["pairing"])
            L += [f"### {split}", "", pair_txt, ""]
            for kind, title in (("all", "all rows and per adapter"), ("cell:", "per cell")):
                names = [n for n in entry["scopes"] if (n == "all" or n.startswith("adapter:")) == (kind == "all")]
                if not names:
                    continue
                L += [f"{title}:", "", "| scope | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
                for n in names:
                    res = entry["scopes"][n]
                    nrows = max((res[m]["n_rows"] for m in metrics if m in res), default=0)
                    label = n.split(":", 1)[-1]
                    L.append(f"| {_md(label)} | {nrows} | " + " | ".join(_cell(m, res.get(m)) for m in metrics) + " |")
                L.append("")
    return "\n".join(L) + "\n"


# ============================================================================== main


def build(args: argparse.Namespace) -> Dict[str, Any]:
    mcnemar, holm = load_trace_tool(args.trace_tool_src)
    metrics = tuple(m for m in args.metrics.split(",") if m)
    splits = [s for s in args.splits.split(",") if s] if args.splits else None
    ds = load_dataset(args.arm, condition=args.condition, metrics=metrics, splits=splits,
                      cluster_key=args.cluster_key, stratum_key=args.stratum_key)
    alias = dict(kv.split("=", 1) for kv in args.family_arms.split(",") if kv) if args.family_arms else {}
    pairs = [tuple(p.split(":", 1)) for p in args.pairs.split(",") if p] if args.pairs else []
    st = Settings(n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed, mcnemar_seed=args.mcnemar_seed, alpha=args.alpha)
    chosen = default_pairs(ds, pairs, alias, not args.no_preregistered)
    comps = run_comparisons(ds, chosen, st, mcnemar, cells=not args.no_cells)
    fams = evaluate_families(comps, alias, st, holm) if not args.no_preregistered else []
    means = arm_means(ds)
    arms = {arm: {(t or "-"): {"files": run.files,
                               "splits": {s: {"rows": v["rows"], "decoding_seeds": v["decoding_seeds"]}
                                          for s, v in means[arm][t or "-"].items()}}
                  for t, run in by.items()} for arm, by in ds.runs.items()}
    return {
        "version": VERSION,
        "config": {"condition": args.condition, "metrics": list(metrics), "splits": splits, "n_boot": st.n_boot,
                   "n_perm": st.n_perm, "seed": st.seed, "mcnemar_seed": st.mcnemar_seed, "alpha": st.alpha,
                   "cluster_key": args.cluster_key, "stratum_key": args.stratum_key, "family_arms": alias,
                   "preregistered": not args.no_preregistered, "rows_skipped_other_condition": ds.skipped_condition},
        "arms": arms,
        "arm_means": means,
        "families": fams,
        "comparisons": comps,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, help="NAME[@TRAINING_SEED]=PATH (repeat; PATH file or dir)")
    p.add_argument("--condition", default="cert_v1", help="trace-tool condition (or public cond) of the rows to compare")
    p.add_argument("--splits", help="comma list (default: every split present)")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help=f"comma list from {sorted(METRICS)}")
    p.add_argument("--pairs", help="extra comparisons X:Y (X - Y), comma separated")
    p.add_argument("--family-arms", help="map the family roles to arm names, e.g. P=p3b,G=g3b,A=a3b")
    p.add_argument("--no-preregistered", action="store_true", help="no DESIGN 8 families (only --pairs, descriptive)")
    p.add_argument("--no-cells", action="store_true", help="skip the per-cell scopes")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mcnemar-seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--cluster-key", default="scene_group")
    p.add_argument("--stratum-key", default="adapter_id")
    p.add_argument("--trace-tool-src", help="trace-tool's src dir (default $TRACE_TOOL_SRC or $TRACE_TOOL_ROOT/src)")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md")
    args = p.parse_args(argv)
    try:
        doc = build(args)
    except CompareError as exc:
        print(f"compare_cert: {exc}", file=sys.stderr)
        return 2
    Path(args.out_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    md = markdown(doc)
    if args.out_md:
        Path(args.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
