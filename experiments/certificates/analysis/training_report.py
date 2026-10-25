"""Training dynamics, compute accounting, matching audit and go/no-go 1 (DESIGN 7.3, 8; RQ3 tokens).

    nice -n 10 $TRACE_CERT_PYTHON experiments/certificates/analysis/training_report.py \\
        --runs-root "$TRACE_CERT_RUNS" --scan "$OUT/rollout_scan.json" --out-dir "$OUT"

``--scan`` is the output of ``training_rollout_scan.py`` (per-rollout reward records of every RL
run). Everything else is read from the raw run directories under ``--runs-root``:

1. **E1-dev step curves.** For every RL run (``rl<size>-<arm>-s<seed>-<step>`` under
   ``eval/ckpt``) and its SFT start (``cert-sft-<size>-s564``, step 0), the cert_v1 E1-dev
   scores of decoding seeds 42-44 (trace preset, temperature 0.6). Per row a metric is its mean
   over the decoding seeds; a model's value is the mean over the 512 rows. Differences (step vs S,
   P - G, P - A, G - A within one training seed) use the paired cluster bootstrap of DESIGN 8
   (``scene_group`` clusters resampled within adapter strata, 10,000 resamples, seed 0, percentile
   95% CI). Tests follow DESIGN 8 / ``compare_cert.py``: binary metrics (A, F, joint) get the
   exact McNemar test on the decoding-seed-42 outcome (``trace_tool.compare.mcnemar_exact``);
   continuous metrics (G, M, tokens) the cluster sign-flip permutation p, which is also kept for the
   binary metrics as a secondary statistic. Across the three 3B training seeds the estimate is the
   mean of per-seed differences with two CIs: the ``compare_cert.py`` two-level bootstrap (resample
   training seeds, then clusters independently within each seed) and a shared-instance two-level
   bootstrap (resample training seeds, and one cluster resample applied to every seed). The second
   keeps the instance-sampling term that the seeds share: all seeds are evaluated on the same 512
   rows, and a step-vs-S difference also shares the S baseline itself, which independent per-seed
   resampling averages down by the number of seeds. Plus the sign-agreement count. E1-dev is the RL
   ``val_files`` split, so these curves are descriptive only (no family, no Holm).
2. **RL reward curves** from ``rl/<size>/<arm>/s<seed>/ckpt/experiment_log.jsonl``: overall
   (the arm's own reward), A, G, M, F, joint, response length, GRPO zero/perfect-solve rates;
   first-10 vs last-10 step means; the E1-dev validation records (every 50 steps, temperature 0.6,
   n = 1); best validation step; EasyR1's checkpoint-tracker best step, checked against its
   save-before-validate order (the save at step s is credited with the latest validation before
   s, so the tracker's best is always 25 steps after the best validation and never step 300); and
   from the scan the share of zero-variance prompt groups and every reward formula evaluated on the
   same rollouts.
3. **Compute**: GPU type x N, wall hours, s/step, GPU-hours, rollout tokens (exact, from the
   reward records), prompt tokens, host RAM and GPU memory peaks, for RL (run receipts, logs) and
   SFT (receipts, train logs); SFT dev curves and the checkpoint selection.
4. **Matching audit** (DESIGN 7.3): the arms of one training seed must share the resolved EasyR1
   config (except the arm keys), SFT init, data, steps, batch, rollouts, hardware class and data
   order (per-step prompt multiset). Every difference is listed.
5. **Go/no-go 1** (DESIGN 8): (a) on the S checkpoint's greedy E1-dev outputs, numeric adapters
   only, the share of rows with G >= 0.9 and M < 1 (threshold 10%) and Spearman(G, M); (b) the
   corruption-suite result recorded by ``section5_cpu.py``.

Writes ``training.json`` (every number), ``training.md`` (rendered from the JSON file only),
CSV tables and SVG figures. Missing inputs (evaluations or merges still running) are listed
under ``pending`` and skipped, so the script re-runs unchanged when they land.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VERSION = "training_report-1.1.0"

ARM_CODE = {"A": "a", "G": "g", "P": "p", "P_gated": "pgated", "G_gated": "ggated"}
EXPECTED_RUNS: Tuple[Tuple[str, str, int], ...] = (
    ("3b", "A", 1), ("3b", "G", 1), ("3b", "P", 1),
    ("3b", "A", 2), ("3b", "G", 2), ("3b", "P", 2),
    ("3b", "A", 3), ("3b", "G", 3), ("3b", "P", 3),
    ("3b", "P_gated", 1), ("3b", "G_gated", 1),
    ("7b", "G", 1), ("7b", "P", 1),
)
# matched sets of DESIGN 7.3 (one training seed, one hardware class, one SFT init)
MATCH_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, str, int], ...]], ...] = (
    ("3b seed 1 (A/G/P)", (("3b", "A", 1), ("3b", "G", 1), ("3b", "P", 1))),
    ("3b seed 2 (A/G/P)", (("3b", "A", 2), ("3b", "G", 2), ("3b", "P", 2))),
    ("3b seed 3 (A/G/P)", (("3b", "A", 3), ("3b", "G", 3), ("3b", "P", 3))),
    ("3b seed 1 gated (P_gated/G_gated)", (("3b", "P_gated", 1), ("3b", "G_gated", 1))),
    ("7b seed 1 (G/P)", (("7b", "G", 1), ("7b", "P", 1))),
)
SFT_TAG = {"3b": "qwen2.5-vl-3b", "7b": "qwen2.5-vl-7b"}
S_SERVED = {"3b": "cert-sft-3b-s564", "7b": "cert-sft-7b-s564"}
STEPS = (100, 200, 300)
DEC_SEEDS = (42, 43, 44)
METRICS = ("A", "G", "M", "F", "joint", "tokens")
BINARY = {"A", "F", "joint"}
FIELD = {"A": "A", "G": "G", "M": "M", "F": "F", "joint": "joint", "tokens": "completion_tokens"}
NUMERIC_ADAPTERS = ("filtered_mean.table_filtered_column_mean", "weighted_sum.lever_side_torque", "path.rpg_movement_cost")
ADAPTER_SHORT = {
    "selected_set.chess_piece_count": "chess",
    "filtered_mean.table_filtered_column_mean": "table",
    "weighted_sum.lever_side_torque": "lever",
    "path.rpg_movement_cost": "rpg",
}
LOG_KEYS = {  # experiment_log reward key -> report name
    "overall": "overall", "accuracy": "A", "grounding": "G", "semantic": "M", "cert_format": "F", "joint": "joint",
}
CONFIG_ARM_KEYS = {
    "trainer.experiment_name", "trainer.save_checkpoint_path",
    "worker.reward.reward_function_kwargs.arm", "worker.reward.reward_function_kwargs.details_dir",
}
EXTERNAL_PENDING_NOT_USED = [
    "external benchmark evaluations (ext_eval/runs, trace_eval_v1 full and q25, including the q25 runs of the step-100/200 checkpoints): "
    "not an input of this analysis, which reads only cert_v1 E1-dev scores, training logs and receipts",
    "base / legacy TRACE ckpt_eval models (eval/ckpt/base-*): not an input of this analysis",
]
MCNEMAR_SEED = 42


class ReportError(RuntimeError):
    pass


# ============================================================================== small helpers


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_time(text: Optional[str]) -> Optional[dt.datetime]:
    if not text:
        return None
    t = text.strip()
    if re.search(r"[+-]\d{4}$", t):
        t = t[:-2] + ":" + t[-2:]
    return dt.datetime.fromisoformat(t)


def hours_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    ta, tb = parse_time(a), parse_time(b)
    if ta is None or tb is None:
        return None
    return (tb - ta).total_seconds() / 3600.0


def fnum(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def get(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def served_name(size: str, arm: str, seed: int, step: int) -> str:
    return f"rl{size}-{ARM_CODE[arm]}-s{seed}-{step}"


def run_key(size: str, arm: str, seed: int) -> str:
    return f"{size}/{arm}/s{seed}"


# ============================================================================== statistics (DESIGN 8)


class Clusters:
    def __init__(self, D: np.ndarray, cluster: Sequence[str], stratum: Sequence[str]) -> None:
        uniq, idx = np.unique(np.asarray(cluster, dtype=str), return_inverse=True)
        C, M = len(uniq), D.shape[1]
        V = ~np.isnan(D)
        self.ids = uniq
        self.S = np.zeros((C, M))
        self.N = np.zeros((C, M))
        np.add.at(self.S, idx, np.where(V, D, 0.0))
        np.add.at(self.N, idx, V.astype(float))
        st = np.empty(C, dtype=object)
        for i, c in enumerate(idx):
            if st[c] is None:
                st[c] = stratum[i]
            elif st[c] != stratum[i]:
                raise ReportError(f"cluster {uniq[c]} spans strata")
        self.stratum = st

    def estimate(self) -> np.ndarray:
        n = self.N.sum(axis=0)
        return np.where(n > 0, self.S.sum(axis=0) / np.where(n > 0, n, 1), np.nan)


def cluster_bootstrap(cl: Clusters, n_boot: int, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
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
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot_n > 0, tot_s / np.where(tot_n > 0, tot_n, 1), np.nan)


def signflip_p(cl: Clusters, n_perm: int, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
    n = cl.N.sum(axis=0)
    obs = np.abs(cl.estimate())
    C, M = cl.S.shape
    exceed = np.zeros(M)
    for b0 in range(0, n_perm, chunk):
        b1 = min(n_perm, b0 + chunk)
        signs = rng.integers(0, 2, size=(b1 - b0, C)).astype(float) * 2.0 - 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.abs((signs @ cl.S) / np.where(n > 0, n, 1)[None, :])
        exceed += (t >= obs[None, :] - 1e-12 * np.maximum(1.0, obs[None, :])).sum(axis=0)
    return np.where(n > 0, (1.0 + exceed) / (1.0 + n_perm), np.nan)


def two_level_bootstrap(cls: Sequence[Clusters], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    K, M = len(cls), cls[0].S.shape[1]
    pick = rng.integers(0, K, size=(n_boot, K))
    vals = np.full((n_boot, K, M), np.nan)
    for k in range(K):
        where = np.argwhere(pick == k)
        if len(where):
            vals[where[:, 0], where[:, 1]] = cluster_bootstrap(cls[k], len(where), rng)
    return np.nanmean(vals, axis=1)


def shared_instance_bootstrap(cls: Sequence[Clusters], n_boot: int, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
    """(n_boot, M): training seeds resampled with replacement, and ONE stratified cluster resample per draw applied to
    every seed (the seeds are evaluated on the same instances, and a step-vs-S difference also shares S itself).
    ``two_level_bootstrap`` (the compare_cert.py method) instead resamples clusters independently within each seed,
    which averages the shared instance-sampling term down by the number of seeds."""
    K, M = len(cls), cls[0].S.shape[1]
    ref = cls[0]
    for cl in cls[1:]:
        if cl.ids.shape != ref.ids.shape or not np.array_equal(cl.ids, ref.ids) or not np.array_equal(cl.stratum, ref.stratum):
            raise ReportError("shared-instance bootstrap needs the same clusters (and strata) in every training seed")
    est = np.zeros((n_boot, K, M))
    for b0 in range(0, n_boot, chunk):
        b1 = min(n_boot, b0 + chunk)
        tot_s = np.zeros((b1 - b0, K, M))
        tot_n = np.zeros((b1 - b0, K, M))
        for h in sorted(set(ref.stratum.tolist())):
            idx = np.flatnonzero(ref.stratum == h)
            k = len(idx)
            counts = rng.multinomial(k, np.full(k, 1.0 / k), size=b1 - b0).astype(float)
            for j, cl in enumerate(cls):
                tot_s[:, j] += counts @ cl.S[idx]
                tot_n[:, j] += counts @ cl.N[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            est[b0:b1] = np.where(tot_n > 0, tot_s / np.where(tot_n > 0, tot_n, 1), np.nan)
    pick = rng.integers(0, K, size=(n_boot, K))
    rows = np.arange(n_boot)[:, None]
    return np.nanmean(est[rows, pick], axis=1)


def load_mcnemar() -> Tuple[Callable[[int, int], float], str]:
    """``trace_tool.compare.mcnemar_exact`` as compare_cert.py loads it; an identical local copy if it cannot be imported."""
    root = os.environ.get("TRACE_TOOL_ROOT")
    for cand in (os.environ.get("TRACE_TOOL_SRC"), os.path.join(root, "src") if root else None):
        if cand and Path(cand, "trace_tool", "compare.py").is_file():
            if cand not in sys.path:
                sys.path.insert(0, cand)
            break
    try:
        import importlib  # noqa: WPS433

        mod = importlib.import_module("trace_tool.compare")
        return mod.mcnemar_exact, f"trace_tool.compare.mcnemar_exact ({mod.__file__})"
    except ImportError:
        def mcnemar_exact(b: int, c: int) -> float:
            n = b + c
            if n == 0:
                return 1.0
            k = min(b, c)
            return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

        return mcnemar_exact, "local copy of trace_tool.compare.mcnemar_exact"


MCNEMAR, MCNEMAR_SOURCE = load_mcnemar()


def mcnemar_counts(x42: np.ndarray, y42: np.ndarray) -> Tuple[int, int, int]:
    """(b, c, n) as compare_cert.py: b = Y-only successes, c = X-only successes, over rows with both outcomes."""
    ok = ~np.isnan(x42) & ~np.isnan(y42)
    xs, ys = x42[ok] > 0.5, y42[ok] > 0.5
    return int(np.sum(ys & ~xs)), int(np.sum(xs & ~ys)), int(ok.sum())


def pct_ci(boot: np.ndarray) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    lo, hi = [], []
    for m in range(boot.shape[1]):
        col = boot[:, m]
        col = col[~np.isnan(col)]
        if col.size:
            a, b = np.percentile(col, [2.5, 97.5])
            lo.append(float(a))
            hi.append(float(b))
        else:
            lo.append(None)
            hi.append(None)
    return lo, hi


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ============================================================================== E1-dev evaluations


class EvalSet:
    """cert_v1 E1-dev rows of one evaluated model: per instance, the mean over decoding seeds."""

    def __init__(self, served: str, root: Path) -> None:
        self.served = served
        self.root = root
        self.ok = False
        self.status = "missing"
        self.problems: List[str] = []
        self.ids: List[str] = []
        self.X = np.zeros((0, len(METRICS)))
        self.per_seed: Dict[int, np.ndarray] = {}
        self.meta: Dict[str, Tuple[str, str]] = {}
        self.error_class: Dict[int, Counter] = {}
        self.files: Dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            self.status = "pending (no eval dir)"
            return
        val = self.root / "validation.json"
        if not val.is_file():
            self.status = "pending (no validation.json)"
        else:
            v = read_json(val)
            if not v.get("ok"):
                self.status = "pending (validation not ok)"
        rows_by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for seed in DEC_SEEDS:
            path = self.root / "e1_dev" / "cert_v1" / f"seed{seed}" / "cert" / "cert_scores.jsonl"
            if not path.is_file():
                self.problems.append(f"missing {path}")
                continue
            rows = read_jsonl(path)
            self.files[seed] = str(path)
            got: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                if r.get("split") != "e1_dev" or r.get("condition") != "cert_v1":
                    continue
                if int(r.get("seed", seed)) != seed:
                    self.problems.append(f"seed field {r.get('seed')} in {path}")
                got[r["instance_id"]] = r
            rows_by_seed[seed] = got
        if self.problems or self.status.startswith("pending"):
            if not self.status.startswith("pending"):
                self.status = "pending (missing decoding seeds)"
            return
        ids = sorted(set.intersection(*[set(v) for v in rows_by_seed.values()]))
        if any(len(v) != len(ids) for v in rows_by_seed.values()):
            self.problems.append("decoding seeds cover different instance sets")
        self.ids = ids
        arr = np.full((len(DEC_SEEDS), len(ids), len(METRICS)), np.nan)
        for si, seed in enumerate(DEC_SEEDS):
            rows = rows_by_seed[seed]
            cnt: Counter = Counter()
            for i, iid in enumerate(ids):
                r = rows[iid]
                for j, m in enumerate(METRICS):
                    arr[si, i, j] = fnum(r.get(FIELD[m])) if fnum(r.get(FIELD[m])) is not None else np.nan
                cnt[str(r.get("error_class"))] += 1
                if iid not in self.meta:
                    self.meta[iid] = (str(r["adapter_id"]), str(r.get("scene_group") or iid))
            self.per_seed[seed] = arr[si]
            self.error_class[seed] = cnt
        self.X = np.nanmean(arr, axis=0)
        self.ok = True
        self.status = "ok"

    def adapters(self) -> List[str]:
        return [self.meta[i][0] for i in self.ids]

    def clusters(self) -> List[str]:
        return [self.meta[i][1] for i in self.ids]

    def summary(self) -> Dict[str, Any]:
        if not self.ok:
            return {"served": self.served, "status": self.status, "problems": self.problems}
        ad = np.array(self.adapters())
        out: Dict[str, Any] = {
            "served": self.served, "status": self.status, "n_rows": len(self.ids), "decoding_seeds": list(DEC_SEEDS),
            "files": {str(k): v for k, v in self.files.items()},
            "metrics": {m: fnum(np.nanmean(self.X[:, j])) for j, m in enumerate(METRICS)},
            "per_decoding_seed": {str(s): {m: fnum(np.nanmean(self.per_seed[s][:, j])) for j, m in enumerate(METRICS)} for s in DEC_SEEDS},
            "by_adapter": {},
            "error_class_share": {},
        }
        for a in sorted(set(ad.tolist())):
            mask = ad == a
            out["by_adapter"][ADAPTER_SHORT.get(a, a)] = {"n": int(mask.sum()), **{m: fnum(np.nanmean(self.X[mask, j])) for j, m in enumerate(METRICS)}}
        tot: Counter = Counter()
        for c in self.error_class.values():
            tot.update(c)
        n = sum(tot.values())
        out["error_class_share"] = {k: v / n for k, v in sorted(tot.items())}
        return out


def paired_diff(x: EvalSet, y: EvalSet, n_boot: int, seed: int) -> Tuple[Dict[str, Any], Optional[Clusters]]:
    common = sorted(set(x.ids) & set(y.ids))
    if not common:
        return {"status": "no common rows"}, None
    xi = {iid: k for k, iid in enumerate(x.ids)}
    yi = {iid: k for k, iid in enumerate(y.ids)}
    X = x.X[[xi[i] for i in common]]
    Y = y.X[[yi[i] for i in common]]
    cl = Clusters(X - Y, [x.meta[i][1] for i in common], [x.meta[i][0] for i in common])
    est = cl.estimate()
    boot = cluster_bootstrap(cl, n_boot, np.random.default_rng([seed, 1]))
    lo, hi = pct_ci(boot)
    perm = signflip_p(cl, n_boot, np.random.default_rng([seed, 2]))
    res = {"n_rows": len(common), "metrics": {}}
    X42 = x.per_seed[MCNEMAR_SEED][[xi[i] for i in common]] if MCNEMAR_SEED in x.per_seed else None
    Y42 = y.per_seed[MCNEMAR_SEED][[yi[i] for i in common]] if MCNEMAR_SEED in y.per_seed else None
    for j, m in enumerate(METRICS):
        rec = {
            "x": fnum(np.nanmean(X[:, j])), "y": fnum(np.nanmean(Y[:, j])), "delta": fnum(est[j]),
            "ci": [lo[j], hi[j]], "perm_p": fnum(perm[j]), "mcnemar": None,
        }
        if m in BINARY and X42 is not None and Y42 is not None:
            b, c, nn = mcnemar_counts(X42[:, j], Y42[:, j])
            if nn:
                rec["mcnemar"] = {"seed": MCNEMAR_SEED, "b_y_only": b, "c_x_only": c, "n": nn, "p": float(MCNEMAR(b, c))}
        if rec["mcnemar"] is not None:
            rec["test"], rec["p"] = f"mcnemar_exact(seed {MCNEMAR_SEED})", rec["mcnemar"]["p"]
        else:
            rec["test"], rec["p"] = "cluster_signflip", rec["perm_p"]
        res["metrics"][m] = rec
    return res, cl


# ============================================================================== RL runs


def load_log(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "ckpt" / "experiment_log.jsonl"
    if not path.is_file():
        return []
    rows = read_jsonl(path)
    by_step: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        s = int(r["step"])
        if s in by_step:  # a resumed run may log a step twice: keep the last record, merging val
            merged = dict(by_step[s])
            merged.update(r)
            by_step[s] = merged
        else:
            by_step[s] = r
    return [by_step[s] for s in sorted(by_step)]


def window_means(rows: Sequence[Mapping[str, Any]], keyfn: Callable[[Mapping[str, Any]], Optional[float]], n: int = 10) -> Dict[str, Any]:
    vals = [keyfn(r) for r in rows]
    first = mean(vals[:n])
    last = mean(vals[-n:])
    return {"first10": first, "last10": last, "delta": (last - first) if first is not None and last is not None else None,
            "mean_all": mean(vals)}


def reward_curve_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys: Dict[str, Callable[[Mapping[str, Any]], Optional[float]]] = {}
    for k, name in LOG_KEYS.items():
        keys[name] = (lambda kk: (lambda r: fnum(get(r, f"reward.{kk}"))))(k)
    keys["resp_len"] = lambda r: fnum(get(r, "response_length.mean"))
    keys["zero_solve_rate"] = lambda r: fnum(get(r, "rlvr_stats.zero_solve_rate"))
    keys["perfect_solve_rate"] = lambda r: fnum(get(r, "rlvr_stats.perfect_solve_rate"))
    keys["entropy_loss"] = lambda r: fnum(get(r, "actor.entropy_loss"))
    keys["grad_norm"] = lambda r: fnum(get(r, "actor.grad_norm"))
    keys["pg_clipfrac_higher"] = lambda r: fnum(get(r, "actor.pg_clipfrac_higher"))
    return {name: window_means(rows, fn) for name, fn in keys.items()}


def val_records(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        v = r.get("val")
        if not v:
            continue
        rew = v.get("reward") or {}
        rec = {"step": int(r["step"])}
        for k, name in LOG_KEYS.items():
            rec[name] = fnum(rew.get(k))
        rec["reward_score"] = fnum(v.get("reward_score"))
        rec["resp_len"] = fnum(get(r, "val_response_length.mean"))
        out.append(rec)
    return out


def best_of(vals: Sequence[Mapping[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    cand = [v for v in vals if v.get(key) is not None]
    if not cand:
        return None
    best = max(cand, key=lambda v: (v[key], -v["step"]))  # ties: earliest step
    return {"step": best["step"], "value": best[key]}


def tracker_check(vals: Sequence[Mapping[str, Any]], tracker: Mapping[str, Any], save_freq: Any, val_freq: Any,
                  resumed: Sequence[Any]) -> Dict[str, Any]:
    """EasyR1 ray_trainer.fit saves the checkpoint of step s BEFORE it validates step s, and _save_checkpoint credits the save with
    the latest validation score (strict '>' keeps the earliest of ties). With save_freq 25 and val_freq 50 the save at s + 25
    carries the validation of step s, the save at step 300 carries step 250 (already credited at 275), so the tracker's best is
    the best validation among steps < max step, shifted by +25, and can never be the last step."""
    sf, vf = int(save_freq or 0), int(val_freq or 0)
    recs = [v for v in vals if fnum(v.get("reward_score")) is not None]
    if not recs or not sf or not tracker or any(r not in (None, 0) for r in resumed):
        return {"status": "not checked", "reason": "no validation records, tracker or save_freq, or a resumed run"}
    last = max(int(v["step"]) for v in recs)
    score = {int(v["step"]): float(v["reward_score"]) for v in recs}
    best_step, best = None, -1.0
    cur = 0.0  # val_reward_score before the first validation (val_before_train False)
    for st in range(sf, last + 1, sf):
        prior = [x for x in score if x < st]
        cur = score[max(prior)] if prior else 0.0
        if cur > best:
            best, best_step = cur, st
    tb = tracker.get("best_global_step")
    tv = fnum(tracker.get("best_val_reward_score"))
    own = best_of(vals, "reward_score")
    return {
        "status": "ok" if (tb == best_step and tv is not None and abs(tv - round(best, 4)) < 1e-9) else "MISMATCH",
        "expected_best_global_step": best_step, "expected_best_val_reward_score": round(best, 4),
        "credited_validation_step": max(x for x in score if x < best_step) if any(x < best_step for x in score) else None,
        "tracker_can_be_last_step": False,
        "best_validation_step_incl_last": (own or {}).get("step"), "best_validation_score_incl_last": (own or {}).get("value"),
        "last_step_is_best_validation": bool(own and own["step"] == last),
        "save_freq": sf, "val_freq": vf,
    }


def tsv_rows(path: Path) -> List[List[str]]:
    if not path.is_file():
        return []
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def host_ram(run_dir: Path) -> Dict[str, Any]:
    used, peak, limit, n = None, None, None, 0
    for path in sorted((run_dir / "logs").glob("host_mem.attempt*.tsv")):
        for parts in tsv_rows(path):
            if len(parts) != 4:
                continue
            n += 1
            u, p, lim = fnum(parts[1]), fnum(parts[2]), fnum(parts[3])
            if u is not None:
                used = u if used is None else max(used, u)
            if p is not None:
                peak = p if peak is None else max(peak, p)
            if lim is not None:
                limit = lim
    return {"unreclaimable_used_gb_max": used, "cgroup_peak_gb_max": peak, "limit_gb": limit, "samples": n}


def gpu_mem(run_dir: Path) -> Dict[str, Any]:
    peak: Dict[str, int] = {}
    util: Dict[str, List[float]] = defaultdict(list)
    for path in sorted((run_dir / "logs").glob("gpu_mem.attempt*.tsv")):
        for parts in tsv_rows(path):
            if len(parts) != 4:
                continue
            try:
                peak[parts[1]] = max(peak.get(parts[1], 0), int(parts[2]))
                util[parts[1]].append(float(parts[3]))
            except ValueError:
                continue
    return {"nvidia_smi_peak_mib": peak, "mean_util_pct": {k: (sum(v) / len(v) if v else None) for k, v in util.items()}}


def scan_calls(scan: Mapping[str, Any], key: str, phase: str) -> List[Tuple[int, Dict[str, Any]]]:
    run = (scan.get("runs") or {}).get(key)
    if not run:
        return []
    calls = run["calls"].get(phase, {})
    out = []
    for c, rec in calls.items():
        if ":" in c:
            continue  # resumed duplicates are reported in the audit, not used for curves
        out.append((int(c), rec))
    return sorted(out)


def analyse_run(runs_root: Path, scan: Mapping[str, Any], size: str, arm: str, seed: int) -> Dict[str, Any]:
    run_dir = runs_root / "rl" / size / arm / f"s{seed}"
    key = run_key(size, arm, seed)
    out: Dict[str, Any] = {"key": key, "size": size, "arm": arm, "seed": seed, "run_dir": str(run_dir)}
    rec_path = run_dir / "run_receipt.json"
    if not rec_path.is_file():
        out["status"] = "pending (no run_receipt.json)"
        return out
    rec = read_json(rec_path)
    cfg_path = run_dir / "resolved_easyr1.yaml"
    import yaml  # noqa: WPS433

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    rows = load_log(run_dir)
    train_calls = scan_calls(scan, key, "train")
    val_calls = scan_calls(scan, key, "val")
    tracker = read_json(run_dir / "ckpt" / "checkpoint_tracker.json") if (run_dir / "ckpt" / "checkpoint_tracker.json").is_file() else {}
    progress = read_json(run_dir / "progress.json") if (run_dir / "progress.json").is_file() else {}
    attempts = rec.get("attempts") or []
    max_steps = int(get(cfg, "trainer.max_steps") or rec.get("max_steps") or 0)
    rollout_batch = int(get(cfg, "data.rollout_batch_size") or 0)
    n_roll = int(get(cfg, "worker.rollout.n") or 0)
    rollouts_per_step = rollout_batch * n_roll
    out["status"] = rec.get("status")
    out["training_complete"] = bool(rows) and int(rows[-1]["step"]) == max_steps and all(a.get("rc") == 0 for a in attempts[-1:])
    out["merged_steps"] = sorted(int(k) for k in (rec.get("merged") or {}))
    out["merges_pending"] = [s for s in STEPS if s not in out["merged_steps"]]
    # ------------------------------------------------------------------ config facts
    out["config"] = {
        "max_steps": max_steps, "rollout_batch_size": rollout_batch, "global_batch_size": get(cfg, "worker.actor.global_batch_size"),
        "rollout_n": n_roll, "rollouts_per_step": rollouts_per_step, "lr": get(cfg, "worker.actor.optim.lr"),
        "rollout_temperature": get(cfg, "worker.rollout.temperature"), "val_freq": get(cfg, "trainer.val_freq"),
        "save_freq": get(cfg, "trainer.save_freq"), "max_prompt_length": get(cfg, "data.max_prompt_length"),
        "max_response_length": get(cfg, "data.max_response_length"), "data_seed": get(cfg, "data.seed"),
        "rollout_seed": get(cfg, "worker.rollout.seed"), "reward_arm": get(cfg, "worker.reward.reward_function_kwargs.arm"),
        "gated": get(cfg, "worker.reward.reward_function_kwargs.gated"), "alpha": get(cfg, "worker.reward.reward_function_kwargs.alpha"),
        "format_weight": get(cfg, "worker.reward.reward_function_kwargs.format_weight"), "n_gpus_per_node": get(cfg, "trainer.n_gpus_per_node"),
        "model_path": get(cfg, "worker.actor.model.model_path"),
    }
    # ------------------------------------------------------------------ reward curves
    out["logged_steps"] = len(rows)
    out["last_logged_step"] = int(rows[-1]["step"]) if rows else None
    out["curve_summary"] = reward_curve_summary(rows) if rows else {}
    vals = val_records(rows)
    out["val"] = vals
    out["best_val"] = {"by_own_reward": best_of(vals, "overall"), "by_A": best_of(vals, "A"), "by_M": best_of(vals, "M"),
                       "tracker": {"best_global_step": tracker.get("best_global_step"), "best_val_reward_score": tracker.get("best_val_reward_score"),
                                   "last_global_step": tracker.get("last_global_step")}}
    out["best_val"]["tracker_check"] = tracker_check(vals, tracker, get(cfg, "trainer.save_freq"), get(cfg, "trainer.val_freq"),
                                                     [a.get("resumed_from_step") for a in attempts])
    # scan-based curve facts
    if train_calls:
        tc = [r for _, r in train_calls]
        zv = [r["zero_var_groups"] / r["n_prompts"] if r["n_prompts"] else None for r in tc]
        zva = [r["zero_var_groups_A"] / r["n_prompts"] if r["n_prompts"] else None for r in tc]
        out["grpo_signal"] = {
            "zero_var_group_share": {"first10": mean(zv[:10]), "last10": mean(zv[-10:]), "mean_all": mean(zv)},
            "zero_var_group_share_on_A": {"first10": mean(zva[:10]), "last10": mean(zva[-10:]), "mean_all": mean(zva)},
        }
        out["reward_by_formula"] = {
            f: {"first10": mean(r["overall_by_formula"][f] for r in tc[:10]), "last10": mean(r["overall_by_formula"][f] for r in tc[-10:])}
            for f in ("A", "G", "P", "P_gated", "G_gated")
        }
        ec_first: Counter = Counter()
        ec_last: Counter = Counter()
        for r in tc[:10]:
            ec_first.update(r["error_class"])
        for r in tc[-10:]:
            ec_last.update(r["error_class"])
        out["rollout_error_class_share"] = {
            "first10": {k: v / sum(ec_first.values()) for k, v in sorted(ec_first.items())},
            "last10": {k: v / sum(ec_last.values()) for k, v in sorted(ec_last.items())},
        }
        by_ad: Dict[str, Dict[str, Any]] = {}
        for a in sorted({a for r in tc for a in r["by_adapter"]}):
            short = ADAPTER_SHORT.get(a, a)
            by_ad[short] = {}
            for m in ("overall", "A", "G", "M", "F"):
                f10 = mean(r["by_adapter"].get(a, {}).get(m) for r in tc[:10])
                l10 = mean(r["by_adapter"].get(a, {}).get(m) for r in tc[-10:])
                by_ad[short][m] = {"first10": f10, "last10": l10, "delta": (l10 - f10) if f10 is not None and l10 is not None else None}
        out["rollout_by_adapter"] = by_ad
        # consistency of the scan with the trainer log: call k <-> step k+1
        log_by_step = {int(r["step"]): r for r in rows}
        diffs, tok_diffs, unmatched = [], [], 0
        for c, r in train_calls:
            lr = log_by_step.get(c + 1)
            if lr is None:
                unmatched += 1
                continue
            lo = fnum(get(lr, "reward.overall"))
            if lo is not None and r["overall"] is not None:
                diffs.append(abs(lo - r["overall"]))
            lm = fnum(get(lr, "response_length.mean"))
            if lm is not None and r["n_rows"]:
                tok_diffs.append(abs(lm - r["response_tokens"] / r["n_rows"]))
        out["scan_log_consistency"] = {"train_calls": len(train_calls), "unmatched_calls": unmatched,
                                       "max_abs_overall_diff": max(diffs) if diffs else None,
                                       "max_abs_resp_len_diff": max(tok_diffs) if tok_diffs else None}
    # ------------------------------------------------------------------ compute
    wall = sum(h for h in (hours_between(a.get("started"), a.get("ended")) for a in attempts) if h is not None)
    n_gpus = int(rec.get("n_gpus") or get(rec, "gpu.n") or 0)
    step_s = [fnum(get(r, "timing_s.step")) for r in rows]
    step_s = [s for s in step_s if s is not None]
    timing_tot: Dict[str, float] = defaultdict(float)
    for r in rows:
        for k, v in (r.get("timing_s") or {}).items():
            if fnum(v) is not None:
                timing_tot[k] += float(v)
    resp_tokens_exact = sum(r["response_tokens"] for _, r in train_calls) if train_calls else None
    rollouts_exact = sum(r["n_rows"] for _, r in train_calls) if train_calls else None
    resp_tokens_log = sum((fnum(get(r, "response_length.mean")) or 0.0) * rollouts_per_step for r in rows) if rows else None
    prompt_tokens_log = sum((fnum(get(r, "prompt_length.mean")) or 0.0) * rollouts_per_step for r in rows) if rows else None
    total_tokens_log = sum((fnum(get(r, "perf.total_num_tokens")) or 0.0) for r in rows) if rows else None
    val_tokens = sum(r["response_tokens"] for _, r in val_calls) if val_calls else None
    placements = [a.get("placement") or {} for a in attempts]
    first_start = attempts[0].get("started") if attempts else None
    out["compute"] = {
        "gpu_name": get(rec, "gpu.name"), "n_gpus": n_gpus, "gpu_memory_mib": get(rec, "gpu.memory_mib"),
        "hosts": sorted({a.get("host") for a in attempts if a.get("host")}),
        "gpu_uuids": sorted({u for a in attempts for u in (get(a, "gpu.uuids") or [])}),
        "slurm_jobs": sorted({str(a.get("slurm_job")) for a in attempts if a.get("slurm_job")}),
        "mode": ("pair: " + ",".join(sorted({p.get("pair") for p in placements if p.get("pair")}))) if any(p.get("pair") for p in placements) else "single",
        "attempts": len(attempts),
        "resumed_from_steps": [a.get("resumed_from_step") for a in attempts],
        "started": first_start, "ended": attempts[-1].get("ended") if attempts else None, "completed": rec.get("completed"),
        "wall_hours_train": wall, "gpu_hours_train": wall * n_gpus,
        "wall_hours_incl_merges": hours_between(first_start, rec.get("completed")) if rec.get("completed") else None,
        "s_per_step_median": statistics.median(step_s) if step_s else None,
        "s_per_step_mean": (sum(step_s) / len(step_s)) if step_s else None,
        "s_per_step_wall": (wall * 3600 / len(rows)) if rows and wall else None,
        "step_time_hours_sum": (sum(step_s) / 3600) if step_s else None,
        "timing_hours_sum": {k: v / 3600 for k, v in sorted(timing_tot.items())},
        "rollouts_per_step": rollouts_per_step,
        "train_rollouts_exact": rollouts_exact,
        "response_tokens_exact": resp_tokens_exact,
        "response_tokens_from_log_mean": resp_tokens_log,
        "prompt_tokens_from_log_mean": prompt_tokens_log,
        "total_tokens_from_log": total_tokens_log,
        "val_rollouts": sum(r["n_rows"] for _, r in val_calls) if val_calls else None,
        "val_response_tokens": val_tokens,
        "host_ram": host_ram(run_dir) | {"share": get(rec, "gpu.host_ram.share"), "share_gb": get(rec, "gpu.host_ram.share_gb"),
                                         "progress_peak_gb": get(progress, "host_ram_gb.peak_gb"),
                                         "trainer_cpu_memory_used_gb_max": max((fnum(get(r, "perf.cpu_memory_used_gb")) or 0.0) for r in rows) if rows else None},
        "gpu_memory": gpu_mem(run_dir) | {
            "torch_max_allocated_gb": max((fnum(get(r, "perf.max_memory_allocated_gb")) or 0.0) for r in rows) if rows else None,
            "torch_max_reserved_gb": max((fnum(get(r, "perf.max_memory_reserved_gb")) or 0.0) for r in rows) if rows else None},
    }
    # ------------------------------------------------------------------ matching facts (compared within groups)
    flat_cfg = {k: v for k, v in flatten(cfg).items() if k not in CONFIG_ARM_KEYS}
    out["_match"] = {
        "config": flat_cfg,
        "receipt": {
            "model_dir.path": get(rec, "model_dir.path"), "model_dir.config_sha256": get(rec, "model_dir.config_sha256"),
            "model_dir.processor_digest": get(rec, "model_dir.processor_digest"),
            "model_dir.weights": json.dumps(get(rec, "model_dir.weights"), sort_keys=True),
            "data.train.sha256": get(rec, "data.train.sha256"), "data.val.sha256": get(rec, "data.val.sha256"),
            "data.store.sha256": get(rec, "data.store.sha256"), "data.system_prompt.sha256": get(rec, "data.system_prompt.sha256"),
            "data.cert_code_sha256_at_export": json.dumps(get(rec, "data.cert_code_sha256_at_export"), sort_keys=True),
            "git.trace.head": get(rec, "git.trace.head"), "git.trace.cert_code_sha256": get(rec, "git.trace.cert_code_sha256"),
            "git.trace.train_rl_sha256": get(rec, "git.trace.train_rl_sha256"), "git.trace_rlvr.head": get(rec, "git.trace_rlvr.head"),
            "env.lock.sha256": get(rec, "env.lock.sha256"), "spec.sha256": get(rec, "spec.sha256"), "release.sha256": get(rec, "release.sha256"),
            "prompt_clip": json.dumps(rec.get("prompt_clip"), sort_keys=True), "n_gpus": rec.get("n_gpus"), "gpu.name": get(rec, "gpu.name"),
            "seed": rec.get("seed"), "reward_kwargs.alpha": get(rec, "reward_kwargs.alpha"),
            "reward_kwargs.format_weight": get(rec, "reward_kwargs.format_weight"), "reward_kwargs.coord_frame": get(rec, "reward_kwargs.coord_frame"),
            "reward_kwargs.store_sha256": get(rec, "reward_kwargs.store_sha256"), "reward_kwargs.gated": get(rec, "reward_kwargs.gated"),
            "attempts": len(attempts), "resumed_from_steps": json.dumps([a.get("resumed_from_step") for a in attempts]),
            "save_freq": json.dumps([a.get("save_freq") for a in attempts]), "logged_steps": len(rows),
            "last_logged_step": out["last_logged_step"], "attempts[].placement (single/pair)": out["compute"]["mode"],
        },
        "train_multisets": [r["prompt_multiset_sha256"] for _, r in train_calls],
        "train_prompts_per_step": sorted({r["n_prompts"] for _, r in train_calls}),
        "train_rollouts_per_step": sorted({r["n_rows"] for _, r in train_calls}),
        "rollouts_per_prompt": sorted({tuple(r["rollouts_per_prompt"]) for _, r in train_calls if r.get("rollouts_per_prompt")}),
        "val_multisets": [r["prompt_multiset_sha256"] for _, r in val_calls],
        "train_response_multisets": [r.get("response_multiset_sha256") for _, r in train_calls],
        "train_response_prefixes": {c: r["response_sha_prefixes"] for c, r in train_calls if r.get("response_sha_prefixes")},
    }
    # ------------------------------------------------------------------ per-step curve rows (CSV + figures)
    scan_by_step = {c + 1: r for c, r in train_calls}
    curve = []
    for r in rows:
        s = int(r["step"])
        sc = scan_by_step.get(s)
        curve.append({
            "step": s,
            **{name: fnum(get(r, f"reward.{k}")) for k, name in LOG_KEYS.items()},
            "resp_len": fnum(get(r, "response_length.mean")),
            "zero_solve_rate": fnum(get(r, "rlvr_stats.zero_solve_rate")),
            "perfect_solve_rate": fnum(get(r, "rlvr_stats.perfect_solve_rate")),
            "entropy_loss": fnum(get(r, "actor.entropy_loss")),
            "grad_norm": fnum(get(r, "actor.grad_norm")),
            "step_s": fnum(get(r, "timing_s.step")),
            "zero_var_group_share": (sc["zero_var_groups"] / sc["n_prompts"]) if sc and sc["n_prompts"] else None,
            "response_tokens_exact": sc["response_tokens"] if sc else None,
        })
    out["_curve"] = curve
    # ------------------------------------------------------------------ merge integrity vs eval manifests
    integ = {}
    for step in STEPS:
        served = served_name(size, arm, seed, step)
        man_path = runs_root / "eval" / "ckpt" / served / "ckpt_manifest.json"
        merged = (rec.get("merged") or {}).get(str(step))
        if not man_path.is_file() or not merged:
            integ[str(step)] = {"status": "pending", "manifest": man_path.is_file(), "merged": bool(merged)}
            continue
        man = read_json(man_path)
        man_sha = {f["name"]: f.get("sha256") for f in man.get("files", []) if str(f.get("name", "")).endswith(".safetensors")}
        want = merged.get("safetensors_sha256") or {}
        expect_dir = str(run_dir / "hf" / f"global_step_{step}")
        integ[str(step)] = {
            "status": "ok" if (man_sha == want and man.get("ckpt_realpath", man.get("ckpt_dir")) == expect_dir) else "MISMATCH",
            "ckpt_dir": man.get("ckpt_dir"), "safetensors_equal": man_sha == want,
        }
    out["eval_manifest_integrity"] = integ
    return out


# ============================================================================== matching audit


def audit_group(name: str, members: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    present = [m for m in members if "_match" in m]
    res: Dict[str, Any] = {"group": name, "members": [m["key"] for m in members], "present": [m["key"] for m in present],
                           "differences": [], "data_order": {}, "verdict": None}
    if len(present) < 2:
        res["verdict"] = "pending"
        return res
    keys = sorted(set().union(*[set(m["_match"]["config"]) for m in present]))
    for k in keys:
        vals = {m["key"]: m["_match"]["config"].get(k) for m in present}
        if len({json.dumps(v, sort_keys=True) for v in vals.values()}) > 1:
            res["differences"].append({"field": f"resolved_easyr1.{k}", "values": vals, "class": classify(k)})
    rkeys = sorted(set().union(*[set(m["_match"]["receipt"]) for m in present]))
    for k in rkeys:
        vals = {m["key"]: m["_match"]["receipt"].get(k) for m in present}
        if len({json.dumps(v, sort_keys=True) for v in vals.values()}) > 1:
            res["differences"].append({"field": f"receipt.{k}", "values": vals, "class": classify(k)})
    for k in ("train_prompts_per_step", "train_rollouts_per_step", "rollouts_per_prompt"):
        vals = {m["key"]: json.dumps(m["_match"][k]) for m in present}
        if len(set(vals.values())) > 1:
            res["differences"].append({"field": f"rollout_scan.{k}", "values": vals, "class": "math"})
    seqs = {m["key"]: m["_match"]["train_multisets"] for m in present}
    n = min(len(s) for s in seqs.values())
    ref_key = present[0]["key"]
    mism = [i + 1 for i in range(n) if len({seqs[k][i] for k in seqs}) > 1]
    res["data_order"] = {"steps_compared": n, "steps_with_different_prompt_multiset": len(mism),
                         "first_mismatch_steps": mism[:10], "reference": ref_key,
                         "lengths": {k: len(v) for k, v in seqs.items()}}
    vseqs = {m["key"]: m["_match"]["val_multisets"] for m in present}
    vn = min(len(s) for s in vseqs.values())
    res["data_order"]["val_calls_compared"] = vn
    res["data_order"]["val_calls_with_different_rows"] = sum(1 for i in range(vn) if len({vseqs[k][i] for k in vseqs}) > 1)
    if mism:
        res["differences"].append({"field": "rollout_scan.train_prompt_multiset", "values": {"steps_differing": len(mism)}, "class": "math"})
    res["rollouts"] = rollout_overlap([(m["key"], m["_match"]) for m in present])
    math_diffs = [d for d in res["differences"] if d["class"] == "math"]
    prov = [d for d in res["differences"] if d["class"] == "provenance"]
    pending = len(present) < len(members)
    res["verdict"] = ("MISMATCH (training math)" if math_diffs else ("matched; provenance differences listed" if prov else "matched")) + (
        " [members pending]" if pending else "")
    return res


def multiset_overlap(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    if not a or not b:
        return None
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values()) / max(len(a), len(b))


def rollout_overlap(members: Sequence[Tuple[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """How many training rollouts the members share step by step (common random numbers: same init, prompts,
    rollout seed and numerics give identical rollouts until the first differing update)."""
    seqs = {k: m.get("train_response_multisets") or [] for k, m in members}
    n = min((len(v) for v in seqs.values()), default=0)
    identical = [i + 1 for i in range(n) if len({seqs[k][i] for k in seqs}) == 1 and seqs[members[0][0]][i]]
    keep_calls = sorted(set.intersection(*[set(m.get("train_response_prefixes") or {}) for _, m in members])) if members else []
    by_step: Dict[str, Any] = {}
    for c in keep_calls:
        pairs = {}
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ki, kj = members[i][0], members[j][0]
                pairs[f"{ki} vs {kj}"] = multiset_overlap(members[i][1]["train_response_prefixes"][c], members[j][1]["train_response_prefixes"][c])
        by_step[str(c + 1)] = {"mean_pairwise_shared_fraction": mean(pairs.values()), "pairs": pairs}
    return {"steps_with_identical_rollouts": identical[:20], "n_steps_identical": len(identical), "shared_fraction_by_step": by_step}


def classify(field: str) -> str:
    prov = ("git.", "env.", "cert_code", "train_rl", "placement", "hosts", "slurm", "gpu.uuids", "release.sha256", "spec.sha256")
    if any(p in field for p in prov):
        return "provenance"
    return "math"


# ============================================================================== SFT


def analyse_sft(runs_root: Path, size: str) -> Dict[str, Any]:
    d = runs_root / "sft" / SFT_TAG[size]
    out: Dict[str, Any] = {"size": size, "dir": str(d)}
    if not (d / "receipt.json").is_file():
        out["status"] = "pending"
        return out
    rec = read_json(d / "receipt.json")
    curve = read_json(d / "dev_curve.json")
    log = read_jsonl(d / "train_log.jsonl")
    launches = get(rec, "training.launches") or []
    plan = get(rec, "training.plan") or {}
    t = [float(r["time_s"]) for r in log if fnum(r.get("time_s")) is not None]
    first = launches[0] if launches else {}
    n_gpus = len(first.get("gpus") or []) or int(get(first, "layout.world_size") or 0)
    wall = hours_between(first.get("launch"), get(rec, "training.finished"))
    out.update({
        "status": "ok",
        "model": get(rec, "model.repo_id"), "revision": get(rec, "model.revision"),
        "gpu_name": (first.get("gpus") or [None])[0], "n_gpus": n_gpus, "host": first.get("host"), "launches": len(launches),
        "launch": first.get("launch"), "finished": get(rec, "training.finished"), "receipt_written": rec.get("written"),
        "wall_hours_train": wall, "gpu_hours_train": (wall or 0.0) * n_gpus,
        "wall_hours_incl_dev_eval": hours_between(first.get("launch"), rec.get("written")),
        "gpu_hours_incl_dev_eval": (hours_between(first.get("launch"), rec.get("written")) or 0.0) * n_gpus,
        "steps": len(log), "s_per_step_median": statistics.median(t) if t else None, "step_time_hours_sum": sum(t) / 3600 if t else None,
        "label_tokens": sum(int(r.get("label_tokens") or 0) for r in log), "seq_tokens": sum(int(r.get("seq_tokens") or 0) for r in log),
        "rows_seen": sum(int(r.get("rows") or 0) for r in log),
        "loss_first10": mean(fnum(r.get("loss")) for r in log[:10]), "loss_last10": mean(fnum(r.get("loss")) for r in log[-10:]),
        "lr": plan.get("lr"), "global_batch": plan.get("global_batch"), "epochs": plan.get("epochs"), "n_train": plan.get("n_train"),
        "layout": plan.get("layout") or first.get("layout"),
        "params_trainable": get(rec, "training.model.params_trainable"), "params_total": get(rec, "training.model.params_total"),
        "rule": curve.get("rule"), "outcome": curve.get("outcome"), "selected_step": curve.get("selected_step"),
        "dev_decoding": get(rec, "dev_eval.decoding"),
        "dev_curve": [{k: p.get(k) for k in ("step", "epoch", "A", "G", "M", "F", "joint", "tokens_mean", "dA", "dM", "qualifies", "reasons", "valid_schema_rate")}
                      for p in curve.get("curve", [])],
        "selected_dir": get(rec, "selected.dir"),
        "selected_safetensors_sha256": {k: v.get("sha256") for k, v in (get(rec, "selected.files") or {}).items() if k.endswith(".safetensors")},
    })
    saves = [int(p["step"]) for p in curve.get("curve", []) if p.get("step") is not None]
    qual = [int(p["step"]) for p in curve.get("curve", []) if p.get("qualifies")]
    out["selection"] = {
        "evaluated_saves": saves, "final_save_step": max(saves) if saves else None, "train_steps_logged": len(log),
        "first_qualifying_step": min(qual) if qual else None,
        "selected_is_final_save": bool(saves) and curve.get("selected_step") == max(saves),
        "final_save_is_last_train_step": bool(saves) and max(saves) == len(log),
        "note": ("the first qualifying save is the final save (end of epoch 3), so the rule and its last-save fallback select the same "
                 "checkpoint; no earlier save met the plateau criterion"
                 if qual and saves and min(qual) == max(saves) == curve.get("selected_step") else None),
    }
    # S eval manifest integrity (the step-0 evaluation is the selected checkpoint)
    man_path = runs_root / "eval" / "ckpt" / S_SERVED[size] / "ckpt_manifest.json"
    if man_path.is_file():
        man = read_json(man_path)
        man_sha = {f["name"]: f.get("sha256") for f in man.get("files", []) if str(f.get("name", "")).endswith(".safetensors")}
        out["s_eval_manifest"] = {"served": S_SERVED[size], "ckpt_dir": man.get("ckpt_dir"),
                                  "safetensors_equal_selected": man_sha == out["selected_safetensors_sha256"] if man_sha else None}
    return out


# ============================================================================== go/no-go


def spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    from scipy import stats  # noqa: WPS433

    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "p": None, "n": int(len(x)), "note": "undefined: a constant input" if len(x) >= 3 else "n < 3"}
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def go_no_go_1a_rows(rows: Sequence[Mapping[str, Any]], threshold_g: float = 0.9) -> Dict[str, Any]:
    num = [r for r in rows if r["adapter_id"] in NUMERIC_ADAPTERS]
    G = np.array([float(r["G"]) for r in num])
    M = np.array([float(r["M"]) for r in num])
    A = np.array([float(r["A"]) for r in num])
    hit = (G >= threshold_g) & (M < 1.0)
    k, n = int(hit.sum()), len(num)
    lo, hi = wilson(k, n)
    out: Dict[str, Any] = {"n_numeric_rows": n, "count_G_ge_thr_and_M_lt_1": k, "share": k / n if n else None,
                           "wilson_ci": [lo, hi], "G_threshold": threshold_g,
                           "spearman_G_M": spearman(G, M),
                           "of_which_A_correct": int((hit & (A >= 0.5)).sum()),
                           "error_class_of_hits": dict(Counter(str(r.get("error_class")) for r, h in zip(num, hit) if h)),
                           "by_adapter": {}}
    for a in NUMERIC_ADAPTERS:
        mask = np.array([r["adapter_id"] == a for r in num])
        if not mask.any():
            continue
        kk = int((hit & mask).sum())
        out["by_adapter"][ADAPTER_SHORT[a]] = {
            "n": int(mask.sum()), "count": kk, "share": kk / int(mask.sum()),
            "n_G_ge_thr": int(((G >= threshold_g) & mask).sum()), "n_M_lt_1": int(((M < 1) & mask).sum()),
            "G_mean": float(G[mask].mean()), "M_mean": float(M[mask].mean()),
            "spearman_G_M": spearman(G[mask], M[mask]),
        }
    # stratified bootstrap CI for the share (rows = clusters here; strata = adapter)
    rng = np.random.default_rng([0, 3])
    strata = np.array([r["adapter_id"] for r in num])
    tot = np.zeros(10000)
    for a in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == a)
        cnt = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=10000).astype(float)
        tot += cnt @ hit[idx].astype(float)
    if n:
        b = tot / n
        out["stratified_bootstrap_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    # share without the path adapter (its G is integral-tile grounding)
    no_path = [i for i, r in enumerate(num) if r["adapter_id"] != "path.rpg_movement_cost"]
    if no_path:
        out["share_excluding_rpg"] = {"n": len(no_path), "count": int(hit[no_path].sum()), "share": float(hit[no_path].mean())}
    return out


def go_no_go(runs_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    primary = runs_root / "sft" / "qwen2.5-vl-3b" / "dev" / "step_000564" / "cert" / "cert_scores.jsonl"
    rc = runs_root / "sft" / "qwen2.5-vl-3b" / "dev" / "step_000564" / "run_config.json"
    one_a: Dict[str, Any] = {"source": str(primary), "threshold_share": 0.10,
                             "rule": "S checkpoint, greedy E1-dev, numeric adapters (table, lever, rpg): share of rows with G >= 0.9 and M < 1 >= 10%; Spearman(G, M) reported"}
    if primary.is_file():
        rows = [r for r in read_jsonl(primary) if r.get("split") == "e1_dev" and r.get("condition") == "cert_v1"]
        cfg = read_json(rc) if rc.is_file() else {}
        one_a["decoding"] = get(cfg, "decoding")
        one_a["greedy"] = fnum(get(cfg, "decoding.temperature")) == 0.0
        one_a["scorer_version"] = sorted({r.get("scorer_version") for r in rows})
        one_a["n_rows_all_adapters"] = len(rows)
        res = go_no_go_1a_rows(rows)
        one_a.update(res)
        one_a["pass"] = bool(res["share"] is not None and res["share"] >= 0.10)
        one_a["prior_read_only_check"] = {"count": 47, "n": 384, "share": 47 / 384, "spearman": 0.21}
        one_a["reproduces_prior_check"] = (res["count_G_ge_thr_and_M_lt_1"] == 47 and res["n_numeric_rows"] == 384
                                           and res["spearman_G_M"]["rho"] is not None and abs(res["spearman_G_M"]["rho"] - 0.21) < 0.005)
        sens: Dict[str, Any] = {}
        for thr in (0.8, 0.95):
            r2 = go_no_go_1a_rows(rows, thr)
            sens[f"G>={thr}"] = {"count": r2["count_G_ge_thr_and_M_lt_1"], "share": r2["share"]}
        one_a["sensitivity_G_threshold"] = sens
    else:
        one_a["status"] = "pending (missing)"
    # descriptive sensitivity: 7B S greedy dev, and the sampled (trace preset) S evaluations
    extra: Dict[str, Any] = {}
    p7 = runs_root / "sft" / "qwen2.5-vl-7b" / "dev" / "step_000564" / "cert" / "cert_scores.jsonl"
    if p7.is_file():
        r7 = go_no_go_1a_rows([r for r in read_jsonl(p7) if r.get("split") == "e1_dev"])
        extra["7b_S_greedy_dev"] = {"source": str(p7), "count": r7["count_G_ge_thr_and_M_lt_1"], "n": r7["n_numeric_rows"],
                                    "share": r7["share"], "spearman": r7["spearman_G_M"],
                                    "by_adapter": {k: {"count": v["count"], "n": v["n"]} for k, v in r7["by_adapter"].items()}}
    for size in ("3b", "7b"):
        for seed in DEC_SEEDS:
            p = runs_root / "eval" / "ckpt" / S_SERVED[size] / "e1_dev" / "cert_v1" / f"seed{seed}" / "cert" / "cert_scores.jsonl"
            if p.is_file():
                rr = go_no_go_1a_rows(read_jsonl(p))
                extra[f"{size}_S_sampled_seed{seed}"] = {"source": str(p), "count": rr["count_G_ge_thr_and_M_lt_1"], "n": rr["n_numeric_rows"],
                                                         "share": rr["share"], "spearman_rho": rr["spearman_G_M"]["rho"]}
    one_a["sensitivity_descriptive"] = extra
    out["1a"] = one_a
    # 1(b): the corruption suite, as recorded by section5_cpu.py (v1 = SECTION5_CPU.md, v2, v2 gate re-run)
    one_b: Dict[str, Any] = {"rule": ">= 3 content corruption types per numeric adapter keep G (|dG| < 0.01) and lower M on every applicable row",
                             "runs": {}}
    for tag in ("section5_cpu", "section5_cpu_v2", "section5_cpu_v2_gate"):
        p = runs_root / "results" / tag / "summary.json"
        if not p.is_file():
            continue
        s = read_json(p)
        g = s.get("go_no_go_1b") or {}
        one_b["runs"][tag] = {
            "source": str(p), "scorer_version": s.get("scorer_version"), "cert_code_sha256": s.get("cert_code_sha256"),
            "corruptions_scored": s.get("corruptions_scored"), "corruption_expectations_unmet": s.get("corruption_expectations_unmet"),
            "by_adapter": {ADAPTER_SHORT.get(a, a): {"content_types": v.get("count_content_all_rows"), "all_types": v.get("count_all_rows"),
                                                    "pass": v.get("pass"), "content_type_names": v.get("content_types_all_rows")}
                           for a, v in sorted(g.items())},
        }
        one_b["runs"][tag]["pass"] = bool(g) and all(v.get("pass") for v in g.values())
    latest = "section5_cpu_v2_gate" if "section5_cpu_v2_gate" in one_b["runs"] else ("section5_cpu_v2" if "section5_cpu_v2" in one_b["runs"] else None)
    one_b["current"] = latest
    one_b["pass"] = one_b["runs"][latest]["pass"] if latest else None
    out["1b"] = one_b
    out["1_pass"] = bool(one_a.get("pass")) and bool(one_b.get("pass"))
    return out


# ============================================================================== SVG figures

COLORS = {"A": "#2a78d6", "G": "#eb6834", "P": "#1baf7a", "S": "#52514e", "P_gated": "#1baf7a", "G_gated": "#eb6834"}
DASH = {"P_gated": "6 4", "G_gated": "6 4"}


def moving_average(xs: Sequence[Optional[float]], w: int = 10) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i in range(len(xs)):
        win = [v for v in xs[max(0, i - w + 1): i + 1] if v is not None]
        out.append(sum(win) / len(win) if win else None)
    return out


def svg_small_multiples(path: Path, title: str, panels: Sequence[Mapping[str, Any]], legend: Sequence[Tuple[str, str]], note: str,
                        cols: int = 3) -> None:
    """panels: {title, xlabel, series: [{name, color, dash, width, opacity, points: [(x, y)], markers}]}"""
    pw, ph, gap, top, left = 300, 190, 24, 64, 46
    rows = math.ceil(len(panels) / cols)
    W = cols * (pw + gap) + 16
    H = top + rows * (ph + 44) + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, sans-serif">',
           "<style>.bg{fill:#fcfcfb}.t{fill:#0b0b0b}.m{fill:#52514e}.ax{stroke:#c9c8c3}.gr{stroke:#ecebe7}"
           "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t{fill:#f0efec}.m{fill:#c3c2b7}.ax{stroke:#55544f}.gr{stroke:#2c2c2a}}</style>",
           f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>',
           f'<text class="t" x="16" y="24" font-size="15" font-weight="600">{esc(title)}</text>']
    lx = 16
    for name, color in legend:
        dash = DASH.get(name, "")
        out.append(f'<line x1="{lx}" y1="42" x2="{lx + 22}" y2="42" stroke="{color}" stroke-width="2.5"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        out.append(f'<text class="m" x="{lx + 28}" y="46" font-size="12">{esc(name)}</text>')
        lx += 40 + 7 * len(name)
    for pi, panel in enumerate(panels):
        cx = 16 + (pi % cols) * (pw + gap)
        cy = top + (pi // cols) * (ph + 44)
        pts = [(x, y) for s in panel["series"] for x, y in s["points"] if y is not None]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pad = (y1 - y0) * 0.08 or (abs(y1) * 0.05 or 0.01)
        y0, y1 = y0 - pad, y1 + pad
        px0, px1, py0, py1 = cx + left, cx + pw, cy + 18, cy + ph

        def X(v: float) -> float:
            return px0 + (v - x0) / ((x1 - x0) or 1) * (px1 - px0)

        def Y(v: float) -> float:
            return py1 - (v - y0) / ((y1 - y0) or 1) * (py1 - py0)

        out.append(f'<text class="t" x="{cx}" y="{cy + 8}" font-size="12.5" font-weight="600">{esc(panel["title"])}</text>')
        for t in nice_ticks(y0, y1, 4):
            out.append(f'<line class="gr" x1="{px0}" y1="{Y(t):.1f}" x2="{px1}" y2="{Y(t):.1f}" stroke-width="1"/>')
            out.append(f'<text class="m" x="{px0 - 4}" y="{Y(t) + 4:.1f}" font-size="10" text-anchor="end">{fmt_tick(t)}</text>')
        for t in nice_ticks(x0, x1, 4):
            out.append(f'<text class="m" x="{X(t):.1f}" y="{py1 + 14}" font-size="10" text-anchor="middle">{fmt_tick(t)}</text>')
        out.append(f'<line class="ax" x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}" stroke-width="1"/>')
        out.append(f'<text class="m" x="{(px0 + px1) / 2:.1f}" y="{py1 + 28}" font-size="10" text-anchor="middle">{esc(panel.get("xlabel", "step"))}</text>')
        for s in panel["series"]:
            seg = [(X(x), Y(y)) for x, y in s["points"] if y is not None]
            if len(seg) >= 2:
                d = " ".join(f"{'M' if i == 0 else 'L'}{a:.1f},{b:.1f}" for i, (a, b) in enumerate(seg))
                out.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{s.get("width", 2)}" stroke-linejoin="round" '
                           f'stroke-linecap="round" opacity="{s.get("opacity", 1)}"' + (f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else "") + "/>")
            if s.get("markers"):
                for (a, b), (x, y) in zip(seg, [(x, y) for x, y in s["points"] if y is not None]):
                    out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{s["color"]}" stroke="#fcfcfb" stroke-width="1.5">'
                               f'<title>{esc(s["name"])} step {fmt_tick(x)}: {y:.4f}</title></circle>')
    out.append(f'<text class="m" x="16" y="{H - 10}" font-size="10.5">{esc(note)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def esc(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(lo: float, hi: float, n: int) -> List[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def fmt_tick(t: float) -> str:
    if abs(t) >= 100:
        return f"{t:.0f}"
    if abs(t - round(t)) < 1e-9:
        return f"{t:.0f}"
    return f"{t:.3g}"


# ============================================================================== build


def build(args: argparse.Namespace) -> Dict[str, Any]:
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    scan = read_json(Path(args.scan)) if args.scan and Path(args.scan).is_file() else {"runs": {}}
    doc: Dict[str, Any] = {"version": VERSION, "runs_root": str(runs_root.resolve()),
                           "scan": {"path": str(Path(args.scan).resolve()), "version": scan.get("version"), "created": scan.get("created")},
                           "settings": {"n_boot": args.n_boot, "bootstrap_seed": 0, "decoding_seeds": list(DEC_SEEDS), "mcnemar_seed": MCNEMAR_SEED,
                                        "mcnemar_source": MCNEMAR_SOURCE,
                                        "tests": "binary metrics (A, F, joint): exact McNemar on the decoding-seed-42 outcome (DESIGN 8, compare_cert.py); "
                                                 "continuous metrics (G, M, tokens): cluster sign-flip p (also reported for binary metrics as a secondary statistic)",
                                        "pooled_cis": {"ci_two_level": "compare_cert.py two-level bootstrap: training seeds resampled, clusters resampled independently within each seed",
                                                       "ci_shared_instance": "training seeds resampled, one stratified cluster resample per draw shared by every seed"}},
                           "pending": [], "pending_not_used": EXTERNAL_PENDING_NOT_USED}
    # ----------------------------------------------------------------- RL runs
    runs: Dict[str, Dict[str, Any]] = {}
    for size, arm, seed in EXPECTED_RUNS:
        r = analyse_run(runs_root, scan, size, arm, seed)
        runs[r["key"]] = r
        if not str(r.get("status", "")).startswith("completed"):
            doc["pending"].append(f"rl run {r['key']}: receipt status {r.get('status')!r} (training complete: {r.get('training_complete')}; merges pending at steps {r.get('merges_pending')})")
    # ----------------------------------------------------------------- matching audit
    doc["matching"] = [audit_group(name, [runs[run_key(*m)] for m in members]) for name, members in MATCH_GROUPS]
    cross = {}
    for label, ka, kb in (("same seed 1, H200 NVL (A/s1) vs RTX PRO 6000 (P_gated/s1)", "3b/A/s1", "3b/P_gated/s1"),
                          ("same seed 3, same GPU type, different node (A/s3 grn074 vs G/s3 grn073)", "3b/A/s3", "3b/G/s3"),
                          ("same seed 1 and GPU type, 3B vs 7B init (G/s1)", "3b/G/s1", "7b/G/s1")):
        ma, mb = runs[ka].get("_match") or {}, runs[kb].get("_match") or {}
        pa, pb = (ma.get("train_response_prefixes") or {}), (mb.get("train_response_prefixes") or {})
        cross[label] = {"runs": [ka, kb], "step1_shared_fraction": multiset_overlap(pa.get(0) or [], pb.get(0) or []),
                        "step1_same_prompts": (ma.get("train_multisets") or [None])[0] == (mb.get("train_multisets") or [None])[0]}
    doc["rollout_cross_checks"] = cross
    s_init = {}
    for size in ("3b", "7b"):
        ks = [k for k, r in runs.items() if r["size"] == size and "_match" in r]
        vals = {k: (runs[k]["_match"]["receipt"]["model_dir.path"], runs[k]["_match"]["receipt"]["model_dir.weights"],
                    runs[k]["_match"]["receipt"]["model_dir.config_sha256"]) for k in ks}
        s_init[size] = {"runs": len(ks), "distinct_inits": len(set(vals.values())), "path": sorted({v[0] for v in vals.values()})}
    doc["sft_init_shared"] = s_init
    # ----------------------------------------------------------------- SFT
    doc["sft"] = {size: analyse_sft(runs_root, size) for size in ("3b", "7b")}
    # ----------------------------------------------------------------- E1-dev step curves
    evals: Dict[str, EvalSet] = {}
    for size in ("3b", "7b"):
        evals[S_SERVED[size]] = EvalSet(S_SERVED[size], runs_root / "eval" / "ckpt" / S_SERVED[size])
    for size, arm, seed in EXPECTED_RUNS:
        for step in STEPS:
            name = served_name(size, arm, seed, step)
            evals[name] = EvalSet(name, runs_root / "eval" / "ckpt" / name)
    doc["e1dev_models"] = {k: v.summary() for k, v in evals.items()}
    for k, v in evals.items():
        if not v.ok:
            doc["pending"].append(f"E1-dev evaluation {k}: {v.status}")
    steps_doc: Dict[str, Any] = {}
    cls_store: Dict[Tuple[str, str, int, int], Clusters] = {}
    for size, arm, seed in EXPECTED_RUNS:
        key = run_key(size, arm, seed)
        S = evals[S_SERVED[size]]
        entry = {"0": {"served": S.served, "status": S.status, "metrics": S.summary().get("metrics")}}
        for step in STEPS:
            e = evals[served_name(size, arm, seed, step)]
            sm = e.summary()
            rec = {"served": e.served, "status": e.status, "metrics": sm.get("metrics"), "by_adapter": sm.get("by_adapter"),
                   "error_class_share": sm.get("error_class_share"), "per_decoding_seed": sm.get("per_decoding_seed")}
            if e.ok and S.ok:
                d, cl = paired_diff(e, S, args.n_boot, 0)
                rec["vs_S"] = d
                if cl is not None:
                    cls_store[(size, arm, seed, step)] = cl
            entry[str(step)] = rec
        steps_doc[key] = entry
    doc["e1dev_steps"] = steps_doc
    # within-seed arm differences per step (descriptive)
    comps: Dict[str, Any] = {}
    pairs_3b = (("P", "G"), ("P", "A"), ("G", "A"))
    for step in (0,) + STEPS:
        if step == 0:
            continue
        for x, y in pairs_3b:
            per_seed = {}
            cls = []
            for seed in (1, 2, 3):
                ex, ey = evals[served_name("3b", x, seed, step)], evals[served_name("3b", y, seed, step)]
                if ex.ok and ey.ok:
                    d, cl = paired_diff(ex, ey, args.n_boot, 0)
                    per_seed[str(seed)] = d
                    cls.append(cl)
            item: Dict[str, Any] = {"pair": f"{x}-{y}", "size": "3b", "step": step, "per_seed": per_seed}
            if len(cls) >= 2:
                boot = two_level_bootstrap(cls, args.n_boot, np.random.default_rng([0, 4, step]))
                lo, hi = pct_ci(boot)
                sboot = shared_instance_bootstrap(cls, args.n_boot, np.random.default_rng([0, 6, step]))
                slo, shi = pct_ci(sboot)
                pooled = {}
                for j, m in enumerate(METRICS):
                    ds = [per_seed[s]["metrics"][m]["delta"] for s in per_seed]
                    est = mean(ds)
                    pooled[m] = {"delta": est, "ci_two_level": [lo[j], hi[j]], "ci_shared_instance": [slo[j], shi[j]], "n_seeds": len(ds),
                                 "n_pos": sum(1 for v in ds if v is not None and v > 0), "n_neg": sum(1 for v in ds if v is not None and v < 0),
                                 "per_seed_test_p": {sd: per_seed[sd]["metrics"][m]["p"] for sd in per_seed},
                                 "per_seed_test": per_seed[next(iter(per_seed))]["metrics"][m]["test"]}
                item["pooled"] = pooled
            comps[f"3b {x}-{y} step {step}"] = item
        for (size, x, y, seed) in (("7b", "P", "G", 1), ("3b", "P_gated", "G_gated", 1)):
            ex, ey = evals[served_name(size, x, seed, step)], evals[served_name(size, y, seed, step)]
            key = f"{size} {x}-{y} step {step}"
            if ex.ok and ey.ok:
                d, _ = paired_diff(ex, ey, args.n_boot, 0)
                comps[key] = {"pair": f"{x}-{y}", "size": size, "step": step, "per_seed": {str(seed): d}}
            else:
                comps[key] = {"pair": f"{x}-{y}", "size": size, "step": step, "status": "pending", "missing": [e.served for e in (ex, ey) if not e.ok]}
    doc["e1dev_arm_diffs"] = comps
    # pooled 3B arm levels per step
    levels: Dict[str, Any] = {}
    for arm in ("A", "G", "P"):
        lv = {}
        for step in (0,) + STEPS:
            vals = []
            for seed in (1, 2, 3):
                m = steps_doc[run_key("3b", arm, seed)][str(step)].get("metrics")
                if m:
                    vals.append(m)
            lv[str(step)] = {mm: {"mean": mean(v[mm] for v in vals), "min": min((v[mm] for v in vals), default=None),
                                  "max": max((v[mm] for v in vals), default=None), "n_seeds": len(vals)} for mm in METRICS}
        levels[arm] = lv
    doc["e1dev_3b_levels"] = levels
    by_ad: Dict[str, Any] = {}
    for arm in ("A", "G", "P"):
        by_ad[arm] = {}
        for step in (0,) + STEPS:
            recs = []
            for seed in (1, 2, 3):
                st = steps_doc[run_key("3b", arm, seed)][str(step)]
                ba = st.get("by_adapter") if step else evals[S_SERVED["3b"]].summary().get("by_adapter")
                if ba:
                    recs.append(ba)
            for ad in sorted({k for r in recs for k in r}):
                by_ad[arm].setdefault(ad, {})[str(step)] = {m: mean(r[ad][m] for r in recs if ad in r) for m in ("A", "G", "M", "joint")} | {"n_seeds": len(recs)}
    doc["e1dev_3b_levels_by_adapter"] = by_ad
    pooled_vs_s: Dict[str, Any] = {}
    for arm in ("A", "G", "P"):
        pooled_vs_s[arm] = {}
        for step in STEPS:
            keys = [(("3b", arm, sd, step)) for sd in (1, 2, 3) if ("3b", arm, sd, step) in cls_store]
            if len(keys) < 2:
                pooled_vs_s[arm][str(step)] = {"status": "pending"}
                continue
            boot = two_level_bootstrap([cls_store[k] for k in keys], args.n_boot, np.random.default_rng([0, 5, step]))
            lo, hi = pct_ci(boot)
            sboot = shared_instance_bootstrap([cls_store[k] for k in keys], args.n_boot, np.random.default_rng([0, 7, step]))
            slo, shi = pct_ci(sboot)
            res = {}
            for j, m in enumerate(METRICS):
                recs = {str(k[2]): steps_doc[run_key("3b", arm, k[2])][str(step)]["vs_S"]["metrics"][m] for k in keys}
                ds = [r["delta"] for r in recs.values()]
                res[m] = {"delta": mean(ds), "ci_two_level": [lo[j], hi[j]], "ci_shared_instance": [slo[j], shi[j]], "n_seeds": len(ds),
                          "n_pos": sum(1 for v in ds if v is not None and v > 0), "n_neg": sum(1 for v in ds if v is not None and v < 0),
                          "per_seed_test_p": {sd: r["p"] for sd, r in recs.items()}, "per_seed_test": next(iter(recs.values()))["test"]}
            pooled_vs_s[arm][str(step)] = res
    doc["e1dev_3b_vs_S_pooled"] = pooled_vs_s
    # completion-token changes from S: seed-pooled vs single runs (so the wording can say which)
    tok: Dict[str, Any] = {"per_run": {}, "pooled_3b": {}}
    for k, entry in steps_doc.items():
        for st in STEPS:
            v = ((entry.get(str(st)) or {}).get("vs_S") or {}).get("metrics", {}).get("tokens")
            if v:
                tok["per_run"][f"{k} step {st}"] = v["delta"]
    for arm, by in pooled_vs_s.items():
        for st, res in by.items():
            if "tokens" in res:
                tok["pooled_3b"][f"{arm} step {st}"] = res["tokens"]["delta"]
    def _extreme(d: Mapping[str, Optional[float]]) -> Dict[str, Any]:
        items = [(kk, vv) for kk, vv in d.items() if vv is not None]
        if not items:
            return {}
        kk, vv = max(items, key=lambda t: abs(t[1]))
        return {"max_abs": abs(vv), "at": kk, "value": vv}
    tok["max_abs_pooled_3b"] = _extreme(tok["pooled_3b"])
    tok["max_abs_per_run_any_step"] = _extreme(tok["per_run"])
    tok["max_abs_per_run_step300"] = _extreme({kk: vv for kk, vv in tok["per_run"].items() if kk.endswith("step 300")})
    tok["S_tokens"] = {size: (evals[S_SERVED[size]].summary().get("metrics") or {}).get("tokens") for size in ("3b", "7b")}
    doc["e1dev_tokens_vs_S"] = tok
    # ----------------------------------------------------------------- RL curve and compute tables
    pub_runs = {}
    for k, r in runs.items():
        pub_runs[k] = {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
    doc["rl_runs"] = pub_runs
    # pooled reward-curve summary for the 3B main arms
    pooled_curve = {}
    for arm in ("A", "G", "P"):
        rs = [runs[run_key("3b", arm, s)] for s in (1, 2, 3) if runs[run_key("3b", arm, s)].get("curve_summary")]
        pooled_curve[arm] = {m: {w: mean(r["curve_summary"][m][w] for r in rs) for w in ("first10", "last10", "delta")}
                             for m in ("overall", "A", "G", "M", "F", "joint", "resp_len")} if rs else {}
        pooled_curve[arm]["n_seeds"] = len(rs)
    doc["reward_curves_3b_pooled"] = pooled_curve
    # compute totals
    tot: Dict[str, Any] = {"by_gpu": {}, "by_arm_3b": {}, "rl_total_gpu_hours": 0.0, "rl_total_response_tokens": 0}
    for k, r in runs.items():
        c = r.get("compute") or {}
        g = c.get("gpu_name") or "unknown"
        b = tot["by_gpu"].setdefault(g, {"runs": 0, "gpu_hours": 0.0, "wall_hours": 0.0})
        b["runs"] += 1
        b["gpu_hours"] += c.get("gpu_hours_train") or 0.0
        b["wall_hours"] += c.get("wall_hours_train") or 0.0
        tot["rl_total_gpu_hours"] += c.get("gpu_hours_train") or 0.0
        tot["rl_total_response_tokens"] += c.get("response_tokens_exact") or 0
    for arm in ("A", "G", "P"):
        rs = [runs[run_key("3b", arm, s)].get("compute") or {} for s in (1, 2, 3)]
        tot["by_arm_3b"][arm] = {"gpu_hours": sum(c.get("gpu_hours_train") or 0.0 for c in rs),
                                 "response_tokens": sum(c.get("response_tokens_exact") or 0 for c in rs),
                                 "response_tokens_per_seed": [c.get("response_tokens_exact") for c in rs],
                                 "mean_resp_len_per_seed": [(c.get("response_tokens_exact") or 0) / c["train_rollouts_exact"] if c.get("train_rollouts_exact") else None for c in rs]}
    sps: Dict[str, List[Tuple[float, float, float, str]]] = {}
    for k, r in runs.items():
        c = r.get("compute") or {}
        if c.get("s_per_step_median") is not None:
            sps.setdefault(f"{r['size']} on {short_gpu(c.get('gpu_name'))} x {c.get('n_gpus')}", []).append(
                (c["s_per_step_median"], c.get("wall_hours_train") or 0.0, c.get("gpu_hours_train") or 0.0, k))
    tot["s_per_step_median_by_size_gpu"] = {
        g: {"runs": [x[3] for x in v], "median_of_run_medians": statistics.median([x[0] for x in v]),
            "min": min(x[0] for x in v), "max": max(x[0] for x in v),
            "wall_hours_min": min(x[1] for x in v), "wall_hours_max": max(x[1] for x in v),
            "gpu_hours_min": min(x[2] for x in v), "gpu_hours_max": max(x[2] for x in v)}
        for g, v in sorted(sps.items())}
    sft_gpu = sum((doc["sft"][s].get("gpu_hours_incl_dev_eval") or 0.0) for s in doc["sft"])
    tot["sft_total_gpu_hours_incl_dev_eval"] = sft_gpu
    tot["training_total_gpu_hours"] = tot["rl_total_gpu_hours"] + sft_gpu
    doc["compute_totals"] = tot
    # ----------------------------------------------------------------- go/no-go
    doc["go_no_go"] = go_no_go(runs_root)
    # ----------------------------------------------------------------- side outputs (CSV + SVG)
    write_csvs(out_dir, runs, steps_doc)
    write_figures(out_dir, runs, steps_doc, doc)
    doc["outputs"] = sorted(p.name for p in out_dir.iterdir() if p.suffix in (".csv", ".svg"))
    return doc


def write_csvs(out_dir: Path, runs: Mapping[str, Mapping[str, Any]], steps_doc: Mapping[str, Any]) -> None:
    with (out_dir / "reward_curves.csv").open("w", newline="", encoding="utf-8") as h:
        cols = ["run", "size", "arm", "seed", "step", "overall", "A", "G", "M", "F", "joint", "resp_len", "zero_solve_rate", "perfect_solve_rate",
                "entropy_loss", "grad_norm", "step_s", "zero_var_group_share", "response_tokens_exact"]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for k, r in runs.items():
            for row in r.get("_curve", []):
                w.writerow({"run": k, "size": r["size"], "arm": r["arm"], "seed": r["seed"], **row})
    with (out_dir / "val_curves.csv").open("w", newline="", encoding="utf-8") as h:
        cols = ["run", "size", "arm", "seed", "step", "overall", "A", "G", "M", "F", "joint", "reward_score", "resp_len"]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for k, r in runs.items():
            for v in r.get("val", []):
                w.writerow({"run": k, "size": r["size"], "arm": r["arm"], "seed": r["seed"], **v})
    with (out_dir / "e1dev_steps.csv").open("w", newline="", encoding="utf-8") as h:
        cols = ["run", "step", "served", "status"] + [f"{m}" for m in METRICS] + [f"d{m}_vs_S" for m in METRICS] + [f"d{m}_vs_S_lo" for m in METRICS] + [f"d{m}_vs_S_hi" for m in METRICS]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for k, entry in steps_doc.items():
            for step, rec in entry.items():
                row = {"run": k, "step": step, "served": rec.get("served"), "status": rec.get("status")}
                for m in METRICS:
                    row[m] = (rec.get("metrics") or {}).get(m)
                    vs = (rec.get("vs_S") or {}).get("metrics", {}).get(m)
                    if vs:
                        row[f"d{m}_vs_S"], row[f"d{m}_vs_S_lo"], row[f"d{m}_vs_S_hi"] = vs["delta"], vs["ci"][0], vs["ci"][1]
                w.writerow(row)
    with (out_dir / "compute_rl.csv").open("w", newline="", encoding="utf-8") as h:
        cols = ["run", "gpu_name", "n_gpus", "hosts", "mode", "started", "ended", "wall_hours_train", "gpu_hours_train", "s_per_step_median",
                "s_per_step_mean", "train_rollouts_exact", "response_tokens_exact", "prompt_tokens_from_log_mean", "total_tokens_from_log",
                "host_unreclaimable_used_gb_max", "host_cgroup_peak_gb_max", "host_limit_gb", "host_share", "torch_max_reserved_gb"]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for k, r in runs.items():
            c = r.get("compute") or {}
            hr = c.get("host_ram") or {}
            w.writerow({"run": k, "gpu_name": c.get("gpu_name"), "n_gpus": c.get("n_gpus"), "hosts": ";".join(c.get("hosts") or []), "mode": c.get("mode"),
                        "started": c.get("started"), "ended": c.get("ended"), "wall_hours_train": c.get("wall_hours_train"),
                        "gpu_hours_train": c.get("gpu_hours_train"), "s_per_step_median": c.get("s_per_step_median"), "s_per_step_mean": c.get("s_per_step_mean"),
                        "train_rollouts_exact": c.get("train_rollouts_exact"), "response_tokens_exact": c.get("response_tokens_exact"),
                        "prompt_tokens_from_log_mean": c.get("prompt_tokens_from_log_mean"), "total_tokens_from_log": c.get("total_tokens_from_log"),
                        "host_unreclaimable_used_gb_max": hr.get("unreclaimable_used_gb_max"), "host_cgroup_peak_gb_max": hr.get("cgroup_peak_gb_max"),
                        "host_limit_gb": hr.get("limit_gb"), "host_share": hr.get("share"),
                        "torch_max_reserved_gb": (c.get("gpu_memory") or {}).get("torch_max_reserved_gb")})


def write_figures(out_dir: Path, runs: Mapping[str, Mapping[str, Any]], steps_doc: Mapping[str, Any], doc: Mapping[str, Any]) -> None:
    metrics = [("overall", "own reward (arm formula)"), ("A", "A (answer)"), ("G", "G (grounding)"), ("M", "M (semantic)"),
               ("F", "F (format)"), ("resp_len", "response tokens (mean)")]
    # 3B A/G/P: per-seed thin lines + mean of seeds, 10-step moving average
    panels = []
    for m, title in metrics:
        series = []
        for arm in ("A", "G", "P"):
            per = []
            for s in (1, 2, 3):
                cur = runs[run_key("3b", arm, s)].get("_curve") or []
                ma = moving_average([c[m] for c in cur])
                pts = [(c["step"], v) for c, v in zip(cur, ma)]
                per.append(dict((x, y) for x, y in pts))
                series.append({"name": f"{arm} s{s}", "color": COLORS[arm], "width": 1, "opacity": 0.3, "points": pts})
            steps = sorted(set().union(*[set(p) for p in per])) if per else []
            mpts = [(x, mean(p.get(x) for p in per)) for x in steps]
            series.append({"name": arm, "color": COLORS[arm], "width": 2.2, "points": mpts})
        panels.append({"title": title, "series": series})
    svg_small_multiples(out_dir / "reward_curves_3b.svg", "3B RL training reward, arms A / G / P (10-step moving average)", panels,
                        [("A", COLORS["A"]), ("G", COLORS["G"]), ("P", COLORS["P"])],
                        "Thick line: mean of training seeds 1-3; thin lines: single seeds. Own reward differs by arm formula; A/G/M/F are comparable.")
    panels = []
    for m, title in metrics:
        series = []
        for key, name in (("7b/G/s1", "G 7B"), ("7b/P/s1", "P 7B"), ("3b/G_gated/s1", "G_gated 3B"), ("3b/P_gated/s1", "P_gated 3B")):
            cur = runs[key].get("_curve") or []
            ma = moving_average([c[m] for c in cur])
            arm = runs[key]["arm"]
            series.append({"name": name, "color": COLORS[arm], "dash": "6 4" if "gated" in arm else ("2 3" if False else None),
                           "width": 2 if runs[key]["size"] == "7b" else 1.6, "points": [(c["step"], v) for c, v in zip(cur, ma)]})
        panels.append({"title": title, "series": series})
    svg_small_multiples(out_dir / "reward_curves_7b_gated.svg", "7B G / P (solid) and 3B gated pair (dashed), seed 1 (10-step moving average)", panels,
                        [("G (7B)", COLORS["G"]), ("P (7B)", COLORS["P"]), ("G_gated", COLORS["G"]), ("P_gated", COLORS["P"])],
                        "Solid: 7B G and P. Dashed: 3B P_gated and G_gated (answer-gated reward).")
    # E1-dev step curves
    panels = []
    for m, title in (("A", "A"), ("G", "G"), ("M", "M"), ("F", "F"), ("joint", "joint"), ("tokens", "completion tokens")):
        series = []
        for arm in ("A", "G", "P"):
            lv = doc["e1dev_3b_levels"][arm]
            pts = [(int(s), lv[s][m]["mean"]) for s in sorted(lv, key=int) if lv[s][m]["mean"] is not None]
            series.append({"name": arm, "color": COLORS[arm], "width": 2, "points": pts, "markers": True})
            for seed in (1, 2, 3):
                e = steps_doc[run_key("3b", arm, seed)]
                pts = [(int(s), (e[s].get("metrics") or {}).get(m)) for s in sorted(e, key=int)]
                series.append({"name": f"{arm} s{seed}", "color": COLORS[arm], "width": 1, "opacity": 0.3, "points": pts})
        for arm in ("P_gated", "G_gated"):
            e = steps_doc.get(run_key("3b", arm, 1)) or {}
            pts = [(int(s), (e[s].get("metrics") or {}).get(m)) for s in sorted(e, key=int)]
            series.append({"name": f"{arm} s1", "color": COLORS[arm], "dash": DASH[arm], "width": 1.6, "points": pts})
        panels.append({"title": title, "series": series, "xlabel": "RL step (0 = S)"})
    svg_small_multiples(out_dir / "e1dev_steps_3b.svg", "3B E1-dev by RL step (cert_v1, decoding seeds 42-44)", panels,
                        [("A", COLORS["A"]), ("G", COLORS["G"]), ("P", COLORS["P"]), ("P_gated", COLORS["P_gated"]), ("G_gated", COLORS["G_gated"])],
                        "Markers: mean of training seeds 1-3; thin lines: single seeds. Dashed: gated pair (seed 1, RTX PRO 6000; not matched to A/G/P). "
                        "Step 0 = shared SFT checkpoint S.")


# ============================================================================== markdown (rendered from the JSON file only)


def pct(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{d}f}"


def pp(v: Optional[float], d: int = 2) -> str:
    return "n/a" if v is None else f"{100 * v:+.{d}f}"


def num(v: Optional[float], d: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def mval(m: str, v: Optional[float]) -> str:
    return num(v, 1) if m == "tokens" else pct(v)


def mdelta(m: str, v: Optional[float]) -> str:
    return ("n/a" if v is None else f"{v:+.1f}") if m == "tokens" else pp(v)


def ci_str(m: str, ci: Optional[Sequence[Optional[float]]]) -> str:
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{mdelta(m, ci[0])}, {mdelta(m, ci[1])}]"


def pval(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def mcn(rec: Optional[Mapping[str, Any]]) -> str:
    """McNemar p with the discordant counts: p (b Y-only / c X-only)."""
    mc = (rec or {}).get("mcnemar")
    if not mc:
        return "n/a"
    return f"p {pval(mc['p'])} ({mc['b_y_only']} / {mc['c_x_only']})"


def gb(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def tokm(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v / 1e6:.2f}"


def render_md(doc: Mapping[str, Any]) -> str:
    L: List[str] = []
    a = L.append
    a("# Training dynamics, compute accounting and go/no-go 1 (training)")
    a("")
    a(f"Generated by `{doc['version']}` from `{doc['runs_root']}` (rollout scan `{doc['scan']['version']}`, {doc['scan']['created']}). "
      "Every number below is read from `training.json`, which the same run wrote. Rates are percent; differences are percentage points (pp); "
      "token counts are in millions (M) unless marked.")
    a("")
    render_headlines(doc, a)
    a("## Pending inputs")
    a("")
    if doc["pending"]:
        for p in doc["pending"]:
            a(f"- {p}")
    else:
        a("- none")
    for p in doc["pending_not_used"]:
        a(f"- (other campaign item) {p}")
    a("")
    # ---------------------------------------------------------------- go / no-go
    g = doc["go_no_go"]
    a("## Go/no-go 1 (DESIGN 8): the semantic signal is distinct from G")
    a("")
    one_a = g["1a"]
    a("### 1(a) S checkpoint, greedy E1-dev, numeric adapters")
    a("")
    if one_a.get("n_numeric_rows"):
        sp = one_a["spearman_G_M"]
        a(f"Source: `{one_a['source']}` (decoding temperature {one_a['decoding'].get('temperature') if one_a.get('decoding') else 'n/a'}, "
          f"scorer {', '.join(one_a['scorer_version'])}; {one_a['n_rows_all_adapters']} E1-dev rows, of which {one_a['n_numeric_rows']} numeric).")
        a("")
        a(f"- **Rows with G >= 0.9 and M < 1: {one_a['count_G_ge_thr_and_M_lt_1']} / {one_a['n_numeric_rows']} = {pct(one_a['share'])}%** "
          f"(Wilson 95% CI {pct(one_a['wilson_ci'][0])}-{pct(one_a['wilson_ci'][1])}%; adapter-stratified bootstrap "
          f"{pct(one_a['stratified_bootstrap_ci'][0])}-{pct(one_a['stratified_bootstrap_ci'][1])}%). Threshold 10%: **{'PASS' if one_a['pass'] else 'FAIL'}**.")
        a(f"- Spearman(G, M) over the {sp['n']} numeric rows: rho = {num(sp['rho'], 3)} (p = {pval(sp['p'])}).")
        pr = one_a["prior_read_only_check"]
        a(f"- The earlier read-only check ({pr['count']}/{pr['n']} = {pct(pr['share'])}%, Spearman {pr['spearman']}) is "
          f"{'reproduced exactly' if one_a['reproduces_prior_check'] else 'NOT reproduced'}.")
        a(f"- Of the qualifying rows, {one_a['of_which_A_correct']} have a correct answer (A = 1); error classes: "
          + ", ".join(f"{k} {v}" for k, v in sorted(one_a["error_class_of_hits"].items())) + ".")
        a("")
        a("| adapter | rows | G >= 0.9 | M < 1 | G >= 0.9 and M < 1 | share | mean G | mean M | Spearman(G, M) |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for ad, v in one_a["by_adapter"].items():
            s = v["spearman_G_M"]
            a(f"| {ad} | {v['n']} | {v['n_G_ge_thr']} | {v['n_M_lt_1']} | {v['count']} | {pct(v['share'])}% | {num(v['G_mean'])} | {num(v['M_mean'])} | "
              + (num(s['rho']) if s['rho'] is not None else s.get('note', 'n/a')) + " |")
        ex = one_a.get("share_excluding_rpg")
        a("")
        a("Reading (descriptive, does not change the pre-registered verdict):")
        if ex:
            a(f"- The pass rests on rpg: rpg G is 1.0 on every row (tile grounding is saturated at S), so on rpg the criterion reduces to M < 1. "
              f"Without rpg the share is {ex['count']}/{ex['n']} = {pct(ex['share'])}%. Lever has M = 1 on every row, so its Spearman is undefined, and the pooled "
              "Spearman mixes between-adapter level differences with within-adapter association.")
        sens = one_a.get("sensitivity_G_threshold", {})
        if sens:
            a("- G threshold sensitivity: " + "; ".join(f"{k}: {v['count']} rows ({pct(v['share'])}%)" for k, v in sens.items()) + ".")
        sd = one_a.get("sensitivity_descriptive", {})
        if sd:
            parts = []
            for k, v in sd.items():
                parts.append(f"{k}: {v['count']}/{v['n']} = {pct(v['share'])}%")
            a("- Other S outputs (not pre-registered): " + "; ".join(parts) + ".")
    else:
        a(f"- pending: {one_a.get('status')}")
    a("")
    one_b = g["1b"]
    a("### 1(b) Corruption suite (from `section5_cpu.py` summaries)")
    a("")
    a("| run | scorer | corruptions scored | unmet expectations | table (content / all types) | lever | rpg | pass |")
    a("|---|---|---:|---:|---|---|---|---|")
    for tag, v in one_b["runs"].items():
        ba = v["by_adapter"]
        cell = lambda k: f"{ba[k]['content_types']} / {ba[k]['all_types']}" if k in ba else "n/a"  # noqa: E731
        a(f"| `{tag}` | {v['scorer_version']} | {v['corruptions_scored']} | {v['corruption_expectations_unmet']} | {cell('table')} | {cell('lever')} | {cell('rpg')} | "
          f"{'PASS' if v['pass'] else 'FAIL'} |")
    a("")
    a(f"Current record: `{one_b['current']}` ({'PASS' if one_b['pass'] else 'FAIL'}; at least 3 content types are needed per numeric adapter). "
      "SECTION5_CPU.md's headline quotes the pre-review `data/v1` run (`section5_cpu`); its review note points to the v2 re-run.")
    r1, rc = one_b["runs"].get("section5_cpu"), one_b["runs"].get(one_b["current"] or "")
    if r1 and rc and "rpg" in r1["by_adapter"] and "rpg" in rc["by_adapter"]:
        added = sorted(set(rc["by_adapter"]["rpg"]["content_type_names"] or []) - set(r1["by_adapter"]["rpg"]["content_type_names"] or []))
        a(f"rpg content types: {r1['by_adapter']['rpg']['content_types']} (v1) -> {rc['by_adapter']['rpg']['content_types']} (current), added: "
          + ", ".join(f"`{x}`" for x in added) + ".")
    a("")
    a(f"**Go/no-go 1 overall: {'PASS' if g['1_pass'] else 'FAIL'}** (1(a) and 1(b) both pass; go/no-go 2, P > G on the primary family, is outside this report).")
    a("")
    # ---------------------------------------------------------------- matching
    a("## Matching audit (DESIGN 7.3)")
    a("")
    for s, v in doc["sft_init_shared"].items():
        a(f"- {s}: {v['runs']} RL runs, {v['distinct_inits']} distinct SFT init(s) ({', '.join(v['path'])}).")
    a("")
    a("| matched set | verdict | steps with a different prompt multiset | val calls compared (differing) | differences |")
    a("|---|---|---:|---|---|")
    for grp in doc["matching"]:
        do = grp.get("data_order", {})
        diffs = "; ".join(f"{d['field']} ({d['class']})" for d in grp["differences"]) or "none"
        a(f"| {grp['group']} | {grp['verdict']} | {do.get('steps_with_different_prompt_multiset', 'n/a')} of {do.get('steps_compared', 'n/a')} | "
          f"{do.get('val_calls_compared', 'n/a')} ({do.get('val_calls_with_different_rows', 'n/a')}) | {diffs} |")
    a("")
    for grp in doc["matching"]:
        for d in grp["differences"]:
            vals = "; ".join(f"{k}: {str(v)[:70]}" for k, v in d["values"].items())
            a(f"- {grp['group']}, `{d['field']}`: {vals}")
    a("")
    a("Shared training rollouts within a matched set (multiset overlap of per-rollout response hashes; 1.0 = identical batches). "
      "Same init, prompts, rollout seed and GPU numerics give identical step-1 rollouts (common random numbers); the arms then diverge "
      "through their different updates.")
    a("")
    steps_shown = ["1", "2", "3", "5", "10", "20", "50", "100", "200", "300"]
    a("| matched set | steps with identical rollouts | " + " | ".join(f"step {x}" for x in steps_shown) + " |")
    a("|---|---|" + "---:|" * len(steps_shown))
    for grp in doc["matching"]:
        ro = grp.get("rollouts") or {}
        sb = ro.get("shared_fraction_by_step") or {}
        a(f"| {grp['group']} | {ro.get('n_steps_identical', 'n/a')} ({', '.join(str(x) for x in ro.get('steps_with_identical_rollouts', []))}) | "
          + " | ".join(num((sb.get(x) or {}).get("mean_pairwise_shared_fraction"), 3) for x in steps_shown) + " |")
    a("")
    for label, v in doc.get("rollout_cross_checks", {}).items():
        a(f"- Across sets, {label}: same step-1 prompts {v['step1_same_prompts']}, share of identical step-1 rollouts {num(v['step1_shared_fraction'], 3)}.")
    a("")
    a("Reading: the overlap counts identical response texts, so it also includes answers that any low-entropy policy reproduces (short, "
      "compact JSON certificates); the 3B-vs-7B row (different weights, same prompts and seed) is the reference for that chance agreement. "
      "Exact step-1 identity within every matched set, and its loss across GPU types at the same seed, confirm that the arms of a seed share "
      "their sampling noise and that the hardware class is part of the training seed (DESIGN 7.3: hardware is nested in training seed). "
      "Cross-set comparisons (for example P_gated seed 1 on RTX PRO 6000 vs P seed 1 on H200 NVL) are therefore not matched.")
    a("")
    a("Classes: `math` = a training input (config, data, init, steps, batch, rollouts, hardware class); `provenance` = code/environment identity or "
      "placement that does not enter the resolved EasyR1 config. The resolved EasyR1 config is compared key by key with the arm keys "
      "(experiment name, save path, reward arm, details dir) removed; `gated` is compared and must agree within a set.")
    a("")
    # ---------------------------------------------------------------- compute
    a("## Compute")
    a("")
    a("### RL runs")
    a("")
    a("| run | GPU x N | host(s) | mode | wall h | GPU-h | s/step median (mean) | rollouts | response tokens (M) | mean resp len | prompt tokens (M) | host RAM unreclaimable / cgroup peak / limit GB | torch reserved max GB |")
    a("|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|")
    for k, r in doc["rl_runs"].items():
        c = r.get("compute") or {}
        if not c:
            a(f"| {k} | pending |")
            continue
        hr = c["host_ram"]
        gpu = short_gpu(c["gpu_name"])
        share = f" (shared x{hr['share']})" if hr.get("share") and hr["share"] > 1 else ""
        mrl = c["response_tokens_exact"] / c["train_rollouts_exact"] if c.get("train_rollouts_exact") else None
        a(f"| {k} | {gpu} x {c['n_gpus']} | {', '.join(c['hosts'])} | {c['mode']} | {num(c['wall_hours_train'], 2)} | {num(c['gpu_hours_train'], 1)} | "
          f"{num(c['s_per_step_median'], 1)} ({num(c['s_per_step_mean'], 1)}) | {c['train_rollouts_exact']} | {tokm(c['response_tokens_exact'])} | {num(mrl, 1)} | "
          f"{tokm(c['prompt_tokens_from_log_mean'])} | {gb(hr['unreclaimable_used_gb_max'])} / {gb(hr['cgroup_peak_gb_max'])} / {gb(hr['limit_gb'])}{share} | "
          f"{gb((c.get('gpu_memory') or {}).get('torch_max_reserved_gb'))} |")
    a("")
    t = doc["compute_totals"]
    a(f"RL total: {num(t['rl_total_gpu_hours'], 1)} GPU-hours (trainer wall x GPUs), {tokm(t['rl_total_response_tokens'])} M generated rollout tokens. By GPU type: "
      + "; ".join(f"{k}: {v['runs']} runs, {num(v['gpu_hours'], 1)} GPU-h" for k, v in t["by_gpu"].items()) + ".")
    a("")
    a("Per model size and GPU type (median of the runs' median s/step; range over runs): " + "; ".join(
        f"{g}: {len(v['runs'])} runs, {num(v['median_of_run_medians'], 1)} s/step ({num(v['min'], 1)}-{num(v['max'], 1)}), "
        f"{num(v['wall_hours_min'], 2)}-{num(v['wall_hours_max'], 2)} wall h, {num(v['gpu_hours_min'], 1)}-{num(v['gpu_hours_max'], 1)} GPU-h"
        for g, v in t["s_per_step_median_by_size_gpu"].items()) + ".")
    a("")
    a("3B rollout tokens per arm (sum over seeds 1-3; per-seed mean response length): "
      + "; ".join(f"{arm}: {tokm(v['response_tokens'])} M ({', '.join(num(x, 1) for x in v['mean_resp_len_per_seed'])})" for arm, v in t["by_arm_3b"].items()) + ".")
    a("")
    a("Wall hours run from the trainer attempt's start to its end (receipt `attempts`); merges, the vLLM load check and evaluation are not included. "
      "Rollout tokens are the exact sums of `response_length` over the 300 x 512 training rollouts in `reward_details` (checked against "
      "`response_length/mean` x 512 of the trainer log; see `scan_log_consistency` in the JSON). In pair mode the two runs of a pair share one "
      "host-RAM cgroup, so their host-RAM figures are the pair's.")
    a("")
    a("Timing breakdown (hours summed over steps):")
    a("")
    keys = ["gen", "old", "update_actor", "reward", "save_checkpoint", "validation"]
    a("| run | " + " | ".join(keys) + " | sum of step times |")
    a("|---|" + "---:|" * (len(keys) + 1))
    for k, r in doc["rl_runs"].items():
        c = r.get("compute") or {}
        th = c.get("timing_hours_sum") or {}
        a(f"| {k} | " + " | ".join(num(th.get(x), 2) for x in keys) + f" | {num(c.get('step_time_hours_sum'), 2)} |")
    a("")
    a("### SFT (shared warm starts)")
    a("")
    a("| model | GPU x N | host | lr | steps | s/step median | train wall h | GPU-h (train) | GPU-h incl. dev eval | label tokens (M) | sequence tokens (M) | loss first10 -> last10 | selected step (rule outcome) |")
    a("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for size, s in doc["sft"].items():
        if s.get("status") != "ok":
            a(f"| {size} | pending |")
            continue
        a(f"| {s['model']} | {(s['gpu_name'] or '?').replace('NVIDIA ', '')} x {s['n_gpus']} | {s['host']} | {s['lr']} | {s['steps']} | {num(s['s_per_step_median'], 2)} | "
          f"{num(s['wall_hours_train'], 2)} | {num(s['gpu_hours_train'], 2)} | {num(s['gpu_hours_incl_dev_eval'], 2)} | {tokm(s['label_tokens'])} | {tokm(s['seq_tokens'])} | "
          f"{num(s['loss_first10'], 3)} -> {num(s['loss_last10'], 3)} | {s['selected_step']} ({s['outcome']}) |")
    a("")
    a(f"Training total (RL trainer + SFT incl. dev evaluation): {num(t['training_total_gpu_hours'], 1)} GPU-hours.")
    a("")
    a("### SFT dev curves and selection (greedy E1-dev, 512 rows)")
    a("")
    for size, s in doc["sft"].items():
        if s.get("status") != "ok":
            continue
        sel = s.get("selection") or {}
        a(f"**{s['model']}** — rule: {s['rule']['text'].rstrip('.')}. Selected: step {s['selected_step']} ({s['outcome']}). "
          f"S evaluation manifest matches the selected weights: {s.get('s_eval_manifest', {}).get('safetensors_equal_selected')}. "
          f"Step {s['selected_step']} is the final save (saves evaluated: {', '.join(str(x) for x in sel.get('evaluated_saves', []))}; "
          f"{sel.get('train_steps_logged')} optimizer steps logged): {sel.get('selected_is_final_save')}. First qualifying save: "
          f"{sel.get('first_qualifying_step')}." + (f" So {sel['note']}." if sel.get("note") else ""))
        a("")
        a("| step | epoch | A | G | M | F | joint | tokens | dA (pp) | dM (pp) | qualifies | reasons |")
        a("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for p in s["dev_curve"]:
            a(f"| {p['step']} | {p['epoch']} | {pct(p['A'])} | {pct(p['G'])} | {pct(p['M'])} | {pct(p['F'])} | {pct(p['joint'])} | {num(p['tokens_mean'], 1)} | "
              f"{pp(p['dA'])} | {pp(p['dM'])} | {p['qualifies']} | {'; '.join(p['reasons'] or [])} |")
        a("")
    # ---------------------------------------------------------------- reward curves
    a("## RL reward curves")
    a("")
    a("Train-batch means from `experiment_log.jsonl` (512 rollouts per step, temperature 1.0). `overall` is the arm's own reward, so it is "
      "not comparable across arms; A, G, M, F are. Figures: `reward_curves_3b.svg`, `reward_curves_7b_gated.svg`; per-step values: `reward_curves.csv`.")
    a("")
    a("| run | own reward first10 -> last10 | A | G | M | F | joint | resp len | zero-variance groups (own reward) | best val step (own) | tracker best |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in doc["rl_runs"].items():
        cs = r.get("curve_summary") or {}
        if not cs:
            a(f"| {k} | pending |")
            continue

        def fl(m: str, scale: str = "pct") -> str:
            v = cs[m]
            if scale == "num":
                return f"{num(v['first10'], 1)} -> {num(v['last10'], 1)}"
            return f"{pct(v['first10'])} -> {pct(v['last10'])} ({pp(v['delta'], 1)})"

        zv = (r.get("grpo_signal") or {}).get("zero_var_group_share") or {}
        bv = r["best_val"]
        a(f"| {k} | {num(cs['overall']['first10'])} -> {num(cs['overall']['last10'])} | {fl('A')} | {fl('G')} | {fl('M')} | {fl('F')} | {fl('joint')} | "
          f"{fl('resp_len', 'num')} | {pct(zv.get('first10'))}% -> {pct(zv.get('last10'))}% | "
          f"{(bv['by_own_reward'] or {}).get('step')} | {bv['tracker'].get('best_global_step')} ({bv['tracker'].get('best_val_reward_score')}) |")
    a("")
    a("Note: the training batches of one seed hold the same prompts in every arm, but different seeds draw different prompts, so "
      "first-10 vs last-10 changes in response length (and to a lesser degree in A/G/M) partly reflect the batch's task mix; compare arms within a seed. "
      "Step-1 rollouts are identical across the arms of a matched set (see the matching audit), so first-10 windows share their first step.")
    a("")
    a("3B pooled over seeds 1-3 (first-10 -> last-10 step means, pp change):")
    a("")
    a("| arm | A | G | M | F | joint | resp len |")
    a("|---|---|---|---|---|---|---|")
    for arm, v in doc["reward_curves_3b_pooled"].items():
        if not v.get("n_seeds"):
            continue
        a(f"| {arm} | " + " | ".join(f"{pct(v[m]['first10'])} -> {pct(v[m]['last10'])} ({pp(v[m]['delta'], 1)})" for m in ("A", "G", "M", "F", "joint"))
          + f" | {num(v['resp_len']['first10'], 1)} -> {num(v['resp_len']['last10'], 1)} |")
    a("")
    a("Validation on E1-dev during training (EasyR1 `val_freq` 50, temperature 0.6, n = 1, 512 rows; A / M in %):")
    a("")
    a("| run | " + " | ".join(f"step {s}" for s in (50, 100, 150, 200, 250, 300)) + " | best A step | best M step |")
    a("|---|" + "---|" * 8)
    for k, r in doc["rl_runs"].items():
        vals = {v["step"]: v for v in r.get("val", [])}
        cells = [f"{pct(vals[s]['A'])} / {pct(vals[s]['M'])}" if s in vals else "n/a" for s in (50, 100, 150, 200, 250, 300)]
        bv = r.get("best_val") or {}
        a(f"| {k} | " + " | ".join(cells) + f" | {(bv.get('by_A') or {}).get('step')} | {(bv.get('by_M') or {}).get('step')} |")
    a("")
    a("Validation own reward (the arm's formula, so comparable within an arm only) and the best validation step:")
    a("")
    a("| run | " + " | ".join(f"step {s}" for s in (50, 100, 150, 200, 250, 300)) + " | best (own) |")
    a("|---|" + "---:|" * 7)
    for k, r in doc["rl_runs"].items():
        vals = {v["step"]: v for v in r.get("val", [])}
        bo = (r.get("best_val") or {}).get("by_own_reward") or {}
        a(f"| {k} | " + " | ".join(num(vals[s]["overall"], 4) if s in vals else "n/a" for s in (50, 100, 150, 200, 250, 300))
          + f" | {bo.get('step')} ({num(bo.get('value'), 4)}) |")
    a("")
    tc_ok = sum(1 for r in doc["rl_runs"].values() if ((r.get("best_val") or {}).get("tracker_check") or {}).get("status") == "ok")
    tc_n = sum(1 for r in doc["rl_runs"].values() if (r.get("best_val") or {}).get("tracker_check"))
    last_best = [k for k, r in doc["rl_runs"].items() if ((r.get("best_val") or {}).get("tracker_check") or {}).get("last_step_is_best_validation")]
    a("How to read the tracker column: EasyR1's `fit` loop saves the checkpoint of step s before it validates step s, and `_save_checkpoint` "
      "credits the save with the latest validation score. With save_freq 25 and val_freq 50, the save at step s + 25 carries the validation of "
      "step s, and the save at step 300 carries the step-250 score, which step 275 already holds (ties keep the earlier step). The tracker's "
      "best step is therefore the best validation among steps 50-250 shifted by +25, and it can never be step 300; it says nothing about "
      f"whether step 300 is past the optimum. This accounts for the tracker value of {tc_ok} of {tc_n} runs (`best_val.tracker_check`). "
      "By the runs' own validation reward, step 300 is the best validation step for "
      + (", ".join(last_best) if last_best else "no run") + ". The reported checkpoint is step 300 by pre-registration (DESIGN 7.3).")
    a("")
    a("Every reward formula on the same training rollouts (last 10 steps; the arm's own formula is the one it optimised):")
    a("")
    a("| run | R_A | R_G | R_P | R_P_gated | R_G_gated |")
    a("|---|---:|---:|---:|---:|---:|")
    for k, r in doc["rl_runs"].items():
        rb = r.get("reward_by_formula") or {}
        if not rb:
            continue
        a(f"| {k} | " + " | ".join(num(rb[f]["last10"]) for f in ("A", "G", "P", "P_gated", "G_gated")) + " |")
    a("")
    # ---------------------------------------------------------------- E1-dev step curves
    a("## E1-dev by RL step (cert_v1, trace preset, decoding seeds 42-44; step 0 = S)")
    a("")
    a("E1-dev is the RL validation split, so this section is descriptive (no family, no Holm). Each value is the mean over 512 rows of the "
      "per-row mean over the three decoding seeds. Figure: `e1dev_steps_3b.svg`; all rows: `e1dev_steps.csv`.")
    a("")
    for size_label, keys in (("3B", [k for k in doc["e1dev_steps"] if k.startswith("3b/")]), ("7B", [k for k in doc["e1dev_steps"] if k.startswith("7b/")])):
        a(f"### {size_label} levels (A / M / joint in %, tokens)")
        a("")
        a("| run | step 0 (S) | step 100 | step 200 | step 300 | step 300 vs S: dA [95% CI] | McNemar p (A, seed 42; S-only / step-only) | dM [95% CI] | dG [95% CI] | dtokens [95% CI] |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            e = doc["e1dev_steps"][k]

            def cell(s: str) -> str:
                m = (e.get(s) or {}).get("metrics")
                if not m:
                    return "pending"
                return f"{pct(m['A'])} / {pct(m['M'])} / {pct(m['joint'])} / {num(m['tokens'], 0)}"

            vs = ((e.get("300") or {}).get("vs_S") or {}).get("metrics") or {}

            def dcell(m: str) -> str:
                v = vs.get(m)
                return f"{mdelta(m, v['delta'])} {ci_str(m, v['ci'])}" if v else "pending"

            a(f"| {k} | {cell('0')} | {cell('100')} | {cell('200')} | {cell('300')} | {dcell('A')} | {mcn(vs.get('A'))} | {dcell('M')} | {dcell('G')} | {dcell('tokens')} |")
        a("")
    a("### 3B pooled over training seeds (mean [min, max] of seeds 1-3)")
    a("")
    a("| arm | metric | step 0 (S) | step 100 | step 200 | step 300 |")
    a("|---|---|---|---|---|---|")
    for arm, lv in doc["e1dev_3b_levels"].items():
        for m in ("A", "G", "M", "F", "joint", "tokens"):
            cells = []
            for s in ("0", "100", "200", "300"):
                v = lv[s][m]
                cells.append(f"{mval(m, v['mean'])} [{mval(m, v['min'])}, {mval(m, v['max'])}]" if v["mean"] is not None else "pending")
            a(f"| {arm} | {m} | " + " | ".join(cells) + " |")
    a("")
    a("### 3B by adapter (mean of seeds 1-3; A / M in %)")
    a("")
    a("| adapter | arm | step 0 (S) | step 100 | step 200 | step 300 |")
    a("|---|---|---|---|---|---|")
    bya = doc.get("e1dev_3b_levels_by_adapter") or {}
    ads = sorted({ad for arm in bya for ad in bya[arm]})
    for ad in ads:
        for arm in ("A", "G", "P"):
            row = (bya.get(arm) or {}).get(ad) or {}
            a(f"| {ad} | {arm} | " + " | ".join(f"{pct(row[s]['A'])} / {pct(row[s]['M'])}" if s in row else "pending" for s in ("0", "100", "200", "300")) + " |")
    a("")
    a("### 3B change from S by step, pooled over training seeds")
    a("")
    a("Each cell: mean of the per-seed paired differences, then two 95% CIs and the seed signs. The first CI is the `compare_cert.py` "
      "two-level bootstrap (training seeds resampled, clusters resampled independently within each seed). The second (`shared`) resamples "
      "training seeds and draws one cluster resample that every seed shares. All three seeds are scored on the same 512 E1-dev rows and every "
      "difference here subtracts the same S outputs, so the instance-sampling term (including S's own per-row values) is common to the seeds; "
      "independent per-seed resampling averages that term down by the number of seeds and gives an interval that is too narrow for "
      "differences from a shared baseline. Read the shared-instance CI for these step curves. Per-seed tests: McNemar on seed 42 for A, F and "
      "joint; sign-flip for G, M and tokens.")
    a("")
    a("| arm | step | dA | dG | dM | dF | djoint | dtokens | per-seed McNemar p on A (s1, s2, s3) |")
    a("|---|---:|---|---|---|---|---|---|---|")
    for arm, by in doc["e1dev_3b_vs_S_pooled"].items():
        for st, res in by.items():
            if res.get("status") == "pending":
                a(f"| {arm} | {st} | pending |")
                continue
            a(f"| {arm} | {st} | " + " | ".join(f"{mdelta(m, res[m]['delta'])} {ci_str(m, res[m]['ci_two_level'])} shared {ci_str(m, res[m].get('ci_shared_instance'))} "
                                                 f"({res[m]['n_pos']}+/{res[m]['n_neg']}-)" for m in METRICS)
              + " | " + ", ".join(pval(v) for v in (res["A"].get("per_seed_test_p") or {}).values()) + " |")
    a("")
    a("### Arm differences by step (paired within training seed)")
    a("")
    a("3B: mean of per-seed differences with the `compare_cert.py` two-level CI and the shared-instance CI (see above), how many seeds are "
      "positive / negative on A, and the per-seed McNemar p on A (seed-42 outcome). 7B and the gated pair: one training seed, CI from instance "
      "sampling only; the test on A is the exact McNemar on decoding seed 42 (DESIGN 8; counts in brackets are rows correct only under the "
      "second arm / only under the first arm), with the cluster sign-flip p given second. All "
      "descriptive: about 18 cells, no multiplicity control, and E1-dev is the RL validation split.")
    a("")
    a("| comparison | step | dA | dG | dM | dF | djoint | dtokens | A: seeds (+/-) and test p |")
    a("|---|---:|---|---|---|---|---|---|---|")
    for k, c in doc["e1dev_arm_diffs"].items():
        if c.get("status") == "pending":
            a(f"| {c['size']} {c['pair']} | {c['step']} | pending ({', '.join(c['missing'])}) |")
            continue
        if "pooled" in c:
            po = c["pooled"]
            cells = [f"{mdelta(m, po[m]['delta'])} {ci_str(m, po[m]['ci_two_level'])} shared {ci_str(m, po[m].get('ci_shared_instance'))}" for m in METRICS]
            sg = (f"{po['A']['n_pos']}+/{po['A']['n_neg']}- of {po['A']['n_seeds']}; McNemar p "
                  + ", ".join(pval(v) for v in (po["A"].get("per_seed_test_p") or {}).values()))
        else:
            (sd, d), = c["per_seed"].items()
            cells = [f"{mdelta(m, d['metrics'][m]['delta'])} {ci_str(m, d['metrics'][m]['ci'])}" for m in METRICS]
            sg = f"1 seed (s{sd}); McNemar {mcn(d['metrics']['A'])}; sign-flip p {pval(d['metrics']['A']['perm_p'])}"
        a(f"| {c['size']} {c['pair']} | {c['step']} | " + " | ".join(cells) + f" | {sg} |")
    a("")
    a("Per-seed differences (with McNemar counts and p for A/F/joint and sign-flip p for every metric) are in `training.json` under "
      "`e1dev_arm_diffs`; the vs-S ones under `e1dev_steps.<run>.<step>.vs_S`.")
    a("")
    # ---------------------------------------------------------------- integrity
    a("## Integrity checks")
    a("")
    bad = []
    for k, r in doc["rl_runs"].items():
        for st, v in (r.get("eval_manifest_integrity") or {}).items():
            if v["status"] not in ("ok", "pending"):
                bad.append(f"{k} step {st}: {v}")
        sc = r.get("scan_log_consistency") or {}
        if sc and ((sc.get("max_abs_overall_diff") or 0) > 1e-6 or (sc.get("max_abs_resp_len_diff") or 0) > 1e-6 or sc.get("unmatched_calls")):
            bad.append(f"{k}: reward_details vs trainer log {sc}")
    n_ok = sum(1 for r in doc["rl_runs"].values() for v in (r.get("eval_manifest_integrity") or {}).values() if v["status"] == "ok")
    n_pend = sum(1 for r in doc["rl_runs"].values() for v in (r.get("eval_manifest_integrity") or {}).values() if v["status"] == "pending")
    a(f"- Evaluated checkpoints whose `ckpt_manifest.json` points at `rl/<size>/<arm>/s<seed>/hf/global_step_<step>` with safetensors sha256 equal to the "
      f"merge receipt: {n_ok} ok, {n_pend} pending.")
    a(f"- Reward records vs trainer log (call k = step k+1; mean own reward and mean response length agree to 1e-6): "
      f"{'all runs agree' if not any('reward_details' in b for b in bad) else 'see below'}.")
    for b in bad:
        a(f"- PROBLEM: {b}")
    a("")
    return "\n".join(L) + "\n"


def short_gpu(name: Optional[str]) -> str:
    return (name or "?").replace("NVIDIA ", "").replace(" Blackwell Max-Q Workstation Edition", " Max-Q")


def render_headlines(doc: Mapping[str, Any], a: Callable[[str], None]) -> None:
    a("## Headlines")
    a("")
    g = doc["go_no_go"]
    o = g["1a"]
    if o.get("n_numeric_rows"):
        ex = o.get("share_excluding_rpg") or {}
        a(f"- **Go/no-go 1(a) {'PASS' if o['pass'] else 'FAIL'}**: {o['count_G_ge_thr_and_M_lt_1']}/{o['n_numeric_rows']} = {pct(o['share'])}% of numeric S-checkpoint "
          f"greedy E1-dev rows have G >= 0.9 and M < 1 (threshold 10%; Wilson CI {pct(o['wilson_ci'][0])}-{pct(o['wilson_ci'][1])}%); Spearman(G, M) = "
          f"{num(o['spearman_G_M']['rho'], 3)}. By adapter: " + ", ".join(f"{k} {v['count']}/{v['n']}" for k, v in o["by_adapter"].items())
          + (f"; without rpg {ex['count']}/{ex['n']} = {pct(ex['share'])}%." if ex else "."))
    b = g["1b"]
    if b.get("current"):
        cur = b["runs"][b["current"]]
        a(f"- **Go/no-go 1(b) {'PASS' if b['pass'] else 'FAIL'}** (`{b['current']}`): content corruption types that keep G and lower M: "
          + ", ".join(f"{k} {v['content_types']}" for k, v in cur["by_adapter"].items()) + " (>= 3 needed).")
    mism = [grp for grp in doc["matching"] if grp["verdict"].startswith("MISMATCH")]
    prov = [f"{grp['group']}: " + ", ".join(d["field"].split(".", 1)[1] for d in grp["differences"]) for grp in doc["matching"] if grp["differences"]]
    a(f"- **Matching**: {len(doc['matching']) - len(mism)} of {len(doc['matching'])} matched sets agree on every training input (config, SFT init, data, "
      f"300 steps x 64 prompts x 8 rollouts, GPU type, per-step prompt multiset); provenance-only differences: " + ("; ".join(prov) if prov else "none") + ".")
    t = doc["compute_totals"]
    a(f"- **Compute**: RL {num(t['rl_total_gpu_hours'], 1)} GPU-h over {len(doc['rl_runs'])} runs + SFT {num(t['sft_total_gpu_hours_incl_dev_eval'], 1)} GPU-h "
      "(incl. dev evaluation). Median s/step by model size and GPU type (2 GPUs per run): "
      + "; ".join(f"{g} {num(v['median_of_run_medians'], 1)} ({len(v['runs'])} runs, {num(v['wall_hours_min'], 2)}-{num(v['wall_hours_max'], 2)} h)"
                  for g, v in t["s_per_step_median_by_size_gpu"].items())
      + "; 3B generated rollout tokens per arm (3 seeds): " + ", ".join(f"{arm} {tokm(v['response_tokens'])} M" for arm, v in t["by_arm_3b"].items()) + ".")
    sft = doc.get("sft") or {}
    sel = [f"{sz} step {v['selected_step']} (final save: {(v.get('selection') or {}).get('selected_is_final_save')}, first qualifying: "
           f"{(v.get('selection') or {}).get('first_qualifying_step')})" for sz, v in sft.items() if v.get("status") == "ok"]
    if sel:
        a("- **SFT selection**: " + "; ".join(sel) + ". The rule takes the first save that meets the plateau criterion, else the last save; "
          "in both sizes the first qualifying save is the final save at the end of epoch 3, so rule and fallback coincide.")
    ps = doc.get("e1dev_3b_vs_S_pooled") or {}
    if all(isinstance(ps.get(arm, {}).get("300"), dict) and "A" in ps[arm]["300"] for arm in ("A", "G", "P")):
        a("- **E1-dev step 300 vs S (3B, mean of 3 seeds; shared-instance 95% CI, compare_cert two-level CI in brackets after it)**: " + "; ".join(
            f"arm {arm}: dA {mdelta('A', ps[arm]['300']['A']['delta'])} {ci_str('A', ps[arm]['300']['A']['ci_shared_instance'])} "
            f"({ci_str('A', ps[arm]['300']['A']['ci_two_level'])}; McNemar p per seed " + ", ".join(pval(v) for v in ps[arm]['300']['A']['per_seed_test_p'].values()) + "), "
            f"dM {mdelta('M', ps[arm]['300']['M']['delta'])} {ci_str('M', ps[arm]['300']['M']['ci_shared_instance'])} ({ci_str('M', ps[arm]['300']['M']['ci_two_level'])}), "
            f"dG {mdelta('G', ps[arm]['300']['G']['delta'])} {ci_str('G', ps[arm]['300']['G']['ci_shared_instance'])} ({ci_str('G', ps[arm]['300']['G']['ci_two_level'])})"
            for arm in ("A", "G", "P")) + ".")
    pg = (doc["e1dev_arm_diffs"].get("3b P-G step 300") or {}).get("pooled")
    if pg:
        a(f"- **E1-dev P - G (3B, step 300)**: A {mdelta('A', pg['A']['delta'])} shared {ci_str('A', pg['A']['ci_shared_instance'])} / two-level "
          f"{ci_str('A', pg['A']['ci_two_level'])} ({pg['A']['n_pos']}+/{pg['A']['n_neg']}- seeds; McNemar p per seed "
          + ", ".join(pval(v) for v in pg["A"]["per_seed_test_p"].values()) + f"), M {mdelta('M', pg['M']['delta'])} shared {ci_str('M', pg['M']['ci_shared_instance'])}, "
          f"joint {mdelta('joint', pg['joint']['delta'])} shared {ci_str('joint', pg['joint']['ci_shared_instance'])} (descriptive: E1-dev is the RL validation split).")
    for size, pair in (("7b", "P-G"), ("3b", "P_gated-G_gated")):
        cells = []
        for st in STEPS:
            c = doc["e1dev_arm_diffs"].get(f"{size} {pair} step {st}") or {}
            if c.get("per_seed"):
                (sd, d), = c["per_seed"].items()
                mA = d["metrics"]["A"]
                cells.append(f"step {st} dA {mdelta('A', mA['delta'])} {ci_str('A', mA['ci'])}, McNemar {mcn(mA)}")
            else:
                cells.append(f"step {st} pending")
        a(f"- **E1-dev {pair} ({size.upper()}, one training seed, instance-sampling CI)**: " + "; ".join(cells) + ". Descriptive; no single step is evidence for either arm.")
    gg = doc["e1dev_steps"].get("3b/G_gated/s1") or {}
    pgt = doc["e1dev_steps"].get("3b/P_gated/s1") or {}
    gv = ((gg.get("300") or {}).get("vs_S") or {}).get("metrics") or {}
    pv = ((pgt.get("300") or {}).get("vs_S") or {}).get("metrics") or {}
    if gv and pv:
        a(f"- **Gated pair, step 300 vs S (3B seed 1, RTX PRO 6000)**: P_gated dA {mdelta('A', pv['A']['delta'])} {ci_str('A', pv['A']['ci'])} (McNemar {mcn(pv['A'])}), "
          f"dM {mdelta('M', pv['M']['delta'])} {ci_str('M', pv['M']['ci'])}; G_gated dA {mdelta('A', gv['A']['delta'])} {ci_str('A', gv['A']['ci'])} "
          f"(McNemar {mcn(gv['A'])}), dM {mdelta('M', gv['M']['delta'])} {ci_str('M', gv['M']['ci'])}.")
    tk = doc.get("e1dev_tokens_vs_S") or {}
    if tk.get("max_abs_pooled_3b"):
        a(f"- **Completion tokens on E1-dev** (S: 3B {num(tk['S_tokens'].get('3b'), 1)}, 7B {num(tk['S_tokens'].get('7b'), 1)}): seed-pooled 3B changes from S stay within "
          f"{num(tk['max_abs_pooled_3b']['max_abs'], 2)} tokens; single runs reach {num(tk['max_abs_per_run_any_step']['value'], 2)} "
          f"({tk['max_abs_per_run_any_step']['at']}), and within {num(tk['max_abs_per_run_step300']['max_abs'], 2)} at step 300.")
    tcs = [((r.get("best_val") or {}).get("tracker_check") or {}) for r in doc["rl_runs"].values()]
    last_best = [k for k, r in doc["rl_runs"].items() if ((r.get("best_val") or {}).get("tracker_check") or {}).get("last_step_is_best_validation")]
    a(f"- **Best validation step**: EasyR1's tracker best (steps {', '.join(sorted({str(r['best_val']['tracker'].get('best_global_step')) for r in doc['rl_runs'].values()}, key=int))}) "
      f"is never step 300 by construction (save before validate; reproduced for {sum(1 for x in tcs if x.get('status') == 'ok')} of {len(tcs)} runs), so it is not a "
      "finding about over-training. By the own validation reward, step 300 is the best validation step for " + (", ".join(last_best) or "no run") + ".")
    pc = doc.get("reward_curves_3b_pooled") or {}
    zv = {}
    for arm in ("A", "G", "P"):
        vals = [r.get("grpo_signal", {}).get("zero_var_group_share", {}) for r in doc["rl_runs"].values() if r["size"] == "3b" and r["arm"] == arm]
        zv[arm] = (mean(v.get("first10") for v in vals), mean(v.get("last10") for v in vals))
    if pc.get("A", {}).get("n_seeds"):
        a("- **Training reward (3B, first-10 -> last-10 train-batch means, 3 seeds)**: " + "; ".join(
            f"{arm}: A {pct(pc[arm]['A']['first10'])} -> {pct(pc[arm]['A']['last10'])}, M {pct(pc[arm]['M']['first10'])} -> {pct(pc[arm]['M']['last10'])}, "
            f"G {pct(pc[arm]['G']['first10'])} -> {pct(pc[arm]['G']['last10'])}, zero-variance prompt groups {pct(zv[arm][0])}% -> {pct(zv[arm][1])}%" for arm in ("A", "G", "P")) + ".")
    a("")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--scan", required=True, help="rollout_scan.json from training_rollout_scan.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--md-only", action="store_true", help="re-render training.md from an existing training.json")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "training.json"
    if not args.md_only:
        doc = build(args)
        tmp = out_dir / f".training.json.tmp.{os.getpid()}"
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=False, allow_nan=False, default=str), encoding="utf-8")
        os.replace(tmp, jpath)
        print(f"wrote {jpath}")
    doc = read_json(jpath)
    (out_dir / "training.md").write_text(render_md(doc), encoding="utf-8")
    print(f"wrote {out_dir / 'training.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
