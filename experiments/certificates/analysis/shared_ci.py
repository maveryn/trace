"""Shared-instance confidence intervals for the pooled 3B certificate claims (results/final/cert_3b_shared_ci).

    python shared_ci.py --model P@1=<eval/ckpt/root> ... --model S=<root> \\
        --cmp-main cmp_main.json --extras-main extras_main.json --out-json shared_ci.json --out-md shared_ci.md

Why. ``compare_cert.two_level_bootstrap`` resamples training seeds and then resamples ``scene_group``
clusters INDEPENDENTLY for every drawn seed. All training seeds are evaluated on the SAME instances, so
instance effects are shared across seeds (fully so for X - S, where the single S checkpoint is the Y
side of every seed's difference). Independent cluster draws ignore that correlation, and the pooled
CI can be too narrow. This script recomputes, for a fixed list of claims, three CIs of the same
estimate (the mean over training seeds of the per-seed paired difference):

(a) ``compare_cert``: ``compare_cert.two_level_bootstrap`` called unchanged on ``compare_cert``'s own
    pairing and clusters, with its RNG stream (``default_rng([seed, 3])``). Its estimate and CI are
    checked against ``cmp_main.json`` (A, M, joint) or ``extras_main.json`` (error classes) to confirm
    that the pairing is the published one.
(b) shared-instance two-level bootstrap: every replicate draws ONE cluster resample, stratified by
    adapter (the same multinomial counts for every training seed), and independently draws K training
    seeds with replacement; the replicate value is the mean over the drawn seeds of each seed's
    row-weighted mean paired difference on that one resample. Cluster counts come from
    ``default_rng([seed, 1])`` drawn exactly as ``compare_cert.cluster_bootstrap`` draws them; seed
    picks from ``default_rng([seed, 2])``.
(c) instance-only: the unit is the row's paired difference averaged over the K training seeds; a
    stratified cluster bootstrap of its row-weighted mean, using the same cluster counts as (b)
    (so (b) and (c) differ only by the training-seed resampling). This CI ignores seed-to-seed
    (training) variance and is a lower reference, not a replacement for (a) or (b).

Percentile 95% CIs. Sign agreement is compare_cert's: the number of training seeds whose per-seed
difference has the sign of the pooled estimate (2 of 3 needed). The error class is final_cert3b.py's
(``cert_values``: the scorer's ``error_class`` field, i.e. the first failing stage, as a 0/1 indicator;
missing -> NaN). Rows, decoding seeds (42-44 averaged per row, those present in both arms) and clusters
are compare_cert's (``pair_runs`` / ``make_clusters``), imported read-only; compare_cert.py and
final_cert3b.py are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_cert as cc  # noqa: E402  (read-only reuse)
import final_cert3b as fc  # noqa: E402  (read-only reuse: loader and error-class definition)

VERSION = "shared_ci-1.0.0"

METRIC_NAMES = ("A", "M", "joint", "ec_grounding")

# (family / role, X, Y, split, metric, reference file key)
CLAIMS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("primary", "P", "G", "e1_test", "A"),
    ("primary", "P", "G", "e2_test", "A"),
    ("secondary", "P", "A", "e1_test", "A"),
    ("secondary", "P", "A", "e2_test", "A"),
    ("secondary", "G", "A", "e1_test", "A"),
    ("secondary", "G", "A", "e2_test", "A"),
    ("descriptive", "P", "G", "e1_test", "M"),
    ("descriptive", "P", "G", "e1_test", "joint"),
    ("SFT contribution", "A", "S", "e1_test", "A"),
    ("SFT contribution", "A", "S", "e2_test", "A"),
    ("SFT contribution", "G", "S", "e1_test", "A"),
    ("SFT contribution", "G", "S", "e2_test", "A"),
    ("SFT contribution", "P", "S", "e1_test", "A"),
    ("SFT contribution", "P", "S", "e2_test", "A"),
    ("RQ3 error class", "P", "A", "e1_test", "ec_grounding"),
    ("RQ3 error class", "P", "S", "e1_test", "ec_grounding"),
)
REPRO_TOL = 1e-9


def _f(x: Any) -> Optional[float]:
    return cc._f(x)


def excludes0(ci: Sequence[Optional[float]]) -> bool:
    return ci[0] is not None and ci[1] is not None and (ci[0] > 0 or ci[1] < 0)


def pct_ci(boot: np.ndarray) -> List[Optional[float]]:
    lo, hi = cc.percentile_ci(boot.reshape(-1, 1))
    return [_f(lo[0]), _f(hi[0])]


def draw_cluster_counts(stratum: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 1000) -> np.ndarray:
    """(n_boot, C) stratified multinomial cluster counts (uint16), drawn in compare_cert.cluster_bootstrap's order."""
    W = np.zeros((n_boot, len(stratum)), dtype=np.uint16)
    for h in sorted(set(stratum.tolist())):
        idx = np.flatnonzero(stratum == h)
        k = len(idx)
        for b0 in range(0, n_boot, chunk):
            b1 = min(n_boot, b0 + chunk)
            W[b0:b1, idx] = rng.multinomial(k, np.full(k, 1.0 / k), size=b1 - b0)
    return W


def wdot(W: np.ndarray, A: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """W @ A in float64, converting the uint16 counts one block of replicates at a time (low memory)."""
    out = np.empty((W.shape[0],) + A.shape[1:])
    for b0 in range(0, W.shape[0], chunk):
        out[b0:b0 + chunk] = W[b0:b0 + chunk].astype(float) @ A
    return out


def reference_result(ref: Dict[str, Any], x: str, y: str, split: str, metric: str) -> Optional[Dict[str, Any]]:
    doc = ref["extras"]["cert"] if metric.startswith("ec_") else ref["cmp"]
    for c in doc.get("comparisons", []):
        if c["x"] == x and c["y"] == y:
            return ((c["splits"].get(split) or {}).get("scopes", {}).get("all") or {}).get(metric)
    return None


def corr_offdiag(Sk: np.ndarray, Nk: np.ndarray) -> Optional[float]:
    """Mean pairwise Pearson correlation across training seeds of the per-cluster mean differences."""
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(Nk > 0, Sk / np.where(Nk > 0, Nk, 1.0), np.nan)  # (C, K)
    ok = ~np.isnan(D).any(axis=1)
    D = D[ok]
    K = D.shape[1]
    if K < 2 or D.shape[0] < 3:
        return None
    r = []
    for a in range(K):
        for b in range(a + 1, K):
            if np.std(D[:, a]) > 0 and np.std(D[:, b]) > 0:
                r.append(float(np.corrcoef(D[:, a], D[:, b])[0, 1]))
    return float(np.mean(r)) if r else None


def moment_diagnostic(D: np.ndarray, strat: np.ndarray, singleton: bool) -> Dict[str, Any]:
    """Crossed random-effects decomposition d[i, k] = mu + a_k + b_i + e_ik (strata fixed), balanced rows only.

    Diagnostic for reading the three bootstrap widths, not a CI. With MS_A (seeds), MS_B (instances, centred
    within stratum) and MS_E (residual), Var(estimate) = s2_a/K + s2_b/n + s2_e/(K n) is estimated without bias
    by (MS_A + MS_B - MS_E)/(n K). The approximate variances each bootstrap targets are:
      compare_cert   (K-1)/K^2 MS_A/n + (s2_b + s2_e)/(K n)     (instance effects counted once per seed: s2_b/K)
      shared         (K-1)/K^2 MS_A/n + s2_b/n + (2K-1)/K^2 s2_e/n  (a repeated seed reuses the same resample)
      instance-only  MS_B/(K n)                                  (no training-seed variance)
    """
    if not singleton:
        return {"status": "n/a (clusters are not single rows)"}
    ok = ~np.isnan(D).any(axis=1)
    D, strat = D[ok], strat[ok]
    n, K = D.shape
    if K < 2 or n < 3:
        return {"status": "n/a"}
    g = D.mean()
    ck, ri = D.mean(axis=0), D.mean(axis=1)
    hs, hidx = np.unique(strat.astype(str), return_inverse=True)
    gh = np.bincount(hidx, weights=ri) / np.bincount(hidx)
    ms_a = n * float(((ck - g) ** 2).sum()) / (K - 1)
    ms_b = K * float(((ri - gh[hidx]) ** 2).sum()) / (n - len(hs))
    ms_e = float(((D - ri[:, None] - ck[None, :] + g) ** 2).sum()) / ((n - 1) * (K - 1))
    s2_e, s2_a, s2_b = ms_e, (ms_a - ms_e) / n, (ms_b - ms_e) / K
    var = (ms_a + ms_b - ms_e) / (n * K)
    pred_cc = (K - 1) / K ** 2 * ms_a / n + (s2_b + s2_e) / (K * n)
    pred_sh = (K - 1) / K ** 2 * ms_a / n + s2_b / n + (2 * K - 1) / K ** 2 * s2_e / n
    pred_in = ms_b / (K * n)
    sq = lambda v: math.sqrt(v) if v > 0 else 0.0  # noqa: E731
    parts = {"seed": s2_a / K, "instance": s2_b / n, "interaction": s2_e / (K * n)}
    return {"status": "ok", "n_rows": int(n), "ms_seed": ms_a, "ms_instance": ms_b, "ms_resid": ms_e,
            "s2_seed": s2_a, "s2_instance": s2_b, "s2_interaction": s2_e,
            "se_moment": sq(var), "var_share": {k: (v / var if var > 0 else None) for k, v in parts.items()},
            "predicted_sd": {"compare_cert": sq(pred_cc), "shared": sq(pred_sh), "instance_only": sq(pred_in)}}


def analyse_pair(ds: cc.Dataset, x: str, y: str, split: str, metrics: Sequence[str], st: cc.Settings) -> Dict[str, Any]:
    """Every CI for every requested metric of one (X, Y, split)."""
    tps = cc.tseed_pairs(ds.runs[x], ds.runs[y])
    paired = [(tp, cc.pair_runs(ds, ds.runs[x][tp[0]], ds.runs[y][tp[1]], split, st.mcnemar_seed)) for tp in tps]
    keys0 = paired[0][1].keys
    for tp, p in paired:
        if p.keys != keys0:
            raise cc.CompareError(f"{x} - {y} {split}: training-seed pair {tp} is paired on different rows "
                                  "(the shared-instance bootstrap needs one common row set)")
    K = len(paired)
    cls = [cc.make_clusters(p.X, p.Y, p.cluster, p.stratum) for _, p in paired]
    stratum = cls[0].stratum
    C = len(stratum)
    # (a) compare_cert, unchanged (RNG stream of compare_scope for K > 1)
    boot_a = cc.two_level_bootstrap(cls, st.n_boot, np.random.default_rng([st.seed, 3])) if K > 1 else None
    # shared cluster counts for (b) and (c); seed picks for (b)
    W = draw_cluster_counts(stratum, st.n_boot, np.random.default_rng([st.seed, 1]))
    pick = np.random.default_rng([st.seed, 2]).integers(0, K, size=(st.n_boot, K))
    out: Dict[str, Any] = {
        "training_seed_pairs": [list(tp) for tp in tps],
        "pairing": [{"tseeds": list(tp), "rows": len(p.keys), "x_only": p.x_only, "y_only": p.y_only,
                     "decoding_seeds": p.seeds} for tp, p in paired],
        "n_clusters": C, "n_strata": len(set(stratum.tolist())), "metrics": {},
    }
    for m in metrics:
        j = ds.metrics.index(m)
        Sk = np.stack([cl.S[:, j] for cl in cls], axis=1)  # (C, K)
        Nk = np.stack([cl.N[:, j] for cl in cls], axis=1)
        per_seed = [_f(cc._ratio(np.array([Sk[:, k].sum()]), np.array([Nk[:, k].sum()]))[0]) for k in range(K)]
        delta = float(np.mean(per_seed)) if all(d is not None for d in per_seed) else None
        sgn = 0 if delta is None else (1 if delta > 0 else -1 if delta < 0 else 0)
        agree = sum(1 for d in per_seed if d is not None and sgn and d * sgn > 0)
        need = cc.sign_needed(K)
        res: Dict[str, Any] = {"K": K, "per_seed_delta": per_seed, "delta": delta, "sign_agree": agree, "sign_needed": need,
                               "n_rows_per_seed": [int(Nk[:, k].sum()) for k in range(K)],
                               "cluster_corr_across_seeds": corr_offdiag(Sk, Nk)}
        # (a)
        if boot_a is not None:
            ci_a = pct_ci(boot_a[:, j])
            res["compare_cert"] = {"ci": ci_a, "excludes0": excludes0(ci_a), "claim": bool(excludes0(ci_a) and agree >= need),
                                   "method": "compare_cert.two_level_bootstrap (seeds, then clusters independently per drawn seed)"}
        # (b) shared instances
        with np.errstate(invalid="ignore", divide="ignore"):
            est = cc._ratio(wdot(W, Sk), wdot(W, Nk))  # (B, K): each seed's mean difference on the SAME resample
        boot_b = est[np.arange(st.n_boot)[:, None], pick].mean(axis=1)
        ci_b = pct_ci(boot_b)
        res["shared"] = {"ci": ci_b, "excludes0": excludes0(ci_b), "claim": bool(excludes0(ci_b) and agree >= need),
                         "nan_replicates": int(np.isnan(boot_b).sum()),
                         "method": "one stratified cluster resample shared by all seeds; seeds resampled with replacement"}
        # (c) instance only: row unit = paired difference averaged over the K seeds
        D = np.stack([p.X[:, j] - p.Y[:, j] for _, p in paired], axis=1)  # (R, K)
        valid = ~np.isnan(D)
        cnt = valid.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dbar = np.where(cnt > 0, np.where(valid, D, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        _, idx = np.unique(paired[0][1].cluster.astype(str), return_inverse=True)
        Sb, Nb = np.zeros(C), np.zeros(C)
        okr = ~np.isnan(dbar)
        np.add.at(Sb, idx[okr], dbar[okr])
        np.add.at(Nb, idx[okr], 1.0)
        boot_c = cc._ratio(wdot(W, Sb), wdot(W, Nb))
        ci_c = pct_ci(boot_c)
        # self-check: identical to compare_cert.cluster_bootstrap on the seed-averaged rows with default_rng([seed, 1])
        cl_bar = cc.Clusters(Sb[:, None], Nb[:, None], np.zeros((C, 1)), np.zeros((C, 1)), stratum)
        chk = cc.cluster_bootstrap(cl_bar, st.n_boot, np.random.default_rng([st.seed, 1]))[:, 0]
        res["instance_only"] = {"delta": _f(Sb.sum() / Nb.sum()) if Nb.sum() > 0 else None, "ci": ci_c,
                                "excludes0": excludes0(ci_c), "n_rows": int(Nb.sum()),
                                "rows_partial_seeds": int(((cnt > 0) & (cnt < K)).sum()),
                                "matches_cc_cluster_bootstrap": bool(np.allclose(boot_c, chk, equal_nan=True, atol=1e-12)),
                                "method": "stratified cluster bootstrap of the 3-seed-mean paired difference per row"}
        res["variance_diagnostic"] = moment_diagnostic(D, paired[0][1].stratum, singleton=(C == len(keys0)))
        res["_arrays"] = (Sk, Nk, stratum, K)
        if boot_a is not None:
            wa = (ci_a[1] - ci_a[0]) if None not in ci_a else None
            wb = (ci_b[1] - ci_b[0]) if None not in ci_b else None
            wc = (ci_c[1] - ci_c[0]) if None not in ci_c else None
            res["width"] = {"compare_cert": wa, "shared": wb, "instance_only": wc,
                            "shared_over_compare_cert": (wb / wa) if wa and wb is not None else None}
            res["boot_sd"] = {"compare_cert": float(np.nanstd(boot_a[:, j])), "shared": float(np.nanstd(boot_b)),
                              "instance_only": float(np.nanstd(boot_c))}
        out["metrics"][m] = res
    return out


def shared_only(Sk: np.ndarray, Nk: np.ndarray, stratum: np.ndarray, K: int, n_boot: int, seed: int) -> List[Optional[float]]:
    """The shared-instance CI (b) recomputed with another RNG seed (Monte Carlo check)."""
    W = draw_cluster_counts(stratum, n_boot, np.random.default_rng([seed, 1]))
    pick = np.random.default_rng([seed, 2]).integers(0, K, size=(n_boot, K))
    est = cc._ratio(wdot(W, Sk), wdot(W, Nk))
    return pct_ci(est[np.arange(n_boot)[:, None], pick].mean(axis=1))


def reproduction(res: Dict[str, Any], ref: Optional[Dict[str, Any]], n_boot_ref: int, n_boot: int) -> Dict[str, Any]:
    if ref is None:
        return {"status": "no reference row"}
    mine_ci = res["compare_cert"]["ci"]
    ref_per = [ps["delta"] for ps in ref.get("per_seed", [])]
    d_delta = abs(res["delta"] - ref["delta"])
    d_seed = max(abs(a - b) for a, b in zip(res["per_seed_delta"], ref_per)) if len(ref_per) == len(res["per_seed_delta"]) else math.inf
    d_ci = max(abs(a - b) for a, b in zip(mine_ci, ref["ci"]))
    same_boot = n_boot == n_boot_ref
    ok_est = d_delta <= REPRO_TOL and d_seed <= REPRO_TOL and res["sign_agree"] == ref.get("sign_agree")
    ok_ci = d_ci <= REPRO_TOL if same_boot else None
    return {"status": "ok" if ok_est and (ok_ci or not same_boot) else "MISMATCH",
            "ref_delta": ref["delta"], "ref_ci": ref["ci"], "ref_sign_agree": ref.get("sign_agree"), "ref_claim": ref.get("claim"),
            "abs_diff_delta": d_delta, "abs_diff_per_seed": d_seed, "abs_diff_ci": d_ci,
            "ci_compared": same_boot, "ref_n_boot": n_boot_ref}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pp(v: Optional[float]) -> str:
    return "-" if v is None else f"{100 * v:+.2f}"


def ci_txt(ci: Sequence[Optional[float]]) -> str:
    return f"[{pp(ci[0])}, {pp(ci[1])}]"


def yn(b: Optional[bool]) -> str:
    return "-" if b is None else ("yes" if b else "no")


def markdown(doc: Dict[str, Any]) -> str:
    cfg = doc["config"]
    L = ["# Shared-instance CIs for the 3B certificate claims", "",
         f"Generated by `{Path(cfg['script']).name}` ({doc['version']}) from the eval roots listed in `shared_ci.json`; "
         f"{cfg['n_boot']} bootstrap replicates, seed {cfg['seed']}; condition `cert_v1`; decoding seeds averaged per row; "
         "cluster `scene_group` stratified by adapter. Differences X - Y in percentage points; estimate = mean over "
         "training seeds of the per-seed paired difference (identical for all three CIs).", "",
         "- **compare_cert CI**: `compare_cert.two_level_bootstrap` unchanged (seeds resampled, then clusters resampled "
         "independently for each drawn seed). Reproduces `cert_3b/cmp_main.json` / `extras_main.json` (column repro).",
         "- **shared CI**: one stratified cluster resample per replicate, used by every training seed; seeds resampled "
         "with replacement; mean of the drawn seeds' differences.",
         "- **instance-only CI**: the row's 3-seed-mean difference is the unit; cluster bootstrap only (no training-seed "
         "variance). A lower reference.",
         "- **sign**: training seeds agreeing in sign with the estimate (need 2 of 3). **claim** = CI excludes 0 and sign "
         "rule holds.", "",
         "| # | role | claim (split) | Δ pp | per-seed Δ | sign | compare_cert CI | excl 0 | shared CI | excl 0 | instance-only CI | excl 0 | width shared / cc | ρ seeds | repro |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(doc["claims"], 1):
        r = c["result"]
        ps = ", ".join(pp(d) for d in r["per_seed_delta"])
        w = r.get("width", {}).get("shared_over_compare_cert")
        rho = r.get("cluster_corr_across_seeds")
        L.append(f"| {i} | {c['role']} | {c['label']} | {pp(r['delta'])} | {ps} | {r['sign_agree']}/{r['K']} | "
                 f"{ci_txt(r['compare_cert']['ci'])} | {yn(r['compare_cert']['excludes0'])} | "
                 f"{ci_txt(r['shared']['ci'])} | {yn(r['shared']['excludes0'])} | "
                 f"{ci_txt(r['instance_only']['ci'])} | {yn(r['instance_only']['excludes0'])} | "
                 f"{'-' if w is None else f'{w:.2f}'} | {'-' if rho is None else f'{rho:.2f}'} | {c['reproduction']['status']} |")
    L += ["", "## Claim rule under each CI", "", "| # | claim | compare_cert claim | shared claim | changed |", "|---|---|---|---|---|"]
    for i, c in enumerate(doc["claims"], 1):
        r = c["result"]
        a, b = r["compare_cert"]["claim"], r["shared"]["claim"]
        L.append(f"| {i} | {c['label']} | {yn(a)} | {yn(b)} | {'**yes**' if a != b else 'no'} |")
    L += ["", "## Variance diagnostic (reading the widths; not a CI)", "",
          "Bootstrap SDs (pp) of the three schemes, the SDs predicted for each from a crossed random-effects decomposition "
          "d[row, seed] = mu + seed + row + interaction (adapter strata fixed; rows are the clusters here), and the moment "
          "SE of the estimate under that model, (MS_seed + MS_row - MS_resid)/(nK), with its split into seed / row "
          "(shared instance) / interaction parts. Under this model compare_cert counts the shared row variance once per "
          "seed (1/K of it), the shared scheme counts it fully but also over-counts the interaction term, because a seed "
          "drawn twice reuses one resample ((2K-1)/K^2 instead of 1/K^2 per seed), so it is conservative when the "
          "interaction dominates (low ρ). Both bootstraps scale the seed part by (K-1)/K.", "",
          "| # | claim | SD cc | SD shared | SD inst | pred cc | pred shared | pred inst | SE moment | share seed / row / interaction |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(doc["claims"], 1):
        r = c["result"]
        v, sd = r["variance_diagnostic"], r.get("boot_sd", {})
        f2 = lambda x: "-" if x is None else f"{100 * x:.2f}"  # noqa: E731
        if v.get("status") != "ok":
            L.append(f"| {i} | {c['label']} | {f2(sd.get('compare_cert'))} | {f2(sd.get('shared'))} | {f2(sd.get('instance_only'))} | {v.get('status')} | | | | |")
            continue
        sh = v["var_share"]
        share = " / ".join("-" if sh[k] is None else f"{sh[k]:.2f}" for k in ("seed", "instance", "interaction"))
        L.append(f"| {i} | {c['label']} | {f2(sd.get('compare_cert'))} | {f2(sd.get('shared'))} | {f2(sd.get('instance_only'))} | "
                 f"{f2(v['predicted_sd']['compare_cert'])} | {f2(v['predicted_sd']['shared'])} | {f2(v['predicted_sd']['instance_only'])} | "
                 f"{f2(v['se_moment'])} | {share} |")
    mcs = [(i, c) for i, c in enumerate(doc["claims"], 1) if c["result"]["shared"].get("mc_check")]
    if mcs:
        L += ["", "## Monte Carlo re-check of the shared CI", "",
              "Claims whose excludes-0 verdict changes (or whose shared CI ends near 0), recomputed with other RNG seeds.", "",
              "| # | claim | seed " + str(doc["config"]["seed"]) + " | " + " | ".join(f"seed {k}" for k in mcs[0][1]["result"]["shared"]["mc_check"]) + " |",
              "|---|---|---|" + "---|" * len(mcs[0][1]["result"]["shared"]["mc_check"])]
        for i, c in mcs:
            sh = c["result"]["shared"]
            L.append(f"| {i} | {c['label']} | {ci_txt(sh['ci'])} | " + " | ".join(ci_txt(v["ci"]) for v in sh["mc_check"].values()) + " |")
    s = doc["summary"]
    L += ["", "## Summary", "",
          f"- compare_cert reproduction: {s['reproduced']}/{s['n_claims']} claims match the published estimate, per-seed "
          f"differences, sign count and CI (tolerance {REPRO_TOL:g}).",
          f"- Shared CI wider than compare_cert's in {s['shared_wider']}/{s['n_claims']} claims; median width ratio "
          f"{s['median_width_ratio']:.2f} (range {s['min_width_ratio']:.2f}-{s['max_width_ratio']:.2f}).",
          f"- Excludes-0 verdict changes (compare_cert -> shared): {s['excl0_changes'] or 'none'}.",
          f"- Claim-rule verdict changes: {s['claim_changes'] or 'none'}.",
          "- Moment check of the claims whose verdict changes (Δ / SE moment, interaction share of the moment variance; "
          "a high interaction share means the shared scheme's extra width there is mostly its over-count, not shared "
          f"instance effects): {s['changed_moment'] or 'none'}.",
          "- ρ seeds = mean pairwise correlation across training seeds of the per-cluster mean differences (instance "
          "effects shared across seeds; ρ > 0 makes independent per-seed cluster resampling too narrow).", ""]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="NAME[@TRAINING_SEED]=PATH as compare_cert --arm")
    ap.add_argument("--splits", default="e1_test,e2_test")
    ap.add_argument("--cmp-main", required=True, help="cert_3b/cmp_main.json (compare_cert reference)")
    ap.add_argument("--extras-main", required=True, help="cert_3b/extras_main.json (error-class reference)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc-seeds", default="1,2,3,4", help="RNG seeds of the Monte Carlo re-check of the shared CI "
                    "(run for claims whose excludes-0 verdict changes or whose shared CI ends within 2.5%% of its width of 0)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    cc.METRICS.setdefault("ec_grounding", ("ec_grounding", True))  # runtime registration, as final_cert3b does
    splits = [s for s in args.splits.split(",") if s]
    args.mc_seeds = [int(x) for x in args.mc_seeds.split(",") if x]
    # final_cert3b's loader and cert_values (the error class needs no stage map; A/M/joint as compare_cert reads them)
    ds, _, info = fc.load_cert_datasets(args.model, METRIC_NAMES, {}, splits)
    print(f"loaded {sum(len(v) for v in ds.runs.values())} runs in {time.time() - t0:.0f}s", flush=True)
    ref = {"cmp": json.loads(Path(args.cmp_main).read_text()), "extras": json.loads(Path(args.extras_main).read_text())}
    st = cc.Settings(n_boot=args.n_boot, n_perm=0, seed=args.seed)
    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    need: Dict[Tuple[str, str, str], List[str]] = {}
    for _, x, y, split, m in CLAIMS:
        need.setdefault((x, y, split), [])
        if m not in need[(x, y, split)]:
            need[(x, y, split)].append(m)
    for key, ms in need.items():
        cache[key] = analyse_pair(ds, key[0], key[1], key[2], ms, st)
        print(f"{key[0]} - {key[1]} {key[2]} {ms}: {time.time() - t0:.0f}s", flush=True)
    claims = []
    for role, x, y, split, m in CLAIMS:
        pr = cache[(x, y, split)]
        res = pr["metrics"][m]
        refrow = reference_result(ref, x, y, split, m)
        ref_n = (ref["extras"] if m.startswith("ec_") else ref["cmp"])["config"]["n_boot"]
        mname = "grounding error-class rate" if m == "ec_grounding" else m
        claims.append({"role": role, "x": x, "y": y, "split": split, "metric": m,
                       "label": f"{x} - {y} {mname} ({split})", "pairing": pr["pairing"], "n_clusters": pr["n_clusters"],
                       "result": res, "reproduction": reproduction(res, refrow, ref_n, args.n_boot)})
    for c in claims:
        Sk, Nk, strat, K = c["result"].pop("_arrays")
        r = c["result"]
        if r["compare_cert"]["excludes0"] != r["shared"]["excludes0"] or (
                min(abs(v) for v in r["shared"]["ci"]) < 0.25 * (r["shared"]["ci"][1] - r["shared"]["ci"][0]) / 10):
            mc = {str(sd): shared_only(Sk, Nk, strat, K, args.n_boot, sd) for sd in args.mc_seeds}
            r["shared"]["mc_check"] = {sd: {"ci": ci, "excludes0": excludes0(ci)} for sd, ci in mc.items()}
    for pr in cache.values():  # the same metric dicts are referenced from claims; arrays already removed
        for res in pr["metrics"].values():
            res.pop("_arrays", None)
    ratios = [c["result"]["width"]["shared_over_compare_cert"] for c in claims]
    summary = {
        "n_claims": len(claims),
        "reproduced": sum(1 for c in claims if c["reproduction"]["status"] == "ok"),
        "shared_wider": sum(1 for r in ratios if r > 1),
        "median_width_ratio": float(np.median(ratios)), "min_width_ratio": float(min(ratios)), "max_width_ratio": float(max(ratios)),
        "excl0_changes": [f"{c['label']}: {yn(c['result']['compare_cert']['excludes0'])} -> {yn(c['result']['shared']['excludes0'])}"
                          for c in claims if c["result"]["compare_cert"]["excludes0"] != c["result"]["shared"]["excludes0"]],
        "claim_changes": [f"{c['label']}: {yn(c['result']['compare_cert']['claim'])} -> {yn(c['result']['shared']['claim'])}"
                          for c in claims if c["result"]["compare_cert"]["claim"] != c["result"]["shared"]["claim"]],
        "changed_moment": [
            f"{c['label']}: Δ/SE {c['result']['delta'] / c['result']['variance_diagnostic']['se_moment']:.2f}, interaction share "
            f"{c['result']['variance_diagnostic']['var_share']['interaction']:.2f}, row share {c['result']['variance_diagnostic']['var_share']['instance']:.2f}"
            for c in claims if c["result"]["compare_cert"]["excludes0"] != c["result"]["shared"]["excludes0"]
            and c["result"]["variance_diagnostic"].get("status") == "ok" and c["result"]["variance_diagnostic"]["se_moment"] > 0],
        "instance_only_selfcheck_ok": all(c["result"]["instance_only"]["matches_cc_cluster_bootstrap"] for c in claims),
    }
    doc = {
        "version": VERSION, "compare_cert_version": cc.VERSION, "final_cert3b_version": fc.VERSION,
        "config": {"script": str(Path(__file__).resolve()), "argv": list(sys.argv[1:]) if argv is None else list(argv),
                   "n_boot": args.n_boot, "seed": args.seed, "splits": splits, "condition": "cert_v1",
                   "cluster_key": "scene_group", "stratum_key": "adapter_id",
                   "rng": {"compare_cert": "default_rng([seed, 3]) inside compare_cert.two_level_bootstrap",
                           "shared_cluster_counts": "default_rng([seed, 1]) (also used by instance-only)",
                           "shared_seed_picks": "default_rng([seed, 2])"},
                   "code_sha256": {p.name: sha256(p) for p in (Path(__file__).resolve(), HERE / "compare_cert.py", HERE / "final_cert3b.py")},
                   "references": {"cmp_main": str(Path(args.cmp_main).resolve()), "extras_main": str(Path(args.extras_main).resolve())}},
        "models": info,
        "claims": claims,
        "summary": summary,
        "runtime_s": round(time.time() - t0, 1),
    }
    Path(args.out_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    md = markdown(doc)
    Path(args.out_md).write_text(md)
    print(md)
    bad = [c["label"] for c in claims if c["reproduction"]["status"] != "ok"]
    if bad:
        print(f"shared_ci: compare_cert reproduction failed for {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
