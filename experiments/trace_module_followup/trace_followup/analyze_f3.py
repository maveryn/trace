"""F3 analysis (design/decision_F3.md): does a norm-matched VERO keep VERO's measured result?

Cells B, T, V (the F1 runs) and VNM = (1 − λ) B + λ V with λ = ‖Δ_T‖/‖Δ_V‖. P1 on the confirmation partition (seeds
42–44) and the development screen; P2 on the development screen as a diagnostic. Linear contrasts use the F1 paired
cluster bootstrap and the same draws on both scales (``analyze_f2c._matrix``). The share of VERO's gain V-NM keeps,
``(S(VNM) − S(B)) / (S(V) − S(B))``, is a ratio: its interval is the percentile interval of the replicate ratio. Descriptive only;
there is no confirmatory family.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .analyze import (NORMALIZED, OFFICIAL, PRIMARY_INTERFACE, STATUS_FINAL, STATUS_PROVISIONAL, TEMPLATE_INTERFACE,
                      Log, _overall, _write_csv, an, contrast_arrays, make_context, run_table, scope_vectors, st)
from .analyze_f2c import _matrix
from .common import FollowupConfig

CELLS = ("B", "T", "V", "VNM")


def f3_contrasts(i: str) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    b, t, v, n = (f"{c}@{i}" for c in CELLS)
    contrasts = {"VNM_minus_V": {n: 1.0, v: -1.0}, "VNM_minus_B": {n: 1.0, b: -1.0}, "V_minus_B": {v: 1.0, b: -1.0},
                 "T_minus_VNM": {t: 1.0, n: -1.0}, "T_minus_B": {t: 1.0, b: -1.0}}
    return contrasts, {k: "f3_descriptive" for k in contrasts}


def retention(m: Mapping[str, Any], interface: str) -> dict[str, Any]:
    """Share of VERO's gain over B that V-NM keeps, per scale, on the overall macro (point and replicate interval)."""

    b, v, n = (f"{c}@{interface}" for c in ("B", "V", "VNM"))
    out = {}
    for scale, res in m["results"].items():
        _level, _scope, _count, vec = next(s for s in scope_vectors(res) if s[0] == "overall")
        num, den = contrast_arrays(res, {n: 1.0, b: -1.0}), contrast_arrays(res, {v: 1.0, b: -1.0})
        if num is None or den is None:
            out[scale] = {"status": "not computable"}
            continue
        point = float(num[0] @ vec) / float(den[0] @ vec)
        reps = (num[1] @ vec) / (den[1] @ vec)
        reps = reps[np.isfinite(reps)]
        lo, hi = st.percentile_interval(reps, res.confidence)
        out[scale] = {"estimate": point, "ci_low": lo, "ci_high": hi, "denominator_V_minus_B": float(den[0] @ vec)}
    return out


def analyze_f3(fcfg: FollowupConfig, *, provisional: bool = False, reviewed_root: Path | str | None = None,
               n_replicates: int | None = None, out_dir: Path | None = None, log: Log = print) -> dict[str, Any]:
    ctx = make_context(fcfg, reviewed_root=reviewed_root, provisional=provisional, n_replicates=n_replicates,
                       behavior_replicates=0, log=log)
    status = STATUS_PROVISIONAL if provisional else STATUS_FINAL
    out = Path(out_dir) if out_dir else fcfg.work("tables", *(["provisional"] if provisional else []))
    runs = run_table(ctx, cells=list(CELLS))
    seeds_cfg = [int(s) for s in fcfg.raw["evaluation"]["analysis"]["seeds"]]
    kw = {"cells": CELLS, "family": (), "name": "F3"}
    matrices = {
        "main_P1": _matrix(ctx, runs, "analysis", "confirmation", seeds_cfg, interface=PRIMARY_INTERFACE,
                           contrasts_roles=f3_contrasts(PRIMARY_INTERFACE), **kw),
        "screen_P1": _matrix(ctx, runs, "screen", "development_screen", [42], interface=PRIMARY_INTERFACE,
                             contrasts_roles=f3_contrasts(PRIMARY_INTERFACE), **kw),
        "screen_P2": _matrix(ctx, runs, "template", "development_screen", [42], interface=TEMPLATE_INTERFACE,
                             contrasts_roles=f3_contrasts(TEMPLATE_INTERFACE), **kw),
    }
    frames = []
    for name, m in matrices.items():
        for section, table in (("effects", m["effects"]), ("scores", m["performance"])):
            t = table.copy()
            t.insert(0, "section", section)
            t.insert(1, "matrix", name)
            t.insert(2, "seeds", "+".join(map(str, m["seeds"])))
            frames.append(t)
    _write_csv(pd.concat(frames, ignore_index=True), out / "f3_norm_matched.csv", status)
    norms = json.loads(fcfg.work("audit", "f3_update_norms.json").read_text(encoding="utf-8"))
    built = json.loads(fcfg.work("audit", "build_verification_F3.json").read_text(encoding="utf-8"))["VNM"]

    def overall(m: Mapping[str, Any], interface: str) -> dict[str, Any]:
        eff = _overall(m["effects"], seed_scope="mean_over_seeds")
        perf = _overall(m["performance"], seed_scope="mean_over_seeds")
        return {"seeds": m["seeds"], "macro_name": m["macro_name"], "missing_runs": m["missing_runs"],
                "macro": {r.model_label: {OFFICIAL: round(r.value_official, 4), NORMALIZED: round(r.value_normalized, 4)}
                          for r in perf.itertuples()},
                "effects": {f"{r.effect}/{r.scale}": [round(r.estimate, 4), round(r.ci_low, 4), round(r.ci_high, 4),
                                                      round(r.p_value, 5)]
                            for r in eff[eff["status"] == "ok"].itertuples()},
                "retention": retention(m, interface)}

    summary = {"status": status, "design": "design/decision_F3.md",
               "lambda": norms["lambda"], "norms": {"T": norms["global_delta_norm"]["T"],
                                                    "V": norms["global_delta_norm"]["V"],
                                                    "VNM_realized": built["realized_norm"],
                                                    "relative_residual": built["relative_residual"],
                                                    "fraction_values_equal_to_B": built["fraction_values_bitwise_equal_to_B"]},
               **{k: overall(m, PRIMARY_INTERFACE if k.endswith("P1") else TEMPLATE_INTERFACE)
                  for k, m in matrices.items()},
               "runs_not_usable": {r.name: r.problem for r in runs[~runs["usable"]].itertuples()
                                   if r.cell == "VNM" or r.tier in ("analysis", "screen", "template")}}
    an._write_json(out / "f3_summary.json", summary)
    log(f"[f3] {status}: wrote f3_norm_matched.csv and f3_summary.json to {out}")
    return summary
