"""F2c analysis: which of TRACE's decoder block groups its measured gain needs (design/decision_F2.md).

Cells B, T (the F1 runs) and TR0/TR1/TR2 (full TRACE with decoder layers 0-8 / 9-18 / 19-27 restored to B), all under
P1. On the confirmation partition, seeds 42-44: ``R_k = S(T) - S(TR_k)`` (reversion effect of block group k) and
``S(TR_k) - S(B)``, official and format-normalized-v2 with the same paired cluster bootstrap draws as F1 (reused from
``analyze``). Holm over the three R_k is a descriptive safeguard, not a confirmatory family; the group effects are not
additive by construction, so no decomposition identity is claimed. The development screen (seed 42) is reported beside
it, with the design §4 technical-stop checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .analyze import (NORMALIZED, OFFICIAL, PRIMARY_INTERFACE, STATUS_FINAL, STATUS_PROVISIONAL, FollowupAnalysisError,
                      Log, _overall, _write_csv, an, draws_record, effects_table, format_rates, load_scored,
                      macro_benchmarks, make_context, performance_table, restrict_partition, run_table,
                      scale_bootstrap, st)
from .common import FollowupConfig

F2C_CELLS = ("TR0", "TR1", "TR2")
REFERENCE_CELLS = ("B", "T")
FAMILY = tuple(f"R_{k}" for k in F2C_CELLS)
BLOCKS = {"TR0": "decoder layers 0-8", "TR1": "decoder layers 9-18", "TR2": "decoder layers 19-27"}


def f2c_contrasts(interface: str = PRIMARY_INTERFACE) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    b, t = f"B@{interface}", f"T@{interface}"
    contrasts: dict[str, dict[str, float]] = {"full_T": {t: 1.0, b: -1.0}}
    roles = {"full_T": "reference_gain"}
    for k in F2C_CELLS:
        contrasts[f"R_{k}"] = {t: 1.0, f"{k}@{interface}": -1.0}
        roles[f"R_{k}"] = "reversion_effect_descriptive_holm"
        contrasts[f"{k}_minus_B"] = {f"{k}@{interface}": 1.0, b: -1.0}
        roles[f"{k}_minus_B"] = "retained_gain_exploratory"
    return contrasts, roles


def _matrix(ctx, runs: pd.DataFrame, tier: str, partition: str, seeds_cfg: list[int], *,
            cells: tuple[str, ...] = (*REFERENCE_CELLS, *F2C_CELLS), interface: str = PRIMARY_INTERFACE,
            contrasts_roles: tuple[dict[str, dict[str, float]], dict[str, str]] | None = None,
            family: tuple[str, ...] = FAMILY, name: str = "F2c") -> dict[str, Any]:
    part = runs[(runs["tier"] == tier) & runs["cell"].isin(cells) & (runs["interface"] == interface)]
    usable, missing = part[part["usable"]], part[~part["usable"]]
    seeds = [s for s in seeds_cfg if set(usable.loc[usable["seed"] == s, "cell"]) >= set(cells)]
    if len(missing) and not ctx.provisional:
        raise FollowupAnalysisError(f"{name} {tier} runs not usable: " + "; ".join(
            f"{r.name}: {r.problem}" for r in missing.itertuples()) + " (pass --provisional for a provisional run)")
    if not seeds:
        raise FollowupAnalysisError(f"{name} {tier}: no seed has all of {cells} usable")
    use = usable[usable["seed"].isin(seeds)]
    labels = [f"{c}@{interface}" for c in cells]
    benches, excluded = macro_benchmarks(ctx, use)
    rows = restrict_partition(load_scored(ctx, use), ctx.ga, partition)
    rows = rows[rows["benchmark_id"].isin(benches)].reset_index(drop=True)
    ctx.log(f"[{name}] {tier} {interface}: {len(rows)} rows, labels {labels}, seeds {seeds}, {len(benches)} benchmarks")
    results = scale_bootstrap(rows, ctx.metrics, ctx.ga, labels=labels, seeds=seeds, categories=ctx.categories,
                              n_replicates=ctx.n_replicates, seed=ctx.seed, confidence=ctx.confidence)
    contrasts, roles = contrasts_roles or f2c_contrasts(interface)
    return {"seeds": seeds, "labels": labels, "results_interface": interface, "benchmarks": benches, "excluded_benchmarks": excluded,
            "macro_name": st.macro_name(len(benches)), "missing_runs": missing["name"].tolist(), "results": results,
            "effects": effects_table(results, contrasts, roles, family), "performance": performance_table(results),
            "draws": draws_record(results)}


def technical_checks(ctx, runs: pd.DataFrame) -> pd.DataFrame:
    """Design §4 technical-stop rule for the F2c cells (smoke + screen, P1) and their build verification."""

    max_tokens = int(ctx.fcfg.base.raw["protocol"]["generation"]["max_tokens"])
    verify_path = ctx.fcfg.work("audit", "build_verification_F2c.json")
    verified = (pd.read_json(verify_path, typ="series").to_dict() if verify_path.is_file() else {})
    use = runs[runs["usable"] & runs["cell"].isin(F2C_CELLS) & runs["subset"].isin(["smoke", "development_screen"])]
    rates = (pd.concat([format_rates(load_scored(ctx, part), max_tokens).assign(subset=subset)
                        for subset, part in use.groupby("subset", sort=True)], ignore_index=True)
             if len(use) else pd.DataFrame())
    out = []
    for cell in F2C_CELLS:
        label = f"{cell}@{PRIMARY_INTERFACE}"
        rec: dict[str, Any] = {"cell": cell, "model_label": label, "restored_to_B": BLOCKS[cell]}
        reasons = []
        for subset, tag in (("smoke", "smoke"), ("development_screen", "screen")):
            r = _overall(rates, model_label=label, subset=subset) if len(rates) else pd.DataFrame()
            if r.empty:
                rec[f"{tag}_status"] = "not available"
                continue
            r = r.iloc[0]
            rec.update({f"{tag}_status": "available", f"{tag}_empty_pct": r["empty_pct"],
                        f"{tag}_truncated_pct": r["truncated_pct"]})
            if r["empty_pct"] > 50.0:
                reasons.append(f"{tag}: {r['empty_pct']:.1f}% empty")
            if r["truncated_pct"] > 80.0:
                reasons.append(f"{tag}: {r['truncated_pct']:.1f}% truncated")
        ok = bool((verified.get(cell) or {}).get("donor_equal"))
        rec["build_verified"] = ok
        if not ok:
            reasons.append("build verification missing or failed")
        rec["technical_stop"] = bool(reasons)
        rec["technical_stop_reasons"] = "; ".join(reasons)
        out.append(rec)
    return pd.DataFrame(out)


def analyze_f2c(fcfg: FollowupConfig, *, provisional: bool = False, reviewed_root: Path | str | None = None,
                n_replicates: int | None = None, out_dir: Path | None = None, log: Log = print) -> dict[str, Any]:
    ctx = make_context(fcfg, reviewed_root=reviewed_root, provisional=provisional, n_replicates=n_replicates,
                       behavior_replicates=0, log=log)
    status = STATUS_PROVISIONAL if provisional else STATUS_FINAL
    out = Path(out_dir) if out_dir else fcfg.work("tables", *(["provisional"] if provisional else []))
    runs = run_table(ctx, cells=[*REFERENCE_CELLS, *F2C_CELLS])
    seeds_cfg = [int(s) for s in fcfg.raw["evaluation"]["analysis"]["seeds"]]
    main = _matrix(ctx, runs, "analysis", "confirmation", seeds_cfg)
    screen = _matrix(ctx, runs, "screen", "development_screen", [42])
    checks = technical_checks(ctx, runs)
    if checks["technical_stop"].any() and not provisional:
        raise FollowupAnalysisError(f"F2c technical stops: {checks[checks['technical_stop']].to_dict('records')}")
    frames = []
    for name, m, part in (("main", main, "confirmation"), ("screen", screen, "development_screen")):
        for section, table in (("effects", m["effects"]), ("scores", m["performance"])):
            t = table.copy()
            t.insert(0, "section", section)
            t.insert(1, "matrix", name)
            t.insert(2, "partition", part)
            t.insert(3, "seeds", "+".join(map(str, m["seeds"])))
            frames.append(t)
    _write_csv(pd.concat(frames, ignore_index=True), out / "f2c_reversion.csv", status)
    _write_csv(checks, out / "f2c_technical.csv", status)

    def overall(m: Mapping[str, Any]) -> dict[str, Any]:
        eff = _overall(m["effects"], seed_scope="mean_over_seeds")
        perf = _overall(m["performance"], seed_scope="mean_over_seeds")
        return {"seeds": m["seeds"], "macro_name": m["macro_name"], "missing_runs": m["missing_runs"],
                "excluded_benchmarks": m["excluded_benchmarks"], "draws": m["draws"],
                "macro": {r.model_label: {"official": round(r.value_official, 4),
                                          "normalized": round(r.value_normalized, 4)} for r in perf.itertuples()},
                "effects": {f"{r.effect}/{r.scale}": [round(r.estimate, 4), round(r.ci_low, 4), round(r.ci_high, 4),
                                                      round(r.p_value, 5),
                                                      None if pd.isna(r.p_holm) else round(r.p_holm, 5)]
                            for r in eff[eff["status"] == "ok"].itertuples()}}

    summary = {"status": status, "design": "design/decision_F2.md", "family_descriptive_holm": list(FAMILY),
               "scales": {"primary": OFFICIAL, "sensitivity": NORMALIZED}, "blocks": BLOCKS,
               "main": overall(main), "screen": overall(screen),
               "technical_stop": checks[["cell", "technical_stop", "technical_stop_reasons"]].to_dict("records"),
               "runs_not_usable": {r.name: r.problem for r in runs[~runs["usable"]].itertuples()}}
    an._write_json(out / "f2c_summary.json", summary)
    log(f"[f2c] {status}: wrote f2c_reversion.csv, f2c_technical.csv, f2c_summary.json to {out}")
    return summary
