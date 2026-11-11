"""Three-seed interface table (design amendment A5): the seven cells under P1 and P2 on the development screen, seeds
42-44 (the seed-42 screen and template runs plus the replication tier), one paired bootstrap for all 14 labels and both
scales. Reports per-cell P2 − P1 and the component decomposition under each interface. Exploratory: nothing here is
confirmatory or selects anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .analyze import (F1_CELLS, NORMALIZED, OFFICIAL, PRIMARY_INTERFACE, STATUS_FINAL, STATUS_PROVISIONAL,
                      TEMPLATE_INTERFACE, FollowupAnalysisError, Log, _overall, _write_csv, an, combine,
                      decomposition_contrasts, draws_record, effects_table, load_scored, macro_benchmarks, make_context,
                      performance_table, restrict_partition, run_table, scale_bootstrap, st)
from .common import FollowupConfig

TIERS = ("screen", "template", "replication")


def analyze_interface(fcfg: FollowupConfig, *, provisional: bool = False, reviewed_root: Path | str | None = None,
                      n_replicates: int | None = None, out_dir: Path | None = None, log: Log = print) -> dict[str, Any]:
    ctx = make_context(fcfg, reviewed_root=reviewed_root, provisional=provisional, n_replicates=n_replicates,
                       behavior_replicates=0, log=log)
    status = STATUS_PROVISIONAL if provisional else STATUS_FINAL
    out = Path(out_dir) if out_dir else fcfg.work("tables", *(["provisional"] if provisional else []))
    runs = run_table(ctx, tiers=list(TIERS))
    sr = runs[runs["cell"].isin(F1_CELLS) & (runs["subset"] == "development_screen")]
    usable, missing = sr[sr["usable"]], sr[~sr["usable"]]
    labels = [f"{c}@{i}" for i in (PRIMARY_INTERFACE, TEMPLATE_INTERFACE) for c in F1_CELLS]
    seeds = [s for s in (42, 43, 44) if set(usable.loc[usable["seed"] == s, "label"]) >= set(labels)]
    if len(missing) and not provisional:
        raise FollowupAnalysisError("interface runs not usable: " + "; ".join(
            f"{r.name}: {r.problem}" for r in missing.itertuples()))
    if not seeds:
        raise FollowupAnalysisError("no seed has all 14 labels usable")
    use = usable[usable["seed"].isin(seeds)]
    benches, excluded = macro_benchmarks(ctx, use)
    rows = restrict_partition(load_scored(ctx, use), ctx.ga, "development_screen")
    rows = rows[rows["benchmark_id"].isin(benches)].reset_index(drop=True)
    log(f"[interface] {len(rows)} rows, {len(labels)} labels, seeds {seeds}")
    results = scale_bootstrap(rows, ctx.metrics, ctx.ga, labels=labels, seeds=seeds, categories=ctx.categories,
                              n_replicates=ctx.n_replicates, seed=ctx.seed, confidence=ctx.confidence)
    contrasts: dict[str, dict[str, float]] = {}
    roles: dict[str, str] = {}
    for c in F1_CELLS:
        contrasts[f"P2_minus_P1:{c}"] = {f"{c}@{TEMPLATE_INTERFACE}": 1.0, f"{c}@{PRIMARY_INTERFACE}": -1.0}
        roles[f"P2_minus_P1:{c}"] = "interface_dependence_per_cell"
    d1, d2 = decomposition_contrasts(PRIMARY_INTERFACE), decomposition_contrasts(TEMPLATE_INTERFACE)
    for name in d1:
        contrasts[f"{name}@P1"], roles[f"{name}@P1"] = d1[name], "decomposition_P1_screen"
        contrasts[f"{name}@P2"], roles[f"{name}@P2"] = d2[name], "decomposition_P2"
        contrasts[f"{name}:P2_minus_P1"] = combine((1.0, d2[name]), (-1.0, d1[name]))
        roles[f"{name}:P2_minus_P1"] = "decomposition_interface_dependence"
    effects = effects_table(results, contrasts, roles)
    perf = performance_table(results)
    table = pd.concat([perf.assign(section="scores"), effects.assign(section="contrasts")], ignore_index=True)
    table.insert(0, "section", table.pop("section"))
    table.insert(1, "partition", "development_screen")
    table.insert(2, "seeds", "+".join(map(str, seeds)))
    _write_csv(table, out / "interface_3seed.csv", status)
    eff = _overall(effects, seed_scope="mean_over_seeds")
    summary = {"status": status, "design": "design/followup_v1.md amendment A5", "seeds": seeds,
               "macro_name": st.macro_name(len(benches)), "excluded_benchmarks": excluded,
               "missing_runs": missing["name"].tolist(), "draws": draws_record(results),
               "macro": {r.model_label: {OFFICIAL: round(r.value_official, 4), NORMALIZED: round(r.value_normalized, 4)}
                         for r in _overall(perf, seed_scope="mean_over_seeds").itertuples()},
               "effects": {f"{r.effect}/{r.scale}": [round(r.estimate, 4), round(r.ci_low, 4), round(r.ci_high, 4),
                                                     round(r.p_value, 5)]
                           for r in eff[eff["status"] == "ok"].itertuples()}}
    an._write_json(out / "interface_3seed_summary.json", summary)
    log(f"[interface] {status}: wrote interface_3seed.csv and interface_3seed_summary.json to {out}")
    return summary
