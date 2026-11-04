#!/usr/bin/env python3
"""Write <work root>/reports/FINDINGS.md: the key findings and every analysis table, built from the work root's tables.

    python experiments/trace_merge/scripts/write_findings.py [--config <yaml>] [--work-root <dir>]

Run after `confirm`, `capture-reference`, `format-sensitivity`, `costs` and `diagnostics all`; `export` then copies the
file to results/reports/. Numbers are read from the tables (half-up rounding); the narrative wording is fixed here and
was written for the trace_vero_qwen25_v1 results, so rerunning on other results needs the wording reviewed.
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
import glob as _glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from trace_merge import reports as rp  # noqa: E402
from trace_merge.config import load_config  # noqa: E402

_args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
_args.add_argument("--config", type=Path, default=None)
_args.add_argument("--work-root", type=Path, default=None)
_a = _args.parse_args()
cfg = load_config(_a.config, work_root=_a.work_root)
W = cfg.work()
T = W / "tables"
MINUS = "−"


def rd(p):
    return pd.read_csv(p, comment="#", low_memory=False)


TIES = set()


def _half_up(x, d):
    q = Decimal(float(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)   # exact binary value, ties away from 0
    if f"{float(x):.{d}f}" != f"{q:.{d}f}":
        TIES.add((float(x), f"{float(x):.{d}f}", f"{q:.{d}f}"))
    return q


def num(x, d=2):
    return f"{_half_up(x, d):.{d}f}".replace("-", MINUS)


def sgn(x, d=2):
    return f"{_half_up(x, d):+.{d}f}".replace("-", MINUS)


def ci(lo, hi, d=2, signed=True):
    f = sgn if signed else num
    return f"[{f(lo, d)}, {f(hi, d)}]"


def est_ci(r, d=2, signed=True, e="estimate", lo="ci_low", hi="ci_high"):
    f = sgn if signed else num
    return f"{f(r[e], d)} {ci(r[lo], r[hi], d, signed)}"


def pfmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


ORDER = rp.suite_order(cfg)                      # [(category, benchmark), ...]
CATS = list(dict.fromkeys(c for c, _ in ORDER))
BNAME = rp.display_names(cfg)
SHORT = {"B": "B", "T": "T", "V@P0": "V@P0", "V@P1": "V@P1", "M050@P1": "M050", "CV050@P1": "CV050",
         "CT050@P1": "CT050", "B@P2": "B@P2", "T@P2": "T@P2", "M050@P2": "M050@P2", "CV050@P2": "CV050@P2",
         "CT050@P2": "CT050@P2", "B (reference)": "B (reference, post hoc)"}
P1 = ["B", "T", "V@P0", "V@P1", "M050@P1", "CV050@P1", "CT050@P1"]
P2 = ["B@P2", "T@P2", "V@P0", "M050@P2", "CV050@P2", "CT050@P2"]


def comp_name(c):
    a, b = c.split("-", 1)
    return f"{SHORT.get(a, a)} {MINUS} {SHORT.get(b, b)}"


def scores_table(s, labels):
    s = s.set_index(["model_label", "level", "scope"])
    overall = s.xs("overall", level=1).reset_index()["scope"].iloc[0]
    rows = []
    for lab in labels:
        r = s.loc[(lab, "overall", overall)]
        rows.append([SHORT[lab], est_ci(r, signed=False)] + [num(s.loc[(lab, "category", c)]["estimate"]) for c in CATS])
    return table(["model", f"{overall} [95% CI]", *CATS], rows)


def contrast_table(d, comps, holm=False):
    d = d.set_index(["comparison", "level", "scope"])
    rows = []
    for c in comps:
        o = d.xs((c, "overall"), level=(0, 1)).iloc[0]
        row = [comp_name(c), est_ci(o), pfmt(o["p_value"])]
        if holm:
            ph = o.get("p_holm")
            row.append(pfmt(ph) if ph == ph and ph is not None else "")
        row += [est_ci(d.loc[(c, "category", k)]) for k in CATS]
        rows.append(row)
    return table(["comparison", "macro [95% CI]", "p", *(["Holm p"] if holm else []), *CATS], rows)


def bench_scores(s, labels, extra=None):
    s = s[s["level"] == "benchmark"].set_index(["model_label", "scope"])["estimate"]
    rows = []
    for cat, b in ORDER:
        row = [cat, BNAME.get(b, b)] + [num(s.loc[(lab, b)]) for lab in labels]
        if extra is not None:
            row.append(est_ci(extra.loc[b]))
        rows.append(row)
    return rows


def bench_contrasts(d, comps):
    d = d[d["level"] == "benchmark"].set_index(["comparison", "scope"])
    return table(["category", "benchmark", *[comp_name(c) for c in comps]],
                 [[cat, BNAME.get(b, b)] + [est_ci(d.loc[(c, b)]) for c in comps] for cat, b in ORDER])


# ------------------------------------------------------------------------------------------------ load
sb_scores = rd(T / "stage_b/official_scores.all.csv")
sb_diff = rd(T / "stage_b/differences.all.csv")
sb_cr = rd(T / "stage_b/corrections_regressions.csv")
sb_joint = rd(T / "stage_b/joint_outcomes.csv")
sb_overlap = rd(T / "stage_b/parent_overlap.csv")
sb_pers = rd(T / "stage_b/persistence.csv")
sc1, sc2 = rd(T / "screen/scores.P1.csv"), rd(T / "screen/scores.P2.csv")
sd1, sd2 = rd(T / "screen/differences.P1.csv"), rd(T / "screen/differences.P2.csv")
s_cap = rd(T / "screen/merge_capture_damage.csv")
s_fmt = rd(T / "screen/answer_format_sensitivity.csv")
c_scores = rd(T / "confirmation/scores.csv")
c_diff = rd(T / "confirmation/differences.csv")
c_seed = rd(T / "confirmation/seed_scores.csv")
c_ref = rd(T / "confirmation/merge_capture_damage.base_reference.csv")
c_fmt = rd(T / "confirmation/answer_format_sensitivity.csv")
c_twin = rd(T / "confirmation/sensitivity_without_dev_twins.csv")
c_env = {k: rd(T / f"confirmation/sensitivity_seed42.{k}.csv") for k in ("archived_BT", "local_rerun_BT")}
audit = rd(T / "length_format_judge_audit.csv")
costs = rd(T / "evaluation_costs.csv")
totals = json.loads((T / "evaluation_costs.totals.json").read_text())
judge = rd(T / "judge_extraction_agreement.csv")
lock = json.loads((W / "reports/selection_lock.json").read_text())
n_recorded = sum(len(json.loads(Path(f).read_text())["attempts"]) for f in _glob.glob(str(W / "runs/*/*/timing.seed*.json")))

ov = c_scores[c_scores["level"] == "overall"].set_index("model_label")
cd = c_diff[c_diff["level"] == "overall"].set_index("comparison")
cdc = c_diff[c_diff["level"] == "category"].set_index(["comparison", "scope"])
cdb = c_diff[c_diff["level"] == "benchmark"].set_index(["comparison", "scope"])
sbd = sb_diff[sb_diff["level"] == "overall"].set_index("comparison")
sbdc = sb_diff[sb_diff["level"] == "category"].set_index(["comparison", "scope"])
FAMILY = ["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1"]
SECOND = ["T-B", "V@P1-B", "V@P1-V@P0", "M050@P1-B", "CV050@P1-B", "CT050@P1-B"]
pers = sb_pers[(sb_pers["level"] == "overall") & (sb_pers["seed_scope"] == "pooled_over_ordered_seed_pairs")].set_index("set_name")
olap = sb_overlap[(sb_overlap["level"] == "overall") & (sb_overlap["seed_scope"] == "mean_over_seeds")
                  & (sb_overlap["aggregation"] == "pooled_rows")].iloc[0]
cap = c_ref[(c_ref["level"] == "overall") & (c_ref["seed_scope"] != "single_condition")]
capi = cap[(cap["benchmark_set"] == "all_binary") & (cap["variant"] == "independent_repeat")].set_index(["merge_label", "group"])
fmt = c_fmt.groupby(["benchmark_id", "model_label"])[["official_accuracy_pct", "latex_letter_ending_pct",
                                                      "official_zero_but_final_letter_correct_pct"]].mean()
vdrop = cdb.loc["V@P1-V@P0"]["estimate"]
share = lambda bs: -sum(vdrop[b] for b in bs) / 24 / -cd.loc["V@P1-V@P0", "estimate"] * 100  # noqa: E731
ctT = (cd.loc["CT050@P1-B", "estimate"] / cd.loc["T-B", "estimate"]) * 100
cvV = cd.loc["CV050@P1-B", "estimate"] - cd.loc["V@P1-B", "estimate"]

# ------------------------------------------------------------------------------------------------ write
L = []
add = L.append
add("# TRACE–VERO merging study: findings and data")
add("")
add("Study `trace_vero_qwen25_v1`, complete as of 2026-11-03. It compares the released TRACE and VERO Qwen2.5-VL-7B "
    "checkpoints with their shared backbone, then tests three training-free 50:50 weight merges. Nothing was trained.")
add("")
add("This file collects the study's key findings and its data in one place.")
add("- **Interpretation, caveats and the decision.** These are in `analysis_report.md` and `decision_memo.md`, in this "
    "directory. The frozen design is `../../DESIGN.md`.")
add("- **Where the numbers come from.** Numbers are read programmatically from the aggregate tables in `../tables/`, "
    "and each section names its source. §6.2 also uses the runner summaries. Two values are quoted from the A8 record "
    "instead: the pre-repair M050 − T estimate and the untimed A40 attempt.")
add("- **Rounding.** Two decimals for percentage-point values and one for rates, rounded half-up.")
add("")
add("## How to read the numbers")
add("")
add("**Models.**")
add("- B: Qwen2.5-VL-7B-Instruct, the base.")
add("- T: the released TRACE model.")
add("- V: the released VERO model. V@P0 is VERO with its own think/answer chat template (the archived canonical "
    "runs); V@P1 is VERO with the backbone's template.")
add("- M050 = ½V + ½T, CV050 = ½V + ½B, CT050 = ½T + ½B. Unsuffixed merge labels are served with the backbone's "
    "interface (P1).")
add("- Labels ending @P2 (merges, B and T) are served with VERO's interface. That is a development-only diagnostic.")
add("")
add("**Metric.** The macro is the unweighted mean of the 24 official benchmark scores, averaged over seeds.")
add("")
add("**Intervals and tests.** Every interval is a 95% paired cluster bootstrap: 10,000 replicates over the bootstrap "
    "units (image/problem groups, with near-duplicate twins joined; DESIGN.md A7), with all models and seeds resampled "
    "together. p-values are two-sided bootstrap p-values; with 10,000 replicates the smallest attainable p is 0.0002, "
    "shown as < 0.001.")
add("- Seeds 42–44 are decoding seeds. The intervals cover item and decoding variability for these fixed checkpoints, "
    "not training-seed variability: each recipe is one released checkpoint.")
add("- Only the four comparisons of the prespecified primary family are confirmatory and Holm-adjusted.")
add("- Everything else is descriptive or exploratory. That includes categories, benchmarks, capture/damage, sensitivity, "
    "the development screen and P2.")
add("")
add("**Tiers.**")
add("- Stage B: the archived runs on the full split (32,805 rows, seeds 42–44).")
add("- Development screen: 2,910 rows, seed 42.")
add("- Confirmation: 26,228 rows, seeds 42–44.")
add("")

# ---------------------------------------------------------------- key findings
fam = {c: cd.loc[c] for c in FAMILY}
add("## 1. Key findings")
add("")
add(f"1. **TRACE and VERO gain similar amounts over the base** (Stage B, native interfaces).")
add(f"   - T − B = {est_ci(sbd.loc['T-B'])} and V@P0 − B = {est_ci(sbd.loc['V@P0-B'])}.")
add(f"   - Their difference, T − V@P0 = {est_ci(sbd.loc['T-V@P0'])}, is not resolved.")
add(f"   - Exploratory category contrasts put TRACE ahead in Charts & Tables "
    f"({est_ci(sbdc.loc[('T-V@P0', 'Charts & Tables')])}) and VERO ahead in Visual Math "
    f"({est_ci(sbdc.loc[('T-V@P0', 'Visual Math')])}).")
add(f"2. **The parents' unique successes are partly reproducible.**")
add(f"   - On a single seed they disagree on {num(olap['trace_only_pct'] + olap['vero_only_pct'], 1)}% of binary-scored "
    f"rows.")
add(f"   - A T-only success set recurs on another decoding seed {num(pers.loc['trace_only', 'persistence_pct'], 1)}% of the time "
    f"(label-swap null {num(pers.loc['trace_only', 'null_persistence_pct'], 1)}%). A V-only set recurs "
    f"{num(pers.loc['vero_only', 'persistence_pct'], 1)}% of the time (null "
    f"{num(pers.loc['vero_only', 'null_persistence_pct'], 1)}%).")
add(f"3. **VERO's gain depends on its interface.**")
add(f"   - With the backbone template, V@P1 − V@P0 = {est_ci(cd.loc['V@P1-V@P0'])} on confirmation.")
add(f"   - {num(share(['puzzlevqa', 'visualpuzzles']), 0)}% of that macro drop is in PuzzleVQA "
    f"({sgn(vdrop['puzzlevqa'])}) and VisualPuzzles ({sgn(vdrop['visualpuzzles'])}). There, many V@P1 answers end in "
    f"a LaTeX-wrapped option letter that the pinned scorers do not read: "
    f"{num(fmt.loc[('puzzlevqa', 'V@P1'), 'official_zero_but_final_letter_correct_pct'], 1)}% of its PuzzleVQA rows "
    f"are scored 0 with the correct final letter.")
add(f"   - TableVQABench ({sgn(vdrop['tablevqabench'])}) is another {num(share(['tablevqabench']), 0)}% and is not "
    f"diagnosed.")
add(f"4. **The merge beats both controls and VERO-on-the-backbone-template, but is not shown to beat TRACE**")
add(f"   (confirmation, Holm-adjusted):")
add(f"   - M050 − T = {est_ci(fam['M050@P1-T'])}, Holm p = {pfmt(fam['M050@P1-T']['p_holm'])}.")
add(f"   - M050 − V@P1 = {est_ci(fam['M050@P1-V@P1'])}.")
add(f"   - M050 − CV050 = {est_ci(fam['M050@P1-CV050@P1'])}.")
add(f"   - M050 − CT050 = {est_ci(fam['M050@P1-CT050@P1'])}.")
add(f"   - The last three have Holm p < 0.001. No noninferiority margin was set.")
add(f"5. **M050 scores {num(ov.loc['M050@P1', 'estimate'])}**, against T {num(ov.loc['T', 'estimate'])} and native VERO "
    f"{num(ov.loc['V@P0', 'estimate'])}. M050 − V@P0 is a difference of point estimates outside the primary family; "
    f"no paired interval was computed.")
add(f"6. **Halving each update costs differently.**")
add(f"   - CT050 keeps {num(ctT, 0)}% of TRACE's gain over the base.")
add(f"   - CV050 scores {num(cvV)} pp above VERO itself under P1, consistent with half of VERO's update softening its "
    f"answer-format shift.")
add(f"7. **Category pattern (descriptive).**")
add(f"   - M050 is below T only in Charts & Tables with an interval excluding 0 "
    f"({est_ci(cdc.loc[('M050@P1-T', 'Charts & Tables')])}).")
add(f"   - It is above T in Science & General ({est_ci(cdc.loc[('M050@P1-T', 'Science & General')])}) and Spatial "
    f"Reasoning ({est_ci(cdc.loc[('M050@P1-T', 'Spatial Reasoning')])}).")
add(f"   - VERO's half-update on top of the TRACE blend gains most in Visual Math "
    f"({est_ci(cdc.loc[('M050@P1-CT050@P1', 'Visual Math')])}).")
add(f"   - TRACE's half-update on top of the VERO blend gains most in Science & General "
    f"({est_ci(cdc.loc[('M050@P1-CV050@P1', 'Science & General')])}).")
mi, mv = capi.loc[("M050@P1", "trace_only")], capi.loc[("M050@P1", "vero_only")]
bi, bv = capi.loc[("B (reference)", "trace_only")], capi.loc[("B (reference)", "vero_only")]
add(f"8. **Row-level capture** (independent repeat; the base reference is post hoc, with no paired intervals):")
add(f"   - M050 keeps VERO-only successes at {num(mv['conditional_rate_pct'], 1)}%, against the base model's "
    f"{num(bv['conditional_rate_pct'], 1)}% on the same rows.")
add(f"   - It keeps TRACE-only successes at {num(mi['conditional_rate_pct'], 1)}%, close to the base model's "
    f"{num(bi['conditional_rate_pct'], 1)}%.")
add("9. **Frozen hypotheses.**")
add("   - H1 (integration beyond shrinkage) holds in direction.")
add("   - H2 (the category trade-off transfers) is half supported.")
add(f"   - H3 (interface dependence) is supported for the macro and for Puzzles & Logic. It is not supported for "
    f"Visual Math being one of the two largest drops: Charts & Tables drops more "
    f"({est_ci(cdc.loc[('V@P1-V@P0', 'Charts & Tables')])}, through TableVQABench), although Visual Math also drops "
    f"({est_ci(cdc.loc[('V@P1-V@P0', 'Visual Math')])}).")
add("10. **One technical repair** (DESIGN amendments A8, A9). One confirmation run's judge (Qwen3-32B at "
    "tensor-parallel 4 on A800 cards) produced degraded extractions. The run was rescored at tensor-parallel 1, and every "
    "table here uses the repaired scores (§6).")
add("11. **Decision** (`decision_memo.md`).")
add("    - Do not adopt a merge as an improvement.")
add("    - Stop the merge search.")
add("    - Do not train.")
add("    - One exploratory, evaluation-only follow-up is proposed. It needs a DESIGN amendment, researcher approval and "
    "GPU authorization.")
add("")

# ---------------------------------------------------------------- runs
add("## 2. Runs")
add("")
add("Every new generation run, with its final attempt's hardware. Source: `../tables/evaluation_costs.csv` and each "
    "run's `timing.seed<S>.json`.")
add("- B, T and V@P0 are not re-run; they come from the archive.")
add(f"- GPU-hours and the attempts column count the {totals['attempts']} attempts with a timed phase. The timing "
    f"files record {n_recorded} attempts; {n_recorded - totals['attempts']} stopped right after starting, with no "
    f"timed phase (smoke M050, smoke V@P1, V@P1 confirmation seed 44).")
add("- Generated tokens are counted from the ingested responses.")
add("")
rows_per = {"smoke": 135, "development_screen": 2910, "confirmation": 26228}
label_of = {"vero-p1": "V@P1", "m050-p1": "M050", "cv050-p1": "CV050", "ct050-p1": "CT050", "qwen25vl7b-p2": "B@P2",
            "trace7b-p2": "T@P2", "m050-p2": "M050@P2", "cv050-p2": "CV050@P2", "ct050-p2": "CT050@P2"}
tier_order = {"smoke": 0, "development_screen": 1, "confirmation": 2}
run_rows = []
for r in costs.sort_values(["tier", "view_slug", "seed"], key=lambda s: s.map(tier_order) if s.name == "tier" else s).itertuples():
    tf = W / "runs" / r.view_slug / r.tier / f"timing.seed{r.seed}.json"
    hw = ""
    if tf.is_file():
        a = json.loads(tf.read_text())["attempts"][-1]
        hw = f"{a.get('host', '')}, {a.get('gpus', '')}".replace(" Max-Q Workstation Edition", "")
    note = " (scoring redone, A8)" if (r.view_slug, r.tier, r.seed) == ("m050-p1", "confirmation", 44) else ""
    run_rows.append([r.tier.replace("_", " "), label_of[r.view_slug], r.seed, f"{rows_per[r.tier]:,}", r.attempts,
                     hw + note, num(r.gpu_hours), f"{int(r.completion_tokens):,}"])
add(table(["tier", "model", "seed", "rows", "attempts", "final attempt: host, GPUs", "GPU-hours", "generated tokens"],
          run_rows))
add("")
add(f"Totals: {totals['gpu_hours']} recorded GPU-hours over {totals['attempts']} attempts "
    f"({totals['failed_or_incomplete_attempts']} failed or incomplete, all resumed) and "
    f"{totals['completion_tokens']:,} generated tokens. Merged weights take {totals['merged_weights_gib']} GiB; run "
    f"directories take {totals['run_dirs_gib']} GiB.")
add("")
add("One attempt is not in the timing files: M050 confirmation seed 44 first generated all its rows on 2× A40 for about "
    "2.3 hours (about 4.6 GPU-hours) before its allocation expired. Its rows were kept.")
add("")

# ---------------------------------------------------------------- stage B
add("## 3. Stage B: TRACE and VERO against the base (archived runs, full split, seeds 42–44)")
add("")
add("### 3.1 Official scores (`stage_b/official_scores.all.csv`)")
add("")
add(scores_table(sb_scores, ["B", "T", "V@P0"]))
add("")
add("### 3.2 Contrasts (`stage_b/differences.all.csv`; categories exploratory, unadjusted)")
add("")
add(contrast_table(sb_diff, ["T-B", "V@P0-B", "T-V@P0"]))
add("")
add("### 3.3 Per-benchmark scores and T − V@P0 (`stage_b/official_scores.all.csv`, `stage_b/differences.all.csv`; "
    "exploratory, unadjusted)")
add("")
sbb = sb_diff[(sb_diff["level"] == "benchmark") & (sb_diff["comparison"] == "T-V@P0")].set_index("scope")
add(table(["category", "benchmark", "B", "T", "V@P0", "T − V@P0 [95% CI]"], bench_scores(sb_scores, ["B", "T", "V@P0"], sbb)))
add("")
add("### 3.4 Row-level behaviour (21 binary-eligible benchmarks, 27,617 rows per seed; mean over seeds, rows pooled)")
add("")
cr = sb_cr[(sb_cr["level"] == "overall") & (sb_cr["seed_scope"] == "mean_over_seeds") & (sb_cr["aggregation"] == "pooled_rows")]
add("Corrections and regressions against B (`stage_b/corrections_regressions.csv`):")
add("")
add(table(["model", "correction rate (% of rows)", "regression rate", "net [95% CI]", "corrections / B errors",
           "regressions / B successes"],
          [[r.model_label, num(r.correction_rate_pct), num(r.regression_rate_pct),
            f"{sgn(r.net_delta_pp)} {ci(r.net_delta_pp_ci_low, r.net_delta_pp_ci_high)}",
            num(r.correction_given_base_error_pct, 1) + "%", num(r.regression_given_base_success_pct, 1) + "%"]
           for r in cr.itertuples()]))
add("")
jo = sb_joint[(sb_joint["level"] == "overall") & (sb_joint["seed_scope"] == "mean_over_seeds")
              & (sb_joint["aggregation"] == "pooled_rows")].set_index("cell")
add("Eight outcome cells, % of rows (`stage_b/joint_outcomes.csv`):")
add("")
add(table(["", "B = 0", "B = 1"],
          [[f"T = {t}, V@P0 = {v}"] + [num(jo.loc[f"B={b}|T={t}|V@P0={v}", "cell_rate_pct"]) for b in (0, 1)]
           for t, v in ((0, 0), (0, 1), (1, 0), (1, 1))]))
add("")
add("Parent overlap (`stage_b/parent_overlap.csv`):")
add(f"- T-only {num(olap['trace_only_pct'], 1)}%, V-only {num(olap['vero_only_pct'], 1)}%, both correct "
    f"{num(olap['both_correct_pct'], 1)}%, both wrong {num(olap['both_wrong_pct'], 1)}%; agreement "
    f"{num(olap['agreement_pct'], 1)}%.")
add(f"- The hindsight \"either parent correct\" rate is {num(olap['hindsight_either_parent_correct_pct'], 1)}%. It uses "
    f"two responses and correctness labels, so it is neither a router nor a bound on a merge.")
add("")
add("Persistence of outcome sets across seeds (`stage_b/persistence.csv`; defined on one seed, checked on another, all "
    "ordered seed pairs):")
add("")
names = {"trace_only": "T-only (T=1, V@P0=0)", "vero_only": "V-only (T=0, V@P0=1)", "trace_correction": "T corrections (B=0, T=1)",
         "vero_correction": "V corrections (B=0, V@P0=1)", "trace_regression": "T regressions (B=1, T=0)",
         "vero_regression": "V regressions (B=1, V@P0=0)"}
add(table(["set", "persists (%)", "label-swap null (%)", "excess (pp) [95% CI]"],
          [[names[k], num(r["persistence_pct"], 1), num(r["null_persistence_pct"], 1),
            f"{sgn(r['persistence_excess_pp'], 1)} {ci(r['persistence_excess_pp_ci_low'], r['persistence_excess_pp_ci_high'], 1)}"]
           for k, r in pers.iterrows() if k in names]))
add("")

# ---------------------------------------------------------------- screen
add("## 4. Development screen (seed 42, 2,910 rows; descriptive)")
add("")
add("### 4.1 Scores under P1 (`screen/scores.P1.csv`)")
add("")
add(scores_table(sc1, P1))
add("")
add("### 4.2 Contrasts under P1 (`screen/differences.P1.csv`)")
add("")
add(contrast_table(sd1, ["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1", "CV050@P1-B", "CT050@P1-B",
                         "T-B", "V@P1-B", "V@P1-V@P0"]))
add("")
add("### 4.3 Scores under P2, VERO's interface (`screen/scores.P2.csv`)")
add("")
add(scores_table(sc2, P2))
add("")
add("### 4.4 Contrasts under P2 (`screen/differences.P2.csv`)")
add("")
add(contrast_table(sd2, ["M050@P2-V@P0", "M050@P2-CV050@P2", "M050@P2-CT050@P2", "M050@P2-T@P2", "T@P2-B@P2", "V@P0-B@P2"]))
add("")
add("### 4.5 Capture and damage, same-seed parents (`screen/merge_capture_damage.csv`)")
add("")
G = [("trace_only", "TRACE-only capture"), ("vero_only", "VERO-only capture"), ("shared_success", "shared-success damage"),
     ("joint_failure", "joint-failure rescue")]
sc_ = s_cap[(s_cap["level"] == "overall")].set_index(["merge_label", "group"])
add(table(["merge", *[f"{n} (n = {int(round(sc_.loc[('M050@P1', g), 'n_group'])):,})" for g, n in G]],
          [[SHORT[m]] + [f"{num(sc_.loc[(m, g), 'conditional_rate_pct'], 1)} "
                         f"{ci(sc_.loc[(m, g), 'conditional_rate_pct_ci_low'], sc_.loc[(m, g), 'conditional_rate_pct_ci_high'], 1, False)}"
                         for g, _ in G] for m in ("M050@P1", "CV050@P1", "CT050@P1")]))
add("")
add("### 4.6 Answer format (`screen/answer_format_sensitivity.csv`; % of rows)")
add("")
sf = s_fmt.set_index(["benchmark_id", "model_label"])
add(table(["model", "PuzzleVQA accuracy", "PuzzleVQA: ends in \\( X \\)", "PuzzleVQA: scored 0, final letter correct",
           "VisualPuzzles accuracy", "VisualPuzzles: ends in \\( X \\)", "VisualPuzzles: scored 0, final letter correct"],
          [[SHORT[m]] + [num(sf.loc[(b, m), c], 1) for b in ("puzzlevqa", "visualpuzzles")
                         for c in ("official_accuracy_pct", "latex_letter_ending_pct", "official_zero_but_final_letter_correct_pct")]
           for m in [*P1, *[x for x in P2 if x != "V@P0"]] if (("puzzlevqa", m) in sf.index)]))
add("")

# ---------------------------------------------------------------- confirmation
add("## 5. Confirmation (P1, 26,228 rows, seeds 42–44)")
add("")
add(f"The confirmed set, the primary family and every input hash were locked at {lock['locked_at']} "
    f"(`selection_lock.json`) before any confirmation run had scores.")
add("")
add("### 5.1 Official scores (`confirmation/scores.csv`)")
add("")
add(scores_table(c_scores, P1))
add("")
add("### 5.2 Macro per seed (`confirmation/seed_scores.csv`; unweighted mean of the 24 benchmarks)")
add("")
ps = c_seed.groupby(["model_label", "seed"])["official_score"].mean().unstack()
add(table(["model", "seed 42", "seed 43", "seed 44", "mean"],
          [[SHORT[m]] + [num(ps.loc[m, s]) for s in (42, 43, 44)] + [num(ps.loc[m].mean())] for m in P1]))
add("")
add("### 5.3 Per-benchmark scores, mean over seeds (`confirmation/scores.csv`)")
add("")
add(table(["category", "benchmark", *[SHORT[m] for m in P1]], bench_scores(c_scores, P1)))
add("")
add("### 5.4 Primary family (Holm-adjusted) and secondary contrasts (`confirmation/differences.csv`)")
add("")
add(contrast_table(c_diff, FAMILY, holm=True))
add("")
add(contrast_table(c_diff, SECOND))
add("")
add("### 5.5 Per-benchmark contrasts (`confirmation/differences.csv`; exploratory, unadjusted)")
add("")
add(bench_contrasts(c_diff, ["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1", "V@P1-V@P0"]))
add("")
add(bench_contrasts(c_diff, ["T-B", "V@P1-B", "M050@P1-B", "CV050@P1-B", "CT050@P1-B"]))
add("")
add("### 5.6 Capture, retention and damage (`confirmation/merge_capture_damage.base_reference.csv`)")
add("")
add("Setup:")
add("- 21 binary-eligible benchmarks, 22,078 rows per seed.")
add("- Groups are defined by T and V@P1 on the confirmation partition.")
add("- Rates are pooled over conditions, and group sizes are mean rows per condition.")
add("- Intervals are for each rate on its own, not for differences.")
add("- The B (reference) rows score the base model as if it were a merge. They were added post hoc.")
add("")
for bset, title in (("all_binary", "all 21 benchmarks"), ("without_puzzlevqa_visualpuzzles", "without PuzzleVQA and VisualPuzzles")):
    add(f"**{title}**")
    add("")
    rows = []
    for v in ("independent_repeat", "contemporaneous", "stable_other_seeds"):
        x = cap[(cap["benchmark_set"] == bset) & (cap["variant"] == v)].set_index(["merge_label", "group"])
        sizes = " / ".join(f"{int(round(x.loc[('M050@P1', g), 'n_group'])):,}" for g, _ in G)
        rows.append([f"**{v.replace('_', ' ')}** ({sizes})", "", "", "", ""])
        for m in ("M050@P1", "CV050@P1", "CT050@P1", "B (reference)"):
            rows.append([SHORT[m]] + [f"{num(x.loc[(m, g), 'conditional_rate_pct'], 1)} "
                                      f"{ci(x.loc[(m, g), 'conditional_rate_pct_ci_low'], x.loc[(m, g), 'conditional_rate_pct_ci_high'], 1, False)}"
                                      for g, _ in G])
    add(table(["model (variant, group sizes)", *[n for _, n in G]], rows))
    add("")
add("### 5.7 Answer format (`confirmation/answer_format_sensitivity.csv`; mean over seeds, % of rows)")
add("")
add(table(["model", "PuzzleVQA accuracy", "PuzzleVQA: ends in \\( X \\)", "PuzzleVQA: scored 0, final letter correct",
           "VisualPuzzles accuracy", "VisualPuzzles: ends in \\( X \\)", "VisualPuzzles: scored 0, final letter correct"],
          [[SHORT[m]] + [num(fmt.loc[(b, m), c], 1) for b in ("puzzlevqa", "visualpuzzles")
                         for c in ("official_accuracy_pct", "latex_letter_ending_pct", "official_zero_but_final_letter_correct_pct")]
           for m in P1]))
add("")
add("### 5.8 Length, finish reasons and extraction (`length_format_judge_audit.csv`)")
add("")
add("B, T and V@P0 come from the archive (full split, 98,415 responses each). The new models come from the "
    "confirmation runs (78,684 responses each). Rates are row-weighted.")
add("")
au = audit[(audit["source"] == "archive") | audit["run_id"].astype(str).str.contains("confirmation")]
au = au[au["model_label"].isin(P1)]
lrows = []
for m in P1:
    g = au[au["model_label"] == m]
    n = g["n_rows"].sum()
    w = lambda c: (g[c] * g["n_rows"]).sum() / n * 100  # noqa: E731
    med = Decimal(str(g["completion_tokens_median"].median())).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    lrows.append([SHORT[m], f"{int(n):,}", int(g["finish_reason_length"].sum()), str(med),
                  num(w("marker_think_close_rate"), 1) + "%", num(w("marker_boxed_rate"), 1) + "%",
                  num(w("extraction_status_not_resolved_rate"), 3) + "%"])
add(table(["model", "responses", "length stops", "median of per-benchmark median completion tokens", "with </think>",
           "with \\boxed{}", "extraction not resolved"], lrows))
add("")
add("### 5.9 Sensitivity analyses")
add("")
add("Without the 130 confirmation rows that have a near-duplicate development twin (`confirmation/sensitivity_without_dev_twins.csv`):")
add("")
tw = c_twin[c_twin["level"] == "overall"].set_index("comparison")
add(table(["comparison", "all rows", "without twins"],
          [[comp_name(c), est_ci(cd.loc[c]), est_ci(tw.loc[c])] for c in FAMILY + SECOND]))
add("")
add("Seed 42 only, with the archived B/T or with local reruns of B/T made in the new environment "
    "(`confirmation/sensitivity_seed42.*.csv`):")
add("")
e1, e2 = (c_env[k][c_env[k]["level"] == "overall"].set_index("comparison") for k in ("archived_BT", "local_rerun_BT"))
add(table(["comparison", "archived B/T", "local-rerun B/T"],
          [[comp_name(c), est_ci(e1.loc[c]), est_ci(e2.loc[c])] for c in e1.index]))
add("")

# ---------------------------------------------------------------- technical
add("## 6. Technical record")
add("")
add("### 6.1 Judge health (`judge_extraction_agreement.csv`)")
add("")
add("For each run's MathVerse slice: the share of rows whose judge-extracted answer equals the response's own final "
    "`\\boxed{}` answer. The floor is 30% (DESIGN.md A8).")
add("")
add(table(["run", "tier", "seed", "quarantined", "rows with a boxed answer", "extracted = boxed (%)", "below floor"],
          [[label_of[r.view_slug], r.subset.replace("_", " "), r.seed, "yes" if r.quarantined else "", r.rows_with_boxed,
            num(r.boxed_equals_extracted_pct, 1), "yes" if r.below_floor else ""]
           for r in judge.sort_values(["subset", "view_slug", "seed", "quarantined"]).itertuples()]))
add("")
add("### 6.2 The A8 repair: M050 confirmation seed 44 before and after rescoring")
add("")
add("The quarantined first scoring was done with the judge at tensor-parallel 4 on A800 cards; the rescore used "
    "tensor-parallel 1 on RTX PRO 6000 cards. The values are the score campaign's own per-benchmark aggregates from the "
    "runner summaries. WeMath is the campaign's fixed-525 value; the study's WeMath metric differs (DESIGN.md A2).")
add("")
run = W / "runs/m050-p1/confirmation"
before = {b["key"]: b["score"] for b in json.loads((run / "quarantine_seed44_tp4_judge/summary.seed44.json").read_text())["benchmarks"]}
after_s = json.loads((run / "summary.seed44.json").read_text())
after = {b["key"]: b["score"] for b in after_s["benchmarks"]}
s42 = c_seed[(c_seed["model_label"] == "M050@P1")].pivot(index="benchmark_id", columns="seed", values="official_score")
brows = []
for cat, b in ORDER:
    if abs(before[b] - after[b]) < 0.005:
        continue
    brows.append([BNAME.get(b, b), num(before[b]), num(after[b]), sgn(after[b] - before[b]),
                  num(s42.loc[b, 42]), num(s42.loc[b, 43])])
add(table(["benchmark", "seed 44 before", "seed 44 after", "change", "seed 42 (study metric)", "seed 43 (study metric)"], brows))
add("")
add("Only benchmarks whose campaign value changed are listed. Every other benchmark is identical before and after.")
add("")
add("Effect on the primary contrast:")
add("- Before the repair, M050 − T was −0.29: the point estimate from the superseded tables, which the second "
    "verification pass recomputed from the quarantined scores.")
add(f"- After it, M050 − T is {est_ci(cd.loc['M050@P1-T'])}.")
add("- The old confirmation tables are superseded. `analysis_report.md` §4 and `decision_memo.md` quote only this "
    "pre-repair estimate and the Visual Math shortfall it showed, to describe the fault.")
add("")

# ---------------------------------------------------------------- files
add("## 7. Where the data lives (`experiments/trace_merge/results/`)")
add("")
add(table(["path", "contents"], [
    ["`reports/analysis_report.md`, `reports/decision_memo.md`", "interpretation, caveats, decision"],
    ["`reports/behavior_report.md`, `reports/merge_screen.md`", "frozen Stage B report; development-screen report"],
    ["`reports/selection_lock.json`", "hashes frozen before confirmation"],
    ["`tables/stage_b/`", "Stage B scores, contrasts, corrections/regressions, joint cells, overlap, persistence"],
    ["`tables/screen/`", "development screen: P1/P2 scores, contrasts, seed scores, capture/damage, answer format"],
    ["`tables/confirmation/`", "confirmation: scores, seed scores, contrasts, capture/damage (+ base reference), answer format, sensitivities"],
    ["`tables/length_format_judge_audit.csv`", "lengths, finish reasons, extraction and judge use per model, seed and benchmark"],
    ["`tables/judge_extraction_agreement.csv`", "judge-health check per run"],
    ["`tables/evaluation_costs.csv`", "GPU-hours, attempts, tokens per run"],
    ["`audit/`", "asset, archive, metric-reproduction, protocol and environment audits"],
    ["`figures/`", "Stage B, screen and confirmation figures"],
    ["`merges/*/merge_manifest.json`", "merge recipes and hashes"],
    ["`tests/test_report.txt`", "test-suite run"],
]))
add("")

out = W / "reports" / "FINDINGS.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out.name} ({len(L)} lines)")
for t in sorted(TIES):
    print("half-up differs from half-even:", t)
