#!/usr/bin/env python3
"""Write <work root>/reports/FINDINGS.md: key findings, experimental settings and every analysis table, from the work root.

    python experiments/trace_merge/scripts/write_findings.py [--config <yaml>] [--work-root <dir>]

Run after `confirm`, `capture-reference`, `format-sensitivity`, `costs` and `diagnostics all`; `export` then copies the
file to results/reports/. Numbers are read from the tables (half-up rounding); the narrative wording is fixed here and
was written for the trace_vero_qwen25_v1 results, so rerunning on other results needs the wording reviewed.
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime
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
    x = 0.0 if abs(float(x)) < 1e-9 else x                                          # float noise at an exact 0
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
c_cap = rd(T / "confirmation/merge_capture_damage.csv")
c_fmt = rd(T / "confirmation/answer_format_sensitivity.csv")
c_twin = rd(T / "confirmation/sensitivity_without_dev_twins.csv")
c_env = {k: rd(T / f"confirmation/sensitivity_seed42.{k}.csv") for k in ("archived_BT", "local_rerun_BT")}
audit = rd(T / "length_format_judge_audit.csv")
costs = rd(T / "evaluation_costs.csv")
totals = json.loads((T / "evaluation_costs.totals.json").read_text())
judge = rd(T / "judge_extraction_agreement.csv")
lock = json.loads((W / "reports/selection_lock.json").read_text())
n_recorded = sum(len(json.loads(Path(f).read_text())["attempts"]) for f in _glob.glob(str(W / "runs/*/*/timing.seed*.json")))
sb_part = {p: (rd(T / f"stage_b/official_scores.{p}.csv"), rd(T / f"stage_b/differences.{p}.csv"))
           for p in ("all", "development", "confirmation")}
sb_seed = rd(T / "stage_b/seed_scores.all.csv")
envd = rd(T / "environment_drift.csv")
status = json.loads((T / "evaluation_status.json").read_text())
n_units = {k: json.loads((T / f).read_text())["n_components"] for k, f in (
    ("Stage B, full split", "stage_b/bootstrap.all.json"), ("Stage B, development partition", "stage_b/bootstrap.development.json"),
    ("Stage B, confirmation partition", "stage_b/bootstrap.confirmation.json"), ("development screen", "screen/bootstrap.P1.json"),
    ("confirmation", "confirmation/bootstrap.json"))}
AU = W / "audit"
compat = json.loads((AU / "compatibility.json").read_text())
upd = json.loads((AU / "parent_update_statistics.json").read_text())
proto = json.loads((AU / "protocol_comparison.json").read_text())
repro = json.loads((AU / "metric_reproduction.json").read_text())
miss = rd(AU / "missingness.csv")
registry = json.loads((W / "manifests/metric_registry.json").read_text())
merge_ids = json.loads((W / "manifests/merges.json").read_text())
merge_man = {m: json.loads((W / "merges" / m / "merge_manifest.json").read_text()) for m in merge_ids}
split = json.loads((W / "splits/split_manifest.json").read_text())
units = json.loads((W / "splits/bootstrap_units.json").read_text())
sb_summary = json.loads((T / "stage_b/summary.json").read_text())
c_summary = json.loads((T / "confirmation/summary.json").read_text())

# server logs: vLLM version and parallelism of every generation (ports 180xx) and judge (ports 181xx) server start
vllm_versions, starts, tp_n, mem_n, bt_n = Counter(), Counter(), Counter(), Counter(), Counter()
for f in sorted(W.glob("runs/*/*/logs/*/*.log")):
    text = f.read_text(encoding="utf-8", errors="replace")
    ver = re.findall(r"vLLM API server version (\S+)", text)
    if not ver:
        continue
    vllm_versions.update(ver)
    role = "judge" if "_port_181" in f.name else "generation"
    starts[role] += 1
    for tp in set(re.findall(r"tensor_parallel_size=(\d+)", text)) or {"not logged"}:
        tp_n[(role, tp)] += 1
    mus = re.findall(r"gpu-memory-utilization ([0-9.]+)|'gpu_memory_utilization': ([0-9.]+)|"
                     r"Desired GPU memory utilization is \(([0-9.]+)", text)
    for mu in {f"{float(next(x for x in m if x)):.2f}" for m in mus} or {"not logged"}:
        mem_n[(role, mu)] += 1
    for bt in set(re.findall(r"max_num_batched_tokens=(\d+)", text)) or {"not logged"}:
        bt_n[(role, bt)] += 1
# benchmarks whose official scorer writes judge outputs (seed-42 confirmation scoring of M050); a file that only carries
# the judge's name in an accuracy table (`*_acc.csv` alone) does not count
judge_files = {d.name for d in (W / "runs/m050-p1/confirmation/scoring/seed_42/benchmark").iterdir()
               if any(("qwen3-32b-judge" in f.name or "qwen32b_judge" in str(f)) and not f.name.endswith("_acc.csv")
                      for f in d.rglob("*") if f.is_file())}
freeze = json.loads((W / "reports/behavior_report.freeze.json").read_text())
gpu_attempts = Counter()
for f in _glob.glob(str(W / "runs/*/*/timing.seed*.json")):
    for a in json.loads(Path(f).read_text())["attempts"]:
        m = re.match(r"(\d+)x (.+)", a.get("gpus", ""))
        if m:
            gpu_attempts[m.group(2).replace(" Max-Q Workstation Edition", "")] += 1

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
add("This file collects the study's key findings, its experimental settings and its data in one place. It covers only "
    "what this study set up and measured; the TRACE evaluation protocol it reuses (benchmarks, prompts, scorers, judge) "
    "is cited by revision and not re-described.")
add("- **Interpretation, caveats and the decision.** These are in `analysis_report.md` and `decision_memo.md`, in this "
    "directory. The frozen design is `../../DESIGN.md`.")
add("- **Where the numbers come from.** Numbers are read programmatically from the aggregate tables in `../tables/`, "
    "and each section names its source. §2 also reads the study config (`../../configs/`), `../manifests/`, "
    "`../splits/`, `../merges/`, `../audit/`, and the work root's run logs and scoring directories (vLLM settings, attempt "
    "hardware, judge use); §8.2 also uses the runner summaries. A few values are quoted from the study's reports "
    "instead, each with its source: the pre-repair M050 − T estimate and the 0.10 pp A7 interval change "
    "(`analysis_report.md`), the development runs' 1e-14 adapter check (`merge_screen.md`), the follow-up's cost "
    "(`decision_memo.md`) and the fault signature (DESIGN.md A9).")
add("- **Rounding.** Two decimals for percentage-point values; one for rates and for differences between rates "
    "(persistence excess); integers for token counts and mean group sizes. Rounding is half-up on the stored value. "
    "Differences and ranges are computed before rounding, so they can differ by 0.01 from the difference of the "
    "displayed values.")
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
UG = upd["groups"]
upd_n = sum(g["tensors"] for g in UG.values())
u_tot = {"base": sum(g["base_norm"] ** 2 for g in UG.values()) ** 0.5,
         "T": sum(g["T_delta_norm"] ** 2 for g in UG.values()) ** 0.5,
         "V": sum(g["V_delta_norm"] ** 2 for g in UG.values()) ** 0.5}
# each group's cosine is over its concatenated tensors, so the all-tensor cosine is the summed dot products over the norms
chance = 1 / compat["tensors"]["parameters"]["B"] ** 0.5
u_tot["cos"] = (sum((g["cos_T_V_updates"] or 0.0) * g["T_delta_norm"] * g["V_delta_norm"] for g in UG.values())
                / (u_tot["T"] * u_tot["V"]))
drift = {r: proto["environment_drift"][f"{r}_seed42_existing_local_rerun"] for r in ("B", "T")}
pc = pd.DataFrame([{"run": run, "reference": r["reference"].split("/seed=")[0].replace("archive/", ""),
                    "conforms": r["conforms"], **r["total"]}
                   for run, v in proto["prompt_conformance"].items() for r in v.values()])
repro_max = max(t["max_abs_error_pp"] for t in repro["summary"]["table"])
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
add(f"3. **The two weight updates differ in size and are nearly orthogonal** (§2.3; a diagnostic, not a test).")
add(f"   - Over all {upd_n} tensors, ‖V − B‖ is {num(u_tot['V'] / u_tot['T'], 1)} times ‖T − B‖ "
    f"({num(100 * u_tot['V'] / u_tot['base'], 3)}% against {num(100 * u_tot['T'] / u_tot['base'], 3)}% of ‖B‖).")
add(f"   - The cosine between the two updates is {num(u_tot['cos'], 4)}: small, though far above the ±{chance:.0e} "
    f"expected of two random directions in this many dimensions. Norms and cosines alone do not show that the updates "
    f"encode separable skills.")
add(f"4. **VERO's gain depends on its interface.**")
add(f"   - With the backbone template, V@P1 − V@P0 = {est_ci(cd.loc['V@P1-V@P0'])} on confirmation.")
add(f"   - {num(share(['puzzlevqa', 'visualpuzzles']), 0)}% of that macro drop is in PuzzleVQA "
    f"({sgn(vdrop['puzzlevqa'])}) and VisualPuzzles ({sgn(vdrop['visualpuzzles'])}). There, many V@P1 answers end in "
    f"a LaTeX-wrapped option letter that the pinned scorers do not read: "
    f"{num(fmt.loc[('puzzlevqa', 'V@P1'), 'official_zero_but_final_letter_correct_pct'], 1)}% of its PuzzleVQA rows "
    f"are scored 0 with the correct final letter.")
add(f"   - TableVQABench ({sgn(vdrop['tablevqabench'])}) is another {num(share(['tablevqabench']), 0)}% and is not "
    f"diagnosed.")
add(f"5. **The merge beats both controls and VERO-on-the-backbone-template, but is not shown to beat TRACE**")
add(f"   (confirmation, Holm-adjusted):")
add(f"   - M050 − T = {est_ci(fam['M050@P1-T'])}, Holm p = {pfmt(fam['M050@P1-T']['p_holm'])}.")
add(f"   - M050 − V@P1 = {est_ci(fam['M050@P1-V@P1'])}.")
add(f"   - M050 − CV050 = {est_ci(fam['M050@P1-CV050@P1'])}.")
add(f"   - M050 − CT050 = {est_ci(fam['M050@P1-CT050@P1'])}.")
add(f"   - The last three have Holm p < 0.001. No noninferiority margin was set.")
add(f"6. **M050 scores {num(ov.loc['M050@P1', 'estimate'])}**, against T {num(ov.loc['T', 'estimate'])} and native VERO "
    f"{num(ov.loc['V@P0', 'estimate'])}. M050 − V@P0 is a difference of point estimates outside the primary family; "
    f"no paired interval was computed.")
add(f"7. **Halving each update costs differently.**")
add(f"   - CT050 keeps {num(ctT, 0)}% of TRACE's gain over the base.")
add(f"   - CV050 scores {num(cvV)} pp above VERO itself under P1, consistent with half of VERO's update softening its "
    f"answer-format shift.")
add(f"8. **Category pattern (descriptive).**")
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
add(f"9. **Row-level capture** (independent repeat; the base reference is post hoc, with no paired intervals):")
add(f"   - M050 keeps VERO-only successes at {num(mv['conditional_rate_pct'], 1)}%, against the base model's "
    f"{num(bv['conditional_rate_pct'], 1)}% on the same rows.")
add(f"   - It keeps TRACE-only successes at {num(mi['conditional_rate_pct'], 1)}%, close to the base model's "
    f"{num(bi['conditional_rate_pct'], 1)}%.")
dec = {m: capi.xs(m, level=0)["merge_correct_contribution_pp"] for m in ("M050@P1", "CV050@P1", "CT050@P1", "B (reference)")}
ddiff = {o: dec["M050@P1"] - dec[o] for o in ("CV050@P1", "CT050@P1", "B (reference)")}
lead_from_shared = all(d.idxmax() == "shared_success" for d in ddiff.values())
fewer_rescue = all(d["joint_failure"] < 0 for d in ddiff.values())
add(f"   - In accuracy terms (§7.7; differences of point estimates, no paired intervals), M050's pooled binary-accuracy "
    f"lead over CV050, CT050 and B ({' / '.join(sgn(d.sum()) for d in ddiff.values())} pp) has its largest term in the "
    + ("shared successes it keeps (" + " / ".join(sgn(d["shared_success"]) for d in ddiff.values()) + " pp)"
       if lead_from_shared else "groups shown in §7.7")
    + f"; against CT050, the VERO-only successes it keeps add {sgn(ddiff['CT050@P1']['vero_only'])} pp"
    + (". It rescues fewer joint failures than any of them." if fewer_rescue else "."))
add("10. **Frozen hypotheses** (§2.10).")
add("    - H1 (integration beyond shrinkage) holds in direction.")
add("    - H2 (the category trade-off transfers) is half supported.")
add(f"    - H3 (interface dependence) is supported for the macro and for Puzzles & Logic. It is not supported for "
    f"Visual Math being one of the two largest drops: Charts & Tables drops more "
    f"({est_ci(cdc.loc[('V@P1-V@P0', 'Charts & Tables')])}, through TableVQABench), although Visual Math also drops "
    f"({est_ci(cdc.loc[('V@P1-V@P0', 'Visual Math')])}).")
within = all(drift[r]["ranges"]["official_macro_delta"]["rerun_vs_seed_pair_range"] == "within" for r in ("B", "T"))
ec = {k: c_env[k][c_env[k]["level"] == "category"].set_index(["comparison", "scope"]) for k in ("archived_BT", "local_rerun_BT")}
env_cat_shift = (ec["local_rerun_BT"]["estimate"] - ec["archived_BT"]["estimate"]).abs().max()
side = lambda r: "+" if r["ci_low"] > 1e-9 else "−" if r["ci_high"] < -1e-9 else "0"  # noqa: E731
env_cflips = [f"{comp_name(c)} in {k}" for c in ec["archived_BT"].index.get_level_values(0).unique() for k in CATS
              if side(ec["archived_BT"].loc[(c, k)]) != side(ec["local_rerun_BT"].loc[(c, k)])]
n_rerun = int(pc["run"].str.endswith("_existing_local_rerun").sum())
add("11. **Measurement checks** (§4).")
add(f"    - All {len(pc) - n_rerun} new runs and the {n_rerun} local reruns match their archived references on every one "
    f"of the {int(pc['rows'].sum()):,} rows checked: the same prompt payload and the same rendered prompt length "
    f"({int(pc['conforms'].sum())} of {len(pc)} conform).")
add(f"    - All {len(repro['summary']['accepted'])} metric adapters reproduce the archived aggregates (largest error "
    f"{repro_max:.1e} pp).")
add(f"    - No row is missing or extra in any of the {len(miss)} model × seed × benchmark slices, and no stored row "
    f"score is missing.")
add(f"    - Local reruns of B and T in the new environment move the macro by "
    f"{sgn(drift['B']['ranges']['official_macro_delta']['rerun'])} and "
    f"{sgn(drift['T']['ranges']['official_macro_delta']['rerun'])} pp, "
    + ("inside the archive's own seed-to-seed range" if within else "outside the archive's own seed-to-seed range")
    + f". B's rerun writes shorter responses (mean {sgn(drift['B']['ranges']['completion_tokens_mean_paired_shift']['rerun'], 1)} "
    f"completion tokens; {drift['B']['length_stops']['rerun']} length stops against "
    f"{' / '.join(str(v) for v in drift['B']['length_stops']['archived_by_seed'].values())} per archived seed). With "
    f"local-rerun B and T the seed-42 macro contrasts lead to the same conclusions, but category contrasts involving B "
    f"or T move by up to {num(env_cat_shift)} pp, and {len(env_cflips)} category intervals change side of 0 "
    f"({'; '.join(env_cflips)}) (§7.11).")
add("12. **One technical repair** (DESIGN amendments A8, A9). One confirmation run's judge (Qwen3-32B at "
    "tensor-parallel 4 on A800 cards) produced degraded extractions. The run was rescored at tensor-parallel 1, and every "
    "table here uses the repaired scores (§8).")
add("13. **Decision** (`decision_memo.md`).")
add("    - Do not adopt a merge as an improvement.")
add("    - Stop the merge search.")
add("    - Do not train.")
add("    - One exploratory, evaluation-only follow-up is proposed: serve M050 and CV050 with VERO's interface on the "
    "confirmation rows (six runs, roughly 30–37 GPU-hours by the memo's estimate, plus preferably three VERO reruns "
    "under P2, roughly 15–20 more). It needs a DESIGN amendment, researcher approval and GPU authorization.")
add("")

# ---------------------------------------------------------------- setup
add("## 2. Experimental setup")
add("")
add("The settings below are this study's own. Everything not listed (benchmark data, prompts, scorer code, judge prompts) "
    "is the published TRACE evaluation protocol at the pinned revisions, used unchanged.")
add("")
add("### 2.1 Models, archive and pinned stack (`configs/trace_merge_qwen25_v1.yaml`)")
add("")
roles, dt = cfg.raw["model_roles"], compat["architecture"]["declared_dtype"]
add(table(["role", "model", "Hugging Face repository", "pinned revision", "archived model id", "declared config dtype"],
          [[r, name, f"`{roles[r]['repo_id']}`", f"`{roles[r]['source_revision']}`", f"`{roles[r]['archived_model_id']}`", dt[r]]
           for r, name in (("B", "base: the shared backbone"), ("T", "TRACE, released"), ("V", "VERO, released"))]))
add("")
arc, pr = cfg.raw["archive"], cfg.raw["protocol"]
add(f"- **Declared bases.** T: `{compat['lineage']['T']['declared_base']}`. V: {compat['lineage']['V']['declared_base']}.")
add(f"- **Archive.** `{arc['repository_id']}@{arc['revision'][:12]}`, run `{arc['run_id']}`: responses, extractions and "
    f"scores of B, T and V (V@P0) for seeds {', '.join(map(str, arc['seeds']))} on all {split['rows']:,} rows of the 24 "
    f"benchmarks. They are never regenerated; every comparison with B, T or V@P0 uses them.")
add("- **Local reruns.** Seed-42 runs of B and T on the full split, made earlier for another study with the same "
    "protocol and VLMEvalKit commits and the same prepared data as the new runs (but a different runner code hash), "
    "serve only the environment check (§4.3) and one sensitivity analysis (§7.11).")
add(f"- **Evaluation stack (reused, pinned).** Protocol code `rlvr@{pr['code_revision'][:7]}`, VLMEvalKit "
    f"`{pr['vlmevalkit_revision'][:8]}`, judge `{pr['judge']['repo_id']}@{pr['judge']['revision'][:8]}`, served with vLLM "
    f"{', '.join(sorted(vllm_versions))} (every one of the {sum(vllm_versions.values())} server starts logs this version).")
add("")
add("### 2.2 Parent compatibility (`audit/compatibility.json`)")
add("")
sig, tz, tok = compat["architecture"]["signature"], compat["tensors"], compat["tokenizer"]
assert not compat["architecture"]["differences_vs_reference"] and not compat["hard_stops"]
assert len(set(tz["tensor_count"].values())) == 1 and len(set(tz["parameters"].values())) == 1
add(f"- **Architecture.** Identical in all three parents (status `{compat['status']}`, no hard stop): "
    f"`{sig['architectures'][0]}` with a {sig['text.num_hidden_layers']}-layer language model (hidden size "
    f"{sig['text.hidden_size']:,}, {sig['text.num_attention_heads']} attention and {sig['text.num_key_value_heads']} "
    f"key-value heads, vocabulary {sig['text.vocab_size']:,}, {'tied' if sig['text.tie_word_embeddings'] else 'untied'} "
    f"input and output embeddings) and a {sig['vision.depth']}-block vision encoder (hidden size {sig['vision.hidden_size']:,}).")
add(f"- **Tensors.** {tz['tensor_count']['B']} tensors and {tz['parameters']['B']:,} parameters in each parent, all stored as "
    f"{'/'.join(sorted({d for ds in tz['dtypes'].values() for d in ds}))} and matched by name, shape and dtype; "
    f"{len(tz['non_floating_tensors']) or 'no'} non-floating tensors. The configs declare B {dt['B']}, T {dt['T']} and "
    f"V {dt['V']}; vLLM serves every model in bf16.")
nwf = compat["non_weight_files"]
tok_diff = [f for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
            if len({nwf[r].get(f) for r in ("B", "T", "V")}) > 1]
tok_missing_b = [f for f in ("added_tokens.json", "special_tokens_map.json")
                 if nwf["B"].get(f) is None and nwf["T"].get(f) and nwf["T"].get(f) == nwf["V"].get(f)]
add(f"- **Tokenizer.** The same tokenizer signature in all parents: {tok['signature']['vocab_size_with_added']:,} tokens "
    f"including {tok['added_token_count']} added tokens, the same vocabulary, and {tok['embedding_rows']:,} embedding "
    f"rows. The files are not all byte-identical: {', '.join(f'`{f}`' for f in tok_diff)} differ between parents, and "
    f"{' and '.join(f'`{f}`' for f in tok_missing_b)} are shipped (identically) by T and V but not by B. Which files are "
    f"served is fixed by the inference view (§2.5).")
add("- **Recorded differences** (none affects merging):")
for x in compat["findings"]:
    add(f"  - {x.rstrip('.')}.")
ct = compat["chat_templates"]
add(f"  - The chat templates differ: B ships `{ct['B']['file']}`, T and V ship "
    f"{'different' if ct['T']['sha256'] != ct['V']['sha256'] else 'identical'} `{ct['T']['file']}` files. T's renders "
    f"the same prompts as B's (§2.5).")
add("")
add("### 2.3 How far each parent moved from the base (`audit/parent_update_statistics.json`; diagnostic)")
add("")
add("Frobenius norms per tensor group, computed in float64 from the stored BF16 weights. Relative update = ‖parent − B‖ / "
    "‖B‖. The cosine is between the flattened updates T − B and V − B. The all-tensor row is derived from the group rows.")
add("")
GN = {"language_embeddings": "language: input embeddings", "language_layers": "language: decoder layers",
      "language_final_norm": "language: final norm", "language_output_head": "language: output head",
      "visual_patch_embed": "vision: patch embedding", "visual_blocks": "vision: blocks", "visual_merger": "vision: merger"}


def urow(name, n, base, t, v, cos, eq):
    return [name, n, num(base, 1), num(t, 3), num(v, 3), num(100 * t / base, 3), num(100 * v / base, 3),
            num(v / t, 2) if t else "–", num(cos, 4) if cos is not None else "–", eq]


eqc = upd["bitwise_equal_tensor_counts"]
add(table(["tensor group", "tensors", "‖B‖", "‖T − B‖", "‖V − B‖", "T relative update (%)", "V relative update (%)",
           "‖V − B‖ / ‖T − B‖", "cosine(T − B, V − B)", "tensors bitwise equal to B (T / V)"],
          [urow(GN.get(k, k), UG[k]["tensors"], UG[k]["base_norm"], UG[k]["T_delta_norm"], UG[k]["V_delta_norm"],
                UG[k]["cos_T_V_updates"], f"{UG[k]['T_tensors_bitwise_equal_to_B']} / {UG[k]['V_tensors_bitwise_equal_to_B']}")
           for k in [*[g for g in GN if g in UG], *[g for g in UG if g not in GN]]]
          + [urow("**all tensors**", upd_n, u_tot["base"], u_tot["T"], u_tot["V"], u_tot["cos"], f"{eqc['T']} / {eqc['V']}")]))
add("")
moved = {k: g for k, g in UG.items() if g["T_delta_norm"] > 0 and g["V_delta_norm"] > 0}
rat = {k: g["V_delta_norm"] / g["T_delta_norm"] for k, g in moved.items()}
cosg = {k: g["cos_T_V_updates"] for k, g in moved.items()}
lo_r, hi_r, hi_c = min(rat, key=rat.get), max(rat, key=rat.get), max(cosg, key=lambda k: abs(cosg[k]))
add(f"- VERO moved the weights {num(u_tot['V'] / u_tot['T'], 1)} times as far as TRACE did, and further in every group "
    f"that changed: the ratio runs from {num(rat[lo_r], 2)} ({GN[lo_r]}) to {num(rat[hi_r], 2)} ({GN[hi_r]}).")
add(f"- The cosines are small: {num(u_tot['cos'], 4)} over all tensors, largest in absolute value {num(cosg[hi_c], 4)} "
    f"({GN[hi_c]}), against about ±{chance:.0e} for two random directions in {compat['tensors']['parameters']['B']:,} "
    f"dimensions. Averaging therefore cancels little of either update, but each enters a 50:50 merge at half "
    f"strength. The audit's caveat applies: {upd['note'].split('; ', 1)[1]}.")
fin = [GN[k] for k, g in UG.items() if g["tensors"] == g["T_tensors_bitwise_equal_to_B"] == g["V_tensors_bitwise_equal_to_B"]]
add(f"- Untouched by both: {', '.join(fin) or 'none'}. Of the {eqc['T']} (T) and {eqc['V']} (V) tensors bitwise equal "
    f"to B, {UG['visual_blocks']['T_tensors_bitwise_equal_to_B']} and {UG['visual_blocks']['V_tensors_bitwise_equal_to_B']} "
    f"are in the vision blocks.")
add("")
add("### 2.4 Merge construction (`merges/*/merge_manifest.json`, `manifests/merges.json`)")
add("")
ISO = {"M050": "M050 − B = ½(T − B) + ½(V − B)", "CV050": "M050 − CV050 = ½(T − B)", "CT050": "M050 − CT050 = ½(V − B)"}
ROLE = {"complementarity_test": "complementarity test", "vero_update_shrinkage_control": "VERO update-shrinkage control",
        "trace_update_shrinkage_control": "TRACE update-shrinkage control"}
add(table(["merge", "weights", "role", "weight-level identity", "recipe sha256", "output content revision"],
          [[m, " + ".join(f"{'½' if c == 0.5 else c}{r}" for r, c in spec["coefficients"].items()),
            ROLE.get(spec["role"], spec["role"]), ISO[m], f"`{merge_ids[m]['recipe_sha256'][:12]}…`",
            f"`{merge_ids[m]['content_revision'][:22]}…`"] for m, spec in cfg.merges.items()]))
add("")
mb, mm = cfg.raw["merge_build"], merge_man["M050"]
assert all(merge_man[m]["compatibility_status"] == "compatible" and merge_man[m]["no_training"] for m in merge_man)
n_shards = sum(1 for k in mm["output_file_hashes"] if k.endswith(".safetensors"))
add(f"- **Arithmetic.** All {upd_n} tensors are interpolated. Each parent's BF16 values are upcast to "
    f"{mb['accumulation_dtype']}, combined with the coefficients above, and cast once to {mb['output_dtype']} (round to "
    f"nearest even). The untied input embeddings and output head are interpolated as separate tensors. Non-floating "
    f"buffers would be copied only if bitwise identical in every parent; there are none.")
add(f"- **Streaming.** Tensors are processed in slices of at most {mb['max_slice_elements']:,} elements; each output keeps "
    f"B's tensor-to-shard layout ({n_shards} shards).")
add("- **Weights only.** No optimizer, trainer state or gradient is involved. The merged directories hold weights only; "
    "the tokenizer, chat template, processor and config come from the inference view (§2.5).")
add(f"- **Provenance.** Parent shard hashes are in `audit/parent_weight_sha256.json`. Each merge manifest records its "
    f"output file hashes and build environment (Python {mm['environment_versions']['python']}, torch "
    f"{mm['environment_versions']['torch']}). The three merges take {totals['merged_weights_gib']} GiB.")
mch = json.loads((AU / "merge_code_hashes.json").read_text())
if all(merge_man[m].get("implementation_dirty") for m in merge_man):
    add(f"- **Build code.** The merges were built from an uncommitted working tree on top of "
        f"`{mm['implementation_commit'][:7]}` (`implementation_dirty` in every manifest); the hashes of the merge and "
        f"view code used are in `audit/merge_code_hashes.json` (recorded {mch['recorded'][:19]}Z).")
add("")
add("### 2.5 Inference interfaces (`manifests/views.json`, `audit/protocol_comparison.json`)")
add("")
add("A merged checkpoint has no chat template of its own, so the non-weight files served with each model's weights (its "
    "inference view) are fixed per protocol in advance. Templates are never averaged and never chosen by score.")
add("")
ptm, PV, xp = proto["prompt_tokens_median"], cfg.raw["protocols"], proto["cross_model_pairs"]


def med_prompt(tier, suffix):
    return "/".join(f"{int(v):,}" for v in sorted({v for k, v in ptm.items() if f"_{tier}_temp06" in k and k.endswith(suffix)}))


add(table(["protocol", "non-weight files from", "role", "models served", "median prompt tokens"], [
    ["P0 (native)", "each parent's own", "the archived canonical runs", "B, T, V@P0",
     f"full split: B {int(ptm['archive/B']):,}, T {int(ptm['archive/T']):,}, V@P0 {int(ptm['archive/V@P0']):,}"],
    ["P1 (backbone)", f"B ({len(PV['P1_backbone']['files'])} files)", "primary, for every merge and control",
     "V@P1, M050, CV050, CT050; B and T need no rerun (their P0 prompts are P1 prompts)",
     f"smoke {med_prompt('smoke', '@P1')}, screen {med_prompt('development_screen', '@P1')}, confirmation "
     f"{med_prompt('confirmation', '@P1')}"],
    ["P2 (VERO)", f"V ({len(PV['P2_vero']['files'])} files)", "diagnostic, development screen only, never used to select",
     "B@P2, T@P2, M050@P2, CV050@P2, CT050@P2; V@P2 is the archived V@P0", f"screen {med_prompt('development_screen', '@P2')}"]]))
add("")
add("Prompt medians differ between tiers because the row sets differ; compare them only within a tier.")
add(f"- **P1 files:** {', '.join(f'`{x}`' for x in PV['P1_backbone']['files'])}.")
add(f"- **P2 files:** {', '.join(f'`{x}`' for x in PV['P2_vero']['files'])}.")
add(f"- **B and T** render every prompt identically: equal prompt token counts on "
    f"{num(100 * xp['B|T']['prompt_tokens_equal_rate'], 1)}% of {xp['B|T']['prompt_tokens_known_both']:,} archived rows. "
    f"T's view lacks `<|endoftext|>` (151643) as an extra stop id; B's and V's list it.")
add(f"- **V's template** replaces any system message with a fixed reasoning prompt (think inside `<think>`, give the final "
    f"answer inside `<answer>` with `\\boxed{{}}`) and pre-fills `<think>\\n` after the assistant header. On the same rows "
    f"its prompts are a median {int(xp['B|V@P0']['prompt_tokens_median_paired_diff_b_minus_a'])} tokens longer than B's.")
add("- **Images.** All three parents load the fast Qwen2-VL image processor with the same pixel bounds and normalization.")
add("")
add("### 2.6 Decoding, serving and hardware")
add("")
g = pr["generation"]
add(table(["setting", "value"], [
    ["temperature", g["temperature"]], ["top-p", g["top_p"]], ["top-k", "off (−1)" if g["top_k"] == -1 else g["top_k"]],
    ["presence penalty", g["presence_penalty"]], ["repetition penalty", g["repetition_penalty"]],
    ["max new tokens", f"{g['max_tokens']:,}"], ["image pixels", f"{g['min_image_pixels']:,} to {g['max_image_pixels']:,}"],
    ["media transport", g["media_transport"]], ["weights served as", "bf16"],
    ["decoding seeds", "42 (smoke, screen); 42, 43, 44 (Stage B, confirmation)"]]))
add("")
ns = proto["normalized_settings"]
producer_rev = sorted({v for part in proto["archive_part_provenance"].values() for m in part.values()
                       for v in m.get("producer_code_revision", [])})[0]
add(f"- **One settings set.** Apart from model id and seed, these settings are identical on all "
    f"{sum(v['slices'] for v in ns['values'].values())} archived and new model × seed × benchmark slices "
    f"({ns['distinct']} distinct set).")
def tally(c, role):
    return ", ".join(f"{k} in {n}" for (r, k), n in sorted(c.items()) if r == role)


add(f"- **Generation servers** ({starts['generation']} starts). One vLLM replica per GPU, data-parallel across a node's "
    f"GPUs; tensor-parallel degree {tally(tp_n, 'generation')}. GPU memory utilization {tally(mem_n, 'generation')}: "
    f"0.90 by default, 0.80 on smaller cards after two A40 servers ran out of memory at 0.90 (A5(b)). Requests in "
    f"flight per endpoint: 64 on cards below 80 GB and 128 on 80 GB or more (protocol default 32; A7(c)). None of this "
    f"changes a decoding setting.")
add(f"- **Judge servers** ({starts['judge']} starts). Qwen3-32B at tensor-parallel degree {tally(tp_n, 'judge')}: 1 on "
    f"≥90 GB cards and 2 on 44–80 GB cards (A7(c))"
    + ("; the single degree-4 start is the faulty scoring of A8 (§8)" if tp_n.get(("judge", "4")) == 1 else "")
    + f". Max batched tokens {tally(bt_n, 'judge')}. Judge weights are verified by marker, the job scripts' default; "
    f"the timing files record it only for attempts after A7 (A9(c)).")
add(f"- **GPUs** (card type of the {sum(gpu_attempts.values())} recorded attempts that name their cards): "
    f"{', '.join(f'{k} ({n})' for k, n in gpu_attempts.most_common())}.")
add(f"- **Archive versus new runs.** The archived runs come from the canonical producer (recorded code revision "
    f"`{producer_rev[:7]}`, not assumed identical to the published `{pr['code_revision'][:7]}`); §4.3 measures what "
    f"the change of environment does.")
add("")
add("### 2.7 Scoring and metrics (`manifests/metric_registry.json`, `audit/metric_reproduction.json`)")
add("")
jb = audit.groupby("benchmark_id")["used_judge_rate"].max()
judged = [b for _, b in ORDER if jb.get(b, 0) > 0]
kinds, binary = registry["kinds"], sb_summary["binary_benchmarks"]
judge_official = [b for _, b in ORDER if b in judge_files and b not in judged]
add(f"- **Scorers.** Each benchmark's official primary metric, computed by the pinned scorer. The Qwen3-32B judge is the "
    f"extraction or grading route on {len(judged)} benchmarks: {', '.join(BNAME.get(b, b) for b in judged)}. The "
    f"official VLMEvalKit scorer also writes judge outputs on {len(judge_official)} more "
    f"({', '.join(BNAME.get(b, b) for b in judge_official)}), where it grades or extracts answers or matches "
    f"multiple-choice answers. So {len(judged) + len(judge_official)} of the 24 benchmarks can depend on the judge. On "
    f"the {len(judged)} route benchmarks every row goes through the judge; for the {len(judge_official)} others the "
    f"bundle does not record how many rows reached it.")
add("- **Fail-closed fixes** (scorer side, DESIGN.md §1). Three are added: the CharXiv judge parser, which the public "
    "release omits; MME-Reasoning official-function exceptions are scored incorrect; judge output still unusable after "
    "the official five attempts is scored incorrect. They change only rows on which the canonical producer would have "
    "raised an error. The bundle does not record how many rows took these paths.")
add(f"- **Row scores.** Paired resampling needs official scores per row. {len(kinds['binary_row'])} benchmarks have 0/1 "
    f"row scores whose mean is the official value. WeMath's official unit is the problem group. On "
    f"{len(kinds['aggregate_only_rederived'])} benchmarks the archive holds only the aggregate, so row scores are re-derived "
    f"from the archived extractions with the pinned scorer functions, with no new judge calls.")
add(f"- **Adapter acceptance.** Each adapter had to reproduce the archived aggregate of all 9 model × seed slices of its "
    f"benchmark to {repro['tolerance_pp']:g} pp. All {len(repro['summary']['accepted'])} were accepted "
    f"({len(repro['summary']['not_accepted'])} rejected); the largest error is {repro_max:.1e} pp.")
add("- **WeMath on subsets** (A2). The pinned score counts every problem group absent from a subset as mastered, so the "
    "study uses 100 × (G − 0.5 IG − RM − IK) / G over the G complete problem groups present. It equals the pinned value on "
    "the full split.")
excl = [BNAME.get(b, b) for _, b in ORDER if b not in binary]
add(f"- **Row diagnostics** (A3). Corrections, regressions, outcome cells, overlap, persistence and capture/damage use the "
    f"{len(binary)} benchmarks whose official value is the mean of 0/1 row scores. {', '.join(excl)} stay in every macro "
    f"but not in these diagnostics (fractional row scores, a mean of five split scores, and a group metric).")
add(f"- **Macro.** Equal weight on each of the {len(ORDER)} benchmarks; each of the {len(CATS)} category scores is the mean "
    f"of its {len(ORDER) // len(CATS)} benchmarks.")
add("")
add("### 2.8 Benchmarks, partition and resampling units (`splits/split_manifest.json`, `splits/bootstrap_units.json`)")
add("")
KIND = {"binary_row": "row mean", "grouped_row": "problem groups (A2)", "aggregate_only_rederived": "aggregate; rows re-derived"}
kind_of = {b: k for k, bs in kinds.items() for b in bs}
twins, PB = units["confirmation_rows_with_development_twin_by_benchmark"], split["per_benchmark"]
prow = [[cat, BNAME.get(b, b), KIND[kind_of[b]], "route" if b in judged else "in scorer" if b in judge_official else "",
         "yes" if b in binary else "",
         f"{PB[b]['rows']:,}", f"{PB[b]['components']:,}",
         f"{PB[b]['development_rows']:,} ({num(100 * PB[b]['development_rows'] / PB[b]['rows'], 1)}%)",
         PB[b]["dev_screen_rows"],
         PB[b]["smoke_rows"], f"{PB[b]['confirmation_rows']:,}", twins.get(b, "")] for cat, b in ORDER]
prow.append(["**all**", "", "", f"{len(judged)} route, {len(judge_official)} in scorer", len(binary), f"{split['rows']:,}", f"{split['components']:,}",
             f"{split['development_rows']:,} ({num(100 * split['development_rows'] / split['rows'], 1)}%)",
             f"{split['dev_screen_rows']:,}", split["smoke_rows"], f"{split['confirmation_rows']:,}",
             units["confirmation_rows_with_development_twin"]])
add(table(["category", "benchmark", "official value", "judge", "row diagnostics", "rows", "components",
           "development rows (share)", "screen rows", "smoke rows", "confirmation rows",
           "confirmation rows with a development twin"], prow))
add("")
add("The `all` row counts components that span several benchmarks once; per-benchmark component counts include them.")
add(f"- **Components.** Rows that share any image (by SHA-256, across all benchmarks) or a WeMath problem family are joined "
    f"into {split['components']:,} connected components: {split['singleton_components']:,} single rows, "
    f"{split['cross_benchmark_components']} that span several benchmarks, and at most {split['largest_component_rows']} "
    f"rows in one. Components are the unit of splitting.")
add(f"- **Partition** (seed {split['seed']}, frozen {split['frozen_at']}). Each component is ranked by "
    f"`blake2b(\"{split['seed']}|\" + component id)`. Cross-benchmark components with rank fraction below 0.20 go to "
    f"development; then, per benchmark, single-benchmark components join development in rank order until it holds "
    f"round(0.20 × rows). The rest is confirmation. Every seed and question variant of a component stays in one partition.")
add("- **Development screen.** Per benchmark, development components in rank order until at least 128 rows (all "
    "development rows when fewer), plus the best-ranked component carrying each missing value of the strata the "
    "official scorer divides by (TableVQABench `split`; PuzzleVQA `category`; VisualPuzzles `category` and `difficulty`).")
add(f"- **Smoke.** The target is 4 rows per benchmark; whole components and stratum coverage make it {split['smoke_rows']} "
    f"rows (A6(c)).")
add(f"- **Near-duplicate twins** (A7). {units['twin_pairs']} row pairs across MathVista/MMStar and MME-Reasoning/PuzzleVQA "
    f"show the same picture re-encoded, with the same question and answer, so byte-level image matching did not join "
    f"them. The partition is kept; for every bootstrap their components are merged, giving {units['bootstrap_units']:,} "
    f"resampling units from {units['components']:,} components (frozen {units['frozen_at']}). The list is a lower bound: "
    f"resized copies and within-benchmark duplicates were not searched.")
add(f"- **Resampling units per analysis:** {'; '.join(f'{k} {v:,}' for k, v in n_units.items())}.")
add("- **Prior exposure.** The suite was examined in earlier work. The confirmation partition was reserved from merge "
    "tuning, but it is not pristine.")
add("")
add("### 2.9 Evaluation matrix")
add("")
add(table(["tier", "rows", "seeds", "new generation runs", "reused from the archive", "use"], [
    ["Stage B", f"{split['rows']:,}", "42, 43, 44", "none", "B, T, V@P0", "parent behaviour; frozen hypotheses"],
    ["smoke", split["smoke_rows"], "42", "P1: V@P1, M050, CV050, CT050", "none",
     "technical gate for each P1 model (the P2 diagnostic runs had none)"],
    ["development screen", f"{split['dev_screen_rows']:,}", "42",
     "P1: V@P1, M050, CV050, CT050; P2: B@P2, T@P2, M050@P2, CV050@P2, CT050@P2", "B, T, V@P0 (= V@P2)",
     "descriptive screen; selects nothing"],
    ["confirmation", f"{split['confirmation_rows']:,}", "42, 43, 44", "P1: V@P1, M050, CV050, CT050 (12 runs)", "B, T, V@P0",
     "primary family and every confirmation table"]]))
add("")
n_ok = sum(1 for s in status if s["state"] == "complete" and s["scored_complete"] and not s["problems"])
add(f"- {len(status)} new generation runs in all; {n_ok} are complete and fully scored (`tables/evaluation_status.json`).")
add("- Not run: training of any kind, coefficients other than 0.5, component-wise merges and other merge algorithms "
    "(Stage D was not opened).")
add("- Confirmation ran concurrently with the development screen (A4). The confirmed set and the family were fixed in the "
    "design, so the screen could not select anything.")
add("")
add("### 2.10 Statistics, selection and frozen hypotheses")
add("")
st, cboot = cfg.raw["statistics"], json.loads((T / "confirmation/bootstrap.json").read_text())
add(f"- **Bootstrap.** {st['bootstrap_replicates']:,} replicates (seed {st['seed']}). Resampling units are drawn with "
    f"replacement globally, with one draw shared by every model and seed, so every contrast is paired. Benchmark weights "
    f"stay equal, and WeMath's group metric is recomputed in each replicate.")
add(f"- **Intervals.** {int(100 * st['confidence'])}%, {cboot['ci_method']}.")
add("- **p-values.** Two-sided percentile-inversion bootstrap p-values: p = min(1, 2 (min(#{d* ≤ 0}, #{d* ≥ 0}) + 1) / "
    "(R + 1)) over the R replicate differences d*. p < 0.05 exactly when 0 lies outside the 95% interval, up to the "
    "add-one correction.")
add(f"- **Multiplicity.** Holm over the primary family only: {', '.join(comp_name(c) for c in FAMILY)} (confirmation, "
    f"24-benchmark macro). Everything else is descriptive or exploratory and unadjusted.")
sc_flag = s_cap[s_cap["below_min_groups"].astype(str) == "True"]["level"].unique().tolist()
small = [BNAME.get(b, b) for _, b in ORDER if PB[b]["dev_screen_components"] < st["exploratory_min_groups_per_slice"]]
add(f"- **Small slices.** The behaviour and capture/damage tables flag slices with fewer than "
    f"{st['exploratory_min_groups_per_slice']} resampling groups; the only flagged ones are the development screen's "
    f"capture/damage slices at {', '.join(sc_flag)} level, which this file does not report. Score and contrast tables "
    f"carry no such flag: on the development screen {len(small)} benchmarks have fewer than "
    f"{st['exploratory_min_groups_per_slice']} components ({', '.join(small)}), so their per-benchmark values "
    f"(§6.7–§6.8) and intervals (§6.9–§6.10) rest on few groups.")
add("- **Selection.** None. There is no coefficient search, so all three midpoints and both controls went to "
    "confirmation as prespecified. No category regression margin or tie tolerance was set, so no noninferiority or "
    "no-harm claim is made.")
lk = c_summary["lock"]
add(f"- **Lock.** Before any confirmation run had scores, `reports/selection_lock.json` froze the config, design, split, "
    f"merge and view hashes and the confirmed labels and seeds ({lock['locked_at']}). `confirm` re-checks them; "
    f"amendments appended since the lock: {', '.join(lk['amendments_after_lock'])}. The lock does not record "
    f"{', '.join(f'`{x}`' for x in lk['fields_not_recorded_by_lock'])}, so it cannot show runner changes: the pandas "
    f"guard was added (A5) and then narrowed (A7(b)) after it.")
add("")
add(f"Frozen hypotheses (`behavior_report.md` §6, written from Stage B and frozen at {freeze['frozen_at']}, before any "
    f"merged model had generated a response) and their outcome on confirmation:")
add("")
h1 = {v: cap[(cap["benchmark_set"] == "all_binary") & (cap["variant"] == v)].set_index(["merge_label", "group"])["conditional_rate_pct"]
      for v in ("independent_repeat", "contemporaneous", "stable_other_seeds")}
h1_all = all(h[("M050@P1", "trace_only")] > h[("CV050@P1", "trace_only")] and h[("M050@P1", "vero_only")] > h[("CT050@P1", "vero_only")]
             for h in h1.values())
hi = h1["independent_repeat"]
cv_cat, ct_cat = cdc.loc["M050@P1-CV050@P1"]["estimate"], cdc.loc["M050@P1-CT050@P1"]["estimate"]
vv_cat = cdc.loc["V@P1-V@P0"]["estimate"].sort_values()
add(table(["hypothesis", "frozen prediction", "confirmation outcome", "verdict"], [
    ["H1: integration beyond shrinkage",
     "M050 keeps more TRACE-only successes than CV050, and more VERO-only successes than CT050 (independent repeat)",
     f"TRACE-only: M050 {num(hi[('M050@P1', 'trace_only')], 1)}% vs CV050 {num(hi[('CV050@P1', 'trace_only')], 1)}%; "
     f"VERO-only: M050 {num(hi[('M050@P1', 'vero_only')], 1)}% vs CT050 {num(hi[('CT050@P1', 'vero_only')], 1)}%; "
     f"{'same direction in all three variants' if h1_all else 'not in every variant'}",
     f"holds in direction; but the base model itself keeps {num(hi[('B (reference)', 'trace_only')], 1)}% of "
     f"TRACE-only successes (post hoc)"],
    ["H2: the category trade-off transfers",
     "M050 − CV050 is largest in Charts & Tables; M050 − CT050 is largest in Visual Math",
     f"M050 − CV050 largest in {cv_cat.idxmax()} ({sgn(cv_cat.max())}; Charts & Tables {sgn(cv_cat['Charts & Tables'])}); "
     f"M050 − CT050 largest in {ct_cat.idxmax()} ({sgn(ct_cat.max())})", "half supported"],
    ["H3: part of VERO's gain is interface-dependent",
     "V@P1 < V@P0 on the macro, with the largest drops in Visual Math and Puzzles & Logic",
     f"macro {sgn(cd.loc['V@P1-V@P0', 'estimate'])}; largest drops {vv_cat.index[0]} ({sgn(vv_cat.iloc[0])}) and "
     f"{vv_cat.index[1]} ({sgn(vv_cat.iloc[1])}); Visual Math {sgn(vv_cat['Visual Math'])}",
     "supported for the macro and for Puzzles & Logic; not for Visual Math"]]))
add("")
add("### 2.11 Design amendments and timeline (`DESIGN.md`)")
add("")
add("Amendment stamps are DESIGN.md's, in MST (UTC−7); file timestamps are UTC.")
add("")
add(table(["amendment", "written (MST)", "subject", "what it changed", "effect on reported numbers"], [
    ["A1", "2026-11-02", "row identity", "Archived CountQA rows take the prepared manifest's row hash: the public code hashes "
     "the cached answer as a string, the canonical producer as an integer (both recomputed on all 1,528 rows)", "none"],
    ["A2", "2026-11-02", "WeMath on subsets", "Study metric over the complete problem groups present (§2.7)",
     "WeMath subset values; full-split values unchanged"],
    ["A3", "2026-11-02", "row diagnostics", "Binary diagnostics restricted to the 21 benchmarks with 0/1 official rows",
     "defines the diagnostic set"],
    ["A4", "2026-11-02 02:05", "schedule", "Confirmation runs concurrently with the development screen, to use idle GPUs",
     "none"],
    ["A5", "2026-11-02 02:15", "environment", "A pandas guard against a parser crash on hash strings; GPU memory "
     "utilization 0.80 on 44–48 GB cards", "none"],
    ["A6", "2026-11-02 02:37", "record", "Amendment timing, the lock's prefix rule for DESIGN.md, the 135-row smoke tier",
     "none"],
    ["A7", "2026-11-02 03:35", "resampling units and serving", "Near-duplicate twins joined into bootstrap units, with a "
     "twin-free sensitivity analysis; guard wording; serving parameters",
     "intervals (those quoted in the frozen Stage B report moved by at most 0.10 pp, per analysis_report.md); point "
     "estimates unchanged"],
    ["A8", "2026-11-03 00:10", "judge fault", "M050 confirmation seed 44 rescored with the judge at tensor-parallel 1 "
     "after a degraded tensor-parallel-4 scoring; faulty outputs moved to quarantine, generation kept",
     "M050 seed-44 scores on judge-dependent benchmarks; the M050 − T estimate moved from −0.29 (superseded tables) to "
     "+0.23 (§8.2)"],
    ["A9", "2026-11-03 02:10", "record", "Corrects A8's description of the fault mechanism and A7(c)'s note on timing "
     "files", "none"]]))
add("")
add(table(["event", "time (UTC)"], [["split frozen", split["frozen_at"]], ["selection lock written", lock["locked_at"]],
                                    ["bootstrap units frozen (A7)", units["frozen_at"]]]))
add("")

# ---------------------------------------------------------------- runs
add("## 3. Runs")
add("")
add("Every new generation run, with its final attempt's hardware. Source: `../tables/evaluation_costs.csv` and each "
    "run's `timing.seed<S>.json`.")
add("- B, T and V@P0 are not re-run; they come from the archive.")
add(f"- GPU-hours and the attempts column count the {totals['attempts']} attempts with a timed phase. The timing "
    f"files record {n_recorded} attempts; {n_recorded - totals['attempts']} stopped right after starting, with no "
    f"timed phase (smoke M050, smoke V@P1, V@P1 confirmation seed 44).")
add("- Wall hours and GPU-hours sum all timed attempts; a resumed attempt continues where the previous one stopped.")
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
                     hw + note, num(r.wall_hours), num(r.gpu_hours), f"{int(r.completion_tokens):,}"])
add(table(["tier", "model", "seed", "rows", "attempts", "final attempt: host, GPUs", "wall hours", "GPU-hours",
           "generated tokens"], run_rows))
add("")
add(f"Totals: {totals['gpu_hours']} recorded GPU-hours over {totals['attempts']} attempts "
    f"({totals['failed_or_incomplete_attempts']} failed or incomplete, all resumed) and "
    f"{totals['completion_tokens']:,} generated tokens. Merged weights take {totals['merged_weights_gib']} GiB; run "
    f"directories take {totals['run_dirs_gib']} GiB.")
add("")
ulog = W / "runs/m050-p1/confirmation/logs/eval_subset.seed44.16193485.3572.1102_052739.log"
uts = [ln[:19] for ln in ulog.read_text(encoding="utf-8", errors="replace").splitlines() if re.match(r"\d{4}-\d\d-\d\d \d\d:", ln)]
uh = (datetime.fromisoformat(uts[-1]) - datetime.fromisoformat(uts[0])).total_seconds() / 3600
add(f"One attempt is not in the timing files: M050 confirmation seed 44 first generated all its rows on 2× A40 in about "
    f"{num(uh, 1)} hours by its step log (about {num(2 * uh, 1)} GPU-hours) before its allocation expired. Its rows were "
    f"kept.")
add("")

# ---------------------------------------------------------------- measurement checks
add("## 4. Measurement checks")
add("")
add("### 4.1 Prompts (`audit/protocol_comparison.json`)")
add("")
add("Each run's prompts were compared row by row with an archived run of the same seed: P1 runs with archived B (B and T "
    "render identically), P2 runs with archived V@P0, and each local rerun with its own archived model. The prompt digest "
    "covers the message payload before the chat template is applied; the rendered prompt's token count is the check on "
    "the template.")
add("")
PC_ORDER = ["smoke, P1", "development screen, P1", "development screen, P2", "confirmation, P1", "local reruns, full split"]
pc["group"] = [("local reruns, full split" if r.endswith("_existing_local_rerun")
                else f"{r.split('.')[1].replace('_', ' ')}, {r.split('.')[0].split('@')[1]}") for r in pc["run"]]
pcg = pc.groupby(["group", "reference"]).agg(runs=("run", "size"), ok=("conforms", "sum"), rows=("rows", "sum"),
                                             prompt=("prompt_sha256_equal", "sum"), tokens=("prompt_tokens_equal", "sum"),
                                             neutral=("neutral_request_sha256_equal", "sum")).reset_index()
pcg = pcg.sort_values(["group", "reference"], key=lambda s: s.map(PC_ORDER.index) if s.name == "group" else s)
add(table(["tier, protocol", "archived reference", "runs", "conforming", "rows", "same prompt payload digest",
           "same rendered prompt token count", "same request digest without model id and seed"],
          [[r.group, r.reference, r.runs, r.ok, f"{r.rows:,}", f"{r.prompt:,}", f"{r.tokens:,}", f"{r.neutral:,}"]
           for r in pcg.itertuples()]))
add("")
add("")
ing = {}
for f in sorted((W / "aligned/local_runs").glob("*.report.json")):
    if "_existing_local_rerun" in f.name:
        continue
    d = json.loads(f.read_text())
    ing[f.name] = (d["status"] == "verified", all(x.get("campaign_summary") == "equal" for x in d["slices"]), len(d["slices"]))
n_ing_ok = sum(1 for v in ing.values() if v[0] and v[1])
add(f"Ingestion (the work root's per-run ingest reports): {n_ing_ok} of the {len(ing)} new runs are `verified`, with the "
    f"study's aggregate equal to the run's own score-campaign aggregate on all {next(iter(ing.values()))[2]} benchmark "
    f"slices. For the nine development-screen runs, `merge_screen.md` also reports the largest adapter difference "
    f"(below 1e-14) and that the served weights hash to the audited parent or merge.")
add("")
add("### 4.2 Completeness (`audit/missingness.csv`)")
add("")
ms = miss[["expected_rows", "response_rows", "missing_rows", "extra_rows", "excluded_rows", "row_score_unavailable_rows",
           "finish_reason_null"]].sum()
SRC = {"archive": "archived", "existing_local_rerun": "local-rerun", "local_run": "new-run"}
src_n = miss.groupby("source").size()
mk = miss.groupby("score_kind")[["expected_rows", "row_score_rows"]].sum()
agg_rows = int(mk.loc["aggregate_only", "expected_rows"]) if "aggregate_only" in mk.index else 0
add(f"Across all {len(miss)} model × seed × benchmark slices ({', '.join(f'{n} {SRC.get(k, k)}' for k, n in src_n.items())}), "
    f"{int(ms['expected_rows']):,} rows were expected and {int(ms['response_rows']):,} have responses: "
    f"{int(ms['missing_rows'])} missing, {int(ms['extra_rows'])} extra, {int(ms['excluded_rows'])} excluded and "
    f"{int(ms['finish_reason_null'])} without a finish reason. The {int(mk['row_score_rows'].sum()):,} rows of the "
    f"row-scored benchmarks all carry a stored row score ({int(ms['row_score_unavailable_rows'])} unavailable); the "
    f"{agg_rows:,} rows of the aggregate-only benchmarks store none, and their row scores are re-derived (§2.7).")
add("")
add("### 4.3 Environment: local reruns of B and T against the archive (`audit/protocol_comparison.json`, "
    "`tables/environment_drift.csv`)")
add("")
add("The archived runs come from the canonical producer; the new runs from the published code on the GPUs listed in §2.6. "
    "Seed-42 reruns of B and T on the full split in the new environment are compared with the archived seed-42 runs. As a yardstick, the same "
    "quantities are shown for the three pairs of archived seeds (42–43, 42–44, 43–44), which differ only in decoding seed.")
add("")
MET = [("official_macro_delta", "macro difference (pp)", sgn),
       ("official_delta_mean_abs", "mean absolute benchmark difference (pp)", num),
       ("official_delta_max_abs", "largest absolute benchmark difference (pp)", num),
       ("row_score_agreement_rate_pooled", "row-score agreement (%)", lambda x: num(100 * x, 1)),
       ("response_text_equal_rate", "identical response text (%)", lambda x: num(100 * x, 1)),
       ("completion_tokens_mean_paired_shift", "mean paired change in completion tokens", lambda x: sgn(x, 1))]
drows = []
for k, name, f in MET:
    row = [name]
    for r in ("B", "T"):
        x = drift[r]["ranges"][k]
        row += [f(x["rerun"]), f"{f(x['seed_pairs_min'])} to {f(x['seed_pairs_max'])}"]
    drows.append(row)
drows.append(["length stops (rerun; archived seeds 42 / 43 / 44)"]
             + [c for r in ("B", "T") for c in (drift[r]["length_stops"]["rerun"],
                                                " / ".join(str(v) for v in drift[r]["length_stops"]["archived_by_seed"].values()))])
add(table(["quantity", "B: rerun − archive, seed 42", "B: archived seed pairs", "T: rerun − archive, seed 42",
           "T: archived seed pairs"], drows))
add("")
pos = {r: drift[r]["ranges"]["official_macro_delta"]["rerun_vs_seed_pair_range"] for r in ("B", "T")}
pb = {f"{r}, {k.split('_')[2]}": drift[r]["ranges"][k]["rerun_vs_seed_pair_range"]
      for r in ("B", "T") for k in ("official_delta_mean_abs", "official_delta_max_abs")}
add((f"- The macro difference is {pos['B']} the seed-pair range for both B and T. " if pos["B"] == pos["T"] else
     f"- The macro difference is {pos['B']} the seed-pair range for B and {pos['T']} it for T. ")
    + ("The mean and the largest absolute per-benchmark differences are below the seed-pair range for both."
       if set(pb.values()) == {"below"} else
       "Per-benchmark differences against the seed-pair range: " + "; ".join(f"{k} {v}" for k, v in pb.items()) + "."))
tok_pos = {r: drift[r]["ranges"]["completion_tokens_mean_paired_shift"]["rerun_vs_seed_pair_range"] for r in ("B", "T")}
add(f"- Responses are where the environment shows: the mean completion-token change is {tok_pos['B']} the seed-pair "
    f"range for B and {tok_pos['T']} it for T, and "
    + ("both reruns have fewer length stops than any archived seed." if all(drift[r]["length_stops"]["rerun_outside_archived_range"]
                                                                          for r in ("B", "T"))
       else "the length stops are in the table."))
add("- Row agreement and identical text are higher than between archived seeds because each rerun shares its sampling "
    "seed with the archived slice it is compared with; they mix that seed coupling with environment drift.")
add("- T's rerun was served from the protocol clone's 18-file snapshot, while the archived T used a content-bound 10-file "
    "view of the same weights. Prompt tokens agree on every row, but T's comparison also contains that view difference.")
add("- The confirmation contrasts with local-rerun B and T are in §7.11.")
add("")
ed = envd[envd["comparison"].str.contains("existing_local_rerun")]
ed = {r: ed[ed["comparison"].str.startswith(f"archive {r} ")].set_index("benchmark_id") for r in ("B", "T")}
spair = {r: envd[envd["comparison"].str.startswith(f"archive {r} seeds ")].groupby("benchmark_id")
         ["official_delta_other_minus_reference"].apply(lambda x: x.abs().max()) for r in ("B", "T")}
exceed = {r: [BNAME.get(b, b) for _, b in ORDER
              if abs(ed[r].loc[b, "official_delta_other_minus_reference"]) > spair[r][b]] for r in ("B", "T")}
add("Per benchmark, seed 42 (`tables/environment_drift.csv`):")
add("")
add(table(["category", "benchmark", "B archive", "B rerun", "B difference", "B largest seed-pair difference",
           "B row agreement (%)", "T archive", "T rerun", "T difference", "T largest seed-pair difference",
           "T row agreement (%)"],
          [[cat, BNAME.get(b, b)] + [c for r in ("B", "T") for c in (
              num(ed[r].loc[b, "official_reference"]), num(ed[r].loc[b, "official_other"]),
              sgn(ed[r].loc[b, "official_delta_other_minus_reference"]), num(spair[r][b]),
              num(100 * ed[r].loc[b, "row_score_agreement_rate"], 1) if pd.notna(ed[r].loc[b, "row_score_agreement_rate"]) else "–")]
           for cat, b in ORDER]))
add("")
add("Row agreement is shown only where row scores exist in both runs. The largest seed-pair difference is the largest "
    "absolute difference between two archived seeds of the same model on that benchmark.")
add(f"- The rerun difference is larger than the largest archived seed-pair difference on {len(exceed['B'])} of "
    f"{len(ORDER)} benchmarks for B ({', '.join(exceed['B']) or 'none'}) and {len(exceed['T'])} for T "
    f"({', '.join(exceed['T']) or 'none'}).")
add("")
ai = json.loads((AU / "archive_ingest.json").read_text())
add("### 4.4 Archive ingestion (`audit/archive_ingest.json`)")
add("")
add(f"Status `{ai['status']}`: {ai['rows']:,} archived rows in {len(ai['slices'])} model × seed × benchmark slices, "
    f"{len(ai['problems'])} problems. Checks run:")
for x in ai["checks"]:
    add(f"- {x[0].upper() + x[1:].rstrip('.')}.")
add("")

# ---------------------------------------------------------------- stage B
add("## 5. Stage B: TRACE and VERO against the base (archived runs, full split, seeds 42–44)")
add("")
add("### 5.1 Official scores (`stage_b/official_scores.all.csv`)")
add("")
add(scores_table(sb_scores, ["B", "T", "V@P0"]))
add("")
add("### 5.2 Contrasts (`stage_b/differences.all.csv`; categories exploratory, unadjusted)")
add("")
add(contrast_table(sb_diff, ["T-B", "V@P0-B", "T-V@P0"]))
add("")
add("### 5.3 Per-benchmark scores and contrasts (`stage_b/official_scores.all.csv`, `stage_b/differences.all.csv`; "
    "exploratory, unadjusted)")
add("")
sbb = sb_diff[sb_diff["level"] == "benchmark"].set_index(["comparison", "scope"])
add(table(["category", "benchmark", "B", "T", "V@P0", "T − B [95% CI]", "V@P0 − B [95% CI]", "T − V@P0 [95% CI]"],
          [row + [est_ci(sbb.loc[(c, b)]) for c in ("T-B", "V@P0-B", "T-V@P0")]
           for row, (_, b) in zip(bench_scores(sb_scores, ["B", "T", "V@P0"]), ORDER)]))
add("")
add("### 5.4 Row-level behaviour (21 binary-eligible benchmarks, 27,617 rows per seed; mean over seeds, rows pooled)")
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
    f"{num(olap['agreement_pct'], 1)}% {ci(olap['agreement_pct_ci_low'], olap['agreement_pct_ci_high'], 1, False)}.")
add(f"- The hindsight \"either parent correct\" rate is {num(olap['hindsight_either_parent_correct_pct'], 1)}% "
    f"{ci(olap['hindsight_either_parent_correct_pct_ci_low'], olap['hindsight_either_parent_correct_pct_ci_high'], 1, False)}. "
    f"It uses two responses and correctness labels, so it is neither a router nor a bound on a merge.")
add("")
add("Persistence of outcome sets across seeds (`stage_b/persistence.csv`; defined on one seed, checked on another, all "
    "ordered seed pairs):")
add("")
names = {"trace_only": "T-only (T=1, V@P0=0)", "vero_only": "V-only (T=0, V@P0=1)", "trace_correction": "T corrections (B=0, T=1)",
         "vero_correction": "V corrections (B=0, V@P0=1)", "trace_regression": "T regressions (B=1, T=0)",
         "vero_regression": "V regressions (B=1, V@P0=0)"}
def holds(r, c):
    return num(r[c], 1) if c in r.index and pd.notna(r[c]) else "–"


add(table(["set", "persists (%)", "label-swap null (%)", "excess (pp) [95% CI]", "T's label recurs (%)",
           "V@P0's label recurs (%)", "B's label recurs (%)"],
          [[names[k], num(r["persistence_pct"], 1), num(r["null_persistence_pct"], 1),
            f"{sgn(r['persistence_excess_pp'], 1)} {ci(r['persistence_excess_pp_ci_low'], r['persistence_excess_pp_ci_high'], 1)}",
            holds(r, "T_holds_pct"), holds(r, "V@P0_holds_pct"), holds(r, "B_holds_pct")]
           for k, r in pers.iterrows() if k in names]))
add("")
add(f"A set persists on the other seed when every label of its pattern recurs there; the last three columns show how "
    f"often each model's own label recurs. On T-only rows, T's success recurs {holds(pers.loc['trace_only'], 'T_holds_pct')}% "
    f"of the time and V@P0's failure {holds(pers.loc['trace_only'], 'V@P0_holds_pct')}%. On V-only rows, V@P0's success "
    f"recurs {holds(pers.loc['vero_only'], 'V@P0_holds_pct')}% and T's failure {holds(pers.loc['vero_only'], 'T_holds_pct')}%.")
add("")
add("### 5.5 Macro per seed (`stage_b/seed_scores.all.csv`; unweighted mean of the 24 benchmarks)")
add("")
psb = sb_seed.groupby(["model_label", "seed"])["official_score"].mean().unstack()
add(table(["model", "seed 42", "seed 43", "seed 44", "mean", "range"],
          [[m] + [num(psb.loc[m, s]) for s in (42, 43, 44)] + [num(psb.loc[m].mean()), num(psb.loc[m].max() - psb.loc[m].min())]
           for m in ("B", "T", "V@P0")]))
add("")
add("### 5.6 The same comparisons on each partition (`stage_b/official_scores.*.csv`, `stage_b/differences.*.csv`)")
add("")
PARTS = (("all", "full split", split["rows"]), ("development", "development partition", split["development_rows"]),
         ("confirmation", "confirmation partition", split["confirmation_rows"]))
prow = [[lab] + [est_ci(sb_part[p][0][sb_part[p][0]["level"] == "overall"].set_index("model_label").loc[lab], signed=False)
                 for p, _, _ in PARTS] for lab in ("B", "T", "V@P0")]
prow += [[comp_name(c)] + [f"{est_ci(sb_part[p][1][sb_part[p][1]['level'] == 'overall'].set_index('comparison').loc[c])}"
                           for p, _, _ in PARTS] for c in ("T-B", "V@P0-B", "T-V@P0")]
add(table(["model or contrast (macro)", *[f"{n} ({r:,} rows)" for _, n, r in PARTS]], prow))
add("")
add("Category contrasts on the confirmation partition (`stage_b/differences.confirmation.csv`):")
add("")
add(contrast_table(sb_part["confirmation"][1], ["T-B", "V@P0-B", "T-V@P0"]))
add("")
add("Category contrasts on the development partition (`stage_b/differences.development.csv`):")
add("")
add(contrast_table(sb_part["development"][1], ["T-B", "V@P0-B", "T-V@P0"]))
add("")
add("### 5.7 Corrections and regressions per benchmark (`stage_b/corrections_regressions.csv`; mean over seeds; % of rows)")
add("")
crb = sb_cr[(sb_cr["level"] == "benchmark") & (sb_cr["seed_scope"] == "mean_over_seeds")].set_index(["model_label", "scope"])
add(table(["category", "benchmark", "T corrections", "T regressions", "T net [95% CI]", "V@P0 corrections",
           "V@P0 regressions", "V@P0 net [95% CI]"],
          [[cat, BNAME.get(b, b)] + [c for m in ("T", "V@P0") for c in (
              num(crb.loc[(m, b), "correction_rate_pct"], 1), num(crb.loc[(m, b), "regression_rate_pct"], 1),
              f"{sgn(crb.loc[(m, b), 'net_delta_pp'])} {ci(crb.loc[(m, b), 'net_delta_pp_ci_low'], crb.loc[(m, b), 'net_delta_pp_ci_high'])}")]
           for cat, b in ORDER if b in binary]))
add("")
add("### 5.8 Persistence of the parents' unique successes by category (`stage_b/persistence.csv`; all ordered seed pairs)")
add("")
pcat = sb_pers[(sb_pers["level"] == "category") & (sb_pers["seed_scope"] == "pooled_over_ordered_seed_pairs")].set_index(["set_name", "scope"])
pscopes = [s for c in CATS for s in pcat.loc["trace_only"].index if s.startswith(c)]
add(table(["category (binary-eligible benchmarks)", "T-only: persists / null (%)", "T-only: excess (pp) [95% CI]",
           "V-only: persists / null (%)", "V-only: excess (pp) [95% CI]"],
          [[s] + [c for k in ("trace_only", "vero_only") for c in (
              f"{num(pcat.loc[(k, s), 'persistence_pct'], 1)} / {num(pcat.loc[(k, s), 'null_persistence_pct'], 1)}",
              f"{sgn(pcat.loc[(k, s), 'persistence_excess_pp'], 1)} "
              f"{ci(pcat.loc[(k, s), 'persistence_excess_pp_ci_low'], pcat.loc[(k, s), 'persistence_excess_pp_ci_high'], 1)}")]
           for s in pscopes]))
add("")
add("Persistence of corrections and regressions by category: excess over the label-swap null, pp [95% CI] "
    "(persists / null in %):")
add("")
CRS = (("trace_correction", "T corrections"), ("vero_correction", "V corrections"), ("trace_regression", "T regressions"),
       ("vero_regression", "V regressions"))
add(table(["category (binary-eligible benchmarks)", *[n for _, n in CRS]],
          [[s] + [f"{sgn(pcat.loc[(k, s), 'persistence_excess_pp'], 1)} "
                  f"{ci(pcat.loc[(k, s), 'persistence_excess_pp_ci_low'], pcat.loc[(k, s), 'persistence_excess_pp_ci_high'], 1)} "
                  f"({num(pcat.loc[(k, s), 'persistence_pct'], 1)} / {num(pcat.loc[(k, s), 'null_persistence_pct'], 1)})"
                  for k, _ in CRS] for s in pscopes]))
add("")
pben = sb_pers[(sb_pers["level"] == "benchmark") & (sb_pers["seed_scope"] == "pooled_over_ordered_seed_pairs")].set_index(["set_name", "scope"])
for k, who in (("trace_regression", "TRACE's"), ("vero_regression", "VERO's")):
    x = pben.loc[k]
    above = x[x["persistence_excess_pp_ci_low"] > 0]
    top = above.sort_values("persistence_excess_pp", ascending=False).head(3)
    add(f"- {who} regressions persist beyond the null on {len(above)} of {len(x)} benchmarks (interval above 0)"
        + (f"; the largest point estimates are on "
           f"{', '.join(f'{BNAME.get(b, b)} ({sgn(r_.persistence_excess_pp, 1)}, about {num(r_.n_defined, 0)} rows per seed pair)' for b, r_ in top.iterrows())}."
           if len(above) else "."))
add("")
add("### 5.9 Row-level behaviour by category (`stage_b/parent_overlap.csv`, `stage_b/corrections_regressions.csv`; mean "
    "over seeds, rows pooled)")
add("")
oc = sb_overlap[(sb_overlap["level"] == "category") & (sb_overlap["seed_scope"] == "mean_over_seeds")
                & (sb_overlap["aggregation"] == "pooled_rows")].set_index("scope")
oscopes = [x for c in CATS for x in oc.index if x.startswith(c)]


def pct_ci(r, c):
    return f"{num(r[c], 1)} {ci(r[c + '_ci_low'], r[c + '_ci_high'], 1, False)}"


add("Parent overlap (% of rows):")
add("")
add(table(["category (binary-eligible benchmarks)", "rows per seed", "T-only [95% CI]", "V-only [95% CI]", "both correct",
           "both wrong", "agreement [95% CI]", "hindsight either-correct [95% CI]"],
          [[x, f"{int(round(oc.loc[x, 'n_units'])):,}", pct_ci(oc.loc[x], "trace_only_pct"), pct_ci(oc.loc[x], "vero_only_pct"),
            num(oc.loc[x, "both_correct_pct"], 1), num(oc.loc[x, "both_wrong_pct"], 1), pct_ci(oc.loc[x], "agreement_pct"),
            pct_ci(oc.loc[x], "hindsight_either_parent_correct_pct")] for x in oscopes]
          + [["**all 21**", f"{int(round(olap['n_units'])):,}", pct_ci(olap, "trace_only_pct"), pct_ci(olap, "vero_only_pct"),
              num(olap["both_correct_pct"], 1), num(olap["both_wrong_pct"], 1), pct_ci(olap, "agreement_pct"),
              pct_ci(olap, "hindsight_either_parent_correct_pct")]]))
add("")
crc = sb_cr[(sb_cr["level"] == "category") & (sb_cr["seed_scope"] == "mean_over_seeds")
            & (sb_cr["aggregation"] == "pooled_rows")].set_index(["model_label", "scope"])
add("Corrections and regressions against B (% of rows; conditional rates in % of B's errors or successes):")
add("")
add(table(["category", *[f"{m} {h}" for m in ("T", "V@P0") for h in ("corrections", "regressions", "net [95% CI]",
                                                                     "corrections / B errors", "regressions / B successes")]],
          [[x] + [c for m in ("T", "V@P0") for c in (
              num(crc.loc[(m, x), "correction_rate_pct"], 1), num(crc.loc[(m, x), "regression_rate_pct"], 1),
              f"{sgn(crc.loc[(m, x), 'net_delta_pp'])} {ci(crc.loc[(m, x), 'net_delta_pp_ci_low'], crc.loc[(m, x), 'net_delta_pp_ci_high'])}",
              num(crc.loc[(m, x), "correction_given_base_error_pct"], 1), num(crc.loc[(m, x), "regression_given_base_success_pct"], 1))]
           for x in [y for c in CATS for y in crc.loc["T"].index if y.startswith(c)]]))
add("")

# ---------------------------------------------------------------- screen
add("## 6. Development screen (seed 42, 2,910 rows; descriptive)")
add("")
add("### 6.1 Scores under P1 (`screen/scores.P1.csv`)")
add("")
add(scores_table(sc1, P1))
add("")
add("### 6.2 Contrasts under P1 (`screen/differences.P1.csv`)")
add("")
add(contrast_table(sd1, ["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1", "CV050@P1-B", "CT050@P1-B",
                         "T-B", "V@P1-B", "V@P1-V@P0"]))
add("")
add("### 6.3 Scores under P2, VERO's interface (`screen/scores.P2.csv`)")
add("")
add(scores_table(sc2, P2))
add("")
add("### 6.4 Contrasts under P2 (`screen/differences.P2.csv`)")
add("")
add(contrast_table(sd2, ["M050@P2-V@P0", "M050@P2-CV050@P2", "M050@P2-CT050@P2", "M050@P2-T@P2", "T@P2-B@P2", "V@P0-B@P2"]))
add("")
add("### 6.5 Capture and damage, same-seed parents (`screen/merge_capture_damage.csv`)")
add("")
G = [("trace_only", "TRACE-only capture"), ("vero_only", "VERO-only capture"), ("shared_success", "shared-success damage"),
     ("joint_failure", "joint-failure rescue")]
sc_ = s_cap[(s_cap["level"] == "overall")].set_index(["merge_label", "group"])
add(table(["merge", *[f"{n} (n = {int(round(sc_.loc[('M050@P1', g), 'n_group'])):,})" for g, n in G]],
          [[SHORT[m]] + [f"{num(sc_.loc[(m, g), 'conditional_rate_pct'], 1)} "
                         f"{ci(sc_.loc[(m, g), 'conditional_rate_pct_ci_low'], sc_.loc[(m, g), 'conditional_rate_pct_ci_high'], 1, False)}"
                         for g, _ in G] for m in ("M050@P1", "CV050@P1", "CT050@P1")]))
add("")
add("### 6.6 Answer format (`screen/answer_format_sensitivity.csv`; % of rows)")
add("")
sf = s_fmt.set_index(["benchmark_id", "model_label"])
add(table(["model", "PuzzleVQA accuracy", "PuzzleVQA: ends in \\( X \\)", "PuzzleVQA: scored 0, final letter correct",
           "VisualPuzzles accuracy", "VisualPuzzles: ends in \\( X \\)", "VisualPuzzles: scored 0, final letter correct"],
          [[SHORT[m]] + [num(sf.loc[(b, m), c], 1) for b in ("puzzlevqa", "visualpuzzles")
                         for c in ("official_accuracy_pct", "latex_letter_ending_pct", "official_zero_but_final_letter_correct_pct")]
           for m in [*P1, *[x for x in P2 if x != "V@P0"]] if (("puzzlevqa", m) in sf.index)]))
add("")
add("### 6.7 Per-benchmark scores under P1 (`screen/scores.P1.csv`)")
add("")
add(table(["category", "benchmark", *[SHORT[m] for m in P1]], bench_scores(sc1, P1)))
add("")
add("### 6.8 Per-benchmark scores under P2 (`screen/scores.P2.csv`)")
add("")
add(table(["category", "benchmark", *[SHORT[m] for m in P2]], bench_scores(sc2, P2)))
add("")
add("### 6.9 Per-benchmark merge contrasts under P1 (`screen/differences.P1.csv`; one seed, wide intervals)")
add("")
add(bench_contrasts(sd1, ["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1", "V@P1-V@P0"]))
add("")
add("### 6.10 Per-benchmark merge contrasts under P2 (`screen/differences.P2.csv`; one seed, wide intervals)")
add("")
add(bench_contrasts(sd2, ["M050@P2-V@P0", "M050@P2-CV050@P2", "M050@P2-CT050@P2", "M050@P2-T@P2"]))
add("")
add("### 6.11 Length and format on the development screen (`length_format_judge_audit.csv`; seed 42)")
add("")
add("The screen runs under P1 and P2, with archived V@P0 seed 42 (full split, not the screen rows) for reference. Rates "
    "are row-weighted. These are observable correlates of the P2 results in §6, not causes.")
add("")
sa = audit[audit["run_id"].astype(str).str.contains("development_screen")]
va = audit[(audit["source"] == "archive") & (audit["model_label"] == "V@P0") & (audit["seed"] == 42)]
SCR = [("V@P1", "V@P1"), ("M050@P1", "M050"), ("CV050@P1", "CV050"), ("CT050@P1", "CT050"), ("B@P2", "B@P2"),
       ("T@P2", "T@P2"), ("M050@P2", "M050@P2"), ("CV050@P2", "CV050@P2"), ("CT050@P2", "CT050@P2")]


def fmt_row(lab, g):
    n = g["n_rows"].sum()
    w = lambda c: (g[c] * g["n_rows"]).sum() / n * 100  # noqa: E731
    med = Decimal(str(g["completion_tokens_median"].median())).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    mean_tok = (g["completion_tokens_mean"] * g["completion_tokens_n_known"]).sum() / g["completion_tokens_n_known"].sum()
    return [lab, f"{int(n):,}", int(g["finish_reason_length"].sum()), num(mean_tok, 1), str(med),
            num(w("marker_think_close_rate"), 1), num(w("marker_answer_tag_rate"), 1), num(w("marker_boxed_rate"), 1)]


add(table(["model", "responses", "length stops", "mean completion tokens", "median of per-benchmark median completion tokens",
           "with </think> (%)", "with <answer> (%)", "with \\boxed{} (%)"],
          [fmt_row(short, sa[sa["model_label"] == lab]) for lab, short in SCR] + [fmt_row("V@P0 (archive, full split)", va)]))
add("")
SP2 = ["B@P2", "T@P2", "M050@P2", "CV050@P2", "CT050@P2"]
sb_ = sa[sa["model_label"].isin(SP2)].set_index(["benchmark_id", "model_label"])
vb_ = va.set_index("benchmark_id")
for col, title, d in (("completion_tokens_median", "Median completion tokens per benchmark under P2", 0),
                      ("marker_answer_tag_rate", "Responses with an `<answer>` tag under P2 (%)", 1),
                      ("marker_boxed_rate", "Responses containing `\\boxed{}` under P2 (%)", 1)):
    f = (lambda x: num(x, d)) if col == "completion_tokens_median" else (lambda x: num(100 * x, d))
    add(f"{title}:")
    add("")
    add(table(["category", "benchmark", *SP2, "V@P0 (archive)"],
              [[cat, BNAME.get(b, b)] + [f(sb_.loc[(b, m), col]) for m in SP2] + [f(vb_.loc[b, col])] for cat, b in ORDER]))
    add("")

# ---------------------------------------------------------------- confirmation
add("## 7. Confirmation (P1, 26,228 rows, seeds 42–44)")
add("")
add(f"The confirmed set, the primary family and every input hash were locked at {lock['locked_at']} "
    f"(`selection_lock.json`) before any confirmation run had scores.")
add("")
add("### 7.1 Official scores (`confirmation/scores.csv`)")
add("")
add(scores_table(c_scores, P1))
add("")
add("### 7.2 Macro per seed (`confirmation/seed_scores.csv`; unweighted mean of the 24 benchmarks)")
add("")
ps = c_seed.groupby(["model_label", "seed"])["official_score"].mean().unstack()
add(table(["model", "seed 42", "seed 43", "seed 44", "mean"],
          [[SHORT[m]] + [num(ps.loc[m, s]) for s in (42, 43, 44)] + [num(ps.loc[m].mean())] for m in P1]))
add("")
add("### 7.3 Per-benchmark scores, mean over seeds (`confirmation/scores.csv`)")
add("")
add(table(["category", "benchmark", *[SHORT[m] for m in P1]], bench_scores(c_scores, P1)))
add("")
add("### 7.4 Per-benchmark scores per seed (`confirmation/seed_scores.csv`)")
add("")
cs = c_seed.pivot_table(index="benchmark_id", columns=["model_label", "seed"], values="official_score")
for labs, title in ((["B", "T", "V@P0"], "Archived models on the confirmation partition"),
                    (["V@P1", "M050@P1", "CV050@P1", "CT050@P1"], "New runs")):
    add(f"{title}:")
    add("")
    add(table(["benchmark", *[f"{SHORT[m]} s{s}" for m in labs for s in (42, 43, 44)]],
              [[BNAME.get(b, b)] + [num(cs.loc[b, (m, s)]) for m in labs for s in (42, 43, 44)] for _, b in ORDER]
              + [["**macro**"] + [num(ps.loc[m, s]) for m in labs for s in (42, 43, 44)]]))
    add("")
add("### 7.5 Primary family (Holm-adjusted) and secondary contrasts (`confirmation/differences.csv`)")
add("")
add(contrast_table(c_diff, FAMILY, holm=True))
add("")
add(contrast_table(c_diff, SECOND))
add("")
add("### 7.6 Per-benchmark contrasts (`confirmation/differences.csv`; exploratory, unadjusted)")
add("")
add(bench_contrasts(c_diff, ["M050@P1-T", "M050@P1-V@P1", "M050@P1-CV050@P1", "M050@P1-CT050@P1", "V@P1-V@P0"]))
add("")
C2 = ["T-B", "V@P1-B", "M050@P1-B", "CV050@P1-B", "CT050@P1-B"]
sbc = sb_part["confirmation"][1]
sbc = sbc[sbc["level"] == "benchmark"].set_index(["comparison", "scope"])
add(table(["category", "benchmark", *[comp_name(c) for c in C2], "V@P0 − B (Stage B tables)"],
          [[cat, BNAME.get(b, b)] + [est_ci(cdb.loc[(c, b)]) for c in C2] + [est_ci(sbc.loc[("V@P0-B", b)])]
           for cat, b in ORDER]))
add("")
same_draws = all(abs(sbc.loc[("T-B", b), k] - cdb.loc[("T-B", b), k]) < 1e-9 for _, b in ORDER for k in ("estimate", "ci_low", "ci_high"))
add("The V@P0 − B column comes from the Stage B tables on the same partition and seeds "
    "(`stage_b/differences.confirmation.csv`)" + (", whose bootstrap uses the same resampling units and draws: its "
                                                  "T − B column equals the one above exactly." if same_draws else "."))
add("")
add("### 7.7 Capture, retention and damage (`confirmation/merge_capture_damage.base_reference.csv`)")
add("")
add("Setup:")
add("- 21 binary-eligible benchmarks, 22,078 rows per seed.")
add("- Groups are defined by T and V@P1 on the confirmation partition.")
add("- Rates are pooled over conditions, and group sizes are mean rows per condition.")
add("- Intervals are for each rate on its own, not for differences.")
add("- The B (reference) rows score the base model as if it were a merge. They were added post hoc.")
add("")
add("- In the stable-other-seeds variant, rows whose T/V@P1 pattern differs between the two other seeds form a fifth "
    "group; its column is the merge's accuracy on those rows, and the five groups partition the rows.")
add("")
G5 = G + [("unstable_parent_pattern", "unstable parent pattern: merge accuracy")]
for bset, title in (("all_binary", "all 21 benchmarks"), ("without_puzzlevqa_visualpuzzles", "without PuzzleVQA and VisualPuzzles")):
    add(f"**{title}**")
    add("")
    rows = []
    for v in ("independent_repeat", "contemporaneous", "stable_other_seeds"):
        x = cap[(cap["benchmark_set"] == bset) & (cap["variant"] == v)].set_index(["merge_label", "group"])
        sizes = " / ".join(f"{int(round(x.loc[('M050@P1', g), 'n_group'])):,}" for g, _ in G5 if ("M050@P1", g) in x.index)
        rows.append([f"**{v.replace('_', ' ')}** ({sizes})", "", "", "", "", ""])
        for m in ("M050@P1", "CV050@P1", "CT050@P1", "B (reference)"):
            rows.append([SHORT[m]] + [(f"{num(x.loc[(m, g), 'conditional_rate_pct'], 1)} "
                                       f"{ci(x.loc[(m, g), 'conditional_rate_pct_ci_low'], x.loc[(m, g), 'conditional_rate_pct_ci_high'], 1, False)}")
                                      if (m, g) in x.index else "–" for g, _ in G5])
    add(table(["model (variant, group sizes)", *[n for _, n in G5]], rows))
    add("")
add("**Where pooled binary accuracy comes from** (independent repeat). Each group's contribution is its share of rows "
    "times the model's rate of being correct on it, in percentage points of all rows; before rounding, the four "
    "contributions add up to the model's pooled accuracy on these benchmarks, which pools rows and is not the "
    "24-benchmark macro.")
add("")
DG = [("trace_only", "TRACE-only kept"), ("vero_only", "VERO-only kept"), ("shared_success", "shared successes kept"),
      ("joint_failure", "joint failures rescued")]
for bset, title in (("all_binary", "all 21 benchmarks"), ("without_puzzlevqa_visualpuzzles", "without PuzzleVQA and VisualPuzzles")):
    x = cap[(cap["benchmark_set"] == bset) & (cap["variant"] == "independent_repeat")].set_index(["merge_label", "group"])
    m0 = x.loc["M050@P1"]
    add(f"*{title}* (group shares: {' / '.join(num(m0.loc[g, 'group_share_pct'], 2) + '%' for g, _ in DG)}):")
    add("")
    drow = []
    for m in ("M050@P1", "CV050@P1", "CT050@P1", "B (reference)"):
        y = x.loc[m]
        a0 = y.iloc[0]
        assert abs(y.loc[[g for g, _ in DG], "merge_correct_contribution_pp"].sum() - a0["acc_merge_pct"]) < 1e-6
        drow.append([SHORT[m]] + [num(y.loc[g, "merge_correct_contribution_pp"]) for g, _ in DG]
                    + [f"{num(a0['acc_merge_pct'])} {ci(a0['acc_merge_pct_ci_low'], a0['acc_merge_pct_ci_high'], 2, False)}"])
    p0 = m0.iloc[0]
    drow.append(["T (same rows)", "", "", "", "", num(p0["acc_trace_pct"])])
    drow.append(["V@P1 (same rows)", "", "", "", "", num(p0["acc_vero_pct"])])
    add(table(["model", *[n for _, n in DG], "pooled accuracy [95% CI]"], drow))
    add("")
    add("Differences, arithmetic on the rows above (no paired interval):")
    add("")
    add(table(["comparison", *[n for _, n in DG], "pooled accuracy"],
              [[f"M050 − {SHORT[o]}"] + [sgn(x.loc[("M050@P1", g), "merge_correct_contribution_pp"]
                                            - x.loc[(o, g), "merge_correct_contribution_pp"]) for g, _ in DG]
               + [sgn(x.loc["M050@P1"].iloc[0]["acc_merge_pct"] - x.loc[o].iloc[0]["acc_merge_pct"])]
               for o in ("CV050@P1", "CT050@P1", "B (reference)")]))
    add("")
add("### 7.8 Capture, retention and damage by category (`confirmation/merge_capture_damage.base_reference.csv`; "
    "independent repeat, descriptive)")
add("")
add("Same setup as §7.7, one table per outcome group. A category marked partial has fewer than four binary-eligible "
    "benchmarks.")
add("")
ccat = c_ref[(c_ref["level"] == "category") & (c_ref["seed_scope"] != "single_condition") & (c_ref["benchmark_set"] == "all_binary")
             & (c_ref["variant"] == "independent_repeat")].set_index(["group", "scope", "merge_label"])
CAPM = ("M050@P1", "CV050@P1", "CT050@P1", "B (reference)")
for gk, gname in G:
    sc_scopes = [s for c in CATS for s in ccat.loc[gk].index.get_level_values(0).unique() if s.startswith(c)]
    add(f"**{gname}**")
    add("")
    add(table(["category", "mean group size per condition", *[SHORT[m] for m in CAPM]],
              [[s, f"{int(round(ccat.loc[(gk, s, 'M050@P1'), 'n_group'])):,}"]
               + [f"{num(ccat.loc[(gk, s, m), 'conditional_rate_pct'], 1)} "
                  f"{ci(ccat.loc[(gk, s, m), 'conditional_rate_pct_ci_low'], ccat.loc[(gk, s, m), 'conditional_rate_pct_ci_high'], 1, False)}"
                  for m in CAPM] for s in sc_scopes]))
    add("")
add("**By benchmark** (`confirmation/merge_capture_damage.csv`; independent repeat; the base reference is not computed "
    "per benchmark):")
add("")
cbm = c_cap[(c_cap["level"] == "benchmark") & (c_cap["seed_scope"] == "pooled_over_conditions")
            & (c_cap["variant"] == "independent_repeat")].set_index(["group", "scope", "merge_label"])
M3 = ("M050@P1", "CV050@P1", "CT050@P1")
BB = [(cat, b) for cat, b in ORDER if b in binary]
for gk, gname in G:
    add(f"*{gname}*")
    add("")
    add(table(["category", "benchmark", "mean group size per condition", *[SHORT[m] for m in M3]],
              [[cat, BNAME.get(b, b), f"{int(round(cbm.loc[(gk, b, 'M050@P1'), 'n_group'])):,}"]
               + [f"{num(cbm.loc[(gk, b, m), 'conditional_rate_pct'], 1)} "
                  f"{ci(cbm.loc[(gk, b, m), 'conditional_rate_pct_ci_low'], cbm.loc[(gk, b, m), 'conditional_rate_pct_ci_high'], 1, False)}"
                  for m in M3] for cat, b in BB]))
    add("")
rate = lambda g, b, m: cbm.loc[(g, b, m), "conditional_rate_pct"]  # noqa: E731
n_low_dmg = sum(rate("shared_success", b, "M050@P1") < min(rate("shared_success", b, "CV050@P1"), rate("shared_success", b, "CT050@P1")) for _, b in BB)
n_v_ct = sum(rate("vero_only", b, "M050@P1") > rate("vero_only", b, "CT050@P1") for _, b in BB)
n_t_ct = sum(rate("trace_only", b, "M050@P1") < rate("trace_only", b, "CT050@P1") for _, b in BB)
n_low_resc = sum(rate("joint_failure", b, "M050@P1") < min(rate("joint_failure", b, "CV050@P1"), rate("joint_failure", b, "CT050@P1")) for _, b in BB)
add(f"Across the {len(BB)} benchmarks, M050 has the lowest shared-success damage of the three merges on {n_low_dmg}, keeps "
    f"more VERO-only successes than CT050 on {n_v_ct}, keeps fewer TRACE-only successes than CT050 on {n_t_ct}, and has the "
    f"lowest joint-failure rescue on {n_low_resc}.")
add("")
add("### 7.9 Answer format (`confirmation/answer_format_sensitivity.csv`; mean over seeds, % of rows)")
add("")
add(table(["model", "PuzzleVQA accuracy", "PuzzleVQA: ends in \\( X \\)", "PuzzleVQA: scored 0, final letter correct",
           "VisualPuzzles accuracy", "VisualPuzzles: ends in \\( X \\)", "VisualPuzzles: scored 0, final letter correct"],
          [[SHORT[m]] + [num(fmt.loc[(b, m), c], 1) for b in ("puzzlevqa", "visualpuzzles")
                         for c in ("official_accuracy_pct", "latex_letter_ending_pct", "official_zero_but_final_letter_correct_pct")]
           for m in P1]))
add("")
add("### 7.10 Length, finish reasons and extraction (`length_format_judge_audit.csv`)")
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
    lrows.append([SHORT[m], f"{int(n):,}", int(g["finish_reason_length"].sum()),
                  f"{int(g['length_stop_score_available'].sum())} / {int(g['length_stop_score_zero'].sum())}", str(med),
                  num(w("marker_think_close_rate"), 1) + "%", num(w("marker_boxed_rate"), 1) + "%",
                  num(w("marker_answer_is_rate"), 1) + "%", num(w("marker_any_final_rate"), 1) + "%",
                  num(w("extraction_status_not_resolved_rate"), 3) + "%", int(g["judge_incomplete_sentinel_rows"].sum())])
add(table(["model", "responses", "length stops", "length stops with a row score / scored 0",
           "median of per-benchmark median completion tokens", "with </think>", "with \\boxed{}", "with \"The answer is\"",
           "with any final-answer marker", "extraction not resolved", "judge \"incomplete\" sentinels"], lrows))
add("")
add("A judge \"incomplete\" sentinel is the judge's own statement that a response gives no answer. Length stops without a "
    "row score are on benchmarks whose rows are re-derived (§2.7).")
add("")
add("Median completion tokens per benchmark (mean of the three seeds' medians; archived models on the full split, new "
    "models on the confirmation partition):")
add("")
aub = au.groupby(["benchmark_id", "model_label"])
medb = aub["completion_tokens_median"].mean()
boxb = (au["marker_boxed_rate"] * au["n_rows"]).groupby([au["benchmark_id"], au["model_label"]]).sum() / aub["n_rows"].sum() * 100
add(table(["category", "benchmark", *[SHORT[m] for m in P1]],
          [[cat, BNAME.get(b, b)] + [num(medb[(b, m)], 0) for m in P1] for cat, b in ORDER]))
add("")
add("Responses containing `\\boxed{}`, per benchmark (% of rows, row-weighted over seeds):")
add("")
add(table(["category", "benchmark", *[SHORT[m] for m in P1]],
          [[cat, BNAME.get(b, b)] + [num(boxb[(b, m)], 1) for m in P1] for cat, b in ORDER]))
add("")
add("### 7.11 Sensitivity analyses")
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
sig = lambda r: "+" if r["ci_low"] > 1e-9 else "−" if r["ci_high"] < -1e-9 else "0"  # noqa: E731
at0 = lambda *rs: any(abs(x[k]) < 1e-9 for x in rs for k in ("ci_low", "ci_high"))  # noqa: E731
moved = [c for c in e1.index if any(abs(ec["archived_BT"].loc[(c, k), "estimate"] - ec["local_rerun_BT"].loc[(c, k), "estimate"]) > 1e-9
                                    for k in CATS)]
add(f"By category, seed 42 (only the comparisons that involve B or T change; the others are identical in both files):")
add("")
add(table(["comparison", "category", "archived B/T", "local-rerun B/T", "shift"],
          [[comp_name(c), k, est_ci(ec["archived_BT"].loc[(c, k)]), est_ci(ec["local_rerun_BT"].loc[(c, k)]),
            sgn(ec["local_rerun_BT"].loc[(c, k), "estimate"] - ec["archived_BT"].loc[(c, k), "estimate"])]
           for c in moved for k in CATS]))
add("")
eb = {k: c_env[k][c_env[k]["level"] == "benchmark"].set_index(["comparison", "scope"]) for k in ("archived_BT", "local_rerun_BT")}
flips = [f"{comp_name(c)} on {BNAME.get(b, b)}"
         + (" (a bound at exactly 0)" if at0(eb["archived_BT"].loc[(c, b)], eb["local_rerun_BT"].loc[(c, b)]) else "")
         for c in moved for _, b in ORDER if sig(eb["archived_BT"].loc[(c, b)]) != sig(eb["local_rerun_BT"].loc[(c, b)])]
cflips = env_cflips
add(f"- Category intervals that change side of 0 (a bound within 1e-9 of 0 counts as touching it): {len(cflips)} "
    f"({'; '.join(cflips) or 'none'}).")
add(f"- Benchmark intervals that change side of 0: {len(flips)} ({'; '.join(flips) or 'none'}).")
twc = c_twin[c_twin["level"] == "category"].set_index(["comparison", "scope"])
twb = c_twin[c_twin["level"] == "benchmark"].set_index(["comparison", "scope"])
twin_shift = max(abs(twc.loc[(c, k), "estimate"] - cdc.loc[(c, k), "estimate"]) for c in FAMILY + SECOND for k in CATS)
twin_flips = [f"{comp_name(c)} on {BNAME.get(b, b)}, {est_ci(cdb.loc[(c, b)])} → {est_ci(twb.loc[(c, b)])}"
              for c in FAMILY + SECOND for _, b in ORDER if sig(cdb.loc[(c, b)]) != sig(twb.loc[(c, b)])]
add(f"- Without the development twins, category contrasts move by at most {num(twin_shift)} pp; benchmark intervals that "
    f"change side of 0: {len(twin_flips)} ({'; '.join(twin_flips) or 'none'}).")
add("")

# ---------------------------------------------------------------- technical
add("## 8. Technical record")
add("")
add("### 8.1 Judge health (`judge_extraction_agreement.csv`)")
add("")
add("For each run's MathVerse slice: the share of rows whose judge-extracted answer equals the response's own final "
    "`\\boxed{}` answer. The floor is 30% (DESIGN.md A8).")
add("")
add(table(["run", "tier", "seed", "quarantined", "rows with a boxed answer", "extracted = boxed (%)", "below floor"],
          [[label_of[r.view_slug], r.subset.replace("_", " "), r.seed, "yes" if r.quarantined else "", r.rows_with_boxed,
            num(r.boxed_equals_extracted_pct, 1), "yes" if r.below_floor else ""]
           for r in judge.sort_values(["subset", "view_slug", "seed", "quarantined"]).itertuples()]))
add("")
add("### 8.2 The A8 repair: M050 confirmation seed 44 before and after rescoring")
add("")
add("The fault's signature (DESIGN.md A9): the phrase \"They give the solutions to the equation\", from one of the judge "
    "prompt's few-shot examples, appears in 133 of the 630 quarantined MathVerse extraction outputs and in 0 of 630 after "
    "the rescore; 297 against 540 are short single-line answers. The cause inside the serving stack was not isolated.")
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
add("- Before the repair, M050 − T was −0.29: the point estimate in the superseded tables, as quoted in "
    "`analysis_report.md` and `decision_memo.md`.")
add(f"- After it, M050 − T is {est_ci(cd.loc['M050@P1-T'])}.")
add("- The old confirmation tables are superseded. `analysis_report.md` §4 and `decision_memo.md` quote only this "
    "pre-repair estimate and the Visual Math shortfall it showed, to describe the fault.")
add("")

# ---------------------------------------------------------------- files
add("## 9. Where the data lives (`experiments/trace_merge/results/`)")
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
    ["`tables/evaluation_status.json`, `tables/environment_drift.csv`", "run completeness; local reruns against the archive per benchmark"],
    ["`audit/compatibility.json`, `audit/parent_update_statistics.json`", "parent compatibility; update norms and cosines"],
    ["`audit/protocol_comparison.json`, `audit/missingness.csv`", "prompt conformance, settings, environment drift; row completeness"],
    ["`audit/metric_reproduction.json`", "metric adapters against the archived aggregates"],
    ["`audit/`", "asset, archive, metric-reproduction, protocol and environment audits"],
    ["`manifests/`", "model roles, inference views, merges, metric registry, input manifest"],
    ["`splits/`", "partition and bootstrap-unit manifests"],
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
