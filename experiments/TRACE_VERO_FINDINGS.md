# TRACE–VERO: findings of the merging study and the training-free follow-up

Two training-free studies of Qwen2.5-VL-7B-Instruct (B), the released TRACE-7B (T) and the released VERO-7B (V): the
analysis-first merging study ([`trace_merge/`](trace_merge/README.md), complete 2026-11-03, PR maveryn/trace#14) and the
follow-up on component transfer, co-adaptation and evaluation reliability
([`trace_module_followup/`](trace_module_followup/README.md), complete 2026-11-10). Nothing was trained or uploaded. The
follow-up is exploratory throughout: its main analysis reuses the confirmation partition that the merging study had
already examined. Intervals reflect evaluation uncertainty (items, decoding seeds) for these fixed checkpoints, not
training-run variation; a non-significant difference is not equivalence, and neither study set a noninferiority margin.

## 1. Status (checked 2026-11-11)

- **All runs have finished.** No TRACE–VERO job remains in Slurm.
- **Merging study.** 25 new generation runs (4 smoke, 9 development screen, 12 confirmation), all complete and fully
  scored ([`evaluation_status.json`](trace_merge/results/tables/evaluation_status.json)); B, T and V@P0 are archived runs.
- **Follow-up.** [`evaluation.json`](trace_module_followup/results/manifests/evaluation.json) lists 99 runs, all
  `"complete": true` and each scoring all 24 benchmarks: 33 confirmation, 47 development screen, 19 smoke, over 19 cells
  (B, T, V, TF, TL, VF, VL at P1 and P2; TR0, TR1, TR2 at P1; VNM at P1 and P2). 63 runs finished on their first
  attempt, 36 needed 2–6. The per-card filler (the machine-specific runner that placed runs on already-held
  allocations) ran 9 waves; every generation-only task (exit 6, a planned handoff) had its judge-only half finish with
  exit 0, and failed attempts were re-run with finished rows kept.
- **Tests, bundles, branch.** Merging study 323 tests passed; follow-up 139 passed (30 need the private overlay clone and
  skip without it). Both `results/` bundles are byte-identical to their work-root outputs, unchanged since the last
  export. The merging study is on `experiments/trace-vero-merge` of the fork (PR maveryn/trace#14, base `rlvr`); the
  follow-up and this file are committed on top of it in the same branch.

## 2. Bottom line

Labels and symbols (V@P0/P1/P2, M050, CV050, CT050, TF/TL/VF/VL, V-NM, F0–F3, S, g_F, g_L, I) are defined in §3.

1. **TRACE and VERO gain similar amounts over B with their own interfaces.** T − B = +4.06 [+3.66, +4.45], V@P0 − B =
   +4.30 [+3.78, +4.82]; T − V@P0 = −0.24 [−0.75, +0.28] is not resolved (Stage B, archived runs;
   [FINDINGS.md](trace_merge/results/reports/FINDINGS.md) §1).
2. **The 50:50 merge M050 is not shown to beat TRACE.** M050 − T = +0.23 [−0.19, +0.65], Holm p = 0.29, while M050
   beats V@P1 (+3.07), CV050 (+1.29) and CT050 (+1.88), each Holm p < 0.001 (confirmatory family; FINDINGS.md §7.5).
3. **Most of VERO's loss under the backbone interface is answer formatting.** Normalization shrinks V@P1 − V@P0 from
   −2.94 [−3.33, −2.56] to −0.83 [−1.21, −0.46] (72%). TableVQABench's −11.41 points (11.41 / 24 = 0.48 percentage
   points (pp) of the macro) are most of the rest: under P1 VERO hedges WTQ answers with `||` lists and gives more
   wrong vtabfact True/False labels ([followup_findings.md](trace_module_followup/results/reports/followup_findings.md)
   §2; [`tablevqa_diagnosis.md`](trace_module_followup/results/audit/tablevqa_diagnosis.md) §5).
4. **Both parents' gains transfer into B through the language component, with no measurable co-adaptation.** TRACE:
   g_L = +3.47 [+3.04, +3.90], g_F = +0.41 [+0.12, +0.69], I = +0.08 [−0.32, +0.48]. VERO: g_L = +1.28 [+0.77, +1.80]
   official and +3.04 [+2.52, +3.55] normalized, g_F and I not resolved; after normalization the two language gains
   differ by +0.33 [−0.18, +0.83] (followup_findings.md §3).
5. **VERO's interface dependence sits in its language component.** P2 − P1 is +3.19 [+2.16, +4.19] for V and +3.35
   [+2.30, +4.43] for VL, but −2.88 for B, −3.40 for T and −2.67 for VF (development screen, three seeds;
   followup_findings.md §4).
6. **Within TRACE's decoder, layers 9–18 matter most.** Restoring them to B removes +1.81 [+1.42, +2.19] of TRACE's
   +3.95; layers 0–8 remove +0.58 [+0.25, +0.89], layers 19–27 +0.21 [−0.11, +0.53] (F2c; followup_findings.md §5).
   The reversions are not additive (sum 2.60) and the groups are unequal (9/10/9 blocks); a large reversion effect says
   the group is needed for the measured gain under this interface and scorer, not where a skill lives.
7. **VERO's 4.4 times larger update is not mostly superfluous along its own direction.** Shrunk by
   λ = ‖T − B‖ / ‖V − B‖ = 0.227 (after the BF16 cast its realized update norm is 1.518, 0.76 of TRACE's 1.994; λ was
   not retuned), V-NM keeps 32% [22, 42] of VERO's gain on the normalized P1 scale and scores 2.19 points below V
   (S(V-NM) − S(V) = −2.19 [−2.71, −1.66]). On the official P1 scale it is not distinguishable from V (−0.29
   [−0.81, +0.25]) only because VERO's official P1 gain is depressed by formatting. Under P2 (screen, seed 42) it keeps
   about half (F3; followup_findings.md §6).
8. **Decisions.** Merging study: adopt no merge, stop the merge search, do not train
   ([decision_memo.md](trace_merge/results/reports/decision_memo.md)). Follow-up: stop; no further component, merge or
   coefficient experiments, no new hybrid campaign, no training proposal (followup_findings.md §9).

Items 1–2 are the merging study's results (item 2 is confirmatory); items 3–7 are the follow-up's and are exploratory.

## 3. How to read the numbers

**Checkpoints.**
- B `Qwen/Qwen2.5-VL-7B-Instruct@cc594898`, T `maveryn/trace-qwen2.5-vl-7b@4d0f1ae8`, V
  `zlab-princeton/Vero-Qwen25-7B@180e84be`; one released checkpoint per recipe.
- V@P0, V@P1, V@P2: VERO under its native interface, the backbone's, or VERO's own files. In the merging study V@P2 is
  the archived V@P0; in the follow-up it is a new run. Unsuffixed labels are P1.
- M050 = ½V + ½T, CV050 = ½V + ½B, CT050 = ½T + ½B: all 729 tensors interpolated in float32, one BF16 cast. At the
  weight level M050 − CV050 = ½(T − B) and M050 − CT050 = ½(V − B).
- TF, TL, VF, VL: exact-copy hybrids. Frontend F = vision patch embedding, 32 vision blocks and merger (390 tensors);
  language L = token embeddings, 28 decoder layers, final norm and output head (339). TF = (F_T, L_B), TL = (F_B, L_T),
  and VF, VL likewise. TR0, TR1, TR2: T with decoder layers 0–8, 9–18 or 19–27 restored to B. V-NM (`VNM` in tables):
  (1 − λ)B + λV with λ = ‖T − B‖ / ‖V − B‖ = 0.226577.

**Follow-up work packages.** F0: answer-format audit; F1: component effects (TF, TL, VF, VL); F2c: TRACE decoder
block-group reversion (TR0–TR2); F3: norm-matched VERO (V-NM).

**Component effects** (S = macro). g_F^P = S(F_P, L_B) − S(B), g_L^P = S(F_B, L_P) − S(B),
I_P = S(P) − S(F_P, L_B) − S(F_B, L_P) + S(B), so S(P) − S(B) = g_F^P + g_L^P + I_P. I depends on the scale.

**Interfaces** (non-weight files served with the weights; never averaged or chosen by score). P0: each parent's own
files (archived runs); B and T render identical prompts, so the merging study used their archived runs as P1 references
(T's archived view lacks B's extra stop id, `<|endoftext|>`). P1: B's 8 non-weight files, primary in both studies. P2:
V's 11 files, whose template replaces any system message with a fixed think/answer prompt and pre-fills `<think>`
(prompts a median 486 tokens longer); development partition only, never confirmation (merging study: screen, seed 42;
follow-up: smoke, seed 42, and screen, seeds 42–44).

**Partitions and metric.** Stage B: archived B, T, V@P0 on the full split (32,805 rows, 24 benchmarks, seeds 42–44).
Smoke: 135 rows. Development screen: 2,910 rows from the 6,577-row development partition. Confirmation: 26,228 rows,
seeds 42–44. The merging study froze the split (rows sharing an image or WeMath problem family stay together; twins are
joined for resampling: 20,957 bootstrap units on confirmation, 2,421 on the screen) and the follow-up reused it. The
macro is the equal-weight mean of the 24 official benchmark scores, averaged over seeds, with six categories of four
benchmarks; WeMath on subsets is scored over the problem groups present (merging DESIGN A2). TableVQABench's score is
the mean of five split scores, among them the two WTQ splits (vwtq, vwtq_syn), whose prompt asks for several answers to
be separated by `||`, and vtabfact (True/False labels).

**Scales.** "Official" (pinned scorers) is primary in both studies. "Normalized" is the follow-up's frozen
format-normalized-v2 diagnostic: gold-blind and only raising scores, it re-reads PuzzleVQA and VisualPuzzles rows where
the pinned parser finds no option and re-scores TableVQABench rows after two narrow rewrites; the other 21 benchmarks are
unchanged. It is never the benchmark score.

**Intervals and tests.** 95% paired cluster bootstraps (10,000 replicates, seed 314159, all models and seeds resampled
together), in percentage points. Confirmatory: only the merging study's four-comparison primary family (M050 against T,
V@P1, CV050, CT050; Holm). The follow-up Holm-adjusts its prespecified six-effect family (g_F, g_L, I for T and V) but
is exploratory as a whole; F2c applies Holm over three effects as a descriptive safeguard. Categories, benchmarks,
row-level capture (the share of one parent's unique successes that a merge also solves), P2, F3 and sensitivity
analyses are descriptive. Decoding seeds are not training replications.

## 4. Merging study ([`trace_merge/`](trace_merge/README.md))

Stage B, three 50:50 merges, a seed-42 screen (P1 and P2), then confirmation under P1 against a selection lock written
before any confirmation run had scores; 57.57 recorded GPU-hours (about 62 with one untimed attempt). Key findings
(FINDINGS.md §1, decision_memo.md):

1. **Different places (native interfaces, Stage B).** Exploratory category contrasts put TRACE ahead in Charts & Tables
   (T − V@P0 = +2.42 [+1.40, +3.43]) and VERO ahead in Visual Math (−2.02 [−3.15, −0.86]). Under P1 the Visual Math point
   estimates are level (V@P1 48.64, T 48.69 on confirmation; no paired interval).
2. **Unique successes are partly reproducible.** The parents disagree on 21.8% of binary-scored rows on one seed. A
   T-only set recurs on another seed 47.1% of the time, a V-only set 50.0% (label-swap null 26.9% for both).
3. **Update geometry (diagnostic).** ‖V − B‖ is 4.4 times ‖T − B‖ (0.406% against 0.092% of ‖B‖), larger in every group
   that changed (ratio 3.81 to 6.90); cosine 0.0073. Norms and cosines alone do not show separable skills.
4. **VERO's gain depends on its interface.** V@P1 − V@P0 = −2.94 [−3.33, −2.56]. PuzzleVQA (−33.10) and VisualPuzzles
   (−13.29) carry 66% of the drop. Many V@P1 answers end in a LaTeX-wrapped option letter that the pinned scorers do not
   read (27.6% of its PuzzleVQA rows end in `\( X \)`), and 32.0% of its PuzzleVQA rows are scored 0 although the final
   letter is correct. TableVQABench (−15.97, about 23%) was not diagnosed here (see F0 in §5).
5. **Confirmed merge result** (P1, confirmation, Holm-adjusted):

   | comparison | difference [95% CI] | Holm p |
   |---|---|---|
   | M050 − T | +0.23 [−0.19, +0.65] | 0.29 |
   | M050 − V@P1 | +3.07 [+2.62, +3.51] | < 0.001 |
   | M050 − CV050 | +1.29 [+0.93, +1.64] | < 0.001 |
   | M050 − CT050 | +1.88 [+1.45, +2.31] | < 0.001 |

   M050 scores 52.36, T 52.13, V@P0 52.23; the study computed no paired interval for M050 − V@P0 (the follow-up did,
   §6). Without the 130 rows with a development twin, M050 − T = +0.23 [−0.19, +0.66].
6. **Halving each update.** CT050 keeps 59% of TRACE's gain (+2.38 of +4.02 over B). CV050 scores 1.77 above V@P1,
   consistent with half of VERO's update softening its answer-format shift.
7. **Categories (descriptive).** M050 − T has an interval below 0 only in Charts & Tables (−0.91 [−1.76, −0.07]) and
   above 0 in Science & General (+1.15 [+0.24, +2.04]) and Spatial Reasoning (+1.13 [+0.14, +2.15]); the other
   categories are not resolved (e.g. Puzzles & Logic −0.72 [−1.97, +0.48]). VERO's half-update gains most in Visual
   Math (M050 − CT050 +3.73 [+2.69, +4.78]), TRACE's in Science & General (M050 − CV050 +2.33 [+1.52, +3.15]).
8. **Row-level capture** (sets defined by T and V@P1 on confirmation, not the V@P0 sets of item 2; independent repeat;
   post-hoc base reference, no paired intervals). M050 keeps VERO-only
   successes at 52.8% (base model 36.7%) and TRACE-only successes at 49.0% (base model 47.6%). In accuracy terms
   (differences of point estimates), M050's pooled binary-accuracy lead over CV050, CT050 and B (+1.19 / +2.01 /
   +4.25 pp) has its largest term in the shared successes it keeps (+0.99 / +2.21 / +4.09 pp); against CT050 the
   VERO-only successes it keeps add +1.59 pp. It rescues fewer joint failures than any of them.
9. **Frozen hypotheses.** H1 (integration beyond shrinkage) holds in direction; H2 (category trade-off transfers) is
   half supported; H3 (interface dependence) holds for the macro and Puzzles & Logic. H3 is not supported for Visual
   Math being one of the two largest drops, because Charts & Tables drops more (−3.89 [−4.73, −3.04], through
   TableVQABench), although Visual Math also drops (−1.77 [−2.68, −0.90]).
10. **Measurement checks.** All 25 new runs and 2 local reruns render the same prompts as their archived references on
    all 407,076 rows checked, and all 24 metric adapters reproduce the archived aggregates (largest error 4.5e-11 pp).
    Local reruns of B and T move the macro by −0.25 and −0.02 pp. With them in place of the archived B and T on seed
    42, category contrasts involving B or T move by up to 1.23 pp, and 3 category intervals change side of 0 (M050 − T
    in Spatial Reasoning; CV050 − B and CT050 − B in Puzzles & Logic).
11. **Decision.** Do not adopt M050 or any merge (standalone TRACE remains the contribution); stop the merge search (no
    α sweep, no TIES/DARE/SLERP, no component experiment from the design's approval-gated Stage D); do not train. The
    memo's proposed exploratory check, M050 and CV050 under P2 on the confirmation rows (six runs, roughly 30–37
    GPU-hours by the memo's estimate, plus preferably three VERO reruns under P2, roughly 15–20 more; it needed a DESIGN
    amendment, researcher approval and GPU authorization), was not run.

**The judge fault** ([DESIGN.md](trace_merge/DESIGN.md) A8, A9; FINDINGS.md §8).
- M050 confirmation seed 44 was scored with the Qwen3-32B judge at tensor-parallel degree 4 on A800 40 GB cards, the
  only such configuration. It scored 16.98 on MathVerse against 43.65 and 47.30 on seeds 42 and 43, with normal-looking
  responses; the judge-extracted answer equalled the response's own `\boxed{}` answer on 14.8% of the 582 MathVerse
  rows with a boxed answer, against 43–88% on all 24 other runs. The faulty judge outputs often ran on and copied the
  judge prompt's few-shot text (133 of 630 MathVerse extraction outputs); the cause in the serving stack was not
  isolated.
- Generation was kept; the scoring outputs were quarantined and the scoring half rerun at degree 1, under a rule fixed
  before any rescored value was read. Seed-44 values changed on MathVerse (16.98 → 42.38), MathVista (67.25 → 76.63),
  LogicVista (47.21 → 49.44), MathVision (27.18 → 27.92) and slightly on MMStar and SpatialVizBench COT.
- Every table in both studies uses the repaired scores; the superseded M050 − T = −0.29 is quoted only to describe the
  fault. The repair followed a first read of the confirmation tables, but nothing locked changed. The fault was found
  from the outlying seed-44 scores during independent verification; the health check adopted with A8 (MathVerse
  boxed-versus-extracted agreement, floor 30%) covers one judge-routed benchmark, not the archived runs.

## 5. Follow-up ([`trace_module_followup/`](trace_module_followup/README.md))

Seven cells under P1 on the confirmation partition, seeds 42–44 (21 runs; B, T, V regenerated), plus smoke, screen, a
P2 template matrix, the A5 replication, F2c and F3: 99 runs, 135.3 GPU-hours on already-held allocations. Builds passed
identity, ownership and parent-hash checks, ingest reproduced every run's aggregates to ≤ 5e-11 pp, and no cell was a
technical stop. Three runs were rescored after scorer or archive raises (its A1, A3), with fail-closed fixes. Section
numbers (§) in this part refer to [followup_findings.md](trace_module_followup/results/reports/followup_findings.md).

**F0, answer-format audit** (§2; [`tablevqa_diagnosis.md`](trace_module_followup/results/audit/tablevqa_diagnosis.md)).
- format-normalized-v2 was frozen before any hybrid output (85 tests). It replaced v1 after a blinded check found two
  TableVQABench defects, before any normalized confirmation value was computed and before any new run existed; a
  blinded check of 80 development rows agreed 80/80.
- On the merging study's stored confirmation outputs, V@P1 − V@P0 goes from −2.94 to −0.83 [−1.21, −0.46]: about 72%
  ((2.94 − 0.83) / 2.94) is formatting. PuzzleVQA −33.1 → +0.4, VisualPuzzles −13.3 → −0.6, TableVQABench −16.0 → −11.4.
- TableVQABench's remainder is behaviour: under P1 VERO lists several `||`-separated values on 47% of its WTQ rows
  (the vwtq and vwtq_syn splits; 4% natively), which the scorer correctly marks wrong, and it gives more wrong
  True/False labels on the vtabfact split. Neither is repaired.
- M050 − T moves from +0.23 [−0.19, +0.65] to +0.62 [+0.20, +1.04] (descriptive; the interval excludes 0 only on a
  diagnostic scale chosen after the merging study, so no superiority claim follows; the confirmatory family is unchanged).

**F1, component effects** (P1, confirmation, three seeds; §3,
[`analysis_summary.json`](trace_module_followup/results/tables/analysis_summary.json)).

| effect | official [95% CI] | Holm p | normalized [95% CI] |
|---|---|---|---|
| g_F^T | +0.41 [+0.12, +0.69] | 0.025 | +0.44 [+0.15, +0.73] |
| g_L^T | +3.47 [+3.04, +3.90] | < 0.001 | +3.36 [+2.93, +3.79] |
| I_T | +0.08 [−0.32, +0.48] | 0.71 | +0.07 [−0.33, +0.47] |
| g_F^V | +0.25 [−0.09, +0.60] | 0.46 | +0.28 [−0.06, +0.63] |
| g_L^V | +1.28 [+0.77, +1.80] | < 0.001 | +3.04 [+2.52, +3.55] |
| I_V | −0.24 [−0.71, +0.23] | 0.62 | −0.10 [−0.56, +0.37] |
| S(T) − S(B) | +3.95 [+3.52, +4.39] | | +3.87 [+3.43, +4.31] |
| S(V) − S(B) | +1.29 [+0.74, +1.83] | | +3.22 [+2.67, +3.76] |

- Macros, official / normalized: B 48.04 / 48.22, T 51.99 / 52.09, V 49.34 / 51.44, TF 48.45 / 48.66, TL 51.51 / 51.58,
  VF 48.29 / 48.50, VL 49.33 / 51.26. Transplanting TRACE's language component alone (TL) recovers 88% of TRACE's gain
  (3.47 / 3.95, ratio of point estimates).
- Removing TRACE's frontend costs S(T) − S(TL) = +0.48 [+0.19, +0.78]; removing VERO's costs +0.01 [−0.35, +0.36]. The
  official TRACE–VERO language gap, +2.19 [+1.67, +2.69], falls to +0.33 [−0.18, +0.83] after normalization; the
  formatting is concentrated in Puzzles & Logic (VERO g_L −5.2 official, +4.3 normalized).
- Categories (exploratory, official scale): TRACE's g_L is largest in Charts & Tables (+5.4) and Visual Math (+4.8),
  VERO's in Visual Math (+6.2) and Science & General (+4.0). Frontend effects are at most +1.02 in every category, the
  largest being TRACE's in Puzzles & Logic (+1.02 [−0.02, +2.10]; +1.23 [+0.18, +2.29] on the normalized scale, the
  only category frontend effect above +1.02 on either scale).
- Success-set retention (rows the parent solves and B does not on seed 42, re-measured on seed 43; 21 binary-scored
  benchmarks; [`behavior_and_format.csv`](trace_module_followup/results/tables/behavior_and_format.csv)). TRACE's set
  (2,773 rows): TL 59.1%, T's own repeat 61.2%, B 38.3%, TF 39.8%, so TL keeps about 91% of the above-floor retention
  ((59.1 − 38.3) / (61.2 − 38.3)). VERO's set (2,834 rows): VL 68.8%, V 78.2%, B 36.0%, VF 39.4%; VL keeps about 78%.

**Interface sensitivity and the A5 replication** (development screen; §4). Seed 42 showed P2 − P1 = −4.00 for B and
+3.29 for V, so A5 reran the 2,910 rows with seeds 43 and 44 under both interfaces for all seven cells (28 runs).
Three-seed P2 − P1 ([`interface_3seed_summary.json`](trace_module_followup/results/tables/interface_3seed_summary.json)):

| | B | T | V | TF | TL | VF | VL |
|---|---|---|---|---|---|---|---|
| official | −2.88 [−4.14, −1.60] | −3.40 [−4.65, −2.11] | +3.19 [+2.16, +4.19] | −2.55 [−3.81, −1.24] | −3.39 [−4.67, −2.10] | −2.67 [−4.00, −1.34] | +3.35 [+2.30, +4.43] |
| normalized | −1.33 | −1.51 | +1.09 | −0.97 | −1.40 | −0.99 | +1.55 |

Under P2, S(V) − S(B) = +7.48 [+5.87, +9.10] (normalized +5.75), with g_L^V = +7.56 and g_F^V = +0.37. TRACE's gain is
not resolved to differ between interfaces (+4.17 under P1, +3.64 under P2; P2 − P1 = −0.53 [−2.02, +0.99]); all
interactions are within ±0.6, intervals including 0.

**F2c, TRACE decoder block-group reversion** (§5; [`decision_F2.md`](trace_module_followup/design/decision_F2.md)).
Chosen on development evidence (the record discloses a rough look at the partly complete confirmation runs, which
selected nothing): the other two options were not triggered (F2a, encoder-versus-merger transplants, because TF and VF
are within 0.2 of B; F2b, cross-parent swaps, because there is no complementary pair).

| checkpoint (T with … from B) | macro | R_k = S(T) − S(TR_k) | Holm p (descriptive) | S(TR_k) − S(B) |
|---|---|---|---|---|
| TR0, layers 0–8 | 51.42 | +0.58 [+0.25, +0.89] | 0.0006 | +3.37 |
| TR1, layers 9–18 | 50.18 | +1.81 [+1.42, +2.19] | 0.0003 | +2.14 |
| TR2, layers 19–27 | 51.79 | +0.21 [−0.11, +0.53] | 0.20 | +3.74 |

The effects are not additive (sum 2.60 of +3.95) and the groups differ in size (9/10/9 blocks). A large R_k says the
group is needed for the measured gain under this interface and scorer, not where a skill lives.

**F3, norm-matched VERO** (§6; [`decision_F3.md`](trace_module_followup/design/decision_F3.md)). λ = 0.226577
(‖T − B‖ = 1.9938, ‖V − B‖ = 8.7998). After the BF16 cast the realized update norm is 1.518 (−23.9% against the
target; 76.7% of stored values equal B's). λ was not retuned (its A4): V-NM is matched in coefficient, not realized norm.

| P1, confirmation, three seeds | official [95% CI] | normalized [95% CI] |
|---|---|---|
| macro: B / V-NM / V / T | 48.04 / 49.05 / 49.34 / 51.99 | 48.22 / 49.25 / 51.44 / 52.09 |
| S(V-NM) − S(B) | +1.00 [+0.68, +1.34] | +1.02 [+0.70, +1.36] |
| S(V-NM) − S(V) | −0.29 [−0.81, +0.25] | −2.19 [−2.71, −1.66] |
| share of VERO's gain kept | 0.78 [0.51, 1.29] | 0.32 [0.22, 0.42] |
| S(T) − S(V-NM) | +2.95 [+2.52, +3.37] | +2.85 [+2.42, +3.26] |

Under P2 (screen, seed 42) V-NM keeps 0.49 [0.29, 0.70] of VERO's gain official, 0.55 [0.31, 0.83] normalized. V-NM
is not distinguishable from V on the official P1 scale (−0.29 [−0.81, +0.25]) only because VERO's official P1 gain is
depressed by formatting: V-NM's official and normalized macros differ by 0.2 and VERO's by 2.1, so V-NM sheds the
format behaviour along with most of the gain.

**Decision** (§9): stop. The open points are properties of evaluation (interface choice) or would need training-time
experiments; a deployment-facing, interface-matched VERO comparison on an untouched set would be a new study, which the
follow-up does not propose.

## 6. How the two studies connect

1. **F0 accounts for most of the merging study's interface finding** (computed: F0 rescored the merging study's own
   stored outputs with a rule frozen before use). The PuzzleVQA diagnostic there (32.0% scored 0 with the correct letter)
   is what F0 repairs (−33.1 → +0.4), and F0 diagnoses the TableVQABench part the merging study left open.
2. **The merging study's contrasts on the normalized scale** (computed, exploratory;
   [`official_vs_normalized.contrasts.csv`](trace_module_followup/results/tables/official_vs_normalized.contrasts.csv),
   whose official values reproduce the merging study's). M050 − V@P1 falls from +3.07 to +1.45 [+1.00, +1.88],
   M050 − CV050 barely moves (+1.29 → +1.33 [+0.97, +1.68]) and M050 − CT050 rises to +2.28 [+1.85, +2.70]. M050's only
   category interval below T, Charts & Tables −0.91 [−1.76, −0.07], becomes −0.32 [−1.15, +0.51] (format-sensitive,
   descriptive); the memo cites that category in its first decision. It also gives the paired interval the merging
   study lacked: M050 − V@P0 = +0.12 [−0.37, +0.62] official, +0.62 [+0.13, +1.11] normalized. This sets new runs
   against archived ones at different interfaces, so the interval does not cover the 0.1–0.3 pp environment shift, and
   the normalized scale was chosen after the merging study, so no superiority claim follows. CV050's lead over V@P1
   falls from +1.77 to +0.11 (difference of the CV050 − B and V@P1 − B rows, no interval), in line with the merging
   study's reading that it reflected a softened format shift.
3. **Fresh references are close to the archived ones** (side by side, no paired test). New P1 confirmation macros:
   B 48.04, T 51.99, V 49.34; the merging study's: 48.10, 52.13 (archived) and 49.29 (its V@P1 runs).
4. **VERO's larger update does not show a larger frontend transfer** (reading). VERO's update is larger than TRACE's in
   every group that changed (4.4 times overall, 4.67 in the vision blocks, 4.34 in the decoder layers), yet VERO's
   frontend transplant is not shown to gain more than TRACE's (g_F^V = +0.25 [−0.09, +0.60], g_F^T = +0.41 [+0.12, +0.69];
   g_F^T − g_F^V = +0.16 [−0.19, +0.50], not resolved); this fits the merging study's caveat that norms do not establish
   skills.
5. **How much of each update is needed** (reading; different run sets and references). On the normalized P1 scale along
   B → V: V-NM (nominal λ = 0.227) +1.02 [+0.70, +1.36] over B, CV050 (λ = 0.5) +3.24 [+2.81, +3.65], V (λ = 1) +3.22
   [+2.67, +3.76] fresh (+3.12 [+2.58, +3.66] for the merging study's V@P1). CV050's point estimate is not below V's
   (no paired interval), while V-NM, whose realized update after BF16 rounding is 1.518 rather than the targeted 1.994,
   keeps about a third of VERO's gain. For TRACE (official scale), half the whole update keeps 59% (CT050) and the whole
   language update keeps 88% (TL).
6. **M050 − T and the component results** (side by side; not tested, and neither study links them). Both parents'
   measured gains transfer mainly through the language component (TRACE's frontend adds +0.41 [+0.12, +0.69]), with
   interactions not resolved (F1, confirmation), and decision_F2.md's trigger check (seed-42 development screen) found
   no complementary pair of transferable components. M050 = B + ½(T − B) + ½(V − B) adds both updates at half strength,
   and each half-update still adds to the other parent's blend (+1.29, +1.88). Neither study tested why the combination
   is not shown to beat T, or interference between the two updates directly.
7. **Interface dependence, replicated in direction and following the language component** (computed in the
   follow-up). The merging study's P2 evidence was one seed with archived V@P0 standing in for V@P2 (screen:
   V@P1 − V@P0 = −4.67 [−6.09, −3.21], −2.56 normalized). The follow-up regenerated every cell under both interfaces on
   the 2,910-row development screen with three seeds: V gains +3.19 official (+1.09 normalized) under its own
   interface, smaller than the merging screen's single-seed 4.67 (2.56 normalized); B and T lose, and VL shows the
   effect while VF does not.
8. **The merging memo's proposed check was not run.** Neither study evaluated M050 or CV050 under P2 on the
   confirmation rows; the only such evidence remains the single-seed screen, M050@P2 − CV050@P2 = −0.64 [−2.13, +0.87].
   The follow-up's component experiments, which the memo had not authorized, ran under a separate request recorded in
   [`followup_v1.md`](trace_module_followup/design/followup_v1.md) §0. Its stop decision rules out further merge
   experiments from its own results but does not address the memo's proposed P2 check, which remains unapproved and
   unrun.
9. **The judge lesson carried over.** After the merging study's amendment A8 the follow-up kept the judge at
   tensor-parallel degree ≤ 2 and ran the same MathVerse agreement check on every run (40–89% on every screen,
   replication and confirmation run).

## 7. Caveats and limits

- **Fixed checkpoints, confounded recipes.** One released checkpoint per recipe; no interval covers training-seed
  variation. TRACE and VERO differ in data, RL algorithm, prompts, reward routes, trainable modules and training length,
  so these are recipe-level findings, not a procedural-versus-natural-image effect. VERO's model card does not pin its
  base revision; reading V − B as an update from B assumes it.
- **Prior exposure.** The suite was examined before both studies; the confirmation partition was reserved from merge
  tuning only, and its reuse makes every follow-up result exploratory.
- **No margins.** "Not shown to be better than TRACE" is not "as good as TRACE"; no hybrid is said to keep a full gain.
- **Scales and the judge.** The normalized scale is a diagnostic; TableVQABench's remaining loss and judge-graded rows
  needing re-adjudication are not normalized. The judge decides extraction on judge-routed benchmarks for every model,
  so a response style it reads poorly is scored wrong.
- **Component effects and F3.** Transfer is not capability localization or evidence about training-time freezing; a
  failed transplant can reflect component mismatch. I depends on the scale; F2c groups are unequal and not additive.
  F3 tests one coefficient, and V-NM moved less than targeted, so a norm-exact construction might keep somewhat more.
- **Interface evidence is small.** P2 results come from the 2,910-row screen. Single-seed screen values also move
  between independent runs: the two studies' seed-42 B@P2 runs score 42.71 and 44.10, their T@P2 runs 47.21 and 48.56
  ([`scores.P2.csv`](trace_merge/results/tables/screen/scores.P2.csv),
  [`template_sensitivity.csv`](trace_module_followup/results/tables/template_sensitivity.csv)); the cause was not examined.
- **Environment.** Archived runs come from the canonical producer, new runs from the published code on mixed GPUs; local
  B/T reruns move seed-42 comparisons by 0.1–0.3 pp, the order of M050 − T. Cross-GPU determinism was not verified.
- **Repairs and twins.** Four runs were rescored after technical faults (merging A8; follow-up A1, A3), each under a rule
  fixed before any rescored value existed. The near-duplicate twin list (465 row pairs) is a lower bound.

## 8. Where everything is

| path (from `experiments/`) | contents |
|---|---|
| `trace_merge/README.md`, `DESIGN.md`, `configs/` | layout and commands; frozen design, amendments A1–A9; config |
| `trace_merge/trace_merge/`, `jobs/`, `scripts/`, `tests/` | package; GPU task runner (reused by the follow-up); entry point, `write_findings.py`; tests |
| [`trace_merge/results/reports/FINDINGS.md`](trace_merge/results/reports/FINDINGS.md) | start here: key findings, settings, checks, every data table |
| `trace_merge/results/reports/` (others) | `analysis_report.md`, `decision_memo.md`, frozen `behavior_report.md`, `merge_screen.md`, `selection_lock.json` |
| `trace_merge/results/tables/` | `stage_b/`, `screen/`, `confirmation/`; run status, costs, environment drift, length/format/judge audit, judge health |
| `trace_merge/results/audit/`, `manifests/`, `splits/`, `merges/`, `figures/` | compatibility, update norms, prompt conformance, metric reproduction; roles, views, metric registry; partition, bootstrap units; merge recipes |
| `trace_module_followup/README.md`, `configs/`, `design/` | layout and commands; config; design (amendments A1–A5), F2 and F3 decision records |
| `trace_module_followup/trace_followup/`, `tests/` | package (builder, views, F0 scorer, analyses, export); tests |
| [`trace_module_followup/results/reports/followup_findings.md`](trace_module_followup/results/reports/followup_findings.md) | start here: summary, §1–§10, decision |
| `trace_module_followup/results/tables/` | component effects, seven-cell scores, official vs normalized, interface, F2c, F3, behaviour and format, costs |
| `trace_module_followup/results/audit/` | component map, build verification, F3 norms, parent hashes, scoring specs, `scoring_changes.md`, `tablevqa_diagnosis.md` |
| `trace_module_followup/results/manifests/`, `hybrids/`, `scaled/`, `dispatch/` | every run (`evaluation.json`), views; hybrid and V-NM recipes; the three protocol-overlay patches |

**What stays local** (work roots, approximate sizes; not committed because of size and because both studies commit
aggregates only, with no prompts, gold answers, images, responses, weights, machine paths or tokens):
- Follow-up: 123.6 GiB of constructed weights (TF, TL, VF, VL, TR0, TR1, TR2, V-NM) and 15.2 GiB of run directories
  (followup_findings.md §8); aligned and scored rows (966 MiB); F0 row ledgers and blinded-check samples (130 MiB);
  inference views (242 MiB); frozen split copies (28 MiB); machine-specific GPU dispatch tooling and task files
  (1.3 MiB); build, analysis and filler logs.
- Merging study: 46.34 GiB of merged weights and 5.33 GiB of run directories (FINDINGS.md §3); aligned rows (503 MiB);
  inference views (120 MiB). Also the private protocol overlay clone (363 MiB; its three patches are committed under
  `trace_module_followup/results/dispatch/`) and the downloaded archived Stage B runs (372 MiB;
  `maveryn/trace-eval-runs`).

**Regenerating the reports** needs the local work roots, named by `TRACE_MERGE_WORK_ROOT`, `TRACE_MERGE_PROTOCOL_CLONE`
and `TRACE_FOLLOWUP_ROOT`; each README lists the full sequence. Merging study ([README](trace_merge/README.md)):
`scripts/trace_merge_study.py analyze`, `screen`, `confirm`, `figures`, then `python scripts/write_findings.py` and
`export`. Follow-up ([README](trace_module_followup/README.md)): `python -m trace_followup analyze`, `analyze-f2c`,
`analyze-f3`, `analyze-interface`, then `export`. Tests (`python -m pytest -q tests` per study) run without a work
root, but the merging study's real-data checks then skip; 30 follow-up tests need the private overlay clone and 3 more
need the pinned protocol clone's VLMEvalKit, and they skip without them.
