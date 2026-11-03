# Decision memo: TRACE–VERO merging study (`trace_vero_qwen25_v1`)

**Question.** Do the existing TRACE and VERO Qwen2.5-VL-7B checkpoints improve on their shared backbone in different
ways that a single training-free merge can combine? And does the evidence justify new training?

**Short answer.**
- **Different strengths.** The two models' strengths are partly different, and partly reproducible across decoding
  seeds.
- **The merge.** Under the backbone interface, a 50:50 merge beats both shrinkage controls and VERO-under-that-interface.
  It is not shown to be better than TRACE (+0.23 [−0.19, +0.65]).
- **Stop merging.** No merge should be adopted as an improvement, and the merging search should stop here.
- **No training yet.** This study does not justify new training. One evaluation-only check, which needs approval, can
  show whether a matched continuation experiment is worth proposing.

Nothing was trained, uploaded or published by this study. The evidence is in `reports/analysis_report.md`; every
confirmation number below uses the scores repaired under DESIGN amendment A8.

## Evidence

1. **Similar gains, different places.**
   - On the archived runs, TRACE and VERO improve the 24-benchmark macro by +4.06 and +4.30 pp over the base. Their
     difference is not resolved.
   - Exploratory category contrasts at native interfaces put TRACE ahead in Charts & Tables (+2.42) and VERO ahead in
     Visual Math (+2.02).
2. **The parents' unique successes are partly reproducible.**
   - They disagree on about 22% of binary-scored rows.
   - About half of each parent's unique-success set recurs on another decoding seed, 20–23 pp above a label-swap null.
3. **VERO's measured gain depends on its own interface.**
   - Served with the backbone's template (P1), VERO loses 2.94 pp on the confirmation macro.
   - Its Puzzles & Logic category score falls 11.20 pp.
   - About two-thirds of the macro loss sits in PuzzleVQA and VisualPuzzles. There, 32% of V@P1's PuzzleVQA rows are
     scored 0 although the final letter is correct: an answer format the pinned scorers do not read.
   - The TableVQABench part of the loss (about 23%) is not diagnosed.
4. **Confirmed merge result** (P1, 26,228 rows, 3 seeds, Holm-adjusted):

   | comparison | difference (pp) [95% CI] | Holm p |
   |---|---|---|
   | M050 − T | +0.23 [−0.19, +0.65] | 0.29 |
   | M050 − V@P1 | +3.07 [+2.62, +3.51] | 0.0008 |
   | M050 − CV050 | +1.29 [+0.93, +1.64] | 0.0008 |
   | M050 − CT050 | +1.88 [+1.45, +2.31] | 0.0008 |

   M050 scores 52.36, against VERO's native 52.23 and TRACE's 52.13. Its gap to native VERO is a difference of point
   estimates with no interval, outside the prespecified family.
5. **How the merge combines its parents.** These use unique-success sets defined by T and V@P1 on the confirmation
   partition, not the V@P0 sets of item 2.
   - **VERO side.** M050 keeps VERO-only successes at 52.8%, well above what the base model gets on the same rows
     (36.7%).
   - **TRACE side.** It keeps TRACE-only successes at 49.0%, close to the base model's own rate (47.6%; no paired
     interval). The TRACE blend CT050 keeps 54.6%. (Independent-repeat variant; the base rates are a post-hoc
     reference.)
   - **Categories.** Descriptively, M050 is below T in Charts & Tables, TRACE's lead category (−0.91 [−1.76, −0.07]),
     and above T in Science & General and Spatial Reasoning.
6. **Frozen hypothesis H2 (descriptive) was half supported on confirmation.** VERO's half-update helps most in Visual
   Math, as predicted. TRACE's helps most in Science & General, not Charts & Tables.
7. **One technical repair.** A judge fault on one confirmation run (degraded, run-on judge extractions; A8, A9) was
   found during verification and repaired by rescoring. Before the repair it had made M050 look worse than TRACE (−0.29) and weak in Visual Math.

## Which outcome this is (handoff §12)

| handoff outcome | applies? |
|---|---|
| TRACE-specific gains persist, and the merge beats the parents and base controls | **Partly.** Under P1 the merge beats both shrinkage controls and V@P1, but is not shown to beat T (+0.23 [−0.19, +0.65]). It is 0.12 pp above native VERO as a difference of point estimates (no interval). M050's TRACE-only capture (49.0%) is close to the post-hoc base reference (47.6%; no paired interval) (item 5). |
| The parents have reliable different strengths, but simple merges interfere | **Partly.** The strengths are partly reproducible (items 1–2). At the macro, each half-update adds over the other parent's blend, and no category contrast of M050 against a parent or control is significantly negative except Charts & Tables against T (−0.91, unadjusted). At the row level, M050's TRACE-only capture is close to the post-hoc base reference (no paired interval). |
| The merge is no better than VERO-to-base interpolation | **No.** M050 − CV050 = +1.29, Holm p < 0.001: under P1, TRACE's half-update adds to the VERO blend. |
| Apparent gains mostly disappear under protocol/format controls | **For comparisons with VERO, largely yes.** M050's +3.07 over V@P1 sits mostly in Puzzles & Logic (+9.89), where V@P1 is format-depressed. Against native VERO, the merge is only 0.12 pp ahead (no interval). Under VERO's own interface (P2), only the single-seed development screen exists. There, M050@P2 − CV050@P2 = −0.64 [−2.13, +0.87]: no gain was detected, and the upper bound lies below the +1.29 confirmed under P1. But the screen is single-seed and small, and its P1 and P2 estimates are not paired (the same screen gave +0.85 [−0.67, +2.35] under P1), so it does not show the gain is absent under P2. |
| Little unique benefit | **No.** Unique successes are partly reproducible, and each half-update adds under P1. |

**Interpretation.**
- **Diversity and compatibility.** These checkpoints show behavioural diversity, and some weight compatibility: the
  merge beats both shrinkage controls.
- **Not shown to be better.** The 50:50 merge M050 is not shown to be better than its better parent under the backbone
  interface, TRACE (+0.23 [−0.19, +0.65]). Other coefficients and merge methods were not tested.
- **The open question.** Under P1, TRACE's half-update adds +1.29 [+0.93, +1.64] to the VERO blend. Whether it adds
  anything under VERO's own interface has not been tested at confirmation scale. That question bears on whether any
  VERO-based continuation is worth proposing.

## Decisions

1. **Do not adopt M050 or any other merge as an improvement.**
   - Under the backbone interface it is not shown to be better than TRACE, and descriptively it is below TRACE in Charts
     & Tables (−0.91 [−1.76, −0.07], unadjusted).
   - Standalone TRACE remains the contribution.
2. **Stop the merging search.**
   - No coefficient sweep (α = 0.25/0.75), no TIES/DARE/SLERP, and no Stage D component experiment.
   - Any of these would need a written development finding and researcher approval (DESIGN.md header). None exists.
   - An α search on this confirmation partition would be tuning on a partition already read, and it would need a new
     untouched confirmation set.
3. **No training.** Run the evaluation-only check below only if it is approved.

## Recommended next step (evaluation only; exploratory; needs a DESIGN amendment, researcher approval and GPU authorization)

**Check, on the confirmation partition and under VERO's own interface, whether TRACE's half-update adds to the
half-VERO blend, and whether the merge stays within δ of released VERO.**

**Scope and approvals.**
- DESIGN.md §3 restricts the P2 interface to the development screen. Running it on the confirmation partition
  therefore extends a frozen protocol.
- It needs, before any run: a written DESIGN amendment fixing the comparisons and the decision rule below, researcher
  approval, and GPU authorization.
- It is chosen after the confirmation results were seen and was not in the locked family, so it is exploratory.

**Runs.**
- Serve M050 and CV050 with VERO's interface on the 26,228 confirmation rows, seeds 42–44: six runs.
- V@P2 is the archived V@P0 by construction (DESIGN.md §3). Comparison (b) then sets new runs against archived ones
  and carries the environment shift measured in analysis report §4.5 (0.1–0.3 pp on seed 42), a sizeable fraction of
  δ. The amendment should either re-run VERO under P2 in the current environment (three more runs, preferred) or state
  this limitation as part of the rule.
- **Estimated cost:** roughly 30–37 GPU-hours for the six runs (roughly 15–20 more if VERO is re-run, since its P2 responses are longer). The 12 P1
  confirmation runs averaged about 4.5 GPU-hours each, counting the attempt the timing files miss; the M050 and CV050
  runs alone averaged about 5.1. P2 responses were about 20% longer than P1 on the development screen. The estimate
  scales by completion tokens only; it leaves out P2's ~490-token longer prompts and the hardware mix.

**Comparisons and decision rule** (proposed, to be fixed in the amendment before the runs):
- **Comparisons.**
  - (a) M050@P2 − CV050@P2: TRACE's half-update on top of the half-VERO blend, under VERO's interface.
  - (b) M050@P2 − V@P0: the merge against released VERO in its own interface. This is the model a continuation would
    start from.
- **Estimand.** The 24-benchmark macro on the 26,228 confirmation rows, seeds 42–44. Intervals are 95% paired cluster
  bootstraps over the A7 bootstrap units (10,000 replicates, seed 314159). p-values are Holm-adjusted over (a) and (b).
- **Margin.** δ = 0.5 pp, proposed: about 40% of the +1.29 pp that the same contrast showed under P1. The researcher
  sets the final value in the amendment.
- **Outcomes,** checked in this order:
  1. *No meaningful addition:* the upper bound of (a) is below +δ. This study then gives no merge-based reason to
     propose training, and standalone TRACE remains the contribution.
  2. *Adds:* (a) is positive with Holm p < 0.05, and the lower bound of (b) is above −δ, so the merge is not more than
     δ below released VERO. A matched continuation pair is then worth *proposing* for approval (handoff §12 option 2):
     released VERO → TRACE-64K against released VERO → VERO-Fresh64K, both evaluated under one fixed interface, with the
     parents re-run under that interface.
  3. *Inconclusive:* anything else. Report it as inconclusive; this study then makes no training proposal.

**What the check can and cannot show.**
- The rule uses a training-free merge as one input. A merge result does not predict how continuation or joint RL
  training would behave.
- A null result means only that this study gives no merge-based reason to propose training. It does not show that
  continuation training would not help.
- A positive result does not show that it would help.

**Not recommended now.**
- Original-Qwen → VERO-Fresh64K (handoff §12 option 1) is the cleaner data-source control, but it does not test
  integration.
- Qwen3 work waits for the Qwen2.5 question above.

## Limits on this decision

- **Single checkpoints.** One released checkpoint per recipe; the intervals do not cover training-seed variability.
- **Confounded recipes.** The recipes differ in data, RL algorithm, prompts and training length. Nothing here isolates
  procedural versus natural image data.
- **Prior exposure.** The suite was studied before. The confirmation partition was reserved from tuning, but it is not
  unseen.
- **No margin.** No noninferiority margin was set. "Not shown to be better than TRACE" must not be read as "as good as
  TRACE".
- **One repaired run.** One run's scores were repaired after the confirmation tables were first read (A8). The repair
  rule was fixed before any rescored value was read, and the confirmed set, family, split, merges and views did not
  change.
