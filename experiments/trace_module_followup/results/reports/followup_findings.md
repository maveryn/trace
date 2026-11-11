# TRACE–VERO follow-up: findings

Study `trace_vero_followup_v1`. Design: `design/followup_v1.md` (amendments A1–A5), `design/decision_F2.md`,
`design/decision_F3.md`. Execution guide: "TRACE–VERO Training-Free Follow-up Experiments" (plan v1.0, 5 Nov 2026).
**Exploratory:** the confirmation partition was examined in the reviewed merging study (PR maveryn/trace#14), so it is
not unseen. Intervals describe evaluation uncertainty for these fixed checkpoints, not training-run variation. A
non-significant difference is not evidence of equivalence. No noninferiority margin was set. Nothing was trained or
uploaded.

## Summary

1. **TRACE's gain transfers through its language component, with no measurable co-adaptation.**
   * On the P1 (backbone) interface, confirmation partition, three seeds and the 24-benchmark macro, TRACE gains
     +3.95 [3.52, 4.39] points over B. It decomposes into language g_L = +3.47 [3.04, 3.90], frontend
     g_F = +0.41 [0.12, 0.69] and interaction I = +0.08 [−0.32, 0.48].
   * Holm-adjusted p is < 0.001 for g_L, 0.025 for g_F and 0.71 for I.
   * The same decomposition holds on the format-normalized scale: g_L +3.36, g_F +0.44, I +0.07.
2. **VERO's gain also sits in its language component.** On the official P1 scale it is small: +1.29 overall, with
   g_L = +1.28 [0.77, 1.80], g_F = +0.25 (n.s.) and I = −0.24 (n.s.). After format normalization VERO's language gain
   is +3.04 [2.52, 3.55], statistically indistinguishable from TRACE's (gap +0.33 [−0.18, 0.83]). The official-scale
   TRACE–VERO language gap of +2.19 [1.67, 2.69] is therefore mostly answer formatting under P1. That formatting is
   concentrated in Puzzles & Logic (VERO g_L −5.2 official, +4.3 normalized).
3. **The interface dependence follows the language component.** On the development screen (three seeds), changing
   from P1 to VERO's interface P2 moves V by +3.19 [2.16, 4.19] and VL by +3.35. It moves B by −2.88, T by −3.40, TF
   by −2.55, TL by −3.39 and VF by −2.67. Under P2, VERO's gain over B is +7.5, almost entirely language-side (+7.6).
   The component pattern (language-side and additive, frontend ≈ 0) holds under both interfaces.
4. **Within TRACE's decoder, the middle block group matters most (F2c).** Restoring layers 9–18 to B removes 1.81
   [1.42, 2.19] points of TRACE's +3.95. Layers 0–8 remove 0.58 [0.25, 0.89], and layers 19–27 remove 0.21
   [−0.11, 0.53] (n.s.). The three reversions are not additive (they sum to 2.60). A large reversion effect says the
   group is needed for the measured gain under this interface and scorer; it does not localize a "skill".
5. **Shrinking VERO's update to TRACE's norm (F3) loses most of VERO's real gain.** On the format-normalized P1
   scale, V-NM = (1 − λ)B + λV with λ = 0.2266 keeps 32% [22, 42] of VERO's gain and scores −2.19 [−2.71, −1.66]
   below V. Under VERO's own interface it keeps about half. The official P1 scale shows V-NM ≈ V (−0.29, n.s.) only
   because VERO's official P1 gain is depressed by formatting. TRACE beats V-NM by +2.95. Caveat: BF16 rounding left
   V-NM's realized update at 1.518 rather than the targeted 1.994, and λ was not retuned.
6. **Hybrids keep parent-specific successes through the language component.**
   * TRACE's success set (rows T solves and B does not on seed 42): TL re-solves 59.1% on seed 43. TRACE's own
     repeat re-solves 61.2% and B's 38.3%, so TL keeps about 91% of the above-floor retention. TF re-solves 39.8%,
     at B's floor.
   * VERO's success set: VL 68.8% against V's own 78.2% and B's 36.0% (about 78% of the above-floor retention).
     VF re-solves 39.4%.
7. **Decision: stop** (§9).

## 1. What was run

| Work package | Checkpoints | Interface × rows × seeds | Runs |
|---|---|---|---|
| F0 scoring audit | archived B/T/V and the reviewed study's runs | stored outputs only | 0 (CPU) |
| F1 main matrix | B, T, V, TF, TL, VF, VL | P1 × confirmation (26,228 rows) × 42/43/44 | 21 |
| F1 smoke / screen | the same 7 | P1 and P2 smoke; P1 development screen (2,910 rows) × 42 | 21 |
| F1 template matrix | the same 7 | P2 × development screen × 42 | 7 |
| A5 replication | the same 7 | P1 and P2 × development screen × 43/44 | 28 |
| F2c (decision_F2.md) | TR0/TR1/TR2 (T with layers 0–8 / 9–18 / 19–27 from B) | P1 smoke, screen × 42; confirmation × 42/43/44 | 15 |
| F3 (decision_F3.md) | VNM = (1 − λ)B + λV | P1 smoke, screen, confirmation × 3; P2 smoke, screen | 7 |

Every hybrid copies each stored tensor byte-for-byte from its designated donor in B's shard layout. The build checks
passed: identity (all-B/T/V reconstructions), ownership (TF ∪ TL = T's 662 changed tensors and VF ∪ VL = V's 666,
with no overlap), finiteness, and parents unchanged. The parent weight files match the reviewed study's hashes.
Ingest of the new runs reproduced every run's own campaign aggregates to ≤ 5e-11 pp, so no benchmark left any macro.
No cell met the technical-stop rule. Empty responses were ≤ 0.18% and truncation at 4,096 tokens ≤ 0.74% on every
run (≤ 0.28% on the main matrix).

## 2. F0: answer-format artifacts (frozen `format-normalized-v2`)

The rule was frozen before any hybrid output existed (spec sha256 `a229a564…`, 85 tests). It only raises scores and
is gold-blind:
* PuzzleVQA and VisualPuzzles rows where the pinned parser finds no option are re-read from the final-answer region
  only.
* TableVQABench rows without full credit are re-scored after two narrow rewrites (a single `\boxed{}` answer; empty
  `||` items dropped).

A blinded check of 80 development rows agreed with the normalized score on 80/80. Main effects on the reviewed
study's stored outputs (confirmation, 3 seeds):
* VERO's interface loss (V@P1 − V@P0) shrinks from −2.94 [−3.33, −2.56] to −0.83 [−1.21, −0.46], so formatting
  explains about 72% of it. By benchmark:
  * PuzzleVQA: −33.1 → +0.4;
  * VisualPuzzles: −13.3 → −0.6;
  * TableVQABench: −16.0 → −11.4.
* TableVQABench's remaining loss is behaviour, not format. Under P1, VERO answers the WTQ questions with several
  `||`-separated values in 47% of rows (4% natively). The pinned scorer correctly marks those wrong, and they are not
  repaired.
* M050 − T moves from +0.23 [−0.19, +0.65] to +0.62 [+0.20, +1.04]. That is descriptive; the reviewed confirmatory
  family is unchanged.

## 3. F1: component effects (primary)

Paired cluster bootstrap over 20,957 units (connected image/problem components with near-duplicate twins merged),
10,000 replicates, seed 314159, all seven models and three seeds attached to each unit. Native metrics are recomputed
in each replicate. p-values are centered bootstrap, Holm across the six-effect family.

| Effect (pp) | Official [95% CI] | Holm p | Normalized [95% CI] |
|---|---|---|---|
| g_F^T | +0.41 [0.12, 0.69] | 0.025 | +0.44 [0.15, 0.73] |
| g_L^T | **+3.47** [3.04, 3.90] | < 0.001 | +3.36 [2.93, 3.79] |
| I_T | +0.08 [−0.32, 0.48] | 0.71 | +0.07 [−0.33, 0.47] |
| g_F^V | +0.25 [−0.09, 0.60] | 0.46 | +0.28 [−0.06, 0.63] |
| g_L^V | **+1.28** [0.77, 1.80] | < 0.001 | **+3.04** [2.52, 3.55] |
| I_V | −0.24 [−0.71, 0.23] | 0.62 | −0.10 [−0.56, 0.37] |
| S(T) − S(B) | +3.95 [3.52, 4.39] | | +3.87 |
| S(V) − S(B) | +1.29 [0.74, 1.83] | | +3.22 |
| S(T) − S(TL) (frontend reversion) | +0.48 [0.19, 0.78] | | +0.51 |
| S(V) − S(VL) (frontend reversion) | +0.01 [−0.35, 0.36] | | +0.18 |

Seven-cell macro, seed mean (official / normalized):

| | B | T | V | TF | TL | VF | VL |
|---|---|---|---|---|---|---|---|
| official | 48.04 | 51.99 | 49.34 | 48.45 | 51.51 | 48.29 | 49.33 |
| normalized | 48.22 | 52.09 | 51.44 | 48.66 | 51.58 | 48.50 | 51.26 |

By category (official g_L): TRACE is largest in Charts & Tables (+5.4) and Visual Math (+4.8). VERO is largest in
Visual Math (+6.2) and Science & General (+4.0), with Puzzles & Logic −5.2 official and +4.3 normalized. Frontend
effects are ≤ 1.0 in every category. These category figures are exploratory.

Figure: `figures/component_effects.png`. Full tables: `tables/component_effects.csv`, `tables/performance_7cell.csv`,
`tables/behavior_and_format.csv`.

## 4. Interface sensitivity (P2; development screen; seeds 42–44, A5)

| Cell | P2 − P1 official [95% CI] | normalized |
|---|---|---|
| B | −2.88 [−4.14, −1.60] | −1.33 |
| T | −3.40 [−4.65, −2.11] | −1.51 |
| V | **+3.19** [2.16, 4.19] | +1.09 |
| TF | −2.55 [−3.81, −1.24] | −0.97 |
| TL | −3.39 [−4.67, −2.10] | −1.40 |
| VF | −2.67 [−4.00, −1.34] | −0.99 |
| VL | **+3.35** [2.30, 4.43] | +1.55 |

Under P2: S(V) − S(B) = +7.48 [5.87, 9.10] (normalized +5.75), with g_L^V = +7.56 and g_F^V = +0.37. TRACE's gain is
similar under both interfaces (+4.17 under P1, +3.64 under P2). The interactions are within ±0.6 under both
interfaces, with intervals including 0. The seed-42 diagnostic replicated (V: +3.29 on one seed, +3.19 over three;
B: −4.00 and −2.88). Tables: `tables/interface_3seed.csv`, `tables/template_sensitivity.csv` (seed 42).

## 5. F2c: TRACE decoder block-group reversion (P1, confirmation, three seeds)

| Checkpoint (T with … from B) | macro | R_k = S(T) − S(TR_k) | Holm (3, descriptive) | S(TR_k) − S(B) |
|---|---|---|---|---|
| TR0, layers 0–8 | 51.42 | +0.58 [0.25, 0.89] | 0.0006 | +3.37 |
| TR1, layers 9–18 | 50.18 | **+1.81** [1.42, 2.19] | 0.0003 | +2.14 |
| TR2, layers 19–27 | 51.79 | +0.21 [−0.11, 0.53] | 0.20 | +3.74 |

The normalized results are within 0.06 pp of these. R_TR1 is largest in Visual Math (3.1) and Charts & Tables (2.8).
The group sizes differ (9/10/9 blocks). Tables: `tables/f2c_reversion.csv`.

## 6. F3: norm-matched VERO

V-NM = (1 − λ)B + λV with λ = ‖Δ_T‖/‖Δ_V‖ = 0.226577 (‖Δ_T‖ = 1.9938, ‖Δ_V‖ = 8.7998; float64 over all 729 stored
tensors). After the single BF16 cast the realized update norm is 1.518, a −23.9% residual against the target; 76.7%
of stored values equal B's. λ was not retuned (A4), so V-NM is matched in coefficient, not in realized norm; its update
is 0.76 of TRACE's. Same stack and interfaces as F1; B, T, V are the F1 runs.

| P1, confirmation, three seeds | official [95% CI] | normalized [95% CI] |
|---|---|---|
| macro: B / V-NM / V / T | 48.04 / 49.05 / 49.34 / 51.99 | 48.22 / 49.25 / 51.44 / 52.09 |
| S(V-NM) − S(B) | +1.00 [0.68, 1.34] | +1.02 [0.70, 1.36] |
| S(V-NM) − S(V) | −0.29 [−0.81, 0.25] | **−2.19** [−2.71, −1.66] |
| share of VERO's gain kept, (S(V-NM) − S(B)) / (S(V) − S(B)) | 0.78 [0.51, 1.29] | **0.32** [0.22, 0.42] |
| S(T) − S(V-NM) | +2.95 [2.52, 3.37] | +2.85 [2.42, 3.26] |

Under VERO's own interface (P2, development screen, seed 42, diagnostic), V-NM keeps 0.49 [0.29, 0.70] of VERO's gain
on the official scale and 0.55 [0.31, 0.83] on the normalized one; S(V-NM) − S(V) is −4.27 [−6.27, −2.25].

Reading:
* On the official P1 scale V-NM looks as good as V (−0.29, n.s.). That reflects how small VERO's official P1 gain is,
  because its formatting failures depress it, not V-NM retaining VERO's capability. V-NM's official and normalized
  scores differ by only 0.2 (VERO's by 2.1), so the shrunken model sheds VERO's answer-format behaviour along with
  most of its gain.
* On the format-normalized scale, and under VERO's own interface, V-NM keeps about a third to a half of VERO's gain.
  Along this interpolation path, much of VERO's displacement is needed for its measured result.
* TRACE's update gives nearly four times V-NM's gain (+3.95 vs +1.00 over B, official) at a realized norm
  1.3 times V-NM's. That is a geometric
  diagnostic for these released BF16 checkpoints, not "quality per parameter moved". The two parents differ in
  training history, and equal parameter distance is not equal functional change.
* Interpretation limit: the BF16 residual means V-NM moved less than the target, so a norm-exact construction might
  keep somewhat more. Nothing here identifies the training corpus's causal role or the minimum movement needed to
  train VERO.

Tables: `tables/f3_norm_matched.csv`, `tables/f3_summary.json`; norms in `audit/f3_update_norms.json` and
`audit/build_verification_F3.json`.

## 7. Checks, deviations and incidents

* **Amendments** (all written before the outputs they govern existed):
  * A1: two P2 scorer raises (ChartQAPro empty final answer; MathVerse judge truncation), fixed by fail-closed
    fix-4/5 in a private overlay of the protocol clone.
  * A2: the F2c choice and the execution changes.
  * A3: an archive-sanitizer raise on V@P1 seed 43, fixed by fix-6.
  * A4: the F3 residual.
  * A5: the replication.
  * The fixes change only rows on which the pinned producer raises. No other run took those paths, and the shared
    clone was never modified.
  * The three rescored runs carry the overlay's code hash on their scoring records, which the ingest accepts only
    for those listed runs and only for code hashes computed from the clone states.
* **Run incidents:**
  * Allocations ended or were preempted mid-run (rc 143).
  * vLLM 0.10.2 "mm_hash" stalls left a server healthy but generating nothing. The fillers' built-in guard had a bug
    and never fired; the pools were stopped by hand and a standalone guard was added.
  * The judge endpoint hit transient read timeouts.
  * Several of this study's own fillers raced for the same card. The runner's claim arbitration refused the
    duplicates; the race no longer counts as a failed attempt.
  * Every affected run resumed with its finished rows kept. Failed attempts are recorded in each run's
    `timing.seed*.json`, and quarantined scoring outputs sit in `quarantine_*` directories.
* **Judge health:** MathVerse boxed-vs-extracted agreement is 40–89% on every screen, replication and confirmation
  run, above the 30% floor (smoke runs have too few boxed rows to gate). The judge ran at TP ≤ 2 throughout, never at the TP4 configuration that caused the earlier fault.
* **Shared cards:** cards were lent by other campaigns (AlmostAgents, WKR, WMX) and coordinated through claims.

## 8. Costs

All GPU work ran inside already-held allocations on granite and notchpeak; nothing was allocated for the study.

* **Compute:** 99 run-seeds, 167 attempts, 135.3 GPU-hours of attempt time (wall 119.0 h).
  * Of the attempts, 49 ended with a non-zero code. These include the intended generation-only handoffs to a
    judge-capable card (rc 6), placement races (rc 3), preemptions or ended allocations (rc 143), and the stalled
    pools.
  * The F1 primary and template runs alone used 54.5 GPU-hours and generated 121M tokens.
* **Storage:** 123.6 GiB of constructed weights (4 F1 hybrids, 3 F2c checkpoints, 1 F3 checkpoint) and 15.2 GiB of
  run directories, both in the local work root.
* **Judge:** Qwen3-32B scoring calls ran inside the runs (the judge-routed benchmarks of every run, plus three
  judge-only rescores).
* **CPU:** F0 and the analyses used CPU only.

## 9. Decision

**Stop.** No further component, merge or coefficient experiments, no new hybrid campaign, and no training
proposal from these results.

The follow-up's questions are answered at the resolution these checkpoints and this suite allow:
* Both recipes' measured gains transfer into the original model through the language component.
* Neither shows a measurable frontend–language co-adaptation on either scale or under either interface; the frontend
  contributes little (TRACE +0.4, VERO n.s.).
* The apparent TRACE–VERO gap under the backbone interface is mostly answer formatting plus interface dependence that
  also lives in the language component.
* Within TRACE's decoder, the measured gain depends most on layers 9–18.
* VERO's larger update is not mostly superfluous along its own direction.

Each remaining open point is either a property of evaluation (interface choice) or would need training-time
experiments that this plan excludes. More training-free surgery would refine numbers rather than change these
conclusions. A deployment-facing comparison of VERO would need a separately designed, interface-matched evaluation on
an untouched set; that is a new study, not a continuation of this one, and it is not proposed here.

## 10. Limitations

* Exploratory reuse of an already examined confirmation partition. Decoding seeds are not training replications.
  The interaction I is scale-dependent and is not an intrinsic measure of synergy.
* Component transfer is not capability localization and not evidence of training-time freezing. A failed
  transplant can reflect component mismatch rather than a useless donor.
* TRACE and VERO differ in recipe and history, not only in image provenance. These are recipe-level findings, not a
  causal synthetic-vs-natural image effect. VERO's model card does not pin its exact base revision.
* The format-normalized scale is a diagnostic beside the official score, never the benchmark score.
  TableVQABench's remaining loss and some judge-graded rows (which would need a judge to re-adjudicate) are not
  normalized.
* The P2 results come from the development screen (2,910 rows), not the confirmation partition.
