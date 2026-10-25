# TRACE program-verifiable certificates: campaign findings

Status as of 2026-10-24. Every planned training run and evaluation is complete. All numbers below come from the
generated analysis outputs under `results/final/` (paths are given per section). Each output directory has a
`run.sh` that regenerates it from the raw per-row files. During the campaign, each analysis was also recomputed
independently by a separate reviewer agent, and every number in this document was checked against its source file
in two passes.

**Arms.** All arms share one output schema (annotation + certificate + answer), one SFT warm start, and identical
rollout settings. Only the RL reward differs:

- **S**: structured SFT only (the shared warm start).
- **A**: S + outcome RL (answer-only reward).
- **G**: S + grounding RL (answer + annotation grounding).
- **P**: S + grounding + semantic-certificate RL. This is the method under test.
- **P_gated / G_gated**: answer-gated variants of P and G (stage 4). The grounding or certificate term pays only when the answer is right: R = 0.95·A·[0.8 + 0.2·B] + 0.05·F, with B as in P or G.

**Scale and seeds.** At 3B (Qwen2.5-VL-3B), A, G and P have 3 training seeds each; S is a single checkpoint; the
gated pair has 1 seed. At 7B (Qwen2.5-VL-7B), G and P have 1 seed each. Every eval uses decoding seeds 42-44
unless noted.

**Metrics.**

- **A**: answer accuracy.
- **G**: annotation grounding score (IoU-based match of boxes and points).
- **M**: certificate semantic truth (operands, bindings and intermediate results checked against the scene).
- **F**: strict format validity.
- **joint**: answer right, strict format valid (F = 1) and certificate fully verified (M = 1).

Where an arm and a metric share a letter, the metric is named in words (e.g. "arm A loses grounding").

**Splits.**

- **E1**: new scenes from the trained programs.
- **E2**: held-out program combinations or render grammars, excluded from both SFT and RL.
- **E1-dev**: the RL validation split. It is also used for SFT selection, so results on it are descriptive.
- **External**: trace_eval_v1, 24 benchmarks with no oracle.

---

## 1. Bottom line

1. **The main hypothesis is not supported.**
   - At 3B with 3 training seeds, adding the semantic-certificate reward (P) to grounding RL (G) shows no detectable difference in answer accuracy under the pre-registered rule.
   - New scenes (E1): +0.06 pp [-0.93, +1.09].
   - Held-out cells (E2: program combinations or render grammars): -1.48 pp [-3.56, +1.03]. Seeds 2 and 3 individually favour G, and the E2 deficit is spread across adapters rather than coming from one cell.
   - With the wider shared-instance CIs (§2), the data cannot rule out effects of about ±1.2-1.3 pp on E1, or anything on E2 from a 3.8 pp loss to a 1.1 pp gain.
   - Pre-registered go/no-go 2 is not met. The P vs A and G vs A secondary tests also make no claim.
2. **There is no transfer benefit either.** On the 24 external benchmarks, neither P - G macro is a claim under the pre-registered rule.
   - Natural images: -0.18 points, CI [-0.36, +0.00], Holm p 0.112, all 3 seeds negative. This lean towards G depends on pooling decoding seeds 43/44; seed 42 alone gives +0.04. It also comes mostly from one benchmark (CountBenchQA).
   - Charts/diagrams/synthetic is flat: +0.02 [-0.13, +0.17].
   - External scores move little during RL. At 3B every arm gains 0.26-0.44 points over S on natural images, and nothing on charts/diagrams. The largest move is G7 at 7B (+1.07 on natural images).
3. **What RL does do at 3B (all arms, on top of SFT; descriptive).**
   - Raises in-distribution (E1) accuracy by 2.3-3.0 pp over the single S checkpoint. SFT dev accuracy was still rising at its last save, so part of this may be what longer SFT would also give (§8).
   - On E1, the two grounding-reward arms (G, P) cut grounding errors by about 5 pp relative to outcome-only RL, and by about 4-5 pp relative to S. The cut comes from the chess control and table adapters, not rpg.
   - On E2 they are also about 3 pp below A, but only because A adds grounding errors over S (+2.9 pp); G and P do not fall below S.
   - Computation errors don't change at 3B.
   - Arms A and P lose about 1.8 pp on E2, mostly in the held-out chess cell. At 7B, P7 instead gains 2.3 pp over S7 on E2.
4. **P's small certificate-quality edge over G is unconfirmed.**
   - On E1, M is +0.55 and joint +1.03, with all 3 seeds in the same direction. These are descriptive, unadjusted cells: M and joint are not in a DESIGN 8 family.
   - Their CIs exclude 0 with compare_cert's bootstrap and just include 0 with the shared-instance bootstrap (§2).
   - The edge does not appear on E1-dev: at step 300 P - G on M is -0.05, and at step 200 G is ahead (M -1.22, shared CI [-2.28, -0.14], all 3 seeds).
   - At 7B it is absent on E1 (P7 - G7 M -0.36) but present on E2 (+1.25 [+0.27, +2.25]; one seed).
   - The matched answer-gated pair (one seed) shows a larger E1 gap (M +1.42, joint +2.32), but mostly because G_gated scored low (§3).
5. **7B (one seed) does not change the picture.**
   - P7 - G7 is -1.0 pp on E1 (the bootstrap CI just excludes 0; the pre-registered McNemar test is not significant, Holm p 0.076).
   - On E2 it is +2.9 pp [+1.4, +4.5] (Holm p 0.011, significant at the instance level only). With one training seed, neither split is a DESIGN 8 claim.
   - 86% of the E2 gain comes from one control cell (chess piece counting, +10.1 pp). That value lies outside the 3B seed range for the cell, but with one 7B seed it cannot be separated from 7B seed variation.
6. **Certificate SFT lowers accuracy on TRACE's own released validation set under its answer-only prompt** (one SFT run per size).
   - The certificate SFT model is 4-7 pp (3B) and 10-12 pp (7B) below the base model.
   - Under that prompt the certificate-trained models answer in about 7 tokens (base about 215), so part of the drop may be lost reasoning output rather than lost knowledge.
   - The released answer-only TRACE checkpoints are 13-18 pp above base. They were RL-trained on the same TRACE programs this set is drawn from, so this is context rather than a like-for-like comparison.
7. **The infrastructure works as designed.**
   - The verifier passes all oracle and corruption checks, and go/no-go 1(b) passes.
   - Go/no-go 1(a) passes narrowly: 12.2% against a 10% threshold, Wilson CI 9.3-15.9%. It passes only because of rpg; without rpg the share is 4.3%.
   - Training arms are matched exactly in data, init, steps and rollouts per step. Generated rollout tokens come out nearly equal (3B: A 79.10 M, G 79.12 M, P 79.10 M).
   - No zero-shot model writes verifiable certificates with the prompts tested (best joint rate 4.7%), so these models needed certificate SFT here.
   - On answer accuracy alone, zero-shot Qwen3.5-9B beats every trained 3B arm on E2 under the same certificate prompt (75.6% vs 68.6-70.4%). It does not on E1 (73.5% vs 83.9-87.3%).

---

## 2. Pre-registered decisions (DESIGN 8)

| Criterion | Result | Source |
|---|---|---|
| Go/no-go 1(a): S-checkpoint E1-dev numeric rows (one greedy pass) with G >= 0.9 and M < 1 make up at least 10% | **Pass, narrowly**: 47/384 = 12.2% (Wilson 95% CI 9.3-15.9%, which includes values below 10%). Descriptively, a G >= 0.95 cut gives 9.6%, and so does the 7B S checkpoint. It rests on rpg (36/128), whose G is 1.0 on every row; without rpg the share is 4.3%. Spearman(G, M) = 0.21 pooled (p 3e-5), which mixes adapter levels: within table it is 0.58, and it is undefined for lever (M constant) and rpg (G constant). | `results/final/training/training.json` `go_no_go.1a` |
| Go/no-go 1(b): at least 3 content-corruption types per numeric adapter (table, lever, rpg) keep G (\|dG\| < 0.01) and lower M on every applicable row | **Pass**: table 9, lever 8, rpg 18 (72,692 corruptions, 0 unmet expectations) | `go_no_go.1b`; `results/section5_cpu_v2_gate/` |
| Go/no-go 2 / primary family: P - G on A, E1-test and E2-test, 3 training seeds. A claim needs the two-level bootstrap CI to exclude 0 and 2 of 3 seeds to agree in sign; Holm adjusts only the per-seed McNemar p values. | **Not met. No claim either way.** | `results/final/cert_3b/cmp_main.json` `families` |
| Secondary family: P - A and G - A on A | **No claim** on any of the 4 tests | same |

### Primary family, 3B (P - G on A)

Seeds 1-2 ran on 2 x H200 NVL and seed 3 on 2 x RTX PRO 6000, so the two-level CI mixes seed and hardware
variation.

| Split | Mean of 3 seeds | 95% CI (two-level) | Per-seed delta (s1 / s2 / s3) | Per-seed Holm p (McNemar, decoding seed 42) | Seeds agreeing |
|---|---|---|---|---|---|
| E1-test (2,000 rows) | +0.06 pp | [-0.93, +1.09] | +0.90 / -0.65 / -0.08 | 0.439 / 0.481 / 0.703 | 1/3 |
| E2-test (1,200 rows) | -1.48 pp | [-3.56, +1.03] | +1.17 / -3.14 / -2.47 | 0.664 / 0.018 / 0.00089 | 2/3 |

On E2, seeds 2 and 3 individually favour G, but the pooled CI includes 0. Neither "P > G" nor "G > P" is
supported. Per DESIGN 8 the result is reported as is and the claim is narrowed.

Descriptively, the E2 deficit is spread over rpg (-3.37 [-7.48, +1.52]) and the chess control (-2.07 [-6.93, +2.81]).
Without the chess control it is +0.67 / -1.85 / -2.67 by seed. So, unlike the 7B E2 gain, it does not come from one
cell.

### Secondary family, 3B (on A)

| Pair | E1-test | E2-test |
|---|---|---|
| P - A | +0.71 [-0.13, +1.56], 3/3 seeds | -0.02 [-1.44, +1.52] |
| G - A | +0.65 [-0.62, +1.76] | +1.46 [-0.05, +2.85], 3/3 seeds (borderline) |

### CI sensitivity: shared-instance bootstrap

The pre-registered script resamples items independently within each training seed, although all seeds are scored
on the same items. Re-estimating with one shared item resample per replicate widens every CI, by a median of
1.24x (range 1.04-1.51). Source: `results/final/cert_3b_shared_ci/shared_ci.md`.

For the two P - G certificate-quality cells (M and joint on E1), the training seeds share little instance effect
(ρ 0.08 and 0.13). There the shared bootstrap over-counts the seed x instance interaction and is conservative:
- Shared bootstrap SDs: 0.29 and 0.62 pp.
- Moment-based standard errors: 0.20 and 0.45 pp.
- compare_cert's SDs: 0.24 and 0.50 pp.

Both intervals are defensible for these descriptive cells.

- **The primary and secondary families are unchanged:** still no claim.
- **The RL-over-SFT (X - S) and RQ3 grounding-error intervals still exclude 0** (descriptive, unadjusted).
  - X - S on E1: A +2.32 [+1.18, +3.46], G +2.97 [+1.49, +4.33], P +3.03 [+1.89, +4.18].
  - X - S on E2: A -1.75 [-3.53, -0.01] and P -1.77 [-3.37, -0.15]. The A - S result is borderline: its upper end stays between -0.01 and -0.06 across bootstrap RNG seeds.
  - Grounding-error rate, P - A on E1: -5.44 [-7.81, -3.08].
- **Two descriptive cells no longer clear 0:** P - G certificate truth M on E1, +0.55 [-0.02, +1.13], and P - G joint on E1, +1.03 [-0.18, +2.28]. All 3 seeds still point the same way.

---

## 3. In-domain results at 3B (RQ1, RQ3)

Sources:
- `results/final/cert_3b/`: `REPORT.md`, `cmp_main.json`, `extras_main.json`, `cmp_gated.json`, `extras_gated.json`.
- `results/final/cert_3b_shared_ci/shared_ci.json`: the shared-instance CIs.

Everything in this section outside the two families above is descriptive and unadjusted.

### Arm means (mean over training seeds, decoding seeds 42-44)

| Arm | E1-test A | E1-test M | E1-test joint | E2-test A |
|---|---|---|---|---|
| S | 83.85 | 87.08 | 74.73 | 70.39 |
| A | 86.17 | 87.03 | 75.66 | 68.64 |
| G | 86.82 | 90.33 | 79.91 | 70.10 |
| P | 86.88 | 90.88 | 80.94 | 68.62 |
| P_gated (1 seed) | 87.30 | 91.22 | 81.43 | 69.53 |
| G_gated (1 seed) | 86.65 | 89.80 | 79.12 | 70.39 |

F is at least 98.78 for S/A/G/P (seed means; the lowest single training seed is 98.03, G on E2-test). The gated
arms reach only 97.6 / 97.1 on E2-test.

- **The certificate prompt carries the accuracy.** With the answer-only prompt on the same E1 rows, A is only about 25-27 for every arm.
- **RL improves on SFT in-distribution, but arms A and P lose on held-out programs.**
  - E1-test, X - S on A: A +2.32, G +2.97, P +3.03, all 3/3 seeds. The CIs are in §2.
  - E2-test: A -1.75 and P -1.77 (both 3/3); G -0.29 [-2.62, +1.59] (shared-instance [-2.89, +2.03]).
  - The E2 loss sits in the held-out chess cell.
- **P's certificates may be slightly better than G's in-distribution, but this is unconfirmed.**
  - E1-test: M +0.55 and joint +1.03, with all 3 seeds in the same direction. These are descriptive cells, outside any DESIGN 8 family.
  - compare_cert's CIs exclude 0 ([+0.09, +1.03] and [+0.08, +2.04]); the shared-instance CIs just include 0 ([-0.02, +1.13] and [-0.18, +2.28]).
  - E2-test: M -0.24 [-0.96, +0.49], joint -0.45 [-1.81, +1.02].
- **What changes (RQ3): grounding errors fall; computation errors don't move.** Error class is the scorer's first failing stage.
  - E1-test grounding-error rate: S 13.53%, A 14.18%, G 9.32%, P 8.74%.
  - P - A -5.44 pp [-7.72, -3.16] and P - S -4.79 pp [-5.53, -4.01], both 3/3. The shared-instance CIs are [-7.81, -3.08] and [-5.92, -3.67].
  - On E2-test, P - A is -2.70 and G - A -3.60. That is because A's grounding-error rate rises (A - S +2.90 [+1.66, +4.31]); P and G do not fall below S (P - S +0.19 [-0.88, +1.23], G - S -0.70 [-1.86, +0.53]).
  - Operand errors move less (P - S -1.16 [-2.28, -0.04]). Computation errors do not move (P - S -0.27 [-0.91, +0.35]).
  - No P - G failure class meets the claim rule.
  - Descriptively, P has fewer right-answer-with-untrue-certificate rows on E1 (-0.98 [-1.68, -0.33], 3/3), and its answer-only accuracy is higher (+0.67 [+0.08, +1.29]).
  - Its operand and computation stage-truth scores are also higher (+0.43, +1.05). These gated scores inherit upstream operand errors, so they do not show better arithmetic: the computation error-class rate is +0.69 [-1.01, +2.09].
  - None of these was re-checked with shared-instance resampling.
- **Verbosity is unchanged on E1-test.** Certificate-prompt completions are about 173 tokens for every arm (P - G -0.29 tokens [-0.86, +0.23]), and nothing is truncated. On E2-test the grounding arms write a few more tokens than A (G - A +5.5, P - A +3.6, both 3/3), and P - G swings by seed (+8.0 / -7.4 / -6.3).
- **The table-adapter signal from the stage-1 interim does not replicate.** The interim used seed 1 only: P - G on table A was +5.53 [+1.67, +9.47]. Over 3 seeds it is +0.40 [-3.60, +5.27] on E1, with per-seed values +5.53 / -3.00 / -1.33.

### Answer-gated variant (stage 4; seed 1, both arms on 2 x RTX PRO 6000; descriptive)

- P_gated - G_gated, E1-test:
  - A +0.65 [-0.52, +1.85].
  - M +1.42 [+0.77, +2.07] and joint +2.32 [+1.03, +3.62].
  - Grounding-error rate -1.97 [-2.98, -0.93] and operand-error rate -1.60 [-2.37, -0.87]. The computation-error rate rises, +1.25 [+0.40, +2.12].
  - The M/joint gain comes from the table and rpg adapters.
- E2-test: A -0.86 [-2.50, +0.78], M +0.54 [-0.42, +1.53], joint +0.00 [-1.56, +1.56].
- **There is no matched ungated reference.**
  - On E1, Pg - Gg on M (+1.42) is larger than the ungated P - G at seed 3 (same hardware, +0.32 [-0.32, +0.98]) or at seed 1 (same data order, +0.91 [+0.28, +1.52]).
  - Against seed 3, most of the gap comes from G_gated scoring below G3 (M -0.76 [-1.36, -0.16]), not from P_gated scoring above P3 (+0.34 [-0.31, +0.99]).
  - On A, the ungated seed-3 P - G is -0.08 [-1.28, +1.13] on E1 and -2.47 [-4.00, -1.00] on E2.
  - Both references confound gating with hardware or seed, and no interaction was tested. On one seed, this does not show that gating favours P.

---

## 4. 7B replication (one training seed; CIs cover instance sampling only)

Source: `results/final/cert_7b/` (`cert_7b.md`, `cert_7b.json`).

| | E1-test A | E2-test A |
|---|---|---|
| base Qwen2.5-VL-7B (zero-shot, cert prompt) | 20.28 (1,217/6,000) | 20.28 (730/3,600; equal to E1 by coincidence) |
| legacy TRACE-7B (zero-shot, cert prompt) | 33.32 | 33.19 |
| S7 | 89.47 | 72.69 |
| G7 | 92.48 | 72.08 |
| P7 | 91.47 | 75.03 |

- **Primary contrast P7 - G7 on A.**
  - E1-test: -1.02 pp [-1.98, -0.05], McNemar p 0.076, Holm p 0.076. Not significant: the all-seed bootstrap CI excludes 0, but the pre-registered seed-42 McNemar test does not reject.
  - E2-test: +2.94 pp [+1.36, +4.50], Holm p 0.011.
  - With one training seed this is not a DESIGN 8 claim.
- **86% of the 7B E2 gain comes from the selected_set control cell (chess piece counting).**
  - That cell alone is +10.11 pp [+5.44, +14.78]. The numeric adapters give +0.56 [-0.93, +2.04] on E2 (post hoc).
  - Across 3B training seeds the same cell's P - G is +2.67 / -7.00 / -1.89 (SD 4.84). The 7B +10.11 lies 7.44 pp above that range, so 3B seed variation does not cover it.
  - With one 7B seed, though, it cannot be separated from 7B seed variation. A second 7B seed is needed before any claim.
- **RL over SFT at 7B.**
  - P7 - S7: E1 +2.00 [+0.93, +3.07], E2 +2.33 [+0.69, +3.94].
  - G7 - S7: E1 +3.02 [+1.95, +4.07], E2 -0.61 [-2.22, +1.00].
- **RQ3 at 7B, P7 - G7 rate differences.**
  - E1: computation errors -1.93 pp, grounding errors +1.50 pp.
  - E2: grounding -2.14, computation -1.56, operand +1.03.
- **P7 shows no reliable early lead over G7 on E1-dev.**
  - P7 - G7 on E1-dev at steps 100 / 200 / 300 is +2.02 [+0.33, +3.71] / -0.07 / -1.11 pp (McNemar p 0.185 / 1.00 / 0.248).
  - At step 100 the bootstrap CI excludes 0 but McNemar does not, and there is no correction across steps.

---

## 5. External transfer (RQ2, trace_eval_v1)

Source: `results/final/external/compare_ext.md` and `compare_ext.json`.

**Setup.**
- 24 benchmarks and 32,805 generated rows (`rows_expected` in each run's ext_eval summary), with no certificates or oracles at inference. That is 31,590 scored items, because WeMath is scored over 525 problem groups.
- Items are paired between arms. Every G/P training seed pools decoding seeds 42-44.
- Multi-seed comparisons use a two-level bootstrap over training seeds and items, resampling within each benchmark.
- Comparisons with a single run on both sides ("one run") cover items only. A single run on one side (S, legacy-3B) is paired with each training seed of the other side and gets the two-level bootstrap.
- This external bootstrap resamples items independently per training seed, the pattern §2 found too narrow in-domain. It was not re-checked here, and the conclusions below are "no claim" or descriptive either way.

**Pre-registered external family (P - G, Holm across the two macros): no claim.**

| Macro | P | G | P - G | 95% CI | p (Holm) | Per-seed delta (s1 / s2 / s3) |
|---|---|---|---|---|---|---|
| Natural images (9 benchmarks) | 49.36 | 49.54 | -0.18 | [-0.36, +0.00] | 0.056 (0.112) | -0.09 / -0.26 / -0.18 |
| Chart / diagram / synthetic (15) | 30.45 | 30.44 | +0.02 | [-0.13, +0.17] | 0.828 (0.828) | +0.02 / -0.08 / +0.11 |

On natural images all three seeds lean slightly towards G, but the effect is under 0.2 points and not a claim. It
depends on pooling decoding seeds 43/44: seed 42 alone gives P - G natural +0.04 [-0.29, +0.38], with 2/3 seeds
agreeing. It also comes mostly from CountBenchQA (-1.19 [-2.10, -0.30], 3/3; descriptive, unadjusted).
The secondary family (P - A, G - A) has no claim either: every delta is within ±0.06 points.

**Descriptive context (3B unless noted).**

- **RL over SFT helps a little on natural images and does nothing on charts/diagrams.**
  - Natural: A - S +0.42 [+0.09, +0.76], G - S +0.44 [+0.17, +0.67], P - S +0.26 [+0.04, +0.47].
  - Chart/diagram/synthetic: all within ±0.1.
  - These X - S CIs pair one shared S run with every seed, the case §2 found most affected by the narrow bootstrap.
- **At 3B, certificate SFT shifts transfer from charts/diagrams to natural images, compared with base.** At 7B, S7 is below base on chart/diagram/synthetic and borderline on natural images (-0.84 [-1.70, -0.00], sign-flip p 0.053; one run each, seed 42).
  - S - base-3B (one run each; CI over items only): natural +1.37 [+0.55, +2.17], chart/diagram/synthetic -2.67 [-3.42, -1.90].
  - The released answer-only TRACE-3B is ahead of P on both groups: natural -0.61 [-1.10, -0.10], chart/diagram/synthetic -6.33 [-6.79, -5.88].
- **Part of the chart/diagram gap is an output-format artifact (DESIGN 10).**
  - At decoding seed 42, the seed of the base and legacy runs, the official PuzzleVQA / VisualPuzzles letter extractor cannot read the answer from 45-59% / 42-49% of the certificate-trained 3B models' responses (scored wrong). At seed 43 this rises to about 95% / 76-85%.
  - For base-3B it fails on 3.5% / 11.7%, and for legacy-3B on 2.5% / 5.6%.
  - Without those two benchmarks, S - base on the chart/diagram/synthetic group shrinks to -1.19 [-2.01, -0.37], and P - legacy to -4.68 [-5.16, -4.20].
  - Within P - G, the format rates differ by less than 1 pp, except these two extractors: up to +4.3 / -3.5 pp with decoding seeds pooled, and up to 5.5 pp at seed 42 alone.
  - DESIGN 10 flags 13 rate differences above 2 pp within the family and secondary pairs. The largest are in P - A and G - A at training seed 2 (PuzzleVQA -13.2 and -8.5 pp).
- **7B (one run per model):**
  - P7 - G7: natural -0.66 [-1.14, -0.20], chart/diagram/synthetic -0.48 [-1.05, +0.08].
  - G7 - S7: natural +1.07 [+0.55, +1.58].
  - S7 - base-7B: natural -0.84 [-1.70, -0.00], chart/diagram/synthetic -1.72 [-2.49, -0.95].
- **Answer-gated pair (one seed):** P_gated - G_gated is natural +0.02 [-0.43, +0.49] and chart/diagram/synthetic -0.19 [-0.55, +0.16].
- **Decoding seed matters a lot for absolute levels.**
  - In all 8 3B models that have both seeds, seed 43 lowers the chart/diagram/synthetic macro by 2.24-2.98 points (mean -2.60).
  - Every request uses the decoding seed as its sampling seed, so near-identical models draw near-identical samples.
  - The biggest per-benchmark swings are PuzzleVQA -16, VisualPuzzles -10, MathVista -8 and PhyX +8. The PuzzleVQA / VisualPuzzles drops coincide with the letter-extractor failures above.
  - Matched-seed pairs cancel this. Every seed-42-only level or single-run comparison (base, legacy, gated, 7B, q25, A seeds 2-3) is seed-sensitive.

**Transfer over training (trace_eval_v1_q25, 8,207-row subset, decoding seed 42; descriptive).**
Step 0 is S; the step-300 values are the full runs restricted to the q25 rows.

| Arm (3B, mean of 3 seeds) | natural @ 0 / 100 / 200 / 300 | chart/diagram/synthetic @ 0 / 100 / 200 / 300 |
|---|---|---|
| A | 49.81 / 49.42 / 49.63 / 49.89 | 31.87 / 31.98 / 31.52 / 31.46 |
| G | 49.81 / 49.78 / 49.89 / 49.88 | 31.87 / 31.90 / 31.83 / 31.76 |
| P | 49.81 / 49.49 / 49.51 / 50.05 | 31.87 / 31.86 / 31.53 / 31.75 |

- **External scores barely move during RL.** Arm means stay within about ±0.4 points of S; single training seeds stay within about ±0.8.
- **No 3B P - G or P - A difference has a CI excluding 0 at any step.**
- **The noise floor is comparable to these differences.**
  - Re-running the same base model's q25 rows gives deltas of -0.36 / -0.28 (3B) and +0.18 / +0.04 (7B), with about 91-92% identical item values.
  - Those re-runs include new generation, a second judge pass and an older scorer build.

**Scorer provenance.**
- Full suite: 24 runs use the old code hash 16d37ed061fd and 11 use the fix-3 hash d9319e7db480.
- q25: the 26 step-100/200 runs use d9319e7db480.
- The 2 base-model q25 probes carry 221335f51667, a build from before fix-2. fix-2 scores MME-Reasoning official-evaluator exceptions as incorrect, 0-5 rows per run. Their regeneration-yardstick deltas therefore also include a scorer-version difference.
- fix-3 scored 4 rows incorrect across the external-suite runs:
  - one MathVerse row each in G seed 3 and P seed 3 (decoding seed 44);
  - the 7B P MME-Reasoning row;
  - one MathVerse row in A seed 2 step 100 (q25).

---

## 6. Breadth cost on TRACE's released validation set

Source: `results/final/cert_3b/trace_val.json` and `results/final/cert_7b/cert_7b.json` `trace_val`.

**Setup.** Answer-only prompt, 2,000 rows (1,000 tasks, 11 domains) per decoding seed, seeds 42-44 pooled (6,000
rows per model). Rows without a judge verdict ("judge-pending") count as wrong, so each accuracy is a lower bound.
Where a range is given, it covers every possible outcome for those rows.

| Model | 3B accuracy (lower bound; pending rows) | 7B accuracy (lower bound; pending rows) |
|---|---|---|
| base Qwen2.5-VL | 24.45 (189 pending; upper 27.60) | 33.55 (111 pending; upper 35.40) |
| legacy TRACE (answer-only RL) | 40.52 (31) | 51.93 (5) |
| S | 20.17 (29) | 23.82 (12) |
| A | 20.58 / 20.05 / 20.82 (seeds 1/2/3; 20 / 20 / 22 pending) | - |
| G | 18.98 / 19.75 / 19.42 (21 / 20 / 20) | 23.93 (14) |
| P | 19.50 / 19.62 / 19.38 (22 / 24 / 24) | 23.10 (13) |

- **Certificate SFT costs breadth on TRACE's own tasks.**
  - 3B: S - base is between -7.43 [-9.25, -5.62] and -3.80 [-5.58, -2.02], depending on how pending rows resolve.
  - 7B: S7 - base7 is between -11.58 and -9.53 pp (official lower bounds: -9.73 [-11.82, -7.68]).
  - Either way it is negative.
  - Under the answer-only prompt the certificate-trained models answer in about 7 tokens (base about 215, legacy 240-320). Part of the cost may be lost reasoning output rather than lost knowledge, and these outputs do not separate the two.
- **The legacy answer-only TRACE checkpoints are far ahead:** 3B legacy - base is +12.9 to +16.6 pp, and 7B +16.5 to +18.5 pp. This set is in-distribution for them: it is drawn IID from the same TRACE programs they were RL-trained on (DESIGN dataset table). So the gap is context, not a like-for-like breadth comparison.
- **On top of S, RL moves TRACE-validation accuracy by about 1 pp or less at 3B.**
  - The grounding arms point slightly lower: G - S ranges from -1.27 to -0.44 and P - S from -1.15 to -0.28.
  - A does not: A - S ranges from -0.17 to +0.66.
  - No RL-over-S direction holds under every judge outcome.
  - On TRACE's own 2,048-token protocol (decoding seed 42 only, 2,000 rows), G < A under every judge outcome ([-1.60, -0.95]). With seeds 42-44 pooled under the answer-only prompt the range is [-1.44, -0.76], and the direction is not established under every outcome.
- **7B:** G7 - S7 is +0.12 [-0.63, +0.87] and P7 - S7 -0.72 [-1.50, +0.07]. Separately, P7 - G7 is -0.83 [-1.55, -0.15] on official lower bounds and between -1.07 and -0.62 under any resolution of the pending rows (one training seed, instance-sampling CI).

---

## 7. Zero-shot models and the certificate format (Section 5)

Source: `results/final/section5/section5_zs.md`. The trained-arm values used for comparison come from
`results/final/cert_3b/REPORT.md` and `results/final/cert_7b/cert_7b.md`.

- **No zero-shot model writes verifiable certificates with the prompts tested.**
  - Six models were tested: Qwen2.5-VL-3B/7B, TRACE-3B/7B, Qwen3.5-4B/9B.
  - The best joint rate is 4.7% (Qwen3.5-9B, E2-test), and the best M is 0.090 (Qwen2.5-VL-7B).
  - Only 1,020 of 57,600 certificate-prompt test rows have a right answer with a fully true certificate.
- **On answer accuracy alone, zero-shot Qwen3.5-9B is strong on E2.**
  - It scores 77.6% on E2 with the answer-only prompt and 75.6% with the certificate prompt.
  - That is above every trained 3B arm (68.6-70.4%) and level with P7 (75.0%). That last gap is untested, and Qwen3.5-9B writes about 1,300 tokens per answer against about 180 for P7.
  - On E1 the order reverses: 77.4% / 73.5%, against 83.9-87.3% for the trained 3B models and 91.5% for P7.
- **The certificate prompt has lower zero-shot accuracy than the answer-only prompt in all 12 model x split point estimates**, by 1.3 to 11.2 pp. 11 of the 12 CIs exclude 0, and 6 of 12 survive Holm (McNemar, seed 42). TRACE-7B pays the most: -11.2 pp on E1, mostly on the lever task.
- **Correct answers almost never come with a true certificate.** P(M < 1 | A = 1) is 87.5% to 99.7%. Grounding is the first failing stage in 65-98% of rows.
- **Qwen3.5's grounding scores are frame-confounded.** It answered in source pixels on lever but in 0-1000 normalised coordinates (rel1000, the frame it was prompted in) on rpg, so compare it with the other families on A only. Its 4.7% best joint rate carries the same confound.
- **The verifier itself is sound.**
  - Oracle certificates round-trip on 3,712/3,712 rows.
  - All 72,692 corruptions behave as expected.
  - Median verifier time is 0.92 ms per row.
  - For the zero-shot models only, moving the identity IoU threshold from 0.5 to 0.3 or 0.7 changes E1-dev M by -1.6 to +3.4 pp. This sensitivity was not run on the trained checkpoints.

---

## 8. Training, matching and compute

Source: `results/final/training/` (`training.md`, `training.json`).

- **Matching (DESIGN 7.3) holds.**
  - All 5 matched sets agree on every training input: config, SFT init, data and store hashes, GPU type x 2, and 300 steps x 64 prompts x 8 rollouts = 153,600 rollouts per run.
  - Step-1 rollouts are identical within each set.
  - The only differences are at the provenance level: seed-2 A ran a different `train_rl.py` revision than G/P with an identical resolved config, and at seed 3 A ran alone on its node while G and P shared one.
- **Compute.**
  - RL: 315.6 GPU-h over 13 runs. SFT: 5.0 GPU-h including dev evaluation (4.1 GPU-h of training).
  - 3B on 2 x H200 NVL: about 9.5 h per run. 3B on 2 x RTX PRO 6000 Max-Q: about 14.7 h. 7B on 2 x H200: about 13.5 h.
- **Reward curves**, 3B mean of seeds, first-10 -> last-10 train batches.
  - G and P raise grounding (88.4-88.5 -> 91.7-91.8) and M (86.2-86.3 -> 90.2-91.0). A raises answer reward (83.3 -> 87.8) but not grounding.
  - Zero-variance prompt groups, which give no GRPO signal, rise from 66% to 87% for A and from 28-31% to 48-51% for G/P. These are measured under each arm's own reward, so the level gap mostly reflects how coarse the answer-only reward is.
- **E1-dev at step 300 vs S, 3B (descriptive).**
  - Both grounding arms gain M (+3.0-3.1 pp) and grounding (+2.7-2.9 pp); arm A loses grounding (-1.3 pp).
  - Accuracy gains (+1.3 to +1.5 pp for G/P) do not exclude 0 under shared-instance resampling.
- **SFT.**
  - Both sizes selected step 564. The first save that qualifies under the pre-registered rule is also the last save (end of epoch 3).
  - Dev A was still rising: 3B went 80.1 -> 85.9 -> 87.1 over the last three saves. Part of the RL-over-SFT gain may therefore be what further SFT would also have given; no longer-SFT control exists.

---

## 9. Caveats and deviations the reader should know

- **Hardware is nested in training seed at 3B.**
  - Seeds 1-2 ran on 2 x H200 NVL; seed 3 and the gated pair ran on 2 x RTX PRO 6000 Max-Q. Arms are matched within each seed, but the two-level CIs mix seed and hardware variation.
  - The gated pair shares seed 1's data order and rollout seed but seed 3's hardware. Neither ungated pair is a matched reference; only P_gated - G_gated is matched.
  - Only 89.8% of step-1 rollouts are identical across the two GPU types at the same seed.
- **One training seed at 7B and for the gated pair.** Their CIs and p values reflect instance sampling only. DESIGN 8's claim rule (training-seed CI plus 2 of 3 seeds agreeing) cannot be applied.
- **Bootstrap CI method.**
  - The pre-registered `compare_cert.py`, and `compare_ext.py` for the external suite, resample test items independently within each training seed, although every seed is scored on the same items. This can make pooled CIs too narrow, especially for X - S with a shared S.
  - The primary and secondary families are unaffected in their conclusions: neither CI excluded 0.
  - The in-domain sensitivity check is in §2. The external CIs were not re-checked.
- **Scorer change (post-release fix-3, 2026-10-23).**
  - **The problem.** Two transfer evals could never finish. In each, one benchmark row's judge output stayed unusable after all 5 attempts, and the scorer then failed the whole benchmark. Because judge outputs are cached, every retry failed the same way.
    - Three later runs hit the same failure and were scored under fix-3 from the start: G seed 3 and P seed 3 at decoding seed 44, inside the pre-registered P - G family with one MathVerse row each, and A seed 2 step 100 on q25.
    - A seed 3, MathVerse: on the fix-3 rescoring the judge was re-run and resolved the row, so fix-3 scored no row of this run.
    - 7B P, MME-Reasoning: a degenerate 4,096-token answer whose extraction never fits. fix-3 scored it incorrect.
  - **The fix.** fix-3 scores such a row as incorrect instead, which is what upstream VLMEvalKit does.
  - **Provenance.**
    - fix-3 changed the scorer code hash from 16d37ed061fd to d9319e7db480.
    - Carrying the new hash: A seed 3 and 7B P (rescored), G_gated, the seed-43/44 runs of G/P training seeds 2-3, and the 26 step-100/200 q25 runs.
    - The 24 full-suite runs scored earlier keep the old hash and were not rescored. That changes nothing: a run that completed under the old scorer cannot contain a row fix-3 would score, because such a row would have failed its benchmark.
  - **Size of the effect.**
    - The fix-3 rule only changes rows whose judge output stayed unusable. Across the campaign it scored 4 rows (§5).
    - Rescoring re-runs the Qwen3-32B judge, so the two rescored runs can also differ by judge noise on other judge-scored rows.
    - The 7B P fix-3 row (1 of 1,188 MME-Reasoning items) moves its chart/diagram/synthetic macro by at most about 0.006 points.
- **E1-dev is not a test split.** It is the RL validation split and was used for SFT selection and go/no-go 1(a), so step curves on it are descriptive.
- **TRACE-validation accuracies are bounds** until the remaining judge-pending rows are judged. Base 3B has 189 and base 7B 111 of 6,000 pending (upper bounds 27.60 and 35.40). Both are under-stated more than the trained and legacy models, which have 5-31 pending.
- **q25 on intermediate checkpoints was added.** DESIGN 6 asks for trace_eval_v1_q25 on intermediate checkpoints; DESIGN 7.4 says steps 100/200 are evaluated on E1-dev only. We ran q25 for all 26 step-100/200 checkpoints and report it as descriptive.
- **Decoding seeds 43/44 for G and P were extended to training seeds 2-3.**
  - DESIGN 7.4 says "G and P also 43/44" for the step-300 models, but only seed 1 had them. Every G/P training seed now has external seeds 42-44.
  - S and A seed 1 also have them, beyond the design. So P - A and G - A pool decoding seeds 42-44 at training seed 1 but use seed 42 only at seeds 2-3.
- **Go/no-go 1(a) passes only because of rpg.** rpg's G is 1.0 on every row; without it the share is 4.3%.
- **Operational incidents, all resolved, none affecting results.**
  - The fleet GPU watcher cancelled G_gated's allocation during its CPU-only merge. Training had finished; the merge and load checks were redone.
  - Two transfer-eval scorer failures, fixed by fix-3 (above).
  - A port race between evals sharing a node: two vLLM servers died with "address already in use". The eval script now holds a per-node lock while choosing ports.
  - Transient judge and generation timeouts resumed on retry.
  - Six old failure flags remain in `STATUS.flags`: one task from 2026-10-18 (an RL smoke test that ran out of host memory), four tasks from 2026-10-19 (an uppercase run-name error), and the 2026-10-22 gated-pair line (the G_gated incident above). All of that work was later redone successfully under different job lines.

---

## 10. Where everything is

Run outputs live under the campaign root `$TRACE_CERT_RUNS` (on the cluster, `/scratch/general/nfs1/u1592009/trace-cert-runs/`); paths below are relative to it unless noted. Each
`results/final/<name>/` directory has a `run.sh` that regenerates it from the raw eval files, and a `README.md`.

| What | Where |
|---|---|
| 3B in-domain comparison, families, gated, TRACE-val | `results/final/cert_3b/REPORT.md` |
| CI sensitivity (shared-instance bootstrap) | `results/final/cert_3b_shared_ci/shared_ci.md` |
| 7B replication | `results/final/cert_7b/cert_7b.md` |
| External transfer incl. q25 trajectory | `results/final/external/compare_ext.md` |
| Zero-shot / Section 5 | `results/final/section5/section5_zs.md` |
| Training curves, matching, compute, go/no-go | `results/final/training/training.md` |
| Per-checkpoint certificate evals | `eval/ckpt/<model>/` (`summary.md`, `validation.json`) |
| Per-model external results | `ext_eval/runs/<model>/{full,q25}/summary.seed<N>.md` |
| Design and pre-registration | `cert_notes/DESIGN.md`, `PLAN_SUMMARY.md` in the trace-tool checkout (not in this repository) |
| Experiment code, README, UNRESOLVED | `experiments/certificates/` in the trace repository (branch `experimental/certificates`) |
| Run status | `STATUS.md` |
