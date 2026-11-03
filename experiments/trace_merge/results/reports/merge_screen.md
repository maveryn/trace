# Development screen: three 50:50 merges and their controls

Tier: **development** (the frozen development screen: 2,910 rows, sampled in whole image/problem groups, 81–129 per
benchmark). Decoding seed 42 only.

**Primary interface: P1.** Every new model is served with the backbone's tokenizer, chat template and processor
(DESIGN.md section 3 explains why). The parents come from three sources:
- B and T from the archive. Their archived outputs are already P1 outputs.
- V@P0 from the archive, generated with VERO's own template.
- V@P1, VERO re-run under P1.

**Diagnostic interface: P2.** B, T and the three merges are re-served with VERO's tokenizer, template and processor
(section 4 below).

**How to read the numbers.**
- Every number is descriptive. Nothing is selected from this screen: the confirmed set was fixed in advance
  (DESIGN.md section 7; locked at 09:04:27Z).
- Intervals are 95% paired cluster bootstraps, 10,000 replicates, unadjusted. The resampling units are the screen's
  bootstrap units: image/problem groups, with near-duplicate twin images joined into one unit (DESIGN amendment A7).
- The tables are in `tables/screen/` (`scores.P{1,2}.csv`, `differences.P{1,2}.csv`, `merge_capture_damage.csv`,
  `answer_format_sensitivity.csv`).

## 1. Technical checks (smoke and development tiers)

The four P1 models passed the 135-row smoke tier. All nine new runs (four P1, five P2) then passed the development
screen, and each passed three checks:
- **Requests.** Every rendered request is byte-identical to the archived reference request for the same row and
  protocol: prompt digest, model-neutral request digest and prompt-token count, 2,910/2,910 rows. The reference is
  the base model's archived request under P1 and VERO's archived request under P2.
- **Metrics.** The metric adapters reproduce the score campaign's own aggregate on all 24 benchmarks (maximum
  difference < 1e-14).
- **Served model.** The served view's weights hash to the audited parent or merge weights.

Two environment failures were repaired before any confirmation run had scores (DESIGN amendments A5, A6):
- **Scorer crash.** Pinned pandas 2.3.3 segfaults when it infers numeric types on SHA-256 request hashes shaped like
  `7649e84210128867284d…`. This crashed M050's LogicVista scorer. A score-neutral guard now rewrites only an exponent
  of 10 or more significant digits, in a private copy passed to the parser.
- **Out of memory.** Generation servers ran out of GPU memory on 44–48 GB cards. Utilization on those cards is now 0.80.

## 2. Official scores under P1 (24-benchmark macro and categories)

| model | macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B (base) | 47.97 [46.18, 49.76] | 53.13 | 42.54 | 49.61 | 60.30 | 47.62 | 34.64 |
| T (TRACE) | 51.40 [49.56, 53.24] | 60.09 | 45.90 | 54.88 | 60.48 | 49.11 | 37.97 |
| V@P0 (VERO, own template) | 53.11 [51.19, 55.08] | 57.61 | 49.06 | 56.45 | 59.41 | 54.60 | 41.55 |
| V@P1 (VERO, backbone template) | 48.44 [46.58, 50.39] | 51.92 | 47.50 | 54.69 | 58.30 | 51.03 | 27.23 |
| M050 = ½V + ½T | 50.38 [48.50, 52.24] | 57.13 | 47.54 | 53.71 | 58.24 | 49.15 | 36.49 |
| CV050 = ½V + ½B | 49.53 [47.66, 51.42] | 53.38 | 44.96 | 54.10 | 58.52 | 48.51 | 37.69 |
| CT050 = ½T + ½B | 48.51 [46.64, 50.37] | 56.71 | 42.30 | 52.93 | 58.74 | 46.63 | 33.77 |

Paired differences on the macro (pp):

| comparison | estimate [95% CI] | p (unadjusted) |
|---|---|---|
| M050 − T | −1.03 [−2.57, +0.54] | 0.20 |
| M050 − V@P1 | +1.93 [+0.34, +3.45] | 0.016 |
| M050 − CV050 | +0.85 [−0.67, +2.35] | 0.26 |
| M050 − CT050 | +1.86 [+0.20, +3.54] | 0.029 |
| CV050 − B | +1.55 [−0.15, +3.32] | 0.075 |
| CT050 − B | +0.54 [−1.08, +2.16] | 0.50 |
| T − B | +3.43 [+1.70, +5.18] | < 0.001 |
| V@P1 − B | +0.47 [−1.29, +2.37] | 0.58 |
| **V@P1 − V@P0** | **−4.67 [−6.09, −3.21]** | < 0.001 |

## 3. What the P1 screen shows

**(a) VERO's measured gain depends on its interface, largely through answer format (H3: supported for the macro and
for Puzzles & Logic; Visual Math's drop was not among the largest).**

- **The gain nearly vanishes.** VERO's own-template advantage over the base model (+5.14 pp on this screen) almost
  disappears under the backbone template: V@P1 − B = +0.47.
- **Where the loss sits.** V@P1 − V@P0 by category: Puzzles & Logic −14.32 [−18.33, −10.18], Charts & Tables −5.69
  [−9.23, −2.29], Perception & Counting −3.57 [−6.59, −0.43]. The other three categories' intervals include 0.
- **Why.** Under P1, VERO no longer writes `<think>`/`<answer>`. It often ends a multiple-choice answer with a
  LaTeX-wrapped letter, `Answer: \( C \)`, and the pinned PuzzleVQA and VisualPuzzles scorers do not read that form.

Answer-format diagnostic, development screen, seed 42 (`answer_format_sensitivity.csv`). Not a metric. The final
option letter is read leniently (bare, `\( X \)`, `$X$`, `\boxed{X}`).

| model | PuzzleVQA: ends in `\( X \)` | PuzzleVQA: scored 0, final letter correct | VisualPuzzles: ends in `\( X \)` | VisualPuzzles: scored 0, final letter correct |
|---|---|---|---|---|
| B | 0.0% | 3.9% | 1.6% | 0.0% |
| T | 0.0% | 3.1% | 0.8% | 0.8% |
| V@P0 | 0.0% | 0.0% | 0.0% | 0.0% |
| **V@P1** | **30.5%** | **37.5%** | **21.1%** | **9.4%** |
| M050 | 1.6% | 7.0% | 7.0% | 3.1% |
| CV050 | 0.0% | 2.3% | 3.1% | 1.6% |
| CT050 | 0.0% | 0.8% | 0.0% | 1.6% |

This is an observable answer-format effect, not evidence that VERO lost the underlying ability. The official scores
stay the primary metric (prespecified). Every reading below has to allow for one fact: under P1, VERO's parent
baseline is depressed partly by format. Under P2, none of the ten cells (five P2 models × two benchmarks) has a
LaTeX-wrapped final letter or a format loss.

**(b) M050 beats V@P1 and the TRACE-blend control, but not TRACE.**
- **Clear wins.** M050 beats CT050 by +1.86. CT050 is the TRACE blend, so this measures VERO's half-update. M050 also
  beats V@P1 by +1.93.
- **Not resolved.** Against CV050, the VERO blend, M050 is +0.85 and the interval includes 0. This contrast measures
  TRACE's half-update.
- **Against TRACE.** M050 is 1.03 pp below TRACE, and that interval also includes 0.

**Halving each update has a different cost.**
- Halving TRACE's update keeps little of TRACE's macro gain: CT050 − B = +0.54, against T − B = +3.43.
- Halving VERO's update scores slightly *better* than VERO itself under P1: CV050 − B = +1.55, against V@P1 − B = +0.47.
  This fits part of VERO's P1 deficit coming from its answer-format shift, which halving the update softens (CV050's
  PuzzleVQA format-loss rate is 2.3%, against V@P1's 37.5%).

**(c) The category pattern matches H2 (descriptive).**
- Adding TRACE's half-update (M050 − CV050) helps most in Charts & Tables: +3.75 [+1.03, +6.61].
- Adding VERO's half-update (M050 − CT050) helps most in Visual Math: +5.23 [+0.45, +9.95].
- No other category difference in either contrast excludes 0.

**(d) Capture and damage.** These rates cover the 21 binary-eligible benchmarks, rows pooled, seed 42. They are
contemporaneous: the parent sets come from the same seed, so the groups include lucky parental failures. The
independent-repeat variant needs the confirmation seeds.

| model | TRACE-only capture P(m=1 \| T=1, V@P1=0) | VERO-only capture P(m=1 \| V@P1=1, T=0) | shared-success damage P(m=0 \| both=1) | joint-failure rescue P(m=1 \| both=0) |
|---|---|---|---|---|
| group size (rows) | 313 | 256 | 974 | 982 |
| M050 | 51.4% [45.7, 57.3] | 52.7% [46.5, 59.0] | 7.3% [5.7, 9.0] | 6.8% [5.3, 8.5] |
| CV050 | 44.4% [38.5, 50.2] | 53.5% [47.4, 59.6] | 10.4% [8.5, 12.4] | 10.6% [8.6, 12.7] |
| CT050 | 56.9% [51.2, 62.6] | 27.0% [21.7, 32.6] | 12.0% [10.0, 14.1] | 11.8% [9.8, 14.0] |

- **Capture.** M050 keeps about half of each parent's unique successes. Its TRACE-only capture is above CV050's, and
  its VERO-only capture is far above CT050's, as H1 predicts.
- **Damage and rescue.** M050 damages fewer shared successes than either control, but also rescues fewer joint
  failures. Read the two columns together: a model closer to both parents moves less in both directions.
- **Status.** With one seed and same-draw parent sets, these are hypotheses for confirmation, not findings.

## 4. P2 diagnostic: the same models under VERO's interface

P2 serves every model with VERO's tokenizer, chat template (its ~490-token think/answer system prompt and `<think>`
prefill) and processor. V@P0 is VERO's archived output under that interface. P2 is a development-only diagnostic and
plays no part in the confirmed comparison.

| model | macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B@P2 | 42.71 [40.94, 44.44] | 39.64 | 33.01 | 46.09 | 57.55 | 47.93 | 32.02 |
| T@P2 | 47.21 [45.44, 49.05] | 43.78 | 39.77 | 52.93 | 59.34 | 49.29 | 38.16 |
| V@P0 | 53.11 [51.19, 55.08] | 57.61 | 49.06 | 56.45 | 59.41 | 54.60 | 41.55 |
| M050@P2 | 52.34 [50.46, 54.29] | 59.77 | 47.89 | 54.30 | 60.83 | 52.10 | 39.14 |
| CV050@P2 | 52.98 [51.05, 54.90] | 60.54 | 47.27 | 58.40 | 58.55 | 50.87 | 42.24 |
| CT050@P2 | 46.80 [44.96, 48.63] | 44.49 | 41.17 | 53.12 | 61.47 | 46.77 | 33.77 |

Paired differences on the P2 macro (pp):

| comparison | estimate [95% CI] | p (unadjusted) |
|---|---|---|
| M050@P2 − V@P0 | −0.78 [−2.40, +0.86] | 0.36 |
| M050@P2 − CV050@P2 | −0.64 [−2.13, +0.87] | 0.43 |
| M050@P2 − CT050@P2 | +5.54 [+3.61, +7.52] | < 0.001 |
| M050@P2 − T@P2 | +5.13 [+3.36, +6.93] | < 0.001 |
| T@P2 − B@P2 | +4.50 [+2.67, +6.42] | < 0.001 |
| V@P0 − B@P2 | +10.41 [+8.40, +12.44] | < 0.001 |

**(e) The merges follow the interface they are served with.**
- **Non-VERO models lose.** Under VERO's interface, both non-VERO models score below their P1 scores: B 42.71 against
  47.97, T 47.21 against 51.40.
- **VERO-containing models gain.** M050 scores 52.34 against 50.38 under P1, and CV050 52.98 against 49.53. Both are
  close to VERO's own 53.11. The P1-versus-P2 differences are differences of point estimates; the protocols are not
  paired, so they carry no interval.
- **CT050 does not.** CT050, which has no VERO update, stays near TRACE (46.80).

**(f) Under P2, no gain from TRACE's half-update on top of the VERO blend was detected.**
- M050@P2 − CV050@P2 is −0.64 on the macro, with an interval including 0.
- One category excludes 0, and it is negative: Science & General −4.10 [−7.16, −0.96]. Charts & Tables is −0.77
  [−4.80, +3.29].
- Under P1, the same contrast was +0.85 on the macro and +3.75 in Charts & Tables.
- So the screen does not show TRACE's half-update under VERO's interface. It does not show its absence either: the P1
  and P2 estimates are not paired, and the P2 interval is wide.

**(g) Under P2, VERO's half-update adds a lot on top of the TRACE blend.**
- M050@P2 − CT050@P2 is +5.54 on the macro.
- Most of it is in Charts & Tables: +15.28 [+10.73, +19.80]. Visual Math adds +6.72 [+1.72, +11.95].

## 5. Pending at the time of writing

**Confirmation** (P1, 26,228 rows, seeds 42–44, the four prespecified models). Six of the twelve runs are complete:
V@P1 seeds 42 and 43, M050 seed 42, CV050 seeds 42 and 43, and CT050 seed 42. The other six are running or queued.

*Update, 2026-11-02 23:41 MST:* all twelve confirmation runs completed. The confirmed results are in
`reports/analysis_report.md` §4. This screen's text above is unchanged.

## 6. Stage D

No component (frontend-preserving) experiment is proposed from this screen. The trigger would be a merge that helps
some categories while regressing on others; on this screen no category contrast of M050 against T, V@P1, CV050 or
CT050 is negative with an interval excluding 0. Category names do not identify skills, so a category pattern would not
by itself show which component holds one.

The dominant effect is an interaction between interface and answer format. On the P2 diagnostic (single seed,
descriptive), the VERO-containing merges' point estimates are near VERO's own (M050@P2 − V@P0 = −0.78 [−2.40, +0.86])
when they are served with VERO's interface. That points to the prompt/format frontend, not to a weight partition, as
the lever. A weight-partition experiment would not separate the two.
