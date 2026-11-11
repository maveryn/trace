# F0: format-normalized-v2 (answer-format artifacts vs capability)

Work package F0 of the TRACE–VERO follow-up. Source data: the reviewed study's stored outputs (work root
`trace_vero_qwen25_v1`, read-only). The follow-up `runs/` directory was never read. No GPU, no generation, no judge call.

`format-normalized-v2` is a **diagnostic beside the official score**. The official 24-benchmark macro stays the primary
metric, and every confirmatory statement of the reviewed study (Holm family) is unchanged. Normalized contrasts are
descriptive.

## 1. Freeze record

| item | value |
|---|---|
| frozen spec | `audit/scoring_spec.format-normalized-v2.json` (sha256 `a229a5640394054b712c257dffb1bd89ad61f7f197b707882372f03259ea1a02`, frozen 2026-11-06T07:17:15Z, read-only) |
| `trace_followup/extract.py` | `6c3fb4578e9511fe781052b726b04420e21d5fdbab6d3f090976107b3a9f3e0c` |
| `trace_followup/rescore.py` | `f203850862c7268a3f463ed1c79b217168d65c36c52b2800c9a71d966e7ba923` |
| tests | `tests/test_extract.py`, `tests/test_rescore.py`: 85 passed (the 3 pinned-scorer tests ran) |
| superseded | `audit/scoring_spec.format-normalized-v1.json` (frozen 2026-11-06, read-only; see §1.1) |

Timeline:
1. Rule developed on development-partition rows only.
2. Unit tests passed.
3. The blinded check of 80 rows agreed fully (§6).
4. v1 frozen.
5. v1 development run.
6. Blinded TableVQABench check (§6) found two v1 defects.
7. The v1 confirmation run was stopped before it wrote anything.
8. v1 development outputs were moved to `audit/superseded_v1/`.
9. v2 frozen.
10. v2 run on development, then on confirmation (the frozen-rule application).

No confirmation number was read before the v2 freeze.

### 1.1 Why v2 superseded v1 (development evidence only)

1. **T1 on vtabfact.** In v1, T1 (read a single `\boxed{}` answer) also applied on vtabfact. The pinned tabfact scorer already reads `true`/`false` anywhere in the prediction, and it scores a prediction holding both labels as unparsed. So on vtabfact a box could only ever override that rule. That credited self-contradictory responses (`The statement is True ... \boxed{False}`) which the same response without a box would not get. v2 restricts T1 to fintabnetqa, vwtq and vwtq_syn.
2. **LaTeX escapes.** v1 did not remove escapes inside a box, so `\boxed{55\%}` was not read as `55%`. v2 removes `\% \$ \& \# \_`.
3. **Effect on development rows.** 23 TableVQABench rows change status and 4 change score: 3 P2-interface rows gain, and 1 contradictory vtabfact row loses its v1 repair. No P1 model and no archived parent changes score. The MC rule is identical in v1 and v2.

## 2. The rule

Status codes and per-row reasons are in the ledger, `audit/scoring_changes.csv`.

**Out of scope (21 benchmarks).** Normalized = official. Reasons are recorded per benchmark in `rescore.OUT_OF_SCOPE`. Development evidence for the scope is in §5.

**PuzzleVQA and VisualPuzzles (`mc_final_option`).**
- The pinned scorer parses the prediction cell with `extract_unambiguous_abcd`.
- If that parse yields an option, the row is untouched (`official_read`), whether the option is right or wrong.
- Only when the pinned parse is unresolved (`Z`, official score 0) does the extractor read the same cell. It gets the row's allowed labels and no gold.
- The extractor's grammar (precedence):
  1. **Final region.** In order of preference: the `<answer>` block(s); else the text after `</think>`; else the last paragraph. Reasoning that never closed has no final region.
  2. **Designators in that region.** Balanced `\boxed{}`, answer statements (`Answer: \( D \)`, `the answer is (C)`, ...), and a final line that is a lone option token.
  3. **Decision.** All designators must name the same allowed label. Two labels, or a different label designated outside the region, give `ambiguous`. Several letters (`AB`, `B and D`) or a non-option give `not_an_option`. There is no "take the last one" rule.
  4. **Harmless wrappers.** `B`, `B.`, `(B)`, `**B**`, `\( B \)`, `\(\text{B}\)`, `$B$`, `$B`, `\boxed{B}`, `<answer>B</answer>`, `Option B`.
- Resolved: normalized = 1 if the extracted option is gold, else 0. Gold is the pinned `to_choice_letter` for PuzzleVQA and the licensed letter for VisualPuzzles.
- Unresolved: normalized = 0, and the row stays in the denominator.

**TableVQABench (`tvqa_final_answer`).**
- Rows with full official credit are untouched.
- On every other row, the exact string the pinned split scorer read is rewritten:
  - **T1** (fintabnetqa, vwtq, vwtq_syn): a single designated `\boxed{}` answer, with whole-content macros and escapes removed, replaces the prediction. Disagreeing boxes give `unresolved`.
  - **T2** (vwtq, vwtq_syn): empty items of a `||` list (`5||`) are dropped when a non-empty item remains.
- The rewritten prediction is re-scored with the **pinned** split scorer, using the same tolerances. Each component becomes max(official, rewritten).
- There is no global or permissive matcher, and a list of several non-empty values is never narrowed.

**Invariants (tested).**
- normalized ≥ official on every row.
- Every row stays in its denominator.
- Gold only scores an extraction made without it.
- One rule for every model, interface, seed and partition.

Benchmark values use the reviewed adapters with native aggregation: TableVQABench is the mean of five split scores; WeMath is the problem-group strict score (A2). The macro is the unweighted mean of the 24 benchmarks. Intervals come from the reviewed paired cluster bootstrap with the same units, seed and 10,000 replicates.

**Reproduction check.** Recomputed official values reproduce the reviewed seed tables: max error 4e-14 pp over 504 confirmation slices and 528 development slices. Official contrast estimates and intervals reproduce the reviewed difference tables: max error 1.4e-14 pp.

## 3. Which rows changed, and why

Changed rows (official 0 → normalized 1, or fintabnetqa half credit → full credit). Counts are summed over seeds. The development columns cover the archived parents on the full development partition (seeds 42–44) and the screen runs (seed 42).

| benchmark | model | development | confirmation (3 seeds) | reason |
|---|---|---|---|---|
| PuzzleVQA | V@P1 | 50 / 128 | 1,605 / 4,785 | `Answer: \( D \)`, `Answer: $B$`, `$B`, `\(\text{A}\)`: the pinned parser only reads a bare `Answer: X` |
| PuzzleVQA | B | 34 / 1,215 | 136 / 4,785 | `Answer: (D)`: parentheses defeat the pinned parser |
| PuzzleVQA | T | 14 / 1,215 | 90 / 4,785 | same |
| PuzzleVQA | M050 / CV050 / CT050 | 9 / 3 / 1 of 128 | 277 / 273 / 83 | same wrappers |
| PuzzleVQA | V@P0 | 0 | 0 | its answers sit in one `<answer>` block, which the parser reads |
| VisualPuzzles | V@P1 | 12 / 128 | 354 / 2,799 | as for PuzzleVQA |
| VisualPuzzles | B / T | 4 / 4 of 705 | 40 / 14 | as for PuzzleVQA |
| VisualPuzzles | M050 / CV050 / CT050 | 5 / 2 / 2 of 128 | 103 / 117 / 15 | as for PuzzleVQA |
| VisualPuzzles | V@P0 | 1 / 705 | 0 | as for PuzzleVQA |
| TableVQABench | V@P1 | 5 / 128 | 250 / 3,600 | T2: `4||`-style empty list items |
| TableVQABench | M050 / CV050 | 3 / 2 of 128 | 128 / 64 | T2 |
| TableVQABench | B / T / CT050 | 0 | 5 / 3 / 2 | T2 |
| TableVQABench | B@P2 / T@P2 / CT050@P2 / M050@P2 / CV050@P2 | 54 / 61 / 63 / 7 / 2 of 128 | (P2 is development only) | T1: the VERO interface makes models box their answer, and the TableVQA path never unwraps boxes |

Rows the rule examined but did not change stay at their official score:
- **`resolved_still_wrong`**, an explicit wrong answer. Confirmation: PuzzleVQA V@P1 1,698, B 194; VisualPuzzles V@P1 492; TableVQABench V@P1 215.
- **`unresolved`.** Confirmation: PuzzleVQA B 61, V@P1 9; VisualPuzzles B 45.
- **`no_rule_applies`** (TableVQABench).

Of V@P1's confirmation PuzzleVQA rows, 69% (3,312 of 4,785) are unreadable by the pinned parser. The extractor resolves 99.7% of those, and 48.6% of the resolved ones are correct. On the 1,473 rows the parser does read, V@P1 is correct on 50.8%.

## 4. Official vs normalized

Full tables: `tables/official_vs_normalized.csv` (per model × benchmark, category and macro, both partitions) and `tables/official_vs_normalized.contrasts.csv` (all contrasts, every level, both scores, same bootstrap draws).

### 4.1 Macro (24 benchmarks; mean over seeds)

| subset | model | official | normalized | Δ |
|---|---|---|---|---|
| confirmation | B | 48.10 | 48.28 | +0.18 |
| confirmation | T | 52.13 | 52.23 | +0.10 |
| confirmation | V@P0 | 52.23 | 52.23 | +0.00 |
| confirmation | V@P1 | 49.29 | 51.40 | **+2.11** |
| confirmation | M050 | 52.36 | 52.85 | +0.49 |
| confirmation | CV050 | 51.06 | 51.52 | +0.46 |
| confirmation | CT050 | 50.48 | 50.57 | +0.10 |
| development screen (seed 42) | V@P1 / V@P0 | 48.44 / 53.11 | 50.55 / 53.11 | +2.11 / 0 |
| development screen P2 | B@P2 / T@P2 / CT050@P2 | 42.71 / 47.21 / 46.80 | 44.48 / 49.06 / 48.76 | +1.77 / +1.85 / +1.96 |

### 4.2 Key contrasts (24-benchmark macro; 95% paired cluster bootstrap)

| subset | contrast | official | normalized | change |
|---|---|---|---|---|
| development partition (B/T/V@P0, seeds 42–44) | T − B | +4.19 [+3.30, +5.10] | +4.12 [+3.23, +5.03] | −0.07 |
| development partition | V@P0 − B | +4.96 [+3.81, +6.12] | +4.82 [+3.68, +5.99] | −0.13 |
| development screen (seed 42) | V@P1 − V@P0 | −4.67 [−6.09, −3.21] | −2.56 [−3.96, −1.13] | +2.11 |
| development screen | M050 − T | −1.03 [−2.57, +0.54] | −0.67 [−2.22, +0.90] | +0.36 |
| development screen | M050 − V@P1 | +1.93 [+0.34, +3.45] | +0.34 [−1.23, +1.86] | −1.59 |
| **confirmation** (seeds 42–44) | **V@P1 − V@P0** | **−2.94 [−3.33, −2.56]** | **−0.83 [−1.21, −0.46]** | **+2.11** |
| confirmation | T − B | +4.02 [+3.58, +4.45] | +3.94 [+3.50, +4.38] | −0.08 |
| confirmation | V@P0 − B | +4.13 [+3.55, +4.71] | +3.95 [+3.37, +4.53] | −0.18 |
| confirmation | V@P1 − B | +1.19 [+0.65, +1.72] | +3.12 [+2.58, +3.66] | +1.93 |
| confirmation | M050 − T | +0.23 [−0.19, +0.65] | +0.62 [+0.20, +1.04] | +0.39 |
| confirmation | M050 − V@P1 | +3.07 [+2.62, +3.51] | +1.45 [+1.00, +1.88] | −1.62 |
| confirmation | M050 − CV050 | +1.29 [+0.93, +1.64] | +1.33 [+0.97, +1.68] | +0.04 |
| confirmation | M050 − CT050 | +1.88 [+1.45, +2.31] | +2.28 [+1.85, +2.70] | +0.40 |
| confirmation | M050 − V@P0 | +0.12 [−0.37, +0.62] | +0.62 [+0.13, +1.11] | +0.49 |
| confirmation | T − V@P0 | −0.11 [−0.68, +0.47] | −0.01 [−0.58, +0.56] | +0.10 |

How the normalized contrasts differ:

- **VERO's interface dependence is mostly a format artifact.** On confirmation, normalization closes 2.11 of the 2.94 pp V@P1 − V@P0 gap (72%):
  - PuzzleVQA −33.10 → +0.44 (macro contribution −1.38 → +0.02).
  - VisualPuzzles −13.29 → −0.64 (−0.55 → −0.03).
  - TableVQABench −15.97 → −11.41 (−0.67 → −0.48).
  - The residual −0.83 is 57% TableVQABench, which is not a format artifact (see `tablevqa_diagnosis.md`). The rest is small losses spread over ERQA, TreeBench and the Visual Math benchmarks, which v2 leaves official.
  - Category level: Puzzles & Logic −11.20 → +0.35 [−0.64, +1.31]; Charts & Tables −3.89 → −2.75; Visual Math unchanged at −1.77.
  - Under the backbone interface VERO keeps +3.12 of its +3.95 normalized gain over B (79%), against 1.19 of 4.13 (29%) officially.
- **TRACE vs base and VERO vs base barely move.** T − B −0.08 and V@P0 − B −0.18. B gains a little in PuzzleVQA from its `Answer: (D)` habit, so both gains shrink slightly. V@P0 is unaffected (0 changed rows).
- **M050 − T.** It becomes +0.62 [+0.20, +1.04]: M050 inherits some of V@P1's wrapper habit (PuzzleVQA −4.62 → −0.71, VisualPuzzles −1.43 → +1.75, TableVQABench −7.63 → −5.29). This is a descriptive sensitivity result. The confirmatory Holm-family result (M050 − T not shown, p = 0.29) is unchanged. The interval now excludes 0 only under a diagnostic metric chosen after the reviewed study, so no superiority claim follows.
- **M050 category results.**
  - Its only interval below T, Charts & Tables −0.91 [−1.76, −0.07], becomes −0.32 [−1.15, +0.51] (format-sensitive).
  - Puzzles & Logic −0.72 → +1.05 [−0.18, +2.27].
- **M050 − V@P1.** It falls from +3.07 to +1.45 because most of V@P1's deficit was format.
- **P2 diagnostic** (development screen):
  - The P2 TableVQA collapse of B@P2 (17.83) and T@P2 (19.39) is almost entirely boxed answers; normalized values are 60.36 and 63.71.
  - V@P0 − B@P2 goes +10.41 → +8.63, and M050@P2 − T@P2 goes +5.13 → +3.60.

## 5. Scope evidence (development partition; `tables/f0/mc_scope_survey.development.csv`)

For every option benchmark, the gold-blind extractor was compared with the official outcome. Rows are from the development partition; archived B/T/V@P0 cover 3 seeds, local runs seed 42.

- **PuzzleVQA and VisualPuzzles.** These are the only benchmarks where the pinned parser leaves clearly designated options unread:
  - "official 0 but the final option is gold": PuzzleVQA 112 rows, VisualPuzzles 30.
  - Whenever the pinned parser reads an option, the extractor never resolves a different one (0 rows).
- **Judge-graded MC benchmarks** (MMStar, BLINK, MMMU-ProVis, WeMath, SpatialViz, ERQA, LogicVista, MME-Reasoning, MathVista, MathVision):
  - "Official 0, final boxed option is gold" occurs almost only for V@P0 (WeMath 34, BLINK 21, MMStar 11, MMMU-ProVis 5).
  - Blind inspection shows these are responses whose reasoning contradicts the boxed answer (e.g. the reasoning concludes "B" and the response boxes "A"). The judge read the whole response, so this is not a format failure. Re-adjudicating them would need a judge; none was called.
- **SpatialViz/ERQA.** "Official 1 but the final option differs" (archive 79, P2 ERQA 58) is matcher or judge leniency, not format. v2 never lowers a score; re-grading would need a judge.
- **PhyX, TreeBench.** The pinned parsers over-read: PhyX takes the last standalone A–D, even a unit "C"; TreeBench takes the first capital A–E of the answer text. They rarely fail to read. Their errors are misreads, which v2 does not override.

## 6. Blinded manual check (acceptance gate)

### 6.1 Main check: `audit/f0_manual_check.development.csv`

**Sample.** 80 development rows: 60 stratified over rule outcomes (24 MC repaired, 6 MC resolved-wrong, 5 MC unresolved, 5 MC official-read, 10 TableVQA repaired, 4 TableVQA resolved-wrong, 6 TableVQA no-rule) plus 20 officially credited MC rows as a baseline.

**Blinding.** Model, seed, status and extracted value were hidden. For each row I recorded the designated final answer, whether the reasoning supports it, and the expected score. The key was joined afterwards (items: `audit/f0_work/manual_check*.items.development.jsonl`).

**Result.**
- Normalized score equals the blind expectation in 80/80 rows.
- The extractor's letter equals my reading in 30/30 resolved MC rows.
- Rows read as unresolved stayed unresolved: truncated or degenerate outputs, `Answer: AB`, `\boxed{R}`.
- Explicit wrong answers stayed wrong in 10/10.

**Reasoning that disagrees with a correct letter.**
- 1 of 24 repaired rows: F0-040 designates `(A)` = circle, which is gold, while its reasoning concludes "diamond".
- 1 of 20 officially credited baseline rows: F0B-008 has the same pattern.

So a correct letter with inconsistent reasoning is credited at the same rate the official parser already credits it. v2 treats a wrapped designation exactly like the official `Answer: X`, and does not judge reasoning. See the assumptions in §7.

### 6.2 TableVQABench check: `audit/f0_tablevqa_blind_check.development.csv`

**Sample.** 73 officially failed rows, up to 3 per split × failure class, model hidden.

**Result.**
- The check found the two v1 defects of §1.1.
- The v2 automatic failure class agrees with the blind reading in 72/73 rows (v1 taxonomy: 56/73). The one disagreement labels prose that boxes rejected answers as a box conflict rather than a malformed response.

## 7. Limits, judge needs, assumptions to review

- **What a wrapped repair does not judge.** A wrapped repair can credit a correct letter whose reasoning concludes something else. The official parser does the same for unwrapped letters (rate 1/24 vs 1/20 in the blind check). Excluding such rows would need a judge or reasoning analysis, and it would make the normalized metric stricter for wrapped answers than the official metric is for unwrapped ones.
- **Repair only.** v2 never lowers a score. Official over-reads (PhyX units, TreeBench first letter) and judge leniency (SpatialViz/ERQA) stay as scored; quantifying them would need a judge.
- **TableVQA hedged lists.** These are not repaired (see `tablevqa_diagnosis.md`). No judge would change that under the pinned single-answer contract.
- **Audit-label wording.** The audit-only group label in the frozen code reads "extraction failure (fixed by v1)". In v2 outputs it means "fixed by the rule".
- **P2 coverage.** P2 rows exist only on the development screen (seed 42), so the P2 numbers above are development-only diagnostics.
