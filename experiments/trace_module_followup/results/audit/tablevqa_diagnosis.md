# TableVQABench: why VERO loses 16 points under the backbone interface

F0 audit. In the reviewed study TableVQABench contributes about 23% of the V@P1 − V@P0 interface loss (−15.97 on confirmation, −0.67 pp of the −2.94 macro). The mechanism had not been diagnosed. Data: stored outputs of the reviewed study only.

Tables:
- `tables/tablevqa_split_scores.csv`: official and normalized score per model × split component (pinned split arithmetic).
- `audit/tablevqa_diagnosis.csv`: failure class × split × model, both partitions.
- `audit/f0_tablevqa_blind_check.development.csv`: blinded sample with verdicts.
- Raw rows (predictions, gold) live only in `audit/f0_work/tvqa_rows.*.parquet`.

## 1. Where the loss sits (confirmation, mean of seeds 42–44)

| split component | rows per seed | V@P0 | V@P1 official | V@P1 normalized (v2) | V@P1 − V@P0 official → normalized |
|---|---|---|---|---|---|
| fintabnetqa relieved | 200 | 95.33 | 95.17 | 95.17 | −0.16 → −0.16 |
| fintabnetqa exact | 200 | 87.00 | 89.67 | 89.67 | +2.67 → +2.67 |
| vtabfact | 193 | 85.15 | 75.65 | 75.65 | −9.50 → −9.50 |
| vwtq | 609 | 59.72 | 26.60 | 35.91 | −33.12 → −23.81 |
| vwtq_syn | 198 | 66.16 | 26.43 | 39.90 | −39.73 → −26.26 |
| **benchmark (mean of 5)** | | 78.67 | 62.70 | 67.26 | **−15.97 → −11.41** |

The official split aggregation and every pinned numeric tolerance are preserved. Normalization changes only the WTQ splits, through rule T2 (empty list items); V@P1 under P1 almost never boxes its answer, so T1 changes none of its rows. The development screen (seed 42) shows the same pattern: V@P1 vwtq 16.4 vs V@P0 57.4, vwtq_syn 16.7 vs 57.1, vtabfact 87.0 vs 89.5.

## 2. Mechanism

1. **WTQ: hedged `||` lists (the main cause; a behaviour, not a parse problem).**
   - The pinned WTQ prompt asks for multiple answers to be separated with `||`. Under the backbone template, VERO takes this literally. 46.9% of its WTQ predictions list several values, against 4.3% for V@P0, 5.4% for T and 5.1% for B; only 4% of gold answers have several values.
   - The pinned denotation check requires the exact value set, so a list of several values is wrong whenever the gold has one. Per seed this is 345 of V@P1's 807 WTQ rows (V@P0: 13.3).
   - About 75% of those lists contain the gold value (e.g. gold `English`, prediction `English||French`). In 79% of those the gold is listed first.
   - V@P1 makes fewer plain wrong-value errors than V@P0 (vwtq 117 vs 228 per seed), so the lists replace answers that would have been both right and wrong.
2. **WTQ: empty list items (a format artifact, fixed by T2).** A prediction like `5||` or `11||` (a trailing separator with nothing after it) fails only because of the empty item. Per seed, 57 vwtq and 27 vwtq_syn V@P1 rows are fixed this way.
3. **vtabfact: real errors.** V@P1 gives the wrong True/False label more often (41.3 vs 25.0 rows per seed). Malformed outputs (both labels or neither) are rare (5.7 vs 3.7). There is no format artifact to remove.
4. **fintabnetqa: no loss.** V@P1's exact accuracy is 2.7 pp higher.

The merges inherit the pattern in proportion to their VERO share. Hedged-list rows per seed: M050 170, CV050 177, CT050 20, T 22. T2 fixes 43 (M050) and 21 (CV050) rows per seed.

## 3. Failure partition by type

Confirmation, mean rows per seed (all 3,600 rows of 3 seeds each in `audit/tablevqa_diagnosis.csv`):

| class (group) | V@P0 | V@P1 | V@P1 split location |
|---|---|---|---|
| extraction failure fixed by v2 (T2 empty items) | 0 | 83.3 | vwtq 56.7, vwtq_syn 26.7 |
| malformed response (empty, prose, both or neither label) | 3.7 | 9.3 | mostly vtabfact |
| numeric/text normalization not fixed (`Episode 7` vs `7`, `6272571` vs `6,272,571`, accounting negatives `$(16.7)` vs `-16.7`) | 17.3 | 14.7 | all splits |
| partial credit by design (fintabnetqa exact misses a scale word) | 16.7 | 11.0 | fintabnetqa |
| real error: hedged `||` list | 13.3 | 345.3 | vwtq 262.3, vwtq_syn 83.0 |
| real error: partial answer (gold is a longer phrase) | 3.0 | 3.3 | vwtq, fintabnetqa |
| real error: wrong value / label / missing values | 313.0 | 193.3 | vtabfact 41.3, vwtq 116.7, vwtq_syn 32.3 |

Normalization misses are of similar size in both models (17.3 vs 14.7 per seed), so they explain none of the gap. They stay unfixed by design: the brief requires pinned tolerances and no global permissive matcher.

## 4. Blinded stratified inspection (development partition)

**Sample.** 73 officially failed rows, up to 3 per split × automatic failure class, drawn across all development models (archived parents, P1 and P2 runs). Model identity was hidden while judging. Verdicts and the joined key are in `audit/f0_tablevqa_blind_check.development.csv`.

**Classes.** My blind class matches the v2 automatic class on 72/73 rows. The exception is TV-056, prose that uses boxes to mark rejected answers, which I called malformed and the code calls a box conflict.

**Fixes.** The inspection separated two harmless cases, both now in v2:
- trailing empty items (`4||`);
- a single boxed final answer (`\boxed{7066}` for `$7,066`), which matters under the VERO interface (P2): development B@P2 17.8 → 60.4, T@P2 19.4 → 63.7.

**v1 defects found** (both corrected in v2 before any confirmation number was read):
- T1 must not apply on vtabfact. TV-013, `The statement is True ... \boxed{False}`, would have escaped the pinned both-labels rule.
- LaTeX escapes inside a box must be removed. TV-005, `\boxed{55\%}`.

**Not fixable.**
- Hedged lists: choosing one listed value would mean choosing between candidates. "Take the first item" depends on knowing that the gold has one value (structural leakage) and would break genuine multi-answer rows.
- Wrong values.
- Numeric/text normalization misses.

## 5. Conclusion

V@P1's TableVQABench loss is mostly not an answer-format artifact.
- The format-normalized rule removes 4.56 of the 15.97 points (empty `||` items).
- The remaining −11.41 comes from two sources:
  - V@P1 answering WTQ questions with hedged lists, in which it usually names the correct value first but alongside wrong ones;
  - more wrong vtabfact labels.
- Both are behaviour changes induced by the backbone template plus the WTQ instruction, and the pinned contract scores them correctly as wrong.
- After normalization, TableVQABench is the largest single source (57%) of the residual V@P1 − V@P0 macro gap (−0.83 pp).
- No judge call is needed or would change this.
