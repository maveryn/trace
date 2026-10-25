# Qwen3.5 results: inference-time tools and forced final answers on TRACE

> **Sources.** All numbers are official scores after the judge pass (2026-10-06). Every number was recomputed for this
> write-up from the per-row files under `runs/` (`scores.jsonl`, `responses.jsonl`, `run_config.json`,
> `judge_results.jsonl`). Numbers quoted from the reports next to this file match those files. Numbers marked **†** come
> from extra paired tests run for this write-up; they are not in the copied reports. Row-level counts (stop reasons,
> forced turns, cut-offs, call counts, judge counts) are not in the reports either. Each section names its source, and
> [Extra tests](#extra-tests) lists the commands.

Generated 2026-10-06 with `run_on_server.sh`, on TRACE validation, for Qwen3.5-4B and Qwen3.5-9B with thinking off.
The questions (Q1–Q3, and Q6 for Qwen3.5) and the decision rule are in [EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md).
The forced conditions are specified in its "Forced final answer (Qwen3.5)" section. The copied reports are described under
[Files](#files).

## Takeaway

**Q1. Zero-shot, under these prompts and loop settings, no tool condition improves on `no_tool` for Qwen3.5, forced or
not.**
- Every unforced tool condition is significantly worse at both sizes on the seeds-pooled bootstrap CI. The losses run from
  −2.32 pp (4B `calculator`) to −25.62 pp (4B `zoom`), and from −3.27 pp (9B `calculator`) to −20.27 pp (9B `all`).
- Nine of the ten forced conditions are significantly worse too. The exception is 9B `calculator_forced`: −0.90 pp,
  CI [−2.12, +0.30], not significant in any seed.
- No condition meets decision rule (a).

**Q2. `no_tool_budget` beats `no_tool` by +5.08 pp (4B) and +5.30 pp (9B), significant in every seed. All of that gain
comes from the 2,048-token cut-off.**
- Even with thinking off, Qwen3.5 writes long reasoning. Under TRACE's 2,048-token limit, `no_tool` is cut off in 19.3%
  (4B) and 17.2% (9B) of episodes. For Qwen2.5-VL the figures were 0.5% (3B) and 0.3% (7B).
- `no_tool` scores 10.8% (4B) and 11.5% (9B) on those rows. With 8,192 tokens, `no_tool_budget` scores 39.8% and 44.5%
  on the same rows. These rows alone contribute +5.60 and +5.67 pp. The other rows contribute −0.52 and −0.37 pp.
- So for Qwen3.5 the cut-off depresses the TRACE-protocol baseline, and `no_tool_budget` is the stronger baseline. Every
  tool condition, forced or not, is 6.20†–30.70 pp below it, significant in every seed (forced: extra test †).

**Q6. The countdown and forced final answer fix most of the "still calling tools at the turn limit" problem. They do not
fix incomplete responses as a whole, and they do not make the tools help.**
- **Where it works.** `zoom` and `visual_ops` at both sizes, and `all` at 9B, mostly fail by calling tools until the
  6-turn limit (`max_turns` in 23.4–36.9% of episodes). For 4B `all`, `max_turns` (14.3%) is slightly below its
  2,048-token cut-offs (16.3%). Forcing turns most of these episodes into answers:
  - The forced turn ends in an answer 54.9–88.9% of the time. In the rest the model still writes a tool call
    (7.6–35.1%) or is cut off at the length limit (2.4–10.1%).
  - No-answer rates fall from 25.6–38.9% to 7.8–26.9%.
  - Accuracy rises by +5.85 to +14.83 pp over the unforced base, significant in every seed after Holm across the 5 pairs†.
- **What it leaves.** The other kind of incomplete response is a turn cut at 2,048 tokens before it writes a call or an
  answer. Forcing does not address it, and it is most of the no-answers for `calculator` and `ocr`:
  - The 4B no-answer rate stays put: `calculator` 12.6% → 13.1%, `ocr` 16.2% → 15.9%. Forced vs unforced is not
    significant there in any seed.
  - For 9B, forcing gains +2.37 pp (`calculator`) and +3.93 pp (`ocr`), significant in 2 of 3 seeds after Holm across
    the 5 pairs†.
- **Tools still lose.** Forced, the tool conditions are still 0.90–11.62 pp below `no_tool` and 6.20–16.70 pp† below
  `no_tool_budget`. On the rows where a forced turn happened, the forced run scores 30.3–42.6%. `no_tool` scores
  43.2–59.8% on the same rows.

The pilot's `oracle_crop` (one seed) shows no significant gain from looking in the right place: +1.6 pp (4B) and
+2.6 pp (9B), Holm p 0.58 and 0.28, with 95% CIs reaching +4.4 pp (4B) and +5.3 pp (9B). One seed can neither confirm
nor rule out a few points of headroom.

## Main result (full run: 2000 rows × seeds 42, 43, 44)

Official accuracy (%), seeds pooled (each row's correctness is averaged over the 3 seeds). Δ is in pp. Brackets are the
95% paired-bootstrap CI vs `no_tool`.

Sources:
- `compare.md`: `no_tool_budget` and the unforced tools vs `no_tool`.
- `compare_forced.md`: forced vs `no_tool`.
- `compare_vs_budget.md`: unforced vs `no_tool_budget`.
- † forced vs `no_tool_budget`, an extra test.

| condition | 4B acc | Δ vs no_tool | Δ vs no_tool_budget | 9B acc | Δ vs no_tool | Δ vs no_tool_budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_tool | 61.65 | — | −5.08 | 65.25 | — | −5.30 |
| no_tool_budget | 66.73 | +5.08 [+4.07, +6.12] | — | 70.55 | +5.30 [+4.38, +6.20] | — |
| zoom | 36.03 | −25.62 [−27.43, −23.83] | −30.70 | 46.12 | −19.13 [−20.93, −17.37] | −24.43 |
| visual_ops | 47.17 | −14.48 [−16.00, −12.93] | −19.57 | 46.63 | −18.62 [−20.33, −16.93] | −23.92 |
| calculator | 59.33 | −2.32 [−3.62, −1.02] | −7.40 | 61.98 | −3.27 [−4.55, −2.03] | −8.57 |
| ocr | 55.48 | −6.17 [−7.50, −4.82] | −11.25 | 58.72 | −6.53 [−7.83, −5.22] | −11.83 |
| all | 51.98 | −9.67 [−11.10, −8.20] | −14.75 | 44.98 | −20.27 [−21.98, −18.60] | −25.57 |
| zoom_forced | 50.03 | −11.62 [−13.30, −9.92] | −16.70† | 54.48 | −10.77 [−12.33, −9.17] | −16.07† |
| visual_ops_forced | 55.82 | −5.83 [−7.35, −4.37] | −10.92† | 59.05 | −6.20 [−7.73, −4.70] | −11.50† |
| calculator_forced | 59.15 | −2.50 [−3.78, −1.22] | −7.58† | 64.35 | −0.90 [−2.12, +0.30] | −6.20† |
| ocr_forced | 56.20 | −5.45 [−6.82, −4.12] | −10.53† | 62.65 | −2.60 [−3.87, −1.37] | −7.90† |
| all_forced | 57.83 | −3.82 [−5.20, −2.43] | −8.90† | 59.82 | −5.43 [−6.95, −3.90] | −10.73† |

`no_tool` per seed (42, 43, 44): 62.60, 61.55, 60.80 (4B); 65.40, 64.60, 65.75 (9B). These are the TRACE-protocol
settings. [EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md) lists published TRACE base scores only for Qwen2.5-VL, so
this run has no built-in external check like Phase 1's.

**Per-seed tests.** Exact McNemar, Holm-adjusted within each seed and family. All families except the
forced-vs-`no_tool_budget` family (†, an extra test) are the ones in the reports.
- **Core family** (`compare.md`: `no_tool_budget` plus the 5 tools).
  - Every unforced tool condition is significantly below `no_tool` in all three seeds at both sizes, with two exceptions,
    both `calculator`: 4B seed 44 (−0.30 pp, Holm p 0.81) and 9B seed 43 (−2.10 pp, Holm p 0.0516).
  - Every other Holm p is ≤ 0.00913 (4B `calculator`, seed 43).
  - `no_tool_budget` is significantly *above* `no_tool` in every seed (Holm p ≤ 2.62e-08 for 4B, ≤ 8.33e-10 for 9B).
- **Forced family** (`compare_forced.md`: the 5 forced conditions).
  - 4B: every forced condition is significantly below `no_tool` in every seed, except `calculator_forced` in seed 44
    (−0.75 pp, Holm p 0.50). Otherwise the largest Holm p is 0.00735 (`all_forced`, seed 44).
  - 9B: `zoom_forced`, `visual_ops_forced` and `all_forced` are significant in every seed (largest Holm p 2.53e-04).
    `ocr_forced` is significant in seeds 43 and 44 (Holm p 0.00265, 0.0384) but not in seed 42 (0.0958).
    `calculator_forced` is not significant in any seed (Holm p 0.263, 0.193, 0.92).
- **Everything in one family** (`compare_all.md`: all 11 conditions vs `no_tool`). The verdicts are the same; 9B
  `calculator` seed 43 has Holm p 0.103 there.
- **Against `no_tool_budget`.** Every tool condition is significantly below it in every seed. Unforced
  (`compare_vs_budget.md`): largest Holm p 1.07e-08 (4B `calculator`, seed 44). Forced†: largest Holm p 5.27e-08
  (9B `calculator_forced`, seed 44).

### Why `no_tool_budget` gains 5 pp (Q2)

Both runs are paired by row and seed over all three seeds, and split by whether `no_tool` was cut off. Computed from
`scores.jsonl` (`stop_reason`, `completion_tokens`, `official_correct`).

| | 4B | 9B |
| --- | ---: | ---: |
| `no_tool` episodes cut at 2,048 tokens (`stop_reason` `length`) | 19.3% (1,158 of 6,000) | 17.2% (1,032) |
| `no_tool` accuracy on those rows | 10.8 | 11.5 |
| `no_tool_budget` accuracy on the same rows | 39.8 | 44.5 |
| `no_tool_budget` on the same rows: answered / again longer than 2,048 tokens | 81.3% / 71.1% | 78.8% / 74.1% |
| contribution of these rows to Δ (pp) | +5.60 | +5.67 |
| other rows: `no_tool` vs `no_tool_budget` accuracy | 73.8 vs 73.2 | 76.4 vs 76.0 |
| contribution of the other rows to Δ (pp) | −0.52 | −0.37 |
| `no_tool_budget` episodes longer than 2,048 tokens / their accuracy | 19.3% / 37.7 | 17.4% / 42.5 |
| `no_tool_budget` cut at 8,192 tokens | 4.2% | 4.2% |
| well-formed final JSON (`strict_format`), `no_tool` → `no_tool_budget` | 80.7 → 95.8 | 82.8 → 95.5 |
| mean generated tokens, `no_tool` → `no_tool_budget` | 763 → 1,354 | 699 → 1,255 |

- A cut-off episode rarely ends in a usable answer. Of the cut rows, 228 (4B) and 208 (9B) were resolved by the judge
  from an answer stated mid-reasoning.
- Without a cut-off, the extra budget changes nothing. Take the rows `no_tool` answered and where `no_tool_budget` also
  stayed within 2,048 tokens: there the two conditions score the same (75.7 vs 75.4 for 4B, 77.5 vs 77.6 for 9B).
- The pilot shows the same picture. `no_tool` is cut in 20.0% (4B) and 19.0% (9B) of episodes, and those rows contribute
  +5.60 and +6.20 pp of the pilot's +5.0 and +5.9 pp.
- The split is descriptive. The two runs are independent samples: the per-request seed does not reproduce the text when
  `max_tokens` differs. In seed 42, only 36% (4B) and 47% (9B) of the rows `no_tool` answered got a byte-identical reply
  in `no_tool_budget`.
- The same 2,048-token limit applies to every turn of every tool condition. It cuts the first turn of 0.4–14.0% of tool
  episodes; see [Why the tools help or hurt](#why-the-tools-help-or-hurt-q3).

## Effect of the budget notice and forced final answer

Each `*_forced` condition is its base condition plus four things together:
- a budget statement in the system prompt;
- a "[Tool calls left: N]" countdown inside the last tool response;
- a 1,024-token answer reserve;
- a forced final turn, announced by the "[No tool calls left ... give your final answer now ...]" notice. Calls written
  in that turn are recorded but not executed.

This design measures the whole package (see [Caveats](#caveats)).

### Forced vs unforced (Q6)

Sources:
- Seeds-pooled Δ and CI: `compare_forced_vs_unforced.md`. It concatenates one compare call per pair, so its Holm p equals
  the raw p.
- Per-seed Δ and exact McNemar p: the same file. The b/c counts were recomputed from `scores.jsonl` and match.
- † Holm across the 5 pairs within each model and seed, computed for this write-up.

Each per-seed cell gives Δ (pp), McNemar p, and Holm p† in parentheses.

| | tool | unforced | forced | Δ [95% CI] | seed 42 | seed 43 | seed 44 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 4B | zoom | 36.03 | 50.03 | +14.00 [+12.52, +15.55] | +12.40, 6.3e-22 (3.1e-21) | +15.30, 3.5e-33 (1.7e-32) | +14.30, 8.6e-29 (4.3e-28) |
| 4B | visual_ops | 47.17 | 55.82 | +8.65 [+7.23, +10.07] | +7.15, 4.4e-09 (1.3e-08) | +7.70, 3.2e-10 (1.3e-09) | +11.10, 2.1e-18 (8.2e-18) |
| 4B | calculator | 59.33 | 59.15 | −0.18 [−1.45, +1.02] | +0.75, 0.50 (1) | −0.85, 0.44 (0.82) | −0.45, 0.70 (0.70) |
| 4B | ocr | 55.48 | 56.20 | +0.72 [−0.57, +1.95] | −0.60, 0.61 (1) | +0.95, 0.41 (0.82) | +1.80, 0.105 (0.21) |
| 4B | all | 51.98 | 57.83 | +5.85 [+4.55, +7.18] | +7.50, 1.0e-10 (4.1e-10) | +4.55, 7.7e-05 (2.3e-04) | +5.50, 1.6e-06 (4.8e-06) |
| 9B | zoom | 46.12 | 54.48 | +8.37 [+6.90, +9.83] | +8.30, 8.1e-12 (2.4e-11) | +8.30, 1.9e-11 (5.8e-11) | +8.50, 4.5e-12 (1.3e-11) |
| 9B | visual_ops | 46.63 | 59.05 | +12.42 [+10.93, +13.88] | +11.10, 9.3e-20 (3.7e-19) | +12.95, 2.0e-25 (7.9e-25) | +13.20, 8.9e-27 (3.6e-26) |
| 9B | calculator | 61.98 | 64.35 | +2.37 [+1.20, +3.53] | +2.45, 0.019 (0.019) | +0.75, 0.49 (0.49) | +3.90, 2.1e-04 (2.9e-04) |
| 9B | ocr | 58.72 | 62.65 | +3.93 [+2.68, +5.15] | +5.30, 1.4e-06 (2.9e-06) | +2.35, 0.028 (0.055) | +4.15, 1.4e-04 (2.9e-04) |
| 9B | all | 44.98 | 59.82 | +14.83 [+13.40, +16.32] | +15.35, 5.2e-36 (2.6e-35) | +14.60, 1.6e-32 (7.8e-32) | +14.55, 1.4e-32 (6.9e-32) |

- After Holm† across the 5 pairs, forcing helps in all three seeds for `zoom`, `visual_ops` and `all` at both sizes.
- For 9B it also helps `calculator` and `ocr` in seeds 42 and 44, but not in seed 43 (Holm p† 0.49, 0.055).
- For 4B `calculator` and `ocr`, no seed is significant.

### Incomplete responses

"No answer" means the episode's last turn is not a final answer (`stop_reason` ≠ `answer`). All figures are % of the
6,000 episodes, seeds pooled, computed from `scores.jsonl` and `responses.jsonl`.

| | tool | unforced: no answer | `max_turns` | `length`, turn 1 | `length`, later turn | forced: no answer | `forced_tool_call` | `length`, forced turn | `length`, turn 1 | `length`, later tool turn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | zoom | 38.9 | 36.9 | 0.4 | 1.7 | 7.8 | 4.2 | 1.9 | 0.8 | 1.0 |
| 4B | visual_ops | 34.8 | 23.4 | 8.0 | 3.4 | 21.8 | 8.2 | 2.4 | 5.0 | 6.2 |
| 4B | calculator | 12.6 | 1.2 | 10.9 | 0.5 | 13.1 | 1.3 | 0.1 | 10.9 | 0.8 |
| 4B | ocr | 16.2 | 1.0 | 14.0 | 1.2 | 15.9 | 0.6 | 0.2 | 11.1 | 4.0 |
| 4B | all | 30.5 | 14.3 | 12.1 | 4.2 | 26.9 | 9.0 | 2.6 | 7.8 | 7.5 |
| 9B | zoom | 25.6 | 23.4 | 0.4 | 1.8 | 14.0 | 9.8 | 1.1 | 1.6 | 1.5 |
| 9B | visual_ops | 30.4 | 27.0 | 1.2 | 2.2 | 18.6 | 11.5 | 2.0 | 1.5 | 3.6 |
| 9B | calculator | 10.2 | 2.3 | 6.3 | 1.6 | 10.0 | 1.5 | 0.0 | 7.0 | 1.5 |
| 9B | ocr | 14.1 | 2.9 | 5.3 | 6.0 | 9.6 | 0.3 | 0.1 | 2.8 | 6.3 |
| 9B | all | 38.2 | 34.1 | 1.6 | 2.6 | 23.5 | 15.0 | 3.4 | 0.5 | 4.7 |

- **Unforced stop reasons.** `token_budget` occurs in 2 of the 30,000 unforced 9B tool episodes (in `calculator`) and in
  none at 4B.
- **Forced runs cannot hit the old limits.** Their no-answers are `forced_tool_call` or `length`, since `max_turns` and
  `token_budget` cannot occur.
- **Forcing replaces `max_turns`, not the cut-offs.** Turn-1 cut-offs move by at most 4.3 pp either way (4B `all`
  12.1% → 7.8%). Cut-offs in later, unforced tool turns are higher under forcing in 7 of the 10 conditions; for 4B
  `ocr` they rise from 74 to 242 of 6,000 episodes. This design does not show why.
- **The token-reserve limit never bit.** The design notes a known limit of the token rule: a tool turn capped below
  2,048 tokens by the reserve and cut before it writes a call. Its signature is a `length` stop in an unforced turn at
  exactly 7,168 generated tokens, and no forced episode has it. Only 35 forced episodes in the full run (23 at 4B, 12 at
  9B) generated more than 5,120 tokens at all.

### The forced turn

Computed from `responses.jsonl` (`forced_turn`, `forced_reason`, `stop_reason`) and `scores.jsonl`, seeds pooled. The
last three columns are accuracy (%) on the rows where the forced run had a forced turn, in three runs on the same row and
seed.

| | tool | forced turn (% of episodes) | → answer | → `forced_tool_call` | → `length` | forced run | `no_tool` | unforced base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | zoom | 3,285 (54.8%) | 88.9% | 7.6% | 3.5% | 41.9 | 57.1 | 26.9 |
| 4B | visual_ops | 1,990 (33.2%) | 68.1% | 24.8% | 7.1% | 41.1 | 51.2 | 35.6 |
| 4B | calculator | 125 (2.1%) | 33.6% | 60.8% | 5.6% | 30.4 | 58.4 | 47.2 |
| 4B | ocr | 132 (2.2%) | 64.4% | 28.0% | 7.6% | 30.3 | 43.2 | 47.0 |
| 4B | all | 1,540 (25.7%) | 54.9% | 35.1% | 10.1% | 37.8 | 50.3 | 35.6 |
| 9B | zoom | 2,793 (46.6%) | 76.5% | 21.2% | 2.4% | 40.0 | 59.1 | 32.8 |
| 9B | visual_ops | 2,612 (43.5%) | 68.9% | 26.5% | 4.6% | 40.4 | 54.4 | 32.0 |
| 9B | calculator | 169 (2.8%) | 47.3% | 51.5% | 1.2% | 37.3 | 59.8 | 46.2 |
| 9B | ocr | 84 (1.4%) | 70.2% | 20.2% | 9.5% | 32.1 | 51.2 | 40.5 |
| 9B | all | 3,017 (50.3%) | 63.4% | 29.9% | 6.7% | 42.6 | 55.9 | 29.6 |

- **`forced_reason`.** All 15,747 forced turns in the full run (and all 2,615 in the pilot) were forced by the turn rule
  (`"turns"`), after 5 tool turns. The token rule (`"tokens"`) never fired, and `"calls"` cannot occur with a budget of
  10. No call was skipped for the budget (no `call_budget` record).
- **Under forcing, far fewer 6th turns are tool calls.** Of the episodes that reach a 6th turn, these shares still
  write a tool call (the package as a whole; this design cannot isolate the notice):

  | | zoom | visual_ops | all |
  | --- | --- | --- | --- |
  | 4B, unforced (`max_turns`) → forced (`forced_tool_call`) | 76.5% → 7.6% | 79.1% → 24.8% | 79.9% → 35.1% |
  | 9B, unforced → forced | 67.2% → 21.2% | 69.4% → 26.5% | 73.5% → 29.9% |

  Under forcing, more episodes reach a 6th turn with these three tools; which part of the package causes it, this
  design cannot say. For `calculator` and `ocr` the share stays small and does not consistently rise (4B 1.4% → 2.1%
  and 1.2% → 2.2%; 9B 2.9% → 2.8% and 3.7% → 1.4%). The share of episodes reaching a 6th turn:

  | | zoom | visual_ops | all |
  | --- | --- | --- | --- |
  | 4B | 48.2% → 54.8% | 29.6% → 33.2% | 17.9% → 25.7% |
  | 9B | 34.8% → 46.6% | 38.8% → 43.5% | 46.3% → 50.3% |
- **Rows the unforced run lost to the turn limit.** On rows where the unforced run ended with `max_turns`, accuracy is:

  | | zoom | visual_ops | all |
  | --- | ---: | ---: | ---: |
  | 4B: unforced / forced / `no_tool` | 1.8 / 42.3 / 57.6 | 7.1 / 46.6 / 54.0 | 9.5 / 44.6 / 51.5 |
  | 9B: unforced / forced / `no_tool` | 2.5 / 40.8 / 60.1 | 3.2 / 44.4 / 58.1 | 3.8 / 43.4 / 56.8 |

  These rows account for most of the forced-vs-unforced gain: +14.93 of +14.00 pp (4B `zoom`), +9.23 of +8.65
  (`visual_ops`), +5.02 of +5.85 (`all`); +8.95 of +8.37, +11.10 of +12.42 and +13.50 of +14.83 at 9B. Forcing recovers
  most of what was lost on these rows, but not all: `no_tool` still scores 7–19 pp higher on them.
- **A forced turn that is only a call usually scores 0.** Its final message usually has no answer to extract (1,348 of
  the 1,395 `forced_tool_call` episodes at 4B, 2,271 of the 2,288 at 9B), so it goes to the judge. 140 (4B) and 86 (9B)
  scored correct anyway: 29 and 5 from a JSON answer the extractor found in the final message before the call, and 111
  and 81 recovered by the judge.
- **These splits are descriptive.** The forced and unforced runs are independent samples, so a split by one run's stop
  reason regresses to the mean. Take 4B `calculator`: the rows the unforced run answered lose 2.53 pp under forcing,
  the rows it cut at the length limit gain 1.90 pp, and its `max_turns` rows gain 0.45 pp, for a net −0.18 pp.

### Forced vs `no_tool`: where the remaining loss is

`scripts/decompose_losses.py runs/full/<model>` (paired rows, seeds pooled; this output is not in the reports). The
script counts `forced_tool_call` as no answer.

| | condition | no answer | Δ (pp) | Δ from no-answer rows | Δ from answered rows | answered rows: acc | `no_tool`, same rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | zoom_forced | 7.8% | −11.62 | −3.72 | −7.90 | 53.9 | 62.5 |
| 4B | visual_ops_forced | 21.8% | −5.83 | −5.92 | +0.08 | 67.9 | 67.8 |
| 4B | calculator_forced | 13.1% | −2.50 | −3.02 | +0.52 | 66.9 | 66.3 |
| 4B | ocr_forced | 15.9% | −5.45 | −3.62 | −1.83 | 65.2 | 67.4 |
| 4B | all_forced | 26.9% | −3.82 | −5.95 | +2.13 | 73.7 | 70.7 |
| 9B | zoom_forced | 14.0% | −10.77 | −7.30 | −3.47 | 62.9 | 66.9 |
| 9B | visual_ops_forced | 18.6% | −6.20 | −7.72 | +1.52 | 71.2 | 69.3 |
| 9B | calculator_forced | 10.0% | −0.90 | −2.25 | +1.35 | 70.8 | 69.3 |
| 9B | ocr_forced | 9.6% | −2.60 | −2.18 | −0.42 | 68.4 | 68.9 |
| 9B | all_forced | 23.5% | −5.43 | −9.48 | +4.05 | 75.5 | 70.2 |

- For 4B `zoom_forced`, the loss now sits mostly on answered rows: −7.90 of −11.62 pp, and the answered rows score 53.9
  vs 62.5 for `no_tool` on the same rows. At 9B the answered rows carry about a third of it (−3.47 of −10.77 pp). On
  these rows `zoom_forced` scores below `no_tool` (53.9 vs 62.5 at 4B, 62.9 vs 66.9 at 9B). This is descriptive: the
  runs differ in more than the zoom, and the forced run decides which rows it answers.
- For the other forced conditions, the no-answer rows carry all or most of the loss. Their answered rows are at or above
  `no_tool` on the same rows, except `ocr_forced` (−2.2 pp at 4B, −0.5 pp at 9B). That comparison is descriptive: the
  forced run decides which rows it answers.

## Decision rule for Phase 2

| rule | result |
| --- | --- |
| (a) beats `no_tool` and `no_tool_budget` by ≥ 2 pp, Holm p < 0.05 | **not met** by any condition, unforced or forced. Every point estimate is negative against both baselines. The best is 9B `calculator_forced`: −0.90 pp vs `no_tool` (not significant in any seed) and −6.20 pp† vs `no_tool_budget` (significant in every seed). The best at 4B is `calculator`: −2.32 / −7.40 pp. |
| (b) `tool_use_rate` ≥ 10% | met by every condition, so the tools were used. Unforced: 11.5–96.3% (4B), 50.6–96.1% (9B). Forced: 23.5–96.2% (4B), 40.4–96.4% (9B). |
| (c) the sign holds for the other size | n/a: no condition passes (a). The point estimates are negative at both sizes for every tool, forced and unforced. |

The design also allows a small Phase 2 pilot for a weaker signal, such as gains confined to one or two domains. No
domain-level gain appears. Source: `scripts/decompose_losses.py runs/full/<model> --by-domain`, normal-approximation
CIs, uncorrected.
- There are 220 per-domain deltas vs `no_tool` (10 tool conditions × 11 domains × 2 models). 18 are positive. None of
  them has a 95% CI that excludes zero, even uncorrected. 128 are significantly negative, uncorrected.
- The largest positive delta is 9B `visual_ops_forced` on `puzzles`: +3.33 pp, CI [−2.73, +9.39], 120 rows.
  `no_tool_budget` gains +8.61 pp on 9B `puzzles` (`compare.md`).

## Why the tools help or hurt (Q3)

Unforced tool conditions vs `no_tool`, each row paired with the same row and seed of `no_tool`, all 3 seeds × 2,000
rows. Reproduce with `scripts/decompose_losses.py runs/full/<model>` (not in the reports).

| | condition | no answer¹ | Δ (pp) | Δ from no-answer rows | Δ from answered rows | answered rows: acc | `no_tool`, same rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | zoom | 38.9% | −25.62 | −21.32 | −4.30 | 57.7 | 64.7 |
| 4B | visual_ops | 34.8% | −14.48 | −13.32 | −1.17 | 68.0 | 69.7 |
| 4B | calculator | 12.6% | −2.32 | −2.73 | +0.42 | 66.9 | 66.4 |
| 4B | ocr | 16.2% | −6.17 | −3.95 | −2.22 | 64.7 | 67.4 |
| 4B | all | 30.5% | −9.67 | −8.75 | −0.92 | 69.5 | 70.8 |
| 9B | zoom | 25.6% | −19.13 | −14.22 | −4.92 | 61.1 | 67.7 |
| 9B | visual_ops | 30.4% | −18.62 | −15.57 | −3.05 | 65.3 | 69.7 |
| 9B | calculator | 10.2% | −3.27 | −3.08 | −0.18 | 68.4 | 68.6 |
| 9B | ocr | 14.1% | −6.53 | −4.20 | −2.33 | 67.3 | 70.1 |
| 9B | all | 38.2% | −20.27 | −18.98 | −1.28 | 70.1 | 72.2 |

¹ `stop_reason` ≠ `answer`; the composition is in [Incomplete responses](#incomplete-responses). `no_tool` itself ends
without an answer in 19.3% (4B) and 17.2% (9B) of episodes, all of them cut at 2,048 tokens.

Sources: decomposition from `scripts/decompose_losses.py`; call counts and cut-offs from `scores.jsonl` and
`responses.jsonl`; format and tool error rates from `report.md`.

- **Not stopping is most of the loss.** The no-answer rows make up 64–94% of the drop in nine of the ten conditions.
  For 4B `calculator` they make up more than all of it (−2.73 of −2.32 pp), because its answered rows are slightly
  better than `no_tool` on the same rows.
- **"No answer" means different things for different tools.**
  - For `zoom` and `visual_ops` at both sizes and 9B `all`, it is mostly the turn limit: the model is still calling
    tools at turn 6 (`max_turns` in 23.4–36.9% of episodes, 67–95% of the no-answers). 4B `all` is split: 14.3%
    `max_turns` vs 16.3% length cut-offs (12.1% in turn 1, 4.2% later).
  - For `calculator` and `ocr`, it is mostly the 2,048-token turn limit. A turn of long reasoning is cut before it writes
    a call or an answer: 10.9% (4B `calculator`) and 14.0% (4B `ocr`) of episodes are cut in turn 1. For 9B `ocr`,
    5.3% are cut in turn 1 and 6.0% in later turns.
- **Answered rows are also worse, mainly with zoom.** On the same rows as `no_tool`:
  - `zoom`: −7.0 pp (4B), −6.6 pp (9B);
  - `visual_ops`: −1.8, −4.4;
  - `ocr`: −2.6, −2.7;
  - `all`: −1.3, −2.1;
  - `calculator`: +0.5, −0.2.

  This comparison is descriptive.
- **Tool use ranges from low to near-universal, and the calls are well-formed.**
  - Tool use rates: 4B 11.5% (`ocr`) to 96.3% (`zoom`); 9B 50.6% (`calculator`) to 96.1% (`zoom`).
  - Requested calls per episode: 0.21–4.22 (4B), 0.83–3.93 (9B). Executed: 0.20–3.85 and 0.80–3.58.
  - Parse errors are rare: 0–54 per 6,000 episodes.
  - The calculator is the error-prone tool. 18.4–19.5% of executed calculator calls fail for 9B, and 7.8–9.1% for 4B
    (per seed, `report.md`). For 9B, 552 of the 981 failures are unsupported characters and 191 are unknown names, e.g.
    an equation in `x`.
- **Qwen3.5 rarely exceeds the per-turn cap, unlike Qwen2.5-VL.**
  - The loop executes at most 2 calls per turn. The cap skips at most 2.47% of requested calls in an unforced condition
    (4B `calculator`; 1.41% for 9B `calculator`, 0–0.32% in the others) and at most 5.54% in a forced one
    (4B `calculator_forced`).
  - At most 0.55% of episodes have any cap skip.
  - In the unforced conditions, the gap between requested and executed calls is mostly calls written in the 6th turn,
    which are never executed (51–100% of unexecuted calls). The same holds for most forced conditions, but not for
    `calculator_forced`, where it is mostly per-turn-cap skips (167 of 248 at 4B, 182 of 282 at 9B), or for 9B
    `ocr_forced` (of its 55 unexecuted calls, 29 name a nonexistent `answer` function and 17 are forced-turn calls).
    Qwen2.5-VL's per-turn cap skipped 22.7–78.9% of requested calls (see [Phase 1](#compared-with-qwen25-vl-phase-1)).
- **Format.** The final message is well-formed JSON in 60.8–89.8% of unforced tool episodes, against 80.7%/82.8% for
  `no_tool` and 95.8%/95.5% for `no_tool_budget` (4B/9B). Forcing raises it where the turn limit was the problem:

  | | zoom | visual_ops | all |
  | --- | --- | --- | --- |
  | 4B | 60.8 → 92.1 | 65.0 → 78.2 | 69.5 → 73.1 |
  | 9B | 74.4 → 86.0 | 69.6 → 81.4 | 61.7 → 76.5 |

## Compared with Qwen2.5-VL (Phase 1)

Descriptive only: different model families, run on different days. Sources: [Phase 1 FINDINGS](../phase1/FINDINGS.md)
and the same per-row computations on `runs/full/qwen2.5-vl-*`.

What is the same and what differs:
- **Same:** TRACE decoding (temperature 0.6, top_p 0.95), 2,048 tokens per turn, 6 turns, 8,192 tokens in total, 2
  calls per turn, the TRACE system prompt, the conditions, the judge and the scoring.
- **Model family:** Qwen3.5 with thinking off vs Qwen2.5-VL Instruct.
- **Tool calls:** Qwen3.5's own XML format and tool block vs the Hermes JSON block.
- **Coordinates:** relative [0, 1000] boxes vs absolute pixels.
- **Patch grid:** 32-px vs 28-px.
- **Server context:** 65,536 vs 32,768 tokens. No Qwen3.5 request came near 32,768: the largest final prompt in any
  Qwen3.5 episode was 14,233 tokens.
- **Conditions:** the forced conditions exist only for Qwen3.5, and its pilot has 16 conditions rather than 11.

| | Qwen2.5-VL-7B | Qwen2.5-VL-3B | Qwen3.5-9B | Qwen3.5-4B |
| --- | ---: | ---: | ---: | ---: |
| `no_tool` accuracy (full, seeds pooled) | 33.80 | 24.25 | 65.25 | 61.65 |
| `no_tool` cut at 2,048 tokens | 0.3% | 0.5% | 17.2% | 19.3% |
| mean generated tokens, `no_tool` | 193 | 194 | 699 | 763 |
| `no_tool_budget` − `no_tool` (pp) | +0.05 | +0.53 | +5.30 | +5.08 |
| well-formed final JSON, `no_tool` | 99.5% | 98.6% | 82.8% | 80.7% |
| unforced tools vs `no_tool` (pp) | −1.00 to −12.97 | −5.60 to −7.65 | −3.27 to −20.27 | −2.32 to −25.62 |
| unforced tools: no-answer rate | 7.9–47.5% | 13.7–26.0% | 10.2–38.2% | 12.6–38.9% |
| unforced tools: requested calls per episode | 1.40–23.31 | 2.68–10.71 | 0.83–3.93 | 0.21–4.22 |
| unforced tools: requested calls skipped by the per-turn cap | 22.7–77.9% | 33.2–78.9% | 0–1.41% | 0.01–2.47% |
| unforced tools: episodes with a per-turn-cap skip | 2.9–46.1% | 6.1–42.8% | 0–0.25% | 0.02–0.15% |
| unforced tools: parse errors per 6,000 episodes | 781–5,236 | 1,589–3,903 | 0–54 | 2–39 |

- **Accuracy.** Qwen3.5 is far stronger without tools: 65.25 (9B) and 61.65 (4B), against 33.80 (7B) and 24.25 (3B).
  9B loses more than 7B with every tool; 4B loses more than 3B with `zoom`, `visual_ops` and `all`, but less with
  `calculator` (−2.32 vs −5.80 pp) and `ocr` (−6.17 vs −7.65 pp).
- **Different failure modes.** Qwen2.5-VL failed by over-calling: many calls per turn beyond the cap, many malformed
  calls, and turn-limit endings. Qwen3.5 writes well-formed calls, one or two per turn, but still often runs to the turn
  limit with `zoom`, `visual_ops` and `all`. Its long non-thinking reasoning also runs into the 2,048-token turn limit,
  which Qwen2.5-VL almost never reached.
- **Q2 differs.** The budget control was null for Qwen2.5-VL and is +5 pp for Qwen3.5. The difference is the cut-off
  rate, not the tools.
- **Same conclusion.** At both model families and all four sizes, no tool condition beats `no_tool`.

## Pilot diagnostics (1000 rows, seed 42)

Δ is vs `no_tool`, with McNemar p and Holm p from `compare_all.md`. The Holm family is all 15 compared conditions of the
pilot.

| condition | 4B acc | Δ vs no_tool (p; Holm p) | 9B acc | Δ vs no_tool (p; Holm p) | reading |
| --- | ---: | ---: | ---: | ---: | --- |
| no_tool | 61.9 | — | 65.0 | — | |
| no_tool_budget | 66.9 | +5.0 (1.4e-05; 1.1e-04) | 70.9 | +5.9 (3.2e-07; 2.6e-06) | the 2,048-token cut-off, as in the full run |
| oracle_crop | 63.5 | +1.6 (0.29; 0.58) | 67.6 | +2.6 (0.071; 0.28) | no significant gain from a ground-truth crop (one seed; CIs [−1.1, +4.4] and [−0.1, +5.3]); it can also leak the answer |
| zoom_cookbook | 30.0 | −31.9 (9.7e-60; 1.5e-58) | 46.7 | −18.3 (8.2e-25; 1.1e-23) | the cookbook prompt does not rescue zoom (see below) |
| no_tool_lowres | 56.5 | −5.4 (6.3e-04; 0.0031) | 61.1 | −3.9 (0.0078; 0.055) | the 0.4 MP copy costs 4B accuracy (significant after Holm), 9B less (not after Holm) |
| zoom_lowres | 33.8 | −28.1 (1.4e-45; 2.0e-44); −22.7 vs no_tool_lowres² | 46.3 | −18.7 (1.6e-23; 2.0e-22); −14.8 vs no_tool_lowres² | zoom still costs a lot at low resolution |
| zoom | 37.3 | −24.6 (1.6e-36; 2.1e-35) | 46.6 | −18.4 (3.0e-23; 3.6e-22) | for reference |
| zoom_forced | 50.7 | −11.2 (1.1e-09; 1.2e-08) | 54.5 | −10.5 (6.2e-10; 6.2e-09) | |
| visual_ops_forced | 53.6 | −8.3 (1.1e-06; 9.9e-06) | 60.9 | −4.1 (0.017; 0.10) | |
| calculator_forced | 60.6 | −1.3 (0.41; 0.58) | 63.2 | −1.8 (0.25; 0.76) | |
| ocr_forced | 56.0 | −5.9 (1.2e-04; 7.0e-04) | 63.5 | −1.5 (0.32; 0.76) | |
| all_forced | 58.4 | −3.5 (0.028; 0.11) | 61.1 | −3.9 (0.017; 0.10) | |

² † `trace-tool-compare runs/pilot/<model> --baseline no_tool_lowres --conditions zoom_lowres`, exact McNemar p 9.7e-32
(4B) and 6.6e-16 (9B), uncorrected. Not in `compare_all.md`, which compares `zoom_lowres` only with `no_tool`.

- **Zoom at low resolution.** Zoom's effect at low resolution (−22.7, −14.8) is only a little smaller than at full
  resolution (−24.6, −18.4). The per-row difference-in-differences is +1.9 pp (4B, paired-bootstrap 95% CI [−2.9, +6.5])
  and +3.6 pp (9B, [−0.6, +7.9]), computed from the pilot `scores.jsonl`. So zoom does not help even when the input is
  downsampled. The interaction the design's ZwZ-based prediction expects (zoom gains growing at low resolution) is in
  the predicted direction at both sizes but not significant, and the one-seed pilot can neither confirm nor rule out a
  moderate interaction.
- **`zoom_cookbook`.** For 4B, the model calls zoom as often as in `zoom` (tool use 96.7% vs 96.4%), but the final format
  gets worse (47.7% vs 62.2% well-formed) and accuracy falls 7.3 pp further. For 9B it calls zoom less (83.6% vs 95.6%) at
  the same accuracy (46.7 vs 46.6). Prompt wording moves behaviour, as in Phase 1.
- **The forced conditions in the pilot** (`compare_forced_vs_unforced.md`, one seed). Gains over the unforced base:

  | | zoom | visual_ops | calculator | ocr | all |
  | --- | --- | --- | --- | --- | --- |
  | 4B | +13.4 (p 9.8e-13) | +7.0 (8.8e-05) | +2.0 (0.19) | +0.6 (0.75) | +5.8 (3.9e-04) |
  | 9B | +7.9 (8.2e-06) | +14.7 (4.9e-18) | −0.7 (0.67) | +6.7 (6.4e-06) | +14.1 (2.0e-16) |

  This is the full run's pattern, except 9B `calculator` (−0.7 here, +2.37 pp in the full run).

## Judge pass

Computed from `judge_results.jsonl` (one record per deduplicated request) and the `official_source` field of
`scores.jsonl`. `judge_pending` is 0 in every pilot and full run, and `judge_resolved` > 0 in all 104 of them.

| model, stage | scored rows | rows needing the judge | judge requests | ok | missing | ambiguous | invalid | rows resolved (correct) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B pilot | 16,000 | 3,634 (22.7%) | 3,632 | 772 | 2,381 | 311 | 168 | 773 (256) |
| 4B full | 72,000 | 14,412 (20.0%) | 14,383 | 3,446 | 9,056 | 1,334 | 547 | 3,450 (1,204) |
| 9B pilot | 16,000 | 3,054 (19.1%) | 3,054 | 536 | 2,142 | 226 | 150 | 536 (167) |
| 9B full | 72,000 | 12,940 (18.0%) | 12,924 | 2,460 | 9,014 | 855 | 595 | 2,460 (734) |
| total | 176,000 | 34,040 (19.3%) | 33,993 | 7,214 | 22,593 | 2,726 | 1,460 | 7,219 (2,361) |

- No request was rejected: none exceeded the judge's 16,384-token context. Three requests per model failed transiently
  (`APITimeoutError`), all in the full run. The 9B ones succeeded on a rerun in the same step. The 4B ones came in the
  cancelled one-card attempt and were retried when the step resumed on 8 cards.
- The 26,821 rows that stayed unresolved score 0.

**Why 3.3× as many judge requests as Phase 1.** Phase 1 had 10,201 requests, from 13,162 rows needing the judge out of
106,000 scored rows. Three things add up:
- **More rows.** 176,000 scored rows vs 106,000: 16 pilot and 12 full conditions vs 11 and 7.
- **A higher share needs the judge.** It is 19.3% vs 12.4%, driven by the length cut-offs:
  - Almost every row that needs the judge has `extraction_status` `missing` (34,001 of 34,040): no JSON answer was
    found.
  - 99.1% of the rows needing the judge did not end in an answer. By stop reason: `length` 48.7% (16,567 rows),
    `max_turns` 38.0%, `forced_tool_call` 12.4%, `answer` 0.9%.
  - In Phase 1, `length` made up 2.2% of such rows and `max_turns` 79.9%.
  - A cut-off reply is usually a reasoning trace with no final answer in it. 76.7% of the length-cut rows sent to the
    judge stay unresolved (3,864 of 16,567 resolved).
- **Almost no deduplication.** Requests are deduplicated by (answer type, response) within a model and stage. Qwen3.5's
  long replies are nearly all distinct: 34,040 rows gave 33,993 requests. Phase 1's 13,162 rows gave 10,201.

The judge's yield was a little higher than in Phase 1: 21.2% of requests came back `ok`, vs 16.6% (1,689 of 10,201).

## Caveats

- **Zero-shot only.** This measures what the base models do with tools untrained. The design treats that as a lower
  bound on what RL could teach.
- **Decoding and thinking.** Every condition uses TRACE decoding, not the non-thinking preset the Qwen3.5 model card
  recommends, and thinking is off. Either choice could change both the baselines and tool use. Thinking variants were
  excluded by design.
- **The 2,048-token turn limit cuts long non-thinking reasoning.**
  - It cuts 17–19% of `no_tool` episodes. That depresses the TRACE-protocol baseline by about 5 pp (see Q2).
  - The same limit caps every tool turn, so it also hurts the tool conditions: at 4B every unforced tool condition
    except `zoom` (11–16% of episodes cut), at 9B mainly unforced `calculator` and `ocr` (8% and 11%). Under forcing,
    3.7–17.9% of episodes are cut at 4B and 4.2–9.3% at 9B, where the four non-zoom tools are all at 7.1–9.3%.
  - For Qwen3.5, `no_tool_budget` is the more informative baseline, and the tools are further behind it.
- **What forced vs unforced measures.**
  - A forced condition differs from its base in prompt text (budget statement, countdown, final notice) and in the
    forcing (reserve, forced last turn) together. This design cannot separate them.
  - The stated budget of 10 never binds. Every forced turn in these runs was forced by the turn rule; the token rule
    never fired.
  - So this is "a stated budget plus a forced 6th turn", not a test of a binding call budget.
- **Pilot diagnostics have one seed.** `oracle_crop` uses ground-truth annotations and can leak the answer.
  `zoom_cookbook` shows that prompt wording alone moves behaviour. No condition isolates the tool block without tools to
  call.
- **The decompositions and row splits are descriptive.** The run being split decides which rows end without an answer.
  Paired runs are also independent samples: the same per-request seed does not reproduce text across conditions. So
  splits by one run's outcome regress to the mean.
- **The loop settings were not varied.** 6 turns, 2,048 tokens per turn, 8,192 in total and 2 calls per turn are the
  design's defaults, as in Phase 1.
- **The code changed during the run** (below). According to the operator, the changes were to comments, docs and
  validation only. The earlier tree was not kept, so this write-up could not diff it.

## How it was run

- **Models.** `Qwen/Qwen3.5-4B@851bf6e` and `Qwen/Qwen3.5-9B@c202236`. There is no Qwen3.5-8B; 9B was chosen as the
  larger size.
  - Thinking off: every request sends `chat_template_kwargs: {"enable_thinking": false}`.
  - TRACE decoding (temperature 0.6, top_p 0.95, top_k −1) in every condition. The model card recommends a different
    non-thinking preset.
  - Tool calls use Qwen3.5's XML format and are parsed client-side; the server runs no tool parser.
  - Boxes are in relative [0, 1000] coordinates. A GPU grounding probe confirmed this: both models returned boxes within
    a few units of the relative-[0, 1000] box in 12 of 12 answers.
  - All 136 Qwen3.5 `run_config.json` files record these settings and TRACE dataset revision `4e5b543`. The settings,
    including the system-prompt hash, tool schemas and forced-prompt hash, are identical between the two models and
    between pilot and full for every condition.
- **Judge.** `Qwen/Qwen3-32B@9216db5`, served with `--chat-template-content-format string` and a 16,384-token context.
- **Software.** vLLM 0.30.0, torch 2.13.0 (CUDA 13.0), Python 3.12.12.
  - The pilot and full generation servers had a 65,536-token context (Qwen2.5-VL used 32,768). They ran data-parallel,
    one replica per card. The smoke gate ran at 32,768: the 65,536 setting in `scripts/_model_env.sh` was added at
    00:54, between the gate and the main runs (scripts are outside `code_sha256`). No smoke prompt exceeded 5,893
    tokens.
  - The judge ran tensor-parallel over 8 cards, except in the first 4B judge attempt, which used one card.
- **Gate.** The smoke test (all 16 conditions × 20 rows, both models) passed. Inside the gate, the processor checks of
  `scripts/check_template.py` passed for both models, but its live-server check was skipped (HTTP 400) for both. A
  separate live-server check was run during the pilot against the running 9B server, with `check_template.py` as
  edited at 01:04, and passed: vLLM renders our flattened messages identically, including `enable_thinking=false` and
  a forced-mode notice. There is no passing live-server check for 4B.
- **Schedule (2026-10-06).** Every step ran on held allocations through the queue supervisor; times are from its log.
  Generation ran from 00:57 to 04:53, and the judges finished at 05:22 (4B) and 06:11 (9B).

  | step | hardware | time | wall clock |
  | --- | --- | --- | --- |
  | check + smoke 4B | 1 × RTX PRO 6000 (96 GB) | 00:38–00:47 | 9 min |
  | check + smoke 9B | 2 × RTX PRO 6000 | 00:38–00:52 | 14 min |
  | pilot 9B | 2 × RTX PRO 6000 | 00:57–02:25 | 1 h 28 min |
  | pilot 4B | 1 × RTX PRO 6000 | 00:57–02:41 | 1 h 43 min |
  | full 4B | 4 × RTX PRO 6000 | 00:57–03:19 | 2 h 22 min |
  | full 9B | 8 × H200 (141 GB) | 01:01–04:53 | 3 h 51 min |
  | judge + compare 4B | 1 × RTX PRO 6000, then 8 × RTX PRO 6000 | 03:21–04:54, then 04:56–05:22 | 1 h 33 min (cancelled) + 26 min |
  | judge + compare 9B | 8 × H200 | 04:56–06:11 | 1 h 15 min |

  - **Full 9B.** A first attempt on 4 RTX PRO 6000 cards (00:57) was cancelled by the operator about 2 minutes after
    launch (at 00:59:32), to move it to an idle 8 × H200 allocation. Its server was still starting, so it generated no rows: all 36 full-9B run
    directories were created by the second attempt, from 01:07 on.
  - **Judge 4B.** On one card the 32B judge was KV-cache-bound (about 97% KV usage). It averaged about 0.6 s per item
    and had slowed to about 3 s per item in its last minutes, with about 8,250 full-run items left, so the step was
    cancelled and resumed on 8 cards; the judge results finished so far were kept.
- **Code.** The runs record commit `faf9069` with `git_dirty: true`: an uncommitted working tree on top of that commit.
  - `code_sha256` (a hash of `src/trace_tool`, `configs` and `prompts`) has two values. `5e4ab1bd…` appears in 100 run
    directories (smoke and pilot of both models, and full 4B). `c9b94f66…` appears in the 36 full-9B directories.
  - The hash is taken when `trace-tool-eval` starts and written when a run directory is first created. A resume with a
    different hash keeps the old file and logs a warning, and no such warning appears in the logs.
  - So the hashed tree changed between the start of full 4B's `trace-tool-eval` (01:01, the last process to record
    `5e4ab1bd…`) and the start of the second full-9B attempt's `trace-tool-eval` (01:07, after its server came up).
    File mtimes put one hashed edit in that window, `src/trace_tool/conditions.py` at 01:01. `src/trace_tool/parsing.py`
    was modified at 01:07, within seconds of the 9B hash, so `c9b94f66…` may or may not include it. According to the
    operator, these edits (like the other mid-run edits) touched comments, docs and validation only.
  - The recorded settings, prompt hashes and tool schemas are identical across the two hashes (see Models). The current
    working tree hashes to a third value, `4847b579…`: `configs/conditions.yaml` (01:09) and `src/trace_tool/agent.py`
    (01:12) were edited after every hash was taken.
- **Validation.**
  - Every condition × seed has the expected rows: 1,000 in the pilot, 2,000 in the full run. `responses.jsonl` has
    exactly that many lines.
  - No row has an endpoint error, and no judge item is pending.

## Files

`<model>/<stage>/` with `<model>` ∈ {`qwen3.5-4b`, `qwen3.5-9b`} and `<stage>` ∈ {`pilot`, `full`}: 24 reports. Each is
byte-identical to the one the run wrote under `runs/<stage>/<model>/`.

| file | contents |
| --- | --- |
| `report.md` | one row per condition × seed: accuracy, format, tool use, calls, errors, turns, tokens, latency; then accuracy by domain |
| `compare.md` | paired comparison vs `no_tool`; the Holm family is the core tool conditions + `no_tool_budget` (decision rule (a)) |
| `compare_vs_budget.md` | the core tool conditions vs `no_tool_budget` |
| `compare_all.md` | every condition vs `no_tool` in one Holm family (in the pilot, the diagnostics are included) |
| `compare_forced.md` | the five `*_forced` conditions vs `no_tool`, as their own Holm family |
| `compare_forced_vs_unforced.md` | each `<tool>_forced` vs its base `<tool>`, one compare call per pair, concatenated (so Holm p = raw p) |

The per-row data (`runs/`, gitignored) is not committed.

### Extra tests

These were run for this write-up, from the repository root. Their outputs were written outside the repository and are
not committed.
- † Forced conditions vs `no_tool_budget`:
  `trace-tool-compare runs/full/<model> --baseline no_tool_budget --conditions zoom_forced,visual_ops_forced,calculator_forced,ocr_forced,all_forced`
  (Holm across the 5, per seed).
- † `zoom_lowres` vs `no_tool_lowres` in the pilot:
  `trace-tool-compare runs/pilot/<model> --baseline no_tool_lowres --conditions zoom_lowres`.
- † Holm across the 5 forced-vs-unforced pairs within each model and seed. It uses the exact McNemar p of
  `compare_forced_vs_unforced.md`, with b/c recomputed from `scores.jsonl`, and the same Holm procedure as
  `trace-tool-compare`.
- Loss decompositions:
  `python scripts/decompose_losses.py runs/full/<model> --conditions <the 5 tools and the 5 forced>`, plus
  `--by-domain` for the domain-level check.
- Row-level counts from `scores.jsonl` and `responses.jsonl`: stop reasons, forced turns, cut-offs, per-turn-cap skips,
  calls, judge sources, and the Q2 split. The pilot difference-in-differences uses a paired bootstrap over rows (10,000
  resamples).
