## qwen3.5-4b (baseline `zoom`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| zoom_forced | 42 | 2000 | 36.70 | 49.10 | +12.40 | [+9.90, +14.90] | 213 / 461 | 6.28e-22 | 6.28e-22 |
| zoom_forced | 43 | 2000 | 35.50 | 50.80 | +15.30 | [+12.90, +17.80] | 182 / 488 | 3.47e-33 | 3.47e-33 |
| zoom_forced | 44 | 2000 | 35.90 | 50.20 | +14.30 | [+11.80, +16.75] | 195 / 481 | 8.58e-29 | 8.58e-29 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| zoom_forced | 2000 | 36.03 | 50.03 | +14.00 | [+12.52, +15.55] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zoom_forced | +17.22 | +12.35 | +18.04 | +17.22 | +6.67 | +12.22 | +10.62 | +13.67 | +9.17 | +17.78 | +8.06 |


## qwen3.5-4b (baseline `visual_ops`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| visual_ops_forced | 42 | 2000 | 47.60 | 54.75 | +7.15 | [+4.80, +9.50] | 224 / 367 | 4.39e-09 | 4.39e-09 |
| visual_ops_forced | 43 | 2000 | 47.80 | 55.50 | +7.70 | [+5.30, +10.05] | 222 / 376 | 3.16e-10 | 3.16e-10 |
| visual_ops_forced | 44 | 2000 | 46.10 | 57.20 | +11.10 | [+8.60, +13.55] | 214 / 436 | 2.05e-18 | 2.05e-18 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| visual_ops_forced | 2000 | 47.17 | 55.82 | +8.65 | [+7.23, +10.07] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_ops_forced | +13.70 | +7.75 | +9.41 | +11.67 | +7.67 | +5.28 | +3.54 | +9.00 | +9.17 | +5.00 | +4.72 |


## qwen3.5-4b (baseline `calculator`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| calculator_forced | 42 | 2000 | 58.75 | 59.50 | +0.75 | [-1.25, +2.75] | 208 / 223 | 0.5 | 0.5 |
| calculator_forced | 43 | 2000 | 58.75 | 57.90 | -0.85 | [-2.90, +1.20] | 222 / 205 | 0.439 | 0.439 |
| calculator_forced | 44 | 2000 | 60.50 | 60.05 | -0.45 | [-2.50, +1.55] | 223 / 214 | 0.702 | 0.702 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| calculator_forced | 2000 | 59.33 | 59.15 | -0.18 | [-1.45, +1.02] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calculator_forced | -0.37 | +0.59 | +1.67 | -3.06 | -2.67 | +0.56 | +2.50 | -0.00 | -3.61 | -3.61 | +0.28 |


## qwen3.5-4b (baseline `ocr`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| ocr_forced | 42 | 2000 | 55.25 | 54.65 | -0.60 | [-2.70, +1.50] | 234 / 222 | 0.607 | 0.607 |
| ocr_forced | 43 | 2000 | 55.70 | 56.65 | +0.95 | [-1.15, +3.10] | 227 / 246 | 0.408 | 0.408 |
| ocr_forced | 44 | 2000 | 55.50 | 57.30 | +1.80 | [-0.30, +3.90] | 215 / 251 | 0.105 | 0.105 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| ocr_forced | 2000 | 55.48 | 56.20 | +0.72 | [-0.57, +1.95] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ocr_forced | -2.69 | +0.49 | +0.20 | +0.56 | +1.33 | -0.28 | +3.12 | +0.33 | +8.33 | +5.28 | -1.39 |


## qwen3.5-4b (baseline `all`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| all_forced | 42 | 2000 | 51.40 | 58.90 | +7.50 | [+5.25, +9.80] | 194 / 344 | 1.01e-10 | 1.01e-10 |
| all_forced | 43 | 2000 | 52.55 | 57.10 | +4.55 | [+2.30, +6.75] | 215 / 306 | 7.74e-05 | 7.74e-05 |
| all_forced | 44 | 2000 | 52.00 | 57.50 | +5.50 | [+3.30, +7.75] | 205 / 315 | 1.61e-06 | 1.61e-06 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all_forced | 2000 | 51.98 | 57.83 | +5.85 | [+4.55, +7.18] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_forced | +5.28 | +7.84 | +3.63 | +6.67 | +6.33 | +0.28 | +4.79 | +4.67 | +5.83 | +12.22 | +8.61 |


