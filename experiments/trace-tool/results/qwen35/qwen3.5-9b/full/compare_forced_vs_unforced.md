## qwen3.5-9b (baseline `zoom`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| zoom_forced | 42 | 2000 | 45.95 | 54.25 | +8.30 | [+5.95, +10.65] | 212 / 378 | 8.06e-12 | 8.06e-12 |
| zoom_forced | 43 | 2000 | 45.70 | 54.00 | +8.30 | [+5.95, +10.70] | 223 / 389 | 1.94e-11 | 1.94e-11 |
| zoom_forced | 44 | 2000 | 46.70 | 55.20 | +8.50 | [+6.10, +10.90] | 217 / 387 | 4.46e-12 | 4.46e-12 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| zoom_forced | 2000 | 46.12 | 54.48 | +8.37 | [+6.90, +9.83] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zoom_forced | +11.02 | +4.51 | +4.80 | +19.17 | +11.00 | +11.39 | +10.83 | +7.00 | +5.83 | +8.89 | +5.28 |


## qwen3.5-9b (baseline `visual_ops`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| visual_ops_forced | 42 | 2000 | 47.30 | 58.40 | +11.10 | [+8.70, +13.45] | 191 / 413 | 9.32e-20 | 9.32e-20 |
| visual_ops_forced | 43 | 2000 | 46.95 | 59.90 | +12.95 | [+10.55, +15.30] | 186 / 445 | 1.97e-25 | 1.97e-25 |
| visual_ops_forced | 44 | 2000 | 45.65 | 58.85 | +13.20 | [+10.80, +15.55] | 179 / 443 | 8.93e-27 | 8.93e-27 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| visual_ops_forced | 2000 | 46.63 | 59.05 | +12.42 | [+10.93, +13.88] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_ops_forced | +13.98 | +10.88 | +8.43 | +19.44 | +13.33 | +7.78 | +16.46 | +8.00 | +16.67 | +13.89 | +12.78 |


## qwen3.5-9b (baseline `calculator`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| calculator_forced | 42 | 2000 | 61.75 | 64.20 | +2.45 | [+0.45, +4.45] | 185 / 234 | 0.0189 | 0.0189 |
| calculator_forced | 43 | 2000 | 62.50 | 63.25 | +0.75 | [-1.25, +2.75] | 199 / 214 | 0.491 | 0.491 |
| calculator_forced | 44 | 2000 | 61.70 | 65.60 | +3.90 | [+1.85, +5.85] | 178 / 256 | 0.000211 | 0.000211 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| calculator_forced | 2000 | 61.98 | 64.35 | +2.37 | [+1.20, +3.53] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calculator_forced | +3.24 | +4.51 | +1.18 | +0.83 | -0.33 | +1.67 | +2.71 | -1.67 | +2.50 | +2.50 | +4.17 |


## qwen3.5-9b (baseline `ocr`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| ocr_forced | 42 | 2000 | 58.00 | 63.30 | +5.30 | [+3.15, +7.40] | 186 / 292 | 1.43e-06 | 1.43e-06 |
| ocr_forced | 43 | 2000 | 59.00 | 61.35 | +2.35 | [+0.30, +4.45] | 195 / 242 | 0.0277 | 0.0277 |
| ocr_forced | 44 | 2000 | 59.15 | 63.30 | +4.15 | [+2.05, +6.25] | 192 / 275 | 0.000143 | 0.000143 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| ocr_forced | 2000 | 58.72 | 62.65 | +3.93 | [+2.68, +5.15] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ocr_forced | +3.24 | +4.41 | +3.92 | +6.11 | +7.67 | +4.17 | +2.08 | +0.67 | +6.67 | +3.61 | +1.94 |


## qwen3.5-9b (baseline `all`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| all_forced | 42 | 2000 | 45.20 | 60.55 | +15.35 | [+13.05, +17.70] | 158 / 465 | 5.2e-36 | 5.2e-36 |
| all_forced | 43 | 2000 | 44.90 | 59.50 | +14.60 | [+12.30, +16.95] | 166 / 458 | 1.55e-32 | 1.55e-32 |
| all_forced | 44 | 2000 | 44.85 | 59.40 | +14.55 | [+12.20, +16.85] | 164 / 455 | 1.39e-32 | 1.39e-32 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all_forced | 2000 | 44.98 | 59.82 | +14.83 | [+13.40, +16.32] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_forced | +19.35 | +18.14 | +10.00 | +18.33 | +18.33 | +11.11 | +11.04 | +11.00 | +10.00 | +14.44 | +16.39 |


