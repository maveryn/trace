## qwen3.5-9b (baseline `zoom`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| zoom_forced | 42 | 1000 | 46.60 | 54.50 | +7.90 | [+4.50, +11.20] | 115 / 194 | 8.19e-06 | 8.19e-06 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| zoom_forced | 1000 | 46.60 | 54.50 | +7.90 | [+4.50, +11.20] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zoom_forced | +3.89 | +8.82 | +8.24 | +5.00 | +2.00 | +8.33 | +18.75 | +6.00 | +5.00 | +11.67 | +10.00 |


## qwen3.5-9b (baseline `visual_ops`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| visual_ops_forced | 42 | 1000 | 46.20 | 60.90 | +14.70 | [+11.50, +17.90] | 75 / 222 | 4.85e-18 | 4.85e-18 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| visual_ops_forced | 1000 | 46.20 | 60.90 | +14.70 | [+11.50, +17.90] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_ops_forced | +11.67 | +17.65 | +17.65 | +16.67 | +24.00 | +8.33 | +11.25 | +14.00 | +10.00 | +15.00 | +13.33 |


## qwen3.5-9b (baseline `calculator`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| calculator_forced | 42 | 1000 | 63.90 | 63.20 | -0.70 | [-3.40, +2.00] | 100 / 93 | 0.666 | 0.666 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| calculator_forced | 1000 | 63.90 | 63.20 | -0.70 | [-3.40, +2.00] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calculator_forced | +2.78 | -2.35 | +1.18 | -6.67 | -14.00 | +0.00 | +0.00 | +0.00 | +5.00 | -3.33 | +0.00 |


## qwen3.5-9b (baseline `ocr`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| ocr_forced | 42 | 1000 | 56.80 | 63.50 | +6.70 | [+3.80, +9.50] | 75 / 142 | 6.35e-06 | 6.35e-06 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| ocr_forced | 1000 | 56.80 | 63.50 | +6.70 | [+3.80, +9.50] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ocr_forced | +8.33 | +5.88 | +10.00 | +6.67 | +8.00 | -1.67 | +5.00 | +4.00 | +16.67 | +0.00 | +3.33 |


## qwen3.5-9b (baseline `all`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| all_forced | 42 | 1000 | 47.00 | 61.10 | +14.10 | [+10.80, +17.40] | 80 / 221 | 1.95e-16 | 1.95e-16 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all_forced | 1000 | 47.00 | 61.10 | +14.10 | [+10.80, +17.40] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_forced | +17.78 | +15.29 | +8.82 | +18.33 | +8.00 | +16.67 | +3.75 | +8.00 | +23.33 | +16.67 | +20.00 |


