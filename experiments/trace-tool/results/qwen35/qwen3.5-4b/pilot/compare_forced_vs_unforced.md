## qwen3.5-4b (baseline `zoom`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| zoom_forced | 42 | 1000 | 37.30 | 50.70 | +13.40 | [+9.70, +17.00] | 111 / 245 | 9.77e-13 | 9.77e-13 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| zoom_forced | 1000 | 37.30 | 50.70 | +13.40 | [+9.70, +17.00] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zoom_forced | +15.00 | +15.29 | +14.71 | +15.00 | +10.00 | +6.67 | +7.50 | +22.00 | +13.33 | +21.67 | +0.00 |


## qwen3.5-4b (baseline `visual_ops`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| visual_ops_forced | 42 | 1000 | 46.60 | 53.60 | +7.00 | [+3.50, +10.50] | 121 / 191 | 8.8e-05 | 8.8e-05 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| visual_ops_forced | 1000 | 46.60 | 53.60 | +7.00 | [+3.50, +10.50] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_ops_forced | +12.78 | +8.24 | +5.88 | +3.33 | +6.00 | +1.67 | +16.25 | +4.00 | -5.00 | +1.67 | +6.67 |


## qwen3.5-4b (baseline `calculator`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| calculator_forced | 42 | 1000 | 58.60 | 60.60 | +2.00 | [-0.80, +4.80] | 96 / 116 | 0.192 | 0.192 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| calculator_forced | 1000 | 58.60 | 60.60 | +2.00 | [-0.80, +4.80] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calculator_forced | -0.56 | +0.00 | +2.94 | +6.67 | +4.00 | +0.00 | +3.75 | +10.00 | +5.00 | -3.33 | +1.67 |


## qwen3.5-4b (baseline `ocr`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| ocr_forced | 42 | 1000 | 55.40 | 56.00 | +0.60 | [-2.50, +3.70] | 120 / 126 | 0.75 | 0.75 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| ocr_forced | 1000 | 55.40 | 56.00 | +0.60 | [-2.50, +3.70] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ocr_forced | -1.11 | +0.59 | -1.18 | +10.00 | +4.00 | +5.00 | +1.25 | -4.00 | +8.33 | -8.33 | -1.67 |


## qwen3.5-4b (baseline `all`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| all_forced | 42 | 1000 | 52.60 | 58.40 | +5.80 | [+2.70, +8.90] | 101 / 159 | 0.000388 | 0.000388 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all_forced | 1000 | 52.60 | 58.40 | +5.80 | [+2.70, +8.90] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_forced | +9.44 | +3.53 | +3.53 | +5.00 | +4.00 | -5.00 | +12.50 | +8.00 | +6.67 | +5.00 | +10.00 |


