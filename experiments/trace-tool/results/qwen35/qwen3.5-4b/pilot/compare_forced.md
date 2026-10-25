## qwen3.5-4b (baseline `no_tool`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| all_forced | 42 | 1000 | 61.90 | 58.40 | -3.50 | [-6.50, -0.50] | 138 / 103 | 0.0283 | 0.0566 |
| calculator_forced | 42 | 1000 | 61.90 | 60.60 | -1.30 | [-4.10, +1.50] | 112 / 99 | 0.409 | 0.409 |
| ocr_forced | 42 | 1000 | 61.90 | 56.00 | -5.90 | [-8.80, -2.90] | 144 / 85 | 0.000117 | 0.000351 |
| visual_ops_forced | 42 | 1000 | 61.90 | 53.60 | -8.30 | [-11.60, -5.10] | 185 / 102 | 1.1e-06 | 4.4e-06 |
| zoom_forced | 42 | 1000 | 61.90 | 50.70 | -11.20 | [-14.80, -7.60] | 225 / 113 | 1.11e-09 | 5.56e-09 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all_forced | 1000 | 61.90 | 58.40 | -3.50 | [-6.50, -0.50] |
| calculator_forced | 1000 | 61.90 | 60.60 | -1.30 | [-4.10, +1.50] |
| ocr_forced | 1000 | 61.90 | 56.00 | -5.90 | [-8.80, -2.90] |
| visual_ops_forced | 1000 | 61.90 | 53.60 | -8.30 | [-11.60, -5.10] |
| zoom_forced | 1000 | 61.90 | 50.70 | -11.20 | [-14.80, -7.60] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_forced | -15.56 | +1.18 | +2.35 | -1.67 | -8.00 | -3.33 | +3.75 | -2.00 | -5.00 | -16.67 | +8.33 |
| calculator_forced | -8.33 | -1.18 | +3.53 | +5.00 | -6.00 | -8.33 | -1.25 | +10.00 | +6.67 | -13.33 | +5.00 |
| ocr_forced | -9.44 | -4.12 | -5.88 | +0.00 | -6.00 | -6.67 | -2.50 | -8.00 | +1.67 | -26.67 | +5.00 |
| visual_ops_forced | -21.67 | -1.18 | -4.12 | -16.67 | -12.00 | -13.33 | +0.00 | -8.00 | -6.67 | -15.00 | +10.00 |
| zoom_forced | -31.11 | -1.18 | -5.29 | -13.33 | -24.00 | -5.00 | -16.25 | -8.00 | +3.33 | -13.33 | +1.67 |

