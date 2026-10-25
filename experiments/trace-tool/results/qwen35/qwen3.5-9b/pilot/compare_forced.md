## qwen3.5-9b (baseline `no_tool`)

Holm adjusts across the compared conditions within each seed.

| condition | seed | n | baseline acc | acc | Δ (pp) | 95% CI | b / c | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| all_forced | 42 | 1000 | 65.00 | 61.10 | -3.90 | [-7.00, -0.80] | 147 / 108 | 0.0172 | 0.0686 |
| calculator_forced | 42 | 1000 | 65.00 | 63.20 | -1.80 | [-4.70, +1.10] | 119 / 101 | 0.252 | 0.503 |
| ocr_forced | 42 | 1000 | 65.00 | 63.50 | -1.50 | [-4.20, +1.20] | 106 / 91 | 0.319 | 0.503 |
| visual_ops_forced | 42 | 1000 | 65.00 | 60.90 | -4.10 | [-7.30, -0.90] | 162 / 121 | 0.0173 | 0.0686 |
| zoom_forced | 42 | 1000 | 65.00 | 54.50 | -10.50 | [-13.70, -7.30] | 197 / 92 | 6.21e-10 | 3.1e-09 |

Seeds pooled (row correctness averaged over shared seeds):

| condition | n rows | baseline acc | acc | Δ (pp) | 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| all_forced | 1000 | 65.00 | 61.10 | -3.90 | [-7.00, -0.80] |
| calculator_forced | 1000 | 65.00 | 63.20 | -1.80 | [-4.70, +1.10] |
| ocr_forced | 1000 | 65.00 | 63.50 | -1.50 | [-4.20, +1.20] |
| visual_ops_forced | 1000 | 65.00 | 60.90 | -4.10 | [-7.30, -0.90] |
| zoom_forced | 1000 | 65.00 | 54.50 | -10.50 | [-13.70, -7.30] |

Δ accuracy (pp) by domain, seeds pooled:

| condition | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_forced | -17.78 | +4.71 | -4.12 | -6.67 | -4.00 | +1.67 | -10.00 | -4.00 | +13.33 | +1.67 | -3.33 |
| calculator_forced | -0.56 | +1.18 | +0.59 | -15.00 | -8.00 | -3.33 | -5.00 | +6.00 | +3.33 | -3.33 | -6.67 |
| ocr_forced | -2.22 | +0.59 | -1.18 | -1.67 | +0.00 | -8.33 | -1.25 | -2.00 | +8.33 | -1.67 | -10.00 |
| visual_ops_forced | -17.78 | +5.88 | -1.76 | -5.00 | +0.00 | -3.33 | -10.00 | +0.00 | -1.67 | +0.00 | -3.33 |
| zoom_forced | -27.22 | -3.53 | -5.88 | -15.00 | -8.00 | -5.00 | -12.50 | -12.00 | +0.00 | -1.67 | -11.67 |

