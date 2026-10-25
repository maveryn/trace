| run | model | condition | seed | n | official acc | det. acc | strict acc | format | tool use | calls/ep | tool err | turns | gen tok | median s | pending judge | endpoint errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | all | 42 | 1000 | 18.40 | 18.20 | 18.20 | 73.60 | 78.20 | 6.35 | 32.22 | 3.16 | 362 | 15.5 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | calculator | 42 | 1000 | 18.00 | 17.60 | 17.60 | 80.20 | 27.20 | 2.69 | 72.03 | 2.41 | 332 | 6.9 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | no_tool | 42 | 1000 | 26.50 | 26.30 | 26.40 | 98.70 | 0.00 | 0.00 | 0.00 | 1.00 | 190 | 7.3 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | no_tool_budget | 42 | 1000 | 25.70 | 25.60 | 25.70 | 98.20 | 0.00 | 0.00 | 0.00 | 1.00 | 217 | 2.0 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | no_tool_lowres | 42 | 1000 | 22.10 | 22.00 | 22.00 | 98.60 | 0.00 | 0.00 | 0.00 | 1.00 | 198 | 4.5 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | ocr | 42 | 1000 | 15.70 | 15.60 | 15.60 | 78.70 | 85.90 | 9.42 | 33.52 | 2.88 | 511 | 53.1 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | oracle_crop | 42 | 1000 | 28.80 | 28.50 | 28.60 | 97.90 | 0.00 | 0.00 | 0.00 | 1.00 | 169 | 9.7 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | visual_ops | 42 | 1000 | 16.20 | 15.70 | 15.70 | 77.40 | 77.70 | 10.62 | 14.00 | 2.76 | 720 | 13.2 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | zoom | 42 | 1000 | 20.60 | 20.10 | 20.10 | 86.40 | 78.50 | 10.81 | 15.87 | 2.36 | 732 | 10.5 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | zoom_cookbook | 42 | 1000 | 15.20 | 12.50 | 12.60 | 62.20 | 87.40 | 7.76 | 15.52 | 3.21 | 601 | 12.1 | 0 | 0 |
| runs/pilot/qwen2.5-vl-3b-instruct | qwen2.5-vl-3b-instruct | zoom_lowres | 42 | 1000 | 16.90 | 16.50 | 16.50 | 87.10 | 77.60 | 9.65 | 14.19 | 2.34 | 660 | 7.6 | 0 | 0 |

Official accuracy (%) by domain

| model | condition | seed | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-vl-3b-instruct | all | 42 | 17.78 | 12.94 | 18.24 | 18.33 | 16.00 | 18.33 | 32.50 | 20.00 | 21.67 | 18.33 | 15.00 |
| qwen2.5-vl-3b-instruct | calculator | 42 | 13.33 | 11.76 | 25.88 | 16.67 | 20.00 | 10.00 | 25.00 | 26.00 | 15.00 | 20.00 | 20.00 |
| qwen2.5-vl-3b-instruct | no_tool | 42 | 30.00 | 22.94 | 30.00 | 20.00 | 18.00 | 28.33 | 36.25 | 24.00 | 31.67 | 13.33 | 25.00 |
| qwen2.5-vl-3b-instruct | no_tool_budget | 42 | 28.33 | 20.59 | 30.59 | 21.67 | 16.00 | 21.67 | 38.75 | 22.00 | 26.67 | 16.67 | 28.33 |
| qwen2.5-vl-3b-instruct | no_tool_lowres | 42 | 21.11 | 15.29 | 27.06 | 26.67 | 28.00 | 21.67 | 23.75 | 24.00 | 13.33 | 26.67 | 21.67 |
| qwen2.5-vl-3b-instruct | ocr | 42 | 16.11 | 11.76 | 14.12 | 20.00 | 6.00 | 15.00 | 23.75 | 14.00 | 23.33 | 18.33 | 15.00 |
| qwen2.5-vl-3b-instruct | oracle_crop | 42 | 33.89 | 17.06 | 30.00 | 30.00 | 26.00 | 26.67 | 33.75 | 34.00 | 25.00 | 31.67 | 36.67 |
| qwen2.5-vl-3b-instruct | visual_ops | 42 | 12.22 | 14.71 | 15.88 | 20.00 | 18.00 | 20.00 | 31.25 | 18.00 | 10.00 | 13.33 | 11.67 |
| qwen2.5-vl-3b-instruct | zoom | 42 | 17.22 | 22.94 | 18.82 | 15.00 | 20.00 | 28.33 | 23.75 | 22.00 | 25.00 | 18.33 | 20.00 |
| qwen2.5-vl-3b-instruct | zoom_cookbook | 42 | 12.78 | 14.71 | 20.59 | 21.67 | 18.00 | 6.67 | 15.00 | 10.00 | 21.67 | 8.33 | 13.33 |
| qwen2.5-vl-3b-instruct | zoom_lowres | 42 | 11.11 | 17.65 | 20.00 | 13.33 | 22.00 | 18.33 | 22.50 | 18.00 | 16.67 | 15.00 | 15.00 |
