| run | model | condition | seed | n | official acc | det. acc | strict acc | format | tool use | calls/ep | tool err | turns | gen tok | median s | pending judge | endpoint errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | all | 42 | 1000 | 25.10 | 24.10 | 24.10 | 64.70 | 73.90 | 8.41 | 9.82 | 3.58 | 543 | 8.6 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | calculator | 42 | 1000 | 32.50 | 32.30 | 32.30 | 96.10 | 25.10 | 1.32 | 32.01 | 1.69 | 227 | 4.9 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | no_tool | 42 | 1000 | 34.30 | 34.30 | 34.30 | 99.00 | 0.00 | 0.00 | 0.00 | 1.00 | 199 | 4.6 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | no_tool_budget | 42 | 1000 | 33.90 | 33.90 | 33.90 | 99.30 | 0.00 | 0.00 | 0.00 | 1.00 | 219 | 1.9 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | no_tool_lowres | 42 | 1000 | 32.90 | 32.80 | 32.80 | 99.70 | 0.00 | 0.00 | 0.00 | 1.00 | 194 | 3.6 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | ocr | 42 | 1000 | 31.20 | 30.40 | 30.40 | 75.80 | 78.00 | 4.89 | 6.52 | 3.54 | 283 | 45.3 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | oracle_crop | 42 | 1000 | 39.20 | 38.90 | 38.80 | 98.40 | 0.00 | 0.00 | 0.00 | 1.00 | 149 | 5.2 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | visual_ops | 42 | 1000 | 20.30 | 19.90 | 20.00 | 57.10 | 64.90 | 23.17 | 12.72 | 3.66 | 1720 | 12.1 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | zoom | 42 | 1000 | 28.10 | 27.50 | 27.40 | 78.70 | 43.50 | 9.45 | 6.99 | 2.77 | 896 | 7.1 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | zoom_cookbook | 42 | 1000 | 27.00 | 26.80 | 26.80 | 89.20 | 23.00 | 3.11 | 5.50 | 1.72 | 458 | 5.0 | 0 | 0 |
| runs/pilot/qwen2.5-vl-7b-instruct | qwen2.5-vl-7b-instruct | zoom_lowres | 42 | 1000 | 27.30 | 26.70 | 26.80 | 80.20 | 44.20 | 10.67 | 4.07 | 2.79 | 975 | 6.0 | 0 | 0 |

Official accuracy (%) by domain

| model | condition | seed | charts | games | geometry | graph | icons | illustrations | pages | physics | puzzles | symbolic | three_d |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-vl-7b-instruct | all | 42 | 28.33 | 11.18 | 29.41 | 28.33 | 16.00 | 21.67 | 40.00 | 34.00 | 26.67 | 23.33 | 23.33 |
| qwen2.5-vl-7b-instruct | calculator | 42 | 33.89 | 17.65 | 45.88 | 30.00 | 28.00 | 28.33 | 37.50 | 40.00 | 30.00 | 35.00 | 30.00 |
| qwen2.5-vl-7b-instruct | no_tool | 42 | 45.00 | 18.82 | 38.24 | 31.67 | 20.00 | 33.33 | 45.00 | 48.00 | 30.00 | 30.00 | 33.33 |
| qwen2.5-vl-7b-instruct | no_tool_budget | 42 | 41.11 | 20.00 | 39.41 | 30.00 | 30.00 | 30.00 | 43.75 | 42.00 | 35.00 | 25.00 | 35.00 |
| qwen2.5-vl-7b-instruct | no_tool_lowres | 42 | 46.11 | 22.94 | 42.94 | 25.00 | 24.00 | 18.33 | 40.00 | 36.00 | 31.67 | 20.00 | 25.00 |
| qwen2.5-vl-7b-instruct | ocr | 42 | 25.00 | 24.71 | 44.71 | 31.67 | 26.00 | 26.67 | 30.00 | 40.00 | 30.00 | 38.33 | 26.67 |
| qwen2.5-vl-7b-instruct | oracle_crop | 42 | 46.11 | 30.59 | 39.41 | 40.00 | 36.00 | 35.00 | 42.50 | 46.00 | 45.00 | 26.67 | 45.00 |
| qwen2.5-vl-7b-instruct | visual_ops | 42 | 20.56 | 10.59 | 22.94 | 11.67 | 22.00 | 20.00 | 36.25 | 20.00 | 30.00 | 20.00 | 16.67 |
| qwen2.5-vl-7b-instruct | zoom | 42 | 32.22 | 15.88 | 39.41 | 25.00 | 22.00 | 18.33 | 37.50 | 38.00 | 21.67 | 28.33 | 21.67 |
| qwen2.5-vl-7b-instruct | zoom_cookbook | 42 | 30.56 | 22.94 | 33.53 | 26.67 | 18.00 | 31.67 | 26.25 | 30.00 | 18.33 | 30.00 | 16.67 |
| qwen2.5-vl-7b-instruct | zoom_lowres | 42 | 26.67 | 22.35 | 38.24 | 33.33 | 8.00 | 20.00 | 28.75 | 32.00 | 26.67 | 26.67 | 25.00 |
