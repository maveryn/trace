# `task_games__counterfactual_board__board_line_count`

## Program Contract

Program: `count(board_grid.visible_lines, orientation=horizontal|vertical, style=xiangqi); scene=counterfactual_board; scope=board_line_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `board_line_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `board_grid`, `visible_lines`, `orientation`, `horizontal`, `vertical`, `style`, `xiangqi`, `counterfactual_board`, `board_line_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; image-pixel line segments for all counted visible board lines; segment-set cardinality equals answer.
Query ids: `horizontal_line_count`, `vertical_line_count`.

## Reasoning Operations

Families: `counting`

## 2) Scene + task contract
1. Entities/relations: A visible Xiangqi-like line board with horizontal and vertical grid lines.
2. Supported `query_id` values: `horizontal_line_count`, `vertical_line_count`
3. `answer_gt.type`: `integer`
4. Annotation schema: `segment_set`
5. Alternate annotation forms: none
6. Annotation witness policy: image-pixel line segments for all counted visible board lines; segment-set cardinality equals answer.
7. Overlap/touch policy: each segment runs along the visible counted line from one board edge to the other.

## 3) Prompt contract
1. `prompt_bundle_id`: `games_counterfactual_board_v1`
2. `scene_key`: `counterfactual_board`
3. `task_key`: `board_grid_count_query`
4. Prompt query keys: `horizontal_line_count`, `vertical_line_count`
5. Required slots: output-mode slots from the prompt bundle only.
6. JSON example validity rule: segment-set cardinality equals the integer answer.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local query/dimension namespaces plus shared counterfactual-board layout/style/noise namespaces.
2. Unique-answer policy: the sampled visible horizontal or vertical line count directly determines the answer.
3. Reject/resample conditions: invalid explicit style, dimensions, or query values raise and retry/propagate.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed.
