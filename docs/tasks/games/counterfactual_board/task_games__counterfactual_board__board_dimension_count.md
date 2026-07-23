# `task_games__counterfactual_board__board_dimension_count`

## Program Contract

Program: `count(board_grid.visible_units, unit=row|column, style=chess_checkers|sudoku); scene=counterfactual_board; scope=board_dimension_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `board_dimension_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `board_grid`, `visible_units`, `unit`, `row`, `column`, `style`, `chess_checkers`, `sudoku`, `counterfactual_board`, `board_dimension_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; image-pixel row or column bboxes for all counted visible board units; bbox-set cardinality equals answer.
Query ids: `row_count`, `column_count`.

## Reasoning Operations

Families: `counting`

## 2) Scene + task contract
1. Entities/relations: A visible board grid rendered in a chess/checkers or Sudoku-like style.
2. Supported `query_id` values: `row_count`, `column_count`
3. `answer_gt.type`: `integer`
4. Annotation schema: `bbox_set`
5. Alternate annotation forms: none
6. Annotation witness policy: image-pixel row or column bboxes for all counted visible board units; bbox-set cardinality equals answer.
7. Overlap/touch policy: counted row/column boxes cover the full visible board span for that unit.

## 3) Prompt contract
1. `prompt_bundle_id`: `games_counterfactual_board_v1`
2. `scene_key`: `counterfactual_board`
3. `task_key`: `board_grid_count_query`
4. Prompt query keys: `row_count`, `column_count`
5. Required slots: output-mode slots from the prompt bundle only.
6. JSON example validity rule: bbox-set cardinality equals the integer answer.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local query/style/dimension namespaces plus shared counterfactual-board layout/style/noise namespaces.
2. Unique-answer policy: the sampled visible row or column count directly determines the answer.
3. Reject/resample conditions: invalid explicit style, dimensions, or query values raise and retry/propagate.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed.
