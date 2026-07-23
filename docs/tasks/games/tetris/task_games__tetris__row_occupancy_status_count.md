# `task_games__tetris__row_occupancy_status_count`

## Contract
1. Domain: `games`
2. Scene: `tetris`
3. Scene id: `tetris`
4. Public task id: `task_games__tetris__row_occupancy_status_count`
5. Supported `query_id` values: `full_row_count`, `one_gap_row_count`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `count(rows where row_occupancy_status(row) = requested_status); scene=tetris; scope=row_occupancy_status_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `row_occupancy_status_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `rows`, `where`, `row_occupancy_status`, `row`, `requested_status`, `tetris`, `row_occupancy_status_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `full_row_count`, `one_gap_row_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. Query ids choose the requested row status: full or exactly one empty cell.
2. Annotation marks whole qualifying row bboxes on the rendered board.
3. Scalar annotation checked: true.
