# `task_games__bingo__near_complete_line_count`

## Contract
1. Domain: `games`
2. Scene id: `bingo`
3. Public task id: `task_games__bingo__near_complete_line_count`
4. Supported `query_id` values: `near_complete_row_count`, `near_complete_column_count`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(filter(bingo_lines, unmarked_cell_count(line) = 1)); scene=bingo; scope=near_complete_line_count`

## Program Contract

Program: `count(filter(bingo_lines, unmarked_cell_count(line) = 1)); scene=bingo; scope=near_complete_line_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `near_complete_line_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `bingo_lines`, `unmarked_cell_count`, `line`, `bingo`, `near_complete_line_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `near_complete_row_count`, `near_complete_column_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation marks the single unmarked gap cell for each qualifying near-complete row or column.
