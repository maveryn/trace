# `task_games__minesweeper__forced_cell_count`

## Contract
1. Domain: `games`
2. Scene id: `minesweeper`
3. Public task id: `task_games__minesweeper__forced_cell_count`
4. Supported `query_id` values: `forced_mine_count`, `forced_safe_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(hidden_cells, forced_status in {mine,safe})); scene=minesweeper; scope=forced_cell_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `forced_cell_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `hidden_cells`, `forced_status`, `mine`, `safe`, `minesweeper`, `forced_cell_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `forced_mine_count`, `forced_safe_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. The scene shows a visible Minesweeper board with opened number cells, hidden cells, and flags.
2. The query branch selects whether to count hidden cells forced to be mines or forced to be safe.
3. Annotation boxes enclose every counted hidden cell.
