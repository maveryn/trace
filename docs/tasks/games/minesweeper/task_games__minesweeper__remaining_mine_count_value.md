# `task_games__minesweeper__remaining_mine_count_value`

## Contract
1. Domain: `games`
2. Scene id: `minesweeper`
3. Public task id: `task_games__minesweeper__remaining_mine_count_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `point`

## Program Contract

Program: `difference(clue_value(marked_clue), adjacent_flag_count(marked_clue)); scene=minesweeper; scope=marked_clue_remaining_mine_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_clue_remaining_mine_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `clue_value`, `marked_clue`, `adjacent_flag_count`, `minesweeper`, `marked_clue_remaining_mine_count`.
Operation: evaluate `difference` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Generation Notes
1. The scene marks exactly one opened clue cell.
2. The answer is how many additional adjacent mines are still needed by that marked clue.
3. Annotation is the scalar point at the center of the marked opened clue cell.
