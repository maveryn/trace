# `task_games__minesweeper__forced_mine_cell_label`

## Contract
1. Domain: `games`
2. Scene id: `minesweeper`
3. Public task id: `task_games__minesweeper__forced_mine_cell_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `point`

## Program Contract

Program: `select(label for hidden option cell where forced_status(cell)=mine); scene=minesweeper; scope=forced_mine_option_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `forced_mine_option_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `label`, `hidden`, `cell`, `where`, `forced_status`, `mine`, `minesweeper`, `forced_mine_option_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`, `matching`

## Generation Notes
1. The scene shows exactly four labeled hidden cells, `A` through `D`.
2. Exactly one labeled hidden cell is guaranteed to be a mine by the visible clue information.
3. The answer is the option letter printed inside that hidden cell.
4. Annotation is the scalar point at the center of the correct labeled hidden cell.
