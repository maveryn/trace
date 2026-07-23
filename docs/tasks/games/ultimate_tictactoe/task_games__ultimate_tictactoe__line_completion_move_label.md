# `task_games__ultimate_tictactoe__line_completion_move_label`

## Contract
1. Domain: `games`
2. Scene: `ultimate_tictactoe`
3. Scene id: `ultimate_tictactoe`
4. Public task id: `task_games__ultimate_tictactoe__line_completion_move_label`
5. Supported `query_id` values: `x_winning_move_label`, `o_winning_move_label`, `x_blocking_move_label`, `o_blocking_move_label`
6. Answer schema: `string_label`
7. Annotation schema: `bbox`

## Program Contract

Program: `label(filter(highlighted_board_empty_cells, line_completion_move(player, tactic_kind))); scene=ultimate_tictactoe; scope=line_completion_move_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `line_completion_move_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `highlighted_board_empty_cells`, `line_completion_move`, `player`, `tactic_kind`, `ultimate_tictactoe`, `line_completion_move_label` plus the active `query_id` branch.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `x_winning_move_label`, `o_winning_move_label`, `x_blocking_move_label`, `o_blocking_move_label`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. Query ids choose X/O and winning/blocking tactic semantics.
2. Annotation contains the selected option-cell bbox.
3. Query ids are internal replay keys and do not define public task units.
4. `scalar_annotation_checked=true`
