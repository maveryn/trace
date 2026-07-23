# `task_games__tic_tac_toe_3d__blocking_move_cell_label`

## Contract
1. Domain: `games`
2. Scene: `tic_tac_toe_3d`
3. Scene id: `tic_tac_toe_3d`
4. Public task id: `task_games__tic_tac_toe_3d__blocking_move_cell_label`
5. Supported `query_id` values: `x_blocking_move_label`, `o_blocking_move_label`
6. Answer schema: `string_label`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `label(unique(candidate_cells where play(target_player, cell) blocks_only_immediate_opponent_3d_tic_tac_toe_line)); scene=tic_tac_toe_3d; scope=blocking_move_cell_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `blocking_move_cell_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique`, `candidate_cells`, `where`, `play`, `target_player`, `cell`, `blocks_only_immediate_opponent_3d_tic_tac_toe_line`, `tic_tac_toe_3d`, `blocking_move_cell_label` plus the active `query_id` branch.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema and contains bboxes for the blocking cell and the two opponent cells in the immediate threat line; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `x_blocking_move_label`, `o_blocking_move_label`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. Query ids are internal replay/sampling keys and do not define public task units.
2. The board is a 3 by 3 by 3 Tic-Tac-Toe state with exactly one immediate opponent winning cell and no immediate winning cell for the target player.
3. Four labeled empty-cell options are shown on the board. The correct option is the single empty cell that blocks the opponent's immediate three-in-a-row.
4. Annotation contains bboxes for the blocking cell and the two opponent cells that form the immediate threat line.
