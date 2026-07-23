# `task_games__ultimate_tictactoe__small_board_status_count`

## Contract
1. Domain: `games`
2. Scene: `ultimate_tictactoe`
3. Scene id: `ultimate_tictactoe`
4. Public task id: `task_games__ultimate_tictactoe__small_board_status_count`
5. Supported `query_id` values: `x_won_board_count`, `o_won_board_count`, `neither_won_board_count`, `drawn_board_count`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(local_boards, board_status=target_status)); scene=ultimate_tictactoe; scope=small_board_status_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `small_board_status_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `local_boards`, `board_status`, `target_status`, `ultimate_tictactoe`, `small_board_status_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `x_won_board_count`, `o_won_board_count`, `neither_won_board_count`, `drawn_board_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. Query ids choose which small-board status category is counted.
2. Annotation contains one small-board bbox for each counted board.
3. Query ids are internal replay keys and do not define public task units.
4. `scalar_annotation_checked=true`
