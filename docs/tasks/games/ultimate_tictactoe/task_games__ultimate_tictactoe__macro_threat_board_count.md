# `task_games__ultimate_tictactoe__macro_threat_board_count`

## Contract
1. Domain: `games`
2. Scene: `ultimate_tictactoe`
3. Scene id: `ultimate_tictactoe`
4. Public task id: `task_games__ultimate_tictactoe__macro_threat_board_count`
5. Supported `query_id` values: `x_immediate_win_board_count`, `o_immediate_win_board_count`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(local_boards, status=open and immediate_win_exists(player))); scene=ultimate_tictactoe; scope=macro_threat_board_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `macro_threat_board_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `local_boards`, `status`, `open`, `immediate_win_exists`, `player`, `ultimate_tictactoe`, `macro_threat_board_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `x_immediate_win_board_count`, `o_immediate_win_board_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Generation Notes
1. Query ids choose whether X or O immediate-win boards are counted.
2. Annotation contains the small-board bboxes for every counted board.
3. Empty annotation is valid when the answer is `0`.
4. `scalar_annotation_checked=true`
