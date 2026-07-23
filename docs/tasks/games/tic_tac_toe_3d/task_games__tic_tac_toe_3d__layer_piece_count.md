# `task_games__tic_tac_toe_3d__layer_piece_count`

## Contract
1. Domain: `games`
2. Scene: `tic_tac_toe_3d`
3. Scene id: `tic_tac_toe_3d`
4. Public task id: `task_games__tic_tac_toe_3d__layer_piece_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`

## Program Contract

Program: `count(layer_cells where mark=target_player); scene=tic_tac_toe_3d; scope=layer_piece_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `layer_piece_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `layer_cells`, `where`, `mark`, `target_player`, `tic_tac_toe_3d`, `layer_piece_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. The target player mark is sampled with `target_player` (`X` or `O`).
2. The target layer is sampled from top, middle, and bottom.
3. Annotation is projected from the centers of every matching X or O piece in the requested layer; zero-count answers use an empty `point_set`.
