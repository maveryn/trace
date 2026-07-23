# `task_games__tetris__drop_collision_time_value`

## Contract
1. Domain: `games`
2. Scene: `tetris`
3. Scene id: `tetris`
4. Public task id: `task_games__tetris__drop_collision_time_value`
5. Supported `query_id` values: `no_shift_collision_time`, `left_shift_collision_time`, `right_shift_collision_time`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set_map`

## Program Contract

Program: `successful_downward_steps(simulate_horizontal_shift_then_vertical_drop(board, falling_piece, requested_shift)); scene=tetris; scope=drop_collision_time_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `drop_collision_time_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `simulate_horizontal_shift_then_vertical_drop`, `board`, `falling_piece`, `requested_shift`, `tetris`, `drop_collision_time_value` plus the active `query_id` branch.
Operation: evaluate `successful_downward_steps` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `no_shift_collision_time`, `left_shift_collision_time`, `right_shift_collision_time`.

## Reasoning Operations

Families: `filtering`, `state_update`

## Generation Notes
1. Query ids choose no shift, left shift, or right shift before the vertical drop.
2. A timestep is one successful downward move by one row; the failed collision move is not counted.
3. Annotation maps `start_piece` to the falling-piece cells and `stop_witness` to the locked cells that stop the shifted drop.
4. Scalar annotation checked: true.
