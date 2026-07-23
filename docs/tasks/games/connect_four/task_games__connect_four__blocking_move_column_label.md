# `task_games__connect_four__blocking_move_column_label`

## Contract
1. Domain: `games`
2. Scene id: `connect_four`
3. Public task id: `task_games__connect_four__blocking_move_column_label`
4. Supported `query_id` values: `single`
5. Answer schema: `label_string`
6. Annotation schema: `point`
7. Program schema: `select(column_label, legal_drop_result=blocks_only_immediate_opponent_win); scene=connect_four; scope=blocking_move_column_label`

## Program Contract

Program: `select(column_label, legal_drop_result=blocks_only_immediate_opponent_win); scene=connect_four; scope=blocking_move_column_label`

Candidate set: the visible Connect Four columns labeled below the board.
Operands: visible board state, current player, opponent player, legal drop rule, and opponent immediate-win line.
Operation: evaluate each legal drop column; select the unique column where the current player occupies the opponent's immediate winning landing cell and leaves the opponent with no immediate win.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; one point marks the selected blocking landing cell.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. Column labels are rendered below the board; annotation marks the center of the selected column's landing cell, not the label text.
2. The visible board has exactly one immediate winning column for the opponent and no immediate winning column for the current player.
3. Dropping the current player's disc in the answer column removes the opponent's immediate win.
