# `task_games__connect_four__winning_move_column_label`

## Contract
1. Domain: `games`
2. Scene id: `connect_four`
3. Public task id: `task_games__connect_four__winning_move_column_label`
4. Supported `query_id` values: `single`
5. Answer schema: `label_string`
6. Annotation schema: `point`
7. Program schema: `select(column_label, legal_drop_result=immediate_win_for_current_player); scene=connect_four; scope=winning_move_column_label`

## Program Contract

Program: `select(column_label, legal_drop_result=immediate_win_for_current_player); scene=connect_four; scope=winning_move_column_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `winning_move_column_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `column_label`, `legal_drop_result`, `immediate_win_for_current_player`, `connect_four`, `winning_move_column_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. Column labels are rendered below the board; annotation marks the center of the selected column's landing cell, not the label text.
2. The answer landing cell is not visibly highlighted in the rendered image.
3. Annotation is projected from the same generated game state used for answer verification.
