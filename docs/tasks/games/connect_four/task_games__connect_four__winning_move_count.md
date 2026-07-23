# `task_games__connect_four__winning_move_count`

## Contract
1. Domain: `games`
2. Scene id: `connect_four`
3. Public task id: `task_games__connect_four__winning_move_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(filter(legal_columns, move_result=win_for_current_player)); scene=connect_four; scope=winning_move_count`

## Program Contract

Program: `count(filter(legal_columns, move_result=win_for_current_player)); scene=connect_four; scope=winning_move_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `winning_move_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `legal_columns`, `move_result`, `win_for_current_player`, `connect_four`, `winning_move_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
