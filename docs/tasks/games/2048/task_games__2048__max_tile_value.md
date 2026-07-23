# `task_games__2048__max_tile_value`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/2048/`
3. Scene id: `2048`
4. Public task id: `task_games__2048__max_tile_value`
5. Supported `query_id` values: `single`
6. Prompt query key: `max_tile_value`
7. Answer schema: `integer_value`
8. Annotation schema: `bbox_set`
9. Program schema: `value(simulate(board, rules=slide_merge_2048, action=move_direction).final_board, property=max_tile_value); scene=2048; scope=max_tile_value`

## Program Contract

Program: `value(simulate(board, rules=slide_merge_2048, action=move_direction).final_board, property=max_tile_value); scene=2048; scope=max_tile_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `max_tile_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `simulate`, `board`, `rules`, `slide_merge_2048`, `action`, `move_direction`, `final_board`, `property`, `max_tile_value`.
Operation: evaluate `value` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
