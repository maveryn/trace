# `task_games__2048__merge_count`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/2048/`
3. Scene id: `2048`
4. Public task id: `task_games__2048__merge_count`
5. Supported `query_id` values: `single`
6. Prompt query key: `merge_count`
7. Answer schema: `integer_count`
8. Annotation schema: `segment_set`
9. Program schema: `count(simulate(board, rules=slide_merge_2048, action=move_direction).merge_events); scene=2048; scope=merge_count`

## Program Contract

Program: `count(simulate(board, rules=slide_merge_2048, action=move_direction).merge_events); scene=2048; scope=merge_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `merge_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `simulate`, `board`, `rules`, `slide_merge_2048`, `action`, `move_direction`, `merge_events`, `merge_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
4. Annotation is a `segment_set`; each segment is `[[x0, y0], [x1, y1]]` and connects the centers of the two original source tile cells that merge. Annotation cardinality equals the merge-count answer.
