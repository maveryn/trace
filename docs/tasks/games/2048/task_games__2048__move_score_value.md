# `task_games__2048__move_score_value`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/2048/`
3. Scene id: `2048`
4. Public task id: `task_games__2048__move_score_value`
5. Supported `query_id` values: `single`
6. Prompt query key: `move_score_value`
7. Answer schema: `integer_value`
8. Annotation schema: `segment_set`
9. Program schema: `sum(simulate(board, rules=slide_merge_2048, action=move_direction).merge_events.merged_value); scene=2048; scope=move_score_value`

## Program Contract

Program: `sum(simulate(board, rules=slide_merge_2048, action=move_direction).merge_events.merged_value); scene=2048; scope=move_score_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `move_score_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `simulate`, `board`, `rules`, `slide_merge_2048`, `action`, `move_direction`, `merge_events`, `merged_value`, `move_score_value`.
Operation: evaluate `sum` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `state_update`, `formula_evaluation`

## Generation Notes
1. The prompt states the scoring rule: each merge scores the value of the new combined tile, and the move score sums over all merges.
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Generation presamples the target score from a balanced support and constructs merge result tiles whose values sum to that score.
4. Annotation is a `segment_set`; each segment is `[[x0, y0], [x1, y1]]` and connects the centers of the two original source tile cells that merge. Annotation cardinality equals the number of scoring merges.
