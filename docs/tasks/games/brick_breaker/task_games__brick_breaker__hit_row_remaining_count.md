# `task_games__brick_breaker__hit_row_remaining_count`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/brick_breaker/`
3. Scene id: `brick_breaker`
4. Public task id: `task_games__brick_breaker__hit_row_remaining_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(bricks_in_hit_row, state=remaining_after_marked_hit)); scene=brick_breaker; scope=hit_row_remaining_count`

## Program Contract

Program: `count(filter(bricks_in_hit_row, state=remaining_after_marked_hit)); scene=brick_breaker; scope=hit_row_remaining_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `hit_row_remaining_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `bricks_in_hit_row`, `state`, `remaining_after_marked_hit`, `brick_breaker`, `hit_row_remaining_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
