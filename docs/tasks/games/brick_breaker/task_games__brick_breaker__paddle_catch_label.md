# `task_games__brick_breaker__paddle_catch_label`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/brick_breaker/`
3. Scene id: `brick_breaker`
4. Public task id: `task_games__brick_breaker__paddle_catch_label`
5. Supported `query_id` values: `single`
6. Answer schema: `string_label`
7. Annotation schema: `point`
8. Program schema: `label(paddle_zone_intersected_by(ball_trajectory)); scene=brick_breaker; scope=paddle_catch_label`

## Program Contract

Program: `label(paddle_zone_intersected_by(ball_trajectory)); scene=brick_breaker; scope=paddle_catch_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `paddle_catch_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `paddle_zone_intersected_by`, `ball_trajectory`, `brick_breaker`, `paddle_catch_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is one point at the center of the selected bottom catch lane pad, projected from the same generated game state used for answer verification.
