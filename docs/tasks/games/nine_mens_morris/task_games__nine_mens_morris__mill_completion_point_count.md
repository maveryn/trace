# `task_games__nine_mens_morris__mill_completion_point_count`

## Contract
1. Domain: `games`
2. Scene id: `nine_mens_morris`
3. Public task id: `task_games__nine_mens_morris__mill_completion_point_count`
4. Supported `query_id` values: `white_mill_completion_point_count`, `black_mill_completion_point_count`
5. Answer schema: `integer_count`
6. Annotation schema: `point_set`

## Program Contract

Program: `count(filter(empty_board_points, completes_mill(point, queried_color)=true)); scene=nine_mens_morris; scope=mill_completion_point_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `mill_completion_point_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `empty_board_points`, `completes_mill`, `queried_color`, `true`, `nine_mens_morris`, `mill_completion_point_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `white_mill_completion_point_count`, `black_mill_completion_point_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
1. Query ids are internal replay/sampling keys and do not define public task units.
2. The two semantic query ids differ only by queried piece color.
3. Annotation is projected from the same generated game state used for answer verification.
