# `task_games__bowling__path_hit_count`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/bowling/`
3. Scene id: `bowling`
4. Public task id: `task_games__bowling__path_hit_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(standing_pins, intersects(extended_straight_ball_path, visible_pin_body))); scene=bowling; scope=path_hit_count`

## Program Contract

Program: `count(filter(standing_pins, intersects(extended_straight_ball_path, visible_pin_body))); scene=bowling; scope=path_hit_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `path_hit_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `standing_pins`, `intersects`, `extended_straight_ball_path`, `visible_pin_body`, `bowling`, `path_hit_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`, `topology`

## Generation Notes
1. Query ids are internal replay/sampling keys and do not define public task units.
2. The answer support is exactly `1..5`.
3. Generated hit pins lie close to the path centerline while every non-hit standing pin has a large clearance from the path, avoiding near-miss ambiguity.
4. The rendered cue is a short dashed aiming line in the same style as `task_games__bowling__first_pin_hit_label`; the task counts only direct intersections of that extended straight path with standing pin bodies and does not model chain reactions after contact.
5. Annotation is projected as a `bbox_set` around every counted standing pin from the same generated state used for answer verification.
