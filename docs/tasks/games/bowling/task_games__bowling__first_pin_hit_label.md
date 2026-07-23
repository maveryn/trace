# `task_games__bowling__first_pin_hit_label`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/bowling/`
3. Scene id: `bowling`
4. Public task id: `task_games__bowling__first_pin_hit_label`
5. Supported `query_id` values: `single`
6. Answer schema: `string_label`
7. Annotation schema: `point`
8. Program schema: `label(first_collision(ball_path, pins)); scene=bowling; scope=first_pin_hit_label`

## Program Contract

Program: `label(first_collision(ball_path, pins)); scene=bowling; scope=first_pin_hit_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `first_pin_hit_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `first_collision`, `ball_path`, `pins`, `bowling`, `first_pin_hit_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `topology`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is one center point for the selected first-hit pin, projected from the same generated game state used for answer verification.
