# `task_games__space_shooter__enemy_ship_count`

## Contract
1. Domain: `games`
2. Scene: `space_shooter`
3. Scene id: `space_shooter`
4. Public task id: `task_games__space_shooter__enemy_ship_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `count(enemy_ships); scene=space_shooter; scope=enemy_ship_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `enemy_ship_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `enemy_ships`, `space_shooter`, `enemy_ship_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Generation Notes
1. `single` is the only public query id; the task-specific prompt key is trace metadata.
2. Annotation is the bbox set of every visible enemy ship; the player ship and all projectiles are distractors.
3. Red enemy-shot distractor lanes may contain one to three visible shots.
4. Blue player shots are placed below all same-lane enemy ships and red enemy shots.
5. Scalar annotation checked: true.
