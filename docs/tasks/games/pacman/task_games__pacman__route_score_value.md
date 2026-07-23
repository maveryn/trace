# `task_games__pacman__route_score_value`

## Contract
1. Domain: `games`
2. Scene id: `pacman`
3. Public task id: `task_games__pacman__route_score_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `point_set`

## Program Contract

Program: `sum(score(collectible) for collectible in route_collectibles); scene=pacman; scope=route_score_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `route_score_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `score`, `collectible`, `route_collectibles`, `pacman`, `route_score_value`.
Operation: evaluate `sum` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `topology`, `formula_evaluation`

## Generation Notes
1. Normal pellets on the highlighted route score 1.
2. Printed-value bonus items on the highlighted route score their printed value, sampled from `2, 3, 4`.
3. Annotation points mark the centers of every normal pellet and printed-value bonus item included in the score.
4. The route includes one or two printed-value bonus items; off-route bonus items are distractors and are not included.
