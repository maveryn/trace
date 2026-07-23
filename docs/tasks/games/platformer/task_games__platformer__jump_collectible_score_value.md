# `task_games__platformer__jump_collectible_score_value`

## Contract
1. Domain: `games`
2. Scene id: `platformer`
3. Public task id: `task_games__platformer__jump_collectible_score_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `point_set`
7. Program schema: `sum(score(collectible) for collectible in jump_arc_collectibles); scene=platformer; scope=jump_collectible_score_value`

## Program Contract

Program: `scene=platformer; scope=jump_collectible_score_value; program=sum(score(collectibles_on_shown_jump_arc))`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `jump_collectible_score_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `platformer`, `jump_collectible_score_value`, `program`, `sum`, `score`, `collectibles_on_shown_jump_arc`.
Operation: evaluate `scene=platformer` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Generation Notes
1. Coins on the dashed jump arc score 1.
2. Printed-value bonus items on the dashed jump arc score their printed value, sampled from `2, 3, 4`.
3. The dashed arc includes one or two printed-value bonus items; off-arc bonus items are distractors and are not included.
4. Query ids are internal replay/sampling keys and do not define public task units.
5. Annotation is projected from the same generated game state used for answer verification.
6. Annotation points mark the centers of the scored collectibles on the shown jump arc.
