# `task_games__platformer__collectible_count`

## Contract
1. Domain: `games`
2. Scene id: `platformer`
3. Public task id: `task_games__platformer__collectible_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `point_set`
7. Program schema: `count(filter(collectibles, collected_by_route=True)); scene=platformer; scope=collectible_count`

## Program Contract

Program: `scene=platformer; scope=collectible_count; program=count(collectibles_on_shown_jump_arc)`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `collectible_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `platformer`, `collectible_count`, `program`, `collectibles_on_shown_jump_arc`.
Operation: evaluate `scene=platformer` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
4. Annotation points mark the centers of the counted coins.
