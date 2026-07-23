# `task_games__pacman__next_item_label`

## Contract
1. Domain: `games`
2. Scene id: `pacman`
3. Public task id: `task_games__pacman__next_item_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `point`

## Program Contract

Program: `label(first(route_item in labeled_bonus_items ordered by route_position)); scene=pacman; scope=next_item_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `next_item_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `first`, `route_item`, `labeled_bonus_items`, `ordered`, `by`, `route_position`, `pacman`, `next_item_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `topology`

## Generation Notes
1. The highlighted route starts at the visible Pac-Man marker.
2. The answer is the label of the first visible labeled bonus item reached along the highlighted route.
3. Annotation is a scalar point at the selected bonus item center.
