# `task_games__dominoes__higher_sum_than_reference_count`

## Contract
1. Domain: `games`
2. Scene: `dominoes`
3. Scene id: `dominoes`
4. Public task id: `task_games__dominoes__higher_sum_than_reference_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(domino_tiles, compare(sum(pips(tile)), sum(pips(reference_tile)), direction=greater_than))); scene=dominoes; scope=higher_sum_than_reference_count`

## Program Contract

Program: `count(filter(domino_tiles, compare(sum(pips(tile)), sum(pips(reference_tile)), direction=greater_than))); scene=dominoes; scope=higher_sum_than_reference_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `higher_sum_than_reference_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `domino_tiles`, `compare`, `sum`, `pips`, `tile`, `reference_tile`, `direction`, `greater_than`, `dominoes`, `higher_sum_than_reference_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `aggregation`

## Generation Notes
1. Renders a face-up domino tableau with one tile marked `REF`; countable tiles are the other visible dominoes.
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
