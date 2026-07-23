# `task_games__dominoes__double_count`

## Contract
1. Domain: `games`
2. Scene: `dominoes`
3. Scene id: `dominoes`
4. Public task id: `task_games__dominoes__double_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(domino_tiles, left_pips(tile) = right_pips(tile))); scene=dominoes; scope=double_count`

## Program Contract

Program: `count(filter(domino_tiles, left_pips(tile) = right_pips(tile))); scene=dominoes; scope=double_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `double_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `domino_tiles`, `left_pips`, `tile`, `right_pips`, `dominoes`, `double_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. Renders a simple face-up domino tableau with no reference chain; all visible tiles are countable.
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
