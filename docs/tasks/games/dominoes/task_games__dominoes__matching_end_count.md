# `task_games__dominoes__matching_end_count`

## Contract
1. Domain: `games`
2. Scene: `dominoes`
3. Scene id: `dominoes`
4. Public task id: `task_games__dominoes__matching_end_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(domino_tiles, matches_open_chain_end(tile)=True)); scene=dominoes; scope=matching_end_count`

## Program Contract

Program: `count(filter(domino_tiles, matches_open_chain_end(tile)=True)); scene=dominoes; scope=matching_end_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `matching_end_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `domino_tiles`, `matches_open_chain_end`, `tile`, `True`, `dominoes`, `matching_end_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `matching`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
