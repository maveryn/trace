# `task_games__bingo__called_number_match_count`

## Contract
1. Domain: `games`
2. Scene id: `bingo`
3. Public task id: `task_games__bingo__called_number_match_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(intersection(called_numbers, card_numbers)); scene=bingo; scope=called_number_match_count`

## Program Contract

Program: `count(intersection(called_numbers, card_numbers)); scene=bingo; scope=called_number_match_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `called_number_match_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `intersection`, `called_numbers`, `card_numbers`, `bingo`, `called_number_match_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation marks the card-cell boxes whose printed numbers are present in the CALLED list.
