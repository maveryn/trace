# `task_games__chess__king_escape_square_count`

## Contract
1. Domain: `games`
2. Scene id: `chess`
3. Public task id: `task_games__chess__king_escape_square_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(legal_escape_squares(king)); scene=chess; scope=king_escape_square_count`

## Program Contract

Program: `count(legal_escape_squares(king)); scene=chess; scope=king_escape_square_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `king_escape_square_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `legal_escape_squares`, `king`, `chess`, `king_escape_square_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from the same generated game state used for answer verification.
