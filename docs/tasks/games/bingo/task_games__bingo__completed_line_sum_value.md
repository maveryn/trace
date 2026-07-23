# `task_games__bingo__completed_line_sum_value`

## Contract
1. Domain: `games`
2. Scene id: `bingo`
3. Public task id: `task_games__bingo__completed_line_sum_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `point_set`
7. Program schema: `sum(values(cells(completed_line))); scene=bingo; scope=completed_line_sum_value`

## Program Contract

Program: `sum(values(cells(completed_line))); scene=bingo; scope=completed_line_sum_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `completed_line_sum_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `values`, `cells`, `completed_line`, `bingo`, `completed_line_sum_value`.
Operation: evaluate `sum` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation marks the center point of each cell in the completed row or column used for the sum.
