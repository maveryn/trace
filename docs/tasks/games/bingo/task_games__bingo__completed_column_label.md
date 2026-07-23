# `task_games__bingo__completed_column_label`

## Contract
1. Domain: `games`
2. Scene id: `bingo`
3. Public task id: `task_games__bingo__completed_column_label`
4. Supported `query_id` values: `single`
5. Answer schema: `string_label`
6. Annotation schema: `segment`
7. Program schema: `label(unique_completed_column(board)); scene=bingo; scope=completed_column_label`

## Program Contract

Program: `label(unique_completed_column(board)); scene=bingo; scope=completed_column_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `completed_column_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique_completed_column`, `board`, `bingo`, `completed_column_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`

## Generation Notes
1. The card has exactly one completed BINGO column.
2. The answer is one of `B`, `I`, `N`, `G`, or `O`.
3. Annotation is one `segment` `[[x0, y0], [x1, y1]]` connecting the top and bottom cell centers of the completed column.
