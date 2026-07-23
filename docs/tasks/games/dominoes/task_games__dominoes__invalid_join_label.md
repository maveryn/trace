# `task_games__dominoes__invalid_join_label`

## Contract
1. Domain: `games`
2. Scene: `dominoes`
3. Scene id: `dominoes`
4. Public task id: `task_games__dominoes__invalid_join_label`
5. Supported `query_id` values: `single`
6. Answer schema: `option_label`
7. Annotation schema: `segment`
8. Program schema: `label(select(join, touching_pip_left(join) != touching_pip_right(join))); scene=dominoes; scope=invalid_join_label`

## Program Contract

Program: `label(select(join, touching_pip_left(join) != touching_pip_right(join))); scene=dominoes; scope=invalid_join_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `invalid_join_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select`, `join`, `touching_pip_left`, `touching_pip_right`, `dominoes`, `invalid_join_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Generation Notes
1. Renders one face-up domino chain with seven dominoes and exactly six labeled joins `A` through `F`.
2. Exactly one adjacent join has mismatched touching halves.
3. The answer is the label of that invalid join.
4. Annotation is one `segment` `[[x0, y0], [x1, y1]]` connecting the centers of the two touching domino halves at the invalid join.
