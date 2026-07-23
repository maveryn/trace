# `task_games__go__group_liberty_count`

## Contract
1. Domain: `games`
2. Scene: `go`
2. Scene id: `go`
3. Public task id: `task_games__go__group_liberty_count`
4. Supported `query_id` values: `marked_group_liberty_count`, `marked_group_shared_liberty_count`
5. Answer schema: `integer_count`
6. Annotation schema: `point_set`
7. Program schema: `count(filter(liberties(marked_group), liberty_filter)); scene=go; scope=group_liberty_count; query_branch=marked_group_liberty_count|marked_group_shared_liberty_count`

## Program Contract

Program: `count(filter(liberties(marked_group), liberty_filter)); scene=go; scope=group_liberty_count; query_branch=marked_group_liberty_count|marked_group_shared_liberty_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `group_liberty_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `liberties`, `marked_group`, `liberty_filter`, `go`, `group_liberty_count`, `query_branch`, `marked_group_liberty_count`, `marked_group_shared_liberty_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `marked_group_liberty_count`, `marked_group_shared_liberty_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Generation Notes
1. `marked_group_liberty_count` counts all empty orthogonal liberties of the marked group.
2. `marked_group_shared_liberty_count` counts marked-group liberties that also touch an opponent stone.
3. Annotation is projected from the same generated game state used for answer verification.
