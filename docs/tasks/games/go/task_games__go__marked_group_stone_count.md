# `task_games__go__marked_group_stone_count`

## Contract
1. Domain: `games`
2. Scene: `go`
2. Scene id: `go`
3. Public task id: `task_games__go__marked_group_stone_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(connected_group(marked_stone)); scene=go; scope=marked_group_stone_count`

## Program Contract

Program: `count(connected_group(marked_stone)); scene=go; scope=marked_group_stone_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_group_stone_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `connected_group`, `marked_stone`, `go`, `marked_group_stone_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Generation Notes
1. The scene draws one red outline around a single black or white reference stone.
2. The answer is the number of same-color stones connected edge-to-edge to that marked stone, including the marked stone.
3. Annotation is the stone box for every stone in the marked stone's connected group.
4. Group-size answers are sampled from `2..6`.
