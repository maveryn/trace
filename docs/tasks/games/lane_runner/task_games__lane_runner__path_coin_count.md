# `task_games__lane_runner__path_coin_count`

## Contract
1. Domain: `games`
2. Scene id: `lane_runner`
3. Public task id: `task_games__lane_runner__path_coin_count`
4. Supported `query_id` values: `single`
5. Annotation schema: `point_set`

## Program Contract

Program: `count(intersection(coins, shown_path_cells)); scene=lane_runner; scope=path_coin_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `path_coin_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `intersection`, `coins`, `shown_path_cells`, `lane_runner`, `path_coin_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `spatial_relations`, `topology`

## Answer And Annotation
1. `answer_gt.type`: `integer`.
2. `annotation_gt.type`: `point_set`.
3. Annotation points mark the centers of every coin collected by the shown path.
4. Off-path coins, including same-row parallel distractors, are not annotation.
