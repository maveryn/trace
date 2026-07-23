# `task_games__hex__candidate_neighbor_count`

## Contract
1. Domain: `games`
2. Scene: `hex`
3. Scene id: `hex`
4. Public task id: `task_games__hex__candidate_neighbor_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`
8. Program schema: `count(filter(adjacent_cells(reference_cell), state=neighbor_target_state)); scene=hex; scope=candidate_neighbor_count`

## Program Contract

Program: `count(filter(adjacent_cells(reference_cell), state=neighbor_target_state)); scene=hex; scope=candidate_neighbor_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `candidate_neighbor_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `adjacent_cells`, `reference_cell`, `state`, `neighbor_target_state`, `hex`, `candidate_neighbor_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Generation Notes
1. The reference cell is filled green in the rendered board and is not counted.
2. The target neighbor state is sampled with `neighbor_target_state` (`red`, `blue`, or `empty`).
3. The generator samples an interior reference cell so each instance has exactly six adjacent cells.
4. Annotation is the set of pixel-space centers for neighboring cells that match the requested state; an empty annotation list is valid when the answer is `0`.
