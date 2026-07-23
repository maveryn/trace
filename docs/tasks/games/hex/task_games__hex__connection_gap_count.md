# `task_games__hex__connection_gap_count`

## Contract
1. Domain: `games`
2. Scene: `hex`
3. Scene id: `hex`
4. Public task id: `task_games__hex__connection_gap_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `point_set`
8. Program schema: `count(unique_minimum_empty_cells_to_connect_sides(player, board_state)); scene=hex; scope=connection_gap_count`

## Program Contract

Program: `count(unique_minimum_empty_cells_to_connect_sides(player, board_state)); scene=hex; scope=connection_gap_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `connection_gap_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique_minimum_empty_cells_to_connect_sides`, `player`, `board_state`, `hex`, `connection_gap_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## Generation Notes
1. `query_id=single` is the public no-branch query id; the prompt uses the Hex connection-gap template.
2. The generator rejects boards with multiple distinct minimum gap sets so annotation has a unique witness set.
3. Annotation is the point set for the empty cells in that unique minimum connection gap.
