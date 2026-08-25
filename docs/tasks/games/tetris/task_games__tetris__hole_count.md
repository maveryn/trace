# `task_games__tetris__hole_count`

## Contract
1. Domain: `games`
2. Scene: `tetris`
3. Scene id: `tetris`
4. Public task id: `task_games__tetris__hole_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `count(cells where cell_is_empty(cell) and filled_block_above(cell)); scene=tetris; scope=hole_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `hole_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `cells`, `where`, `cell_is_empty`, `filled_block_above`, `cell`, `tetris`, `hole_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Generation Notes
1. The prompt states the hole rule: a hole is an empty cell with at least one filled block above it in the same column.
2. Generation presamples the exact hole count from a balanced support, then carves interior cells out of a supported stack so the board matches that count.
3. Annotation marks one cell bbox for each covered hole on the rendered board.
