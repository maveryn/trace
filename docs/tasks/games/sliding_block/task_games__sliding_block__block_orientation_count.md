# `task_games__sliding_block__block_orientation_count`

## Contract
1. Domain: `games`
2. Scene id: `sliding_block`
3. Public task id: `task_games__sliding_block__block_orientation_count`
4. Supported `query_id` values: `horizontal_block_count`, `vertical_block_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(blocks_matching_orientation(board_state, requested_orientation)); scene=sliding_block; scope=block_orientation_count`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(blocks_matching_orientation(board_state, requested_orientation)); scene=sliding_block; scope=block_orientation_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `block_orientation_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `blocks_matching_orientation`, `board_state`, `requested_orientation`, `sliding_block`, `block_orientation_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `horizontal_block_count`, `vertical_block_count`.

## Reasoning Operations

Families: `counting`

## Generation Notes
1. The board has no target block or exit arrow for this task.
2. `horizontal_block_count` counts blocks wider than they are tall.
3. `vertical_block_count` counts blocks taller than they are wide.
4. The generated answer support is `1..6`.
5. Annotation is the bbox set of all blocks matching the requested orientation.
