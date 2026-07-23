# `task_games__sliding_block__movable_block_count`

## Contract
1. Domain: `games`
2. Scene id: `sliding_block`
3. Public task id: `task_games__sliding_block__movable_block_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(blocks_with_legal_one_cell_slide_along_orientation(board_state)); scene=sliding_block; scope=movable_block_count`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(blocks_with_legal_one_cell_slide_along_orientation(board_state)); scene=sliding_block; scope=movable_block_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `movable_block_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `blocks_with_legal_one_cell_slide_along_orientation`, `board_state`, `sliding_block`, `movable_block_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
1. The board has no target block for this task.
2. Horizontal blocks slide horizontally and vertical blocks slide vertically.
3. A block is counted if at least one one-cell slide stays inside the board and does not overlap another block.
4. Annotation is the bbox set of all counted blocks.
