# `task_games__tower_draughts_board__marked_stack_capture_count`

## Contract
1. Domain: `games`
2. Scene: `tower_draughts_board`
3. Scene id: `tower_draughts_board`
4. Public task id: `task_games__tower_draughts_board__marked_stack_capture_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `count(adjacent_opponent_stack for diagonal_jump in legal_immediate_captures(x_marked_stack)); scene=tower_draughts_board; scope=marked_stack_capture_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_stack_capture_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `adjacent_opponent_stack`, `diagonal_jump`, `legal_immediate_captures`, `x_marked_stack`, `tower_draughts_board`, `marked_stack_capture_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`, `state_update`

## Generation Notes
1. The X-marked stack is controlled by the color of its top disk.
2. A capture jumps diagonally over one adjacent opponent-controlled stack and lands on the empty playable square immediately beyond it.
3. Regular top disks capture forward only; crowned top disks capture in either diagonal direction.
4. The answer range is `0..4`.
5. Annotation marks the bounding box of every opponent-controlled stack that can be captured; an empty bbox set is valid when the answer is `0`.
