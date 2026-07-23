# `task_games__sliding_block__sliding_block_blocker_count`

## Contract
1. Domain: `games`
2. Scene id: `sliding_block`
3. Public task id: `task_games__sliding_block__sliding_block_blocker_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(blocks_intersecting(target_exit_path)); scene=sliding_block; scope=sliding_block_blocker_count`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count(blocks_intersecting(target_exit_path)); scene=sliding_block; scope=sliding_block_blocker_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `sliding_block_blocker_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `blocks_intersecting`, `target_exit_path`, `sliding_block`, `sliding_block_blocker_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`

## Generation Notes
1. The red target block labeled `T` and exit arrow define the straight exit path.
2. The answer is the number of non-target blocks occupying cells on that path; supported counts include `0`.
3. Annotation is the bbox set of the blocking blocks, or an empty set when the answer is `0`.
