# `task_games__match3__swap_clear_count`

## Contract
1. Domain: `games`
2. Scene id: `match3`
3. Public task id: `task_games__match3__swap_clear_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract

Program: `count(immediate_clear_cells_after(marked_swap)); scene=match3; scope=swap_clear_count`

Candidate set: the visible match-3 board, colored gems, and the marked adjacent-swap arrow inside the `swap_clear_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `marked_swap`, `immediate_clear_cells_after`, `match3`, and `swap_clear_count`.
Operation: swap the two adjacent gems connected by the marked arrow, find every straight horizontal or vertical run of three or more equal gems immediately after the swap, and count the unique gems in those runs. Same-color gems that merely touch a cleared run are not counted unless they are also in a straight run. Generation enforces the answer support `0, 3, 4, 5, 6`.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; boxes cover exactly the gems that clear immediately after the marked swap. If no gems clear, the box set is empty.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes
1. The board is fixed at 5 by 5 with five gem colors for this task.
2. The board has no pre-existing horizontal or vertical run before the marked swap.
3. The immediate clear rule counts only the post-swap clear; no falling, refill, special effects, or cascades are applied.
4. The marked swap avoids same-color non-clearing branches touching cleared cells, so the visual connected component does not exceed the straight-run clear set.
5. Answer support is sampled from `0, 3, 4, 5, 6`.
