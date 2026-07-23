# `task_puzzles__star_battle__remaining_valid_cell_count`

## Program Contract

Program: `count(star_battle.valid_cells, scope=marked_row|marked_column); scene=star_battle; scope=marked_scope_cells`

Candidate set: the visible Star Battle grid, regions, clues, placed stars, and highlighted row or column inside the `marked_scope_cells` objective scope.
Operands: visible scene state and prompt-bound operands named by `star_battle`, `valid_cells`, `marked_row`, `marked_column`, `marked_scope_cells`.
Operation: evaluate `count` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the bbox set of counted legal cells only.
Query ids: `remaining_valid_cells_in_marked_row_count`, `remaining_valid_cells_in_marked_column_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `matching`

## Behavior

The task renders a partial Star Battle board with visible fixed stars and one highlighted row or column. The prompt names the highlighted row index from the top or column index from the left. The model counts empty cells in that highlighted scope where another star could legally be placed under the Star Battle rules. The annotation is the bbox set of counted legal cells only.
