# `task_puzzles__tents__missing_tent_cell_label`

## Program Contract

Program: `label(single(legal_tent_cells(marked_tree, board_state, row_clues, col_clues))); scene=tents; scope=missing_tent_cell_label`

Candidate set: the visible Tents grid, tree cells, tent cells, row/column clues, labels, and candidate or violating cell markers inside the `missing_tent_cell_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `legal_tent_cells`, `marked_tree`, `board_state`, `row_clues`, `col_clues`, `tents`, `missing_tent_cell_label`.
Operation: evaluate `label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Contract
1. Domain: `puzzles`
2. Scene id: `tents`
3. Public task id: `task_puzzles__tents__missing_tent_cell_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`

## Annotation
`annotation` is one image-pixel bounding box `[x0,y0,x1,y1]` for the selected labeled candidate cell.

## Generation Notes
The generated board has exactly one legal labeled candidate for the marked tree. The four labels are the four orthogonally adjacent cells around the marked tree; no far candidate cells are used. Candidate label assignment, grid size, scene variant, palette, and background style are generation/render metadata, not public query ids.
