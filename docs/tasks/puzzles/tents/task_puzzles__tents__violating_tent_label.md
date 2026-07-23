# `task_puzzles__tents__violating_tent_label`

## Program Contract

Program: `label(single(violating_tent(labeled_tents, tree_cells, adjacency_rule))); scene=tents; scope=violating_tent_label`

Candidate set: the visible Tents grid, tree cells, tent cells, row/column clues, labels, and candidate or violating cell markers inside the `violating_tent_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `violating_tent`, `labeled_tents`, `tree_cells`, `adjacency_rule`, `tents`, `violating_tent_label`.
Operation: evaluate `label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Contract
1. Domain: `puzzles`
2. Scene id: `tents`
3. Public task id: `task_puzzles__tents__violating_tent_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`

## Annotation
`annotation` is one image-pixel bounding box `[x0,y0,x1,y1]` for the selected labeled tent cell.

## Generation Notes
The generated board has a 6x6 to 8x8 Tents grid with four labeled visible tents. Exactly one labeled tent is not orthogonally adjacent to any tree; the other labeled tents each have an orthogonally adjacent tree and the labeled tents do not touch one another.
