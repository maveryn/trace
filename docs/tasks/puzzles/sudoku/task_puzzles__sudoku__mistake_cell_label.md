# `task_puzzles__sudoku__mistake_cell_label`

## Program Contract

Program: `select_label(option_cell, rule=filled_cell_value_violates_sudoku_constraints); scene=sudoku; scope=mistake_cell_label`

Candidate set: the visible Sudoku grid, givens, filled values, marked cell, candidate values, and option labels when present inside the `mistake_cell_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_cell`, `filled_cell_value_violates_sudoku_constraints`, `sudoku`, `mistake_cell_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Contract
1. Domain: `puzzles`
2. Scene id: `sudoku`
3. Public task id: `task_puzzles__sudoku__mistake_cell_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`

## Annotation
`annotation` is one image-pixel bbox for the selected lettered filled cell.

## Generation Notes
One lettered filled cell is injected with a wrong digit that conflicts with a visible unlettered peer. The other lettered cells remain valid solution givens.
