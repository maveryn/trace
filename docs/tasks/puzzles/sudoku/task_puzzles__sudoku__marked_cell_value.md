# `task_puzzles__sudoku__marked_cell_value`

## Program Contract

Program: `value(single(candidate_digits(marked_cell, board_state))); scene=sudoku; scope=marked_cell_value`

Candidate set: the visible Sudoku grid, givens, filled values, marked cell, candidate values, and option labels when present inside the `marked_cell_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `candidate_digits`, `marked_cell`, `board_state`, `sudoku`, `marked_cell_value`.
Operation: evaluate `value` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`, `matching`

## Contract
1. Domain: `puzzles`
2. Scene id: `sudoku`
3. Public task id: `task_puzzles__sudoku__marked_cell_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox`

## Annotation
`annotation` is one image-pixel bbox for the red outlined marked empty cell.

## Generation Notes
The answer and annotation are bound from the same generated Sudoku board. Scene density, style, font, layout jitter, and target digit are generation/render metadata, not public query ids.
