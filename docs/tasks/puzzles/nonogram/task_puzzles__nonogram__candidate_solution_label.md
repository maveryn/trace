# `task_puzzles__nonogram__candidate_solution_label`

## Summary
1. Domain: `puzzles`
2. Scene id: `nonogram`
3. Objective: choose the labeled filled-grid option that satisfies all visible row and column clues.
4. Answer type: `option_letter`
5. Annotation schema: `bbox`

## Program Contract

Program: `select_label(nonogram.option_grid, rule=all_row_and_column_clues_match); scene=nonogram; scope=filled_grid_satisfying_all_row_and_column_clues`

Candidate set: the visible nonogram clues, row or grid cells, filled/empty states, and labeled candidate strips or grids inside the `filled_grid_satisfying_all_row_and_column_clues` objective scope.
Operands: visible scene state and prompt-bound operands named by `nonogram`, `option_grid`, `all_row_and_column_clues_match`, `filled_grid_satisfying_all_row_and_column_clues`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Generation
1. Query id: `single`
2. Grid size: `3x3..5x5`
3. Option count: `{4}`
4. Scene variants: `nonogram_classic|nonogram_card|nonogram_blueprint`

## Contract Notes
The annotation marks only the chosen visual option panel. The clue rails are question context, not answer witnesses.
