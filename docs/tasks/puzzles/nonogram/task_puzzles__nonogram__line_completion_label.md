# `task_puzzles__nonogram__line_completion_label`

## Summary
1. Domain: `puzzles`
2. Scene id: `nonogram`
3. Objective: choose the labeled row-strip option that completes the marked nonogram row.
4. Answer type: `option_letter`
5. Annotation schema: `bbox`

## Program Contract

Program: `select_label(nonogram.row_strip, rule=row_clue_matches_and_visible_cells_match); scene=nonogram; scope=marked_row_strip_satisfying_row_clue_and_visible_cells`

Candidate set: the visible nonogram clues, row or grid cells, filled/empty states, and labeled candidate strips or grids inside the `marked_row_strip_satisfying_row_clue_and_visible_cells` objective scope.
Operands: visible scene state and prompt-bound operands named by `nonogram`, `row_strip`, `row_clue_matches_and_visible_cells_match`, `marked_row_strip_satisfying_row_clue_and_visible_cells`.
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
The annotation marks only the chosen visual option panel. The clue rail and marked row are question context, not answer witnesses.
