# `task_puzzles__star_battle__valid_cell_anywhere_label`

## Program Contract

Program: `select_option(star_battle.valid_cell, scope=whole_board); scene=star_battle; scope=visible_board_candidates`

Candidate set: the visible Star Battle grid, regions, clues, placed stars, scope markers, and labeled candidate cells inside the `visible_board_candidates` objective scope.
Operands: visible scene state and prompt-bound operands named by `star_battle`, `valid_cell`, `whole_board`, `visible_board_candidates`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the label of that legal candidate. The annotation is the selected candidate cell bbox.
Annotation witnesses: `annotation` uses the `bbox` schema; the selected candidate cell bbox.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Behavior

The task renders a partial Star Battle board with visible fixed stars and labeled candidate cells. Exactly one candidate cell is legal under the rules: each row, column, and colored region has exactly one star, and stars may not touch by edge or corner. The answer is the label of that legal candidate. The annotation is the selected candidate cell bbox.
