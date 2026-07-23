# `task_puzzles__toggle_grid__toggle_result_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `toggle_grid`
3. Source scene: `toggle_grid`
4. Task id: `task_puzzles__toggle_grid__toggle_result_label`

## Query Contract
1. Supported `query_id`: `single`
2. Prompt asks for the labeled result grid produced by pressing the red marked switch once.
3. Internal variation: grid size, selected press cell, option label, scene variant, and style are generation/render metadata.

## Program Contract

Program: `select_label(result_grid_options, option_grid = simulate(start_grid, rule=orthogonal_toggle, action=red_marked_switch)); scene=toggle_grid; scope=toggle_result_label`

Candidate set: the visible start/target/result grids, toggle rule markers, switch cells, labels, and labeled grid options inside the `toggle_result_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `result_grid_options`, `option_grid`, `simulate`, `start_grid`, `orthogonal_toggle`, `action`, `red_marked_switch`, `toggle_grid`, `toggle_result_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the correct result option panel.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `annotation_gt.type = bbox`
3. Annotation schema: scalar `bbox`
4. Annotation target: the correct result option panel.
5. `scalar_annotation_checked = true`.

## Prompt Contract
1. Bundle: `puzzles_toggle_grid_v1`
2. Scene key: `toggle_grid`
3. Task key: `toggle_result_label_query`
4. Query key: `toggle_result_label`
