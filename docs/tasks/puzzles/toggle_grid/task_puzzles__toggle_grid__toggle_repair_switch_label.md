# `task_puzzles__toggle_grid__toggle_repair_switch_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `toggle_grid`
3. Source scene: `toggle_grid`
4. Task id: `task_puzzles__toggle_grid__toggle_repair_switch_label`

## Query Contract
1. Supported `query_id`: `single`
2. Prompt asks for the one lettered start-grid switch that transforms the start grid into the target grid.
3. Internal variation: grid size, selected switch, option label, scene variant, and style are generation/render metadata.

## Program Contract

Program: `select_label(candidate_switch_cells, cell = inverse_one_step_toggle(start_grid, target_grid, rule=orthogonal_toggle)); scene=toggle_grid; scope=toggle_repair_switch_label`

Candidate set: the visible start/target/result grids, toggle rule markers, switch cells, labels, and labeled grid options inside the `toggle_repair_switch_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `candidate_switch_cells`, `cell`, `inverse_one_step_toggle`, `start_grid`, `target_grid`, `orthogonal_toggle`, `toggle_grid`, `toggle_repair_switch_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the start-grid cell containing the correct switch label.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `annotation_gt.type = bbox`
3. Annotation schema: scalar `bbox`
4. Annotation target: the start-grid cell containing the correct switch label.
5. `scalar_annotation_checked = true`.

## Prompt Contract
1. Bundle: `puzzles_toggle_grid_v1`
2. Scene key: `toggle_grid`
3. Task key: `toggle_repair_switch_label_query`
4. Query key: `toggle_repair_switch_label`
