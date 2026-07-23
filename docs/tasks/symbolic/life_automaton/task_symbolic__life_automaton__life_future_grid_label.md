# `task_symbolic__life_automaton__life_future_grid_label`

## Summary
1. Domain: `symbolic`
2. Scene: `life_automaton`
3. Task id: `task_symbolic__life_automaton__life_future_grid_label`
4. Goal: apply a cellular-life neighbor rule to the grid marked `START` and choose the option showing the future grid.

## Program Contract
Program: `life_automaton.future_grid_option_label(scene=life_automaton, scope=marked_start_grid_plus_option_set, grid_size=3x3|4x4|5x5, update_steps=one|two, output=option_letter)`

Candidate set: the visible future-grid option panels.
Operands: the `START` grid state, grid size, life update rule, and requested one-step or two-step horizon.
Operation: apply the life rule for the requested number of updates and select the unique option grid matching the result.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `source_grid` and `selected_option` roles.
Query ids: `one_step_future_grid`, `two_step_future_grid`.

## Reasoning Operations

Families: `state_update`

## Contract
1. Query ids: `one_step_future_grid|two_step_future_grid`
2. Answer type: `option_letter`
3. Annotation schema: `bbox_map`
4. Annotation target: role-keyed bboxes for `source_grid` and `selected_option`
5. Scene variants: `clean_grid|lab_panel|notebook_grid`
6. Prompt bundle: `symbolic_life_automaton_v1`
7. Source file: `src/trace_tasks/tasks/symbolic/life_automaton/life_future_grid_label.py`
8. Render metadata: records panel style, role-aware font family, unit-size jitter, and annotation-safe layout jitter before annotation projection.
9. Visual contract: source grid is visibly marked `START`; source grid and option grids use the same rendered cell scale.
10. Scene-local variation: records `life_board.board_style`, `life_board.cell_palette_id`, resolved RGBs, and alive/dead/marker contrast checks.

## Annotation Contract
The annotation is a `bbox_map` because the task has two distinct witness roles:
the source grid that must be updated and the selected visual option panel.
