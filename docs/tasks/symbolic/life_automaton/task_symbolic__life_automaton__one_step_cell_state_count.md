# `task_symbolic__life_automaton__one_step_cell_state_count`

## Summary
1. Domain: `symbolic`
2. Scene: `life_automaton`
3. Task id: `task_symbolic__life_automaton__one_step_cell_state_count`
4. Goal: apply one cellular-life update to the grid marked `START` and count cells in a requested future state.

## Program Contract
Program: `life_automaton.one_step_cell_state_count(scene=life_automaton, scope=marked_start_grid, grid_size=3x3, target_state=alive|dead, update_steps=one, output=integer)`

Candidate set: all cells in the visible `START` grid.
Operands: the current alive/dead state of every cell and the requested future state.
Operation: apply one life update, then count cells whose future state equals the requested state.
Output binding: `answer` is the matching cell count as an integer.
Annotation witnesses: a homogeneous `bbox_set` of source-grid cell positions that have the requested state after the update; the set is empty when the answer is `0`.
Query ids: `one_step_alive_cell_count`, `one_step_dead_cell_count`.

## Reasoning Operations

Families: `counting`, `state_update`

## Contract
1. Query ids: `one_step_alive_cell_count|one_step_dead_cell_count`
2. Answer type: `integer`
3. Annotation schema: `bbox_set`
4. Annotation target: unordered bboxes for the visible source-grid cell positions that have the requested state after one update
5. Scene variants: `clean_grid|lab_panel|notebook_grid`
6. Prompt bundle: `symbolic_life_automaton_v1`
7. Source file: `src/trace_tasks/tasks/symbolic/life_automaton/one_step_cell_state_count.py`
8. Render metadata: records panel style, role-aware font family, unit-size jitter, and annotation-safe layout jitter before annotation projection.
9. Visual contract: source grid is visibly marked `START`; the future grid is not shown.
10. Scene-local variation: records `life_board.board_style`, `life_board.cell_palette_id`, resolved RGBs, and alive/dead/marker contrast checks.

## Annotation Contract
The annotation is a `bbox_set` because the task counts an unordered set of same-role cell witnesses.
If the answer is `0`, the annotation is an empty bbox set.
