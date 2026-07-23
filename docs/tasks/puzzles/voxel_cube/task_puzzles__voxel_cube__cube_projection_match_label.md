# `task_puzzles__voxel_cube__cube_projection_match_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `voxel_cube`
3. Source scene: `voxel_cube`
4. Task id: `task_puzzles__voxel_cube__cube_projection_match_label`

## Query Contract
1. Supported `query_id`: `single`
2. Prompt asks which labeled projection option matches the selected view of the voxel structure, using the rendered front/right direction cue.
3. Internal variation: selected `view_direction`, option count, and correct option label are trace metadata.

## Program Contract

Program: `select_label(projection_options, option = orthographic_projection(stack, view_direction)); scene=voxel_cube; scope=cube_projection_match_label`

Candidate set: the visible voxel stack, unit cubes, projections, changed/reference stacks, and labeled candidate views inside the `cube_projection_match_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `projection_options`, `orthographic_projection`, `stack`, `view_direction`, `voxel_cube`, `cube_projection_match_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the correct projection option panel.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `annotation_gt.type = bbox`
3. Annotation schema: scalar `bbox`
4. Annotation target: the correct projection option panel.
5. `scalar_annotation_checked = true`.

## Prompt Contract
1. Bundle: `puzzles_voxel_cube_v1`
2. Scene key: `voxel_cube`
3. Task key: `cube_projection_match_label_query`
4. Query key: `projection_match_label`
