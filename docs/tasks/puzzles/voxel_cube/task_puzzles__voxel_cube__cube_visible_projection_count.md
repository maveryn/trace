# `task_puzzles__voxel_cube__cube_visible_projection_count`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `voxel_cube`
3. Source scene: `voxel_cube`
4. Task id: `task_puzzles__voxel_cube__cube_visible_projection_count`

## Query Contract
1. Supported `query_id`: `single`
2. Prompt asks for how many cells are filled in a selected top/front/right orthographic projection, using the rendered front/right direction cue.
3. Internal variation: selected `view_direction` is trace metadata and a prompt slot.

## Program Contract

Program: `count(filled_cells(orthographic_projection(stack, view_direction))); scene=voxel_cube; scope=cube_visible_projection_count`

Candidate set: the visible voxel stack, unit cubes, projections, changed/reference stacks, and labeled candidate views inside the `cube_visible_projection_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filled_cells`, `orthographic_projection`, `stack`, `view_direction`, `voxel_cube`, `cube_visible_projection_count`.
Operation: evaluate `count` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; all filled cells in the target projection grid.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `transformation`

## Answer And Annotation
1. `answer_gt.type = integer`
2. `annotation_gt.type = bbox_set`
3. Annotation schema: unordered `bbox_set`
4. Annotation target: all filled cells in the target projection grid.
5. `scalar_annotation_checked = true`.

## Prompt Contract
1. Bundle: `puzzles_voxel_cube_v1`
2. Scene key: `voxel_cube`
3. Task key: `cube_visible_projection_count_query`
4. Query key: `visible_projection_count`
