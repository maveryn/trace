# `task_puzzles__voxel_cube__cube_count`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `voxel_cube`
3. Source scene: `voxel_cube`
4. Task id: `task_puzzles__voxel_cube__cube_count`

## Query Contract
1. Supported `query_id`: `single`
2. Prompt asks for the total number of unit cubes in one visible isometric voxel structure.
3. Internal variation: stack dimensions, cube count, scene treatment, palette, and font/render style are generation/render metadata.

## Program Contract

Program: `count(unit_cubes(stack)); scene=voxel_cube; scope=cube_count`

Candidate set: the visible voxel stack, unit cubes, projections, changed/reference stacks, and labeled candidate views inside the `cube_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `unit_cubes`, `stack`, `voxel_cube`, `cube_count`.
Operation: evaluate `count` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the full rendered voxel structure.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Answer And Annotation
1. `answer_gt.type = integer`
2. `annotation_gt.type = bbox`
3. Annotation schema: scalar `bbox`
4. Annotation target: the full rendered voxel structure.
5. `scalar_annotation_checked = true`.

## Prompt Contract
1. Bundle: `puzzles_voxel_cube_v1`
2. Scene key: `voxel_cube`
3. Task key: `cube_count_query`
4. Query key: `cube_count`
