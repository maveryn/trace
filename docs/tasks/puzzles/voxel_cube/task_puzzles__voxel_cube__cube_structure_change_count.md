# `task_puzzles__voxel_cube__cube_structure_change_count`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `voxel_cube`
3. Source scene: `voxel_cube`
4. Task id: `task_puzzles__voxel_cube__cube_structure_change_count`

## Query Contract
1. Supported `query_id`: `single`
2. Prompt asks for the number of cubes added or removed between a reference and changed structure.
3. Internal variation: `change_type` is recorded as trace metadata and rendered through the selected prompt query key.

## Program Contract

Program: `count(cube_difference(reference_stack, changed_stack)); scene=voxel_cube; scope=cube_structure_change_count`

Candidate set: the visible voxel stack, unit cubes, projections, changed/reference stacks, and labeled candidate views inside the `cube_structure_change_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `cube_difference`, `reference_stack`, `changed_stack`, `voxel_cube`, `cube_structure_change_count`.
Operation: evaluate `count` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the reference structure and the changed structure.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `matching`

## Answer And Annotation
1. `answer_gt.type = integer`
2. `annotation_gt.type = bbox_set`
3. Annotation schema: unordered `bbox_set`
4. Annotation target: the reference structure and the changed structure.
5. `scalar_annotation_checked = true`.

## Prompt Contract
1. Bundle: `puzzles_voxel_cube_v1`
2. Scene key: `voxel_cube`
3. Task key: `cube_structure_change_count_query`
4. Query keys: `missing_to_complete_cuboid_count`, `removed_cube_count`
