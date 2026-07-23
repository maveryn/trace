# `task_puzzles__cell_board__reachable_region_size`

## Program Contract

Program: `count_reachable_cells(cell_board, start=green_S_cell, passable=non_wall_cells, adjacency=orthogonal_4_neighbor); scene=cell_board; scope=reachable_region_size`

Candidate set: the visible grid cells, cell colors/states, labels, walls, start/goal markers, and mirror or connectivity cues inside the `reachable_region_size` objective scope.
Operands: visible scene state and prompt-bound operands named by `cell_board`, `start`, `green_S_cell`, `passable`, `non_wall_cells`, `adjacency`, `orthogonal_4_neighbor`, `reachable_region_size`.
Operation: evaluate `count_reachable_cells` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; one image-pixel cell bbox for every reachable light passable cell, including the start cell.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## 2) Scene + task contract
1. Entities/relations: A rectangular board with dark wall cells, light passable cells, disconnected light distractor regions, and a green start cell marked `S`.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Annotation schema: `bbox_set`
5. Alternate annotation forms: none
6. Annotation witness policy: one image-pixel cell bbox for every reachable light passable cell, including the start cell.
7. Overlap/touch policy: dark walls block movement; diagonal/corner touching between light cells does not connect regions.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_cell_board_v1`
2. `scene_key`: `cell_board`
3. `task_key`: `cell_board_topology_query`
4. Prompt query key: `reachable_region_size`
5. Required slots: output-mode slots from the prompt bundle.
6. JSON example validity rule: bbox-set cardinality equals the integer answer.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local region namespaces plus shared cell-board layout/style/font/noise namespaces.
2. Unique-answer policy: construction creates and verifies an exact reachable region size in answer range `1..8`.
3. Reject/resample conditions: reachable-region size mismatch or insufficient disconnected passable distractors raises and retries.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed.
