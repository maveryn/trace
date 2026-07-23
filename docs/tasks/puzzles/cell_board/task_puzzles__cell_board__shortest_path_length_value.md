# `task_puzzles__cell_board__shortest_path_length_value`

## Program Contract

Program: `shortest_path_length(cell_board, start=green_S_cell, goal=red_G_cell, passable=non_wall_cells, adjacency=orthogonal_4_neighbor); scene=cell_board; scope=shortest_path_length_value`

Candidate set: the visible grid cells, cell colors/states, labels, walls, start/goal markers, and mirror or connectivity cues inside the `shortest_path_length_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `cell_board`, `start`, `green_S_cell`, `goal`, `red_G_cell`, `passable`, `non_wall_cells`, `adjacency`, `orthogonal_4_neighbor`, `shortest_path_length_value`.
Operation: evaluate `shortest_path_length` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; one image-pixel segment for each adjacent step along a shortest path from `S` to `G`; segment count equals the answer.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## 2) Scene + task contract
1. Entities/relations: A rectangular board with dark wall cells, light passable cells, a green start cell marked `S`, a red goal cell marked `G`, and disconnected passable distractors.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Annotation schema: `segment_set`
5. Alternate annotation forms: none
6. Annotation witness policy: one image-pixel segment for each adjacent step along a shortest path from `S` to `G`; segment count equals the answer.
7. Overlap/touch policy: walls block movement and diagonal touching is not a step.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_cell_board_v1`
2. `scene_key`: `cell_board`
3. `task_key`: `cell_board_topology_query`
4. Prompt query key: `shortest_path_length_value`
5. Required slots: output-mode slots from the prompt bundle.
6. JSON example validity rule: segment-set cardinality equals the integer answer.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local shortest-path namespaces plus shared cell-board layout/style/font/noise namespaces.
2. Unique-answer policy: construction creates a verified detour corridor whose shortest S-to-G path is 4..8 steps and longer than direct Manhattan distance.
3. Reject/resample conditions: path construction, distractor placement, or path-length mismatch raises and retries.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed.
