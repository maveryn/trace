# `task_puzzles__cell_board__largest_component_size`

## Program Contract

Program: `max_component_size(cell_board, cells=color_equals_query_color, adjacency=orthogonal_4_neighbor); scene=cell_board; scope=largest_component_size`

Candidate set: the visible grid cells, cell colors/states, labels, walls, start/goal markers, and mirror or connectivity cues inside the `largest_component_size` objective scope.
Operands: visible scene state and prompt-bound operands named by `cell_board`, `cells`, `color_equals_query_color`, `adjacency`, `orthogonal_4_neighbor`, `largest_component_size`.
Operation: evaluate `max_component_size` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; image-pixel cell bboxes for all cells in the unique largest component; bbox-set cardinality equals the answer.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## 2) Scene + task contract
1. Entities/relations: A rectangular colored cell board with a single largest target-color component containing one or more cells.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Annotation schema: `bbox_set`
5. Alternate annotation forms: none
6. Annotation witness policy: image-pixel cell bboxes for all cells in the unique largest component; bbox-set cardinality equals the answer.
7. Overlap/touch policy: diagonal contact does not merge components.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_cell_board_v1`
2. `scene_key`: `cell_board`
3. `task_key`: `cell_board_topology_query`
4. Prompt query key: `largest_component_size`
5. Required slots: `query_color`, plus output-mode slots from the prompt bundle.
6. JSON example validity rule: bbox-set cardinality equals the integer answer.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local largest-component/color namespaces plus shared cell-board layout/style/font/noise namespaces.
2. Unique-answer policy: construction verifies a single target-color component has the configured largest size.
3. Reject/resample conditions: non-unique largest component or size mismatch raises and retries.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed.
