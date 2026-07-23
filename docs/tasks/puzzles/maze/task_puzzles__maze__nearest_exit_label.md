# `task_puzzles__maze__nearest_exit_label`

## Contract

1. Domain: `puzzles`
2. Scene package: `src/trace_tasks/tasks/puzzles/maze/`
3. Scene id: `maze`
4. Public task id: `task_puzzles__maze__nearest_exit_label`
5. Supported `query_id` values: `single`
6. Prompt query key: `nearest_exit_label`
7. Answer schema: `string`
8. Annotation schema: `point`
9. Program schema: `select_label(argmin_exit maze.shortest_path_length_from_start); scene=maze; scope=nearest_reachable_exit`

## Program Contract

Program: `select_label(argmin_exit maze.shortest_path_length_from_start); scene=maze; scope=nearest_reachable_exit`

Candidate set: the visible maze cells, walls, start marker, exits, labels, and reachability/path structure inside the `nearest_reachable_exit` objective scope.
Operands: visible scene state and prompt-bound operands named by `argmin_exit`, `maze`, `shortest_path_length_from_start`, `nearest_reachable_exit`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; the selected nearest exit label.
Annotation witnesses: `annotation` uses the `point` schema; one image-pixel point centered on the selected nearest exit marker.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `topology`

## Query Contract

- Supported public `query_id`: `single`
- The task shows four reachable boundary exits and asks for the one with the shortest corridor path from `START`.
- Maze dimensions, exit placement, path lengths, scene treatment, font, marker shape, and theme are generation/render axes, not public taxonomy axes.

## Generation Contract

- The renderer shows one orthogonal wall maze with one `START` cell and exactly four labeled exits on the outer boundary.
- All four exits are reachable from `START`.
- Movement follows open corridors only; walls block motion.
- The nearest exit is unique by construction and has a configured path-length margin over the second-nearest exit.
- Supported visual variants are `classic_wall_maze`, `paper_labyrinth_maze`, and `block_wall_maze`.

## Prompt Contract

- Bundle: `puzzles_maze_v1`
- `scene_key`: `maze`
- `task_key`: `nearest_exit_label_query`
- `query_key`: `nearest_exit_label`
- Prompt-facing answer is the selected nearest exit label.
- Prompt-facing annotation is one image-pixel point centered on the selected nearest exit marker.

## Annotation + Trace Contract

- `answer_gt.type`: `string`
- `annotation_gt.type`: `point`
- `projected_annotation` includes `point`, `pixel_point`, and `value`.
- `render_map.item_points_px` stores exit marker centers keyed by exit item id.
- `execution_trace` records public query `single`, internal query `nearest_exit_label`, all exit path lengths, nearest exit label/cell/path, answer label, and solver trace.
- Answer and annotation are projected from the same selected nearest exit item id.
- `scalar_annotation_checked=true`.

## Determinism

- Deterministic sampling/rendering from `instance_seed`, scene config, prompt bundle, and code version.
- No semantic auto-relaxation is used to force acceptance.
