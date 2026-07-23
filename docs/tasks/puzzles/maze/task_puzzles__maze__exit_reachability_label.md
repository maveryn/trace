# `task_puzzles__maze__exit_reachability_label`

## Contract

1. Domain: `puzzles`
2. Scene package: `src/trace_tasks/tasks/puzzles/maze/`
3. Scene id: `maze`
4. Public task id: `task_puzzles__maze__exit_reachability_label`
5. Supported `query_id` values: `single`
6. Prompt query key: `exit_reachability_label`
7. Answer schema: `string`
8. Annotation schema: `point`
9. Program schema: `select_label(maze.exit, target_reachability=reachable|unreachable); scene=maze; scope=exit_reachability_label`

## Program Contract

Program: `select_label(maze.exit, target_reachability=reachable|unreachable); scene=maze; scope=exit_reachability_label`

Candidate set: the visible maze cells, walls, start marker, exits, labels, and reachability/path structure inside the `exit_reachability_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `maze`, `exit`, `target_reachability`, `reachable`, `unreachable`, `exit_reachability_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; the selected exit label.
Annotation witnesses: `annotation` uses the `point` schema; one image-pixel point centered on the selected exit marker.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Query Contract

- Supported public `query_id`: `single`
- The task samples whether the requested target is reachable or unreachable and records it as `target_reachability` in trace metadata.
- Maze dimensions, exit count, target reachability, scene treatment, font, marker shape, and theme are generation/render axes, not public taxonomy axes.

## Generation Contract

- The renderer shows one orthogonal wall maze with one `START` cell and labeled exits on the outer boundary.
- Exactly one exit has the requested reachability status from `START`.
- Movement follows open corridors only; walls block motion.
- Supported visual variants are `classic_wall_maze`, `paper_labyrinth_maze`, and `block_wall_maze`.

## Prompt Contract

- Bundle: `puzzles_maze_v1`
- `scene_key`: `maze`
- `task_key`: `exit_reachability_label_query`
- `query_key`: `exit_reachability_label`
- Prompt-facing answer is the selected exit label.
- Prompt-facing annotation is one image-pixel point centered on the selected exit marker.

## Annotation + Trace Contract

- `answer_gt.type`: `string`
- `annotation_gt.type`: `point`
- `projected_annotation` includes `point`, `pixel_point`, and `value`.
- `render_map.item_points_px` stores exit marker centers keyed by exit item id.
- `execution_trace` records public query `single`, internal query `exit_reachability_label`, target reachability, maze topology, reachable/unreachable labels, answer label, and solver trace.
- Answer and annotation are projected from the same selected exit item id.
- `scalar_annotation_checked=true`.

## Determinism

- Deterministic sampling/rendering from `instance_seed`, scene config, prompt bundle, and code version.
- No semantic auto-relaxation is used to force acceptance.
