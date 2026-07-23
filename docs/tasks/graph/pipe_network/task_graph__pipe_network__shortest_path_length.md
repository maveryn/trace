# `task_graph__pipe_network__shortest_path_length`

## Summary
1. Domain: `graph`
2. Scene id: `pipe_network`
3. Task id: `task_graph__pipe_network__shortest_path_length`
4. Objective: count open pipe segments in the unique shortest open route between two labeled junctions.
5. Implementation: `src/trace_tasks/tasks/graph/pipe_network/shortest_path_length.py`.

## Query IDs
1. Supported `query_id` values: `single`.
2. Internal prompt key: `pipe_shortest_path_length`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: find the unique shortest path in the open-pipe graph between the named source and goal, then output its edge length.
2. Answer schema: `integer`.
3. Annotation schema: `point_sequence`.
4. Target path length, node count, grid shape, label style, color, font, background, context text, and noise are generation/render metadata, not public query branches.

## Program Contract

Program: `length(unique_shortest_open_pipe_path(source, goal)); output=integer; annotation=point_sequence(path_junction_centers_in_order); scene=pipe_network; scope=shortest_path_length`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `shortest_path_length` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique_shortest_open_pipe_path`, `source`, `goal`, `path_junction_centers_in_order`, `pipe_network`, `shortest_path_length`.
Operation: evaluate `length` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_sequence` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `point_sequence`.
3. Annotation marks the ordered junction-center points along the shortest path, including both endpoints.
4. `answer == len(annotation) - 1`, and both come from the same finalized open-pipe graph.

## Rendering Contract
1. The scene shows labeled junction fittings connected by open pipes and visible blocked pipes marked with a red X.
2. Only open pipes are traversable; blocked pipes are visible distractors.
3. Visual style, fonts, panel treatment, layout jitter, context text, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/pipe_network/graph_pipe_network_v1.json`.
2. `scene_key`: `pipe_network`.
3. `task_key`: `pipe_network_query`.
4. `query_key`: `pipe_shortest_path_length`.
5. Answer mode emits `{"answer": ...}`.
6. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with `annotation` matching the `point_sequence` schema.
