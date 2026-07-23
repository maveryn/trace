# `task_graph__metro__shortest_path_length`

## Summary
1. Domain: `graph`
2. Scene id: `metro`
3. Task id: `task_graph__metro__shortest_path_length`
4. Objective: count route segments in the unique shortest station path between two labeled stations.
5. Implementation: `src/trace_tasks/tasks/graph/metro/shortest_path_length.py`.

## Query IDs
1. Supported `query_id` values: `single`.
2. Internal prompt key: `metro_shortest_path_length`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: find the unique shortest station path between the named source and goal, then output its route-segment length.
2. Answer schema: `integer`.
3. Annotation schema: `point_sequence`.
4. Path length target, route count, station labels, route colors, style, font, background, and layout jitter are generation/render metadata, not public query branches.

## Program Contract

Program: `length(unique_shortest_station_path(source, goal)); output=integer; annotation=point_sequence(path_station_centers_after_source_in_order); scene=metro; scope=shortest_path_length`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `shortest_path_length` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique_shortest_station_path`, `source`, `goal`, `path_station_centers_after_source_in_order`, `metro`, `shortest_path_length`.
Operation: evaluate `length` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_sequence` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `point_sequence`.
3. Annotation marks the ordered station-center points along the unique shortest path after the source station, including the goal station.
4. `answer == len(annotation)`, and both come from the same finalized metro-route graph.

## Rendering Contract
1. The scene shows a labeled metro route map with colored routes and station nodes.
2. Visual style, fonts, panel treatment, layout jitter, context text, and post-render noise are non-semantic and recorded in trace metadata.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/metro/graph_metro_v1.json`.
2. `scene_key`: `metro_route_map`.
3. `task_key`: `metro_route_query`.
4. `query_key`: `metro_shortest_path_length`.
5. Answer mode emits `{"answer": ...}`.
6. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with `annotation` matching the `point_sequence` schema.
