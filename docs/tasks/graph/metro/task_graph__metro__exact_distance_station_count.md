# `task_graph__metro__exact_distance_station_count`

## Summary
1. Domain: `graph`
2. Scene id: `metro`
3. Task id: `task_graph__metro__exact_distance_station_count`
4. Objective: count stations at shortest metro-route distance exactly `k` route segments from a named station.
5. Implementation: `src/trace_tasks/tasks/graph/metro/exact_distance_station_count.py`.

## Query IDs
1. Supported `query_id` values: `single`.
2. Internal prompt key: `metro_exact_distance_count`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: compute shortest-path distances from the named station and count stations at the sampled exact distance.
2. Answer schema: `integer`.
3. Annotation schema: `point_set`.
4. Query distance, route count, station labels, route colors, style, font, background, and layout jitter are generation/render metadata, not public query branches.

## Program Contract

Program: `count(station where shortest_route_distance(query_station, station) == k); output=integer; annotation=point_set(matching_station_centers); scene=metro; scope=exact_distance_station_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `exact_distance_station_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `station`, `where`, `shortest_route_distance`, `query_station`, `k`, `matching_station_centers`, `metro`, `exact_distance_station_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `point_set`.
3. Annotation marks one `[x,y]` station-center point per station at the queried distance.
4. The answer equals the number of annotation points, and both come from the same finalized metro-route graph.

## Rendering Contract
1. The scene shows a labeled metro route map with colored routes and station nodes.
2. The queried station is visually highlighted.
3. Visual style, fonts, panel treatment, layout jitter, context text, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/metro/graph_metro_v1.json`.
2. `scene_key`: `metro_route_map`.
3. `task_key`: `metro_route_query`.
4. `query_key`: `metro_exact_distance_count`.
5. Answer mode emits `{"answer": ...}`.
6. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with `annotation` matching the `point_set` schema.
