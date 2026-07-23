# `task_graph__metro__route_condition_station_count`

## Summary
1. Domain: `graph`
2. Scene id: `metro`
3. Task id: `task_graph__metro__route_condition_station_count`
4. Objective: count stations on a named colored route that satisfy a route-membership condition.
5. Implementation: `src/trace_tasks/tasks/graph/metro/route_condition_station_count.py`.

## Query IDs
1. Supported `query_id` values: `metro_route_transfer_station_count`, `metro_route_single_route_station_count`.
2. Internal prompt keys match the supported query ids.

## Taxonomy Notes
1. Program contract: count stations on one named route that satisfy the queried route-membership predicate.
2. Answer schema: `integer`.
3. Annotation schema: `point_set`.
4. Named route, route count, station labels, route colors, style, font, background, and layout jitter are generation/render metadata, not public query branches.

## Program Contract

Program: `count(station where station_on_route(named_route) and route_membership(station) satisfies queried_predicate); output=integer; annotation=point_set(matching_station_centers); scene=metro; scope=route_condition_station_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `route_condition_station_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `station`, `where`, `station_on_route`, `named_route`, `route_membership`, `satisfies`, `queried_predicate`, `matching_station_centers`, `metro`, `route_condition_station_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `metro_route_transfer_station_count`, `metro_route_single_route_station_count`, `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `point_set`.
3. Annotation marks one `[x,y]` station-center point per station on the named route satisfying the queried condition.
4. The annotation is an empty array when no station on the named route satisfies the queried condition.
5. For `metro_route_transfer_station_count`, counted stations are on the named route and served by at least two colored routes.
6. For `metro_route_single_route_station_count`, counted stations are on the named route and served only by that route.

## Rendering Contract
1. The scene shows a labeled metro route map with colored routes and station nodes.
2. Route names are visible in the route legend.
3. Transfer stations are visually indicated by multi-route overlap and larger station styling.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/metro/graph_metro_v1.json`.
2. `scene_key`: `metro_route_map`.
3. `task_key`: `metro_route_query`.
4. `query_key`: selected supported `query_id`.
