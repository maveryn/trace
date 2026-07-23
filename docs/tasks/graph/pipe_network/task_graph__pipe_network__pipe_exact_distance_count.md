# `task_graph__pipe_network__pipe_exact_distance_count`

## Summary
1. Domain: `graph`
2. Scene id: `pipe_network`
3. Task id: `task_graph__pipe_network__pipe_exact_distance_count`
4. Objective: count junctions at an exact shortest open-pipe distance from a named junction.
5. Implementation: `src/trace_tasks/tasks/graph/pipe_network/pipe_exact_distance_count.py`.

## Query IDs
1. Supported `query_id` values: `single`.
2. Internal prompt key: `pipe_exact_distance_count`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: compute shortest-path distances from the named junction using only open pipes, then count junctions at the sampled exact distance.
2. Answer schema: `integer`.
3. Annotation schema: `point_set`.
4. Query distance, target count, node count, grid shape, label style, color, font, background, context text, and noise are generation/render metadata, not public query branches.

## Program Contract

Program: `count(junction where shortest_open_pipe_distance(query_junction, junction) == k); output=integer; annotation=point_set(exact_distance_junction_centers); scene=pipe_network; scope=pipe_exact_distance_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `pipe_exact_distance_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `junction`, `where`, `shortest_open_pipe_distance`, `query_junction`, `k`, `exact_distance_junction_centers`, `pipe_network`, `pipe_exact_distance_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `point_set`.
3. Annotation marks one `[x,y]` junction-center point per junction at the queried shortest open-pipe distance.
4. The answer equals the number of annotation points, and both come from the same finalized open-pipe graph.

## Rendering Contract
1. The scene shows labeled junction fittings connected by open pipes and visible blocked pipes marked with a red X.
2. Only open pipes are used for shortest-distance computation; blocked pipes are visible distractors.
3. Visual style, fonts, panel treatment, layout jitter, context text, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/pipe_network/graph_pipe_network_v1.json`.
2. `scene_key`: `pipe_network`.
3. `task_key`: `pipe_network_query`.
4. `query_key`: `pipe_exact_distance_count`.
5. Answer mode emits `{"answer": ...}`.
6. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with `annotation` matching the `point_set` schema.
