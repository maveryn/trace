# `task_graph__flow_network__min_cut_edge_count`

## Summary
1. Domain: `graph`
2. Scene id: `flow_network`
3. Task id: `task_graph__flow_network__min_cut_edge_count`
4. Objective: count the directed edges in the unique minimum `S`-`T` cut.
5. Implementation: `src/trace_tasks/tasks/graph/flow_network/min_cut_edge_count.py`.

## Query IDs
1. Supported `query_id` values: `single`
2. Internal prompt key: `minimum_cut_edge_count`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: read a directed capacity graph, find the unique minimum `S`-`T` cut, and count the directed edges in that cut.
2. Answer schema: `integer`.
3. Annotation schema: `segment_set`.
4. Node count, capacity values, target cut-edge count, target flow value, distractor-edge count, graph style, font, background, and layout transform are generation/render metadata, not public query branches.

## Program Contract

Program: `count(edges(unique_minimum_cut(source=S, sink=T, directed_capacity_graph))); output=integer; annotation=segment_set(unique_minimum_cut_edges); scene=flow_network; scope=min_cut_edge_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `min_cut_edge_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `edges`, `unique_minimum_cut`, `source`, `S`, `sink`, `T`, `directed_capacity_graph`, `unique_minimum_cut_edges`, `flow_network`, `min_cut_edge_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `segment_set`.
3. Annotation marks every directed edge in the unique minimum `S`-`T` cut as a pixel-space segment between endpoint node centers.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)`.

## Rendering Contract
1. The scene shows one directed capacity network with highlighted `S` and `T` nodes.
2. Every visible directed edge has a readable integer capacity label.
3. Visual style, fonts, context text, background, layout transform, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final graph layout and image-level variation.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/flow_network/graph_flow_network_v1.json`.
2. `scene_key`: `capacity_network`.
3. `task_key`: `flow_network_query`.
4. `query_key`: `minimum_cut_edge_count`.
5. Answer mode emits `{"answer": ...}`.
6. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with `annotation` matching the `segment_set` schema.
