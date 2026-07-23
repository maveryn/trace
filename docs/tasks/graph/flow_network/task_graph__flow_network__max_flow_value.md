# `task_graph__flow_network__max_flow_value`

## Summary
1. Domain: `graph`
2. Scene id: `flow_network`
3. Task id: `task_graph__flow_network__max_flow_value`
4. Objective: compute the maximum flow value from source `S` to sink `T`.
5. Implementation: `src/trace_tasks/tasks/graph/flow_network/max_flow_value.py`.

## Query IDs
1. Supported `query_id` values: `single`
2. Internal prompt key: `max_flow_value`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: read a directed capacity graph, find the unique minimum `S`-`T` cut, and return its capacity as the maximum-flow value.
2. Answer schema: `integer`.
3. Annotation schema: `segment_set`.
4. Node count, capacity values, cut-edge count, distractor-edge count, graph style, font, background, and layout transform are generation/render metadata, not public query branches.

## Program Contract

Program: `value(max_flow(source=S, sink=T, directed_capacity_graph)); output=integer; annotation=segment_set(unique_minimum_cut_edges); scene=flow_network; scope=max_flow_value`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `max_flow_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `max_flow`, `source`, `S`, `sink`, `T`, `directed_capacity_graph`, `unique_minimum_cut_edges`, `flow_network`, `max_flow_value`.
Operation: evaluate `value` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `segment_set`.
3. Annotation marks every directed edge in the unique minimum `S`-`T` cut as a pixel-space segment between endpoint node centers.
4. The answer equals the sum of capacities on the annotated cut edges.

## Rendering Contract
1. The scene shows one directed capacity network with highlighted `S` and `T` nodes.
2. Every visible directed edge has a readable integer capacity label.
3. This objective uses a left-to-right identity layout transform and rejects rendered edge crossings so the directed flow channels are clear.
4. Visual style, fonts, context text, background, and post-render noise are non-semantic and recorded in trace metadata.
5. Annotation projection is computed after final graph layout and image-level variation.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/flow_network/graph_flow_network_v1.json`.
2. `scene_key`: `capacity_network`.
3. `task_key`: `flow_network_query`.
4. `query_key`: `max_flow_value`.
5. Answer mode emits `{"answer": ...}`.
6. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with `annotation` matching the `segment_set` schema.
