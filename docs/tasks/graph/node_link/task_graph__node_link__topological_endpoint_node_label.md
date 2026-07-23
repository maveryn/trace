# `task_graph__node_link__topological_endpoint_node_label`

## Program Contract

Program: `label(endpoint(topological_order(dag), endpoint=first_or_last)); scene=node_link; scope=topological_endpoint_node_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `topological_endpoint_node_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `endpoint`, `topological_order`, `dag`, `first_or_last`, `node_link`, `topological_endpoint_node_label`.
Operation: evaluate `label` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `topology`

## Summary
1. Domain: `graph`
2. Scene id: `node_link`
3. Task id: `task_graph__node_link__topological_endpoint_node_label`
4. Objective: return the first or last node label in a directed graph's unique topological order.

## Query IDs
1. `first_in_topological_order_label`
2. `last_in_topological_order_label`
3. Query ids are internal replay metadata; public sampling is at the task-id level.

## Answer And Annotation
1. Answer type: `string`.
2. Annotation type: `point`.
3. Annotation marks one minimal pixel-space witness: the answered node center.
4. The scene generator guarantees a unique topological order before selecting the endpoint.

## Rendering Contract
1. The scene uses the graph-domain renderer for `node_link`.
2. Directed edges render arrowheads.
3. Visual style, fonts, panel treatment, layout jitter, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from graph prompt templates and scene config, not hardcoded user-facing text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
