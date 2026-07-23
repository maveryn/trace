# `task_graph__binary_tree__depth_level_node_count`

## Program Contract

Program: `count(filter(nodes(binary_tree), depth=target_depth)); scene=binary_tree; scope=depth_level_node_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `depth_level_node_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `nodes`, `binary_tree`, `depth`, `target_depth`, `depth_level_node_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Summary
1. Domain: `graph`
2. Scene id: `binary_tree`
3. Scene: `counting`
4. Task id: `task_graph__binary_tree__depth_level_node_count`
5. Objective: count binary-tree nodes at a specified depth.

## Query IDs
1. `single`
2. Query ids are internal replay metadata; public sampling is at the task-id level.

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation schema: `point_set`.
3. Annotation marks node-center pixel points for every node at the requested depth.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)` unless the annotation schema is keyed or sequence based.

## Rendering Contract
1. The scene uses the graph-domain renderer for `binary_tree`.
2. Visual style, fonts, panel treatment, layout jitter, and post-render noise are non-semantic and must be recorded in trace metadata.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from graph prompt assets, not hardcoded user-facing text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
