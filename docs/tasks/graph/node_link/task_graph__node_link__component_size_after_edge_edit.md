# `task_graph__node_link__component_size_after_edge_edit`

## Program Contract

Program: `count(component_nodes(apply_edge_edit(graph, edit_edge), reference_node)); scene=node_link; scope=component_size_after_edge_edit`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `component_size_after_edge_edit` objective scope.
Operands: visible scene state and prompt-bound operands named by `component_nodes`, `apply_edge_edit`, `graph`, `edit_edge`, `reference_node`, `node_link`, `component_size_after_edge_edit`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`, `state_update`

## Summary
1. Domain: `graph`
2. Scene id: `node_link`
3. Scene: `relation`
4. Task id: `task_graph__node_link__component_size_after_edge_edit`
5. Objective: count the component containing a reference node after one edge edit.

## Query IDs
1. `component_size_after_edge_removal|component_size_after_edge_addition`
2. Query ids are internal replay metadata; public sampling is at the task-id level.

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `point_set`.
3. Annotation marks minimal pixel-space visual witnesses for the answer, not answer labels or non-witness annotations.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)` unless the annotation schema is keyed or sequence based.

## Rendering Contract
1. The scene uses the graph-domain renderer for `node_link`.
2. Visual style, fonts, panel treatment, layout jitter, and post-render noise are non-semantic and must be recorded in trace metadata.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from graph prompt templates and scene config, not hardcoded user-facing text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
