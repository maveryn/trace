# `task_graph__node_link__unique_related_node_label`

## Program Contract

Program: `label(unique(related_nodes(reference_node, relation))); scene=node_link; scope=unique_related_node_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `unique_related_node_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique`, `related_nodes`, `reference_node`, `relation`, `node_link`, `unique_related_node_label`.
Operation: evaluate `label` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Summary
1. Domain: `graph`
2. Scene id: `node_link`
3. Scene: `relation`
4. Task id: `task_graph__node_link__unique_related_node_label`
5. Objective: return the unique node label satisfying the requested direct relation.

## Query IDs
1. `unique_neighbor_label|unique_successor_label|unique_predecessor_label`
2. Query ids are internal replay metadata; public sampling is at the task-id level.

## Answer And Annotation
1. Answer type: `string`.
2. Annotation type: `point`.
3. Annotation marks the node-center pixel point for the unique answer node.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)` unless the annotation schema is keyed or sequence based.

## Rendering Contract
1. The scene uses the graph-domain renderer for `node_link`.
2. Visual style, fonts, panel treatment, layout jitter, and post-render noise are non-semantic and must be recorded in trace metadata.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from graph prompt templates and scene config, not hardcoded user-facing text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
