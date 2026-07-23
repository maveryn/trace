# `task_graph__adjacency__directed_strong_component_count`

## Summary
1. Domain: `graph`
2. Scene id: `adjacency`
3. Task id: `task_graph__adjacency__directed_strong_component_count`
4. Objective: count strongly connected components in a directed adjacency representation.
5. Implementation: `src/trace_tasks/tasks/graph/adjacency/directed_strong_component_count.py`.

## Query IDs
1. `single`
2. Internal prompt key: `directed_strong_component_count`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: compute strongly connected components of the directed adjacency graph, count the components, and annotate the topmost displayed row label for each component.
2. Stable schemas: answer is `integer`; annotation is `bbox_set`.
3. Adjacency list versus matrix display, component count target, node labels, node count, font, style, and layout are generation/render metadata, not public query branches.

## Program Contract

Program: `count(strongly_connected_components(directed_adjacency_graph)); output=integer; annotation=bbox_set(component_representative_row_labels); scene=adjacency; scope=directed_strong_component_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `directed_strong_component_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `strongly_connected_components`, `directed_adjacency_graph`, `component_representative_row_labels`, `adjacency`, `directed_strong_component_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `bbox_set`.
3. Annotation marks the topmost displayed row label in each strongly connected component.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)` unless the annotation schema is keyed or sequence based.

## Rendering Contract
1. The scene uses the graph-domain renderer for `adjacency`.
2. Visual style, fonts, panel treatment, layout jitter, and post-render noise are non-semantic and must be recorded in trace metadata.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from graph prompt templates and scene config, not hardcoded user-facing text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
