# `task_graph__adjacency__directed_pair_reciprocity_count`

## Summary
1. Domain: `graph`
2. Scene id: `adjacency`
3. Task id: `task_graph__adjacency__directed_pair_reciprocity_count`
4. Objective: count unordered node pairs in a directed adjacency matrix where both directed edges are present.
5. Implementation: `src/trace_tasks/tasks/graph/adjacency/directed_pair_reciprocity_count.py`.

## Query IDs
1. `single`: count unordered pairs where both mirrored off-diagonal matrix cells are `1`.
2. Internal prompt key: `mutual_pair_count`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: build the unordered off-diagonal node-pair candidate set, filter pairs whose two directed matrix cells are both present, count the filtered pairs, and annotate one mirrored-cell segment per counted pair.
2. Stable schemas: answer is `integer`; annotation is `segment_set`.
3. `target_count`, node labels, node count, font, style, and matrix layout are generation/render metadata, not public query branches.

## Program Contract

Program: `count(filter(unordered_node_pairs, has_edge(source,target) and has_edge(target,source))); output=integer; annotation=segment_set(mutual_edge_cell_centers); scene=adjacency; scope=directed_pair_reciprocity_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `directed_pair_reciprocity_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `unordered_node_pairs`, `has_edge`, `source`, `mutual_edge_cell_centers`, `adjacency`, `directed_pair_reciprocity_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `segment_set`.
3. Annotation marks one segment per counted unordered pair, using the centers of the two mirrored matrix cells.
4. Zero-answer instances use an empty annotation array.

## Rendering Contract
1. The scene uses the graph-domain adjacency-matrix renderer.
2. The graph is directed; rows point to columns.
3. Diagonal cells are excluded from pair counting.
4. Visual style, fonts, panel treatment, context text, and post-render noise are non-semantic and recorded in trace metadata.
5. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from graph prompt templates and scene config, not hardcoded user-facing text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
