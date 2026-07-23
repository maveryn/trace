# `task_graph__adjacency__traversal_kth_label`

## Summary
1. Domain: `graph`
2. Scene: `adjacency`
3. Task id: `task_graph__adjacency__traversal_kth_label`
4. Objective: return the node label at a requested position in a BFS or DFS traversal from an adjacency list.
5. Implementation: `src/trace_tasks/tasks/graph/adjacency/traversal_kth_label.py`.

## Query IDs
1. `bfs_kth_visit_label`: breadth-first search from a named source row, using each row's neighbor order left to right; the source node counts as visit position 1.
2. `dfs_kth_visit_label`: recursive depth-first search from a named source row, using each row's neighbor order left to right; the source node counts as visit position 1.

## Taxonomy Contract
1. Program contract: traverse the directed adjacency-list graph from the sampled source using the query-selected traversal operator, return the label at the requested one-indexed visit position, and annotate the ordered row-label prefix from source through answer.
2. Stable schemas: answer is `string`; annotation is `bbox_sequence`.
3. The BFS versus DFS operator is a semantic query branch. Source label, visit position, node labels, node count, extra edge count, font, style, and layout are generation/render metadata.

## Program Contract

Program: `label_at(traverse(adjacency_list_graph, source_node, traversal_operator), visit_position); output=string; annotation=bbox_sequence(visited_row_label_prefix); scene=adjacency; scope=traversal_kth_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `traversal_kth_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `traverse`, `adjacency_list_graph`, `source_node`, `traversal_operator`, `visit_position`, `visited_row_label_prefix`, `adjacency`, `traversal_kth_label`.
Operation: evaluate `label_at` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_sequence` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `topology`

## Annotation
1. Answer type: `string`.
2. Annotation type: `bbox_sequence`.
3. Annotation boxes are `[x0,y0,x1,y1]` pixel boxes around the visited row labels, ordered from the source row through the answer row.

## Generation Notes
1. The scene renders a directed graph as an adjacency-list panel.
2. Default node count is `5..8`; default traversal position is `2..8`.
3. Every node is reachable from the sampled source row by construction.
4. Node labels use graph label variants `letters|numbers|named`.
5. The adjacency-list panel samples approved font families and readable list styles; non-answer header context chips may appear.
6. Post-render graph noise follows the graph-domain coordinate-preserving noise policy.
