# `task_graph__adjacency__mst_weight`

## Summary
1. Domain: `graph`
2. Scene: `adjacency`
3. Task id: `task_graph__adjacency__mst_weight`
4. Objective: compute the total weight of the unique minimum spanning tree from a weighted adjacency matrix.
5. Implementation: `src/trace_tasks/tasks/graph/adjacency/mst_weight.py`.

## Query IDs
1. `single`: find the minimum spanning tree in a connected undirected weighted graph shown as a matrix.
2. Internal prompt key: `weighted_matrix_mst_weight`.

## Taxonomy Contract
1. Program contract: read the weighted undirected adjacency matrix, find the unique minimum spanning tree, sum its edge weights, and annotate one visible matrix cell for each MST edge.
2. Stable schemas: answer is `integer`; annotation is `bbox_set`.
3. Node labels, node count, extra edge count, sampled edge-weight range, font, style, and layout are generation/render metadata, not public query branches.

## Program Contract

Program: `sum(edge_weight(edge) for edge in minimum_spanning_tree(weighted_undirected_adjacency_graph)); output=integer; annotation=bbox_set(mst_edge_matrix_cells); scene=adjacency; scope=mst_weight`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `mst_weight` objective scope.
Operands: visible scene state and prompt-bound operands named by `edge_weight`, `edge`, `minimum_spanning_tree`, `weighted_undirected_adjacency_graph`, `mst_edge_matrix_cells`, `adjacency`, `mst_weight`.
Operation: evaluate `sum` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`

## Annotation
1. Answer type: `integer`.
2. Annotation type: `bbox_set`.
3. Annotation boxes are `[x0,y0,x1,y1]` pixel boxes around the MST-edge matrix cell whose row label is topmost among the two endpoint rows.

## Generation Notes
1. Blank off-diagonal cells mean no edge; the diagonal uses `-`.
2. Default node count is `4..7`; default extra non-tree edge count is `1..3`.
3. Edge weights are sampled so the intended MST is unique.
4. Node labels use graph label variants `letters|numbers|named`.
5. The weighted matrix panel samples approved font families and readable table styles; non-answer header context chips may appear.
6. Post-render graph noise follows the graph-domain coordinate-preserving noise policy.
