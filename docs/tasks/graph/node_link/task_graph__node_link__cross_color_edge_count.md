# `task_graph__node_link__cross_color_edge_count`

## Program Contract

Program: `count(filter(edges(graph), endpoint_colors={source_color,target_color})); scene=node_link; scope=cross_color_edge_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `cross_color_edge_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `edges`, `graph`, `endpoint_colors`, `source_color`, `target_color`, `node_link`, `cross_color_edge_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; a `segment_set` for every counted edge; each segment is `[[x0, y0], [x1, y1]]`, where each endpoint is an `[x, y]` pixel point at an endpoint node center.
Query ids: `cross_color_edge_count`, `directed_cross_color_edge_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `counting`
3. Scene id: `node_link`
4. Task id: `task_graph__node_link__cross_color_edge_count`
5. Objective: count graph edges whose endpoint nodes have two queried semantic colors.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `cross_color_edge_count` for undirected graphs, `directed_cross_color_edge_count` for directed graphs
3. Supported `graph_directionality` values: `undirected|directed`
4. Supported node colors: the 10-color shared Trace named-color palette: red, blue, green, yellow, orange, purple, brown, cyan, magenta, and maroon
5. `answer_gt.type`: `integer`
6. `annotation_gt.type`: `segment_set`
7. Undirected query: count edges with one endpoint in the first queried color and the other endpoint in the second queried color.
8. Directed query: count arrows whose source node has the first queried color and target node has the second queried color.
9. The queried source/target color names are sampled as an ordered distinct pair unless both are provided explicitly.

## 3) Prompt contract
1. Bundle: `graph_node_link_counting_v1`
2. `scene_key`: `single_graph_counting`
3. `task_key`: `cross_color_edge_count_query`
4. Modes: `answer_only`, `answer_and_annotation`
5. Answer JSON shape: `{"answer":2}`
6. Answer+annotation JSON shape: `{"annotation":[[[180,220],[310,180]],[[180,220],[430,260]]],"answer":2}`
7. Prompt-facing color text uses `<color_name> [#RRGGBB]`.

## 4) Annotation + trace contract
1. Prompt-facing annotation is a `segment_set` for every counted edge; each segment is `[[x0, y0], [x1, y1]]`, where each endpoint is an `[x, y]` pixel point at an endpoint node center.
2. For directed graphs, segment endpoints correspond to the source and target node centers.
3. `answer_gt.value == len(annotation_gt.value)` by construction, including zero-answer cases.
4. `execution_trace.node_color_names_by_label` records every node's semantic color.
5. `execution_trace.matching_edges` records the symbolic counted edge labels.

## 5) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. The sampler assigns node colors so the realized graph has exactly the requested cross-color edge count for the selected color pair.
3. Failures reject/resample; generation does not relax color, directionality, or target-count constraints.

## 6) Complexity + tests
1. Complexity components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Tests: `tests/test_graph_counting_cross_color_edge_count_tasks.py`
