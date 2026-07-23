# `task_graph__node_link__named_node_degree_value`

## Program Contract

Program: `value(degree_metric(query_node)); scene=node_link; scope=named_node_degree_value`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `named_node_degree_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `degree_metric`, `query_node`, `node_link`, `named_node_degree_value` plus the active `query_id` branch.
Operation: evaluate `value` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the queried degree value for the named node.
Annotation witnesses: `annotation` uses the `segment_set` schema; the `segment_set` of endpoint-center segments for all edges counted toward the queried degree value.
Query ids: `undirected_named_node_degree_value`, `directed_named_node_in_degree_value`, `directed_named_node_out_degree_value`, `directed_named_node_total_degree_value`.

## Reasoning Operations

Families: `filtering`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `counting`
3. Scene id: `node_link`
4. Task id: `task_graph__node_link__named_node_degree_value`
5. Objective: ask for the degree, in-degree, out-degree, or total degree of one specific labeled node.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `undirected_named_node_degree_value`, `directed_named_node_in_degree_value`, `directed_named_node_out_degree_value`, or `directed_named_node_total_degree_value`
3. Supported `graph_directionality` values: `undirected`, `directed`
4. Supported directed `degree_mode` values: `in_degree`, `out_degree`, `total_degree`
5. Supported `scene_variant` values: `circular`, `shell`, `spring`, `grid_jitter`, `layered`, `component_clustered`, `path_spine`, `radial_tree`
6. `answer_gt.type`: `integer`
7. `annotation_gt.type`: `segment_set`
8. Scene contract:
   - one single-panel labeled node-link graph,
   - simple unweighted graph only,
   - no self-loops or multi-edges,
   - directed branches also reject reciprocal directed edge pairs,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`),
   - node count is sampled from `5..10`.
9. Query contract:
   - undirected branch asks for the degree of a named node,
   - directed branches ask for the named node's in-degree, out-degree, or total degree,
   - `target_degree` support is `0..4`,
   - answer is the queried degree value for the named node.

## 3) Prompt contract
1. Bundle: `graph_node_link_counting_v1`
2. `scene_key`: `single_graph_counting`
3. `task_key`: `named_node_degree_value_query`
4. Modes: `answer_only`, `answer_and_annotation`
5. Answer JSON shape: `{"answer":2}`
6. Answer+annotation JSON shape: `{"annotation":[[[180,220],[310,180]],[[180,220],[430,260]]],"answer":2}`
7. Prompt-facing annotation uses `segment_set`; each segment is `[[x0, y0], [x1, y1]]`, where each endpoint is an `[x, y]` pixel point at an endpoint node center for one counted edge.
8. If the node label format is `named`, prompt references quote the node label, for example node `"Abby"`.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the `segment_set` of endpoint-center segments for all edges counted toward the queried degree value.
2. For directed branches, segment endpoints correspond to the source and target node centers.
3. `answer_gt.value == len(annotation_gt.value)` by construction, including zero-degree cases where annotation is an empty array.
4. `execution_trace.graph_directionality` records `undirected` or `directed`; `execution_trace.degree_mode` records `degree`, `in_degree`, `out_degree`, or `total_degree`.
5. `execution_trace.query_label` records the canonical node label and `execution_trace.query_label_prompt` records the prompt-facing label after short-name quoting.
6. `scene_ir.entities` stores node labels, total degrees, directed degrees, queried degree values, neighbors/successors/predecessors, center points, and bboxes.
7. `scene_ir.entities` stores `is_counted` on graph edges that contribute to the answer.
8. `projected_annotation` includes `segment_set`.

## 5) Visual policy
1. Rendering uses the shared graph light-panel style from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. Directed branches render arrowheads and reject reciprocal edge pairs so the queried degree mode remains readable.
3. Node label format, edge routing, glyph style, named node color, layout transform, and layout are visual variation only.
4. The prompt never refers to node position, color, or shape as semantic annotation.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same adjacency map and rendered node assignment.
3. Generation rejects samples that cannot realize the requested `(graph_directionality, degree_mode, target_degree)` exactly.

## 7) Complexity + tests
1. Complexity components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Tests: `tests/test_graph_counting_named_node_degree_value_contracts.py`, `tests/test_graph_counting_named_node_degree_value_tasks.py`, `tests/test_scene_config.py`
