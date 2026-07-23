# `task_graph__node_link__shortest_path_length`

## Program Contract

Program: `length(shortest_path(graph, source_node, goal_node)); scene=node_link; scope=shortest_path_length`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `shortest_path_length` objective scope.
Operands: visible scene state and prompt-bound operands named by `shortest_path`, `graph`, `source_node`, `goal_node`, `node_link`, `shortest_path_length` plus the active `query_id` branch.
Operation: evaluate `length` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the number of edges on that path,.
Annotation witnesses: `annotation` uses the `point_sequence` schema; the `point_sequence` of node-center pixel points after the source node along the unique shortest path, ending at the goal node.
Query ids: `undirected_shortest_path_length`, `directed_shortest_path_length`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `path`
3. Scene id: `node_link`
4. Task id: `task_graph__node_link__shortest_path_length`
5. Objective: count how many edges lie on the unique shortest path between two labeled nodes in an undirected or directed graph.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `undirected_shortest_path_length` or `directed_shortest_path_length`
3. Supported `graph_directionality` values: `undirected`, `directed`
4. Supported `scene_variant` values: `circular`, `shell`, `spring`, `grid_jitter`, `layered`, `component_clustered`, `path_spine`, `radial_tree`
5. `answer_gt.type`: `integer`
6. `annotation_gt.type`: `point_sequence`
7. Scene contract:
   - one single-panel labeled node-link graph,
   - simple unweighted graph only,
   - no self-loops or multi-edges,
   - directed branches also reject reciprocal directed edge pairs,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`),
   - node count is sampled from `5..15`,
   - generation leaves at least one node outside the queried shortest path.
8. Query contract:
   - undirected branch asks how many edges are in the unique shortest path from node X to node Y,
   - directed branch asks how many edges are in the unique path from node X to node Y when following arrow direction,
   - `target_shortest_path_length` support is `2..5`,
   - answer is the number of edges on that path,
   - annotation is the ordered node-center pixel path after the source node, ending at the goal node.

## 3) Prompt contract
1. Bundle: `graph_node_link_path_v1`
2. `scene_key`: `single_graph_path`
3. `task_key`: `shortest_path_length_query`
4. Modes: `answer_only`, `answer_and_annotation`
5. Answer JSON shape: `{"answer":2}`
6. Answer+annotation JSON shape: `{"annotation":[[310,180],[430,260]],"answer":2}`
7. Prompt-facing annotation uses an ordered pixel point path because shortest-path semantics depend on source-to-goal order.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the `point_sequence` of node-center pixel points after the source node along the unique shortest path, ending at the goal node.
2. The annotation path excludes the source endpoint.
3. `answer_gt.value == len(annotation_gt.value)` by construction.
4. `execution_trace.query_id` records the concrete public branch: `undirected_shortest_path_length` or `directed_shortest_path_length`.
5. The internal generator records `internal_query_id == "shortest_path_length"` or `internal_query_id == "directed_shortest_path_length"` for diagnostic compatibility.
6. `execution_trace.graph_directionality` records `undirected` or `directed`.
7. `projected_annotation` includes `point_sequence`, `pixel_point_sequence`, and `pixel_bbox_set`.

## 5) Visual policy
1. Rendering uses the shared graph light-panel style from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. Directed branches render arrowheads and use explicit direction-following wording.
3. Node label format, edge routing, glyph style, named node color, layout transform, and layout are visual variation only.
4. The task enables the full node-link layout set for generation: the three original layouts plus grid, layered, clustered, path-spine, and radial-tree layouts.
5. When `label_variant=named`, prompt references to source and goal nodes are quoted, for example node `"Lima"`.
6. The prompt never refers to node position, color, or shape as semantic annotation.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized adjacency map and verified unique-shortest-path computation.
3. Generation rejects graphs with no path or more than one shortest path between the queried endpoints under the selected directionality.

## 7) Complexity + tests
1. Complexity components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Tests: `tests/test_graph_path_shortest_path_length_contracts.py`, `tests/test_graph_path_shortest_path_length_tasks.py`, `tests/test_scene_config.py`
