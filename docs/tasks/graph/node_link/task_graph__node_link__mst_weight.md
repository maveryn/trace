# `task_graph__node_link__mst_weight`

## Program Contract

Program: `sum(weights(minimum_spanning_tree(graph))); scene=node_link; scope=mst_weight`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `mst_weight` objective scope.
Operands: visible scene state and prompt-bound operands named by `weights`, `minimum_spanning_tree`, `graph`, `node_link`, `mst_weight`.
Operation: evaluate `sum` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the sum of the weights on the unique minimum spanning tree,.
Annotation witnesses: `annotation` uses the `segment_set` schema; the `segment_set` of endpoint-node-center segments for all MST edges.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `optimization`
3. Scene id: `weighted_node_link`
4. Task id: `task_graph__node_link__mst_weight`
5. Objective: return the total weight of the graph's unique minimum spanning tree.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `single`
3. Supported `scene_variant` values: `circular`, `shell`, `spring`, `grid_jitter`, `layered`, `component_clustered`, `path_spine`, `radial_tree`
3. `answer_gt.type`: `integer`
4. `annotation_gt.type`: `segment_set`
5. Scene contract:
   - one single-panel labeled connected weighted graph per image,
   - undirected graph only,
   - no self-loops,
   - no multi-edges,
   - visible node labels use uppercase letters by default (`A..J`),
   - node count is sampled from `4..7`,
   - generation always includes at least one non-tree edge so the scene never collapses to a bare tree.
6. Query contract:
   - the prompt defines a spanning tree as connecting every node without cycles and asks for the smallest total edge weight,
   - answer is the sum of the weights on the unique minimum spanning tree,
     - annotation is the unordered set of MST edges, represented as endpoint-node-center pixel segments `[[x0, y0], [x1, y1]]`, with each endpoint as an `[x, y]` pixel point.
7. Weight policy:
   - edge weights are distinct integers from `1..9`,
   - generation samples `1..2` non-tree edges,
   - the heaviest sampled weights are assigned to non-tree edges so the intended spanning tree is the unique MST by construction,
   - the finalized weighted graph is rechecked with `networkx.minimum_spanning_tree(...)` before exposing answer or annotation.
8. Topology variation:
   - `topology_profile` values are `balanced`, `low_degree`, and `hub_heavy`,
   - topology profile affects both the spanning-tree shape and which non-tree edges are added,
   - graph semantics always come from adjacency + weights, never from layout position.
9. Layout variation:
   - `scene_variant` records the realized graph layout,
   - requested layout variants are all reusable node-link layouts: `circular|shell|spring|grid_jitter|layered|component_clustered|path_spine|radial_tree`,
   - the renderer may fall back to a cleaner node-link layout when a sampled layout is too cramped for readable node or edge-label separation.
10. Visual variation:
   - one whole-image named node color is sampled from the shared Trace named-color palette,
   - one whole-image node glyph style is sampled from `circle|rounded_square|hexagon`,
   - one whole-image label format is sampled as `letters` by default; numeric labels remain renderer-supported but are not part of this task's default generation distribution,
   - one whole-image layout transform is sampled from `identity|rotate_90|rotate_180|rotate_270|mirror_left_right|mirror_up_down`,
   - edge weights are rendered as small boxed integers near each edge,
   - these style axes are non-semantic for this task and are recorded in trace metadata.

## 3) Prompt contract
1. Bundle: `graph_node_link_optimization_v1`
2. `scene_key`: `single_graph_optimization`
3. `task_key`: `minimum_spanning_tree_weight_query`
4. Required slots:
   - scene: `object_description`
   - task: `question_text`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Answer JSON shape: `{"answer":12}`
7. Answer+annotation JSON shape: `{"annotation":[[[180,220],[310,180]],[[310,180],[430,260]],[[430,260],[520,340]]],"answer":12}`
8. Prompt-facing annotation uses `segment_set`; each edge is one endpoint-node-center segment `[[x0, y0], [x1, y1]]`, with each endpoint as an `[x, y]` pixel point.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the `segment_set` of endpoint-node-center segments for all MST edges.
2. Each annotation item is an undirected segment; endpoint order is unordered semantically for reward matching, and the corresponding endpoint labels remain in `witness_symbolic`.
3. `scene_ir.entities` stores one node entity per rendered node plus one edge entity per rendered edge with:
   - endpoint labels,
   - rendered segment,
   - integer edge weight,
   - weight-label bbox,
   - MST-membership flag.
4. `scene_ir.relations` stores:
   - graph directionality,
   - full edge-weight list,
   - MST edge list,
   - adjacency map,
   - degree map.
5. `projected_annotation` includes:
   - `segment_set`
   - `segment_map`
6. `execution_trace` records:
   - `query_id` (the concrete public query branch)
   - `scene_variant`
   - `node_count`
   - `extra_edge_count`
   - edge-weight range
   - total MST weight
   - MST edges
   - weighted edge list
   - topology profile
   - requested and realized layout variants

## 5) Visual policy
1. Background and post-image noise use the merged graph-domain visual defaults from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. Current weighted graph scenes use a single rounded light panel on a light solid background.
3. Node labels are rendered inside the nodes; edge weights are rendered in small boxed labels near edge midpoints.
4. Layout and styling may vary for readability and diversity, but the prompt never refers to node position, node color, or node shape as the semantic source of truth.
5. Weight labels stay semantic for this task, so trace metadata and rendered labels come from the same canonical edge-weight map.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized weighted graph.
3. Unique-answer policy: generation enforces a unique MST by construction and rejects any graph whose final weighted edges do not preserve that unique witness.
4. Reject/resample conditions:
   - infeasible node-count / extra-edge-count / weight-range combination,
   - failure to realize a connected weighted graph with at least one non-tree edge,
   - failure to preserve the intended unique MST after weight assignment,
   - unreadable layout that cannot keep nodes sufficiently separated.
5. No semantic auto-relaxation: failures do not weaken the connectivity, weight, or unique-MST contract.

## 7) Complexity + tests
1. Complexity definition/components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Determinism/build tests: `tests/test_graph_optimization_minimum_spanning_tree_weight_contracts.py`
3. Behavior/trace/prompt tests: `tests/test_graph_optimization_minimum_spanning_tree_weight_tasks.py`
4. Prompt bundle/config tests: `tests/test_prompt_system.py`, `tests/test_scene_config.py`
