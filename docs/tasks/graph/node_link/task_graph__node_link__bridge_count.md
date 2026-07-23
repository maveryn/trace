# `task_graph__node_link__bridge_count`

## Program Contract

Program: `count(filter(edges(graph), bridge=True)); scene=node_link; scope=bridge_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `bridge_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `edges`, `graph`, `bridge`, `True`, `node_link`, `bridge_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the number of edges whose removal increases the number of connected components of the graph.
Annotation witnesses: `annotation` uses the `segment_set` schema; the `segment_set` of endpoint-node-center segments for all bridge edges.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `counting`
3. Task id: `task_graph__node_link__bridge_count`
4. Objective: count how many labeled graph edges are bridges.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `single`
3. Supported `scene_variant` values: `circular`, `shell`, `spring`, `grid_jitter`, `layered`, `component_clustered`, `path_spine`, `radial_tree`
3. `answer_gt.type`: `integer`
4. `annotation_gt.type`: `segment_set`
5. Scene contract:
   - one single-panel labeled undirected node-link graph per image,
   - simple unweighted graph only,
   - no self-loops,
   - no multi-edges,
   - node count sampled from `5..10`,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`).
6. Query contract:
   - the prompt asks `How many edges are bridges?`,
   - answer is the number of edges whose removal increases the number of connected components of the graph.
7. Count policy:
   - `target_count` is sampled from `0..5`,
   - node count is chosen from the feasible support that can realize the requested bridge count,
   - the sampler verifies the final bridge-edge set from the realized adjacency map before emitting answer/annotation.
8. Topology variation:
   - `topology_profile` values are `balanced`, `low_degree`, and `hub_heavy`,
   - topology profile affects how the bridge skeleton and any bridgeless blocks are assembled,
   - graph semantics always come from adjacency, never from layout position.
9. Layout variation:
   - `scene_variant` records the realized graph layout,
   - requested layout variants are all reusable node-link layouts: `circular|shell|spring|grid_jitter|layered|component_clustered|path_spine|radial_tree`,
   - the renderer may fall back to `circular` when a sampled layout is too cramped for readable node separation.
10. Visual variation:
   - one whole-image named node color is sampled from the shared Trace named-color palette,
   - one whole-image node glyph style is sampled from `circle|rounded_square|hexagon`,
   - one whole-image label format is sampled from `letters|numbers|named`,
   - one whole-image edge routing style is sampled from `straight|mixed_arc`,
   - one whole-image layout transform is sampled from `identity|rotate_90|rotate_180|rotate_270|mirror_left_right|mirror_up_down`,
   - these style axes are non-semantic for this task and are recorded in trace metadata.

## 3) Prompt contract
1. Bundle: `graph_node_link_counting_v1`
2. `scene_key`: `single_graph_counting`
3. `task_key`: `bridge_count_query`
4. Required slots:
   - scene: `object_description`
   - task: `question_text`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Answer JSON shape: `{"answer":2}`
7. Answer+annotation JSON shape: `{"annotation":[[[180,220],[310,180]],[[310,180],[430,260]]],"answer":2}`
8. Prompt-facing annotation uses a `segment_set` because each witness is a graph edge. Each segment is `[[x0, y0], [x1, y1]]`, where each endpoint is an `[x, y]` pixel point at an endpoint node center.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the `segment_set` of endpoint-node-center segments for all bridge edges.
2. Each bridge witness is represented as an undirected segment `[[x0, y0], [x1, y1]]`, where each endpoint is an `[x, y]` pixel point; endpoint order is unordered semantically for reward matching, and the implementation keeps the corresponding endpoint labels in `witness_symbolic`.
3. The list of bridge-edge pairs is unordered semantically as well; the implementation only canonicalizes the symbolic outer order internally for deterministic serialization.
4. `answer_gt.value == len(annotation_gt.value)` by construction.
5. `scene_ir.entities` stores:
   - one node entity per rendered node with visible label, degree, adjacency, center, and bbox,
   - one edge entity per rendered edge with endpoint labels, pixel segment, and bridge flag.
6. `scene_ir.relations` stores:
   - the counting rule `edge_is_bridge`,
   - the undirected adjacency map,
   - the full edge-label list,
   - the matching bridge-edge label pairs.
7. `projected_annotation` includes:
   - `segment_set`
   - `segment_map`
8. `execution_trace` records:
   - `query_id` (the concrete public query branch)
   - `scene_variant`
   - `target_count`
   - feasible support distributions for node count / target count
   - `topology_profile`
   - requested and realized layout variants
   - bridge-edge topology and adjacency map

## 5) Visual policy
1. Background and post-image noise use the merged graph-domain visual defaults from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. Current graph scenes use a single rounded light panel on a light solid background.
3. Node labels stay readable across layout, label-format, shape, and color variants.
4. Layout and styling may vary for readability and diversity, but the prompt never refers to node position, node color, or node shape as the semantic source of truth.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized adjacency map and bridge computation.
3. Unique-answer policy: the sampler targets one explicit bridge count and rejects any graph whose realized bridge-edge set does not match it exactly.
4. Reject/resample conditions:
   - no feasible node-count support for the requested bridge count,
   - failure to realize a simple connected graph with the requested bridge support,
   - unreadable layout that cannot keep nodes sufficiently separated.
5. No semantic auto-relaxation: failures do not weaken the graph, label, or bridge contract.

## 7) Complexity + tests
1. Complexity definition/components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Determinism/build tests: `tests/test_graph_counting_bridge_count_contracts.py`
3. Behavior/trace/prompt tests: `tests/test_graph_counting_bridge_count_tasks.py`
4. Prompt bundle/config tests: `tests/test_prompt_system.py`, `tests/test_scene_config.py`
