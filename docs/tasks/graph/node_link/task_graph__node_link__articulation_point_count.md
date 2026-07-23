# `task_graph__node_link__articulation_point_count`

## Program Contract

Program: `count(filter(nodes(graph), cut_vertex=True)); scene=node_link; scope=articulation_point_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `articulation_point_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `nodes`, `graph`, `cut_vertex`, `True`, `node_link`, `articulation_point_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the number of nodes whose removal increases the number of connected components of the graph.
Annotation witnesses: `annotation` uses the `point_set` schema; the `point_set` of pixel centers for all articulation-point nodes.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `counting`
3. Task id: `task_graph__node_link__articulation_point_count`
4. Objective: count how many labeled graph nodes are articulation points.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `single`
3. Supported `scene_variant` values: `circular`, `shell`, `spring`, `grid_jitter`, `layered`, `component_clustered`, `path_spine`, `radial_tree`
3. `answer_gt.type`: `integer`
4. `annotation_gt.type`: `point_set`
5. Scene contract:
   - one single-panel labeled undirected node-link graph per image,
   - simple unweighted graph only,
   - no self-loops,
   - no multi-edges,
   - node count sampled from `5..10`,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`).
6. Query contract:
   - the prompt asks `How many nodes are articulation points?`,
   - answer is the number of nodes whose removal increases the number of connected components of the graph.
7. Count policy:
   - `target_count` is sampled from `0..5`,
   - node count is chosen from the feasible support that can realize the requested articulation-point count,
   - the sampler verifies the final articulation-point set from the realized adjacency map before emitting answer/annotation.
8. Topology variation:
   - `topology_profile` values are `balanced`, `low_degree`, and `hub_heavy`,
   - topology profile affects how articulation-supporting substructures are assembled,
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
3. `task_key`: `articulation_point_count_query`
4. Required slots:
   - scene: `object_description`
   - task: `question_text`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Answer JSON shape: `{"answer":3}`
7. Answer+annotation JSON shape: `{"annotation":[[180,220],[310,180],[430,260]],"answer":3}`
8. Prompt-facing annotation uses pixel-space node-center points; node labels remain in `witness_symbolic`.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the `point_set` of pixel centers for all articulation-point nodes.
2. The witness set is unordered semantically; the implementation keeps the corresponding labels in `witness_symbolic` and canonicalizes label order internally for deterministic serialization.
3. `answer_gt.value == len(annotation_gt.value)` by construction.
4. `scene_ir.entities` stores one node entity per rendered node with:
   - visible label,
   - articulation-point flag,
   - neighbors,
   - node center,
   - node bbox.
5. `scene_ir.relations` stores one undirected edge relation per graph edge.
6. `projected_annotation` includes:
   - `point_set`
   - `pixel_point_set`
   - `pixel_bbox_set`
7. `execution_trace` records:
   - `query_id` (the concrete public query branch)
   - `scene_variant`
   - `target_count`
   - feasible support distributions for node count / target count
   - `topology_profile`
   - requested and realized layout variants
   - articulation-point labels and adjacency map

## 5) Visual policy
1. Background and post-image noise use the merged graph-domain visual defaults from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. Current graph scenes use a single rounded light panel on a light solid background.
3. Node labels are rendered inside the nodes and stay visually stable across layout, label-format, shape, and color variants.
4. Layout and styling may vary for readability and diversity, but the prompt never refers to node position, node color, or node shape as the semantic source of truth.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized adjacency map and articulation-point computation.
3. Unique-answer policy: the sampler targets one explicit articulation-point count and rejects any graph whose realized articulation-point set does not match it exactly.
4. Reject/resample conditions:
   - no feasible node-count support for the requested articulation-point count,
   - failure to realize a simple graph with the requested articulation-point support,
   - unreadable layout that cannot keep nodes sufficiently separated.
5. No semantic auto-relaxation: failures do not weaken the graph, label, or articulation-point contract.

## 7) Complexity + tests
1. Complexity definition/components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Determinism/build tests: `tests/test_graph_counting_articulation_point_count_contracts.py`
3. Behavior/trace/prompt tests: `tests/test_graph_counting_articulation_point_count_tasks.py`
4. Prompt bundle/config tests: `tests/test_prompt_system.py`, `tests/test_scene_config.py`
