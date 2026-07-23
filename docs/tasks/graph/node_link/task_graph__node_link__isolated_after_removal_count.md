# `task_graph__node_link__isolated_after_removal_count`

## Program Contract

Program: `count(filter(nodes(remove_node(graph, reference_node)), degree=0)); scene=node_link; scope=isolated_after_removal_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `isolated_after_removal_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `nodes`, `remove_node`, `graph`, `reference_node`, `degree`, `node_link`, `isolated_after_removal_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the count of remaining isolated nodes.
Annotation witnesses: `annotation` uses the `point_set` schema; the unordered `point_set` of pixel centers for all remaining nodes that would have total degree zero after the queried node is removed.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`, `state_update`

## 1) Identity
1. Domain: `graph`
2. Scene: `counting`
3. Scene id: `node_link`
4. Task id: `task_graph__node_link__isolated_after_removal_count`
5. Objective: count how many remaining nodes would be isolated after one named node is removed.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `single`
3. Supported `graph_directionality` values: `undirected|directed`
4. `answer_gt.type`: `integer`
5. `annotation_gt.type`: `point_set`
6. Scene contract:
   - one single-panel labeled node-link graph,
   - simple unweighted graph only,
   - no self-loops or multi-edges,
   - directed variants reject reciprocal edge pairs,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`),
   - node count is sampled from `5..10`.
7. Query contract:
   - prompt names one node to hypothetically remove,
   - removal also removes every incident edge,
   - in directed graphs, an isolated remaining node has zero incoming and zero outgoing edges,
   - answer support is `0..5`, including zero-answer cases,
   - answer is the count of remaining isolated nodes.

## 3) Prompt contract
1. Bundle: `graph_node_link_counting_v1`
2. `scene_key`: `single_graph_counting`
3. `task_key`: `isolated_node_count_after_node_removal_query`
4. Modes: `answer_only`, `answer_and_annotation`
5. Answer JSON shape: `{"answer":2}`
6. Answer+annotation JSON shape: `{"annotation":[[180,220],[310,180]],"answer":2}`
7. Prompt references to `named` labels are quoted, for example node `"Lima"`.
8. Prompt-facing annotation uses pixel-space node-center points for every remaining node that would be isolated.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the unordered `point_set` of pixel centers for all remaining nodes that would have total degree zero after the queried node is removed.
2. `answer_gt.value == len(annotation_gt.value)` by construction, including zero-answer cases where annotation is an empty array.
3. `execution_trace.removed_node_label` records the hypothetically removed node.
4. `execution_trace.matching_labels` records the symbolic witness labels in deterministic label order.
5. `execution_trace.pre_removal_*_by_label` and `execution_trace.post_removal_*_by_label` record adjacency, successor, predecessor, degree, in-degree, and out-degree metadata.
6. `scene_ir.entities` stores node labels, pre-removal degrees, post-removal degrees, center points, bboxes, `is_removed_query_node`, and `is_post_removal_isolated`.
7. `projected_annotation` includes `point_set`, `pixel_point_set`, and `pixel_bbox_set`.

## 5) Visual policy
1. Rendering uses the shared graph light-panel style from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. The graph is rendered in its pre-removal state; the prompt asks the model to reason about the hypothetical post-removal state.
3. Node label format, edge routing, glyph style, layout transform, node color, and layout remain visual variation only.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same rendered graph and the same post-removal execution trace.
3. Generation constructs exactly the requested number of post-removal isolated nodes.
4. No semantic auto-relaxation: failures do not weaken the graph, directionality, or count contract.

## 7) Complexity + tests
1. Complexity components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Tests: `tests/test_graph_counting_isolated_node_count_after_node_removal_tasks.py`, `tests/test_prompt_system.py`, `tests/test_scene_config.py`
