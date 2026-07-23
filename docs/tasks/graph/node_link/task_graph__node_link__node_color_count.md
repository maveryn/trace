# `task_graph__node_link__node_color_count`

## Program Contract

Program: `count(filter(nodes(graph), node_color=target_color)); scene=node_link; scope=node_color_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `node_color_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `nodes`, `graph`, `node_color`, `target_color`, `node_link`, `node_color_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the count of matching nodes.
Annotation witnesses: `annotation` uses the `point_set` schema; the unordered `point_set` of pixel centers for all nodes whose semantic fill color matches the queried color.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## 1) Identity
1. Domain: `graph`
2. Scene: `counting`
3. Scene id: `node_link`
4. Task id: `task_graph__node_link__node_color_count`
5. Objective: count how many labeled nodes use one queried semantic node color.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `single`
3. Supported `graph_directionality` values: `undirected|directed`
4. Supported target colors: shared Trace named-color palette (`red`, `blue`, `green`, `yellow`, `orange`, `purple`, `brown`, `cyan`, `magenta`, `maroon`)
5. `answer_gt.type`: `integer`
6. `annotation_gt.type`: `point_set`
7. Scene contract:
   - one single-panel labeled node-link graph,
   - simple unweighted graph only,
   - no self-loops or multi-edges,
   - directed variants reject reciprocal edge pairs,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`),
   - node count is sampled from `8..12`.
8. Query contract:
   - prompt asks for the count of nodes colored with one shared named color plus its hex code,
   - answer support is `3..7`,
   - answer is the count of matching nodes.

## 3) Prompt contract
1. Bundle: `graph_node_link_counting_v1`
2. `scene_key`: `single_graph_counting`
3. `task_key`: `node_color_count_query`
4. Modes: `answer_only`, `answer_and_annotation`
5. Answer JSON shape: `{"answer":3}`
6. Answer+annotation JSON shape: `{"annotation":[[180,220],[310,180],[430,260]],"answer":3}`
7. Prompt-facing color text uses `<color_name> [#RRGGBB]`, for example `green [#37B94B]`.
8. Prompt-facing annotation uses pixel-space node-center points for every matching node.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the unordered `point_set` of pixel centers for all nodes whose semantic fill color matches the queried color.
2. `answer_gt.value == len(annotation_gt.value)` by construction.
3. `execution_trace.target_color_name` records the queried color name.
4. `execution_trace.node_color_names_by_label` records every rendered node's semantic color.
5. `execution_trace.matching_labels` records the symbolic witness labels in deterministic label order.
6. `scene_ir.entities` stores node labels, node colors, degrees, in-degrees, out-degrees, neighbors/successors/predecessors, center points, node bboxes, and `is_target_color_node`.
7. `projected_annotation` includes `point_set`, `pixel_point_set`, and `pixel_bbox_set`.

## 5) Visual policy
1. Rendering uses the shared graph light-panel style from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. This task promotes per-node fill color from visual style into explicit task semantics.
3. The non-semantic theme color still controls panel and edge styling; `render_spec.style.semantic_node_color_names_by_label` records the actual node colors used for the query.
4. Node label format, edge routing, glyph style, layout transform, and layout remain visual variation only.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same rendered node-color assignment.
3. Generation assigns exactly the requested number of nodes to the queried color, and all other nodes receive non-target colors.
4. No semantic auto-relaxation: failures do not weaken the graph, directionality, color, or count contract.

## 7) Complexity + tests
1. Complexity components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Tests: `tests/test_graph_counting_node_color_count_tasks.py`, `tests/test_prompt_system.py`, `tests/test_scene_config.py`
