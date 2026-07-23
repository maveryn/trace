# `task_graph__node_link__degree_extremum_value`

## Program Contract

Program: `value(extreme(metric(nodes(graph), degree_metric), direction=max_or_min)); scene=node_link; scope=degree_extremum_value`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `degree_extremum_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `extreme`, `metric`, `nodes`, `graph`, `degree_metric`, `direction`, `max_or_min`, `node_link`, `degree_extremum_value` plus the active `query_id` branch.
Operation: evaluate `value` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the queried extreme degree value,.
Annotation witnesses: `annotation` uses the `point` schema; one point at the node center whose branch-specific degree value equals the requested maximum or minimum.
Query ids: `undirected_max_degree_value`, `undirected_min_degree_value`, `directed_max_in_degree_value`, `directed_max_out_degree_value`.

## Reasoning Operations

Families: `ranking`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `comparison`
3. Scene id: `node_link`
4. Task id: `task_graph__node_link__degree_extremum_value`
5. Objective: ask for the highest or lowest degree-style value present in one labeled node-link graph.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: one of `undirected_max_degree_value`, `undirected_min_degree_value`, `directed_max_in_degree_value`, or `directed_max_out_degree_value`
3. Supported `graph_directionality` values: `undirected`, `directed`
4. Supported directed `degree_mode` values: `in_degree`, `out_degree`
5. Supported `scene_variant` values: `circular`, `shell`, `spring`, `grid_jitter`, `layered`, `component_clustered`, `path_spine`, `radial_tree`
6. `answer_gt.type`: `integer`
7. `annotation_gt.type`: `point`
8. Scene contract:
   - one single-panel labeled node-link graph,
   - simple unweighted graph only,
   - no self-loops or multi-edges,
   - directed branches also reject reciprocal directed edge pairs,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`),
   - node count is sampled from `5..10`.
9. Query contract:
   - undirected branches ask for the maximum or minimum node degree value,
   - directed branches ask for the maximum in-degree or out-degree value,
   - the default sampler is uniform over the four public `query_id` branches,
   - `graph_directionality`, `degree_mode`, and `extremum_mode` are derived from
     the selected public `query_id`,
   - branch answer supports are `undirected_max=2..5`, `undirected_min=0..3`, and `directed_max_in/out=1..4`,
   - answer is the queried extreme degree value,
   - every generated instance has exactly one node attaining the queried extreme value.

## 3) Prompt contract
1. Bundle: `graph_node_link_comparison_v1`
2. `scene_key`: `single_graph_comparison`
3. `task_key`: `extreme_degree_value_query`
4. Modes: `answer_only`, `answer_and_annotation`
5. Answer JSON shape: `{"answer":3}`
6. Answer+annotation JSON shape: `{"annotation":[[180,220]],"answer":3}`
7. Prompt-facing annotation uses one pixel-space node center for the unique node attaining the active branch's maximum/minimum degree or maximum directed in/out-degree value.
8. If the node label format is `named`, prompt references quote the node label, for example node `"Lima"`.

## 4) Annotation + trace contract
1. Prompt-facing annotation is one point at the node center whose branch-specific degree value equals the requested maximum or minimum.
2. `answer_gt.value == execution_trace.target_degree` by construction.
3. `len(annotation_gt.value) == 1` by construction.
4. `execution_trace.graph_directionality` records `undirected` or `directed`; `execution_trace.degree_mode` records `degree`, `in_degree`, or `out_degree`.
5. `execution_trace.extremum_mode` records `max` or `min`.
6. `execution_trace.query_id_probabilities` records the public branch sampling distribution.
7. `execution_trace.queried_degrees_by_label` stores the degree map used to compute the answer.
8. `execution_trace.matching_labels` and `witness_symbolic.labels` store the label attaining the queried extreme value.
9. `scene_ir.entities` stores node labels, total degrees, directed degrees, queried degree values, center points, bboxes, and `is_extreme_degree_node`.
10. `projected_annotation` includes `point`.

## 5) Visual policy
1. Rendering uses the shared graph light-panel style from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. Directed branches render arrowheads and reject reciprocal edge pairs so the queried degree mode remains readable.
3. Node label format, edge routing, glyph style, named node color, layout transform, and layout are visual variation only.
4. The prompt never refers to node position, color, or shape as semantic annotation.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same adjacency map and rendered node assignment.
3. Generation rejects samples that cannot realize the requested `(query_id, target_degree)` exactly with one unique extremum node.
4. Explicit `query_id` overrides must map to one supported public branch.

## 7) Complexity + tests
1. Complexity components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Tests: `tests/test_graph_comparison_extreme_degree_value_contracts.py`, `tests/test_graph_comparison_extreme_degree_value_tasks.py`, `tests/test_prompt_system.py`, `tests/test_scene_config.py`
