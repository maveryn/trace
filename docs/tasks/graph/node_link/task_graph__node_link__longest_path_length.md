# `task_graph__node_link__longest_path_length`

## Program Contract

Program: `length(longest_path(directed_acyclic_graph)); scene=node_link; scope=longest_path_length`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `longest_path_length` objective scope.
Operands: visible scene state and prompt-bound operands named by `longest_path`, `directed_acyclic_graph`, `node_link`, `longest_path_length`.
Operation: evaluate `length` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the number of directed edges in that unique longest path.
Annotation witnesses: `annotation` uses the `point_sequence` schema; the ordered `point_sequence` of node-center pixel points along the unique longest directed path.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## 1) Identity
1. Domain: `graph`
2. Scene: `path`
3. Scene id: `node_link`
4. Task id: `task_graph__node_link__longest_path_length`
5. Objective: count the number of directed edges in the unique longest path of a directed acyclic graph.

## 2) Scene + task contract
1. Branch metadata: `query_id`
2. `query_id`: `single`
3. Supported `graph_directionality`: `directed`
4. `answer_gt.type`: `integer`
5. `annotation_gt.type`: `point_sequence`
6. Scene contract:
   - one single-panel labeled directed node-link graph,
   - graph is a DAG,
   - simple unweighted graph only,
   - no self-loops, multi-edges, or reciprocal directed edge pairs,
   - visible node labels use one whole-image label format (`letters`, `numbers`, or `named`),
   - node count is sampled from `5..10`.
7. Query contract:
   - the prompt states that the directed graph has a unique longest directed path,
   - answer support is `2..6`,
   - answer is the number of directed edges in that unique longest path.

## 3) Prompt contract
1. Bundle: `graph_node_link_path_v1`
2. `scene_key`: `single_graph_path`
3. `task_key`: `longest_path_length_query`
4. Modes: `answer_only`, `answer_and_annotation`
5. Answer JSON shape: `{"answer":2}`
6. Answer+annotation JSON shape: `{"annotation":[[180,220],[310,180],[430,260]],"answer":2}`
7. Prompt-facing annotation uses an ordered pixel-space node-center sequence from the start of the longest path to its end.

## 4) Annotation + trace contract
1. Prompt-facing annotation is the ordered `point_sequence` of node-center pixel points along the unique longest directed path.
2. `answer_gt.value == len(annotation_gt.value) - 1` by construction.
3. `execution_trace.longest_path_labels` records the symbolic path in order.
4. `execution_trace.source_label` and `execution_trace.goal_label` record the first and last nodes of the longest path.
5. `scene_ir.entities` stores node labels, in-degrees, out-degrees, successors, predecessors, center points, node bboxes, and `is_on_longest_path`.
6. `projected_annotation` includes `point_sequence`, `pixel_point_sequence`, and `pixel_bbox_set`.

## 5) Visual policy
1. Rendering uses the shared graph light-panel style from `src/trace_tasks/resources/configs/domains/graph/base.yaml`.
2. Directed edges render arrowheads.
3. Node label format, edge routing, glyph style, named node color, layout transform, and layout are visual variation only.
4. The prompt never refers to node position, color, or shape as semantic annotation.

## 6) Determinism + constraints
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized successor adjacency map.
3. Generation starts from a directed backbone path of the requested length and only keeps side branches or extra edges when a longest-path DP check confirms that the global longest path is unique and unchanged.
4. No semantic auto-relaxation: failures do not weaken the DAG, label, or unique-longest-path contract.

## 7) Complexity + tests
1. Complexity components: `topology_reasoning`, `visual_scan`, `ambiguity`, `clutter`
2. Tests: `tests/test_graph_path_longest_path_length_contracts.py`, `tests/test_graph_path_longest_path_length_tasks.py`, `tests/test_prompt_system.py`, `tests/test_scene_config.py`
