# `task_graph__node_link__largest_chordless_cycle_size`

## Program Contract

Program: `length(argmax(chordless_cycles(graph), metric=cycle_size)); scene=node_link; scope=largest_chordless_cycle_size`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `largest_chordless_cycle_size` objective scope.
Operands: visible scene state and prompt-bound operands named by `argmax`, `chordless_cycles`, `graph`, `metric`, `cycle_size`, `node_link`, `largest_chordless_cycle_size`.
Operation: evaluate `length` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `topology`

## Summary
1. Domain: `graph`
2. Scene id: `node_link`
3. Scene: `relation`
4. Task id: `task_graph__node_link__largest_chordless_cycle_size`
5. Objective: return the number of nodes in the largest chordless cycle of one undirected graph.

## Query IDs
1. `single`
2. Query ids are internal replay metadata; public sampling is at the task-id level.

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation type: `point_set`.
3. Annotation is an unordered set of node-center pixel points for all nodes in one largest chordless cycle.
4. `answer_gt.value == len(annotation_gt.value)` by construction.

## Rendering Contract
1. The scene uses the shared graph-domain `node_link` renderer.
2. The graph is simple, undirected, connected, and may contain multiple cycles.
3. Node labels, node colors, node shapes, edge routing, layout, transforms, backgrounds, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final layout and style placement.

## Generation Contract
1. Target largest chordless cycle size is sampled from `3..7`.
2. Node count is sampled from `8..10`.
3. The sampler constructs a target chordless cycle, adds a second small cycle so the graph is not unicyclic, attaches remaining nodes, and keeps extra distractor edges only when the finalized graph's largest chordless cycle size remains the target.
4. Final verification uses the finalized adjacency map; answers and annotation are not inferred from the construction recipe alone.

## Prompt Contract
1. Prompt text comes from `graph_node_link_relation_v1`, not hardcoded task text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with point-set annotation.

## Tests
1. Behavior tests: `tests/test_graph_relation_largest_chordless_cycle_size_tasks.py`
2. Contract tests: `tests/test_graph_relation_largest_chordless_cycle_size_contracts.py`
