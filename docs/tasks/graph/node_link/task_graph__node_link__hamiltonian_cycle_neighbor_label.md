# `task_graph__node_link__hamiltonian_cycle_neighbor_label`

## Program Contract

Program: `label(adjacent_node_on_cycle(unique_hamiltonian_cycle(graph), reference_node, direction=next_or_previous)); scene=node_link; scope=hamiltonian_cycle_neighbor_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `hamiltonian_cycle_neighbor_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `adjacent_node_on_cycle`, `unique_hamiltonian_cycle`, `graph`, `reference_node`, `direction`, `next_or_previous`, `node_link`, `hamiltonian_cycle_neighbor_label`.
Operation: evaluate `label` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Summary
1. Domain: `graph`
2. Scene id: `node_link`
3. Scene: `relation`
4. Task id: `task_graph__node_link__hamiltonian_cycle_neighbor_label`
5. Objective: return the node label immediately before or after a queried node along the graph's unique Hamiltonian cycle.

## Query IDs
1. `next_in_hamiltonian_cycle_label`
2. `previous_in_hamiltonian_cycle_label`
3. Query ids are internal replay metadata; public sampling is at the task-id level.

## Answer And Annotation
1. Answer type: `string`.
2. Annotation type: `point`.
3. Annotation is a single node-center pixel point for the answer node.
4. The answer is the visible label of the requested previous or next node in that traversal.

## Rendering Contract
1. The scene uses the shared graph-domain `node_link` renderer.
2. The graph is simple, undirected, connected, and has exactly one Hamiltonian cycle.
3. Node labels, node colors, node shapes, edge routing, layout, transforms, backgrounds, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final layout and style placement.

## Generation Contract
1. Node count is sampled from `4..6`.
2. The sampler starts from a cycle through all nodes and keeps `0..2` extra distractor edges only when exhaustive validation confirms the finalized graph still has exactly one Hamiltonian cycle.
3. The prompt fixes traversal direction by naming the start node and the final node before the cycle returns to that start.
4. The `next` query asks only for the node after the start node; the `previous` query asks only for the node before the final node.
5. Final verification uses the finalized adjacency map; answers and annotation are not inferred from the construction recipe alone.

## Prompt Contract
1. Prompt text comes from `graph_node_link_relation_v1`, not hardcoded task text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with single-point annotation.

## Tests
1. Behavior tests: `tests/test_graph_relation_hamiltonian_cycle_neighbor_label_tasks.py`
2. Contract tests: `tests/test_graph_relation_hamiltonian_cycle_neighbor_label_contracts.py`
