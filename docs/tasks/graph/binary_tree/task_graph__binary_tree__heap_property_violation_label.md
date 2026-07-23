# `task_graph__binary_tree__heap_property_violation_label`

## Program Contract

Program: `label(child_node(heap_property_violation(binary_tree))); scene=binary_tree; scope=heap_property_violation_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `heap_property_violation_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `child_node`, `heap_property_violation`, `binary_tree`, `heap_property_violation_label`.
Operation: evaluate `label` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `matching`

## Summary
1. Domain: `graph`
2. Scene: `binary_tree`
3. Scene: `relation`
4. Task id: `task_graph__binary_tree__heap_property_violation_label`
5. Objective: find the child node that violates a min-heap property.

## Query IDs
1. `single`: find the child node whose key is smaller than its parent key.

## Annotation
1. Answer type: `string`.
2. Annotation schema: `point_map`.
3. Annotation uses keys `parent` and `child`, with each value a node-center `[x,y]` pixel point.

## Generation Notes
1. Instances render a complete numeric-key binary tree with exactly one min-heap violation.
2. Default node count is `7..10` for this task.
3. The violating child is sampled with a larger numeric gap below its parent so the unique heap violation is visually easier to compare.
4. The renderer is the shared top-down `binary_tree` scene.
5. Title and node labels use the role-appropriate shared font pool and readable text styles with recorded contrast metadata.
6. Binary-tree rendering includes sampled tree treatments, node shapes/colors, light optional non-answer context text outside the tree content, bounded content jitter before projection, and scene-derived connector styles.
