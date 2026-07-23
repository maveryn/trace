# `task_graph__binary_tree__bst_path_operation_label`

## Program Contract

Program: `label(path_terminal(binary_search_tree_operation(tree, operation, key))); scene=binary_tree; scope=bst_path_operation_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `bst_path_operation_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `path_terminal`, `binary_search_tree_operation`, `tree`, `operation`, `key`, `binary_tree`, `bst_path_operation_label`.
Operation: evaluate `label` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_sequence` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `state_update`

## Summary
1. Domain: `graph`
2. Scene: `binary_tree`
3. Scene: `relation`
4. Task id: `task_graph__binary_tree__bst_path_operation_label`
5. Objective: answer label-valued binary-search-tree path operation queries.

## Query IDs
1. `bst_search_terminal_label`: search for a key in a BST and answer the final visited node label.
2. `bst_insert_parent_label`: insert a missing key into a BST and answer the existing parent node label.

## Annotation
1. Answer type: `string`.
2. Annotation schema: `point_sequence`.
3. Annotation points are the ordered node-center search or insertion path from the root through the answer node.

## Generation Notes
1. Instances render numeric keys in a bounded-depth binary search tree.
2. Default node count is `7..13`.
3. The renderer is the shared top-down `binary_tree` scene.
4. Title and node labels use the role-appropriate shared font pool and readable text styles with recorded contrast metadata.
5. Binary-tree rendering includes sampled tree treatments, node shapes/colors, light optional non-answer context text outside the tree content, bounded content jitter before projection, and scene-derived connector styles.
