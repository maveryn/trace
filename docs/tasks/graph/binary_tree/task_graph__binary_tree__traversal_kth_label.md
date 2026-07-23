# `task_graph__binary_tree__traversal_kth_label`

## Program Contract

Program: `label(kth_node(traversal(binary_tree, traversal_order), k)); scene=binary_tree; scope=traversal_kth_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `traversal_kth_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `kth_node`, `traversal`, `binary_tree`, `traversal_order`, `k`, `traversal_kth_label`.
Operation: evaluate `label` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_sequence` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `topology`

## Summary
1. Domain: `graph`
2. Scene: `binary_tree`
3. Scene: `order`
4. Task id: `task_graph__binary_tree__traversal_kth_label`
5. Objective: return the node label at a requested position in a binary-tree traversal.

## Query IDs
1. `preorder_kth_node_label`: root, left subtree, right subtree.
2. `inorder_kth_node_label`: left subtree, root, right subtree.
3. `postorder_kth_node_label`: left subtree, right subtree, root.
4. `level_order_kth_node_label`: top to bottom, left to right within each level.

## Annotation
1. Answer type: `string`.
2. Annotation schema: `point_sequence`.
3. Annotation points are an ordered prefix of node-center `[x,y]` pixel points, from the first visited node through the answer node.

## Generation Notes
1. The renderer is a top-down ordered binary tree; left and right children are determined by visible position.
2. Default node count is `7..10`; default requested traversal position is `2..6`.
3. Node labels use graph label variants `letters|numbers|named`.
4. Title and node labels use the role-appropriate shared font pool and readable text styles with recorded contrast metadata.
5. Binary-tree rendering includes sampled tree treatments, node shapes/colors, light optional non-answer context text outside the tree content, bounded content jitter before projection, and scene-derived connector styles.
