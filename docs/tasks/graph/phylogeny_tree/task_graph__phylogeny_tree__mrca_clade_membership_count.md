# `task_graph__phylogeny_tree__mrca_clade_membership_count`

## Summary
1. Domain: `graph`
2. Scene id: `phylogeny_tree`
3. Source package: `phylogeny_tree`
4. Task id: `task_graph__phylogeny_tree__mrca_clade_membership_count`
5. Objective: count descendant taxa of the most recent common ancestor of two queried taxa.

## Program Contract

Program: `count(leaves(descendant_of(mrca(query_leaf_1,query_leaf_2)))); output=integer; annotation=point_set(descendant_leaf_terminal_centers); scene=phylogeny_tree; scope=mrca_clade_membership_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `mrca_clade_membership_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `leaves`, `descendant_of`, `mrca`, `query_leaf_1`, `query_leaf_2`, `descendant_leaf_terminal_centers`, `phylogeny_tree`, `mrca_clade_membership_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`

## Query IDs
1. Supported `query_id`: `single`.
2. The prompt objective key is `mrca_leaf_count`; it is trace metadata, not a public query branch.

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation schema: `point_set`.
3. Annotation points are the terminal centers of every descendant taxon counted under the MRCA clade.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)`.
5. Query leaf labels, MRCA node id, and descendant leaf labels are also recorded in trace metadata.

## Rendering Contract
1. The scene renders a rooted cladogram with labeled terminal taxa.
2. The queried MRCA clade is not visually highlighted; the solver must infer it from the two named taxa and the generated tree topology.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/phylogeny_tree/graph_phylogeny_tree_v1.json`.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
