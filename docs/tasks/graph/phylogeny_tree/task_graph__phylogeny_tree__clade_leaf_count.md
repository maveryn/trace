# `task_graph__phylogeny_tree__clade_leaf_count`

## Summary
1. Domain: `graph`
2. Scene id: `phylogeny_tree`
3. Source package: `phylogeny_tree`
4. Task id: `task_graph__phylogeny_tree__clade_leaf_count`
5. Objective: count terminal taxa descending from the marked clade.

## Program Contract

Program: `count(leaves(descendant_of(marked_clade))); output=integer; annotation=point_set(descendant_leaf_terminal_centers); scene=phylogeny_tree; scope=clade_leaf_count`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `clade_leaf_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `leaves`, `descendant_of`, `marked_clade`, `descendant_leaf_terminal_centers`, `phylogeny_tree`, `clade_leaf_count`.
Operation: evaluate `count` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`

## Query IDs
1. Supported `query_id`: `single`.
2. The prompt objective key is `marked_clade_leaf_count`; it is trace metadata, not a public query branch.

## Answer And Annotation
1. Answer type: `integer`.
2. Annotation schema: `point_set`.
3. Annotation marks terminal centers for every descendant taxon in the marked clade.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)`.

## Rendering Contract
1. The scene renders a rooted cladogram with leaf labels and one marked clade.
2. Branch routing, child order, fonts, panel treatment, color, background, and post-render noise are nonsemantic and recorded in trace metadata.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/phylogeny_tree/graph_phylogeny_tree_v1.json`.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
