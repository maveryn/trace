# `task_graph__phylogeny_tree__topology_outlier_label`

## Summary
1. Domain: `graph`
2. Scene id: `phylogeny_tree`
3. Source package: `phylogeny_tree`
4. Task id: `task_graph__phylogeny_tree__topology_outlier_label`
5. Objective: choose the only option cladogram with a different rooted topology.

## Program Contract

Program: `select(option_label where rooted_topology(option) != rooted_topology(other_options)); output=string; annotation=bbox(selected_option_panel); scene=phylogeny_tree; scope=topology_outlier_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `topology_outlier_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_label`, `where`, `rooted_topology`, `other_options`, `selected_option_panel`, `phylogeny_tree`, `topology_outlier_label`.
Operation: evaluate `select` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `matching`

## Query IDs
1. Supported `query_id`: `single`.
2. The prompt objective key is `topology_outlier_label`; it is trace metadata, not a public query branch.

## Answer And Annotation
1. Answer type: `string` option letter.
2. Annotation schema: `bbox`.
3. Annotation is the option-panel bbox for the selected visual option.
4. Option bboxes are allowed because this is a true visual option-image task.

## Rendering Contract
1. The scene renders four labeled rooted phylogeny option panels.
2. Options `A` through `D` use fixed panel positions.
3. Three options are topology-equivalent rotations/layout variants; one option has a different rooted clade signature over the same taxon labels.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/phylogeny_tree/graph_phylogeny_tree_v1.json`.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.
