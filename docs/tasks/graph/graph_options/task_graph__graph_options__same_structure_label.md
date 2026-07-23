# `task_graph__graph_options__same_structure_label`

## Summary
1. Domain: `graph`
2. Scene id: `graph_options`
3. Task id: `task_graph__graph_options__same_structure_label`
4. Objective: select the option graph with the same labeled structure as the Reference graph.
5. Implementation: `src/trace_tasks/tasks/graph/graph_options/same_structure_label.py`.

## Query IDs
1. Supported `query_id` values: `single`.
2. Internal prompt key: `same_structure_label`.
3. Public sampling is at the task-id level.

## Taxonomy Contract
1. Program contract: compare the Reference labeled graph with four visual option graphs, ignoring option layout, and choose the option with the same node labels and edge connections.
2. Answer schema: `option_letter`.
3. Annotation schema: `bbox`.
4. Edge direction, node count, correct option slot, style, font, background, and layout jitter are generation/render metadata, not public query branches.

## Program Contract

Program: `select(option_graph where labeled_structure(option_graph) == labeled_structure(reference_graph)); output=option_letter; annotation=bbox(selected_option_panel); scene=graph_options; scope=same_structure_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `same_structure_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_graph`, `where`, `labeled_structure`, `reference_graph`, `selected_option_panel`, `graph_options`, `same_structure_label`.
Operation: evaluate `select` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `matching`

## Answer And Annotation
1. Answer type: `option_letter`.
2. Annotation type: `bbox`.
3. Annotation marks the selected visual option panel as one `[x0,y0,x1,y1]` pixel-space box.
4. The selected option panel and answer letter come from the same execution trace.

## Rendering Contract
1. The scene shows one Reference graph above four visual graph-option panels.
2. Directed samples use arrows and require exact direction matching.
3. Visual style, fonts, panel treatment, layout jitter, and post-render noise are non-semantic and recorded in trace metadata.
4. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/graph_options/graph_options_v1.json`.
2. `scene_key`: `graph_options`.
3. `task_key`: `structure_match_label_query`.
4. `query_key`: `same_structure_label`.
5. Answer mode emits `{"answer": ...}`.
6. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with `annotation` matching the scalar `bbox` schema.
