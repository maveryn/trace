# `task_graph__pedigree_chart__relationship_label`

## Summary
1. Domain: `graph`
2. Scene id: `pedigree_chart`
3. Source package: `src/trace_tasks/tasks/graph/pedigree_chart/relationship_label.py`
4. Task id: `task_graph__pedigree_chart__relationship_label`
5. Objective: select the rendered option letter that gives the family relationship of one labeled person to another.

## Program Contract

Program: `select(option_label where option_value == relationship(person_a, person_b)); output=option_letter; annotation=bbox_set(person_symbol_witnesses); scene=pedigree_chart; scope=relationship_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `relationship_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_label`, `where`, `option_value`, `relationship`, `person_a`, `person_b`, `person_symbol_witnesses`, `pedigree_chart`, `relationship_label`.
Operation: evaluate `select` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Answer And Annotation
1. Answer type: `option_letter`.
2. Annotation schema: `bbox_set`.
3. Annotation boxes mark the queried people plus any intermediate/shared person symbols in the family connection.
4. Role-to-person ids are recorded in trace metadata.
5. The selected option is the answer, not annotation.

## Rendering Contract
1. The scene uses pedigree notation: squares are male individuals and circles are female individuals.
2. Generation rows, individual labels, spouse connectors, and descent/sibling connectors are semantic.
3. The queried people may be highlighted for scan support.
4. Six relationship options are rendered inside the image.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/pedigree_chart/graph_pedigree_chart_v1.json`.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with bbox-set annotation.
