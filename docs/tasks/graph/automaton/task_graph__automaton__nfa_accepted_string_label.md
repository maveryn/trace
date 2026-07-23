# `task_graph__automaton__nfa_accepted_string_label`

## Summary
1. Domain: `graph`
2. Scene id: `automaton`
3. Source package: `automaton`
4. Task id: `task_graph__automaton__nfa_accepted_string_label`
5. Objective: choose the candidate input string accepted by a nondeterministic finite automaton.

## Query IDs
1. `nfa_accepted_string_label`
2. Query ids are internal replay metadata; public sampling is at the task-id level.

## Program Contract

Program: `select(candidate_label for candidate_string in options if accepts(nfa_transition_graph, candidate_string)); output=string; annotation=point_sequence(accepted_state_path); scene=automaton; scope=nfa_accepted_string_label`

Candidate set: the visible graph, tree, network, route, matrix, table, node, edge, label, weight, path, and option elements inside the `nfa_accepted_string_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `candidate_label`, `candidate_string`, `if`, `accepts`, `nfa_transition_graph`, `accepted_state_path`, `automaton`, `nfa_accepted_string_label`.
Operation: evaluate `select` over the candidate set using the visible graph structure, labels, weights, directions, reachability, paths, connectivity, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_sequence` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `state_update`, `matching`

## Answer And Annotation
1. Answer type: `string`.
2. Annotation type: `point_sequence`.
3. Annotation marks minimal pixel-space visual witnesses for the answer, not answer labels or non-witness annotations.
4. Count tasks require `answer_gt.value == len(annotation_gt.value)` unless the annotation schema is keyed or sequence based.

## Rendering Contract
1. The scene uses the graph-domain renderer for `automaton`.
2. Visual style, fonts, panel treatment, layout jitter, and post-render noise are non-semantic and must be recorded in trace metadata.
3. Annotation projection is computed after final layout and style placement.

## Prompt Contract
1. Prompt text comes from `src/trace_tasks/resources/prompts/graph/automaton/automaton_v1.json` and `src/trace_tasks/resources/configs/domains/graph/automaton.yaml`, not hardcoded user-facing text.
2. Answer mode emits `{"answer": ...}`.
3. Answer-and-annotation mode emits `{"annotation": ..., "answer": ...}` with annotation matching the schema above.

## Files
1. Implementation: `src/trace_tasks/tasks/graph/automaton/nfa_accepted_string_label.py`
2. Shared scene logic: `src/trace_tasks/tasks/graph/automaton/shared/sampling.py`
3. Config: `src/trace_tasks/resources/configs/domains/graph/automaton.yaml`
4. Prompts: `src/trace_tasks/resources/prompts/graph/automaton/automaton_v1.json`
