# `task_symbolic__agent_automaton__agent_final_pose_label`

## Summary
1. Domain: `symbolic`
2. Scene: `agent_automaton`
3. Task id: `task_symbolic__agent_automaton__agent_final_pose_label`
4. Goal: simulate a turning agent automaton and choose the option showing the final cell and direction.

## Program Contract
Program: `agent_automaton.final_pose(scene=agent_automaton, scope=source_grid_and_pose_options, rule_variant=binary_rule|three_state_rule, output=option_letter)`

Candidate set: the four visible final-pose option cards.
Operands: the source grid cell states, start arrow cell and direction, selected rule variant, and requested step count.
Operation: simulate the turning-agent rule for the requested number of updates and select the unique option with the final cell and direction.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `start_marker` and `selected_option` roles.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Contract
1. Query ids: `single`
2. Rule variant metadata: `binary_rule|three_state_rule`
3. Answer type: `option_letter`
4. Annotation schema: `bbox_map`
5. Annotation target: role-keyed bboxes for `start_marker` and `selected_option`
6. Scene variants: `clean_grid|lab_panel|notebook_grid`
7. Generation controls: board size `3..5` rows/cols, step count `3..6`, pose options `4`
8. Prompt bundle: `symbolic_agent_automaton_v1`
9. Source file: `src/trace_tasks/tasks/symbolic/agent_automaton/agent_final_pose_label.py`

## Annotation Contract
The annotation is a `bbox_map` with two roles:

1. `start_marker`: bbox around the visible start arrow marker.
2. `selected_option`: bbox around the correct option card.

The answer and annotation are bound from the same simulated execution trace.
