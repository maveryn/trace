# `task_symbolic__turing_tape__final_head_position_value`

## Summary
1. Domain: `symbolic`
2. Scene: `turing_tape`
3. Task id: `task_symbolic__turing_tape__final_head_position_value`
4. Goal: simulate a compact tape-machine transition table for a fixed number of steps and report the final numbered tape cell under the head.

## Program Contract
Program: `turing_tape.final_head_position_value(scene=turing_tape, scope=single_tape_machine_with_transition_table, output=integer_cell_index)`

Candidate set: the visible numbered tape cells after simulating the machine.
Operands: the starting tape contents, initial head position and state, visible transition table, and requested step count.
Operation: simulate the transition table for the requested number of steps and read the final 1-based visible tape cell number under the head.
Output binding: `answer` is the final cell number as an integer.
Annotation witnesses: a `bbox_map` with `machine_panel` and `transition_table` roles.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Query Contract
1. Public `query_id`: `single`
2. Internal prompt/event key: `final_head_position_value`
3. Answer type: `integer`
4. Annotation type: `bbox_map`
5. Annotation schema: `bbox_map`
6. Annotation target: keyed visual witnesses:
   - `machine_panel`: starting tape, head marker, state label, step count, and symbol alphabet panel
   - `transition_table`: visible transition-table panel
7. Scene variants: `clean_grid|lab_panel|notebook_grid`
8. Generation controls: tape length `8..11`, step count `3..6`, symbol alphabet size `2`
9. Answer convention: 1-based visible tape cell number
10. Prompt bundle: `symbolic_turing_tape_v1`
11. Source file: `src/trace_tasks/tasks/symbolic/turing_tape/final_head_position_value.py`
