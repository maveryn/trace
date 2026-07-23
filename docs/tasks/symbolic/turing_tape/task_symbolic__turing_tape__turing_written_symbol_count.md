# `task_symbolic__turing_tape__turing_written_symbol_count`

## Summary
1. Domain: `symbolic`
2. Scene: `turing_tape`
3. Task id: `task_symbolic__turing_tape__turing_written_symbol_count`
4. Goal: simulate a compact tape-machine transition table for a fixed number of steps and count a queried tape symbol afterward.

## Program Contract
Program: `turing_tape.written_symbol_count(scene=turing_tape, scope=single_tape_machine_with_transition_table, output=integer_count)`

Candidate set: the visible tape cells after simulating the machine.
Operands: the starting tape contents, initial head position and state, visible transition table, requested step count, and queried symbol.
Operation: simulate the transition table for the requested number of steps and count visible tape cells containing the queried symbol afterward.
Output binding: `answer` is the matching cell count as an integer.
Annotation witnesses: a `bbox_map` with `machine_panel` and `transition_table` roles.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Query Contract
1. Public `query_id`: `single`
2. Internal prompt/event key: `written_symbol_count`
3. Answer type: `integer`
4. Annotation type: `bbox_map`
5. Annotation schema: `bbox_map`
6. Annotation target: keyed visual witnesses:
   - `machine_panel`: starting tape, head marker, state label, step count, and queried symbol panel
   - `transition_table`: visible transition-table panel
7. Scene variants: `clean_grid|lab_panel|notebook_grid`
8. Generation controls: tape length `8..11`, step count `3..6`, symbol alphabet size `2`, answer range `1..7`
9. Prompt bundle: `symbolic_turing_tape_v1`
10. Source file: `src/trace_tasks/tasks/symbolic/turing_tape/turing_written_symbol_count.py`
