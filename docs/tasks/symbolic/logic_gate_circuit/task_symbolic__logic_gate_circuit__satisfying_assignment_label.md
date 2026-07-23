# `task_symbolic__logic_gate_circuit__satisfying_assignment_label`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `logic_gate_circuit`
3. Source scene: `logic_gate_circuit`
4. Task id: `task_symbolic__logic_gate_circuit__satisfying_assignment_label`

## Program Contract
Program: `logic_gate_circuit.satisfying_assignment_label(scene=logic_gate_circuit, scope=source_circuit_plus_assignment_options, target_output=0|1, output=option_letter)`

Candidate set: the four visible assignment rows labeled `A..D`.
Operands: the source Boolean circuit, each assignment row's input values, and the requested final `OUT` value.
Operation: evaluate the source circuit for every assignment row and select the unique row whose final `OUT` value equals the requested value.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `source_circuit` and `selected_option` roles.
Query ids: `assignment_outputs_one_label`, `assignment_outputs_zero_label`.

## Reasoning Operations

Families: `logical_composition`, `topology`, `formula_evaluation`, `matching`

## Query Contract
1. Query metadata: `query_id`
2. Supported `query_id` values:
   - `assignment_outputs_one_label`
   - `assignment_outputs_zero_label`
3. Prompts ask which visible assignment option makes the source circuit's final `OUT` node evaluate to `1` or `0`.
4. Each instance shows one source circuit and exactly four assignment-option rows labeled `A..D`.
5. Inputs are named `x`, `y`, and `z`.

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the single correct assignment-option label.
3. Annotation schema: `bbox_map`
4. `annotation_gt.type = bbox_map`
5. Annotation uses keys `source_circuit` and `selected_option`.
6. Distractor assignment rows, gate bboxes, and output-node points are not prompt-facing annotation.

## Trace Contract
1. `execution_trace.source_circuit` records the visible circuit grammar.
2. `execution_trace.candidates` records each option label, assignment values, computed output value, and correctness flag.
3. `execution_trace.logic_gate_metadata.correct_assignment` records the unique satisfying assignment.
4. `execution_trace.target_output_value` records the queried output value.
5. `render_map.item_bboxes_px` exposes the source-circuit bbox and candidate-row bboxes after final layout.

## Prompt Contract
1. Bundle: `symbolic_logic_gate_circuit_v1`
2. Scene key: `logic_gate_circuit`
3. Task key: `logic_gate_satisfying_assignment_label`
4. Query keys: `assignment_outputs_one_label` or `assignment_outputs_zero_label`
5. Prompt wording must refer to the visible source circuit and visual assignment rows, not prompt-only answer choices.

## Source Files
1. Task source: `src/trace_tasks/tasks/symbolic/logic_gate_circuit/satisfying_assignment_label.py`
2. Scene shared package: `src/trace_tasks/tasks/symbolic/logic_gate_circuit/shared/`
3. Config: `src/trace_tasks/resources/configs/domains/symbolic/logic_gate_circuit.yaml`
4. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/logic_gate_circuit/symbolic_logic_gate_circuit_v1.json`

## Determinism + Constraints
1. Deterministic generation and rendering from `instance_seed`.
2. Unique-answer policy: the source circuit is constructed so exactly one shown assignment row has the queried output value.
3. Supported visible gates are `AND`, `OR`, `NOT`, `XOR`, `NAND`, and `NOR`.
4. The renderer does not rely on wire fanout or crossover conventions.
