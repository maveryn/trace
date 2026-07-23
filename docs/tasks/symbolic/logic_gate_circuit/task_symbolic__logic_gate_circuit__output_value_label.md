# `task_symbolic__logic_gate_circuit__output_value_label`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `logic_gate_circuit`
3. Source scene: `logic_gate_circuit`
4. Task id: `task_symbolic__logic_gate_circuit__output_value_label`

## Program Contract
Program: `logic_gate_circuit.output_value_label(scene=logic_gate_circuit, scope=four_independent_circuit_options, target_output=0|1, output=option_letter)`

Candidate set: the four independent Boolean circuit-option panels labeled `A..D`.
Operands: the visible input values, gate topology, and gate operators in each option circuit, plus the requested final `OUT` value.
Operation: evaluate each option circuit and select the unique circuit whose final `OUT` node equals the requested value.
Output binding: `answer` is the selected option letter.
Annotation witnesses: the scalar bbox of the selected circuit-option panel.
Query ids: `output_one_label`, `output_zero_label`.

## Reasoning Operations

Families: `logical_composition`, `topology`, `formula_evaluation`

## Query Contract
1. Query metadata: `query_id`
2. Supported `query_id` values:
   - `output_one_label`
   - `output_zero_label`
3. Prompts ask which shown circuit has final `OUT` value `1` or `0`.
4. Each instance shows exactly four independent circuit options labeled `A..D`.
5. Each option panel uses exactly two visible input values.

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the single correct circuit-option label.
3. Annotation schema: `bbox`
4. `annotation_gt.type = bbox`
5. Annotation is the bbox of the selected circuit-option panel.
6. Gate bboxes, input labels, wire segments, and output nodes are not prompt-facing annotation.

## Trace Contract
1. `execution_trace.circuits` records visible inputs, gates, and computed output value for each circuit.
2. `execution_trace.target_output_value` records the queried output value.
3. `execution_trace.annotation_item_id` records the selected circuit id used for bbox projection.
4. `render_map.item_bboxes_px` exposes option-panel, gate, input, and output-node boxes after final layout.
5. `render_map.output_points_px` exposes final output-node centers for audit tooling.

## Prompt Contract
1. Bundle: `symbolic_logic_gate_circuit_v1`
2. Scene key: `logic_gate_circuit`
3. Task key: `logic_gate_output_value_label`
4. Query keys: `output_one_label` or `output_zero_label`
5. Prompt wording must ask from visible gate symbols, wires, and input values. It must not reveal hidden computed output values.

## Source Files
1. Task source: `src/trace_tasks/tasks/symbolic/logic_gate_circuit/output_value_label.py`
2. Scene shared package: `src/trace_tasks/tasks/symbolic/logic_gate_circuit/shared/`
3. Config: `src/trace_tasks/resources/configs/domains/symbolic/logic_gate_circuit.yaml`
4. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/logic_gate_circuit/symbolic_logic_gate_circuit_v1.json`

## Determinism + Constraints
1. Deterministic generation and rendering from `instance_seed`.
2. Unique-answer policy: the sampler constructs exactly one circuit with the queried output value.
3. Supported visible gates are `AND`, `OR`, `NOT`, `XOR`, `NAND`, and `NOR`.
4. Every visible input and every visible gate contributes to the final output.
5. The circuit sampler does not introduce fanout or wire crossings as semantic requirements.
