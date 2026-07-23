# `task_symbolic__logic_gate_circuit__internal_output_count`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `logic_gate_circuit`
3. Task id: `task_symbolic__logic_gate_circuit__internal_output_count`
4. Objective contract: `internal_output_count`

## Program Contract
Program: `logic_gate_circuit.internal_output_count(scene=logic_gate_circuit, scope=single_circuit_with_input_values, target_output=0|1, output=integer)`

Candidate set: all visible gate symbols in the single rendered circuit.
Operands: the visible input values, gate topology, each gate's operator, and the requested local output value.
Operation: evaluate every gate's local output and count gates whose output equals the requested value.
Output binding: `answer` is the matching gate count as an integer.
Annotation witnesses: a homogeneous `bbox_set` of gate-symbol bboxes whose computed output matches the requested value.
Query ids: `internal_output_one_count`, `internal_output_zero_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `topology`, `formula_evaluation`

## 2) Scene + task contract
1. Entities/relations: one fanout-free expression-tree circuit with visible input values, standard gate symbols, wires, and a final `OUT` node.
2. Supported `query_id` values: `internal_output_one_count`, `internal_output_zero_count`
3. `answer_gt.type`: `integer`
4. Default `annotation_gt.type`: `bbox_set`
5. Annotation schema: `bbox_set`
6. Annotation witness policy: annotation marks the gate-symbol bounding boxes whose computed output equals the requested value; nonmatching gates, input labels, wires, output node, panel borders, and decorative background are not prompt-facing annotation.
7. Overlap/touch policy: gate-symbol bounding boxes are laid out by the shared expression-tree renderer and should not overlap.

| Query id | User-facing operation |
|---|---|
| `internal_output_one_count` | Count gate symbols whose computed local output is `1`. |
| `internal_output_zero_count` | Count gate symbols whose computed local output is `0`. |

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_logic_gate_circuit_v1`
2. `scene_key`: `logic_gate_circuit`
3. `task_key`: `logic_gate_internal_output_count`
4. Query keys: `internal_output_one_count`, `internal_output_zero_count`
5. Required slots:
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. JSON examples use an integer answer and a `bbox_set` annotation array.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: `<task_id>.dataset` and `<task_id>:scene_variant`.
2. Unique-answer policy: the sampler chooses a target answer and resamples four-gate circuits until the per-gate evaluation trace has exactly that many gates with the requested output.
3. Reject/resample conditions: invalid query id, target answer outside `0..gate_count`, or incompatible gate-count settings.
4. No-auto-relaxation guarantee: semantic constraints are validated and never relaxed after sampling.

## 5) Tests
1. Determinism test: covered by source-layout generation tests.
2. Answer/annotation consistency test: `tests/test_symbolic_logic_gate_tasks.py`
3. Prompt metadata/placeholder test: `tests/test_prompt_system.py`
4. Constraint-specific tests: focused logic-gate task tests assert per-gate output counts and `bbox_set` annotation cardinality.
