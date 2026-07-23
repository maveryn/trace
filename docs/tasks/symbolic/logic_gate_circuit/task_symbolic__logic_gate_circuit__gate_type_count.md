# `task_symbolic__logic_gate_circuit__gate_type_count`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `logic_gate_circuit`
3. Task id: `task_symbolic__logic_gate_circuit__gate_type_count`
4. Objective contract: `gate_type_count`

## Program Contract
Program: `logic_gate_circuit.gate_type_count(scene=logic_gate_circuit, scope=single_circuit, predicate=target_gate_type, output=integer)`

Candidate set: all visible gate symbols in the single rendered circuit.
Operands: each gate symbol's type and the sampled target gate type.
Operation: count gate symbols whose type equals the target gate type.
Output binding: `answer` is the matching gate count as an integer.
Annotation witnesses: a homogeneous `bbox_set` of matching gate-symbol bboxes.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## 2) Scene + task contract
1. Entities/relations: one fanout-free expression-tree circuit with labeled inputs, standard gate symbols, wires, and a final `OUT` node.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Default `annotation_gt.type`: `bbox_set`
5. Annotation schema: `bbox_set`
6. Annotation witness policy: annotation marks the bounding boxes of gate symbols whose shape matches the requested gate type; nonmatching gates, input labels, wires, output node, panel borders, and decorative background are not prompt-facing annotation.
7. Overlap/touch policy: gate-symbol bounding boxes are laid out by the shared expression-tree renderer and should not overlap.

| Query id | User-facing operation |
|---|---|
| `single` | Count visible gate symbols with the sampled target gate type. |

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_logic_gate_circuit_v1`
2. `scene_key`: `logic_gate_circuit`
3. `task_key`: `logic_gate_gate_type_count`
4. Query key: `gate_type_count`; public single-operation tasks expose `query_id=single`.
5. Required slots:
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
   - query slot: `target_gate_type`
6. JSON examples use an integer answer and a `bbox_set` annotation array.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: `<task_id>.dataset` and `<task_id>:scene_variant`.
2. Unique-answer policy: the sampler constructs exactly the requested count of target gates before rendering.
3. Reject/resample conditions: invalid target gate type, target answer outside `0..gate_count`, or incompatible gate-count settings.
4. No-auto-relaxation guarantee: semantic constraints are validated and never relaxed after sampling.

## 5) Tests
1. Determinism test: covered by source-layout generation tests.
2. Answer/annotation consistency test: `tests/test_symbolic_logic_gate_tasks.py`
3. Prompt metadata/placeholder test: `tests/test_prompt_system.py`
4. Constraint-specific tests: focused logic-gate task tests assert exact target gate counts and `bbox_set` annotation cardinality.
