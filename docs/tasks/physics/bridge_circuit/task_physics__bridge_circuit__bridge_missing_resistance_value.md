# `task_physics__bridge_circuit__bridge_missing_resistance_value`

## Summary
- Domain: `physics`
- Scene id: `bridge_circuit`
- Implementation scene: `bridge_circuit`
- Implementation source: `src/trace_tasks/tasks/physics/bridge_circuit/bridge_missing_resistance_value.py`

## Task Contract
Computes the missing resistor value in a balanced bridge circuit from the visible resistor labels and zero-current bridge meter.

## Program Contract

Program: `integer(solve_balanced_bridge_resistance(known_resistor_values, zero_meter_condition, unknown_resistor_slot)); scene=bridge_circuit; scope=bridge_missing_resistance_value`

Candidate set: the visible bridge circuit components, branch labels, and balance/readout markings inside the `bridge_missing_resistance_value` objective scope.
Operands: `resistors_r1_r2_r3_r4` (semantic_role, allowed `visible_bridge_resistors_with_known_values_and_one_unknown`, source `program_schema_concrete`); `zero_meter_condition` (semantic_role, allowed `visible_bridge_meter_reading_zero`, source `program_schema_concrete`); `unknown_slot` (query_operand, allowed `R1|R2|R3|R4`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the missing resistance in ohms.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation marks the question-mark target resistor. The zero meter reading and known resistor labels remain visible context in the image and trace metadata, but they are not public annotation witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(solve_balanced_bridge_resistance(known_resistor_values, zero_meter_condition, unknown_resistor_slot)); scene=bridge_circuit; scope=bridge_missing_resistance_value` |

## Program Metadata
- Program signatures: `physics.bridge_missing_resistance_value`
- Base program contract: `integer(solve_balanced_bridge_resistance(known_resistor_values, zero_meter_condition, unknown_resistor_slot)); scene=bridge_circuit; scope=bridge_missing_resistance_value`
- Parameter axes: `scene_variant`, `missing_resistor`, `target_answer`
- Arguments:
  - `resistors_r1_r2_r3_r4`: semantic_role; allowed `visible_bridge_resistors_with_known_values_and_one_unknown`; source `program_schema_concrete`
  - `zero_meter_condition`: semantic_role; allowed `visible_bridge_meter_reading_zero`; source `program_schema_concrete`
  - `unknown_slot`: query_operand; allowed `R1|R2|R3|R4`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the missing resistance in ohms.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation marks the question-mark target resistor.
- The zero meter reading and known resistor labels remain visible context in the image and trace metadata, but they are not public annotation witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics bridge-circuit v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, known resistor values, missing slot, zero-meter condition, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep the bridge topology, resistor labels, question-mark target, and zero meter visible.
