# `task_physics__circuit_equivalent__total_resistance_value`

## Summary
- Domain: `physics`
- Scene id: `circuit_equivalent`
- Implementation scene: `circuit_equivalent`
- Implementation source: `src/trace_tasks/tasks/physics/circuit_equivalent/total_resistance_value.py`

## Task Contract
Computes equivalent resistance for a visible mixed series-parallel resistor network between terminals A and B.

## Program Contract

Program: `integer(equivalent_resistance(visible_resistors_between_terminals, series_parallel_topology)); scene=circuit_equivalent; scope=total_resistance_value`

Candidate set: the visible components between terminals A and B, their values, and the series-parallel topology inside the `total_resistance_value` objective scope.
Operands: `visible_resistors_between_terminals` (semantic_role, allowed `labeled_resistors_between_A_B`, source `program_schema_concrete`); `series_parallel_topology` (semantic_role, allowed `mixed_series_parallel_topology`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the exact equivalent resistance in ohms.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation is one final-image pixel box around the full resistor network between terminals `A` and `B`, including resistor symbols, value labels, and connecting wires. Annotation must not mark decorative frame chrome, answer text, or inferred equivalent-resistance calculations.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `topology`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(equivalent_resistance(visible_resistors_between_terminals, series_parallel_topology)); scene=circuit_equivalent; scope=total_resistance_value` |

## Program Metadata
- Program signatures: `physics.equivalent_resistance_value`
- Base program contract: `integer(equivalent_resistance(visible_resistors_between_terminals, series_parallel_topology)); scene=circuit_equivalent; scope=total_resistance_value`
- Parameter axes: `scene_variant`, `target_answer`, `accent_color_name`
- Arguments:
  - `visible_resistors_between_terminals`: semantic_role; allowed `labeled_resistors_between_A_B`; source `program_schema_concrete`
  - `series_parallel_topology`: semantic_role; allowed `mixed_series_parallel_topology`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id`s: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the exact equivalent resistance in ohms.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is one final-image pixel box around the full resistor network between terminals `A` and `B`, including resistor symbols, value labels, and connecting wires.
- Annotation must not mark decorative frame chrome, answer text, or inferred equivalent-resistance calculations.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.
- Scalar annotation checked: `true`.

## Prompt And Trace Requirements
- Prompt text must come from the physics circuit-equivalent v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, component values, topology blocks, target answer, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep terminals A/B, all resistor labels, resistor values, and the mixed series-parallel topology visible.
