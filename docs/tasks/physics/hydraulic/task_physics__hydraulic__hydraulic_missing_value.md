# `task_physics__hydraulic__hydraulic_missing_value`

## Summary
- Domain: `physics`
- Scene id: `hydraulic`
- Implementation scene: `hydraulic`
- Implementation source: `src/trace_tasks/tasks/physics/hydraulic/hydraulic_missing_value.py`

## Program Contract

Program: `solve_pascal_law(input_force, input_area, output_force, output_area, unknown_slot); scene=hydraulic; scope=hydraulic_missing_value`

Candidate set: the visible piston sides, force labels, area labels, chambers, and missing-value marker inside the `hydraulic_missing_value` objective scope.
Operands: `input_area` (semantic_role, allowed `visible_input_piston_area`, source `program_schema_concrete`); `input_force` (semantic_role, allowed `visible_or_unknown_input_force`, source `program_schema_concrete`); `output_area` (semantic_role, allowed `visible_or_unknown_output_piston_area`, source `program_schema_concrete`); `output_force` (semantic_role, allowed `visible_or_unknown_output_force`, source `program_schema_concrete`); `unknown_slot` (semantic_role, allowed `input_area`, `input_force`, `output_area`, `output_force`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `solve_pascal_law` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is an exact integer produced by the symbolic physics construction.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation is keyed because input/output side roles are distinct; keys are `input_side` and `output_side`. Each annotation box marks the corresponding piston side, including its force label, chamber, and area label.
Query ids: `missing_input_area`, `missing_input_force`, `missing_output_force`, `missing_piston_area`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `missing_input_area` | `solve_pascal_law(input_force, input_area, output_force, output_area, unknown_slot); scene=hydraulic; scope=hydraulic_missing_value; query_branch=missing_input_area` |
| `missing_input_force` | `solve_pascal_law(input_force, input_area, output_force, output_area, unknown_slot); scene=hydraulic; scope=hydraulic_missing_value; query_branch=missing_input_force` |
| `missing_output_force` | `solve_pascal_law(input_force, input_area, output_force, output_area, unknown_slot); scene=hydraulic; scope=hydraulic_missing_value; query_branch=missing_output_force` |
| `missing_piston_area` | `solve_pascal_law(input_force, input_area, output_force, output_area, unknown_slot); scene=hydraulic; scope=hydraulic_missing_value; query_branch=missing_piston_area` |

## Program Metadata
- Program signatures: `physics.pascal_law_solve`
- Base program contract: `solve_pascal_law(input_force, input_area, output_force, output_area, unknown_slot); scene=hydraulic; scope=hydraulic_missing_value`
- Parameter axes: `unknown_slot`
- Arguments:
  - `input_area`: semantic_role; allowed `visible_input_piston_area`; source `program_schema_concrete`
  - `input_force`: semantic_role; allowed `visible_or_unknown_input_force`; source `program_schema_concrete`
  - `output_area`: semantic_role; allowed `visible_or_unknown_output_piston_area`; source `program_schema_concrete`
  - `output_force`: semantic_role; allowed `visible_or_unknown_output_force`; source `program_schema_concrete`
  - `unknown_slot`: semantic_role; allowed `input_area`, `input_force`, `output_area`, `output_force`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id`s: `missing_input_area`, `missing_input_force`, `missing_output_force`, `missing_piston_area`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is an exact integer produced by the symbolic physics construction.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation is keyed because input/output side roles are distinct; keys are `input_side` and `output_side`.
- Each annotation box marks the corresponding piston side, including its force label, chamber, and area label.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.
- Scalar annotation checked: `true`; two keyed boxes are retained because both input and output side roles are needed for the Pascal-law relation.

## Prompt And Trace Requirements
- Prompt text must come from the physics prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
