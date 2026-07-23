# `task_physics__electrostatic_field__potential_value`

## Summary
- Domain: `physics`
- Scene id: `electrostatic_field`
- Implementation scene: `electrostatic_field`
- Implementation source: `src/trace_tasks/tasks/physics/electrostatic_field/potential_value.py`

## Task Contract
Computes the signed integer electric potential at point P from visible point charges and distance labels with k=1.

## Program Contract

Program: `sum(point_charge_potential(charges_q1_q2_q3, point_p, k=1)); scene=electrostatic_field; scope=potential_value`

Candidate set: the visible point charges, query point marker, test-charge cue, and direction option arrows inside the `potential_value` objective scope.
Operands: `charges_q1_q2_q3` (semantic_role, allowed `visible_point_charges_Q1_Q2_Q3`, source `program_schema_concrete`); `point_p` (semantic_role, allowed `visible_point_P`, source `program_schema_concrete`).
Operation: evaluate `sum` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is an exact integer produced by the symbolic physics construction.
Annotation witnesses: `point_map` witnesses from the finalized render. Annotation role keys are `Q1`, `Q2`, `Q3`, and `P`; each maps to the final-image pixel point at the center of the corresponding marker. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, decorative chrome, or derived numeric annotations.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `sum(point_charge_potential(charges_q1_q2_q3, point_p, k=1)); scene=electrostatic_field; scope=potential_value` |

## Program Metadata
- Program signatures: `physics.electric_potential_value`
- Base program contract: `sum(point_charge_potential(charges_q1_q2_q3, point_p, k=1)); scene=electrostatic_field; scope=potential_value`
- Parameter axes: `target_answer`, `scene_variant`, `accent_color_name`
- Arguments:
  - `charges_q1_q2_q3`: semantic_role; allowed `visible_point_charges_Q1_Q2_Q3`; source `program_schema_concrete`
  - `point_p`: semantic_role; allowed `visible_point_P`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id` values: `single`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is an exact integer produced by the symbolic physics construction.

## Annotation Contract
- Annotation schema: `point_map`
- Generator `annotation_gt.type`: `point_map`
- Annotation role keys are `Q1`, `Q2`, `Q3`, and `P`; each maps to the final-image pixel point at the center of the corresponding marker.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, decorative chrome, or derived numeric annotations.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/electrostatic_field/physics_electrostatic_field_v1.json`, with scene, task, query, and output layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
- `scalar_annotation_checked=true`: map annotation is retained because multiple role-bound point witnesses are required.
