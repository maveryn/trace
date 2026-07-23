# `task_physics__electrostatic_field__zero_field_point_label`

## Summary
- Domain: `physics`
- Scene id: `electrostatic_field`
- Implementation scene: `electrostatic_field`
- Implementation source: `src/trace_tasks/tasks/physics/electrostatic_field/zero_field_point_label.py`

## Task Contract
Selects the labeled candidate point where the net electric field is zero for a two-charge setup.

## Program Contract

Program: `option_letter(select(candidate_points, net_field(charges_q1_q2, candidate_point)=zero_field)); scene=electrostatic_field; scope=zero_field_point_label`

Candidate set: the visible point charges, query point marker, test-charge cue, and direction option arrows inside the `zero_field_point_label` objective scope.
Operands: `candidate_points` (semantic_role, allowed `visible_candidate_points`, source `program_schema_concrete`); `charges_q1_q2` (semantic_role, allowed `fixed_same_sign_charge_pair`, source `program_schema_concrete`); `zero_field` (semantic_role, allowed `zero_field`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter.
Annotation witnesses: `point_map` witnesses from the finalized render. Annotation role keys are `Q1`, `Q2`, and `zero_point`; each maps to the final-image pixel point at the center of the corresponding marker. Annotation must mark the charge markers and selected zero-field point, not all candidate points, answer labels, option letters, or decorative chrome.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(select(candidate_points, net_field(charges_q1_q2, candidate_point)=zero_field)); scene=electrostatic_field; scope=zero_field_point_label` |

## Program Metadata
- Program signatures: `physics.electric_zero_field_selection`
- Base program contract: `option_letter(select(candidate_points, net_field(charges_q1_q2, candidate_point)=zero_field)); scene=electrostatic_field; scope=zero_field_point_label`
- Parameter axes: `correct_option_letter`, `scene_variant`, `accent_color_name`
- Arguments:
  - `candidate_points`: semantic_role; allowed `visible_candidate_points`; source `program_schema_concrete`
  - `charges_q1_q2`: semantic_role; allowed `fixed_same_sign_charge_pair`; source `program_schema_concrete`
  - `zero_field`: semantic_role; allowed `zero_field`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id` values: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter.

## Annotation Contract
- Annotation schema: `point_map`
- Generator `annotation_gt.type`: `point_map`
- Annotation role keys are `Q1`, `Q2`, and `zero_point`; each maps to the final-image pixel point at the center of the corresponding marker.
- Annotation must mark the charge markers and selected zero-field point, not all candidate points, answer labels, option letters, or decorative chrome.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/electrostatic_field/physics_electrostatic_field_v1.json`, with scene, task, query, and output layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
- `scalar_annotation_checked=true`: map annotation is retained because multiple role-bound point witnesses are required.
