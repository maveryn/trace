# `task_physics__electrostatic_field__field_direction_choice`

## Summary
- Domain: `physics`
- Scene id: `electrostatic_field`
- Implementation scene: `electrostatic_field`
- Implementation source: `src/trace_tasks/tasks/physics/electrostatic_field/field_direction_choice.py`

## Task Contract
Selects the option arrow matching the electric field direction or the force direction for a positive/negative test charge at point P.

## Program Contract

Program: `option_letter(direction_from_charges(charges_q1_q2_q3, point_p, query_id)); scene=electrostatic_field; scope=field_direction_choice`

Candidate set: the visible point charges, query point marker, test-charge cue, and direction option arrows inside the `field_direction_choice` objective scope.
Operands: `charges_q1_q2_q3` (semantic_role, allowed `visible_point_charges_Q1_Q2_Q3`, source `program_schema_concrete`); `query_id` (semantic_role, allowed `electric_field_direction`, `force_on_negative_charge`, `force_on_positive_charge`, source `query_branch`); `point_p` (semantic_role, allowed `visible_point_P`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter.
Annotation witnesses: `point_map` witnesses from the finalized render. Annotation role keys are `Q1`, `Q2`, `Q3`, and `P`; each maps to the final-image pixel point at the center of the corresponding marker. For force-query branches, the P marker contains the visible sign of the test charge; for the electric-field branch, P is neutral.
Query ids: `electric_field_direction`, `force_on_positive_charge`, `force_on_negative_charge`.

## Reasoning Operations

Families: `spatial_relations`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `electric_field_direction` | `option_letter(direction(net_field(charges_q1_q2_q3, point_p))); scene=electrostatic_field; scope=field_direction_choice` |
| `force_on_positive_charge` | `option_letter(direction(force_on_positive_test_charge(net_field(charges_q1_q2_q3, point_p)))); scene=electrostatic_field; scope=field_direction_choice` |
| `force_on_negative_charge` | `option_letter(direction(force_on_negative_test_charge(net_field(charges_q1_q2_q3, point_p)))); scene=electrostatic_field; scope=field_direction_choice` |

## Program Metadata
- Program signatures: `physics.electric_field_direction_choice`
- Base program contract: `option_letter(direction_from_charges(charges_q1_q2_q3, point_p, query_id)); scene=electrostatic_field; scope=field_direction_choice`
- Parameter axes: `query_id`, `target_direction`, `correct_option_letter`, `scene_variant`, `accent_color_name`
- Arguments:
  - `charges_q1_q2_q3`: semantic_role; allowed `visible_point_charges_Q1_Q2_Q3`; source `program_schema_concrete`
  - `query_id`: semantic_role; allowed `electric_field_direction`, `force_on_negative_charge`, `force_on_positive_charge`; source `query_branch`
  - `point_p`: semantic_role; allowed `visible_point_P`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id` values: `electric_field_direction`, `force_on_positive_charge`, `force_on_negative_charge`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter.

## Annotation Contract
- Annotation schema: `point_map`
- Generator `annotation_gt.type`: `point_map`
- Annotation role keys are `Q1`, `Q2`, `Q3`, and `P`; each maps to the final-image pixel point at the center of the corresponding marker.
- For force-query branches, the P marker contains the visible sign of the test charge; for the electric-field branch, P is neutral.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option letters, decorative chrome, or derived numeric annotations.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/electrostatic_field/physics_electrostatic_field_v1.json`, with scene, task, query, and output layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
- `scalar_annotation_checked=true`: map annotation is retained because multiple role-bound point witnesses are required.
