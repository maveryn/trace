# `task_physics__magnetic_force__force_direction_choice`

## Summary
- Domain: `physics`
- Scene id: `magnetic_force`
- Implementation scene: `magnetic_force`
- Implementation source: `src/trace_tasks/tasks/physics/magnetic_force/force_direction_choice.py`

## Program Contract

Program: `option_letter(direction(charge_sign * cross_product(velocity_vector, magnetic_field_orientation))); scene=magnetic_force; scope=force_direction_choice`

Candidate set: the visible charged particle, velocity vector, magnetic-field orientation marker, and direction options inside the `force_direction_choice` objective scope.
Operands: `charge_sign` (semantic_role, allowed `negative_charge`, `positive_charge`, source `program_schema_concrete`); `magnetic_field_orientation` (semantic_role, allowed `into_page`, `out_of_page`, source `program_schema_concrete`); `velocity_vector` (semantic_role, allowed `visible_velocity_arrow`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation is keyed because witness roles are distinct; each key maps to the minimal final-image pixel box for that role. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(direction(charge_sign * cross_product(velocity_vector, magnetic_field_orientation))); scene=magnetic_force; scope=force_direction_choice` |

## Program Metadata
- Program signatures: `physics.lorentz_force_direction_choice`
- Base program contract: `option_letter(direction(charge_sign * cross_product(velocity_vector, magnetic_field_orientation))); scene=magnetic_force; scope=force_direction_choice`
- Parameter axes: `charge_sign`, `magnetic_field_orientation`
- Arguments:
  - `charge_sign`: semantic_role; allowed `negative_charge`, `positive_charge`; source `program_schema_concrete`
  - `magnetic_field_orientation`: semantic_role; allowed `into_page`, `out_of_page`; source `program_schema_concrete`
  - `velocity_vector`: semantic_role; allowed `visible_velocity_arrow`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation is keyed because witness roles are distinct; each key maps to the minimal final-image pixel box for that role.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
