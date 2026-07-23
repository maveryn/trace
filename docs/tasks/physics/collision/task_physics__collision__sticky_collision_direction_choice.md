# `task_physics__collision__sticky_collision_direction_choice`

## Summary
- Domain: `physics`
- Scene id: `collision`
- Implementation scene: `collision`
- Implementation source: `src/trace_tasks/tasks/physics/collision/sticky_collision_direction_choice.py`

## Task Contract
Selects the final direction of two perpendicular pucks after a sticky collision using conserved momentum.

## Program Contract

Program: `option_letter(direction(momentum_sum(pucks_a_b), mode=final_sticky_velocity)); scene=collision; scope=sticky_collision_direction_choice`

Candidate set: the visible puck states, motion arrows or trails, impact marker, labels, and candidate option cells inside the `sticky_collision_direction_choice` objective scope.
Operands: `pucks_a_b` (semantic_role, allowed `visible_input_pucks_A_B`, source `program_schema_concrete`); `final_sticky_velocity` (semantic_role, allowed `velocity_after_sticky_collision`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter from `A`, `B`, `C`, or `D`.
Annotation witnesses: `segment_set` witnesses from the finalized render. Annotation marks the two incoming velocity-arrow segments for puck A and puck B. The selected answer option remains answer context; it is not the public annotation witness.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(direction(momentum_sum(pucks_a_b), mode=final_sticky_velocity)); scene=collision; scope=sticky_collision_direction_choice` |

## Program Metadata
- Program signatures: `physics.momentum_direction_choice`
- Base program contract: `option_letter(direction(momentum_sum(pucks_a_b), mode=final_sticky_velocity)); scene=collision; scope=sticky_collision_direction_choice`
- Parameter axes: `correct_option_letter`, `scene_variant`, `accent_color_name`
- Arguments:
  - `pucks_a_b`: semantic_role; allowed `visible_input_pucks_A_B`; source `program_schema_concrete`
  - `final_sticky_velocity`: semantic_role; allowed `velocity_after_sticky_collision`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter from `A`, `B`, `C`, or `D`.

## Annotation Contract
- Annotation schema: `segment_set`
- Generator `annotation_gt.type`: `segment_set`
- Annotation marks the two incoming velocity-arrow segments for puck A and puck B.
- The selected answer option remains answer context; it is not the public annotation witness.
- Annotation and answer are projected from the same generated execution trace.

## Prompt And Trace Requirements
- Prompt text comes from `src/trace_tasks/resources/prompts/physics/collision/physics_collision_v1.json`.
- Render randomness, sampled fonts/styles, visible masses and speeds, chosen option letter, and verifier payloads are explicit in the instance trace.
- Diagrams must keep all quantities required for the momentum computation visible.
