# `task_physics__collision__sticky_collision_speed_value`

## Summary
- Domain: `physics`
- Scene id: `collision`
- Implementation scene: `collision`
- Implementation source: `src/trace_tasks/tasks/physics/collision/sticky_collision_speed_value.py`

## Task Contract
Computes the speed magnitude of the combined puck after a sticky collision.

## Program Contract

Program: `speed(final_velocity(momentum_sum(pucks_a_b), combined_mass), rounding=one_decimal); scene=collision; scope=sticky_collision_speed_value`

Candidate set: the visible puck states, motion arrows or trails, impact marker, and labels inside the `sticky_collision_speed_value` objective scope.
Operands: `pucks_a_b` (semantic_role, allowed `visible_input_pucks_A_B`, source `program_schema_concrete`); `combined_mass` (semantic_role, allowed `mass_A_plus_mass_B`, source `program_schema_concrete`); `one_decimal` (semantic_role, allowed `round_to_one_decimal_place`, source `program_schema_concrete`).
Operation: evaluate `speed` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `number` schema; Answer precision: `one_decimal`
Annotation witnesses: `segment_set` witnesses from the finalized render. Annotation marks the two incoming velocity-arrow segments for puck A and puck B. Annotation and answer are projected from the same generated execution trace.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `speed(final_velocity(momentum_sum(pucks_a_b), combined_mass), rounding=one_decimal); scene=collision; scope=sticky_collision_speed_value` |

## Program Metadata
- Program signatures: `physics.momentum_speed_value`
- Base program contract: `speed(final_velocity(momentum_sum(pucks_a_b), combined_mass), rounding=one_decimal); scene=collision; scope=sticky_collision_speed_value`
- Parameter axes: `target_answer`, `target_speed_tenths`, `correct_option_letter`, `scene_variant`, `accent_color_name`
- Arguments:
  - `pucks_a_b`: semantic_role; allowed `visible_input_pucks_A_B`; source `program_schema_concrete`
  - `combined_mass`: semantic_role; allowed `mass_A_plus_mass_B`; source `program_schema_concrete`
  - `one_decimal`: semantic_role; allowed `round_to_one_decimal_place`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `number`
- Answer precision: `one_decimal`
- Generator `answer_gt.type`: `number`
- The answer value is the final speed in m/s rounded to one decimal place.
- Current tuned answer support: `2.8`, `3.6`, `4.2`, `5.0`, and `5.7` m/s.

## Annotation Contract
- Annotation schema: `segment_set`
- Generator `annotation_gt.type`: `segment_set`
- Annotation marks the two incoming velocity-arrow segments for puck A and puck B.
- Annotation and answer are projected from the same generated execution trace.

## Prompt And Trace Requirements
- Prompt text comes from `src/trace_tasks/resources/prompts/physics/collision/physics_collision_v1.json`.
- Speed prompts explicitly remind the solver to compute horizontal and vertical final velocity components from momentum divided by combined mass, then combine the perpendicular components.
- Render randomness, sampled fonts/styles, visible masses and speeds, rounded speed target, and verifier payloads are explicit in the instance trace.
- Diagrams must keep all quantities required for the speed calculation visible.
