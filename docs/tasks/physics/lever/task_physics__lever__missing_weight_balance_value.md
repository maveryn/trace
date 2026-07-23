# `task_physics__lever__missing_weight_balance_value`

## Summary
- Domain: `physics`
- Scene id: `lever`
- Implementation scene: `lever`
- Implementation source: `src/trace_tasks/tasks/physics/lever/missing_weight_balance_value.py`

## Program Contract

Program: `solve_torque_balance(left_weight_distance_terms, right_weight_distance_terms, unknown_weight); scene=lever; scope=missing_weight_balance_value`

Candidate set: the visible lever beam, fulcrum, distance marks, weight blocks, and missing-weight marker inside the `missing_weight_balance_value` objective scope.
Operands: `left_weight_distance_terms` (semantic_role, allowed `visible_left_weight_distance_terms`, source `program_schema_concrete`); `right_weight_distance_terms` (semantic_role, allowed `visible_right_weight_distance_terms`, source `program_schema_concrete`); `unknown_weight` (semantic_role, allowed `marked_missing_weight`, source `program_schema_concrete`).
Operation: evaluate `solve_torque_balance` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is an exact integer produced by the symbolic physics construction.
Annotation witnesses: `bbox_set_map` witnesses from the finalized render. Annotation uses keys `known_weights` and `target_weight`; each key maps to final-image pixel boxes for the corresponding visible weight blocks. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `solve_torque_balance(left_weight_distance_terms, right_weight_distance_terms, unknown_weight); scene=lever; scope=missing_weight_balance_value` |

## Program Metadata
- Program signatures: `physics.torque_balance_solve`
- Base program contract: `solve_torque_balance(left_weight_distance_terms, right_weight_distance_terms, unknown_weight); scene=lever; scope=missing_weight_balance_value`
- Parameter axes: `scene_variant`, `target_answer`, `accent_color_name`
- Supported `query_id`s: `single`
- Arguments:
  - `left_weight_distance_terms`: semantic_role; allowed `visible_left_weight_distance_terms`; source `program_schema_concrete`
  - `right_weight_distance_terms`: semantic_role; allowed `visible_right_weight_distance_terms`; source `program_schema_concrete`
  - `unknown_weight`: semantic_role; allowed `marked_missing_weight`; source `program_schema_concrete`
- Argument metadata status: `curated`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is an exact integer produced by the symbolic physics construction.

## Annotation Contract
- Annotation schema: `bbox_set_map`
- Generator `annotation_gt.type`: `bbox_set_map`
- Annotation uses keys `known_weights` and `target_weight`; each key maps to final-image pixel boxes for the corresponding visible weight blocks.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/lever/physics_lever_v1.json`.
- Public `query_id` is `single`; the internal diagnostic branch is recorded as `missing_weight_to_balance`.
- Render randomness, sampled fonts/styles, operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
