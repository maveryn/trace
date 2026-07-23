# `task_physics__lever__side_torque_value`

## Summary
- Domain: `physics`
- Scene id: `lever`
- Implementation scene: `lever`
- Implementation source: `src/trace_tasks/tasks/physics/lever/side_torque_value.py`

## Program Contract

Program: `sum(weight_i * distance_i for weight_i in weights_on_queried_side); scene=lever; scope=side_torque_value`

Candidate set: the visible lever beam, fulcrum, distance marks, weight blocks, and missing-weight marker inside the `side_torque_value` objective scope.
Operands: `distance_i` (semantic_role, allowed `visible_distance_from_fulcrum`, source `program_schema_concrete`); `weight_i` (semantic_role, allowed `visible_weight_block`, source `program_schema_concrete`); `weights_on_queried_side` (semantic_role, allowed `visible_weights_on_queried_side`, source `program_schema_concrete`).
Operation: evaluate `sum` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is an exact integer produced by the symbolic physics construction.
Annotation witnesses: `bbox_set` witnesses from the finalized render. Annotation is an unordered set of final-image pixel boxes around every weight block on the queried side. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `sum(weight_i * distance_i for weight_i in weights_on_queried_side); scene=lever; scope=side_torque_value` |

## Program Metadata
- Program signatures: `physics.torque_sum_value`
- Base program contract: `sum(weight_i * distance_i for weight_i in weights_on_queried_side); scene=lever; scope=side_torque_value`
- Parameter axes: `torque_side`, `scene_variant`, `target_answer`, `accent_color_name`
- Supported `query_id`s: `single`
- Arguments:
  - `distance_i`: semantic_role; allowed `visible_distance_from_fulcrum`; source `program_schema_concrete`
  - `weight_i`: semantic_role; allowed `visible_weight_block`; source `program_schema_concrete`
  - `weights_on_queried_side`: semantic_role; allowed `visible_weights_on_queried_side`; source `program_schema_concrete`
- Argument metadata status: `curated`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is an exact integer produced by the symbolic physics construction.
- Current tuned side-weight cap: at most `3` visible weight blocks are sampled on the queried side.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set | unordered`
- Annotation is an unordered set of final-image pixel boxes around every weight block on the queried side.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/lever/physics_lever_v1.json`.
- Public `query_id` is `single`; the sampled side is recorded as `torque_side`.
- Render randomness, sampled fonts/styles, operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
