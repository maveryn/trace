# `task_physics__pulley__pulley_mechanical_advantage`

## Summary
- Domain: `physics`
- Scene id: `pulley`
- Implementation scene: `pulley`
- Implementation source: `src/trace_tasks/tasks/physics/pulley/pulley_mechanical_advantage.py`

## Program Contract

Program: `integer(solve_ideal_pulley(load_force, effort_force, support_strand_count, unknown_slot)); scene=pulley; scope=pulley_mechanical_advantage`

Candidate set: the visible pulley system, rope/support segments, load markers, and cut or connected segment cues inside the `pulley_mechanical_advantage` objective scope.
Operands: `effort_force` (semantic_role, allowed `visible_or_unknown_effort_force`, source `program_schema_concrete`); `load_force` (semantic_role, allowed `visible_or_unknown_load_force`, source `program_schema_concrete`); `support_strand_count` (semantic_role, allowed `counted_full_supporting_strands`, source `program_schema_concrete`); `unknown_slot` (semantic_role, allowed `effort_force`, `load_force`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is an exact integer produced by the symbolic physics construction.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys: `supporting_strands_region`, `known_force_label`, `unknown_force_label`. Annotation is keyed because witness roles are distinct; each key maps to the final-image pixel box for that role.
Query ids: `missing_effort_force_value`, `missing_load_force_value`.

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `missing_effort_force_value` | `integer(load_force / support_strand_count); scene=pulley; scope=pulley_mechanical_advantage` |
| `missing_load_force_value` | `integer(effort_force * support_strand_count); scene=pulley; scope=pulley_mechanical_advantage` |

## Program Metadata
- Program signatures: `physics.pulley_force_solve`
- Base program contract: `integer(solve_ideal_pulley(load_force, effort_force, support_strand_count, unknown_slot)); scene=pulley; scope=pulley_mechanical_advantage`
- Parameter axes: `query_id`, `support_strand_count`, `scene_variant`, `accent_color_name`
- Arguments:
  - `effort_force`: semantic_role; allowed `visible_or_unknown_effort_force`; source `program_schema_concrete`
  - `load_force`: semantic_role; allowed `visible_or_unknown_load_force`; source `program_schema_concrete`
  - `support_strand_count`: semantic_role; allowed `counted_full_supporting_strands`; source `program_schema_concrete`
  - `unknown_slot`: semantic_role; allowed `effort_force`, `load_force`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `missing_effort_force_value`, `missing_load_force_value`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is an exact integer produced by the symbolic physics construction.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys: `supporting_strands_region`, `known_force_label`, `unknown_force_label`.
- Annotation is keyed because witness roles are distinct; each key maps to the final-image pixel box for that role.
- Annotation must mark the region containing the full supporting strands and the visible force labels needed to compute the missing force. It must not mark cut non-supporting strands or decorative pulley hardware.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/pulley/physics_pulley_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Public `query_id` determines the unknown force slot: `missing_effort_force_value` asks for the effort force, and `missing_load_force_value` asks for the load force.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
