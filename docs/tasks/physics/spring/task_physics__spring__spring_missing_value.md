# `task_physics__spring__spring_missing_value`

## Summary
- Domain: `physics`
- Scene id: `spring`
- Implementation scene: `spring`
- Implementation source: `src/trace_tasks/tasks/physics/spring/spring_missing_value.py`

## Program Contract

Program: `solve_hooke_ratio(reference_weight, reference_extension, query_weight, query_extension, unknown_slot); scene=spring; scope=spring_missing_value`

Candidate set: the visible spring setups, weight blocks, extension markers, and measurement labels inside the `spring_missing_value` objective scope.
Operands: `reference_weight` (semantic_role, allowed `left_spring_weight_block`, source `program_schema_concrete`); `reference_extension` (semantic_role, allowed `left_spring_extension_marker`, source `program_schema_concrete`); `query_weight` (semantic_role, allowed `right_spring_weight_block_or_missing_marker`, source `program_schema_concrete`); `query_extension` (semantic_role, allowed `right_spring_extension_marker_or_missing_marker`, source `program_schema_concrete`); `unknown_slot` (query_operand, allowed `weight|extension`, source `query_id`); active `query_id` branch when present.
Operation: evaluate `solve_hooke_ratio` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is the exact integer weight or extension produced by the symbolic spring-ratio construction.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys are `reference_weight`, `reference_extension`, `query_weight`, and `query_extension`. Annotation must mark the corresponding visible or marked weight blocks and extension markers in the final rendered image. Extension-marker boxes are vertically padded around the marker so the annotation target is not too thin.
Query ids: `missing_weight_for_extension`, `missing_extension_for_weight`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `missing_weight_for_extension` | `solve_hooke_ratio(reference_weight, reference_extension, query_extension, unknown_slot=weight); scene=spring; scope=spring_missing_value` |
| `missing_extension_for_weight` | `solve_hooke_ratio(reference_weight, reference_extension, query_weight, unknown_slot=extension); scene=spring; scope=spring_missing_value` |

## Program Metadata
- Program signatures: `physics.hooke_law_solve`
- Base program contract: `solve_hooke_ratio(reference_weight, reference_extension, query_weight, query_extension, unknown_slot); scene=spring; scope=spring_missing_value`
- Parameter axes: `scene_variant`, `unknown_slot`, `target_answer`, `accent_color_name`
- Arguments:
  - `reference_weight`: semantic_role; allowed `left_spring_weight_block`; source `program_schema_concrete`
  - `reference_extension`: semantic_role; allowed `left_spring_extension_marker`; source `program_schema_concrete`
  - `query_weight`: semantic_role; allowed `right_spring_weight_block_or_missing_marker`; source `program_schema_concrete`
  - `query_extension`: semantic_role; allowed `right_spring_extension_marker_or_missing_marker`; source `program_schema_concrete`
  - `unknown_slot`: query_operand; allowed `weight|extension`; source `query_id`
- Argument metadata status: `curated`
- Supported query ids: `missing_weight_for_extension`, `missing_extension_for_weight`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is the exact integer weight or extension produced by the symbolic spring-ratio construction.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys are `reference_weight`, `reference_extension`, `query_weight`, and `query_extension`.
- Annotation must mark the corresponding visible or marked weight blocks and extension markers in the final rendered image. Extension-marker boxes are vertically padded around the marker so the annotation target is not too thin.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/spring/physics_spring_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, scene variant, accent color, target answer, scale factor, and verifier payloads must be explicit in the instance trace.
- The right spring contains the missing marker; the left spring provides the visible reference ratio.
